"""MARVIN V3 only. No training/Slurm invocation occurs during preparation."""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from wakeword.data import SpeechCommandsBinary
from wakeword.metrics import compute_metrics
from wakeword.model import MarvinDSCNN, parameter_count
from wakeword.train import predict
from wakeword.plots import save_training_plots
from .data import TrainingData, CyclingSampler, inventory, AUGMENTATION, FEATURES
from .decision import threshold_sweep
from .model import MarvinV3
from .streaming import replay_scores, replay_report, STREAM_CONFIG


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protect_output(output):
    # Both the program and launcher refuse even incomplete existing experiments.
    if output.exists():
        raise FileExistsError(f'Refusing existing experiment output: {output}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset/speech_commands'))
    parser.add_argument('--output', type=Path, default=Path('models/marvin_v3'))
    parser.add_argument('--baseline', type=Path, default=Path('models/marvin_exp2_ratio1to2_60ep'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--validation-far-cap', type=float, default=0.0025)
    return parser.parse_args()


def latency(model, example, device):
    model.eval()
    timings = []
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(50):
            model(example)
        sync()
        for _ in range(200):
            sync()
            begin = time.perf_counter()
            model(example)
            sync()
            timings.append((time.perf_counter()-begin)*1000)
    values = torch.tensor(timings, dtype=torch.float64)
    return dict(device=str(device), warmups=50, measured=200, batch_size=1,
                excludes='audio loading and feature extraction', mean_ms=float(values.mean()),
                median_ms=float(values.quantile(.5)), p95_ms=float(values.quantile(.95)))


def save_predictions(path, dataset, probabilities):
    rows = [dict(sample_id=p.relative_to(dataset.root).as_posix(),
                 speaker_id=p.name.split('_nohash_')[0], word=p.parent.name,
                 label=y, probability=float(score))
            for (p, y), score in zip(dataset.items, probabilities)]
    write_csv(path, rows)


def main():
    args = parse_args()
    protect_output(args.output)
    if not 0 <= args.validation_far_cap < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError('Invalid training/threshold configuration')
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    audit = inventory(args.data.resolve())  # test IDs only, never test waveforms here
    baseline_metrics = json.loads((args.baseline/'metrics.json').read_text())
    if baseline_metrics['split_counts']['test'] != dict(examples=11005, marvin=195, other=10810):
        raise ValueError('Frozen baseline test mismatch')
    baseline_path = args.baseline/'best.pt'
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    train = TrainingData(args.data)
    validation = SpeechCommandsBinary(args.data, 'validation')
    sampler = CyclingSampler(train.labels, args.seed, args.batch_size)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    model = MarvinV3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    config = dict(experiment='MARVIN V3', family='binary DSCNN', args={k:str(v) if isinstance(v,Path) else v
                  for k,v in vars(args).items()}, features=FEATURES, augmentation=AUGMENTATION,
                  architecture=str(model), parameters=parameter_count(model),
                  raw_float32_parameter_bytes=4*parameter_count(model),
                  training=dict(optimizer='AdamW', weight_decay=1e-4, loss='unweighted BCEWithLogitsLoss',
                                scheduler='CosineAnnealingLR', workers=0, ratio='1 MARVIN : 2 non-MARVIN',
                                negatives='seeded full-pool cycling; no excluded words',
                                checkpoint_selection='validation F1 at validation-only FAR-constrained threshold',
                                threshold_selection='maximum validation F1 subject to FAR<=cap; ties recall then fewer FP'),
                  streaming=STREAM_CONFIG, split_audit=audit, baseline_checkpoint_sha256=baseline_hash,
                  started=started, environment=dict(python=platform.python_version(), torch=torch.__version__,
                  cuda=torch.version.cuda, hostname=platform.node()))
    write_json(args.output/'config.json', config)
    history, best_score, stale = [], -1.0, 0
    begin = time.perf_counter()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        loader = DataLoader(train, batch_sampler=sampler, num_workers=0)
        model.train()
        losses, correct, count = 0.0, 0, 0
        lr = optimizer.param_groups[0]['lr']
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses += float(loss.detach())*len(labels)
            correct += int(((logits.detach() >= 0) == labels.bool()).sum())
            count += len(labels)
        scheduler.step()
        p, y, val_loss = predict(model, val_loader, device)
        selected, _ = threshold_sweep(p, y, args.validation_far_cap)
        row = dict(epoch=epoch+1, learning_rate=lr, train=dict(loss=losses/count, accuracy=correct/count),
                   validation=dict(loss=val_loss, fixed_threshold_0_5=compute_metrics(p,y,.5).to_dict(),
                                   selected_threshold=selected))
        history.append(row)
        write_json(args.output/'training_history.json', history)
        print(json.dumps(row), flush=True)
        if selected['f1'] > best_score:
            best_score, stale = selected['f1'], 0
            torch.save(dict(model_state=model.state_dict(), epoch=epoch+1,
                            threshold=selected['threshold'], validation_metrics=selected,
                            validation_loss=val_loss, config=config), args.output/'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                break
    training_seconds = time.perf_counter()-begin
    checkpoint = torch.load(args.output/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    # Validation only: freeze both clip and replay-event thresholds BEFORE test construction.
    p, y, val_loss = predict(model, val_loader, device)
    selected, sweep = threshold_sweep(p,y,args.validation_far_cap)
    threshold = selected['threshold']
    write_csv(args.output/'validation_threshold_sweep.csv', sweep)
    save_predictions(args.output/'validation_scores.csv', validation, p)
    vp, vy, vi, windows = replay_scores(model, validation.items, device)
    streaming_selected, streaming_sweep = threshold_sweep(vp, vy, args.validation_far_cap)
    stream_threshold = streaming_selected['threshold']
    write_csv(args.output/'validation_streaming_threshold_sweep.csv', streaming_sweep)
    write_json(args.output/'threshold_selection.json', dict(source='validation only',
               far_definition='FP / negative examples; streaming uses placement events',
               far_cap=args.validation_far_cap, clip=selected, streaming=streaming_selected,
               streaming_rule=STREAM_CONFIG))
    validation_stream = replay_report(vp,vy,vi,windows,stream_threshold)
    # FIRST test waveform access: selected checkpoint reloaded/eval, thresholds already frozen.
    test = SpeechCommandsBinary(args.data, 'test')
    assert len(test) == 11005 and sum(test.labels) == 195
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    tp, ty, test_loss = predict(model, test_loader, device)
    test_selected = compute_metrics(tp,ty,threshold).to_dict()
    save_predictions(args.output/'test_scores.csv', test, tp)
    ts, tl, ti, tw = replay_scores(model, test.items, device)
    test_stream = replay_report(ts,tl,ti,tw,stream_threshold)
    write_csv(args.output/'test_streaming_scores.csv',
              [dict(**row, score=float(score)) for row,score in zip(ti,ts)])
    # Frozen V2 model/threshold, same test ordering; never fit V2 to test or V3 validation.
    baseline = MarvinDSCNN().to(device)
    frozen = torch.load(baseline_path, map_location=device, weights_only=True)
    baseline.load_state_dict(frozen['model_state'])
    baseline.eval()
    bp, by, _ = predict(baseline,test_loader,device)
    assert torch.equal(by, ty)
    baseline_selected = compute_metrics(bp,by,float(frozen['threshold'])).to_dict()
    # Stop rather than silently comparing a non-reproduced baseline.
    for key in ['true_positives','true_negatives','false_accepts','false_rejects']:
        if baseline_selected[key] != baseline_metrics['test'][key]:
            raise RuntimeError(f'Frozen V2 reproduction mismatch: {key}')
    save_predictions(args.output/'frozen_v2_test_scores.csv', test, bp)
    bs, bl, bi, bw = replay_scores(baseline,test.items,device)
    baseline_stream = replay_report(bs,bl,bi,bw,float(frozen['threshold']))
    example = test[0][0].unsqueeze(0).to(device)
    comparison = []
    for name, net, scores, selected_metrics, path in [
        ('MARVIN V2',baseline,bp,baseline_selected,baseline_path),
        ('MARVIN V3',model,tp,test_selected,args.output/'best.pt')]:
        comparison.append(dict(model=name, **selected_metrics, parameters=parameter_count(net),
                               checkpoint_bytes=path.stat().st_size,
                               checkpoint_decimal_MB=path.stat().st_size/1e6,
                               raw_float32_parameter_bytes=parameter_count(net)*4,
                               at_0_5=compute_metrics(scores,ty,.5).to_dict(),
                               model_latency=latency(net,example,device)))
    write_json(args.output/'v2_v3_comparison.json', comparison)
    write_csv(args.output/'v2_v3_comparison.csv',
              [{k:v for k,v in r.items() if k not in ('at_0_5','model_latency')} for r in comparison])
    report = dict(model='MarvinV3', parameters=parameter_count(model), best_epoch=checkpoint['epoch'],
                  epochs_completed=len(history), training_wall_seconds=training_seconds,
                  validation=dict(**selected, loss=val_loss), test=dict(**test_selected, loss=test_loss),
                  validation_at_0_5=compute_metrics(p,y,.5).to_dict(),
                  test_at_0_5=compute_metrics(tp,ty,.5).to_dict(), history=history,
                  streaming=dict(validation=validation_stream, test=test_stream, frozen_v2_test=baseline_stream),
                  comparison=comparison, configuration=config,
                  ended=datetime.now(timezone.utc).isoformat(),
                  success_definition='lower MARVIN FRR with acceptable FAR, not increased accuracy alone')
    write_json(args.output/'metrics.json', report)
    write_csv(args.output/'training_history.csv',
              [dict(epoch=r['epoch'], learning_rate=r['learning_rate'], train_loss=r['train']['loss'],
                    train_accuracy=r['train']['accuracy'], validation_loss=r['validation']['loss'],
                    **r['validation']['selected_threshold']) for r in history])
    save_training_plots(report,args.output)
    # Per-word false accepts retain difficult negatives and expose their performance.
    word_rows = []
    for word in sorted({path.parent.name for path,_ in test.items} - {'marvin'}):
        mask = torch.tensor([path.parent.name == word for path,_ in test.items])
        word_rows.append(dict(word=word, examples=int(mask.sum()),
                              v2_false_accepts=int((bp[mask]>=float(frozen['threshold'])).sum()),
                              v3_false_accepts=int((tp[mask]>=threshold).sum())))
    write_csv(args.output/'hard_negative_word_results.csv',word_rows)
    deployable = torch.jit.script(model.cpu().eval())
    deployable.save(str(args.output/'marvin_v3.ts'))
    report['deployable_model_bytes'] = (args.output/'marvin_v3.ts').stat().st_size
    report['deployable_model_decimal_MB'] = report['deployable_model_bytes']/1e6
    write_json(args.output/'metrics.json', report)
    write_json(args.output/'deployment_config.json',dict(features=FEATURES, streaming=STREAM_CONFIG,
               clip_threshold=threshold, streaming_threshold=stream_threshold,
               architecture='MarvinV3', runtime='TorchScript CPU offline',
               raspberry_pi_performance='not measured'))
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != baseline_hash:
        raise RuntimeError('Frozen V2 changed during evaluation')
    (args.output/'COMPLETED').write_text(report['ended']+'\n')


if __name__ == '__main__':
    main()
