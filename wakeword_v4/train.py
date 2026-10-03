"""MARVIN V4: upgraded binary wake-word training on Speech Commands v0.02.

Unchanged from V3: official train/validation/test lists (inventory-checked), V2 40-bin
log-mel window features, validation-only threshold = max F1 subject to clip FAR <= cap,
streaming replay protocol, test opened only after best reload and threshold freeze.

Upgrades:
  model     residual DSCNN + squeeze-excitation, mean+max pooling (wakeword_v4.model, ~65k params)
  data      1 MARVIN : 4 non-MARVIN per epoch with full-pool negative cycling, plus pure-noise
            negatives (Speech Commands + DEMAND background, labelled non-MARVIN)
  augment   speed 0.9-1.1, gain +/-6 dB, shift +/-100 ms, noise (Speech Commands + DEMAND)
            p=.8 at 0-20 dB SNR, SpecAugment (GPU, 1 freq + 1 time mask)
  optim     AdamW wd .01, warmup -> per-step cosine, label-smoothed BCE (.05), EMA weights
Compared on test against frozen V2 (exp2) and, if present and complete, V3.
"""
import argparse, hashlib, json, math, random, time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Sampler

from wakeword.data import (SpeechCommandsBinary, fit_length, load_random_segment, load_wav,
                           CLIP_SAMPLES)
from wakeword.metrics import compute_metrics
from wakeword.model import MarvinDSCNN, parameter_count
from wakeword.train import predict
from wakeword.plots import save_training_plots
from wakeword_v3.data import inventory, FEATURES
from wakeword_v3.decision import threshold_sweep
from wakeword_v3.model import MarvinV3
from wakeword_v3.streaming import replay_scores, replay_report, STREAM_CONFIG
from wakeword_v3.train import write_json, write_csv, latency, save_predictions
from .model import MarvinV4

AUGMENTATION = {
    "speed_perturbation": {"probability": 0.5, "factor": [0.9, 1.1], "method": "linear resampling"},
    "time_shift": {"probability": 0.5, "ms": [-100, 100], "mode": "zero-filled"},
    "gain": {"probability": 0.5, "db": [-6, 6]},
    "noise": {"probability": 0.8, "snr_db": [0, 20], "sources": "Speech Commands _background_noise_ + DEMAND 16 kHz"},
    "spec_augment": {"freq_masks": 1, "freq_width_max": 6, "time_masks": 1, "time_width_max": 12},
    "noise_only_negatives_per_epoch": "0.25 x positives (pure background, label 0)",
    "training_only": True,
}


class TrainingDataV4(SpeechCommandsBinary):
    """Indices >= len(items) are synthetic pure-noise negatives."""

    def __init__(self, root, extra_noise_dir):
        super().__init__(root, "train", augment=True, extra_noise_dir=extra_noise_dir)

    def _noise(self):
        return fit_length(load_random_segment(random.choice(self.noise_paths), CLIP_SAMPLES), True)

    def _augment(self, waveform):
        if random.random() < 0.5:
            shift = random.randint(-1600, 1600)
            waveform = waveform.roll(shift)
            if shift > 0: waveform[:shift] = 0
            elif shift < 0: waveform[shift:] = 0
        if random.random() < 0.5:
            waveform = waveform * 10 ** (random.uniform(-6, 6) / 20)
        if random.random() < 0.8:
            noise = self._noise()
            gain = waveform.square().mean().sqrt().clamp_min(1e-4) / noise.square().mean().sqrt().clamp_min(1e-4)
            waveform = waveform + noise * gain * 10 ** (-random.uniform(0, 20) / 20)
        return waveform.clamp(-1, 1)

    def __getitem__(self, index):
        if index >= len(self.items):  # pure background clip, random level
            waveform = (self._noise() * 10 ** (random.uniform(-30, 0) / 20)).clamp(-1, 1)
            return self.features(waveform).unsqueeze(0), torch.tensor(0.0)
        path, label = self.items[index]
        waveform = load_wav(path)
        if random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            waveform = F.interpolate(waveform[None, None], size=max(400, round(len(waveform) / factor)),
                                     mode='linear', align_corners=False)[0, 0]
        waveform = self._augment(fit_length(waveform, training=True))
        return self.features(waveform).unsqueeze(0), torch.tensor(float(label))


class CyclingSamplerV4(Sampler):
    """All positives once per epoch, `ratio` cycled negatives each, plus noise-only negatives."""

    def __init__(self, labels, ratio=4, noise_fraction=0.25, seed=1337, batch_size=128):
        self.positives = [i for i, y in enumerate(labels) if y]
        negatives = [i for i, y in enumerate(labels) if not y]
        g = torch.Generator().manual_seed(seed)
        self.negatives = [negatives[i] for i in torch.randperm(len(negatives), generator=g).tolist()]
        self.noise_count = round(noise_fraction * len(self.positives))
        self.noise_base = len(labels)
        self.ratio, self.seed, self.batch_size, self.epoch = ratio, seed, batch_size, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(((1 + self.ratio) * len(self.positives) + self.noise_count) / self.batch_size)

    def __iter__(self):
        count = self.ratio * len(self.positives)
        items = list(self.positives)
        items += [self.negatives[(self.epoch * count + i) % len(self.negatives)] for i in range(count)]
        items += [self.noise_base + i for i in range(self.noise_count)]
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        items = [items[i] for i in torch.randperm(len(items), generator=g).tolist()]
        for start in range(0, len(items), self.batch_size):
            yield items[start:start + self.batch_size]


def spec_augment(x, freq_width=6, time_width=12):
    b, _, n_mels, t = x.shape; dev = x.device
    mel = torch.arange(n_mels, device=dev)[None]; frames = torch.arange(t, device=dev)[None]
    fw = torch.randint(0, freq_width + 1, (b, 1), device=dev); fs = (torch.rand(b, 1, device=dev) * (n_mels - fw + 1)).floor()
    tw = torch.randint(0, time_width + 1, (b, 1), device=dev); ts = (torch.rand(b, 1, device=dev) * (t - tw + 1)).floor()
    keep = ~((mel >= fs) & (mel < fs + fw))[:, :, None] & ~((frames >= ts) & (frames < ts + tw))[:, None, :]
    return x * keep[:, None].to(x.dtype)


def lr_at(step, total, warmup, peak, floor=1e-5):
    if step < warmup:
        return peak * (step + 1) / warmup
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('dataset/speech_commands'))
    p.add_argument('--extra-noise-dir', type=Path, default=Path('dataset/noise/demand/wav16k'))
    p.add_argument('--output', type=Path, required=True)
    # Optional frozen baselines for the test comparison; skipped when absent (not shipped in this repo).
    p.add_argument('--baseline', type=Path, default=Path('models/marvin_exp2_ratio1to2_60ep'))
    p.add_argument('--v3', type=Path, default=Path('models/marvin_v3'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--learning-rate', type=float, default=3e-3)
    p.add_argument('--weight-decay', type=float, default=0.01); p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--negative-ratio', type=int, default=4); p.add_argument('--noise-fraction', type=float, default=0.25)
    p.add_argument('--ema-decay', type=float, default=0.998); p.add_argument('--patience', type=int, default=25)
    p.add_argument('--width', type=int, default=48); p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=6); p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--validation-far-cap', type=float, default=0.0025)
    p.add_argument('--smoke', action='store_true', help='CPU check: 2 short epochs, small validation/test subsets')
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing existing experiment output: {args.output}')
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)
    audit = inventory(args.data.resolve())  # test IDs only, never test waveforms here
    baseline_path = args.baseline / 'best.pt'
    has_baseline = baseline_path.is_file()
    baseline_metrics = json.loads((args.baseline / 'metrics.json').read_text()) if has_baseline else None
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest() if has_baseline else None

    train = TrainingDataV4(args.data, args.extra_noise_dir)
    validation = SpeechCommandsBinary(args.data, 'validation')
    if args.smoke:
        keep = [i for i, (_, y) in enumerate(validation.items) if y][:20] + [i for i, (_, y) in enumerate(validation.items) if not y][:300]
        validation.items = [validation.items[i] for i in keep]
        train.items = [it for it in train.items if it[1]][:64] + [it for it in train.items if not it[1]][:2000]
        args.epochs, args.warmup_epochs, args.workers = 2, 1, 0
    sampler = CyclingSamplerV4(train.labels, args.negative_ratio, args.noise_fraction, args.seed, args.batch_size)
    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, batch_sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(validation, batch_size=256, num_workers=args.workers)

    model = MarvinV4(args.width, args.dropout).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
    decay = [p for p in model.parameters() if p.ndim > 1]; no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                                   {'params': no_decay, 'weight_decay': 0.0}], lr=args.learning_rate)
    steps_per_epoch = len(sampler); total = args.epochs * steps_per_epoch; warmup = args.warmup_epochs * steps_per_epoch
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(experiment='MARVIN V4', family='residual DSCNN + SE', args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  features=FEATURES, augmentation=AUGMENTATION, architecture=str(model), parameters=parameter_count(model),
                  raw_float32_parameter_bytes=4 * parameter_count(model), noise_sources=train.noise_source_counts,
                  training=dict(optimizer='AdamW', weight_decay=args.weight_decay, loss=f'BCEWithLogits, label smoothing {args.label_smoothing}',
                                schedule=f'{args.warmup_epochs}-epoch linear warmup, per-step cosine', ema_decay=args.ema_decay,
                                ratio=f'1 MARVIN : {args.negative_ratio} non-MARVIN + {sampler.noise_count} noise-only negatives/epoch',
                                negatives='seeded full-pool cycling; no excluded words', steps_per_epoch=steps_per_epoch,
                                checkpoint_selection='EMA model, validation F1 at validation-only FAR-constrained threshold',
                                threshold_selection='maximum validation F1 subject to FAR<=cap; ties recall then fewer FP'),
                  streaming=STREAM_CONFIG, split_audit=audit, baseline_checkpoint_sha256=baseline_hash,
                  started=datetime.now(timezone.utc).isoformat())
    write_json(args.output / 'config.json', config)
    print(json.dumps(dict(parameters=config['parameters'], steps_per_epoch=steps_per_epoch, noise=train.noise_source_counts)), flush=True)

    history, best_score, stale, step = [], -1.0, 0, 0
    begin = time.perf_counter()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train(); losses = correct = count = 0
        for features, labels in train_loader:
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            for g in optimizer.param_groups: g['lr'] = lr_at(step, total, warmup, args.learning_rate)
            logits = model(spec_augment(features))
            targets = labels * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            ema.update_parameters(model); step += 1
            losses += float(loss.detach()) * len(labels); correct += int(((logits.detach() >= 0) == labels.bool()).sum()); count += len(labels)
        p, y, val_loss = predict(ema, val_loader, device)
        selected, _ = threshold_sweep(p, y, args.validation_far_cap)
        row = dict(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]['lr'], train=dict(loss=losses / count, accuracy=correct / count),
                   validation=dict(loss=val_loss, fixed_threshold_0_5=compute_metrics(p, y, .5).to_dict(), selected_threshold=selected))
        history.append(row); write_json(args.output / 'training_history.json', history); print(json.dumps(row), flush=True)
        if selected['f1'] > best_score:
            best_score, stale = selected['f1'], 0
            torch.save(dict(model_state=ema.module.state_dict(), epoch=epoch + 1, threshold=selected['threshold'],
                            validation_metrics=selected, validation_loss=val_loss, model_config=dict(width=args.width, dropout=args.dropout)),
                       args.output / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                break
    training_seconds = time.perf_counter() - begin

    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state']); model.eval()
    # Validation only: freeze clip and replay-event thresholds BEFORE test construction.
    p, y, val_loss = predict(model, val_loader, device)
    selected, sweep = threshold_sweep(p, y, args.validation_far_cap); threshold = selected['threshold']
    write_csv(args.output / 'validation_threshold_sweep.csv', sweep)
    save_predictions(args.output / 'validation_scores.csv', validation, p)
    vp, vy, vi, vw = replay_scores(model, validation.items, device)
    streaming_selected, streaming_sweep = threshold_sweep(vp, vy, args.validation_far_cap)
    stream_threshold = streaming_selected['threshold']
    write_csv(args.output / 'validation_streaming_threshold_sweep.csv', streaming_sweep)
    write_json(args.output / 'threshold_selection.json', dict(source='validation only', far_cap=args.validation_far_cap,
               far_definition='FP / negative examples; streaming uses placement events', clip=selected,
               streaming=streaming_selected, streaming_rule=STREAM_CONFIG))
    validation_stream = replay_report(vp, vy, vi, vw, stream_threshold)

    # FIRST test waveform access.
    test = SpeechCommandsBinary(args.data, 'test')
    if args.smoke:
        test.items = [it for it in test.items if it[1]][:20] + [it for it in test.items if not it[1]][:300]
    else:
        assert len(test) == 11005 and sum(test.labels) == 195
    test_loader = DataLoader(test, batch_size=256, num_workers=args.workers)
    tp, ty, test_loss = predict(model, test_loader, device)
    test_selected = compute_metrics(tp, ty, threshold).to_dict()
    save_predictions(args.output / 'test_scores.csv', test, tp)
    ts, tl, ti, tw = replay_scores(model, test.items, device)
    test_stream = replay_report(ts, tl, ti, tw, stream_threshold)
    write_csv(args.output / 'test_streaming_scores.csv', [dict(**r, score=float(s)) for r, s in zip(ti, ts)])

    # Frozen baselines on the same test ordering with their own frozen thresholds.
    rivals = [('MARVIN V2 (exp2)', MarvinDSCNN(), baseline_path)] if has_baseline else []
    if (args.v3 / 'COMPLETED').is_file():
        rivals.append(('MARVIN V3', MarvinV3(), args.v3 / 'best.pt'))
    comparison, per_word, baseline_notes = [], {}, {}
    example = test[0][0].unsqueeze(0).to(device)
    for name, net, path in rivals:
        frozen = torch.load(path, map_location=device, weights_only=False)
        net.load_state_dict(frozen['model_state']); net.to(device).eval()
        bp, by, _ = predict(net, test_loader, device); assert torch.equal(by, ty)
        metrics = compute_metrics(bp, by, float(frozen['threshold'])).to_dict()
        if name.startswith('MARVIN V2') and not args.smoke:
            stored = baseline_metrics['test']
            mismatch = {k: (metrics[k], stored[k]) for k in ('true_positives', 'true_negatives', 'false_accepts', 'false_rejects') if metrics[k] != stored[k]}
            baseline_notes[name] = 'reproduced stored test counts' if not mismatch else f'recomputed counts differ from stored: {mismatch}'
        bs, bl, bi, bw = replay_scores(net, test.items, device)
        comparison.append(dict(model=name, **metrics, parameters=parameter_count(net), checkpoint_bytes=path.stat().st_size,
                               at_0_5=compute_metrics(bp, by, .5).to_dict(), streaming=replay_report(bs, bl, bi, bw, float(frozen['threshold'])),
                               model_latency=latency(net, example, device)))
        per_word[name] = (bp, float(frozen['threshold']))
    comparison.append(dict(model='MARVIN V4', **test_selected, parameters=parameter_count(model),
                           checkpoint_bytes=(args.output / 'best.pt').stat().st_size, at_0_5=compute_metrics(tp, ty, .5).to_dict(),
                           streaming=test_stream, model_latency=latency(model, example, device)))
    per_word['MARVIN V4'] = (tp, threshold)
    write_json(args.output / 'comparison.json', dict(rows=comparison, baseline_reproduction=baseline_notes))
    write_csv(args.output / 'comparison.csv', [{k: v for k, v in r.items() if k not in ('at_0_5', 'model_latency', 'streaming')} for r in comparison])
    word_rows = []
    for word in sorted({p.parent.name for p, _ in test.items} - {'marvin'}):
        mask = torch.tensor([p.parent.name == word for p, _ in test.items])
        word_rows.append(dict(word=word, examples=int(mask.sum()),
                              **{f'{n} false_accepts': int((s[mask] >= t).sum()) for n, (s, t) in per_word.items()}))
    write_csv(args.output / 'hard_negative_word_results.csv', word_rows)

    report = dict(model='MarvinV4', parameters=parameter_count(model), best_epoch=checkpoint['epoch'], epochs_completed=len(history),
                  training_wall_seconds=training_seconds, validation=dict(**selected, loss=val_loss), test=dict(**test_selected, loss=test_loss),
                  validation_at_0_5=compute_metrics(p, y, .5).to_dict(), test_at_0_5=compute_metrics(tp, ty, .5).to_dict(),
                  history=history, streaming=dict(validation=validation_stream, test=test_stream), comparison=comparison,
                  configuration=config, ended=datetime.now(timezone.utc).isoformat(),
                  success_definition='lower MARVIN FRR at validation-capped FAR, not accuracy alone')
    write_json(args.output / 'metrics.json', report)
    write_csv(args.output / 'training_history.csv',
              [dict(epoch=r['epoch'], learning_rate=r['learning_rate'], train_loss=r['train']['loss'], train_accuracy=r['train']['accuracy'],
                    validation_loss=r['validation']['loss'], **r['validation']['selected_threshold']) for r in history])
    save_training_plots(report, args.output)
    torch.jit.script(model.cpu().eval()).save(str(args.output / 'marvin_v4.ts'))
    report['deployable_model_bytes'] = (args.output / 'marvin_v4.ts').stat().st_size
    write_json(args.output / 'metrics.json', report)
    write_json(args.output / 'deployment_config.json', dict(features=FEATURES, streaming=STREAM_CONFIG, clip_threshold=threshold,
               streaming_threshold=stream_threshold, architecture='MarvinV4', model_config=checkpoint['model_config'],
               runtime='TorchScript CPU offline', raspberry_pi_performance='not measured'))
    if has_baseline and hashlib.sha256(baseline_path.read_bytes()).hexdigest() != baseline_hash:
        raise RuntimeError('Frozen V2 changed during evaluation')
    print(json.dumps([{k: r[k] for k in ('model', 'parameters', 'f1', 'recall', 'precision', 'false_accepts', 'false_rejects')} for r in comparison], indent=1), flush=True)
    (args.output / 'COMPLETED').write_text(report['ended'] + '\n')


if __name__ == '__main__':
    main()
