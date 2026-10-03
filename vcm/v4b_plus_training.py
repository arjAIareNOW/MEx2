"""Train/evaluate TC-DSCNN V4-B+ on ai231 (19 classes) with an upgraded recipe.

Same speaker split, front end, class set and selection metric as the V4-B final run
(vcm.final19.prepare, seed 23101, validation macro-F1). Upgrades:
  model   : vcm.tc_dscnn_v4b_plus (width 128, SE, 6 dilated blocks, attentive stats pooling)
  augment : V4-B shift+noise (wider SNR) + random gain + speed perturbation (waveform),
            SpecAugment time/frequency masks and mixup (batch, on GPU)
  optim   : AdamW, linear warmup -> per-step cosine, label smoothing, bf16 autocast, EMA weights
After training, the selected EMA checkpoint and the frozen V4-B final checkpoint are both scored on the
official test split and on the holdout (RPi5 benchmark set, opened at user request).

Usage: python -m vcm.v4b_plus_training --output DIR [--data ~/ai231] [--seed 23101] [--smoke]
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel

from .final19 import CLASSES, PROJECT, eligible, prepare, read, stats
from .tc_dscnn_v4b_plus import IntentTCDSCNNV4BPlus, architecture_config
from .v2_experiment_common import ConservativeWaveformAugmentor, V2IntentDataset
from .train import class_weights, evaluate, make_loader, set_reproducible, write_history
from .dscnn_exp1 import collect_environment, sha256_file

BASELINE_FINAL = PROJECT / 'models/vcm_tc_dscnn_v4_b_19class_final'


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + '\n')


class StrongWaveformAugmentor(ConservativeWaveformAugmentor):
    """Gain and speed perturbation, then V4-B shift + background noise (SNR 5-25 dB)."""

    def __call__(self, waveform: Tensor) -> Tensor:
        if torch.rand(()) < 0.5:  # random gain, +/-6 dB
            waveform = waveform * 10.0 ** ((torch.rand(()) * 12 - 6) / 20)
        if torch.rand(()) < 0.5:  # speed perturbation 0.9-1.1 (tempo + pitch), linear resampling
            factor = 0.9 + 0.2 * float(torch.rand(()))
            length = max(400, int(round(waveform.shape[-1] / factor)))
            waveform = nn.functional.interpolate(waveform[None], size=length, mode='linear',
                                                 align_corners=False)[0][..., :80_000]
        waveform = super().__call__(waveform.clamp(-1.0, 1.0))
        if torch.rand(()) < 0.3:  # extra low-SNR noise pass (total range ~5-25 dB)
            waveform = self._noise(waveform, 5.0 + 5.0 * float(torch.rand(())))
        return waveform

    def _noise(self, waveform: Tensor, snr_db: float) -> Tensor:
        noise = self.noises[int(torch.randint(len(self.noises), ()).item())]
        if len(noise) < waveform.shape[-1]:
            noise = noise.repeat((waveform.shape[-1] + len(noise) - 1) // len(noise))
        start = int(torch.randint(len(noise) - waveform.shape[-1] + 1, ()).item())
        segment = noise[start:start + waveform.shape[-1]].unsqueeze(0)
        scale = waveform.square().mean().sqrt().clamp_min(1e-5) / (
            segment.square().mean().sqrt().clamp_min(1e-5) * 10.0 ** (snr_db / 20.0))
        return (waveform + segment * scale).clamp(-1.0, 1.0)


def spec_augment(x: Tensor, lengths: Tensor, freq_masks=2, freq_width=8, time_masks=2, time_frac=0.1) -> Tensor:
    """Per-utterance frequency/time masking inside the valid length; masked cells -> 0 (utterance mean)."""
    b, _, n_mels, t = x.shape; dev = x.device
    keep = torch.ones(b, n_mels, t, dtype=torch.bool, device=dev)
    mel = torch.arange(n_mels, device=dev)[None]; frames = torch.arange(t, device=dev)[None]
    for _ in range(freq_masks):
        w = torch.randint(0, freq_width + 1, (b, 1), device=dev)
        s = (torch.rand(b, 1, device=dev) * (n_mels - w + 1)).floor()
        keep &= ~((mel >= s) & (mel < s + w))[:, :, None]
    for _ in range(time_masks):
        w = (torch.rand(b, 1, device=dev) * (lengths[:, None] * time_frac).clamp(max=40)).floor()
        s = (torch.rand(b, 1, device=dev) * (lengths[:, None] - w + 1).clamp_min(1)).floor()
        keep &= ~((frames >= s) & (frames < s + w))[:, None, :]
    return x * keep[:, None].to(x.dtype)


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path.home() / 'ai231')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=23101, help='training seed (split seed is fixed at 23101)')
    p.add_argument('--epochs', type=int, default=120); p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, default=0.05); p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--mixup-alpha', type=float, default=0.3); p.add_argument('--mixup-prob', type=float, default=0.5)
    p.add_argument('--ema-decay', type=float, default=0.999); p.add_argument('--patience', type=int, default=30)
    p.add_argument('--width', type=int, default=128); p.add_argument('--dropout', type=float, default=0.30)
    p.add_argument('--workers', type=int, default=6); p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--smoke', action='store_true', help='CPU pipeline check on tiny train/val/test/holdout subsets')
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = a.output.resolve()
    if out == BASELINE_FINAL.resolve():
        raise SystemExit('Refusing to write into the frozen V4-B run')
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Refusing to overwrite non-empty {out}')
    out.mkdir(parents=True, exist_ok=True)

    # Identical split to the V4-B final run (prepare() uses seed 23101 internally via a.seed).
    train_seed = a.seed; a.seed = 23101
    table, ti, vi, detail = prepare(a)
    baseline_manifest = BASELINE_FINAL / 'split_manifest.json'
    if baseline_manifest.is_file():
        frozen = json.loads(baseline_manifest.read_text())
        assert frozen['train_indices'] == ti and frozen['validation_indices'] == vi, 'Split differs from V4-B'
    a.seed = train_seed
    set_reproducible(a.seed, a.cpu_threads)
    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True  # speed over bitwise repro
    dump(out / 'split_manifest.json', {k: v for k, v in detail.items()})

    device = torch.device('cpu' if a.smoke else 'cuda')
    if not a.smoke:
        assert torch.cuda.is_available(), 'CUDA required (use --smoke for a CPU check)'
    if a.smoke:
        rng = np.random.default_rng(0)
        ti = sorted(rng.choice(ti, 256, replace=False).tolist()); vi = sorted(rng.choice(vi, 128, replace=False).tolist())
        a.epochs, a.warmup_epochs, a.batch_size, a.workers = 2, 1, 32, 0

    mapping = {c: i for i, c in enumerate(CLASSES)}
    aug = StrongWaveformAugmentor(PROJECT / 'dataset/speech_commands/_background_noise_')
    tl = make_loader(V2IntentDataset(table, ti, mapping, aug), a, device, True)
    vl = make_loader(V2IntentDataset(table, vi, mapping), a, device, False)

    model = IntentTCDSCNNV4BPlus(19, a.dropout, a.width).to(device)
    # EMA with warm-up: decay ramps min(ema_decay, (1+n)/(10+n)) so early averages track the model.
    def ema_avg(averaged, current, num_averaged):
        decay = min(a.ema_decay, (1 + int(num_averaged)) / (10 + int(num_averaged)))
        return averaged * decay + current * (1 - decay)
    ema = AveragedModel(model, avg_fn=ema_avg, use_buffers=True)
    weights = class_weights(CLASSES, detail['train']['class_counts'], device)
    train_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=a.label_smoothing)
    eval_ce = nn.CrossEntropyLoss(weight=weights)
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        (no_decay if prm.ndim <= 1 else decay).append(prm)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=a.lr)
    total_steps = a.epochs * len(tl); warmup_steps = a.warmup_epochs * len(tl)
    params = sum(p.numel() for p in model.parameters())
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')

    config = {'status': 'TRAINING', 'classes': CLASSES, 'class_to_index': mapping, 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'architecture': architecture_config(a.width), 'parameter_count': params, 'float32_parameter_bytes': 4 * params,
              'split': {'train': len(ti), 'validation': len(vi), 'source': 'vcm.final19.prepare seed 23101 (identical to V4-B final)'},
              'augmentation': {'waveform': 'gain +/-6dB p.5; speed 0.9-1.1 p.5; V4-B shift +/-100ms p.5; noise 10-25dB p.5; extra noise 5-10dB p.3',
                               'spec_augment': '2 freq masks <=8 bins, 2 time masks <=10% len (<=40 frames)',
                               'mixup': {'alpha': a.mixup_alpha, 'probability': a.mixup_prob}},
              'optimizer': {'name': 'AdamW', 'lr': a.lr, 'weight_decay': a.weight_decay, 'warmup_epochs': a.warmup_epochs,
                            'schedule': 'linear warmup then per-step cosine to 1e-5', 'label_smoothing': a.label_smoothing,
                            'ema_decay': a.ema_decay, 'precision': 'bf16 autocast' if device.type == 'cuda' else 'fp32',
                            'loss_weights': 'TRAIN-only inverse frequency, mean 1 (as V4-B)'},
              'selection': 'better of raw / EMA (warm-up) weights each epoch, validation macro-F1', 'environment': collect_environment(device),
              'node': os.environ.get('SLURMD_NODENAME', os.uname().nodename), 'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'UNAVAILABLE'),
              'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
              'source_sha256': {str(p.relative_to(PROJECT)): sha256_file(p) for p in
                                [Path(__file__), PROJECT / 'vcm/tc_dscnn_v4b_plus.py', PROJECT / 'vcm/final19.py']}}
    dump(out / 'config.json', config)
    print(json.dumps({'parameters': params, 'train': len(ti), 'validation': len(vi), 'steps_per_epoch': len(tl)}), flush=True)

    history, best, stale, step, start = [], -1.0, 0, 0, time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); loss_sum = correct = seen = 0
        for x, lengths, y in tl:
            x, lengths, y = x.to(device, non_blocking=True), lengths.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for g in opt.param_groups: g['lr'] = lr_at(step, total_steps, warmup_steps, a.lr)
            x = spec_augment(x, lengths)
            lam = 1.0
            if a.mixup_alpha > 0 and float(torch.rand(())) < a.mixup_prob:
                lam = float(np.random.beta(a.mixup_alpha, a.mixup_alpha)); perm = torch.randperm(len(y), device=device)
                x = lam * x + (1 - lam) * x[perm]; lengths = torch.maximum(lengths, lengths[perm]); y2 = y[perm]
            with torch.autocast(**amp):
                logits = model(x, lengths)
            logits = logits.float()
            loss = train_ce(logits, y) if lam == 1.0 else lam * train_ce(logits, y) + (1 - lam) * train_ce(logits, y2)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); ema.update_parameters(model); step += 1
            loss_sum += float(loss) * len(y); correct += int((logits.argmax(1) == y).sum()); seen += len(y)
        # Score both the raw and EMA weights; keep whichever is better on validation macro-F1.
        candidates = {name: evaluate(net, vl, eval_ce, device, CLASSES) for name, net in (('raw', model), ('ema', ema.module))}
        chosen = max(candidates, key=lambda k: candidates[k][0].summary['macro_f1'])
        report, vloss = candidates[chosen]
        row = {'epoch': epoch, 'learning_rate': opt.param_groups[0]['lr'], 'train_loss': loss_sum / seen,
               'train_accuracy': correct / seen, 'validation_loss': vloss,
               **{'validation_' + k: v for k, v in report.summary.items()}}
        print(json.dumps({'epoch': epoch, 'raw_macro_f1': candidates['raw'][0].summary['macro_f1'], 'ema_macro_f1': candidates['ema'][0].summary['macro_f1'], 'selected': chosen}), flush=True)
        history.append(row); write_history(out / 'training_history.csv', history); print(json.dumps(row), flush=True)
        if report.summary['macro_f1'] > best:
            best, stale = report.summary['macro_f1'], 0
            torch.save({'weights': chosen, 'model_state': (model if chosen == 'raw' else ema.module).state_dict(), 'model_config': {'num_classes': 19, 'dropout': a.dropout, 'width': a.width},
                        'class_to_index': mapping, 'epoch': epoch,
                        'validation_report': {'loss': vloss, **report.summary, 'per_class': report.per_class,
                                              'confusion_matrix': report.confusion_matrix}}, out / 'best.pt')
        else:
            stale += 1
            if stale >= a.patience:
                print(f'Early stop: no validation macro-F1 gain for {a.patience} epochs', flush=True); break

    ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    config.update(status='TRAINED', best_epoch=ck['epoch'], epochs_completed=len(history),
                  training_wall_seconds=time.perf_counter() - start, validation_metrics=ck['validation_report'],
                  checkpoint_sha256=sha256_file(out / 'best.pt'), checkpoint_size_bytes=(out / 'best.pt').stat().st_size)
    dump(out / 'config.json', config)

    if not a.skip_test:
        # Final scoring of the selected V4-B+ and, for a same-conditions comparison, the frozen
        # V4-B final checkpoint: official test split + holdout (the RPi5 benchmark set).
        best_model = IntentTCDSCNNV4BPlus(19, a.dropout, a.width).to(device)
        best_model.load_state_dict(ck['model_state'])
        models = {'v4b_plus': best_model}
        base_ck = BASELINE_FINAL / 'training/best.pt'
        if base_ck.is_file():
            from .tc_dscnn_v4 import IntentTCDSCNNV4B
            base = IntentTCDSCNNV4B(19, .2).to(device)
            base.load_state_dict(torch.load(base_ck, map_location='cpu', weights_only=True)['model_state'])
            models['v4b_final'] = base
        results = {}
        for split in ('test', 'holdout'):
            files = sorted((a.data / 'data').glob(f'{split}-*.parquet'))
            tab = read(files, True)
            commands = tab['command'].to_pylist()
            indices = [i for i, c in enumerate(commands) if c in CLASSES][:64 if a.smoke else None]
            loader = make_loader(V2IntentDataset(tab, indices, mapping), a, device, False)
            for name, m in models.items():
                rep, rloss = evaluate(m, loader, eval_ce, device, CLASSES)
                result = {'loss': rloss, **rep.summary, 'per_class': rep.per_class, 'confusion_matrix': rep.confusion_matrix,
                          'dataset': stats(tab, indices), 'oos_excluded': len(tab) - len(indices),
                          'file_sha256': {str(f): sha256_file(f) for f in files}}
                dump(out / f'{split}_metrics_{name}.json', result)
                results.setdefault(split, {})[name] = {k: v for k, v in result.items() if k not in ('per_class', 'confusion_matrix', 'file_sha256')}
                print(split, name, json.dumps(results[split][name]), flush=True)
        config.update(status='TRAINED_AND_EVALUATED', test_metrics=results['test']['v4b_plus'],
                      holdout_metrics=results['holdout']['v4b_plus'], same_run_comparison=results,
                      holdout_note='Holdout opened at user request for an offline look; it was reserved for the RPi5 benchmark.')
        dump(out / 'config.json', config)

    write_comparison(out, config)
    (out / 'COMPLETED').write_text('done\n')


def write_comparison(out: Path, config: dict) -> None:
    """Side-by-side vs V4-B (frozen final run validation; any V4-B retrain dir with test metrics)."""
    def summary(d):
        return {k: d[k] for k in ('accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall') if k in d} if isinstance(d, dict) else d
    rows = {'V4-B+ (this run)': {'parameters': config['parameter_count'], 'validation': summary(config.get('validation_metrics')),
                                  'test': summary(config.get('test_metrics', 'NOT_RUN')),
                                  'holdout': summary(config.get('holdout_metrics', 'NOT_RUN'))}}
    for split, by_model in config.get('same_run_comparison', {}).items():
        if 'v4b_final' in by_model:
            rows.setdefault('V4-B final (scored in this run)', {})[split] = summary(by_model['v4b_final'])
    base = BASELINE_FINAL / 'submission_metadata.json'
    if base.is_file():
        m = json.loads(base.read_text())
        rows['V4-B final'] = {'parameters': m['parameter_count'], 'validation': summary(m['validation_metrics']), 'test': summary(m['test_metrics'])}
    for d in sorted(PROJECT.glob('models/vcm_tc_dscnn_v4_b_19class_retrain_*')):
        f = d / 'final_test_metrics.json'
        if f.is_file():
            m = json.loads((d / 'submission_metadata.json').read_text())
            rows[f'V4-B {d.name}'] = {'parameters': m['parameter_count'], 'validation': summary(m['validation_metrics']),
                                      'test': summary(json.loads(f.read_text()))}
    dump(out / 'comparison_vs_v4b.json', rows); print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
