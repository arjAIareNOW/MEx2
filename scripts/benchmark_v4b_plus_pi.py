"""Raspberry Pi 5 (8 GB) benchmark for V4-B+ (ONNX Runtime, CPU). Needs only numpy, pyarrow, onnxruntime.

Pipeline assumption: the intent model runs only after the MARVIN wake word fires; each clip is
the command audio after the wake word. Latency = WAV decode + NumPy log-mel + ONNX inference,
per clip, batch 1, after warm-up. RTF = processing time / audio duration.

Accuracy/macro-F1 over the 19 in-scope intents. OUT_OF_SCOPE clips measure false accepts:
without a reject gate every OOS clip is accepted as some intent (FAR = 1.0); a max-softmax
threshold table is reported for information only (no threshold is tuned on this data).

Usage (on the Pi, from the project root):
  python scripts/benchmark_v4b_plus_pi.py RUN_DIR --parquet ai231/data/holdout-00000-of-00001.parquet --threads 4
Writes RUN_DIR/pi_benchmark_<split>_<threads>t.json
"""
import argparse, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4b_plus_frontend import decode_wav, log_mel


def device_model():
    for path in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        try:
            return Path(path).read_text().strip('\x00\n ')
        except OSError:
            pass
    return platform.processor() or platform.machine()


def cpu_temp_c():
    try:
        return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    except (OSError, ValueError):
        return None


def macro_f1(y, p, n):
    f1 = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1)), f1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--parquet', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    config = json.loads((a.run / 'config.json').read_text())
    classes = config['classes']; index = {c: i for i, c in enumerate(classes)}
    options = ort.SessionOptions(); options.intra_op_num_threads = a.threads; options.inter_op_num_threads = 1
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(a.run / 'v4b_plus.onnx'), options, providers=['CPUExecutionProvider'])
    load_ms = (time.perf_counter() - t0) * 1000
    table = pq.read_table(a.parquet, columns=['audio', 'command'])
    if a.limit:
        table = table.slice(0, a.limit)
    audio, commands = table['audio'], table['command'].to_pylist()

    def run(wav_bytes):
        start = time.perf_counter()
        wave_ = decode_wav(wav_bytes); features = log_mel(wave_)
        t_feat = time.perf_counter()
        logits = session.run(['logits'], {'log_mel': features, 'frame_lengths': np.array([features.shape[-1]], dtype=np.int64)})[0][0]
        end = time.perf_counter()
        return logits, (end - start) * 1000, (end - t_feat) * 1000, len(wave_) / 16000

    for i in range(min(a.warmup, len(commands))):
        run(audio[i].as_py()['bytes'])
    temp_start = cpu_temp_c(); temp_peak = temp_start or 0
    total_ms, model_ms, rtf, y_true, y_pred, oos_conf, in_conf = [], [], [], [], [], [], []
    for i, command in enumerate(commands):
        logits, ms, inf_ms, seconds = run(audio[i].as_py()['bytes'])
        prob = np.exp(logits - logits.max()); prob /= prob.sum()
        total_ms.append(ms); model_ms.append(inf_ms); rtf.append(ms / 1000 / seconds)
        if command in index:
            y_true.append(index[command]); y_pred.append(int(prob.argmax())); in_conf.append(float(prob.max()))
        else:
            oos_conf.append(float(prob.max()))
        if i % 25 == 0:
            temp_peak = max(temp_peak, cpu_temp_c() or 0)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mf1, per_class = macro_f1(y_true, y_pred, len(classes))
    q = lambda v, p: float(np.percentile(v, p))
    gate = [{'max_softmax_threshold': t,
             'oos_false_accept_rate': float(np.mean(np.array(oos_conf) >= t)) if oos_conf else None,
             'in_scope_accept_rate': float(np.mean(np.array(in_conf) >= t)),
             'in_scope_accuracy_of_accepted': float(np.mean((y_pred == y_true)[np.array(in_conf) >= t])) if np.any(np.array(in_conf) >= t) else None}
            for t in (0.0, 0.5, 0.7, 0.9)]
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        rss_mb = None
    result = {
        'device': device_model(), 'machine': platform.machine(), 'os': platform.platform(), 'python': platform.python_version(),
        'runtime': 'ONNX Runtime', 'onnxruntime_version': ort.__version__, 'intra_op_threads': a.threads,
        'model_file': 'v4b_plus.onnx', 'model_load_ms': load_ms, 'data': str(a.parquet), 'clips': len(commands),
        'in_scope_clips': int(len(y_true)), 'oos_clips': len(oos_conf), 'warmup_clips': min(a.warmup, len(commands)),
        'pipeline_assumption': 'intent model invoked after MARVIN wake word; wake word not part of this measurement',
        'intent_accuracy': float(np.mean(y_true == y_pred)), 'intent_macro_f1': mf1,
        'per_class_f1': dict(zip(classes, map(float, per_class))),
        'oos_false_accept_rate_no_gate': 1.0 if oos_conf else None, 'oos_reject_gate_table_informational': gate,
        'latency_ms': {'end_to_end_p50': q(total_ms, 50), 'end_to_end_p95': q(total_ms, 95), 'end_to_end_mean': float(np.mean(total_ms)),
                       'model_only_p50': q(model_ms, 50), 'model_only_p95': q(model_ms, 95),
                       'scope': 'decode + NumPy log-mel + ONNX inference, batch 1'},
        'rtf': {'p50': q(rtf, 50), 'p95': q(rtf, 95), 'mean': float(np.mean(rtf))},
        'peak_rss_mb': rss_mb, 'cpu_temp_c': {'start': temp_start, 'peak': temp_peak or None},
    }
    out = a.run / f"pi_benchmark_{a.parquet.stem.split('-')[0]}_{a.threads}t.json"
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('device', 'intent_accuracy', 'intent_macro_f1', 'latency_ms', 'rtf')}, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
