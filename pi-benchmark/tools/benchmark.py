"""ELKIT / MRVN benchmark: model speed on this machine + real usage from the voice log + system health.

Run on the Raspberry Pi 5 (MRVN can keep running; that is the realistic load):

    ~/marvin/venv/bin/python ~/marvin/marvin_local/tools/benchmark.py
    ~/marvin/venv/bin/python ~/marvin/marvin_local/tools/benchmark.py --hours 24     # voice log: last 24 h only

Sections:
  1. Model speed   - wake word (one 1 s window, run every 100 ms), command recognition (intent + slot) for
                     1.5 / 3 / 5 s of audio; mean, median, p95 over many runs.
  2. Real usage    - from marvin/data/voice_log.jsonl: wake scores and timing, recognition time after you stop
                     speaking, how many commands ran / were rejected / ignored, per intent.
  3. Accuracy      - the figures measured when the models were trained (models/config.json).
  4. System        - Pi model, CPU temperature, power/throttling, memory, MRVN's CPU and RAM.
A copy of the results is saved to marvin/data/benchmark_<time>.json.
"""
import argparse, json, os, platform, shutil, statistics, subprocess, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

LOG = ROOT / 'marvin' / 'data' / 'voice_log.jsonl'


def stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return {'n': len(v), 'mean': statistics.fmean(v), 'median': statistics.median(v),
            'p95': v[min(len(v) - 1, int(round(.95 * (len(v) - 1))))], 'max': v[-1]}


def fmt(s, unit='ms', digits=1):
    if not s:
        return 'no data'
    return (f"median {s['median']:.{digits}f} {unit} | mean {s['mean']:.{digits}f} | p95 {s['p95']:.{digits}f} | "
            f"max {s['max']:.{digits}f}  (n={s['n']})")


def timed(fn, runs, warmup=5):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); out.append((time.perf_counter() - t) * 1000)
    return stats(out)


def model_speed(runs):
    from marvin_engine.models import CommandRecognizer, WakeWord
    rng = np.random.default_rng(0)
    speech_like = lambda seconds: (rng.standard_normal(int(16000 * seconds)) * .05).astype(np.float32)
    wake = WakeWord()
    window = speech_like(1.0)
    result = {'wake_window_ms': timed(lambda: wake.probability(window), runs)}
    commands = CommandRecognizer()
    for seconds in (1.5, 3.0, 5.0):
        audio = speech_like(seconds)
        result[f'command_{seconds:g}s_ms'] = timed(lambda: commands.predict(audio), max(20, runs // 5))
    # The wake model runs once per 100 ms of audio: the share of one CPU core that costs.
    result['wake_cpu_share_pct'] = result['wake_window_ms']['mean'] / 100 * 100
    return result


def real_usage(hours):
    if not LOG.is_file():
        return {'error': f'no voice log yet ({LOG})'}
    since = time.time() - hours * 3600 if hours else 0
    events = []
    for line in LOG.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('t', 0) >= since:
            events.append(e)
    by = lambda kind: [e for e in events if e.get('type') == kind]
    wakes, commands, decisions = by('wake'), by('command'), by('decision')
    return {
        'events': len(events), 'from': min((e['t'] for e in events if 't' in e), default=None), 'to': max((e['t'] for e in events if 't' in e), default=None),
        'wakes': len(wakes), 'listening_timeouts': len(by('listening_timeout')), 'commands': len(commands),
        'wake_score': stats([e.get('probability') for e in wakes]),
        'wake_infer_ms': stats([e.get('infer_ms') for e in wakes]),
        'recognition_after_speech_ms': stats([e.get('endpoint_to_result_ms') for e in commands]),
        'intent_ms': stats([e.get('intent_ms') for e in commands]),
        'slot_ms': stats([e.get('slot_ms') for e in commands if e.get('slot_ms')]),
        'command_audio_s': stats([e.get('audio_s') for e in commands]),
        'confidence': stats([e.get('confidence') for e in commands]),
        'model_accepted': sum(1 for e in commands if e.get('accepted')),
        'decisions': dict(Counter(e.get('result') for e in decisions)),
        'ran_by_intent': dict(Counter(e.get('intent') for e in decisions if e.get('result') == 'ran').most_common()),
    }


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ''


def system():
    info = {'machine': platform.machine(), 'python': platform.python_version(), 'cpus': os.cpu_count()}
    model = Path('/proc/device-tree/model')
    info['model'] = model.read_text(errors='replace').strip('\x00 \n') if model.exists() else platform.platform()
    if shutil.which('vcgencmd'):
        info['cpu_temp'] = run(['vcgencmd', 'measure_temp']).replace('temp=', '')
        info['throttled'] = run(['vcgencmd', 'get_throttled']).replace('throttled=', '')
        info['arm_clock_mhz'] = round(int(run(['vcgencmd', 'measure_clock', 'arm']).split('=')[-1] or 0) / 1e6)
    if Path('/proc/meminfo').exists():
        mem = dict(l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l)
        info['ram_total_mb'] = int(mem['MemTotal'].split()[0]) // 1024
        info['ram_available_mb'] = int(mem['MemAvailable'].split()[0]) // 1024
    if Path('/proc/loadavg').exists():
        info['load_1_5_15'] = Path('/proc/loadavg').read_text().split()[:3]
    ps = run(['ps', '-eo', 'pid,pcpu,rss,args', '--no-headers']) if os.name != 'nt' else ''
    service = [l.split(None, 3) for l in ps.splitlines() if 'marvin.preview' in l and 'benchmark' not in l]
    if service:
        pid, cpu, rss, _ = service[0]
        info['mrvn_process'] = {'pid': int(pid), 'cpu_pct': float(cpu), 'ram_mb': round(int(rss) / 1024)}
    return info


def main():
    parser = argparse.ArgumentParser(description='ELKIT / MRVN benchmark')
    parser.add_argument('--runs', type=int, default=200, help='wake-model runs (command runs = runs / 5)')
    parser.add_argument('--hours', type=float, help='voice log: only the last N hours (default: everything)')
    args = parser.parse_args()

    print('Benchmarking ... (about 30 s)\n')
    report = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'system': system(), 'model_speed': model_speed(args.runs),
              'real_usage': real_usage(args.hours),
              'accuracy_reported': json.loads((ROOT / 'models' / 'config.json').read_text())['reported_metrics']}

    s, m, u, a = report['system'], report['model_speed'], report['real_usage'], report['accuracy_reported']
    print('=' * 78); print(f"  SYSTEM   {s['model']}   ({s['cpus']} cores, Python {s['python']})"); print('=' * 78)
    for key, label in (('cpu_temp', 'CPU temperature'), ('arm_clock_mhz', 'CPU clock (MHz)'), ('throttled', 'Power/throttle flags'),
                       ('ram_available_mb', 'RAM available (MB)'), ('load_1_5_15', 'Load average')):
        if key in s:
            note = '  (0x0 = no power or heat problems)' if key == 'throttled' else ''
            print(f'  {label:24} {s[key]}{note}')
    if 'mrvn_process' in s:
        p = s['mrvn_process']; print(f"  {'MRVN service':24} CPU {p['cpu_pct']:.1f} %  |  RAM {p['ram_mb']} MB")

    print('\n  MODEL SPEED (this machine)')
    print(f"  {'Wake word, per 1 s window':34} {fmt(m['wake_window_ms'], digits=2)}")
    print(f"  {'  -> share of one CPU core':34} {m['wake_cpu_share_pct']:.1f} %  (runs every 100 ms)")
    for seconds in ('1.5', '3', '5'):
        print(f"  {'Command (intent+slot), ' + seconds + ' s':34} {fmt(m[f'command_{seconds}s_ms'])}")

    print('\n  REAL USAGE (voice log' + (f', last {args.hours:g} h' if args.hours else '') + ')')
    if 'error' in u:
        print('  ' + u['error'])
    else:
        span = (time.strftime('%Y-%m-%d %H:%M', time.localtime(u['from'])) + ' -> ' + time.strftime('%Y-%m-%d %H:%M', time.localtime(u['to']))) if u['from'] else '-'
        print(f"  {'Period':34} {span}")
        print(f"  {'Wakes / commands / timeouts':34} {u['wakes']} / {u['commands']} / {u['listening_timeouts']}")
        print(f"  {'Wake score when detected':34} {fmt(u['wake_score'], '', 3)}")
        print(f"  {'Wake inference':34} {fmt(u['wake_infer_ms'], digits=2)}")
        print(f"  {'Recognition after you stop':34} {fmt(u['recognition_after_speech_ms'])}")
        print(f"  {'Command length spoken':34} {fmt(u['command_audio_s'], 's', 2)}")
        print(f"  {'Intent confidence':34} {fmt(u['confidence'], '', 2)}")
        d = u['decisions']
        print(f"  {'MRVN decisions':34} ran {d.get('ran', 0)} | sorry {d.get('sorry', 0)} | ignored {d.get('ignored', 0)}")
        if u['ran_by_intent']:
            print(f"  {'Commands run, by intent':34} " + ', '.join(f'{k} {v}' for k, v in list(u['ran_by_intent'].items())[:8]))

    print('\n  ACCURACY (measured at training time, models/config.json)')
    w, r = a['wake_test_speech_commands'], a['intent_reject']['test']
    print(f"  {'Wake word MARVIN':34} recall {w['recall']:.1%} | false accepts {w['false_accept_rate']:.3%} of clips")
    print(f"  {'Intent (test set)':34} accuracy {r['in_scope_accuracy_all']:.1%} | correct+accepted {r['in_scope_correct_and_accepted']:.1%}")
    print(f"  {'Slot values (test set)':34} accuracy {a['slot_test_clean']['accuracy']:.1%} | real voices {a['slot_test_clean']['human_accuracy']:.1%}")

    out = ROOT / 'marvin' / 'data' / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
