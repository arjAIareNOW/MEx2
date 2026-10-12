"""ELKIT false-wake test on real speech that never says "Hi/Hello Elkit".

Each clip (SLURP "slurp_real", 16 kHz FLAC) is padded with 1 s of silence on both sides and streamed through the
ELKIT detector in 80 ms steps; every step's score is kept. A clip is a false wake for a setting (threshold, k) when
k consecutive scores reach the threshold. One model pass per clip serves all settings, and the scores are saved, so
other settings can be checked later without re-running the model.

The ELKIT model is not in this repository (see elkit/README.md). Usage:

    python scripts/elkit_false_accept.py --elkit-dir path/to/elkit --slurp-dir path/to/slurp_real [--n 2000]

Released result (2026-10-12, n=2000, seed 23101): see elkit/README.md.
"""
import argparse, glob, importlib.util, os, random, time
import numpy as np
import soundfile as sf

THRESHOLDS = (0.7575, 0.88, 0.92, 0.93, 0.94, 0.95)
READINGS = (1, 2)


def load_detector(elkit_dir):
    spec = importlib.util.spec_from_file_location('elkit_wakeword', os.path.join(elkit_dir, 'elkit_wakeword.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detector = module.WakeWord()
    detector.threshold = 2.0          # never fires: only the per-step scores are recorded
    return detector


def fires(scores, threshold, k):
    run = 0
    for v in scores:
        run = run + 1 if v >= threshold else 0
        if run >= k:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--elkit-dir', required=True, help='folder with elkit_wakeword.py and models/')
    ap.add_argument('--slurp-dir', required=True, help='folder with SLURP slurp_real *.flac')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=23101)
    ap.add_argument('--scores', default='elkit_fa_scores.npz', help='where to save every clip\'s score sequence')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.slurp_dir, '*.flac')))
    random.seed(args.seed)
    pick = random.sample(files, min(args.n, len(files)))
    detector = load_detector(args.elkit_dir)
    settings = [(t, k) for k in READINGS for t in THRESHOLDS]
    hits = dict.fromkeys(settings, 0)
    scores, seconds, t0 = {}, 0.0, time.time()

    for n, f in enumerate(pick, 1):
        x, sr = sf.read(f, dtype='float32')
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000 or x.size < 1600:
            continue
        seconds += x.size / 16000
        audio = np.concatenate([np.zeros(16000, np.float32), x, np.zeros(16000, np.float32)])
        detector.reset()
        p = []
        for i in range(0, audio.size, 1280):          # 80 ms steps
            detector.push(audio[i:i + 1280])
            p.append(detector.last_probability)
        scores[os.path.basename(f)] = np.array(p, np.float32)
        for s in settings:
            hits[s] += fires(p, *s)
        if n % 100 == 0:
            print(f'{n}/{len(pick)} clips ({(time.time() - t0) / 60:.1f} min)', flush=True)

    used, hours = len(scores), seconds / 3600
    print(f'\n{used} clips = {hours:.2f} h of real speech (SLURP real, seed {args.seed})')
    for s in settings:
        k = hits[s]
        print(f'  threshold {s[0]:.4f}, {s[1]} reading(s) in a row: {k} false wakes = {k / used:.2%} of clips, '
              f'{k / hours:.1f} per hour of continuous speech')
    np.savez_compressed(args.scores, **scores)
    print(f'scores saved to {args.scores}')


if __name__ == '__main__':
    main()
