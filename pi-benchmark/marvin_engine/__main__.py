"""Command line:
  python -m marvin_engine listen [--device N] [--mic-rate 48000]   live microphone, prints events as JSON lines
  python -m marvin_engine command FILE.wav                          intent + slot for a recorded command
  python -m marvin_engine wake FILE.wav                             MARVIN score of a recording (streaming rule)
  python -m marvin_engine devices                                   list audio input devices
"""
import argparse
import json

from . import Assistant, CommandRecognizer, WakeWord, read_wav


def main():
    ap = argparse.ArgumentParser(prog="marvin_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("listen"); li.add_argument("--device", default=None); li.add_argument("--mic-rate", type=int)
    li.add_argument("--wake-threshold", type=float); li.add_argument("--show-levels", action="store_true")
    sub.add_parser("command").add_argument("wav")
    sub.add_parser("wake").add_argument("wav")
    sub.add_parser("devices")
    a = ap.parse_args()
    if a.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
    elif a.cmd == "command":
        r = CommandRecognizer().predict(read_wav(a.wav)); print(json.dumps(r, indent=1))
    elif a.cmd == "wake":
        w = WakeWord(); s = w.score_clip(read_wav(a.wav))
        print(json.dumps({"score": s, "threshold": w.threshold, "marvin": s >= w.threshold}))
    else:
        dev = int(a.device) if a.device is not None and str(a.device).isdigit() else a.device

        def show(e):
            if e["type"] != "level" or a.show_levels:
                print(json.dumps({k: v for k, v in e.items() if k != "audio_wav"}), flush=True)

        asst = Assistant(on_event=show, device=dev, mic_rate=a.mic_rate, wake_threshold=a.wake_threshold)
        print('Say "Marvin", pause briefly, then a command (e.g. "set a timer for 30 seconds"). Ctrl+C to quit.', flush=True)
        try:
            asst.run_microphone()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
