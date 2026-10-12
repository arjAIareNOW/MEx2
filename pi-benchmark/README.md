# Raspberry Pi 5 benchmark (ELKIT appliance)

Reproduces the on-device numbers on the ME2 poster. Speed only: no recordings are included or needed
(the test audio is generated). Runs the same ONNX models the appliance uses.

## Run (on the Pi, from this folder)
    pip install numpy onnxruntime
    python tools/benchmark.py

## Result (Raspberry Pi 5, 2026-10-12)
- Command recognition (intent + slot), 5 s of audio: p95 193 ms, real-time factor 0.03
- onnxruntime, 2 threads; CPU 59 °C, no throttling

The "real usage" section needs the appliance's own voice log and is skipped here.
