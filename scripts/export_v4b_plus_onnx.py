"""Export a V4-B+ best.pt to dynamic batch/time ONNX and check it against PyTorch.

Usage: python scripts/export_v4b_plus_onnx.py RUN_DIR   (writes RUN_DIR/v4b_plus.onnx + onnx_export.json)
"""
import json, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from vcm.tc_dscnn_v4b_plus import IntentTCDSCNNV4BPlus
from vcm.dscnn_exp1 import sha256_file


def main():
    run = Path(sys.argv[1])
    onnx_path, status_path = run / 'v4b_plus.onnx', run / 'onnx_export.json'
    if onnx_path.exists():
        raise FileExistsError(f'Refusing to overwrite {onnx_path}')
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    cfg = ck['model_config']
    model = IntentTCDSCNNV4BPlus(cfg['num_classes'], cfg['dropout'], cfg['width']).eval()
    model.load_state_dict(ck['model_state'])
    x, lengths = torch.randn(2, 1, 64, 301), torch.tensor([301, 211])
    torch.onnx.export(model, (x, lengths), onnx_path, input_names=['log_mel', 'frame_lengths'],
                      output_names=['logits'], opset_version=17,
                      dynamic_axes={'log_mel': {0: 'batch', 3: 'time'}, 'frame_lengths': {0: 'batch'}, 'logits': {0: 'batch'}})
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    cases = []
    for frames in (41, 173, 501):
        x = torch.randn(1, 1, 64, frames); lens = torch.tensor([frames])
        with torch.inference_mode():
            expected = model(x, lens).numpy()
        actual = session.run(['logits'], {'log_mel': x.numpy(), 'frame_lengths': lens.numpy()})[0]
        cases.append({'frames': frames, 'max_abs_logit_diff': float(np.abs(expected - actual).max()),
                      'same_argmax': bool((expected.argmax(1) == actual.argmax(1)).all())})
    status = {'exported': True, 'opset': 17, 'onnx_file': onnx_path.name, 'onnx_bytes': onnx_path.stat().st_size,
              'onnx_sha256': sha256_file(onnx_path), 'checkpoint_sha256': sha256_file(run / 'best.pt'),
              'inputs': {'log_mel': 'float32 [B,1,64,T]', 'frame_lengths': 'int64 [B]'}, 'output': 'logits [B,19]',
              'onnxruntime_version': ort.__version__, 'pytorch_vs_onnxruntime': cases}
    assert all(c['same_argmax'] and c['max_abs_logit_diff'] < 1e-3 for c in cases), cases
    status_path.write_text(json.dumps(status, indent=2) + '\n'); print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
