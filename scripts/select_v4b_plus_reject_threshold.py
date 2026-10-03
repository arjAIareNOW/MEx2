"""Choose a V4-B+ out-of-scope reject threshold on VALIDATION speakers only, then report it.

Rule (fixed in advance): max-softmax threshold = 5th percentile of in-scope validation
confidences, i.e. 95% of validation commands are accepted. OOS validation clips are the
main-train OUT_OF_SCOPE rows of validation speakers (never used for training the 19-way head).
The frozen threshold is then applied to test and holdout (in-scope accept/accuracy, OOS FAR).

Usage: python scripts/select_v4b_plus_reject_threshold.py RUN_DIR [--data ~/ai231]
Writes RUN_DIR/reject_threshold.json
"""
import argparse, json, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import pyarrow as pa
import torch
from vcm.final19 import CLASSES, read
from vcm.tc_dscnn_v4b_plus import IntentTCDSCNNV4BPlus
from vcm.v2_experiment_common import V2IntentDataset
from vcm.data import collate_variable_audio

ACCEPT_TARGET = 0.95


@torch.inference_mode()
def scores(model, table, indices):
    mapping = {c: i for i, c in enumerate(CLASSES)}; mapping['OUT_OF_SCOPE'] = -1
    loader = torch.utils.data.DataLoader(V2IntentDataset(table, indices, mapping), batch_size=64,
                                         collate_fn=collate_variable_audio)
    conf, pred, label = [], [], []
    for x, lengths, y in loader:
        p = model(x, lengths).softmax(1)
        conf += p.max(1).values.tolist(); pred += p.argmax(1).tolist(); label += y.tolist()
    return np.array(conf), np.array(pred), np.array(label)


def report(conf, pred, label, t):
    ins, oos = label >= 0, label < 0
    accepted = conf >= t
    return {'in_scope_clips': int(ins.sum()), 'oos_clips': int(oos.sum()),
            'in_scope_accept_rate': float(accepted[ins].mean()) if ins.any() else None,
            'in_scope_accuracy_all': float((pred[ins] == label[ins]).mean()) if ins.any() else None,
            'in_scope_correct_and_accepted': float(((pred == label) & accepted)[ins].mean()) if ins.any() else None,
            'oos_false_accept_rate': float(accepted[oos].mean()) if oos.any() else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); ap.add_argument('--data', type=Path, default=Path.home() / 'ai231')
    a = ap.parse_args()
    out = a.run / 'reject_threshold.json'
    if out.exists():
        raise FileExistsError(out)
    ck = torch.load(a.run / 'best.pt', map_location='cpu', weights_only=True); cfg = ck['model_config']
    model = IntentTCDSCNNV4BPlus(cfg['num_classes'], cfg['dropout'], cfg['width']).eval(); model.load_state_dict(ck['model_state'])
    torch.set_num_threads(8)
    manifest = json.loads((a.run / 'split_manifest.json').read_text())
    val_speakers = set(manifest['validation_speaker_ids'])
    main_train = read(sorted((a.data / 'data').glob('train-*.parquet')), True)
    fil = read(sorted((a.data / 'supplemental_fil').glob('train-*.parquet')), True)
    table = pa.concat_tables([main_train, fil])
    commands, speakers = table['command'].to_pylist(), table['speaker_id'].to_pylist()
    val_in = manifest['validation_indices']
    val_oos = [i for i in range(len(main_train)) if commands[i] == 'OUT_OF_SCOPE' and speakers[i] in val_speakers]
    conf, pred, label = scores(model, table, val_in + val_oos)
    threshold = float(np.quantile(conf[label >= 0], 1 - ACCEPT_TARGET))
    result = {'rule': f'5th percentile of in-scope validation max-softmax (accept {ACCEPT_TARGET:.0%} of validation commands)',
              'threshold': threshold, 'validation': report(conf, pred, label, threshold),
              'pipeline_assumption': 'runs after MARVIN wake word; rejected clips -> OUT_OF_SCOPE'}
    for split in ('test', 'holdout'):
        tab = read(sorted((a.data / 'data').glob(f'{split}-*.parquet')), True)
        idx = [i for i, c in enumerate(tab['command'].to_pylist()) if c in CLASSES or c == 'OUT_OF_SCOPE']
        result[split] = report(*scores(model, tab, idx), threshold)
        result[split + '_no_gate_oos_false_accept_rate'] = 1.0
    out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
