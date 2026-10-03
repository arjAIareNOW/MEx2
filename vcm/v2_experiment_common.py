from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import time
import wave
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
import matplotlib
from torch import Tensor, nn
from torch.utils.data import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import collate_variable_audio, split_files, split_fingerprint, split_from_validation_speakers
from .dscnn_exp1 import (
    build_oos_metrics, collect_environment, evaluate_final_test, sha256_file,
    write_environment_files, write_oos_csv,
)
from .metrics import selected_confusions, write_class_metrics
from .model import parameter_count
from .train import class_weights, evaluate, make_loader, resolve_device, set_reproducible, train_epoch, write_history

DATASET_REVISION = "da92a79ffde3031d5bb2a25138d9dd7d9f7ed006"
EXPECTED_CLASSES = (
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF", "LIGHT_ON",
    "LIST_REMINDERS", "MESSAGE", "NEXT", "OUT_OF_SCOPE", "PAUSE", "PLAY_MUSIC", "STOP",
    "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP", "WEATHER",
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    model_name: str
    output_name: str
    model_factory: Callable[[int, float], nn.Module]
    architecture: Callable[[], dict[str, object]]
    architecture_family: str


class ConservativeWaveformAugmentor:
    def __init__(self, noise_dir: Path, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.noises: list[Tensor] = []
        self.noise_files = sorted(noise_dir.glob("*.wav"))
        if not self.noise_files:
            raise FileNotFoundError(f"No background-noise WAV files in {noise_dir}")
        for path in self.noise_files:
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != sample_rate or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                    raise ValueError(f"Noise must be mono 16-kHz PCM16: {path}")
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
            self.noises.append(torch.from_numpy(pcm).float().div_(32768.0))

    def __call__(self, waveform: Tensor) -> Tensor:
        # Non-circular zero-filled time shift, p=0.5, maximum +/-100 ms.
        if torch.rand(()) < 0.5:
            shift = int(torch.randint(-1600, 1601, ()).item())
            if shift > 0:
                waveform = nn.functional.pad(waveform, (shift, 0))[..., : waveform.shape[-1]]
            elif shift < 0:
                waveform = nn.functional.pad(waveform[..., -shift:], (0, -shift))
        # Training-only background mixing, p=0.5, uniformly sampled 10-25 dB SNR.
        if torch.rand(()) < 0.5:
            noise = self.noises[int(torch.randint(len(self.noises), ()).item())]
            if len(noise) < waveform.shape[-1]:
                noise = noise.repeat((waveform.shape[-1] + len(noise) - 1) // len(noise))
            start = int(torch.randint(len(noise) - waveform.shape[-1] + 1, ()).item())
            segment = noise[start:start + waveform.shape[-1]].unsqueeze(0)
            speech_rms = waveform.square().mean().sqrt().clamp_min(1e-5)
            noise_rms = segment.square().mean().sqrt().clamp_min(1e-5)
            snr_db = 10.0 + 15.0 * float(torch.rand(()))
            scale = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
            waveform = (waveform + segment * scale).clamp(-1.0, 1.0)
        return waveform


class V2IntentDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, table: pa.Table, indices: Sequence[int], mapping: dict[str, int],
                 augmentor: ConservativeWaveformAugmentor | None = None) -> None:
        self.audio = table.column("audio"); self.commands = table.column("command")
        self.indices = np.asarray(indices, dtype=np.int64); self.mapping = mapping; self.augmentor = augmentor
        self.max_samples = 80_000; self.win_length = 400
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000, n_fft=512, win_length=400, hop_length=160,
            f_min=20.0, f_max=8000.0, n_mels=64, power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        row = int(self.indices[item]); audio = self.audio[row].as_py()["bytes"]
        with wave.open(io.BytesIO(audio), "rb") as handle:
            if handle.getframerate() != 16_000 or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Expected mono 16-kHz PCM16 WAV")
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
        waveform = torch.from_numpy(pcm).float().div_(32768.0).unsqueeze(0)[..., :self.max_samples]
        if self.augmentor is not None: waveform = self.augmentor(waveform)
        if waveform.shape[-1] < self.win_length:
            waveform = nn.functional.pad(waveform, (0, self.win_length - waveform.shape[-1]))
        features = self.to_db(self.mel(waveform))
        features = (features - features.mean()) / features.std().clamp_min(1e-5)
        return features, self.mapping[self.commands[row].as_py()]


def parse_args(spec: ExperimentSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=spec.experiment)
    parser.add_argument("--data", type=Path, default=Path("datasetv2"))
    parser.add_argument("--output", type=Path, default=Path("models") / spec.output_name)
    parser.add_argument("--split-manifest", type=Path, default=Path("splits/main_vcm_speaker_split_v2.json"))
    parser.add_argument("--noise-dir", type=Path, default=Path("dataset/speech_commands/_background_noise_"))
    parser.add_argument("--augmentation", choices=("conservative", "none"), default="conservative")
    parser.add_argument("--epochs", type=int, default=60); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=10); parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=0); parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23101); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    fixed = {"epochs": 60, "batch_size": 64, "learning_rate": .001, "weight_decay": .0001,
             "patience": 10, "dropout": .20, "workers": 0, "cpu_threads": 4, "seed": 23101}
    changed = {key: (getattr(args, key), value) for key, value in fixed.items() if getattr(args, key) != value}
    if changed: raise ValueError(f"Fixed V2 protocol changed: {changed}")


def prepare_v2(args: argparse.Namespace) -> tuple[list[str], dict[str, int], Any, list[Path], pa.Table, dict[str, Any]]:
    git_executable = shutil.which("git")
    if git_executable is None:
        revision = DATASET_REVISION
        revision_verification = "frozen_expected_revision_git_unavailable"
    else:
        try:
            revision = subprocess.check_output(
                [git_executable, "-C", str(args.data), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except subprocess.CalledProcessError as error:
            message = error.stderr.strip() if error.stderr else str(error)
            raise RuntimeError(f"Git is available but Dataset V2 revision verification failed: {message}") from error
        revision_verification = "verified_with_git"
    if revision != DATASET_REVISION: raise ValueError(f"datasetv2 revision mismatch: {revision}")
    files = sorted((args.data / "data").glob("train-*.parquet"))
    if not files: raise FileNotFoundError("datasetv2 default/train Parquet files missing")
    tables = [pq.read_table(path, columns=["audio", "command", "speaker_id", "source"]) for path in files]
    table = pa.concat_tables(tables)
    if len(table) != 10_733: raise ValueError(f"Expected 10,733 V2 train rows, got {len(table)}")
    manifest = json.loads(args.split_manifest.read_text())
    if manifest["dataset_revision"] != DATASET_REVISION or manifest["train_parquet_fingerprint"] != split_fingerprint(files):
        raise ValueError("V2 split does not match datasetv2")
    commands = table.column("command").to_pylist(); speakers = table.column("speaker_id").to_pylist()
    classes = sorted(set(commands)); mapping = {name: i for i, name in enumerate(classes)}
    if tuple(classes) != EXPECTED_CLASSES: raise ValueError("V2 class mapping changed")
    split = split_from_validation_speakers(commands, speakers, frozenset(manifest["validation_speaker_ids"]))
    if (len(split.train_indices), len(split.validation_indices)) != (manifest["train_sample_count"], manifest["validation_sample_count"]):
        raise ValueError("V2 split reconstruction count mismatch")
    if split.train_speakers & split.validation_speakers: raise ValueError("V2 speaker overlap")
    details = {"manifest": str(args.split_manifest), "manifest_sha256": sha256_file(args.split_manifest),
               "dataset_revision": revision, "dataset_revision_verification": revision_verification,
               "train_parquet_fingerprint": split_fingerprint(files),
               "train": len(split.train_indices), "validation": len(split.validation_indices),
               "train_speakers": len(split.train_speakers), "validation_speakers": len(split.validation_speakers),
               "speaker_overlap": 0, "train_class_distribution": split.train_class_counts,
               "validation_class_distribution": split.validation_class_counts}
    return classes, mapping, split, files, table, details


def augmentation_config(args: argparse.Namespace, augmentor: ConservativeWaveformAugmentor | None) -> dict[str, Any]:
    return {"enabled": augmentor is not None, "training_only": True, "validation_test_augmented": False,
            "time_shift": {"probability": .5, "maximum_ms": 100, "mode": "zero-filled non-circular"},
            "background_noise": {"probability": .5, "snr_db": [10, 25], "source": str(args.noise_dir),
                                 "files": [str(path) for path in (augmentor.noise_files if augmentor else [])]}}


def save_v2_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]], classes: Sequence[str], title: str) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    positions = np.arange(len(classes))
    axis.set(xticks=positions, yticks=positions, xticklabels=classes, yticklabels=classes,
             xlabel="Predicted command", ylabel="True command", title=f"{title} — V2 test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def padding_difference(model: nn.Module, sample: Tensor, length: int) -> float:
    model.eval(); original = sample[..., :length]; padded = nn.functional.pad(original, (0, 197)); lengths = torch.tensor([length])
    with torch.inference_mode(): return float((model(original, lengths) - model(padded, lengths)).abs().max())


def run_preflight(spec: ExperimentSpec, args: argparse.Namespace) -> None:
    if args.output.exists(): raise FileExistsError(f"Refusing existing production output: {args.output}")
    classes, mapping, split, _, table, details = prepare_v2(args)
    augmentor = ConservativeWaveformAugmentor(args.noise_dir) if args.augmentation == "conservative" else None
    plain = V2IntentDataset(table, split.train_indices[:32], mapping, None)
    samples = sorted([plain[i] for i in range(len(plain))], key=lambda value: value[0].shape[-1])
    batch = [samples[0], samples[len(samples)//2], samples[-1]]
    features, lengths, labels = collate_variable_audio(batch)
    model = spec.model_factory(20, .20); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    before = next(model.parameters()).detach().clone(); model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(features, lengths); loss = nn.CrossEntropyLoss()(logits, labels); loss.backward(); optimizer.step()
    updated = not torch.equal(before, next(model.parameters()).detach())
    diff = padding_difference(model, features[:1], int(lengths.min()))
    augmented_shape = list(V2IntentDataset(table, split.train_indices[:1], mapping, augmentor)[0][0].shape)
    report = {"experiment": spec.experiment, "dataset_revision": DATASET_REVISION, "split": details,
              "classes": len(classes), "real_audio_shape": list(features.shape), "frame_lengths": lengths.tolist(),
              "propagated_lengths": model.output_lengths(lengths).tolist(), "output_shape": list(logits.shape),
              "forward": True, "backward": True, "adamw_update": updated, "padding_difference": diff,
              "trainable_parameters": parameter_count(model), "total_parameters": sum(p.numel() for p in model.parameters()),
              "class_weight_source_rows": sum(split.train_class_counts.values()),
              "augmentation_real_audio_shape": augmented_shape, "augmentation": augmentation_config(args, augmentor),
              "test_opened": False, "holdout_opened": False, "numerals_opened": False,
              "supplemental_synth_opened": False, "production_output_created": args.output.exists()}
    if logits.shape != (3, 20) or diff > 1e-5 or not updated: raise AssertionError(report)
    print(json.dumps(report, indent=2))


def create_comparison_if_ready(models: Path) -> None:
    sources = (("ER-DSCNN V2", "pure convolutional", "vcm_er_dscnn_v2"),
               ("Light-CRNN V2", "CNN plus temporal recurrence", "vcm_light_crnn_v2"),
               ("TC-DSCNN V2", "pure convolutional temporal", "vcm_tc_dscnn_v2"))
    paths = [(name, family, models / directory / "metrics.json") for name, family, directory in sources]
    if not all(path.is_file() for _, _, path in paths): return
    rows = []
    for name, family, path in paths:
        data = json.loads(path.read_text()); test = data["final_test"]; complexity = data["complexity"]
        latency = test["a100_gpu_inference_latency"]; oos = test["oos"]
        rows.append({"Model": name, "Architecture family": family,
                     "Parameters": complexity["trainable_parameters"], "Checkpoint MB": complexity["checkpoint_size_mb"],
                     "Test accuracy": test["accuracy"], "Test macro-F1": test["macro_f1"],
                     "Test weighted-F1": test["weighted_f1"], "OOS FAR": oos["false_accept_rate"]["decimal"],
                     "A100 median latency": latency["median_ms"], "A100 p95 latency": latency["p95_ms"],
                     "Validation→test macro-F1 difference": test["macro_f1"] - data["best_validation"]["macro_f1"],
                     "Dataset revision": DATASET_REVISION,
                     "Split SHA-256": data["configuration"]["split"]["manifest_sha256"]})
    csv_path = models / "main_vcm_v2_comparison.csv"; json_path = models / "main_vcm_v2_comparison.json"
    if csv_path.exists() or json_path.exists(): return
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps({"external_crnn_v2_included": False, "rows": rows}, indent=2) + "\n")


def train(spec: ExperimentSpec, args: argparse.Namespace) -> None:
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    device = resolve_device(args.device)
    if device.type != "cuda" or "A100" not in torch.cuda.get_device_properties(device).name:
        raise RuntimeError("Production V2 training requires the requested A100")
    classes, mapping, split, train_files, train_table, split_details = prepare_v2(args)
    args.output.mkdir(parents=True, exist_ok=False); environment = collect_environment(device); write_environment_files(args.output, environment)
    augmentor = ConservativeWaveformAugmentor(args.noise_dir) if args.augmentation == "conservative" else None
    model = spec.model_factory(20, .20).to(device); trainable = parameter_count(model); total = sum(p.numel() for p in model.parameters())
    config = {"experiment": spec.experiment, "model": spec.model_name, "architecture": spec.architecture(),
              "dataset": {"path": str(args.data), "revision": DATASET_REVISION, "config": "default",
                          "training_source": "default/train only", "expected_test_rows": 4443,
                          "excluded": ["supplemental_synth", "holdout", "numerals"]},
              "features": {"sample_rate": 16000, "n_mels": 64, "n_fft": 512, "window": 400,
                           "hop": 160, "f_min": 20, "f_max": 8000,
                           "normalization": "per-utterance log-mel mean/std", "variable_duration": True},
              "split": split_details, "train_speaker_ids": sorted(split.train_speakers),
              "validation_speaker_ids": sorted(split.validation_speakers),
              "training": {"optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001,
                           "loss": "training-only inverse-frequency weighted CrossEntropyLoss",
                           "scheduler": "CosineAnnealingLR", "batch_size": 64, "maximum_epochs": 60,
                           "patience": 10, "seed": 23101, "workers": 0, "cpu_threads": 4,
                           "selection": "validation macro-F1 only"},
              "augmentation": augmentation_config(args, augmentor),
              "complexity": {"trainable_parameters": trainable, "total_parameters": total,
                             "macs": "not measured", "flops": "not measured"},
              "environment_at_start": environment}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_set = V2IntentDataset(train_table, split.train_indices, mapping, augmentor)
    validation_set = V2IntentDataset(train_table, split.validation_indices, mapping, None)
    train_loader = make_loader(train_set, args, device, True); validation_loader = make_loader(validation_set, args, device, False)
    criterion = nn.CrossEntropyLoss(weight=class_weights(classes, split.train_class_counts, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    history=[]; best_f1=-1.; best_epoch=0; stale=0; stopped=False
    start=datetime.now(timezone.utc); wall=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path=args.output/"best.pt"
    for epoch in range(1,61):
        lr=optimizer.param_groups[0]["lr"]; train_loss,train_accuracy=train_epoch(model,train_loader,criterion,optimizer,device)
        report,val_loss=evaluate(model,validation_loader,criterion,device,classes)
        row={"epoch":epoch,"learning_rate":lr,"train_loss":train_loss,"train_accuracy":train_accuracy,
             "validation_loss":val_loss,**{f"validation_{k}":v for k,v in report.summary.items()}}
        history.append(row); write_history(args.output/"training_history.csv",history); print(json.dumps(row),flush=True)
        if report.summary["macro_f1"]>best_f1:
            best_f1=report.summary["macro_f1"];best_epoch=epoch;stale=0
            torch.save({"model_state":model.state_dict(),"model_name":spec.model_name,
                        "model_config":{"num_classes":20,"dropout":.20},"class_to_index":mapping,"epoch":epoch,
                        "validation_report":{**report.summary,"loss":val_loss,"per_class":report.per_class,
                                             "confusion_matrix":report.confusion_matrix},"configuration":config},checkpoint_path)
        else: stale+=1
        scheduler.step()
        if stale>=10: stopped=True;break
    end=datetime.now(timezone.utc);wall_seconds=time.perf_counter()-wall
    peak_alloc=torch.cuda.max_memory_allocated(device);peak_reserved=torch.cuda.max_memory_reserved(device)
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True);model.load_state_dict(checkpoint["model_state"]);model.eval()
    # First test access: only after validation-only selection, best.pt reload, and eval().
    test_files=split_files(args.data,"test"); test_tables=[pq.read_table(p,columns=["audio","command","file","speaker_id","source","transcript","duration_s"]) for p in test_files]
    test_table=pa.concat_tables(test_tables) if len(test_tables)>1 else test_tables[0]
    test_commands=test_table.column("command").to_pylist()
    if len(test_table)!=4443 or test_commands.count("OUT_OF_SCOPE")!=76: raise ValueError("Expected V2 test 4,443 rows and 76 OOS")
    test_set=V2IntentDataset(test_table,np.arange(len(test_table)),mapping,None)
    test_report,test_loss,latency,targets,predictions,confidences=evaluate_final_test(model,test_set,criterion,device,classes)
    records=[{"file":test_table.column("file")[i].as_py(),"speaker_id":test_table.column("speaker_id")[i].as_py(),
              "source":test_table.column("source")[i].as_py(),"transcript":test_table.column("transcript")[i].as_py()} for i in range(len(test_table))]
    oos=build_oos_metrics(targets,predictions,confidences,classes,records);false=oos["oos_falsely_classified_as_valid_commands"];total_oos=oos["oos_test_sample_count"]
    oos["false_accept_rate"]={"numerator":false,"denominator":total_oos,"decimal":false/total_oos,
                              "percentage":100*false/total_oos,"definition":"OOS accepted as valid / total OOS"}
    write_oos_csv(args.output/"oos_false_classifications.csv",oos);write_class_metrics(args.output/"class_metrics.csv",{"test":test_report})
    save_v2_confusion_matrix(args.output/"confusion_matrix.png",test_report.confusion_matrix,classes,spec.experiment)
    complexity=config["complexity"]|{"checkpoint_size_bytes":checkpoint_path.stat().st_size,
                                     "checkpoint_size_mb":checkpoint_path.stat().st_size/1_000_000}
    training={"epochs":len(history),"best_epoch":best_epoch,"wall_time_seconds":wall_seconds,
              "best_validation":checkpoint["validation_report"],"start_utc":start.isoformat(),"end_utc":end.isoformat(),
              "stopped_early":stopped,"peak_cuda_allocated_bytes":peak_alloc,"peak_cuda_reserved_bytes":peak_reserved}
    final={**test_report.summary,"loss":test_loss,"per_class":test_report.per_class,"confusion_matrix":test_report.confusion_matrix,
           "selected_confusions":selected_confusions(test_report.confusion_matrix,mapping),"oos":oos,
           "a100_gpu_inference_latency":latency}
    metrics={"experiment":spec.experiment,"configuration":config,"training_result":training,
             "best_validation":checkpoint["validation_report"],"final_test":final,"complexity":complexity,
             "holdout_used":False,"numerals_used":False,"supplemental_synth_used":False}
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (args.output/"submission_summary.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (args.output/"submission_summary.txt").write_text(json.dumps(metrics,indent=2)+"\n")
    (args.output/"COMPLETED").write_text("completed; V2 test evaluated once\n")
    create_comparison_if_ready(args.output.parent)


def main_for_experiment(spec: ExperimentSpec) -> None:
    args=parse_args(spec);validate_args(args);set_reproducible(args.seed,args.cpu_threads)
    if args.preflight: run_preflight(spec,args)
    else: train(spec,args)
