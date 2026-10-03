#!/usr/bin/env python3
"""Decode embedded DEMAND Opus rows once for efficient random WAV access."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import wave
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("dataset/noise/demand/data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/noise/demand/wav16k"),
    )
    parser.add_argument("--expected-count", type=int, default=6800)
    return parser.parse_args()


def valid_cached_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getframerate() == 16_000
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getnframes() >= 16_000
            )
    except (FileNotFoundError, wave.Error):
        return False


def safe_output_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".ogg":
        raise ValueError(f"Unsafe or unexpected DEMAND row name: {name}")
    return root.joinpath(*relative.parts).with_suffix(".wav")


def main() -> None:
    args = parse_args()
    shards = sorted(args.parquet_dir.glob("16k-*.parquet"))
    if len(shards) != 6:
        raise RuntimeError(f"Expected 6 DEMAND 16 kHz shards, found {len(shards)}")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = 0
    decoded = 0
    reused = 0
    names: set[str] = set()
    environments: set[str] = set()
    shard_records = []

    for shard in shards:
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        parquet = pq.ParquetFile(shard)
        shard_records.append(
            {"name": shard.name, "rows": parquet.metadata.num_rows, "sha256": digest}
        )
        for batch in parquet.iter_batches(batch_size=32, columns=["audio", "name"]):
            for payload_scalar, name_scalar in zip(batch.column(0), batch.column(1)):
                name = name_scalar.as_py()
                if name in names:
                    raise RuntimeError(f"Duplicate DEMAND row name: {name}")
                names.add(name)
                environments.add(name.split("/", 1)[0])
                destination = safe_output_path(args.output, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if valid_cached_wav(destination):
                    reused += 1
                else:
                    audio, sample_rate = sf.read(
                        io.BytesIO(payload_scalar.as_py()),
                        dtype="int16",
                        always_2d=False,
                    )
                    if sample_rate != 16_000 or audio.ndim != 1:
                        raise RuntimeError(
                            f"Expected mono 16 kHz DEMAND audio, got {sample_rate} Hz "
                            f"with shape {audio.shape}: {name}"
                        )
                    temporary = destination.with_suffix(".wav.tmp")
                    sf.write(temporary, audio, sample_rate, format="WAV", subtype="PCM_16")
                    os.replace(temporary, destination)
                    decoded += 1
                rows += 1
                if rows % 250 == 0:
                    print(json.dumps({"rows_processed": rows, "decoded": decoded, "reused": reused}), flush=True)

    if rows != args.expected_count or len(names) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} unique rows, got {rows} rows and {len(names)} names"
        )
    cached_files = sorted(args.output.rglob("*.wav"))
    if len(cached_files) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} cached WAVs, found {len(cached_files)}")

    manifest = {
        "format": "mono PCM16 WAV",
        "sample_rate": 16_000,
        "samples": len(cached_files),
        "environments": sorted(environments),
        "source_shards": shard_records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"complete": True, "decoded": decoded, "reused": reused, **manifest}), flush=True)


if __name__ == "__main__":
    main()
