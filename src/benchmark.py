#!/usr/bin/env python3
"""Benchmark the converted packages against the PyTorch reference.

Reports four things per precision: size on disk, cold load time, warm
inference latency at four input lengths, and token agreement with PyTorch.

    python src/benchmark.py --models models > benchmarks/results.md

Numbers are only meaningful on the machine named in the output. An M1 and
an M3 do not produce the same table.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForCTC

DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
INPUT_NAME = "input_values"
SAMPLE_RATE = 16_000
LENGTHS = (1, 5, 10, 30)
PRECISIONS = ("FP32", "FP16", "INT8")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def device_name() -> str:
    if platform.system() != "Darwin":
        return platform.processor() or platform.machine()
    for key in ("machdep.cpu.brand_string",):
        try:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                 text=True, check=True).stdout.strip()
            if out:
                return out
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return platform.machine()


def audio(seconds: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, seconds * SAMPLE_RATE)).astype(np.float32)


def load_torch(model_id: str) -> AutoModelForCTC:
    model = AutoModelForCTC.from_pretrained(model_id)
    model.eval()
    return model


def torch_logits(model: AutoModelForCTC, samples: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(samples))
    return out.logits.detach().numpy()


def agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Fraction of time steps where the argmax token matches."""
    reference = np.squeeze(reference)
    candidate = np.squeeze(candidate)
    steps = min(reference.shape[0], candidate.shape[0])
    if steps == 0:
        return 0.0
    a = reference[:steps].argmax(axis=-1)
    b = candidate[:steps].argmax(axis=-1)
    return float((a == b).mean())


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
    return ordered[index]


def measure(package: Path, model: AutoModelForCTC, runs: int) -> dict:
    load_started = time.perf_counter()
    mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
    load_seconds = time.perf_counter() - load_started

    output_name = mlmodel.get_spec().description.output[0].name
    row = {
        "name": package.name,
        "size_mb": dir_size(package) / 1e6,
        "load_s": load_seconds,
        "latency": {},
        "first_run_s": None,
        "agreement": None,
    }

    for seconds in LENGTHS:
        samples = audio(seconds, seed=seconds)
        started = time.perf_counter()
        mlmodel.predict({INPUT_NAME: samples})
        first = time.perf_counter() - started
        if row["first_run_s"] is None:
            row["first_run_s"] = first

        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            mlmodel.predict({INPUT_NAME: samples})
            timings.append(time.perf_counter() - started)
        row["latency"][seconds] = (statistics.median(timings),
                                   percentile(timings, 0.95))

        if seconds == 5:
            reference = torch_logits(model, samples)
            predicted = mlmodel.predict({INPUT_NAME: samples})[output_name]
            row["agreement"] = agreement(reference, predicted)

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    packages = [args.models / f"Phoneme-{p}.mlpackage" for p in PRECISIONS]
    packages = [p for p in packages if p.exists()]
    if not packages:
        raise SystemExit(f"no packages in {args.models}; run `make convert` first")

    model = load_torch(args.model_id)
    rows = [measure(package, model, args.runs) for package in packages]

    lines = [
        f"Device: **{device_name()}** · macOS {platform.mac_ver()[0]} · "
        f"{args.runs} runs per length · median (p95)",
        "",
        "| Package | Size | Cold load | First run | "
        + " | ".join(f"{s}s" for s in LENGTHS)
        + " | Token agreement |",
        "|---|---:|---:|---:|"
        + "---:|" * len(LENGTHS)
        + "---:|",
    ]
    for row in rows:
        latency = " | ".join(
            f"{row['latency'][s][0] * 1000:.0f} ms ({row['latency'][s][1] * 1000:.0f})"
            for s in LENGTHS if s in row["latency"]
        )
        agreement_text = ("—" if row["agreement"] is None
                          else f"{row['agreement'] * 100:.1f}%")
        lines.append(
            f"| {row['name']} | {row['size_mb']:.1f} MB | "
            f"{row['load_s']:.2f} s | {row['first_run_s']:.2f} s | "
            f"{latency} | {agreement_text} |"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
