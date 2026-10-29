#!/usr/bin/env python3
"""Convert a wav2vec2 CTC model to Core ML at three precisions.

The interesting part is not the conversion call. It is that the ONNX route
does not survive contact with this model, so the script traces the PyTorch
module and converts it directly. See REPORT.md for what failed and why.

    python src/convert.py --precision fp32
    python src/convert.py --precision all --out models
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import wave
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForCTC

DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
INPUT_NAME = "input_values"
SAMPLE_RATE = 16_000
MAX_SAMPLES = 30 * SAMPLE_RATE  # 30 s, the longest input the app recorded

PRECISIONS = ("fp32", "fp16", "int8")


def load_model(model_id: str) -> AutoModelForCTC:
    print(f"loading {model_id}")
    model = AutoModelForCTC.from_pretrained(model_id, torchscript=True)
    model.eval()
    return model


def trace(model: AutoModelForCTC) -> torch.jit.ScriptModule:
    """Trace at one second, then declare a flexible range on the input.

    Tracing fixes the graph; RangeDim fixes the shape the Core ML model will
    accept at runtime. Both are needed, and they are not the same thing.
    """
    dummy = torch.rand(1, SAMPLE_RATE)
    with torch.no_grad():
        return torch.jit.trace(model, dummy)


def input_types() -> list[ct.TensorType]:
    sequence = ct.RangeDim(lower_bound=1, upper_bound=MAX_SAMPLES, default=SAMPLE_RATE)
    return [ct.TensorType(name=INPUT_NAME, shape=(1, sequence))]


def read_wav(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz. Refuses anything else rather than resampling."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path.name}: need mono 16 kHz, got "
                             f"{handle.getnchannels()}ch {handle.getframerate()}Hz")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def calibration_data(directory: Path | None, count: int = 10) -> list[dict]:
    """Representative audio for INT8.

    Without a directory this falls back to random noise, which is a weak
    calibration set and is called out as a limitation in REPORT.md. Point
    --calibration-dir at ten 16 kHz mono wavs for numbers you can trust.
    """
    samples: list[dict] = []
    if directory is not None:
        paths = sorted(directory.glob("*.wav"))[:count]
        if not paths:
            raise SystemExit(f"no .wav files in {directory}")
        for path in paths:
            samples.append({INPUT_NAME: read_wav(path)[None, :]})
        print(f"calibrating on {len(samples)} wav files from {directory}")
        return samples

    rng = np.random.default_rng(0)
    print(f"calibrating on {count} random noise samples (weak; see REPORT.md)")
    for _ in range(count):
        length = int(rng.integers(SAMPLE_RATE, 10 * SAMPLE_RATE))
        noise = rng.standard_normal((1, length)).astype(np.float32)
        samples.append({INPUT_NAME: noise})
    return samples


def convert(model_id: str, precision: str, out_dir: Path,
            calibration_dir: Path | None) -> Path:
    traced = trace(load_model(model_id))

    kwargs: dict = {
        "inputs": input_types(),
        "convert_to": "mlprogram",
        "minimum_deployment_target": ct.target.iOS15,
    }
    if precision == "fp16":
        kwargs["compute_precision"] = ct.precision.FLOAT16
    elif precision == "int8":
        kwargs["optimization_config"] = ct.OptimizationConfig(
            global_config=ct.OpLinearQuantizerConfig(mode="linear_symmetric"),
        )
        kwargs["calibration_data"] = calibration_data(calibration_dir)

    print(f"converting {precision} ...")
    started = time.perf_counter()
    mlmodel = ct.convert(traced, **kwargs)
    elapsed = time.perf_counter() - started

    mlmodel.author = "Liman Shen"
    mlmodel.short_description = (
        f"wav2vec2 CTC phoneme recognition, {precision}, variable-length audio"
    )
    mlmodel.version = "1.0"
    mlmodel.input_description[INPUT_NAME] = (
        f"16 kHz mono audio, 1 to {MAX_SAMPLES} samples"
    )

    target = out_dir / f"Phoneme-{precision.upper()}.mlpackage"
    if target.exists():
        shutil.rmtree(target)
    mlmodel.save(str(target))
    print(f"  {target}  ({dir_size(target) / 1e6:.1f} MB, {elapsed:.1f}s)")
    return target


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--precision", choices=(*PRECISIONS, "all"), default="all")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--calibration-dir", type=Path, default=None,
                        help="directory of 16 kHz mono wavs for INT8 calibration")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = PRECISIONS if args.precision == "all" else (args.precision,)
    for precision in wanted:
        try:
            convert(args.model_id, precision, args.out, args.calibration_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"{precision} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if precision == "int8":
                print("INT8 is optional; FP32 and FP16 are the required artifacts.",
                      file=sys.stderr)
                continue
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
