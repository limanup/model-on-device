#!/usr/bin/env python3
"""Convert a wav2vec2 CTC model to Core ML.

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
from pathlib import Path

import coremltools as ct
import torch
from transformers import AutoModelForCTC

DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
INPUT_NAME = "input_values"
SAMPLE_RATE = 16_000
MAX_SAMPLES = 30 * SAMPLE_RATE  # 30 s, the longest input the app recorded

PRECISIONS = ("fp32", "fp16")


def load_model(model_id: str) -> AutoModelForCTC:
    print(f"loading {model_id}")
    # torchscript=True is what makes this model traceable: it returns a tuple
    # of tensors rather than a ModelOutput dataclass, which the tracer can
    # flatten. transformers 5 removed the flag, which is why it is pinned
    # below 5 in requirements.txt.
    model = AutoModelForCTC.from_pretrained(model_id, torchscript=True)
    model.eval()
    return model


def trace(model: AutoModelForCTC) -> torch.jit.ScriptModule:
    """Trace at one second, then declare a flexible range on the input.

    Tracing fixes the graph; RangeDim fixes the shape the Core ML model will
    accept at runtime. Both are needed, and they are not the same thing.

    model.eval() is not optional. A traced dropout is a wrong model and it
    fails silently, scoring nothing like the checkpoint it came from.
    """
    dummy = torch.rand(1, SAMPLE_RATE)
    with torch.no_grad():
        return torch.jit.trace(model, dummy)


def input_types() -> list[ct.TensorType]:
    sequence = ct.RangeDim(lower_bound=1, upper_bound=MAX_SAMPLES, default=SAMPLE_RATE)
    return [ct.TensorType(name=INPUT_NAME, shape=(1, sequence))]


def convert(model_id: str, precision: str, out_dir: Path) -> Path:
    traced = trace(load_model(model_id))

    kwargs: dict = {
        "inputs": input_types(),
        "convert_to": "mlprogram",
        "minimum_deployment_target": ct.target.iOS15,
    }
    if precision == "fp16":
        kwargs["compute_precision"] = ct.precision.FLOAT16
    else:
        # mlprogram defaults to FLOAT16 weights. Without this line "FP32"
        # produces a 631 MB model that is already half precision, and the
        # comparison is meaningless.
        kwargs["compute_precision"] = ct.precision.FLOAT32

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
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = PRECISIONS if args.precision == "all" else (args.precision,)
    for precision in wanted:
        convert(args.model_id, precision, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
