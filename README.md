# model-on-device

Phoneme recognition on an iPhone, without a server.

A reproducible conversion of a wav2vec2 CTC model to Core ML at three precisions, with
the benchmarks that decide which one ships.

## The problem

A speech model that scores pronunciation has to listen to a child read, return feedback
in well under a second, and do it without uploading a recording of a child to a server.
That rules out the obvious architecture: audio → network → model → network → result.

Running it locally creates three constraints at once: the model has to fit in an app
bundle, it has to load fast enough that the first word is not delayed, and it has to accept
audio of whatever length the user actually speaks instead of a fixed window.

## Why this repo exists

This is a clean rebuild of work I did on a production iOS app. The original lives in a
private company repo, so the code here is rewritten around a public model
(`vitouphy/wav2vec2-xls-r-300m-timit-phoneme`) and the same technique. No customer data,
no product code, no company paths.

The interesting part is not that it converts. It is that the obvious route — ONNX in the
middle — does not work, and the [write-up](docs/REPORT.md) explains exactly where it
fails and why the direct trace does not.

## Quick start

Requires macOS on Apple Silicon. Python 3.11: `coremltools` does not ship wheels for
newer interpreters yet.

```sh
make setup          # .venv with coremltools, torch, transformers
make convert        # FP32, FP16, INT8 into models/   (~1.3 GB download, first run)
make bench          # writes benchmarks/results.md
```

One precision at a time:

```sh
.venv/bin/python src/convert.py --precision fp16
```

## Results

Run `make bench` on your own machine — the numbers are device-specific and a table
copied from someone else's laptop is marketing, not measurement. It reports size on
disk, cold load, warm latency at 1/5/10/30 seconds, and token agreement with the PyTorch
reference.

| Package | Size | Cold load | 1 s | 5 s | 10 s | 30 s | Token agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phoneme-FP32 | — | — | — | — | — | — | — |
| Phoneme-FP16 | — | — | — | — | — | — | — |
| Phoneme-INT8 | — | — | — | — | — | — | — |

The row that matters is not the fastest one. It is the smallest package that still
agrees with PyTorch on the tokens, because a pronunciation score that is 2% wrong is
worse than a slow one that is right.

## How it works

```
PyTorch (transformers)  →  torch.jit.trace  →  coremltools  →  .mlpackage
                              + RangeDim                        (mlprogram, iOS 15+)
```

- **Trace, don't export.** The ONNX detour fails on frontend support and adds a numerical
  approximation. `docs/REPORT.md` has the four error messages.
- **One second to trace, thirty seconds to run.** `torch.jit.trace` fixes the graph;
  `ct.RangeDim(1, 480_000, 16_000)` declares the runtime input range. Tracing and shape
  declaration are separate steps.
- **Three precisions, one interface.** FP32 is the reference, FP16 is the default
  shipping candidate, INT8 is the experiment. Each keeps the same input name and shape so
  they can be swapped behind one Swift protocol.

## Layout

```
src/convert.py       trace + convert, one precision or all three
src/benchmark.py     size, load, latency, and agreement vs PyTorch
swift/SmokeTest.swift loads a package and runs one inference on device
docs/REPORT.md       why the ONNX route failed
benchmarks/          generated results
models/              generated packages (gitignored)
```

## Limitations

- **INT8 calibration defaults to random noise.** Without `--calibration-dir` the
  quantizer calibrates on synthetic audio, which is a weak proxy for speech. Point it at
  ten 16 kHz mono wavs before trusting the INT8 row.
- **Thirty seconds is a hard ceiling.** Longer audio has to be chunked by the caller.
- **No streaming.** This is utterance-level: record, then score. A real-time version
  needs a different input contract and a VAD in front of it.
- **One model, one vocabulary.** The tokenizer config ships with the package; a different
  phoneme set means a different conversion.

## What I'd do next

- Calibrate INT8 on real speech and report the accuracy cost per millisecond saved.
- Add a `swift/` sample app with the microphone and a latency readout, so the benchmark
  matches the experience.
- Compare against Apple's own `SFSpeechRecognizer` for the same task, because "we need a
  custom model" is a claim that should be tested.

## License

MIT. See [LICENSE](LICENSE).
