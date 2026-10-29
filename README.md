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
reference. Mine, Apple M4, ten runs per length, median:

| Package | Size | Cold load | 1 s | 5 s | 10 s | 30 s | Token agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phoneme-FP32 | 1262 MB | 5.4 s | 24 ms | 83 ms | 166 ms | 534 ms | 100.0% |
| Phoneme-FP16 | 631 MB | 64 s | 13 ms | 65 ms | 127 ms | 402 ms | 100.0% |
| Phoneme-INT8 | 318 MB | 4.0 s | 24 ms | 83 ms | 166 ms | 536 ms | 99.6% |

Two rows tell the story. FP16 is the model that shipped: warm latency roughly half of
FP32, same token agreement, half the size — but a cold load that costs about a minute,
because Core ML compiles the half-precision program in-process on first use. A one-time
runtime download hides that behind a progress bar; a bundled model would show it, so the
real fix is shipping the compiled `.mlmodelc`. INT8 lands exactly where weight-only
quantization implies: half of FP16's disk footprint, no latency win (activations stay in
float), and argmax divergence on 0.4% of frames.

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
  shipping candidate, INT8 is weight-only quantization of the FP32 package. Each keeps the
  same input name and shape so they can be swapped behind one Swift protocol.

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

- **INT8 is weight-only.** Activations stay in float, because coremltools' activation
  quantizer raises a scale-dtype mismatch on this graph in both 8.3 and 9.0. The exact
  error is in `docs/REPORT.md`. The size win is real; the latency win is not the one
  "INT8" usually implies.
- **Thirty seconds is a hard ceiling.** Longer audio has to be chunked by the caller.
- **No streaming.** This is utterance-level: record, then score. A real-time version
  needs a different input contract and a VAD in front of it.
- **One model, one vocabulary.** The tokenizer config ships with the package; a different
  phoneme set means a different conversion.

## What I'd do next

- Pre-compile the FP16 package to `.mlmodelc` at build time — a bundled model that takes
  a minute on first launch is not acceptable, and the runtime download that hid it only
  works when the model ships over the network.
- Ship a real calibration/eval set, even a small one, so INT8's accuracy cost is measured
  on speech instead of noise.
- Add a `swift/` sample app with the microphone and a latency readout, so the benchmark
  matches the experience.
- Compare against Apple's own `SFSpeechRecognizer` for the same task, because "we need a
  custom model" is a claim that should be tested.

## License

MIT. See [LICENSE](LICENSE).
