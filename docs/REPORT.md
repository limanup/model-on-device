# The ONNX route did not work

The obvious path for PyTorch → Core ML is PyTorch → ONNX → Core ML. It failed here, and
the failure is worth writing down because the error messages point at Core ML when the
problem is the frontend.

## What I tried, in order

**1. Export to ONNX.** `optimum-cli export onnx --task automatic-speech-recognition`
produced a graph, but with a warning that mattered:

```
values not close enough, max diff: 6.103515625e-05 (atol: 1e-05)
- logits: max diff = 6.103515625e-05
```

A CTC model's logits feed an argmax. A 6e-05 drift is harmless for argmax in most frames
and it is not harmless at a decision boundary. The export "succeeded" with a documented
numerical divergence before Core ML was involved.

**2. Convert the ONNX graph.** `coremltools.convert(onnx.load(path))` failed to identify
the source framework:

```
Unable to determine the type of the model, i.e. the source framework.
Please provide the value of argument "source"
```

Passing `source="pytorch"` for an ONNX graph is a hint, not a fix — it tells the
converter to use the wrong frontend and hope the ops line up.

**3. Fix the shape.** The next error was real and unrelated:

```
For mlprogram, inputs with infinite upper_bound is not allowed.
```

ONNX exported the sequence axis as unbounded. Core ML's `mlprogram` requires a finite
upper bound, so the input has to be declared explicitly — which is also why the direct
route needs `ct.RangeDim`.

**4. Fix the argument.** With a bounded shape, the converter then rejected the input
object itself:

```
Model must either be a TorchScript object ... received: <class 'pathlib.PosixPath'>
```

At that point the pattern was clear: every failure was the converter being asked to
accept a graph it did not have a first-class frontend for. `optimum` does not offer
PyTorch → Core ML, and the ONNX detour adds a numerical approximation plus an op-coverage
lottery.

## What worked instead

Trace the PyTorch module and convert the traced program directly:

```python
model = AutoModelForCTC.from_pretrained(model_id, torchscript=True)
model.eval()
traced = torch.jit.trace(model, torch.rand(1, 16_000))

coreml_model = ct.convert(
    traced,
    inputs=[ct.TensorType(
        name="input_values",
        shape=(1, ct.RangeDim(lower_bound=1, upper_bound=480_000, default=16_000)),
    )],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
)
```

Three things make that work:

- `torchscript=True` makes the model return tuples instead of a dataclass, which the
  tracer can handle. The output is flattened and renamed (`var_1732` in one build), so
  read the output name from the spec rather than hard-coding it.
- `torch.jit.trace` fixes the graph at one second. `RangeDim` separately declares the
  shape the model will accept at runtime. They solve different problems and both are
  required.
- Tracing warns that a Python boolean in the attention path becomes a constant. It is
  benign for this model at this input range, and it is the kind of thing to re-check if
  the audio length changes by an order of magnitude.

## What I would try first next time

Go straight to tracing. Budget the ONNX detour at zero unless the model has custom ops
that the PyTorch frontend rejects — and test the export's numerical agreement before
converting, not after.
