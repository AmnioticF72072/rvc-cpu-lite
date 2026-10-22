# rvc-cpu-lite

RVC inference, ONNX only. No PyTorch, no fairseq, no CUDA. CPU.

Dependencies are `numpy`, `onnxruntime`, `soundfile`, `soxr` — about 140 MB installed,
versus 3.5 GB for a torch environment. The code here is a few hundred KB.

Measured on an i5-10400 (6C/12T, UHD 630, no dGPU): 20 s of input produced 20.00 s of output
in 30 s wall clock, RTF 1.51. An i5-12400 should land around RTF 1.0 — that number is an
estimate, I have not tested that CPU.

Inference only. No training, no realtime, no source separation.

## Usage

```bash
pip install -r requirements.txt
python convert.py -i input.mp3 -o out.wav -m models/your_voice.onnx --key 0
```

`--key` shifts pitch in semitones: +12 male→female, -12 female→male, 0 keeps the source range.

You need three ONNX files in `models/`: the voice model (export it with
`tools/export_generator.py`), `vec768_dyn.onnx` (ContentVec, `tools/export_contentvec.py`)
and `rmvpe.onnx`. They are not in the repo. See `docs/models.md`.

## The two things that will bite you

**The generator can only be exported at a fixed sequence length.** In the official
`attentions.py`, `_relative_position_to_absolute_position()` calls
`F.pad(x_flat, [0, int(length) - 1, ...])`. `int(length)` is a plain Python int, so the
tracer bakes the number in as a constant. The exported graph then only accepts the exact
frame count used at export time; anything else raises
`Reshape input_shape_size == size was false`, with shapes in the message that look plausible
and send you chasing the wrong bug. `do_constant_folding=False` does not help — the constant
comes from tracing, not from graph optimisation.

We export at 1024 frames (10.24 s) and chunk at inference time to match. 2048 frames killed a
16 GB machine during tracing.

**The chunking constants are coupled.** Max distance between two cut points
(`X_CENTER + 2 * X_QUERY`) plus context on both sides (`2 * X_PAD`) must stay ≤ `GEN_T` frames,
otherwise chunks get truncated and the output comes out shorter than the input. `X_QUERY` also
has to be smaller than `X_CENTER`, or cut points land at the start of a segment and you get a
pile of four-second fragments.

## Status

- Training / fine-tuning: out of scope.
- Realtime: no. Whole-file processing with overlap-trimmed chunk joins, a few seconds latency.
- m4a / aac / wma: libsndfile does not read them, convert to mp3 or wav first.
- faiss index retrieval: the hook exists (`--index`), faiss is not bundled.
- OpenVINO / iGPU / NPU: branch exists in `hw_profile.py`, untested — I had no compatible
  hardware. PRs welcome.

## License

MIT for the code. Upstream credits in `NOTICE.md`. No model weights are distributed here.
