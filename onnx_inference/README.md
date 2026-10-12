# LiveAnon: ONNX / TensorRT inference (optional)

This folder exports the full LiveAnon streaming pipeline to ONNX and runs it with ONNX Runtime or TensorRT, without PyTorch.
The main implementation is the PyTorch one in the repository root. This is an optional deployment path.

Each streaming module is exported as a graph that runs one step, with its state as explicit inputs and outputs:

| Graph | Runs | Input → output |
|---|---|---|
| `zipformer_chunk` | once per encoder chunk | fbank frames + encoder states → content features |
| `decoder_prefill` | once per prompt | prompt tokens + prompt content → decoder KV cache |
| `decoder_step` | once per frame | fed-back tokens + content frame + KV cache → codec tokens |
| `codec_encode` | once per prompt | reference waveform → prompt tokens |
| `codec_decode` | once per frame | codec tokens + codec KV cache → 320 samples |

The decoder KV cache is a fixed-size ring buffer with the prompt pinned at the front, so a stream of any length runs with fixed shapes.

## Model weights

Only the LiveAnon checkpoints are distributed by this project (Hugging Face: to be continued).
[JHCodec](https://github.com/jhcodec843/jhcodec) (MIT) and the icefall streaming Zipformer are not redistributed. `export_onnx.py` downloads them from their official releases and converts them on your machine, together with the LiveAnon checkpoint.

## Export

Run in the PyTorch environment of the main README, with `onnx` and `onnxruntime-gpu` installed:
```bash
python onnx_inference/export_onnx.py \
    --ckpt <path_to_checkpoint> \
    --out onnx_inference/exported/<name>
```
After exporting, the script runs each graph on the same inputs as the PyTorch modules and prints the differences.

## Inference

PyTorch is not needed at runtime:
```bash
pip install -r onnx_inference/requirements.txt

python onnx_inference/inference.py \
    --onnx_dir onnx_inference/exported/<name> \
    --source <path_to_source_audio> \
    --reference <path_to_reference_audio> \
    --output <path_to_output_audio> \
    --provider tensorrt  # or cuda / cpu
```
- The source is always processed as a stream (`--chunk_ms`, default 20 ms); the output does not depend on the block size.
- With `--provider tensorrt`, the engines for the per-frame graphs are built on the first run and cached in `<onnx_dir>/trt_cache`. The prompt graphs run on the CUDA provider.
- TF32 is disabled (`NVIDIA_TF32_OVERRIDE=0`), so the FP32 engines give the same tokens as PyTorch. `--fp16` builds FP16 engines; their tokens differ from PyTorch.

Python API:
```python
from runtime import OnnxLiveAnon          # run from onnx_inference/

model = OnnxLiveAnon("exported/<name>", provider="tensorrt")
stream = model.stream(reference)          # 16 kHz float32 numpy array
for block in blocks:
    out = stream.push(block)              # anonymized samples available so far
out = stream.finish()
```

## Matching the PyTorch output

The speaker prompt (the reference's codec tokens and content features) can come from two places:

- `--reference`: computed by the ONNX graphs. The streaming Zipformer differs from the offline one on the last chunk, and the codec attention is not flash-attn. The prompt is therefore slightly different, and the autoregressive decoding takes a different path.
- `--prompt`: precomputed once with PyTorch. The ONNX output then matches the PyTorch streaming output token for token.
```bash
python onnx_inference/make_prompt.py \
    --ckpt <path_to_checkpoint> \
    --reference <path_to_reference_audio> \
    --out prompt.npz

python onnx_inference/inference.py --onnx_dir onnx_inference/exported/<name> \
    --prompt prompt.npz --source <path_to_source_audio> --output <path_to_output_audio>
```
A fixed pseudo-speaker only needs one prompt file.
