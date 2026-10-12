"""LiveAnon inference with ONNX Runtime / TensorRT (no PyTorch needed).

Export the graphs first with `export_onnx.py`. The source is always processed as a
stream (`--chunk_ms` blocks); the output does not depend on the block size.

    python onnx_inference/inference.py --onnx_dir onnx_inference/exported/liveanon_chunk320ms \
        --source path/to/source.wav --reference path/to/reference.wav \
        --output output/anonymized.wav --provider tensorrt
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from audio import load_audio, prepare_reference, save_audio
from runtime import OnnxLiveAnon, load_prompt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx_dir", required=True, help="output of export_onnx.py")
    p.add_argument("--source", required=True, help="utterance to anonymize")
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--reference", help="pseudo-speaker reference audio")
    ref.add_argument("--prompt", help="prompt .npz (make_prompt.py or --save_prompt)")
    p.add_argument("--output", default="output/anonymized.wav")
    p.add_argument("--ref_sec", type=float, default=3.0, help="reference prompt length")
    p.add_argument("--save_prompt", default=None, help="also save the computed prompt (.npz)")
    p.add_argument("--chunk_ms", type=int, default=20, help="input block size")
    p.add_argument("--provider", default="cuda", choices=["cuda", "tensorrt", "cpu"])
    p.add_argument("--fp16", action="store_true", help="TensorRT FP16 engines")
    p.add_argument("--device", type=int, default=0, help="GPU index")
    args = p.parse_args()

    model = OnnxLiveAnon(args.onnx_dir, provider=args.provider, device_id=args.device,
                         trt_fp16=args.fp16)
    sr = model.meta["config"]["sample_rate"]
    if args.prompt:
        prompt = load_prompt(args.prompt)
    else:
        prompt = model.prompt(prepare_reference(load_audio(args.reference, sr), sr,
                                                args.ref_sec))
        if args.save_prompt:
            np.savez(args.save_prompt, **prompt)

    source = load_audio(args.source, sr)
    audio = model.generate(prompt, source, block=args.chunk_ms * sr // 1000)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_audio(args.output, audio, sr)


if __name__ == "__main__":
    main()
