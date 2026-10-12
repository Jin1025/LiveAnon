"""Precompute a speaker prompt (codec tokens + content features) for the ONNX runtime.

The PyTorch model encodes the reference with JHCodec and the offline Zipformer. The ONNX
runtime can do the same with its own graphs (`OnnxLiveAnon.prompt`), but its streaming
Zipformer differs from the offline one on the last chunk and its codec attention is not
flash-attn, so the AR decoding starts from a slightly different prompt. A prompt saved
here makes the ONNX output follow the PyTorch model exactly.

    python onnx_inference/make_prompt.py --ckpt checkpoints/liveanon_chunk320ms.pt \
        --reference path/to/reference.wav --out prompt.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liveanon import LiveAnon  # noqa: E402
from liveanon.audio import load_audio, prepare_reference  # noqa: E402


@torch.no_grad()
def make_prompt(model: LiveAnon, reference: torch.Tensor) -> dict:
    """reference: (T,) 16 kHz, already trimmed / cropped. Same steps as `ARSession`."""
    ref = reference.unsqueeze(0).to(model.device)
    codes = model.codec.encode(ref)[:, :model.cfg.n_codebooks]
    content = model._align(model.encode_content(ref), codes.shape[2])
    return {"codes": codes[0].cpu().numpy(), "content": content[0].cpu().numpy()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref_sec", type=float, default=3.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LiveAnon.from_pretrained(args.ckpt, device=args.device)
    sr = model.cfg.sample_rate
    ref = prepare_reference(load_audio(args.reference, sr), sr, args.ref_sec)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **make_prompt(model, ref))


if __name__ == "__main__":
    main()
