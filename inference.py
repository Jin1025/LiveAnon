"""LiveAnon inference: anonymize a source utterance with a pseudo-speaker reference.

    # offline: the whole source at once
    python inference.py --ckpt checkpoints/liveanon_chunk160ms.pt \
        --source path/to/source.wav --reference path/to/reference.wav \
        --output output/anonymized.wav

    # streaming: source samples arrive in --chunk_ms blocks, audio comes out as it is ready
    python inference.py ... --streaming --chunk_ms 20
"""
from __future__ import annotations

import argparse
import os

import torch

from liveanon import LiveAnon
from liveanon.audio import load_audio, prepare_reference, save_audio


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="LiveAnon checkpoint")
    p.add_argument("--source", required=True, help="utterance to anonymize")
    p.add_argument("--reference", required=True, help="pseudo-speaker prompt")
    p.add_argument("--output", default="output/anonymized.wav")
    p.add_argument("--ref_sec", type=float, default=3.0, help="reference prompt length")
    p.add_argument("--streaming", action="store_true",
                   help="feed the source chunk by chunk (real streaming)")
    p.add_argument("--chunk_ms", type=int, default=20, help="input block size for --streaming")
    p.add_argument("--delay_cap", type=int, default=None,
                   help="override the checkpoint's delay cap (use the value it was trained with)")
    p.add_argument("--no_fast", action="store_true",
                   help="disable the inference optimisations (plain eager decoding)")
    p.add_argument("--temperature", type=float, default=0.0, help="<= 0: greedy decoding")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    overrides = {} if args.delay_cap is None else {"delay_cap": args.delay_cap}
    model = LiveAnon.from_pretrained(args.ckpt, device=args.device, fast=not args.no_fast,
                                     **overrides)
    sr = model.cfg.sample_rate

    source = load_audio(args.source, sr).unsqueeze(0).to(args.device)
    reference = prepare_reference(load_audio(args.reference, sr), sr, args.ref_sec)
    reference = reference.unsqueeze(0).to(args.device)
    sampling = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

    if args.streaming:
        stream = model.stream(reference, **sampling)
        n = args.chunk_ms * sr // 1000
        pieces = [stream.push(source[:, i:i + n]) for i in range(0, source.shape[-1], n)]
        pieces.append(stream.finish())
        audio = torch.cat(pieces, dim=-1)[0]
    else:
        codes = model.generate(reference, source, **sampling)
        audio = model.decode_to_audio(codes)[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_audio(args.output, audio, sr)


if __name__ == "__main__":
    main()
