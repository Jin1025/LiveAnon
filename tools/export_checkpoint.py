"""Convert a training checkpoint into a self-contained LiveAnon inference checkpoint.

The output holds the LiveAnon decoder weights and config only. The frozen third-party
modules (JHCodec, Zipformer encoder) are not included: they are loaded from their official
releases at inference time.

    python tools/export_checkpoint.py \
        --train_ckpt /path/to/step_latest.ckpt \
        --out checkpoints/liveanon.pt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liveanon.config import LiveAnonConfig  # noqa: E402

TRAIN_CONFIG_KEY = "livevoice_config"

# The released model is exactly this configuration; anything else is a different model.
REQUIRED = {
    "codec": "jhcodec",
    "content_source": "zipformer",
    "zipformer_layer": -1,
    "content_conditioning": "film",
    "content_cmn": "off",
    "content_refiner_layers": 0,
    "use_content_fsq": False,
    "speaker_encoder_type": "codec_prompt",
    "speaker_conditioning": "prefix",
    "codec_prompt_continuation": True,
    "codec_prompt_content": True,
    "codec_prompt_alibi_exempt": True,
    "use_delay_pattern": True,
    "use_mpm": False,
    "use_prosody": False,
    "use_cepstral": False,
}

# old name (regex) -> new name
RENAMES = [
    (r"^transformer\.start_token$", "bos"),
    (r"^transformer\.decoder_layers\.", "decoder.layers."),
    (r"^transformer\.decoder_norm\.", "decoder.norm."),
    (r"^transformer\.film_mlps\.", "decoder.content_film."),
    (r"^null_content_embedding$", "null_content"),
    (r"^null_film_feature$", "null_content_film"),
    (r"^decoder_input_projs\.", "input_projs."),
    (r"^codebook_heads\.", "heads."),
    (r"^sw2v_proj\.", "content_proj."),
]

# Parameters of the training graph that inference never touches.
DROP = [
    r"^codec_model\.",                                   # frozen JHCodec, released separately
    r"^content_extractor\.",                             # frozen Zipformer, released separately
    r"^null_speaker_embedding$", r"^null_prev_embedding$",  # speaker CFG / AR-input dropout
    r"^sw2v_to_hidden\.",                                # additive content path (unused with FiLM)
    r"^transformer\.encoder_norm\.",
    r"\.ln2\.",                                          # cross-attention LayerNorm (no cross-attn)
    r"\.self_attn\.(causal_attn_bias|alibi_attn_bias)$",  # deterministic buffers, rebuilt
]


def convert(train_ckpt: str) -> dict:
    obj = torch.load(train_ckpt, map_location="cpu", weights_only=False)
    tc = obj[TRAIN_CONFIG_KEY]
    bad = {k: (tc.get(k), v) for k, v in REQUIRED.items() if str(tc.get(k)) != str(v)}
    if bad:
        raise SystemExit(f"not a LiveAnon checkpoint; config differs (got, expected): {bad}")

    cfg = LiveAnonConfig(
        sample_rate=tc["sample_rate"],
        n_codebooks=tc["n_codebooks_predict"],
        codebook_size=tc["jhcodec_codebook_size"],
        hidden_dim=tc["hidden_dim"],
        num_heads=tc["num_heads"],
        num_decoder_layers=tc["num_decoder_layers"],
        ffn_dim=tc["ffn_dim"],
        dropout=tc["dropout"],
        max_seq_len=tc["max_seq_len"],
        delay_cap=tc["delay_cap"],
        content_proj_dim=tc["content_proj_dim"],
        # checkpoints saved before this field existed were trained with the fixed value 16
        zipformer_chunk_size=tc.get("zipformer_chunk_size", 16),
        zipformer_align_pad_frames=tc["zipformer_align_pad_frames"],
    )

    sd, vectors = {}, {}
    for key, value in obj["state_dict"].items():
        key = key[len("model."):]
        if any(re.search(p, key) for p in DROP):
            continue
        m = re.match(r"^codebook_vectors_(\d+)$", key)
        if m:
            vectors[int(m.group(1))] = value
            continue
        for pat, new in RENAMES:
            if re.search(pat, key):
                key = re.sub(pat, new, key)
                break
        else:
            raise SystemExit(f"unmapped parameter: {key}")
        sd[key] = value
    sd["codebook_vectors"] = torch.stack([vectors[k] for k in sorted(vectors)])

    return {
        "config": cfg.to_dict(),
        "state_dict": sd,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = convert(args.train_ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    n = sum(v.numel() for v in ckpt["state_dict"].values())
    print(f"saved {args.out}  ({len(ckpt['state_dict'])} tensors, {n / 1e6:.1f}M values)")


if __name__ == "__main__":
    main()
