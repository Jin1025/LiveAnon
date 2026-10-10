"""LiveAnon configuration (inference)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class LiveAnonConfig:
    # audio / codec (JHCodec: 16 kHz, 320-sample hop -> 50 fps, 8 RVQ codebooks of 1024)
    sample_rate: int = 16000
    n_codebooks: int = 8
    codebook_size: int = 1024

    # AR decoder
    hidden_dim: int = 768
    num_heads: int = 8
    num_decoder_layers: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 1024          # KV-cache capacity (ring buffer; the prompt is pinned)

    # capped delay pattern: delay(k) = min(k, delay_cap)
    delay_cap: int = 3

    # content: streaming Zipformer bottleneck features -> FiLM
    content_dim: int = 512           # Zipformer tap dimension
    content_proj_dim: int = 256
    zipformer_chunk_size: int = 16
    zipformer_left_context: int = 128
    zipformer_align_pad_frames: int = -6   # <0 trims the front = lookahead (-6 -> 120 ms)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LiveAnonConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
