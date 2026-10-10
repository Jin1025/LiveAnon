"""JHCodec wrapper: 16 kHz, 320-sample hop (50 fps), 8 RVQ codebooks of size 1024.

JHCodec: https://github.com/jhcodec843/jhcodec (weights: https://huggingface.co/jhcodec/jhcodec)
"""
from __future__ import annotations

import torch
import torch.nn as nn

JHCODEC_HF_REPO = "jhcodec/jhcodec"
JHCODEC_CKPT_NAME = "jhcodec_mimi_1000000.pt"


class JHCodec(nn.Module):
    sample_rate = 16000
    hop_length = 320

    def __init__(self, n_codebooks: int = 8, config_path: str | None = None,
                 ckpt_path: str | None = None):
        super().__init__()
        import omegaconf
        from jhcodec.model.codec import JHCodecMimi

        if config_path is None or ckpt_path is None:
            from huggingface_hub import hf_hub_download
            config_path = config_path or hf_hub_download(JHCODEC_HF_REPO, "config.json")
            ckpt_path = ckpt_path or hf_hub_download(JHCODEC_HF_REPO, JHCODEC_CKPT_NAME)

        cfg = omegaconf.OmegaConf.load(config_path)
        self.model = JHCodecMimi(cfg.model)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state.get("model_state_dict", state), strict=True)
        # eval() precomputes the up-projected codebooks, so it must run AFTER loading.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_codebooks = n_codebooks

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()            # always frozen
        return self

    def _k(self, k: int, device) -> torch.Tensor:
        return torch.tensor([int(k)], dtype=torch.long, device=device)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T_samples) 16 kHz -> codes (B, K, T_frames)."""
        audio = audio.float()
        indices, _ = self.model.encode(audio, self._k(self.n_codebooks, audio.device),
                                       inference_cache=None)
        return indices.permute(0, 2, 1).contiguous().long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, cache=None, return_cache: bool = False):
        """codes (B, K, T_frames) -> audio (B, T_samples) at 16 kHz.

        For streaming, pass the returned `cache` back with the next frames
        (`return_cache=True`): the causal decoder continues from its KV cache.
        """
        indices = codes.long().permute(0, 2, 1).contiguous()
        audio, cache = self.model.decode(indices, self._k(codes.shape[1], codes.device),
                                         inference_cache=cache)
        audio = audio.squeeze(1) if audio.dim() == 3 else audio
        return (audio, cache) if return_cache else audio
