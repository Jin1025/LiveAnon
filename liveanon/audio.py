"""Audio I/O and reference-prompt preparation."""
from __future__ import annotations

import soundfile as sf
import torch
import torchaudio

_FRAME_MS = 20.0


def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
    """Mono float32 waveform at `sr` (no level normalisation)."""
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).mean(dim=1)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav


def save_audio(path: str, wav: torch.Tensor, sr: int = 16000) -> None:
    sf.write(path, wav.detach().float().cpu().numpy(), sr, subtype="PCM_16")


def _frame_rms(w: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    hop = max(1, int(sr * _FRAME_MS / 1000.0))
    n = w.numel() // hop
    return w[: n * hop].reshape(n, hop).pow(2).mean(dim=1).sqrt(), hop


def _trim_silence(w: torch.Tensor, sr: int, top_db: float) -> torch.Tensor:
    rms, hop = _frame_rms(w, sr)
    if rms.numel() == 0:
        return w
    idx = (rms > rms.max() * 10.0 ** (-abs(top_db) / 20.0)).nonzero().flatten()
    if idx.numel() == 0:
        return w
    return w[int(idx[0]) * hop: min(w.numel(), (int(idx[-1]) + 1) * hop)]


def _crop_ending_on_speech(w: torch.Tensor, sr: int, crop_sec: float, top_db: float,
                           tail_ms: float = 200.0) -> torch.Tensor:
    """The `crop_sec` window whose last `tail_ms` is loudest (then: loudest overall)."""
    n = int(crop_sec * sr)
    if n <= 0 or w.numel() <= n:
        return w
    rms, hop = _frame_rms(w, sr)
    if rms.numel() == 0:
        return w[-n:]
    floor = rms.max() * 10.0 ** (-abs(top_db) / 20.0)
    n_tail, n_win = max(1, int(tail_ms / _FRAME_MS)), max(1, n // hop)
    best, best_key = None, None
    for end in range(n_win, rms.numel() + 1):
        seg = rms[end - n_win: end]
        tail = seg[-n_tail:].mean()
        key = (float(tail >= floor), float(tail), float(seg.mean()))
        if best_key is None or key > best_key:
            best, best_key = end, key
    stop = min(w.numel(), best * hop)
    return w[max(0, stop - n): stop]


def prepare_reference(wav: torch.Tensor, sr: int = 16000, crop_sec: float = 3.0,
                      top_db: float = 35.0) -> torch.Tensor:
    """Trim silence and crop a `crop_sec` prompt that ENDS ON SPEECH.

    The model continues the prompt's token stream, so a prompt that ends in a pause is a
    strong prior to keep generating silence.
    """
    wav = _trim_silence(wav, sr, top_db)
    return _crop_ending_on_speech(wav, sr, crop_sec, top_db)
