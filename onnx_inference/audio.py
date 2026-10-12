"""numpy versions of `liveanon.audio` for the PyTorch-free ONNX runtime."""
from __future__ import annotations

import numpy as np
import soundfile as sf

_FRAME_MS = 20.0


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Mono float32 waveform at `sr`. Resampling uses scipy's polyphase filter, which is
    not bit-identical to torchaudio's (use `make_prompt.py` for an exact prompt)."""
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, file_sr)
        wav = resample_poly(wav, sr // g, file_sr // g).astype(np.float32)
    return wav


def save_audio(path: str, wav: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, np.asarray(wav, np.float32), sr, subtype="PCM_16")


def _frame_rms(w: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    hop = max(1, int(sr * _FRAME_MS / 1000.0))
    n = len(w) // hop
    return np.sqrt((w[: n * hop].reshape(n, hop) ** 2).mean(axis=1)), hop


def prepare_reference(wav: np.ndarray, sr: int = 16000, crop_sec: float = 3.0,
                      top_db: float = 35.0) -> np.ndarray:
    """Trim silence and crop a `crop_sec` prompt that ends on speech
    (same rule as `liveanon.audio.prepare_reference`)."""
    rms, hop = _frame_rms(wav, sr)
    if len(rms):
        idx = np.nonzero(rms > rms.max() * 10.0 ** (-abs(top_db) / 20.0))[0]
        if len(idx):
            wav = wav[idx[0] * hop: min(len(wav), (idx[-1] + 1) * hop)]
    n = int(crop_sec * sr)
    if n <= 0 or len(wav) <= n:
        return wav
    rms, hop = _frame_rms(wav, sr)
    if len(rms) == 0:
        return wav[-n:]
    floor = rms.max() * 10.0 ** (-abs(top_db) / 20.0)
    n_tail, n_win = max(1, int(200.0 / _FRAME_MS)), max(1, n // hop)
    best, best_key = None, None
    for end in range(n_win, len(rms) + 1):
        seg = rms[end - n_win: end]
        tail = seg[-n_tail:].mean()
        key = (float(tail >= floor), float(tail), float(seg.mean()))
        if best_key is None or key > best_key:
            best, best_key = end, key
    stop = min(len(wav), best * hop)
    return wav[max(0, stop - n): stop]
