"""Export-friendly views of the LiveAnon modules (weights shared with the PyTorch model).

PyTorch keeps streaming state in Python objects (KV caches, Zipformer states) and uses
dynamic shapes. For ONNX every state becomes an explicit input/output tensor of fixed
shape, and each streaming module is exported as a graph that runs one step:

    zipformer_chunk   fbank chunk + encoder states       -> projected content (1, chunk, C)
    decoder_prefill   prompt tokens + prompt content     -> decoder KV cache
    decoder_step      fed-back tokens + content frame    -> codes of one AR step, KV cache
    codec_encode      reference waveform frames          -> prompt codes   (once per prompt)
    codec_decode      codes of one frame + codec KV cache -> 320 samples

Decoder KV cache: (layers, 1, heads, capacity, head_dim). As in `KVCache`, the first `sink`
slots hold the prompt and the rest is a ring buffer. Attention runs over the buffer slots
directly (softmax does not depend on their order): prompt slots get no ALiBi penalty, ring
slots the penalty of their age, empty slots are masked.

JHCodec runs its attention in bfloat16 (flash-attn); the graphs round q/k/v and the
attention output to bfloat16 at the same points, and compute the rest in float32.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from liveanon.layers import _alibi_slopes
from liveanon.model import LiveAnon, _film


def _attend(q, k, v, bias):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + bias
    return torch.softmax(scores, dim=-1) @ v


# ============================================================================ AR decoder
class _DecoderBase(nn.Module):
    def __init__(self, model: LiveAnon):
        super().__init__()
        dec = model.decoder
        self.layers, self.films, self.norm = dec.layers, dec.content_film, dec.norm
        attn = dec.layers[0].self_attn
        self.n_heads, self.head_dim = attn.num_heads, attn.head_dim
        self.capacity = dec.max_seq_len
        with torch.no_grad():
            tables = torch.stack([proj(model.codebook_vectors[k])          # (K, V, H)
                                  for k, proj in enumerate(model.input_projs)])
        self.K, self.V = tables.shape[:2]
        self.register_buffer("tables", tables.reshape(self.K * self.V, -1))

    def embed(self, tokens, mask):
        """Sum of the codebook embeddings of the fed-back tokens: (1, T, K) -> (1, T, H)."""
        offs = torch.arange(self.K, device=tokens.device) * self.V
        rows = self.tables[tokens.clamp(0, self.V - 1) + offs]
        return (rows * mask.unsqueeze(-1)).sum(dim=2)

    def qkv(self, layer, x):
        a = layer.self_attn
        B, L, _ = x.shape
        split = lambda t: t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        return split(a.query(x)), split(a.key(x)), split(a.value(x))

    def block_out(self, i, layer, x, y, content):
        B, L = x.shape[:2]
        x = x + layer.self_attn.proj(y.transpose(1, 2).reshape(B, L, -1))
        x = x + layer.ff(layer.ln3(x))
        return _film(x, self.films[i], content)


class DecoderPrefill(_DecoderBase):
    """Prompt prefill. Row 0 of the stream is BOS; all columns are prompt (no ALiBi)."""

    def __init__(self, model: LiveAnon):
        super().__init__(model)
        self.bos = model.bos

    def forward(self, tokens, mask, content):
        x = self.embed(tokens, mask)
        T = x.shape[1]
        first = (torch.arange(T, device=x.device) == 0).view(1, T, 1)
        x = torch.where(first, self.bos, x)
        i = torch.arange(T, device=x.device)
        bias = torch.zeros(T, T, device=x.device).masked_fill(i[None, :] > i[:, None],
                                                              float("-inf"))
        ks, vs = [], []
        for n, layer in enumerate(self.layers):
            q, k, v = self.qkv(layer, layer.ln1(x))
            x = self.block_out(n, layer, x, _attend(q, k, v, bias), content)
            pad = (0, 0, 0, self.capacity - T)
            ks.append(F.pad(k, pad))
            vs.append(F.pad(v, pad))
        return torch.stack(ks), torch.stack(vs)


class DecoderStep(_DecoderBase):
    """One AR step on the explicit ring-buffer KV cache.

    tokens/mask (1, K): the token of codebook k at frame t - delay(k) (mask 0: none);
    x_add (1, 1, H):    added to the input (the null-content embedding in the delay tail);
    pos (1,):           positions already in the stream (prompt included);
    sink (1,):          prompt length (pinned slots).
    """

    def __init__(self, model: LiveAnon):
        super().__init__(model)
        self.register_buffer("slopes", _alibi_slopes(self.n_heads).view(self.n_heads, 1))
        with torch.no_grad():
            self.register_buffer("head_w", torch.cat([h.weight for h in model.heads]))
            self.register_buffer("head_b", torch.cat([h.bias for h in model.heads]))

    def forward(self, tokens, mask, x_add, content, k_cache, v_cache, pos, sink):
        x = self.embed(tokens.unsqueeze(1), mask.unsqueeze(1)) + x_add
        C = self.capacity
        ring = C - sink
        slot = torch.where(pos < C, pos, sink + (pos - sink) % ring)          # write slot
        s = torch.arange(C, device=x.device)
        age = (slot - s + ring) % ring
        n_ring = torch.clamp(pos + 1 - sink, max=ring)
        alibi = self.slopes * (-age).to(x.dtype)                             # (heads, C)
        bias = torch.where(s < sink, torch.zeros_like(alibi),
                           torch.where(age < n_ring, alibi,
                                       torch.full_like(alibi, float("-inf"))))
        bias = bias.view(1, self.n_heads, 1, C)
        write = (s == slot).view(1, 1, C, 1)

        ks, vs = [], []
        for n, layer in enumerate(self.layers):
            q, k_new, v_new = self.qkv(layer, layer.ln1(x))
            k = torch.where(write, k_new, k_cache[n])
            v = torch.where(write, v_new, v_cache[n])
            x = self.block_out(n, layer, x, _attend(q, k, v, bias), content)
            ks.append(k)
            vs.append(v)
        h = self.norm(x)[:, -1]
        logits = F.linear(h, self.head_w, self.head_b).view(1, self.K, -1)
        return logits.argmax(dim=-1), torch.stack(ks), torch.stack(vs)


# ============================================================================ Zipformer
class ZipformerChunk(nn.Module):
    """One chunk of the streaming Zipformer (as `ZipformerStream._run_encoder`) followed by
    LiveAnon's content projection.

    fbank (1, 2 * chunk + 13, 80):  fbank frames of the chunk (zero padded at the end);
    conv_keep (1,):  subsampled frames that exist (the rest are zeroed, as the EOF flush does);
    valid (1,):      output frames of this chunk (< chunk only for the last one).
    """

    def __init__(self, model: LiveAnon):
        super().__init__()
        self.enc, self.proj = model.content_encoder, model.content_proj
        self.chunk = int(self.enc.encoder.chunk_size[0])
        self.left = int(self.enc.encoder.left_context_frames[0])

    def forward(self, fbank, conv_keep, valid, embed_state, cache_padding, *states):
        sub = self.enc.encoder_embed
        x = sub.conv(fbank.unsqueeze(1))                                      # (1, C, chunk + 3, F)
        keep = torch.arange(x.shape[2], device=x.device) < conv_keep
        x = x * keep.view(1, 1, -1, 1).to(x.dtype)
        x, new_embed_state = sub.convnext.streaming_forward(x, cached_left_pad=embed_state)
        b, c, t, f = x.shape
        x = sub.out_norm(sub.out(x.transpose(1, 2).reshape(b, t, c * f)))     # (1, chunk, D)
        pad = torch.arange(self.chunk, device=x.device).view(1, -1) >= valid
        x = x * (~pad).unsqueeze(-1).to(x.dtype)
        mask = torch.cat([cache_padding, pad], dim=1)
        _, _, new_states = self.enc.encoder.streaming_forward(
            x.permute(1, 0, 2), valid, list(states), mask)
        feats = self.enc._tapped()                                            # (1, chunk, 512)
        return (self.proj(feats), mask[:, -self.left:], new_embed_state, *new_states)


# ============================================================================ JHCodec
def _bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def _rope(x, pos, inv_freq):
    """Interleaved (GPT-J) rotary embedding. x (1, T, H, D), pos (T,) float."""
    freqs = pos.view(-1, 1) * inv_freq.view(1, -1)                            # (T, D/2)
    cos, sin = torch.cos(freqs).unsqueeze(1), torch.sin(freqs).unsqueeze(1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class _CodecStack(nn.Module):
    """JHCodec transformer stack (`jhcodec.model.attention.Decoder`)."""

    def __init__(self, stack):
        super().__init__()
        self.layers = stack.layers
        a = stack.layers[0].self_attention
        self.n_heads, self.head_dim, self.window = a.n_heads, a.head_dim, a.window_size
        self.register_buffer("inv_freq", stack.rotary_emb.inv_freq.clone())

    def qkv(self, layer, x, pos):
        B, T, _ = x.shape
        qkv = layer.self_attention.Wqkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q = _bf16(_rope(qkv[:, :, 0], pos, self.inv_freq))
        k = _bf16(_rope(qkv[:, :, 1], pos, self.inv_freq))
        return q, k, _bf16(qkv[:, :, 2])

    def finish_layer(self, layer, x, context):
        B, T = x.shape[:2]
        a = layer.self_attention
        sa = a.out_proj(_bf16(context).reshape(B, T, -1))
        x = x + sa * (layer.layer_scale_factor * layer.sa_scale)
        mlp = layer.mlp(layer.pre_mlp_norm(x))
        return x + mlp * (layer.layer_scale_factor * layer.mlp_scale)


def _attn_bhtd(q, k, v, bias):
    """q (1, Tq, H, D), k/v (1, Tk, H, D) -> (1, Tq, H, D)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return _attend(q, k, v, bias).transpose(1, 2)


class CodecEncode(nn.Module):
    """Waveform frames (1, T, 320) -> codes (1, T, K). Used once per reference prompt."""

    def __init__(self, codec):
        super().__init__()
        m = codec.model
        self.linear_in, self.rvq_in, self.rvq = m.linear_in, m.rvq_in, m.rvq
        self.stack = _CodecStack(m.encoder)
        self.K = codec.n_codebooks

    @staticmethod
    def _nearest(x, codebook):
        d = (x.square().sum(-1, keepdim=True) + codebook.square().sum(-1).view(1, 1, -1)
             - 2 * x @ codebook.T)
        return d.argmin(dim=-1)

    def forward(self, frames):
        x = self.linear_in(frames)
        T = x.shape[1]
        i = torch.arange(T, device=x.device)
        rel = i.view(-1, 1) - i.view(1, -1)
        bias = torch.zeros(T, T, device=x.device).masked_fill(
            (rel < 0) | (rel > self.stack.window), float("-inf"))
        pos = i.to(x.dtype)
        for layer in self.stack.layers:
            q, k, v = self.stack.qkv(layer, layer.pre_sa_norm(x), pos)
            x = self.stack.finish_layer(layer, x, _attn_bhtd(q, k, v, bias))
        x = self.rvq_in(x)
        rvq = self.rvq
        codes = [self._nearest(rvq.semantic_down(x), rvq.semantic_vq.codebook.weight)]
        res = x
        for i in range(self.K - 1):
            idx = self._nearest(rvq.downs[i](res), rvq.vqs[i].codebook.weight)
            codes.append(idx)
            res = res - rvq.ups[i](rvq.vqs[i].codebook.weight[idx])
        return torch.stack(codes, dim=-1)


class CodecDecode(nn.Module):
    """One frame: codes (1, 1, K) + sliding-window KV cache -> 320 samples.

    k_cache/v_cache (layers, 1, window, heads, head_dim) hold the previous `window` frames
    (oldest first); pos (1,) is the index of this frame.
    """

    def __init__(self, codec):
        super().__init__()
        m = codec.model
        rvq = m.rvq
        self.register_buffer("tables", torch.stack(
            [rvq.semantic_up_vq.codebook.weight] + [u.codebook.weight for u in rvq.up_vqs]))
        self.rvq_out, self.linear_out = m.rvq_out, m.linear_out
        self.stack = _CodecStack(m.decoder)
        self.K = codec.n_codebooks

    def forward(self, codes, k_cache, v_cache, pos):
        x = self.tables[0][codes[:, :, 0]]
        for k in range(1, self.K):
            x = x + self.tables[k][codes[:, :, k]]
        x = self.rvq_out(x)
        W = self.stack.window
        key_pos = pos - W + torch.arange(W + 1, device=x.device)               # (W + 1,)
        bias = torch.where(key_pos >= 0, torch.zeros(W + 1, device=x.device),
                           torch.full((W + 1,), float("-inf"), device=x.device)).view(1, 1, 1, -1)
        p = pos.to(x.dtype)
        ks, vs = [], []
        for n, layer in enumerate(self.stack.layers):
            q, k_new, v_new = self.stack.qkv(layer, layer.pre_sa_norm(x), p)
            k = torch.cat([k_cache[n], k_new], dim=1)
            v = torch.cat([v_cache[n], v_new], dim=1)
            x = self.stack.finish_layer(layer, x, _attn_bhtd(q, k, v, bias))
            ks.append(k[:, 1:])
            vs.append(v[:, 1:])
        audio = self.linear_out(x).view(1, -1)
        return audio, torch.stack(ks), torch.stack(vs)
