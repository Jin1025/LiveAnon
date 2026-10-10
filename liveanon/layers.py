"""Causal Transformer decoder layers with ALiBi and a ring-buffer KV cache.

The reference prompt occupies the first positions of the AR stream. Those positions are
pinned in the KV cache as an attention sink (never evicted) and exempted from the ALiBi
distance penalty, so the speaker information stays reachable regardless of how long the
generated continuation gets.

Two decoding paths share the weights:
  * eager  -- `forward` with a `KVCache` (any batch size / device);
  * graph  -- `graph_*`: a fixed-capacity cache that a CUDA Graph can replay one token at a
              time (B = 1). Used until the cache is full, then the state is handed back to
              the eager ring buffer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Per-layer KV cache: `sink` pinned slots followed by a ring buffer.

    Layout of the backing tensors (B, H, capacity, head_dim):
        [0, sink)          the first `sink` tokens ever written (the prompt)
        [sink, capacity)   ring buffer holding the most recent tokens
    """

    def __init__(self, capacity: int, sink: int):
        self.capacity = int(capacity)
        self.sink = max(1, int(sink))
        self.k = self.v = None
        self.write_pos = self.sink
        self.filled = 0          # valid entries in the ring
        self.length = 0          # sink + filled

    def reset(self, k: torch.Tensor, v: torch.Tensor) -> None:
        B, H, L, D = k.shape
        self.k = torch.zeros(B, H, self.capacity, D, device=k.device, dtype=k.dtype)
        self.v = torch.zeros(B, H, self.capacity, D, device=v.device, dtype=v.dtype)
        n_sink = min(self.sink, L)
        self.k[:, :, :n_sink] = k[:, :, :n_sink]
        self.v[:, :, :n_sink] = v[:, :, :n_sink]
        self.write_pos, self.filled = self.sink, 0
        if L > n_sink:
            self.append(k[:, :, n_sink:], v[:, :, n_sink:])
        else:
            self.length = n_sink

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.size(2)
        ring = self.capacity - self.sink
        src = 0
        while src < n:
            pos = self.write_pos
            take = min(n - src, ring - (pos - self.sink))
            self.k[:, :, pos:pos + take] = k[:, :, src:src + take]
            self.v[:, :, pos:pos + take] = v[:, :, src:src + take]
            src += take
            self.write_pos = self.sink + ((pos - self.sink + take) % ring)
        self.filled = min(ring, self.filled + n)
        self.length = self.sink + self.filled

    def load_linear(self, k: torch.Tensor, v: torch.Tensor, length: int) -> None:
        """Adopt a not-yet-wrapped cache whose first `length` slots are valid."""
        self.k, self.v = k, v
        self.filled = length - self.sink
        self.length = length
        self.write_pos = self.sink + self.filled % (self.capacity - self.sink)

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v) in chronological order: [sink] + [oldest .. newest]."""
        ring = self.capacity - self.sink
        if self.filled < ring and self.write_pos == self.sink + self.filled:
            # ring has not wrapped yet: valid entries are one contiguous prefix
            return self.k[:, :, :self.length], self.v[:, :, :self.length]
        start = (self.write_pos - self.filled - self.sink) % ring
        idx = (start + torch.arange(self.filled, device=self.k.device)) % ring + self.sink
        idx = torch.cat([torch.arange(self.sink, device=self.k.device, dtype=idx.dtype), idx])
        B, H, _, D = self.k.shape
        index = idx.view(1, 1, -1, 1).expand(B, H, -1, D)
        return torch.gather(self.k, 2, index), torch.gather(self.v, 2, index)


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    x = (2 ** 8) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)]).view(-1, 1, 1)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

        rel = torch.arange(max_seq_len)[None, :] - torch.arange(max_seq_len)[:, None]
        causal = torch.zeros(max_seq_len, max_seq_len).masked_fill(
            torch.ones(max_seq_len, max_seq_len).bool().triu(1), float("-inf"))
        self.register_buffer("causal_bias", causal[None, None], persistent=False)
        self.register_buffer("alibi_bias", (_alibi_slopes(num_heads) * rel)[None], persistent=False)
        self.prefix_len = 0
        self.fused_qkv = None        # (weight, bias) packed for one-GEMM inference

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

    def set_fused_qkv(self, enabled: bool) -> None:
        if not enabled:
            self.fused_qkv = None
            return
        with torch.no_grad():
            self.fused_qkv = (
                torch.cat([self.query.weight, self.key.weight, self.value.weight]).detach(),
                torch.cat([self.query.bias, self.key.bias, self.value.bias]).detach(),
            )

    def _qkv(self, x: torch.Tensor):
        if self.fused_qkv is not None:
            q, k, v = F.linear(x, *self.fused_qkv).chunk(3, dim=-1)
            return self._split(q), self._split(k), self._split(v)
        return self._split(self.query(x)), self._split(self.key(x)), self._split(self.value(x))

    def forward(self, x: torch.Tensor, cache: KVCache, use_cache: bool) -> torch.Tensor:
        B, L, _ = x.shape
        q, k_new, v_new = self._qkv(x)
        if use_cache:
            cache.append(k_new, v_new)
            q_start = cache.length - L
        else:
            cache.reset(k_new, v_new)
            q_start = 0
        k, v = cache.view()
        n = cache.length

        alibi = self.alibi_bias[:, :, q_start:q_start + L, :n]
        p = min(self.prefix_len, n)
        if p > 0:                                  # prompt columns: no distance penalty
            alibi = alibi.clone()
            alibi[..., :p] = 0.0
        bias = self.causal_bias[:, :, q_start:q_start + L, :n] + alibi

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, L, -1))

    # ------------------------------------------------------------ CUDA-graph path
    @torch.no_grad()
    def graph_alloc(self, like: KVCache) -> None:
        """Static buffers for graph replay (capacity-sized, B = 1)."""
        self.g_k = torch.zeros_like(like.k)
        self.g_v = torch.zeros_like(like.v)
        self.g_pos = torch.zeros(1, dtype=torch.long, device=like.k.device)
        bias = (self.causal_bias + self.alibi_bias).clone()
        bias[..., :min(self.prefix_len, bias.shape[-1])] = 0.0
        self.g_bias = bias.contiguous()

    @torch.no_grad()
    def graph_load(self, cache: KVCache) -> None:
        """Copy a primed (not wrapped) eager cache into the static buffers."""
        self.g_k.copy_(cache.k)
        self.g_v.copy_(cache.v)
        self.g_pos.fill_(cache.length)

    @torch.no_grad()
    def graph_export(self, cache: KVCache, length: int) -> None:
        """Hand the static-buffer state back to an eager ring-buffer cache."""
        cache.load_linear(self.g_k.clone(), self.g_v.clone(), length)

    def graph_forward(self, x: torch.Tensor) -> torch.Tensor:
        """One token, fixed shapes: graph-capturable."""
        q, k_new, v_new = self._qkv(x)
        self.g_k.index_copy_(2, self.g_pos, k_new)
        self.g_v.index_copy_(2, self.g_pos, v_new)
        bias = self.g_bias.index_select(2, self.g_pos)
        y = F.scaled_dot_product_attention(q, self.g_k, self.g_v, attn_mask=bias, dropout_p=0.0)
        self.g_pos.add_(1)
        return self.proj(y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))


class DecoderBlock(nn.Module):
    """Pre-LN block: causal self-attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, max_seq_len: int,
                 dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads, max_seq_len, dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cache: KVCache, use_cache: bool) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), cache, use_cache)
        return x + self.ff(self.ln3(x))

    def graph_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.graph_forward(self.ln1(x))
        return x + self.ff(self.ln3(x))
