"""LiveAnon: streamable speaker anonymization with an autoregressive codec language model.

Pipeline (see assets/architecture.png):

    reference audio --JHCodec--> prompt tokens ─┐
                                                 ├─> causal Transformer decoder --> codec tokens --JHCodec--> audio
    source audio --Zipformer--> content ──FiLM──┘

* Speaker: VALL-E-style prompt continuation. The reference's codec tokens are the first
  part of the AR stream (embedded with the same codebook embeddings as the generated
  tokens); the model continues that stream in the reference voice.
* Content: streaming Zipformer ASR bottleneck features, per-layer FiLM.
* Capped delay pattern: codebook k is delayed by min(k, delay_cap) frames.

Offline (`generate`) and streaming (`stream`) decoding run the same `ARSession`, so for the
same input they produce identical tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from liveanon.codec import JHCodec
from liveanon.config import LiveAnonConfig
from liveanon.content import (ZipformerContentEncoder, ZipformerStream, derive_zipformer_arch,
                              load_zipformer_state_dict)
from liveanon.layers import DecoderBlock, KVCache


def _film_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 2 * hidden_dim),
    )


def _film(x: torch.Tensor, mlp: nn.Module, cond: torch.Tensor) -> torch.Tensor:
    gamma, beta = mlp(cond).chunk(2, dim=-1)
    return x * (1.0 + gamma) + beta


class Decoder(nn.Module):
    """Causal Transformer decoder with per-layer content FiLM."""

    def __init__(self, cfg: LiveAnonConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, cfg.max_seq_len, cfg.dropout)
            for _ in range(cfg.num_decoder_layers)
        ])
        self.content_film = nn.ModuleList([
            _film_mlp(cfg.content_proj_dim, cfg.hidden_dim) for _ in range(cfg.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.max_seq_len = cfg.max_seq_len

    def set_prompt_len(self, n: int) -> None:
        for layer in self.layers:
            layer.self_attn.prefix_len = int(n)

    def new_caches(self, prompt_len: int) -> list[KVCache]:
        return [KVCache(self.max_seq_len, prompt_len) for _ in self.layers]

    def forward(self, x: torch.Tensor, content: torch.Tensor, caches: list[KVCache],
                use_cache: bool) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, caches[i], use_cache)
            x = _film(x, self.content_film[i], content)
        return self.norm(x)

    def graph_forward(self, x: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer.graph_forward(x)
            x = _film(x, self.content_film[i], content)
        return self.norm(x)


class DecoderGraph:
    """CUDA Graph of one full decoder step (B = 1), replayed once per generated frame.

    Works on fixed-capacity KV buffers, so it can only run until the cache is full
    (prompt + generated frames == max_seq_len); `export` then hands the state back to the
    eager ring-buffer caches. The capture is reused across sessions with the same prompt
    length.
    """

    def __init__(self, decoder: Decoder, caches: list[KVCache], content_dim: int):
        self.decoder = decoder
        self.prompt_len = caches[0].sink
        self.capacity = caches[0].capacity
        for layer, cache in zip(decoder.layers, caches):
            layer.self_attn.graph_alloc(cache)
        dev, dt = caches[0].k.device, caches[0].k.dtype
        self.x = torch.zeros(1, 1, decoder.norm.normalized_shape[0], device=dev, dtype=dt)
        self.content = torch.zeros(1, 1, content_dim, device=dev, dtype=dt)

        self.load(caches)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):                                    # warm-up before capture
                decoder.graph_forward(self.x, self.content)
        torch.cuda.current_stream().wait_stream(side)
        self.load(caches)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = decoder.graph_forward(self.x, self.content)
        self.load(caches)

    def load(self, caches: list[KVCache]) -> None:
        for layer, cache in zip(self.decoder.layers, caches):
            layer.self_attn.graph_load(cache)
        self.length = caches[0].length

    def can_step(self) -> bool:
        return self.length < self.capacity

    @torch.no_grad()
    def step(self, x: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        self.x.copy_(x)
        self.content.copy_(content)
        self.graph.replay()
        self.length += 1
        return self.out

    def export(self, caches: list[KVCache]) -> None:
        for layer, cache in zip(self.decoder.layers, caches):
            layer.self_attn.graph_export(cache, self.length)


class LiveAnon(nn.Module):
    def __init__(self, cfg: LiveAnonConfig, zipformer_arch: dict, codec: JHCodec | None = None):
        super().__init__()
        self.cfg = cfg
        K, V, H = cfg.n_codebooks, cfg.codebook_size, cfg.hidden_dim

        self.codec = codec
        self.content_encoder = ZipformerContentEncoder(
            zipformer_arch, cfg.zipformer_chunk_size, cfg.zipformer_left_context,
            cfg.zipformer_align_pad_frames)
        self.content_proj = nn.Linear(cfg.content_dim, cfg.content_proj_dim)

        # Token embedding: codebook vectors (initialised from JHCodec) -> per-codebook projection,
        # summed over codebooks. Shared by the reference prompt and the generated tokens.
        self.register_buffer("codebook_vectors", torch.zeros(K, V, 1024))
        self.input_projs = nn.ModuleList([nn.Linear(1024, H) for _ in range(K)])
        self.heads = nn.ModuleList([nn.Linear(H, V) for _ in range(K)])
        self.bos = nn.Parameter(torch.zeros(1, 1, H))
        self.null_content = nn.Parameter(torch.zeros(1, 1, H))
        self.null_content_film = nn.Parameter(torch.zeros(1, 1, cfg.content_proj_dim))
        self.decoder = Decoder(cfg)

        self.fast = False
        self._packed_heads = None      # (weight, bias): all codebook heads in one GEMM
        self._token_tables = None      # (K, V, H): input_projs[k](codebook_vectors[k])
        self._graph = None

    # ------------------------------------------------------------------ loading
    # Frozen third-party modules, loaded from their official releases.
    EXTERNAL = ("codec.", "content_encoder.")

    @classmethod
    def from_pretrained(cls, ckpt_path: str, device: str | torch.device = "cuda",
                        codec: JHCodec | None = None, zipformer_ckpt: str | None = None,
                        fast: bool = True, **config_overrides) -> "LiveAnon":
        """Load a LiveAnon checkpoint.

        JHCodec and the Zipformer encoder are downloaded from Hugging Face unless given
        (`codec`, `zipformer_ckpt`). `config_overrides` replace fields of the stored
        config (e.g. `delay_cap=...`). `fast` enables the inference optimisations
        (see `set_fast`).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = LiveAnonConfig.from_dict({**ckpt["config"], **config_overrides})
        zipformer_sd = load_zipformer_state_dict(zipformer_ckpt)
        model = cls(cfg, derive_zipformer_arch(zipformer_sd),
                    codec=codec or JHCodec(cfg.n_codebooks))
        model.content_encoder.load_state_dict(zipformer_sd, strict=True)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        missing = [k for k in missing if not k.startswith(cls.EXTERNAL)]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} "
                               f"unexpected={unexpected[:5]}")
        model = model.to(device).eval()
        model.set_fast(fast)
        return model

    @torch.no_grad()
    def set_fast(self, enabled: bool = True) -> None:
        """Inference optimisations for the AR loop:
          * Q/K/V projections packed into one GEMM;
          * all K codebook heads in one GEMM (greedy decoding);
          * token feedback as a lookup in precomputed input_projs[k](codebook_vectors[k])
            tables instead of K small GEMMs;
          * the whole decoder step replayed as a CUDA Graph (CUDA, batch size 1).
        """
        self.fast = bool(enabled)
        for layer in self.decoder.layers:
            layer.self_attn.set_fused_qkv(self.fast)
        self._graph = None
        if not self.fast:
            self._packed_heads = self._token_tables = None
            return
        self._packed_heads = (torch.cat([h.weight for h in self.heads]),
                              torch.cat([h.bias for h in self.heads]))
        self._token_tables = torch.stack([proj(self.codebook_vectors[k])
                                          for k, proj in enumerate(self.input_projs)])

    # ------------------------------------------------------------------ helpers
    @property
    def delays(self) -> list[int]:
        """Capped delay pattern: codebook k is delayed by min(k, delay_cap) frames."""
        cap = int(self.cfg.delay_cap)
        if not 0 <= cap < self.cfg.n_codebooks:
            raise ValueError(f"delay_cap must be in [0, {self.cfg.n_codebooks - 1}], got {cap}")
        return [min(k, cap) for k in range(self.cfg.n_codebooks)]

    @property
    def device(self) -> torch.device:
        return self.bos.device

    def _embed(self, k: int, codes: torch.Tensor) -> torch.Tensor:
        return self.input_projs[k](self.codebook_vectors[k][codes])

    def _delayed_prompt(self, codes: torch.Tensor) -> torch.Tensor:
        """Prompt codes (B, K, T) -> delayed AR input stream (B, T + max_delay, H), BOS first."""
        B, K, T = codes.shape
        T_out = T + max(self.delays)
        emb = torch.zeros(B, T_out, self.cfg.hidden_dim, device=codes.device)
        for k in range(K):
            start = self.delays[k] + 1
            end = min(start + T, T_out)
            emb[:, start:end] += self._embed(k, codes[:, k])[:, :end - start]
        emb[:, 0:1] = self.bos.expand(B, -1, -1)
        return emb

    @staticmethod
    def _align(feats: torch.Tensor, n: int) -> torch.Tensor:
        """Align (B, T_src, D) features to n codec frames (identity when T_src == n)."""
        T_src = feats.size(1)
        if T_src == n:
            return feats
        if T_src < n:
            idx = torch.linspace(0, T_src - 1, n, device=feats.device).long()
            return feats[:, idx]
        ratio = T_src / n
        return torch.stack([feats[:, max(0, int(t * ratio)):int((t + 1) * ratio)].mean(1)
                            for t in range(n)], dim=1)

    def encode_content(self, audio: torch.Tensor) -> torch.Tensor:
        return self.content_proj(self.content_encoder(audio, self.cfg.sample_rate))

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, reference_audio: torch.Tensor, source_audio: torch.Tensor,
                 temperature: float = 0.0, top_k: int | None = None,
                 top_p: float | None = None) -> torch.Tensor:
        """Offline: anonymize the whole `source_audio` into the voice of `reference_audio`.

        Args:
            reference_audio: (B, T_ref) 16 kHz pseudo-speaker prompt (~3 s).
            source_audio:    (B, T_src) 16 kHz speech to anonymize.
            temperature / top_k / top_p: sampling; temperature <= 0 is greedy.

        Returns:
            codec tokens (B, K, T_frames); decode with `decode_to_audio`.
        """
        session = ARSession(self, reference_audio, temperature, top_k, top_p)
        n_frames = max(1, -(-source_audio.shape[-1] // JHCodec.hop_length))
        session.push(self._align(self.encode_content(source_audio), n_frames))
        session.finish()
        return session.codes()

    @torch.no_grad()
    def stream(self, reference_audio: torch.Tensor, temperature: float = 0.0,
               top_k: int | None = None, top_p: float | None = None) -> "LiveAnonStream":
        """Streaming: returns a session that takes raw source samples as they arrive."""
        return LiveAnonStream(self, reference_audio, temperature, top_k, top_p)

    @torch.no_grad()
    def decode_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(codes)


def _sample(logits, temperature, top_k, top_p):
    if temperature is None or temperature <= 0 or top_k == 1:
        return logits.argmax(dim=-1)
    if top_k:
        kth = torch.topk(logits, min(int(top_k), logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        remove = F.softmax(sorted_logits, dim=-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.scatter(-1, idx, sorted_logits.masked_fill(remove, float("-inf")))
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


class ARSession:
    """Autoregressive decoding state: content frames in, codec-token frames out.

    `push` consumes content frames (one AR step each) and returns the codec frames whose
    K codebooks are all decoded; `finish` runs the delay tail (null content) and returns
    the rest. Offline generation is a single `push` of all frames followed by `finish`.
    """

    def __init__(self, model: LiveAnon, reference_audio: torch.Tensor,
                 temperature: float = 0.0, top_k: int | None = None,
                 top_p: float | None = None):
        m = self.m = model
        self.K, self.delays = m.cfg.n_codebooks, m.delays
        self.cap = max(self.delays)
        self.sampling = (temperature, top_k, top_p)
        self.greedy = temperature is None or temperature <= 0 or top_k == 1
        self.B = reference_audio.shape[0]
        self.device = reference_audio.device

        # speaker prompt: the reference codec tokens are the start of the AR stream
        self.prompt = m.codec.encode(reference_audio)[:, :self.K]           # (B, K, T_ref)
        self.T_ref = self.prompt.shape[2]
        m.decoder.set_prompt_len(self.T_ref)
        stream = m._delayed_prompt(self.prompt)
        prefix, self.prev = stream[:, :self.T_ref], stream[:, self.T_ref:self.T_ref + 1]
        # prefill the KV cache (the prompt region carries the reference's own content)
        prefix_content = m._align(m.encode_content(reference_audio), self.T_ref)
        self.caches = m.decoder.new_caches(self.T_ref)
        m.decoder(prefix, prefix_content, self.caches, use_cache=False)

        self.graph = None
        if m.fast and self.device.type == "cuda" and self.B == 1:
            g = m._graph
            if g is None or g.prompt_len != self.T_ref:
                g = m._graph = DecoderGraph(m.decoder, self.caches, m.cfg.content_proj_dim)
            else:
                g.load(self.caches)
            self.graph = g

        self.generated: list[list[torch.Tensor]] = [[] for _ in range(self.K)]
        self.t = 0
        self.emitted = 0
        self.finished = False

    def _hidden(self, x: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.graph is not None:
            if self.graph.can_step():
                return self.graph.step(x, content)[:, -1]
            self.graph.export(self.caches)        # cache full: continue on the ring buffer
            self.graph = None
        return self.m.decoder(x, content, self.caches, use_cache=True)[:, -1]

    def _step(self, content: torch.Tensor, token_add: torch.Tensor | None = None,
              n_frames: int | None = None) -> None:
        m, t, d = self.m, self.t, self.delays
        x = self.prev if token_add is None else self.prev + token_add
        h = self._hidden(x, content)
        valid = [k for k in range(self.K)
                 if t - d[k] >= 0 and (n_frames is None or t - d[k] < n_frames)]
        if m.fast and self.greedy:
            codes = F.linear(h, *m._packed_heads).view(self.B, self.K, -1).argmax(dim=-1)
            for k in valid:
                self.generated[k].append(codes[:, k])
        else:
            for k in valid:
                self.generated[k].append(_sample(m.heads[k](h), *self.sampling))

        # next input: the token of codebook k at frame t - delay(k)
        # (taken from the prompt while that frame still lies inside it)
        if m.fast and t >= self.cap and n_frames is None:
            idx = torch.stack([self.generated[k][t - d[k]] for k in range(self.K)], dim=1)
            ks = torch.arange(self.K, device=self.device).unsqueeze(0)
            self.prev = m._token_tables[ks, idx].sum(dim=1, keepdim=True)
        else:
            prev = torch.zeros(self.B, 1, m.cfg.hidden_dim, device=self.device)
            for k in range(self.K):
                src = t - d[k]
                if 0 <= src < len(self.generated[k]):
                    prev += m._embed(k, self.generated[k][src]).unsqueeze(1)
                elif src < 0 and self.T_ref + src >= 0:
                    prev += m._embed(k, self.prompt[:, k, self.T_ref + src]).unsqueeze(1)
            self.prev = prev
        self.t += 1

    def _completed(self) -> torch.Tensor:
        done = min(len(g) for g in self.generated)
        out = torch.stack([torch.stack(g[self.emitted:done], dim=1) if done > self.emitted
                           else torch.zeros(self.B, 0, dtype=torch.long, device=self.device)
                           for g in self.generated], dim=1)
        self.emitted = done
        return out

    @torch.no_grad()
    def push(self, content: torch.Tensor) -> torch.Tensor:
        """content: (B, n, content_proj_dim) -> newly completed codes (B, K, m)."""
        if self.finished:
            raise RuntimeError("session already finished")
        for i in range(content.shape[1]):
            self._step(content[:, i:i + 1])
        return self._completed()

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        """Decode the delay tail; returns the remaining codes (B, K, m)."""
        if self.finished:
            raise RuntimeError("session already finished")
        self.finished = True
        m, n_frames = self.m, self.t
        null_content = m.null_content_film.expand(self.B, 1, -1)
        null_add = m.null_content.expand(self.B, 1, -1)
        for _ in range(self.cap):
            self._step(null_content, null_add, n_frames=n_frames)
        return self._completed()

    def codes(self) -> torch.Tensor:
        """All codes generated so far (B, K, T)."""
        n = min(len(g) for g in self.generated)
        return torch.stack([torch.stack(g[:n], dim=1) for g in self.generated], dim=1)


class LiveAnonStream:
    """Real streaming: raw source samples in, anonymized audio out (batch size 1).

        stream = model.stream(reference)
        for chunk in microphone:            # (1, n) 16 kHz samples, any n
            out = stream.push(chunk)        # anonymized samples available so far
        out = stream.finish()

    Content features come from the stateful streaming Zipformer, every finalized frame
    advances the AR decoder by one step, and completed codec frames are decoded
    incrementally by JHCodec (with its KV cache).
    """

    def __init__(self, model: LiveAnon, reference_audio: torch.Tensor, temperature=0.0,
                 top_k=None, top_p=None):
        if reference_audio.shape[0] != 1:
            raise ValueError("streaming supports batch size 1")
        self.m = model
        self.session = ARSession(model, reference_audio, temperature, top_k, top_p)
        self.content = ZipformerStream(model.content_encoder, model.cfg.sample_rate,
                                       use_cuda_graph=model.fast)
        self.codec_cache = None

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] == 0:
            return torch.zeros(1, 0, device=codes.device)
        audio, self.codec_cache = self.m.codec.decode(codes, cache=self.codec_cache,
                                                      return_cache=True)
        return audio

    @torch.no_grad()
    def push(self, samples: torch.Tensor) -> torch.Tensor:
        feats = self.content.push(samples)
        codes = self.session.push(self.m.content_proj(feats))
        return self._decode(codes)

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        feats = self.content.finish()
        codes = torch.cat([self.session.push(self.m.content_proj(feats)),
                           self.session.finish()], dim=-1)
        return self._decode(codes)

    def codes(self) -> torch.Tensor:
        return self.session.codes()
