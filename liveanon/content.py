"""Streaming Zipformer ASR encoder as the content (bottleneck feature) extractor.

The encoder is the icefall streaming Zipformer trained on LibriSpeech
(https://github.com/k2-fsa/icefall, Apache-2.0); its code is vendored under
`liveanon/zipformer/` and its weights are downloaded from the official release
(https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17).
We tap the features just before the final 50 -> 25 Hz downsampling, which gives 512-dim
features at 50 fps -- exactly the codec frame rate.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torchaudio

from liveanon.zipformer import Conv2dSubsampling, Zipformer2

ZIPFORMER_HF_REPO = "Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17"
ZIPFORMER_CKPT_NAME = "exp/pretrained.pt"


def load_zipformer_state_dict(ckpt_path: str | None = None) -> dict:
    """Encoder weights (`encoder_embed.*`, `encoder.*`) of the pretrained icefall model."""
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(ZIPFORMER_HF_REPO, ZIPFORMER_CKPT_NAME)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("model", obj)
    return {k: v for k, v in sd.items() if k.startswith(("encoder_embed.", "encoder."))}


def derive_zipformer_arch(sd: dict) -> dict:
    """Recover the Zipformer architecture from the tensor shapes of its state_dict
    (keys `encoder_embed.*` / `encoder.*`)."""
    def get(stack: int, name: str):
        for k in (f"encoder.encoders.{stack}.layers.0.{name}",
                  f"encoder.encoders.{stack}.encoder.layers.0.{name}"):
            if k in sd:
                return sd[k]
        return None

    n_stacks = 1 + max(int(m.group(1)) for m in
                       (re.match(r"encoder\.encoders\.(\d+)\.", k) for k in sd) if m)
    layers, dims, ffs, heads, kernels = [], [], [], [], []
    for s in range(n_stacks):
        layers.append(1 + max(
            int(m.group(1)) for m in
            (re.match(rf"encoder\.encoders\.{s}\.(?:encoder\.)?layers\.(\d+)\.", k) for k in sd)
            if m))
        dims.append(get(s, "self_attn_weights.in_proj.weight").shape[1])
        # three FeedforwardModules per layer sized (3/4 d, d, 5/4 d); feed_forward2 is d
        ffs.append(get(s, "feed_forward2.in_proj.weight").shape[0])
        heads.append(get(s, "self_attn_weights.linear_pos.weight").shape[0] // 4)
        cm = get(s, "conv_module1.depthwise_conv.chunkwise_conv.weight")
        kernels.append(cm.shape[-1] if cm is not None else 31)
    return {
        "num_encoder_layers": tuple(layers),
        "encoder_dim": tuple(dims),
        "feedforward_dim": tuple(ffs),
        "num_heads": tuple(heads),
        "cnn_module_kernel": tuple(kernels),
        "pos_dim": int(get(0, "self_attn_weights.linear_pos.weight").shape[1]),
        "query_head_dim": (get(0, "self_attn_weights.in_proj.weight").shape[0] // heads[0] - 4) // 2,
        "value_head_dim": get(0, "self_attn1.in_proj.weight").shape[0] // heads[0],
    }


class ZipformerContentEncoder(nn.Module):
    """(B, T_samples) 16 kHz waveform -> (B, T_frames, 512) content features at 50 fps."""

    def __init__(self, arch: dict, chunk_size: int = 16, left_context: int = 128,
                 align_pad_frames: int = -6):
        super().__init__()
        self.align_pad_frames = int(align_pad_frames)
        self.encoder_embed = Conv2dSubsampling(
            in_channels=80, out_channels=arch["encoder_dim"][0], layer1_channels=8,
            layer2_channels=32, layer3_channels=128, dropout=0.0,
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=(1, 2, 4, 8, 4, 2)[: len(arch["encoder_dim"])],
            num_encoder_layers=arch["num_encoder_layers"],
            encoder_dim=arch["encoder_dim"],
            encoder_unmasked_dim=tuple(min(192, d) for d in arch["encoder_dim"]),
            query_head_dim=arch["query_head_dim"],
            pos_head_dim=4,
            value_head_dim=arch["value_head_dim"],
            num_heads=arch["num_heads"],
            feedforward_dim=arch["feedforward_dim"],
            cnn_module_kernel=arch["cnn_module_kernel"],
            pos_dim=arch["pos_dim"],
            dropout=0.0,
            causal=True,
            chunk_size=(chunk_size,),
            left_context_frames=(left_context,),
        )
        self.out_dim = max(arch["encoder_dim"])
        # tap: the input of the final 50 -> 25 Hz downsampling
        self._tap = None
        self.encoder.downsample_output.register_forward_pre_hook(
            lambda _m, inp: setattr(self, "_tap", inp[0]))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _fbank(wav: torch.Tensor, sr: int) -> torch.Tensor:
        """80-bin Kaldi fbank at 100 Hz, matching icefall's kaldifeat options."""
        return torch.stack([
            torchaudio.compliance.kaldi.fbank(
                w.unsqueeze(0), num_mel_bins=80, sample_frequency=float(sr),
                dither=0.0, snip_edges=False, high_freq=-400.0)
            for w in wav
        ], dim=0)

    def align_audio(self, audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        """Align the waveform to the codec grid; returns (audio, n_frames).

        The causal encoder's long left context pulls each frame's centre of mass behind its
        index; trimming the front by 6 frames re-aligns it (= 120 ms lookahead). The tail is
        padded so that exactly ceil(len / hop) frames come out.
        """
        hop = sample_rate // 50
        n_target = -(-int(audio.shape[-1]) // hop)          # ceil(len / hop), the codec grid
        pad = self.align_pad_frames
        if pad > 0:
            audio = nn.functional.pad(audio, (pad * hop, 0))
        elif pad < 0:
            audio = audio[..., -pad * hop:]
        need = (2 * n_target + 7) * (sample_rate // 100) - audio.shape[-1]
        if need > 0:
            audio = nn.functional.pad(audio, (0, int(need)))
        return audio, n_target

    def _tapped(self) -> torch.Tensor:
        feats, self._tap = self._tap.permute(1, 0, 2), None  # (T, N, C) -> (N, T, C)
        return feats

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Whole-utterance (offline) encoding. The chunked attention mask makes it
        identical to running the encoder chunk by chunk (`ZipformerStream`)."""
        audio, n_target = self.align_audio(audio, sample_rate)
        x = self._fbank(audio, sample_rate).to(next(self.parameters()).device)
        x_lens = torch.full((x.size(0),), x.size(1), dtype=torch.int64, device=x.device)
        x, x_lens = self.encoder_embed(x, x_lens)
        x = x.permute(1, 0, 2)                                # (N, T, C) -> (T, N, C)
        mask = torch.arange(x.size(0), device=x.device)[None, :] >= x_lens[:, None]
        self.encoder(x, x_lens, mask)
        feats = self._tapped()
        if feats.shape[1] < n_target:
            raise RuntimeError(f"zipformer emitted {feats.shape[1]} frames, need {n_target}")
        return feats[:, :n_target]


class _EncoderChunkGraph:
    """CUDA Graph of one full-chunk `Zipformer2.streaming_forward` call (B = 1).

    The streaming encoder runs hundreds of small kernels per chunk, so on a GPU it is
    launch-bound; replaying a captured graph removes that overhead with identical results.
    States, padding mask and the chunk are copied in and out of static buffers.
    """

    def __init__(self, enc: "ZipformerContentEncoder", chunk: int, left_context: int,
                 embed_dim: int, device: torch.device):
        e = enc.encoder
        self.x = torch.zeros(chunk, 1, embed_dim, device=device)
        self.lens = torch.tensor([chunk], dtype=torch.int64, device=device)
        self.states = e.get_init_states(batch_size=1, device=device)
        self.mask = torch.zeros(1, left_context + chunk, dtype=torch.bool, device=device)

        def fn():
            _, _, new_states = e.streaming_forward(self.x, self.lens, self.states, self.mask)
            return enc._tap, new_states

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.tap, self.new_states = fn()
        enc._tap = None

    @torch.no_grad()
    def run(self, embedded: torch.Tensor, states: list, mask: torch.Tensor):
        self.x.copy_(embedded.permute(1, 0, 2))
        self.mask.copy_(mask)
        for dst, src in zip(self.states, states):
            dst.copy_(src)
        self.graph.replay()
        return self.tap.permute(1, 0, 2).clone(), [t.clone() for t in self.new_states]


class ZipformerStream:
    """Stateful streaming wrapper: waveform samples in, finalized 50 fps features out.

    No future samples are read. Fbank frames whose 25 ms analysis window reaches past the
    received audio are held back; the subsampler and the Zipformer keep their streaming
    states. A chunk of `chunk_size` feature frames is emitted as soon as its input
    (2 * chunk_size + 13 fbank frames, incl. the ConvNeXt lookahead) has arrived.
    `finish()` flushes the tail with exactly the padding the offline path uses, so the
    concatenated output equals `ZipformerContentEncoder.forward` on the full waveform.
    Batch size 1.
    """

    def __init__(self, encoder: ZipformerContentEncoder, sample_rate: int = 16000,
                 use_cuda_graph: bool = True):
        self.enc = encoder
        self.use_cuda_graph = use_cuda_graph
        self.sr = sample_rate
        self.hop = sample_rate // 50
        self.chunk = int(encoder.encoder.chunk_size[0])
        self.left_context = int(encoder.encoder.left_context_frames[0])
        self.device = next(encoder.parameters()).device
        self.embed_state = encoder.encoder_embed.get_init_states(1, self.device)
        self.states = encoder.encoder.get_init_states(batch_size=1, device=self.device)
        self.cache_padding = torch.ones(1, self.left_context, dtype=torch.bool,
                                        device=self.device)
        self.audio = torch.zeros(1, 0, device=self.device)
        self.fbank_offset = 0
        self.emitted = 0
        pad = encoder.align_pad_frames
        self.front_zeros = max(0, pad) * self.hop
        self.front_skip = max(0, -pad) * self.hop

    def _graph(self, embed_dim: int) -> _EncoderChunkGraph | None:
        if not (self.use_cuda_graph and self.device.type == "cuda"):
            return None
        g = getattr(self.enc, "_chunk_graph", None)
        if g is None:                       # captured once, shared by all streams
            g = self.enc._chunk_graph = _EncoderChunkGraph(
                self.enc, self.chunk, self.left_context, embed_dim, self.device)
        return g

    def _run_encoder(self, embedded: torch.Tensor, valid: int) -> torch.Tensor:
        if valid < self.chunk:
            embedded = nn.functional.pad(embedded, (0, 0, 0, self.chunk - valid))
        current = torch.arange(self.chunk, device=self.device)[None, :] >= valid
        mask = torch.cat([self.cache_padding, current], dim=1)
        graph = self._graph(embedded.shape[-1]) if valid == self.chunk else None
        if graph is not None:
            feats, self.states = graph.run(embedded, self.states, mask)
        else:
            _, _, self.states = self.enc.encoder.streaming_forward(
                embedded.permute(1, 0, 2),
                torch.tensor([valid], dtype=torch.int64, device=self.device),
                self.states, mask)
            feats = self.enc._tapped()
        self.cache_padding = mask[:, -self.left_context:]
        self.emitted += valid
        return feats[:, :valid]

    @torch.no_grad()
    def push(self, samples: torch.Tensor) -> torch.Tensor:
        """Append (1, n) samples; return newly finalized features (1, m, 512), m >= 0."""
        self.audio = torch.cat([self.audio, samples.to(self.device)], dim=-1)
        prefix = self.audio
        if self.front_zeros:
            prefix = nn.functional.pad(prefix, (self.front_zeros, 0))
        elif self.front_skip:
            prefix = prefix[..., min(self.front_skip, prefix.shape[-1]):]
        # a snip_edges=False 25 ms frame i needs samples up to i * 10 ms + 17.5 ms
        shift, right = self.sr // 100, 7 * self.sr // 400
        stable = max(0, (int(prefix.shape[-1]) - right - 1) // shift + 1)
        need = 2 * self.chunk + 13      # fbank frames -> chunk frames + ConvNeXt lookahead
        out = []
        if stable >= self.fbank_offset + need:
            fbanks = self.enc._fbank(prefix, self.sr).to(self.device)
            while stable >= self.fbank_offset + need:
                seg = fbanks[:, self.fbank_offset:self.fbank_offset + need]
                emb, self.embed_state = self.enc.encoder_embed.streaming_forward_exact(
                    seg, self.embed_state, self.chunk)
                out.append(self._run_encoder(emb, self.chunk))
                self.fbank_offset += 2 * self.chunk
        if not out:
            return torch.zeros(1, 0, self.enc.out_dim, device=self.device)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        """Flush the remaining frames (end of stream)."""
        audio, n_target = self.enc.align_audio(self.audio, self.sr)
        fbanks = self.enc._fbank(audio, self.sr).to(self.device)
        out = []
        while self.emitted < n_target:
            valid = min(self.chunk, n_target - self.emitted)
            emb, self.embed_state = self.enc.encoder_embed.streaming_forward_exact(
                fbanks[:, self.fbank_offset:], self.embed_state, valid)
            out.append(self._run_encoder(emb, valid))
            self.fbank_offset += 2 * valid
        if not out:
            return torch.zeros(1, 0, self.enc.out_dim, device=self.device)
        return torch.cat(out, dim=1)
