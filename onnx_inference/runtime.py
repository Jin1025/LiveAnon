"""LiveAnon on ONNX Runtime: the full streaming pipeline without PyTorch.

Dependencies: numpy, onnxruntime(-gpu), kaldi-native-fbank. Graphs come from
`export_onnx.py` (see `graphs.py`):

    samples --fbank (kaldi-native-fbank)--> zipformer_chunk --> content ─┐
    reference --codec_encode--> prompt codes --decoder_prefill--> KV ────┤
                                                    decoder_step (per frame)
                                                                         └--> codec_decode --> samples

    model = OnnxLiveAnon("exported/liveanon_chunk320ms")        # provider="tensorrt" / "cpu"
    stream = model.stream(reference)                             # 16 kHz float32 numpy
    for block in microphone:
        out = stream.push(block)                                 # anonymized samples so far
    out = stream.finish()

Recurrent states (KV caches, Zipformer states) stay on the device: they are bound as
OrtValues and ping-ponged between two buffers. Only the small per-step tensors (fed-back
tokens, one content frame, the K codes of a frame, 320 output samples) cross to the host.
Greedy decoding, batch size 1.
"""
from __future__ import annotations

import json
import os

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

def _preload_gpu_libs(tensorrt: bool) -> None:
    """Load the CUDA / cuDNN (and TensorRT) shared libraries from their pip packages
    (`nvidia-*-cu12`, `tensorrt-cu12`) so the GPU providers find them without PyTorch."""
    import ctypes
    import glob
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        root = list(spec.submodule_search_locations)[0]
        for rel in (("cublas", "libcublasLt.so.12"), ("cublas", "libcublas.so.12"),
                    ("cuda_nvrtc", "libnvrtc.so.12"), ("curand", "libcurand.so.10"),
                    ("cufft", "libcufft.so.11"), ("cuda_runtime", "libcudart.so.12"),
                    ("cudnn", "libcudnn.so.9")):
            path = os.path.join(root, rel[0], "lib", rel[1])
            if os.path.isfile(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    if tensorrt:
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec is not None and spec.submodule_search_locations:
            for lib in sorted(glob.glob(os.path.join(
                    list(spec.submodule_search_locations)[0], "libnv*.so.10"))):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


GRAPHS = ("zipformer_chunk", "decoder_prefill", "decoder_step", "codec_encode", "codec_decode")


class _Graph:
    """An InferenceSession plus device-resident state buffers (ping-pong)."""

    def __init__(self, path: str, providers: list, device: str, device_id: int):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=providers)
        want = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if want not in self.sess.get_providers():
            raise RuntimeError(f"{os.path.basename(path)}: {want} could not be loaded "
                               f"(got {self.sess.get_providers()})")
        self.device, self.device_id = device, device_id
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def zeros(self, shape, dtype=np.float32) -> ort.OrtValue:
        return ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype), self.device,
                                                self.device_id)

    def run(self, host: dict, device: dict, device_out: dict) -> dict:
        """host: name -> numpy input; device: name -> OrtValue input;
        device_out: name -> OrtValue to write. Other outputs come back as numpy."""
        b = self.sess.io_binding()
        for k, v in host.items():
            b.bind_cpu_input(k, np.ascontiguousarray(v))
        for k, v in device.items():
            b.bind_ortvalue_input(k, v)
        for k in self.out_names:
            if k in device_out:
                b.bind_ortvalue_output(k, device_out[k])
            else:
                b.bind_output(k, "cpu")
        self.sess.run_with_iobinding(b)
        return {k: v.numpy() for k, v in zip(self.out_names, b.get_outputs())
                if k not in device_out}


STEP_GRAPHS = ("zipformer_chunk", "decoder_step", "codec_decode")   # run per chunk / frame


class OnnxLiveAnon:
    """provider: "cuda" (CUDA EP), "tensorrt" (TensorRT EP for the per-step graphs, built
    and cached under <onnx_dir>/trt_cache* on first use) or "cpu"."""

    def __init__(self, onnx_dir: str, provider: str = "cuda", device_id: int = 0,
                 trt_fp16: bool = False):
        self.meta = json.load(open(os.path.join(onnx_dir, "meta.json")))
        # conv1d as (N, C, 1, D): cuDNN is ~8x faster on the Zipformer depthwise convs
        cuda = ("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0,
                                          "cudnn_conv_algo_search": "HEURISTIC",
                                          "cudnn_conv1d_pad_to_nc1d": "1"})
        trt = ("TensorrtExecutionProvider", {
            "device_id": device_id, "trt_fp16_enable": trt_fp16,
            "trt_engine_cache_enable": True, "trt_timing_cache_enable": True,
            "trt_engine_cache_path": os.path.join(
                onnx_dir, "trt_cache_fp16" if trt_fp16 else "trt_cache")})
        device = "cpu" if provider == "cpu" else "cuda"
        if device == "cuda":
            # TensorRT (and cuBLAS) use TF32 for float32 by default; that changes the greedy
            # tokens. Must be set before the libraries create their handles.
            os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
            _preload_gpu_libs(provider == "tensorrt")

        def providers(name):
            if provider == "cpu":
                return ["CPUExecutionProvider"]
            if provider == "tensorrt" and name in STEP_GRAPHS:
                return [trt, cuda]
            return [cuda]

        self.g = {n: _Graph(os.path.join(onnx_dir, f"{n}.onnx"), providers(n), device,
                            device_id) for n in GRAPHS}

    def prompt(self, reference: np.ndarray) -> dict:
        """Speaker prompt of a 16 kHz reference: its codec tokens and content features.
        Compute once and reuse for every utterance (`np.savez` / `load_prompt`)."""
        reference = np.asarray(reference, np.float32)
        hop = self.meta["hop"]
        n = -(-len(reference) // hop)
        frames = np.zeros(n * hop, np.float32)
        frames[:len(reference)] = reference
        codes = self.g["codec_encode"].run({"frames": frames.reshape(1, n, hop)}, {}, {})[
            "codes"][0].T.astype(np.int64)                         # (K, T_ref)
        cs = _ContentStream(self)
        content = _cat(cs.push(reference), cs.finish())[:codes.shape[1]]
        return {"codes": codes, "content": content}

    def stream(self, prompt) -> "OnnxStream":
        """prompt: a dict from `prompt()` / `load_prompt()`, or a 16 kHz reference waveform."""
        if not isinstance(prompt, dict):
            prompt = self.prompt(prompt)
        return OnnxStream(self, prompt)

    def generate(self, prompt, source: np.ndarray, block: int | None = None) -> np.ndarray:
        """Whole utterance (fed in `block`-sample pieces, default: at once)."""
        s = self.stream(prompt)
        block = block or len(source)
        out = [s.push(source[i:i + block]) for i in range(0, len(source), block)]
        out.append(s.finish())
        return np.concatenate(out)


# ============================================================================ content
class _ContentStream:
    """`liveanon.content.ZipformerStream` on ONNX: samples in, projected content out."""

    def __init__(self, m: OnnxLiveAnon):
        meta = m.meta["zipformer"]
        self.g = m.g["zipformer_chunk"]
        self.chunk, self.hop = meta["chunk"], meta["hop"]
        self.need = 2 * self.chunk + 13
        self.skip = self.skip_total = meta["front_skip"]   # 120 ms lookahead alignment
        self.state_names = meta["state_names"]          # embed_state, cache_padding, state_*
        self.states = []
        for _ in range(2):
            st = {n: self.g.zeros(shape) for n, shape in zip(self.state_names, meta["state_shapes"])}
            # every left-context frame starts out as padding
            st["cache_padding"] = ort.OrtValue.ortvalue_from_numpy(
                np.ones(meta["state_shapes"][1], np.bool_), self.g.device, self.g.device_id)
            self.states.append(st)
        self.cur = 0
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400.0
        self.fbank = knf.OnlineFbank(opts)
        self.frames = np.zeros((0, 80), np.float32)
        self.offset = 0                                  # first unconsumed fbank frame
        self.n_samples = 0
        self.emitted = 0

    def _accept(self, samples: np.ndarray) -> None:
        if self.skip > 0:
            drop = min(self.skip, len(samples))
            samples, self.skip = samples[drop:], self.skip - drop
        if len(samples):
            self.fbank.accept_waveform(16000, samples.astype(np.float32).tolist())

    def _collect(self) -> None:
        n = self.fbank.num_frames_ready
        if n > len(self.frames):
            new = np.stack([self.fbank.get_frame(i) for i in range(len(self.frames), n)])
            self.frames = np.concatenate([self.frames, new.astype(np.float32)])

    def _run(self, seg: np.ndarray, conv_keep: int, valid: int) -> np.ndarray:
        if len(seg) < self.need:
            seg = np.concatenate([seg, np.zeros((self.need - len(seg), 80), np.float32)])
        src, dst = self.states[self.cur], self.states[1 - self.cur]
        out = self.g.run({"fbank": seg[None, :self.need],
                          "conv_keep": np.array([conv_keep], np.int64),
                          "valid": np.array([valid], np.int64)},
                         {k: src[k] for k in self.state_names},
                         {f"new_{k}": dst[k] for k in self.state_names})
        self.cur = 1 - self.cur
        self.emitted += valid
        return out["content"][0, :valid]

    def push(self, samples: np.ndarray) -> np.ndarray:
        self.n_samples += len(samples)
        self._accept(samples)
        self._collect()
        out = []
        while len(self.frames) >= self.offset + self.need:
            out.append(self._run(self.frames[self.offset:self.offset + self.need],
                                 self.chunk + 3, self.chunk))
            self.offset += 2 * self.chunk
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def finish(self) -> np.ndarray:
        """Flush with the same zero padding as the offline path (`align_audio`)."""
        n_target = -(-self.n_samples // self.hop)
        trimmed = max(0, self.n_samples - self.skip_total)
        pad = (2 * n_target + 7) * (self.hop // 2) - trimmed
        if pad > 0:
            self._accept(np.zeros(pad, np.float32))
        self.fbank.input_finished()
        self._collect()
        out = []
        while self.emitted < n_target:
            valid = min(self.chunk, n_target - self.emitted)
            seg = self.frames[self.offset:]
            keep = min((len(seg) - 7) // 2, valid + 3)
            out.append(self._run(seg[:self.need], keep, valid))
            self.offset += 2 * valid
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


# ============================================================================ AR + codec
class OnnxStream:
    """`liveanon.model.LiveAnonStream` on ONNX (greedy, batch size 1)."""

    def __init__(self, m: OnnxLiveAnon, prompt: dict):
        self.m, meta = m, m.meta
        self.K, self.delays = meta["n_codebooks"], meta["delays"]
        self.cap = max(self.delays)
        self.consts = {k: np.asarray(v, np.float32) for k, v in meta["constants"].items()}
        self.prompt = np.asarray(prompt["codes"], np.int64)                # (K, T_ref)
        self.T_ref = T = self.prompt.shape[1]
        prompt_content = np.asarray(prompt["content"], np.float32)

        # delayed prompt stream: row r carries token r - delay(k) - 1 of codebook k
        rows = np.arange(T)[:, None] - np.asarray(self.delays)[None] - 1
        valid = (rows >= 0) & (rows < T)
        toks = self.prompt[np.arange(self.K)[None], rows.clip(0, T - 1)]
        dg = m.g["decoder_prefill"]
        cache_shape = meta["decoder_cache_shape"]
        self.kv = [{"k_cache": dg.zeros(cache_shape), "v_cache": dg.zeros(cache_shape)}
                   for _ in range(2)]
        dg.run({"tokens": toks[None], "mask": valid[None].astype(np.float32),
                "content": prompt_content[None]}, {}, self.kv[0])
        self.cur = 0
        self.pos = T

        # codec decoder state
        cg = m.g["codec_decode"]
        cshape = meta["codec_cache_shape"]
        self.ckv = [{"k_cache": cg.zeros(cshape), "v_cache": cg.zeros(cshape)} for _ in range(2)]
        self.ccur = 0
        self.cpos = 0

        self.content = _ContentStream(m)
        self.generated: list[list[int]] = [[] for _ in range(self.K)]
        self.t = 0
        self.emitted = 0

    def _feedback(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Tokens fed into step t: codebook k at frame t - 1 - delay(k), taken from the
        prompt while that frame lies inside it (at t = 0 this is the last row of the delayed
        prompt stream)."""
        toks, mask = np.zeros(self.K, np.int64), np.zeros(self.K, np.float32)
        for k in range(self.K):
            s = t - 1 - self.delays[k]
            if 0 <= s < len(self.generated[k]):
                toks[k], mask[k] = self.generated[k][s], 1.0
            elif s < 0 and self.T_ref + s >= 0:
                toks[k], mask[k] = self.prompt[k, self.T_ref + s], 1.0
        return toks, mask

    def _step(self, content: np.ndarray, n_frames: int | None = None) -> None:
        t = self.t
        toks, mask = self._feedback(t)
        x_add = self.consts["null_content"] if n_frames is not None else \
            np.zeros_like(self.consts["null_content"])
        src, dst = self.kv[self.cur], self.kv[1 - self.cur]
        out = self.m.g["decoder_step"].run(
            {"tokens": toks[None], "mask": mask[None], "x_add": x_add.reshape(1, 1, -1),
             "content": content.reshape(1, 1, -1).astype(np.float32),
             "pos": np.array([self.pos], np.int64), "sink": np.array([self.T_ref], np.int64)},
            {"k_cache_in": src["k_cache"], "v_cache_in": src["v_cache"]}, dst)
        self.cur = 1 - self.cur
        self.pos += 1
        codes = out["codes"][0]
        for k in range(self.K):
            f = t - self.delays[k]
            if f >= 0 and (n_frames is None or f < n_frames):
                self.generated[k].append(int(codes[k]))
        self.t += 1

    def _decode_ready(self) -> np.ndarray:
        done = min(len(g) for g in self.generated)
        out = []
        g = self.m.g["codec_decode"]
        for f in range(self.emitted, done):
            codes = np.array([[[self.generated[k][f] for k in range(self.K)]]], np.int64)
            src, dst = self.ckv[self.ccur], self.ckv[1 - self.ccur]
            out.append(g.run({"codes": codes, "pos": np.array([self.cpos], np.int64)},
                             {"k_cache_in": src["k_cache"], "v_cache_in": src["v_cache"]},
                             dst)["audio"][0])
            self.ccur = 1 - self.ccur
            self.cpos += 1
        self.emitted = done
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        for frame in self.content.push(np.asarray(samples, np.float32)):
            self._step(frame)
        return self._decode_ready()

    def finish(self) -> np.ndarray:
        for frame in self.content.finish():
            self._step(frame)
        n_frames = self.t
        for _ in range(self.cap):
            self._step(self.consts["null_content_film"], n_frames)
        return self._decode_ready()

    def codes(self) -> np.ndarray:
        n = min(len(g) for g in self.generated)
        return np.array([g[:n] for g in self.generated], np.int64)


def load_prompt(path: str) -> dict:
    with np.load(path) as z:
        return {"codes": z["codes"], "content": z["content"]}


def _cat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parts = [x for x in (a, b) if x.size]
    return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
