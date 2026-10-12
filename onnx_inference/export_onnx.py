"""Export a LiveAnon checkpoint (with its Zipformer encoder and JHCodec) to ONNX.

Writes five graphs (see `graphs.py`) and `meta.json` to `--out`, then checks every
graph against PyTorch with ONNX Runtime. `runtime.py` runs the
exported pipeline without PyTorch.

    python onnx_inference/export_onnx.py --ckpt checkpoints/liveanon_chunk320ms.pt \
        --out onnx_inference/exported/liveanon_chunk320ms
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liveanon import LiveAnon  # noqa: E402
from graphs import CodecDecode, CodecEncode, DecoderPrefill, DecoderStep, ZipformerChunk  # noqa: E402,E501

OPSET = 17


def _export(module, args, path, inputs, outputs, dynamic=None):
    torch.onnx.export(module.eval(), args, path, input_names=inputs, output_names=outputs,
                      dynamic_axes=dynamic or {}, opset_version=OPSET,
                      do_constant_folding=True, dynamo=False)


def export(model: LiveAnon, out: str) -> dict:
    cfg, dev = model.cfg, model.device
    K, H, C = cfg.n_codebooks, cfg.hidden_dim, cfg.content_proj_dim
    os.makedirs(out, exist_ok=True)
    p = lambda name: os.path.join(out, f"{name}.onnx")  # noqa: E731
    T = 150

    # ---- AR decoder
    attn = model.decoder.layers[0].self_attn
    dec_cache = [cfg.num_decoder_layers, 1, attn.num_heads, cfg.max_seq_len, attn.head_dim]
    tok = torch.zeros(1, T, K, dtype=torch.long, device=dev)
    _export(DecoderPrefill(model).to(dev),
            (tok, torch.ones(1, T, K, device=dev), torch.randn(1, T, C, device=dev)),
            p("decoder_prefill"), ["tokens", "mask", "content"], ["k_cache", "v_cache"],
            {n: {1: "T"} for n in ("tokens", "mask", "content")})
    _export(DecoderStep(model).to(dev),
            (tok[:, 0], torch.ones(1, K, device=dev), torch.zeros(1, 1, H, device=dev),
             torch.randn(1, 1, C, device=dev), torch.zeros(dec_cache, device=dev),
             torch.zeros(dec_cache, device=dev), torch.tensor([T], device=dev),
             torch.tensor([T], device=dev)),
            p("decoder_step"),
            ["tokens", "mask", "x_add", "content", "k_cache_in", "v_cache_in", "pos", "sink"],
            ["codes", "k_cache", "v_cache"])

    # ---- JHCodec
    codec = model.codec
    stack = codec.model.decoder
    a = stack.layers[0].self_attention
    codec_cache = [len(stack.layers), 1, a.window_size, a.n_heads, a.head_dim]
    hop = codec.hop_length
    _export(CodecEncode(codec).to(dev), (torch.randn(1, T, hop, device=dev) * 0.1,),
            p("codec_encode"), ["frames"], ["codes"], {"frames": {1: "T"}, "codes": {1: "T"}})
    _export(CodecDecode(codec).to(dev),
            (torch.zeros(1, 1, K, dtype=torch.long, device=dev),
             torch.zeros(codec_cache, device=dev), torch.zeros(codec_cache, device=dev),
             torch.tensor([20], device=dev)),
            p("codec_decode"), ["codes", "k_cache_in", "v_cache_in", "pos"],
            ["audio", "k_cache", "v_cache"])

    # ---- Zipformer (traced on CPU: its ONNX branches build index tensors on the CPU)
    enc = model.content_encoder
    model.content_encoder.cpu(), model.content_proj.cpu()
    zc = ZipformerChunk(model)
    states = [enc.encoder_embed.get_init_states(1), torch.ones(1, zc.left, dtype=torch.bool)]
    states += enc.encoder.get_init_states(1)
    names = ["embed_state", "cache_padding"] + [f"state_{i}" for i in range(len(states) - 2)]
    _export(zc, (torch.randn(1, 2 * zc.chunk + 13, 80), torch.tensor([zc.chunk + 3]),
                 torch.tensor([zc.chunk]), *states),
            p("zipformer_chunk"), ["fbank", "conv_keep", "valid"] + names,
            ["content", "new_cache_padding", "new_embed_state"] + [f"new_{n}" for n in names[2:]])
    model.content_encoder.to(dev), model.content_proj.to(dev)

    meta = {"n_codebooks": K, "codebook_size": cfg.codebook_size, "hidden_dim": H,
            "content_proj_dim": C, "delays": model.delays, "hop": hop,
            "decoder_cache_shape": dec_cache, "codec_cache_shape": codec_cache,
            "zipformer": {"chunk": zc.chunk, "hop": hop,
                          "front_skip": max(0, -enc.align_pad_frames) * hop,
                          "state_names": names, "state_shapes": [list(s.shape) for s in states]},
            "constants": {"null_content": model.null_content.detach().cpu().view(-1).tolist(),
                          "null_content_film":
                              model.null_content_film.detach().cpu().view(-1).tolist()},
            "opset": OPSET, "config": cfg.to_dict()}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    return meta


@torch.no_grad()
def check(model: LiveAnon, out: str, device_id: int) -> None:
    """Each graph vs its PyTorch counterpart on the same inputs."""
    from liveanon.content import ZipformerStream

    dev = model.device
    sess = lambda n: ort.InferenceSession(  # noqa: E731
        os.path.join(out, f"{n}.onnx"),
        providers=[("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0})])
    np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
    g = torch.Generator(device=dev).manual_seed(0)
    cfg = model.cfg
    K, H, C, T = cfg.n_codebooks, cfg.hidden_dim, cfg.content_proj_dim, 150

    # ---- codec: encode real-ish audio (noise-free sine mix) and decode frame by frame
    t = torch.arange(T * 320, device=dev) / 16000
    wav = (0.3 * torch.sin(2 * np.pi * 220 * t) * torch.sin(2 * np.pi * 3 * t)
           + 0.05 * torch.randn(t.shape, device=dev, generator=g)).view(1, -1)
    ref_codes = model.codec.encode(wav)                                     # (1, K, T)
    codes = sess("codec_encode").run(None, {"frames": np_(wav.view(1, T, 320))})[0]
    match = (codes.transpose(0, 2, 1) == np_(ref_codes)).mean()
    print(f"codec_encode     token match {match * 100:.2f}%")

    ref_audio = model.codec.decode(ref_codes)[0]
    s = sess("codec_decode")
    a = model.codec.model.decoder.layers[0].self_attention
    shape = [len(model.codec.model.decoder.layers), 1, a.window_size, a.n_heads, a.head_dim]
    kc, vc = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    audio = []
    for f in range(T):
        y, kc, vc = s.run(None, {"codes": np_(ref_codes[:, :, f:f + 1].permute(0, 2, 1)),
                                 "k_cache_in": kc, "v_cache_in": vc,
                                 "pos": np.array([f], np.int64)})
        audio.append(y[0])
    audio = np.concatenate(audio)
    ref_audio = np_(ref_audio)
    snr = 10 * np.log10((ref_audio ** 2).sum() / ((audio - ref_audio) ** 2).sum())
    print(f"codec_decode     SNR vs PyTorch {snr:.1f} dB")

    # ---- decoder: prefill + steps vs the eager decoder, past the ring-buffer wrap
    tok = torch.randint(0, cfg.codebook_size, (1, T, K), device=dev, generator=g)
    mask = (torch.rand(1, T, K, device=dev, generator=g) > 0.2).float()
    c = torch.randn(1, T, C, device=dev, generator=g)
    x = DecoderPrefill(model).to(dev).embed(tok, mask)
    x[:, 0] = model.bos[0]
    model.decoder.set_prompt_len(T)
    caches = model.decoder.new_caches(T)
    model.decoder(x, c, caches, use_cache=False)
    kc, vc = sess("decoder_prefill").run(None, {"tokens": np_(tok), "mask": np_(mask),
                                                "content": np_(c)})
    diff = np.abs(kc[:, :, :, :T] - np_(torch.stack([cc.k for cc in caches]))[:, :, :, :T]).max()
    print(f"decoder_prefill  max|k diff| {diff:.2e}")

    step = sess("decoder_step")
    heads_w = torch.cat([h.weight for h in model.heads])
    heads_b = torch.cat([h.bias for h in model.heads])
    embed = DecoderStep(model).to(dev).embed
    n_steps, same = cfg.max_seq_len - T + 40, 0
    for i in range(n_steps):
        tk = torch.randint(0, cfg.codebook_size, (1, K), device=dev, generator=g)
        mk = torch.ones(1, K, device=dev)
        cs = torch.randn(1, 1, C, device=dev, generator=g)
        h = model.decoder(embed(tk[:, None], mk[:, None]), cs, caches, use_cache=True)[:, -1]
        ref = torch.nn.functional.linear(h, heads_w, heads_b).view(1, K, -1).argmax(-1)
        codes, kc, vc = step.run(None, {
            "tokens": np_(tk), "mask": np_(mk), "x_add": np.zeros((1, 1, H), np.float32),
            "content": np_(cs), "k_cache_in": kc, "v_cache_in": vc,
            "pos": np.array([T + i], np.int64), "sink": np.array([T], np.int64)})
        same += (codes == np_(ref)).all()
    print(f"decoder_step     identical codes in {same}/{n_steps} steps "
          f"(ring buffer wraps after {cfg.max_seq_len - T})")

    # ---- Zipformer: one stream of chunks vs ZipformerStream (eager)
    zs = ZipformerStream(model.content_encoder, use_cuda_graph=False)
    z = sess("zipformer_chunk")
    names = [i.name for i in z.get_inputs()][3:]
    st = {n: np.zeros(i.shape, np.float32) for n, i in zip(names, z.get_inputs()[3:])}
    st["cache_padding"] = np.ones(st["cache_padding"].shape, bool)
    ref, got = [], []
    fb = model.content_encoder._fbank(wav[:, 6 * 320:], 16000)
    need, ch = 2 * zs.chunk + 13, zs.chunk
    for off in range(0, fb.shape[1] - need + 1, 2 * ch):
        seg = fb[:, off:off + need]
        emb, zs.embed_state = model.content_encoder.encoder_embed.streaming_forward_exact(
            seg, zs.embed_state, ch)
        ref.append(model.content_proj(zs._run_encoder(emb, ch)))
        outs = z.run(None, {"fbank": np_(seg), "conv_keep": np.array([ch + 3]),
                            "valid": np.array([ch]), **st})
        got.append(outs[0])
        st = dict(zip(["cache_padding", "embed_state"] + names[2:], outs[1:]))
    ref, got = np_(torch.cat(ref, 1)), np.concatenate(got, 1)
    print(f"zipformer_chunk  {len(ref[0])} frames, max|diff| {np.abs(ref - got).max():.2e} "
          f"(|x| max {np.abs(ref).max():.1f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--no_check", action="store_true")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LiveAnon.from_pretrained(args.ckpt, device=f"cuda:{args.device}", fast=False)
    export(model, args.out)
    for f in sorted(os.listdir(args.out)):
        if f.endswith((".onnx", ".json")):
            print(f"{f:22s} {os.path.getsize(os.path.join(args.out, f)) / 2**20:8.1f} MiB")
    if not args.no_check:
        check(model, args.out, args.device)


if __name__ == "__main__":
    main()
