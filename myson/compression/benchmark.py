"""Honest compression benchmark — real measurements, no synthetic numbers.

Runs a small real MySonModel and the three compression blocks, measuring what each one
actually buys on THIS machine (CUDA if present, else CPU for correctness):

  * BitNet  : full-model weight memory + one real train step (fwd+bwd) vs the dense model.
  * MLA      : KV-cache bytes per sequence vs standard multi-head attention (the real win).
  * MoR      : inference compute (tokens x recursion-steps) vs fixed max-depth (real gather).

Caveats printed with the numbers: BitNet's memory figure assumes a true 1.58-bit packed
store; we ship an int8 store + float matmul, so wall-clock speed is NOT improved here.

Run:  python -m myson.compression.benchmark
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn

from ..training.model import MySonModel, ModelConfig
from . import bitnet, mla, mor


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _small_config() -> ModelConfig:
    # small enough to run fast anywhere, real architecture (GQA, ConvSwiGLU, RoPE, TRM)
    return ModelConfig(vocab_size=1024, hidden_size=256, intermediate_size=1024,
                       num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
                       max_position_embeddings=256)


def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _train_step(model, ids, dev, warmup: int = 1, iters: int = 3):
    # warmup (kernel compile / allocator) so timings aren't dominated by the first call
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _, loss = model(ids, labels=ids); loss.backward()
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        _, loss = model(ids, labels=ids); loss.backward(); last = float(loss)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
    return dt, peak, last


def bench_bitnet(dev):
    print("\n== BitNet (full MySonModel, dense vs 1.58-bit) ==")
    cfg = _small_config()
    B, T = 4, 64
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

    dense = MySonModel(cfg).to(dev)
    dt_d, peak_d, loss_d = _train_step(dense, ids, dev)
    rep_d = bitnet.memory_report(dense)

    bit = bitnet.convert_to_bitnet(MySonModel(cfg)).to(dev)
    dt_b, peak_b, loss_b = _train_step(bit, ids, dev)
    rep_b = bitnet.memory_report(bit)

    print(f"  params:            {_params(dense):,}")
    print(f"  weight mem  dense: {rep_d['total_MB']:.2f} MB (all fp32)")
    print(f"  weight mem  bitnet:{rep_b['total_MB']:.2f} MB  "
          f"({rep_b['compression_vs_fp32']:.2f}x smaller, ternary linears @1.58b)")
    print(f"  train step  dense: {dt_d*1e3:.1f} ms, loss {loss_d:.3f}"
          + (f", peak VRAM {peak_d:.0f} MB" if dev == 'cuda' else ""))
    print(f"  train step  bitnet:{dt_b*1e3:.1f} ms, loss {loss_b:.3f}"
          + (f", peak VRAM {peak_b:.0f} MB" if dev == 'cuda' else ""))
    print("  note: memory win is real; wall-clock not improved (int8 store, float matmul,")
    print("        no packed-ternary kernel — see module docstring).")


def bench_mla(dev):
    print("\n== MLA vs standard MHA (KV-cache at a long context) ==")
    d_model, heads, d_latent, T = 256, 8, 32, 512
    x = torch.randn(1, T, d_model, device=dev)

    m = mla.MultiHeadLatentAttention(d_model, heads, d_latent).to(dev).eval()
    with torch.no_grad():
        _, latent, info = m(x, use_cache=True)
    # standard MHA caches full K and V per token; MLA caches one latent per token
    std_kv_bytes = 2 * T * d_model * 4
    mla_bytes = latent.numel() * 4
    print(f"  seq len {T}, d_model {d_model}, d_latent {d_latent}")
    print(f"  standard K+V cache: {std_kv_bytes/1024:.1f} KB")
    print(f"  MLA latent cache:   {mla_bytes/1024:.1f} KB  "
          f"({std_kv_bytes/mla_bytes:.1f}x smaller, {info['kv_cache_saved']} saved)")
    print("  (verified correct for prefill + incremental decode in test_compression.py)")


def bench_mor(dev):
    print("\n== MoR adaptive depth vs fixed max-depth (inference compute) ==")
    hidden, max_depth, B, T = 256, 6, 2, 128
    m = mor.MixtureOfRecursions(hidden, max_depth=max_depth, min_depth=1).to(dev).eval()
    with torch.no_grad():
        m.depth_predictor.predictor[-1].weight.normal_(0, 2.0)   # spread the depths out
        x = torch.randn(B, T, hidden, device=dev)
        _, info = m(x)
    print(f"  tokens {B*T}, max_depth {max_depth}")
    print(f"  fixed max-depth cost: {info['dense_tokens_x_steps']} token-steps")
    print(f"  MoR adaptive cost:    {info['tokens_x_steps']} token-steps  "
          f"({info['compute_saved']} saved, mean depth {info['mean_depth']:.2f})")
    print("  (hard gather/scatter — only active tokens are computed per depth)")


def main():
    dev = _device()
    torch.manual_seed(0)
    print("=" * 64)
    print(f"myson compression benchmark — device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    print("=" * 64)
    bench_bitnet(dev)
    bench_mla(dev)
    bench_mor(dev)
    print("\n" + "=" * 64)
    print("done. these are measured on this machine, not synthetic.")
    print("=" * 64)


if __name__ == "__main__":
    main()
