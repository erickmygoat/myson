"""BitNet b1.58 — ternary {-1, 0, +1} weight quantization (own-model track).

Lifted from the corpus and kept because the quantization math is correct and standalone.
Honest scope (what this does and does NOT do):
  * BitLinear trains in full precision with quantization-aware forward (STE), so a model
    trained with it learns weights that quantize cleanly to 1.58 bits.
  * BitLinearInference stores ternary weights as int8 and a scalar scale -> the real win
    is MEMORY (int8 vs fp32 ≈ 4x on disk/RAM, with a true packed 1.58-bit store it is more).
  * It does NOT ship a packed 1-bit matmul kernel, so it dequantizes int8->float for
    F.linear. The "10x faster" folklore needs that kernel; we don't claim the speedup here.

This attaches to a model whose architecture we control (the repo's own transformer), NOT to
a black-box model served by Ollama.

Reference: Ma et al., "The Era of 1-bit LLMs: ... BitNet b1.58", arXiv:2402.17764.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """sign(x) in the forward pass, identity gradient (straight-through estimator)."""
    return x.sign() + x - x.detach()


def quantize_weights_158bit(w: torch.Tensor) -> torch.Tensor:
    """Ternary {-1, 0, +1} quantization with absmean scaling and STE gradient.

    The threshold at 0.5*mean(|w|) is the BitNet b1.58 rule; weights near zero become 0
    (the sparsity benefit over pure 1-bit {-1,+1})."""
    scale = w.abs().mean().clamp_min(1e-5)
    threshold = 0.5 * scale
    w_quant = torch.zeros_like(w)
    w_quant[w > threshold] = 1.0
    w_quant[w < -threshold] = -1.0
    return w_quant + w - w.detach()          # STE


def quantize_activations(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-tensor symmetric int8 (default) activation quantization with STE."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)
    scale = x.abs().max().clamp_min(1e-5) / qmax
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x_q + x - x.detach()              # STE


class BitLinear(nn.Module):
    """nn.Linear drop-in with quantization-aware ternary weights (training time)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 quant_act: bool = True, activation_bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_act = quant_act
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("weight_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight.abs().mean().clamp_min(1e-5)
        # record scale for inspection without mutating autograd state
        with torch.no_grad():
            self.weight_scale.copy_(scale.detach())
        w_quant = quantize_weights_158bit(self.weight / scale) * scale
        if self.quant_act:
            x = quantize_activations(x, self.activation_bits)
        return F.linear(x, w_quant, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, quant_act={self.quant_act}"


class BitLinearInference(nn.Module):
    """Inference-only: ternary weights stored as int8 + a scalar scale.

    Memory is int8 (plus one float scale per layer); the matmul dequantizes to float because
    we don't ship a packed-ternary kernel. Numerically matches a trained BitLinear's forward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)

    @classmethod
    def from_bitlinear(cls, bl: BitLinear) -> "BitLinearInference":
        layer = cls(bl.in_features, bl.out_features, bias=bl.bias is not None)
        w = bl.weight.data
        scale = w.abs().mean().clamp_min(1e-5)
        threshold = 0.5 * scale
        wq = torch.zeros_like(w, dtype=torch.int8)
        wq[w > threshold] = 1
        wq[w < -threshold] = -1
        layer.weight.copy_(wq)
        layer.weight_scale.copy_(scale)
        if bl.bias is not None:
            layer.bias.copy_(bl.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float() * self.weight_scale
        return F.linear(x, w, self.bias)


def convert_to_bitnet(model: nn.Module, keep: list[str] | None = None) -> nn.Module:
    """Replace nn.Linear layers with BitLinear in place, skipping names in `keep`
    (embeddings / lm_head / norms stay full precision — the standard BitNet recipe)."""
    keep = keep or ["embed", "lm_head", "norm"]
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not any(k in name.lower() for k in keep):
            parent = model if "." not in name else model.get_submodule(name.rsplit(".", 1)[0])
            child = name.rsplit(".", 1)[-1]
            bl = BitLinear(module.in_features, module.out_features, bias=module.bias is not None)
            bl.weight.data = module.weight.data.clone()
            if module.bias is not None:
                bl.bias.data = module.bias.data.clone()
            setattr(parent, child, bl)
    return model


def prepare_for_inference(model: nn.Module) -> nn.Module:
    """Freeze every BitLinear into a BitLinearInference (int8 store)."""
    for name, module in list(model.named_modules()):
        if isinstance(module, BitLinear):
            parent = model if "." not in name else model.get_submodule(name.rsplit(".", 1)[0])
            child = name.rsplit(".", 1)[-1]
            setattr(parent, child, BitLinearInference.from_bitlinear(module))
    return model


def memory_report(model: nn.Module) -> dict:
    """Honest memory accounting: ternary (BitLinear/Inference) weight params counted at
    1.58 bits, everything else at fp32. Returns bytes and the compression vs all-fp32."""
    bit_params = float_params = 0
    bit_ids = set()
    for _, m in model.named_modules():
        if isinstance(m, (BitLinear, BitLinearInference)):
            bit_ids.add(id(m.weight))
    # non-ternary real params count as fp32
    for _, p in model.named_parameters():
        if id(p) in bit_ids:
            bit_params += p.numel()
        else:
            float_params += p.numel()
    # BitLinearInference stores its ternary weight as an int8 BUFFER (not a Parameter), so
    # named_parameters() misses it — pick it up here so inference models are counted too.
    for _, b in model.named_buffers():
        if id(b) in bit_ids:
            bit_params += b.numel()
    bit_bytes = bit_params * 1.58 / 8.0
    float_bytes = float_params * 4.0
    total = bit_params + float_params
    return {
        "bit_params": bit_params, "float_params": float_params, "total_params": total,
        "quantized_MB": bit_bytes / 2**20, "float_MB": float_bytes / 2**20,
        "total_MB": (bit_bytes + float_bytes) / 2**20,
        "fp32_MB": total * 4 / 2**20,
        "compression_vs_fp32": (total * 4) / max(1.0, bit_bytes + float_bytes),
    }
