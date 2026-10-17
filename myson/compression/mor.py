"""Mixture-of-Recursions (MoR) — adaptive per-token depth (own-model track).

Not every token needs the same amount of refinement; MoR lets a learned predictor spend
more recursive steps on hard tokens and fewer on easy ones.

Lifted from the corpus and FIXED — the original had two real bugs:
  1. NO actual compute saving: it ran the block on ALL tokens at every depth and then
     masked the result (torch.where), so it did MORE work than a dense block, not less.
  2. NON-trainable routing: depth = argmax(logits) is non-differentiable and the soft
     probabilities were computed but never used in any loss, so the predictor got no
     gradient and never learned which tokens are hard.

The fix follows the standard adaptive-compute recipe (PonderNet / Mixture-of-Depths):
  * TRAINING: fully differentiable. We compute the per-depth states and return their
    depth-distribution-weighted mixture, so task-loss gradient flows into the predictor;
    a ponder aux_loss penalizes expected depth so it learns to be economical. (Training
    does full compute — that is expected; the saving is an inference-time property.)
  * INFERENCE: hard routing with real gather/scatter — at depth d only the tokens whose
    predicted depth >= d are gathered, refined, and scattered back. That is the genuine
    compute saving.

Attaches to a model we define, NOT a black-box Ollama model.
Reference: Bae et al., "Mixture-of-Recursions", 2025; Banino et al., "PonderNet", 2021.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveBlock(nn.Module):
    """One gated-residual refinement step. Works on [..., D] (so [B,T,D] or [N,D])."""

    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * expansion), nn.SiLU(),
                                 nn.Linear(hidden_size * expansion, hidden_size))
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        return x + torch.sigmoid(self.gate(normed)) * self.mlp(normed)


class DepthPredictor(nn.Module):
    """Per-token depth distribution over 1..max_depth."""

    def __init__(self, hidden_size: int, max_depth: int = 8):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(),
                                        nn.Linear(hidden_size // 2, max_depth))
        self.log_temperature = nn.Parameter(torch.zeros(1))   # temp = softplus(.)+eps > 0

    def forward(self, x: torch.Tensor):
        logits = self.predictor(x)                              # [B,T,max_depth]
        temp = F.softplus(self.log_temperature) + 1e-3
        probs = F.softmax(logits / temp, dim=-1)                # [B,T,max_depth], d=1..max
        depths = logits.argmax(dim=-1) + 1                      # [B,T] in [1,max_depth]
        return depths, probs


class MixtureOfRecursions(nn.Module):
    def __init__(self, hidden_size: int, max_depth: int = 8, min_depth: int = 1,
                 shared_block: bool = True, ponder_lambda: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        self.min_depth = max(1, min_depth)
        self.shared_block = shared_block
        self.ponder_lambda = ponder_lambda
        self.depth_predictor = DepthPredictor(hidden_size, max_depth)
        self.blocks = nn.ModuleList(
            [RecursiveBlock(hidden_size)] if shared_block
            else [RecursiveBlock(hidden_size) for _ in range(max_depth)])
        self.register_buffer("depth_histogram", torch.zeros(max_depth))
        self.register_buffer("total_tokens", torch.zeros(1))

    def _block(self, d: int):
        return self.blocks[0] if self.shared_block else self.blocks[d - 1]

    def forward(self, x: torch.Tensor, adaptive: bool = True) -> Tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        if not adaptive:
            out = x
            for d in range(1, self.max_depth + 1):
                out = self._block(d)(out)
            return out, {"mean_depth": float(self.max_depth), "aux_loss": x.new_zeros(())}

        depths, probs = self.depth_predictor(x)                 # [B,T], [B,T,max_depth]

        if self.training:
            # ---- differentiable soft mixture: out = sum_d P(depth=d) * state_d ----
            state = x
            out = torch.zeros_like(x)
            for d in range(1, self.max_depth + 1):
                state = self._block(d)(state)    # state after d refinements
                out = out + probs[..., d - 1:d] * state
            # ponder penalty on expected depth (differentiable), normalized to [0,1]
            depth_idx = torch.arange(1, self.max_depth + 1, device=x.device, dtype=probs.dtype)
            expected_depth = (probs * depth_idx).sum(-1)        # [B,T]
            aux_loss = self.ponder_lambda * (expected_depth / self.max_depth).mean()
            with torch.no_grad():
                for d in range(self.max_depth):
                    self.depth_histogram[d] += (depths == d + 1).sum().float()
                self.total_tokens += B * T
            info = {"mean_depth": expected_depth.mean().item(), "aux_loss": aux_loss}
            return out, info

        # ---- inference: hard routing with REAL gather/scatter ----
        eff = depths.clamp_min(self.min_depth).reshape(-1)      # [N]
        out = x.reshape(-1, D).clone()                          # [N, D]
        computed = 0
        for d in range(1, self.max_depth + 1):
            idx = (eff >= d).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            out[idx] = self._block(d)(out[idx])                 # only active tokens
            computed += idx.numel()
        out = out.reshape(B, T, D)
        dense_cost = B * T * self.max_depth
        info = {"mean_depth": eff.float().mean().item(),
                "tokens_x_steps": computed, "dense_tokens_x_steps": dense_cost,
                "compute_saved": f"{(1 - computed / max(1, dense_cost)) * 100:.1f}%",
                "aux_loss": x.new_zeros(())}
        return out, info

    def efficiency_stats(self) -> dict:
        if float(self.total_tokens) == 0:
            return {}
        dist = (self.depth_histogram / self.total_tokens)
        mean_depth = sum((i + 1) * dist[i].item() for i in range(self.max_depth))
        return {"mean_depth": mean_depth, "max_depth": self.max_depth,
                "compute_saved_vs_max": f"{(1 - mean_depth / self.max_depth) * 100:.1f}%",
                "distribution": dist.tolist()}


class MoRTransformerBlock(nn.Module):
    """Attention (unchanged) + MoR adaptive-depth refinement in place of a fixed FFN."""

    def __init__(self, hidden_size: int, num_heads: int = 8, max_depth: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.mor = MixtureOfRecursions(hidden_size, max_depth)
        self.mor_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        a = self.attn_norm(x)
        a, _ = self.attention(a, a, a, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m, info = self.mor(self.mor_norm(x))
        return x + m, info
