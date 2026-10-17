"""Multi-head Latent Attention (MLA) — low-rank KV-cache compression (own-model track).

DeepSeek-V2/V3's trick: instead of caching full K and V per token, cache one small latent
vector per token and up-project to K/V on the fly. The KV cache shrinks from 2*d_model to
d_latent per token.

Lifted from the corpus and FIXED — the original had two real bugs:
  1. RoPE was mis-wired: apply_rope(q, k, cos[:T], cos[:kv_len]) passed `cos` into the
     `sin` slot, so the rotation used cosine twice and never sine.
  2. In cached decode (T=1, kv_len=N) that mismatched-length `sin` broadcast q from length
     1 up to N, which then crashed at the final reshape(B, T, D).
Here RoPE is applied to q and k SEPARATELY, each with its own position slice (q sits at the
last T positions of the sequence, k spans 0..kv_len), which is correct for both prefill and
incremental generation.

Attaches to a model whose attention we define (the repo's own transformer), NOT to a
black-box Ollama model. Reference: DeepSeek-V2, arXiv:2405.04434.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCompressor(nn.Module):
    """x -> latent (down) and latent -> K, V (up). d_latent << 2*d_model."""

    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.compress = nn.Linear(d_model, d_latent, bias=False)
        self.decompress_k = nn.Linear(d_latent, d_model, bias=False)
        self.decompress_v = nn.Linear(d_latent, d_model, bias=False)
        # cache size per token vs full K+V per token
        self.compression_ratio = d_latent / (2 * d_model)

    def forward(self, x: torch.Tensor):
        latent = self.compress(x)
        return latent, self.decompress_k(latent), self.decompress_v(latent)


class RotaryEmbeddingMLA(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)             # [seq, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate one tensor x [B, H, L, head_dim] by position cos/sin [L, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, L, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, d_latent: Optional[int] = None,
                 rope_theta: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_latent = d_latent or max(1, d_model // 4)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_compressor = LatentCompressor(d_model, self.d_latent)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbeddingMLA(self.head_dim, max_seq_len, rope_theta)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_latent: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, T, D = hidden_states.shape
        q = self.q_proj(hidden_states)
        latent, k_new, v_new = self.kv_compressor(hidden_states)

        if past_latent is not None:
            past_k = self.kv_compressor.decompress_k(past_latent)
            past_v = self.kv_compressor.decompress_v(past_latent)
            k = torch.cat([past_k, k_new], dim=1)
            v = torch.cat([past_v, v_new], dim=1)
            latent = torch.cat([past_latent, latent], dim=1)
        else:
            k, v = k_new, v_new
        kv_len = k.shape[1]

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)        # [B,H,T,hd]
        k = k.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)   # [B,H,kv,hd]
        v = v.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE — q occupies the LAST T positions of a kv_len-long sequence; k spans all.
        cos, sin = self.rope(kv_len)
        q = apply_rope(q, cos[kv_len - T:kv_len], sin[kv_len - T:kv_len])
        k = apply_rope(k, cos[:kv_len], sin[:kv_len])

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,kv]
        if attention_mask is not None:
            attn = attn + attention_mask
        elif T > 1:
            # causal: query i (global pos kv_len-T+i) may attend to keys <= its own pos
            causal = torch.triu(torch.ones(T, kv_len, device=hidden_states.device),
                                diagonal=kv_len - T + 1).bool()
            attn = attn.masked_fill(causal, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                      # [B,H,T,hd]

        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)  # [B,T,H,1]
        out = out.transpose(1, 2) * gate                 # [B,T,H,hd]
        out = out.reshape(B, T, D)
        out = self.o_proj(out)

        info = {"compression_ratio": self.kv_compressor.compression_ratio,
                "latent_size": self.d_latent,
                "kv_cache_saved": f"{(1 - self.kv_compressor.compression_ratio) * 100:.1f}%"}
        return out, (latent if use_cache else None), info


class MLATransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 d_latent: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads, d_latent)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_dim), nn.SiLU(),
                                 nn.Linear(mlp_dim, d_model))

    def forward(self, x, use_cache=False, past_latent=None):
        a, new_latent, info = self.attn(self.norm1(x), use_cache=use_cache, past_latent=past_latent)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_latent, info
