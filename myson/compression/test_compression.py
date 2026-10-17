"""Tests for the compression modules — forward/backward correctness AND regressions for
the specific bugs that were fixed when lifting them from the corpus.

Run:  python myson/compression/test_compression.py
 or:  python -m unittest myson.compression.test_compression -v
"""
from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from myson.compression import bitnet, mla, mor


class TestBitNet(unittest.TestCase):
    def test_ternary_quant_values_and_ste_gradient(self):
        w = torch.tensor([[2.0, -2.0, 0.0, 0.01]], requires_grad=True)
        q = bitnet.quantize_weights_158bit(w)
        # forward values are in {-1,0,+1}
        self.assertTrue(set(q.detach().flatten().tolist()).issubset({-1.0, 0.0, 1.0}))
        # STE: gradient passes through (d q / d w == 1)
        q.sum().backward()
        self.assertTrue(torch.allclose(w.grad, torch.ones_like(w)))

    def test_bitlinear_trains(self):
        layer = bitnet.BitLinear(16, 8)
        x = torch.randn(4, 16)
        out = layer(x)
        self.assertEqual(out.shape, (4, 8))
        out.sum().backward()
        self.assertIsNotNone(layer.weight.grad)

    def test_inference_matches_training_forward(self):
        bl = bitnet.BitLinear(32, 32, quant_act=False)
        x = torch.randn(2, 32)
        with torch.no_grad():
            train_out = bl(x)
        inf = bitnet.BitLinearInference.from_bitlinear(bl)
        self.assertTrue(torch.allclose(train_out, inf(x), atol=1e-5))

    def test_convert_and_memory_report(self):
        m = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        bitnet.convert_to_bitnet(m)
        self.assertTrue(any(isinstance(mod, bitnet.BitLinear) for mod in m.modules()))
        rep = bitnet.memory_report(m)
        self.assertGreater(rep["compression_vs_fp32"], 1.5)   # quantized is smaller than fp32


class TestMLA(unittest.TestCase):
    def test_prefill_forward_backward(self):
        m = mla.MultiHeadLatentAttention(d_model=64, num_heads=8, d_latent=16)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out, latent, info = m(x, use_cache=True)
        self.assertEqual(out.shape, (2, 10, 64))
        self.assertEqual(latent.shape, (2, 10, 16))     # latent cache << full KV
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_cached_decode_runs_and_shapes_are_right(self):
        """The regression: the old code crashed here (RoPE sin-slot / broadcast bug)."""
        m = mla.MultiHeadLatentAttention(d_model=64, num_heads=8, d_latent=16).eval()
        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            _, latent, _ = m(x, use_cache=True)
            for _ in range(4):                          # generate 4 tokens, one at a time
                tok = torch.randn(1, 1, 64)
                out, latent, _ = m(tok, use_cache=True, past_latent=latent)
                self.assertEqual(out.shape, (1, 1, 64)) # stays length 1 — no broadcast blowup
        self.assertEqual(latent.shape[1], 10)           # 6 prefill + 4 decoded

    def test_rope_actually_uses_sine(self):
        """Guard against the 'cos passed into the sin slot' bug: with real sin, a rotated
        tensor must differ from one rotated with sin forced to zero."""
        head = torch.randn(1, 2, 5, 8)
        cos = torch.randn(5, 8); sin = torch.randn(5, 8)
        with_sin = mla.apply_rope(head, cos, sin)
        no_sin = mla.apply_rope(head, cos, torch.zeros_like(sin))
        self.assertFalse(torch.allclose(with_sin, no_sin))


class TestMoR(unittest.TestCase):
    def test_training_is_differentiable_and_trains_predictor(self):
        """The regression: the predictor used to get zero gradient (argmax + unused probs)."""
        m = mor.MixtureOfRecursions(hidden_size=32, max_depth=4).train()
        x = torch.randn(2, 8, 32)
        out, info = m(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIn("aux_loss", info)
        (out.sum() + info["aux_loss"]).backward()
        pred_grads = [p.grad for p in m.depth_predictor.parameters() if p.grad is not None]
        self.assertTrue(pred_grads, "depth predictor received no gradient")
        self.assertTrue(any(g.abs().sum() > 0 for g in pred_grads))

    def test_inference_does_real_gather_savings(self):
        """The regression: inference used to run the block on ALL tokens every depth.
        Now a mix of shallow/deep tokens must cost < dense (every token x max_depth)."""
        m = mor.MixtureOfRecursions(hidden_size=32, max_depth=6, min_depth=1).eval()
        # bias the predictor so different tokens pick different depths
        with torch.no_grad():
            m.depth_predictor.predictor[-1].weight.normal_(0, 3.0)
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            out, info = m(x)
        self.assertEqual(out.shape, x.shape)
        self.assertLess(info["tokens_x_steps"], info["dense_tokens_x_steps"])

    def test_inference_matches_sequential_depth_semantics(self):
        """A token with hard depth k must equal applying the shared block k times."""
        m = mor.MixtureOfRecursions(hidden_size=16, max_depth=5, min_depth=1,
                                    shared_block=True).eval()
        with torch.no_grad():
            x = torch.randn(1, 4, 16)
            depths, _ = m.depth_predictor(x)
            out, _ = m(x)
            blk = m.blocks[0]
            for t in range(4):
                ref = x[:, t:t + 1, :]
                for _ in range(int(depths[0, t].item())):
                    ref = blk(ref)
                self.assertTrue(torch.allclose(out[:, t:t + 1, :], ref, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
