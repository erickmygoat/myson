"""Glue tests for LocalLLM — the logic WE wrote (prompt rendering, generate/decode slicing,
embed pooling), exercised with fake tokenizer/model so no multi-GB weights are downloaded.

We do NOT re-test transformers' generate(); we test that our wiring around it is correct.

Run:  python -m unittest myson.core.test_local_llm -v
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from myson.core.local_llm import LocalLLM


class _FakeInputs(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self, has_template=True):
        self.chat_template = "TEMPLATE" if has_template else None
        self.eos_token_id = 0
        self.last_prompt = None

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "CHAT:" + "|".join(f"{m['role']}={m['content']}" for m in msgs)

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        self.last_prompt = text
        # 3 prompt tokens, all attended
        return _FakeInputs(input_ids=torch.tensor([[5, 6, 7]]),
                           attention_mask=torch.tensor([[1, 1, 1]]))

    def decode(self, ids, skip_special_tokens=True):
        # ids are the NEW tokens only (we slice before calling); map to text
        return "GEN(" + ",".join(str(int(i)) for i in ids) + ")"


class FakeModel:
    device = "cpu"

    def eval(self):
        return self

    def to(self, _d):
        return self

    def generate(self, **kw):
        input_ids = kw["input_ids"]
        # echo the 3 prompt tokens + 2 "generated" tokens (9, 10)
        new = torch.tensor([[9, 10]])
        return torch.cat([input_ids, new], dim=1)

    def __call__(self, **kw):
        class Out:
            # [1, T=3, H=4] hidden states
            hidden_states = [torch.ones(1, 3, 4) * 2.0]
        return Out()


class TestLocalLLMGlue(unittest.TestCase):
    def _llm(self, has_template=True):
        llm = LocalLLM(model="fake/model")
        llm._tok = FakeTokenizer(has_template=has_template)
        llm._model = FakeModel()
        llm._dev = "cpu"
        return llm

    def test_render_uses_chat_template_when_present(self):
        llm = self._llm(has_template=True)
        p = llm._render([{"role": "user", "content": "hi"}], system="S")
        self.assertTrue(p.startswith("CHAT:"))
        self.assertIn("system=S", p)
        self.assertIn("user=hi", p)

    def test_render_falls_back_without_template(self):
        llm = self._llm(has_template=False)
        p = llm._render([{"role": "user", "content": "hi"}], system=None)
        self.assertIn("USER: hi", p)
        self.assertTrue(p.strip().endswith("ASSISTANT:"))

    def test_chat_slices_off_the_prompt_and_decodes_only_new_tokens(self):
        llm = self._llm()
        msg = llm.chat([{"role": "user", "content": "hi"}], system="S", temperature=0.0)
        # only the 2 generated tokens (9,10) should be decoded, not the 3 prompt tokens
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "GEN(9,10)")

    def test_ask_wraps_chat(self):
        llm = self._llm()
        self.assertEqual(llm.ask("hello"), "GEN(9,10)")

    def test_embed_mean_pools_hidden_states(self):
        llm = self._llm()
        v = llm.embed("whatever")
        # hidden states are all 2.0 -> mean pool is 2.0 across H=4 dims
        self.assertEqual(len(v), 4)
        self.assertTrue(all(abs(x - 2.0) < 1e-6 for x in v))

    def test_available_reflects_transformers_presence(self):
        # transformers IS installed in this env, so available() is True
        self.assertTrue(LocalLLM(model="x").available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
