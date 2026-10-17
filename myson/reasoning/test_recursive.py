"""Unit tests for myson/reasoning/recursive.py.

Offline and hermetic: no Ollama, no network, no real embeddings. The LLM is a
scripted mock (same dependency-injection pattern as the SEAL module's tests), and
the verifier is injected with a scripted/constant scorer where a test needs to drive
the control flow deterministically. Together these prove: a run completes, the
verifier gate drives backtracking (and respects the budget cap), each halting path
fires, self-consistency votes, and the verifier degrades cleanly when the LLM judge
returns junk.

Run:  python -m unittest myson.reasoning.test_recursive -v
 or:  python myson/reasoning/test_recursive.py
"""

import importlib.util
import os
import sys
import unittest
from collections import Counter

# Load recursive.py directly by path so the test runs without importing the whole
# myson package (reasoning/__init__ pulls in torch via rat_urm; we don't need it).
# The module must be registered in sys.modules BEFORE exec so @dataclass (under
# `from __future__ import annotations`) can resolve its own module namespace.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("myson_recursive", os.path.join(_HERE, "recursive.py"))
rec = importlib.util.module_from_spec(_spec)
sys.modules["myson_recursive"] = rec
_spec.loader.exec_module(rec)


# ======================================================================================
# Test doubles
# ======================================================================================

class MockLLM:
    """A scripted stand-in for LLMClient. Routes by the stable prompt tags so it can
    answer the four call kinds (next-step / final-answer / done-check / judge)
    independently and record how many of each it saw."""

    def __init__(self, steps=None, answers=None, done=None, judge=None):
        self.counts = Counter()
        self.prompts = []
        self._steps = list(steps) if steps else None
        self._answers = list(answers) if answers else None
        self._ans_i = 0
        self._done = list(done) if done else None
        self._done_i = 0
        self._step_i = 0
        self._judge = judge

    def _cycle(self, lst, i, default):
        if not lst:
            return default, i
        val = lst[i] if i < len(lst) else lst[-1]
        return val, i + 1

    def ask(self, prompt, system="", temperature=0.0, **kw):
        self.prompts.append(prompt)
        if rec.TAG_JUDGE in prompt:
            self.counts["judge"] += 1
            if callable(self._judge):
                return self._judge(prompt)
            return self._judge if self._judge is not None else ""
        if rec.TAG_DONE in prompt:
            self.counts["done"] += 1
            val, self._done_i = self._cycle(self._done, self._done_i, "NO")
            return "YES" if (val is True or str(val).upper().startswith("YES")) else "NO"
        if rec.TAG_FINAL in prompt:
            self.counts["final"] += 1
            val, self._ans_i = self._cycle(self._answers, self._ans_i, "(no answer)")
            return val
        if rec.TAG_NEXT in prompt:
            self.counts["next"] += 1
            if self._steps:
                val = self._steps[self._step_i % len(self._steps)]
                self._step_i += 1
                return val
            return f"reasoning step {self.counts['next']}"
        self.counts["other"] += 1
        return ""

    def chat(self, messages, system="", temperature=0.7, **kw):
        return {"role": "assistant",
                "content": self.ask(messages[-1]["content"], system=system, temperature=temperature)}


class ConstVerifier:
    """Returns the same score for every candidate -- isolates halting/flow tests from
    verifier behavior."""
    def __init__(self, score):
        self.score = score
        self.calls = 0

    def score_batch(self, cands, z_steps, goal, prev_step=""):
        self.calls += 1
        return [self.score] * len(cands)


class ScriptedVerifier:
    """Returns a scripted score batch per call (last batch repeats) -- drives the
    accept/backtrack decision deterministically."""
    def __init__(self, batches):
        self.batches = [list(b) for b in batches]
        self.i = 0
        self.calls = 0

    def score_batch(self, cands, z_steps, goal, prev_step=""):
        self.calls += 1
        b = self.batches[self.i] if self.i < len(self.batches) else self.batches[-1]
        self.i += 1
        if len(b) < len(cands):
            b = b + [b[-1]] * (len(cands) - len(b))
        return b[:len(cands)]


# ======================================================================================
# Tests
# ======================================================================================

class TestBasicRun(unittest.TestCase):
    def test_run_completes_with_real_heuristic_verifier(self):
        """End-to-end with the real (text-heuristic) verifier, judge off, offline."""
        llm = MockLLM(steps=["add the two numbers", "double-check the sum"],
                      answers=["The sum is 43.", "The answer is 43."])
        trace = rec.reason_trace(
            [{"role": "user", "content": "What is 17 + 26?"}], llm=llm,
            max_steps=2, inner_steps=1, n_candidates=2, tau=0.0,  # accept-all: deterministic flow
            use_judge=False, use_done_check=False,
        )
        self.assertTrue(trace.answer)                       # produced an answer
        self.assertEqual(trace.passes, 2)                   # ran the full (small) budget
        self.assertEqual(trace.halt_reason, "max_steps")
        self.assertEqual(len(trace.scratchpad), 2)          # one net step per pass
        self.assertEqual(llm.counts["final"], 2)            # one re-draft per pass
        self.assertEqual(llm.counts["next"], 4)             # inner_steps*n_candidates*passes
        self.assertEqual(llm.counts["judge"], 0)            # judge disabled

    def test_reason_returns_plain_string(self):
        llm = MockLLM(answers=["42"])
        out = rec.reason("What is 6 times 7?", llm=llm,
                         max_steps=1, inner_steps=1, n_candidates=1,
                         use_judge=False, use_done_check=False, verifier=ConstVerifier(0.9))
        self.assertIsInstance(out, str)
        self.assertEqual(out, "42")


class TestBacktrack(unittest.TestCase):
    def test_low_score_triggers_backtrack(self):
        """A sub-tau best candidate must drop a step and resample (backtrack)."""
        v = ScriptedVerifier([[0.9, 0.1],   # accept -> z grows to 1
                              [0.05, 0.02],  # below tau -> BACKTRACK -> z back to 0
                              [0.8, 0.1],    # accept
                              [0.85, 0.2]])  # accept -> inner target met
        llm = MockLLM(answers=["done"])
        trace = rec.reason_trace(
            "solve it", llm=llm, verifier=v,
            max_steps=1, inner_steps=2, n_candidates=2, tau=0.3,
            max_backtracks=3, use_done_check=False,
        )
        self.assertEqual(trace.n_backtracks, 1)
        self.assertTrue(any(s.kind == "backtrack" for s in trace.steps))
        self.assertTrue(any(s.kind == "accept" for s in trace.steps))

    def test_backtrack_budget_is_capped(self):
        """Persistently low scores must not backtrack more than max_backtracks, and the
        loop must still terminate (forced-accept once the budget is spent)."""
        v = ScriptedVerifier([[0.01, 0.01]])   # always below tau
        llm = MockLLM(answers=["x"])
        trace = rec.reason_trace(
            "hard", llm=llm, verifier=v,
            max_steps=1, inner_steps=3, n_candidates=2, tau=0.3,
            max_backtracks=2, use_done_check=False,
        )
        self.assertEqual(trace.n_backtracks, 2)             # capped, never exceeds
        self.assertTrue(any(s.kind == "forced-accept" for s in trace.steps))
        self.assertTrue(trace.answer)                       # terminated, not hung


class TestHalting(unittest.TestCase):
    def test_convergence_halt(self):
        """Identical consecutive drafts (sim >= conv_threshold) halts early."""
        llm = MockLLM(answers=["The answer is 7.", "The answer is 7."])
        trace = rec.reason_trace(
            "q", llm=llm, verifier=ConstVerifier(0.9),
            max_steps=16, inner_steps=1, n_candidates=1,
            use_done_check=False, conv_threshold=0.98,
        )
        self.assertEqual(trace.halt_reason, "convergence")
        self.assertEqual(trace.passes, 2)
        self.assertIsNotNone(trace.pass_records[-1].converged_sim)
        self.assertGreaterEqual(trace.pass_records[-1].converged_sim, 0.98)

    def test_done_check_halt(self):
        """An LLM 'YES, complete' self-check halts after the first pass."""
        llm = MockLLM(answers=["ans one", "ans two"], done=["YES"])
        trace = rec.reason_trace(
            "q", llm=llm, verifier=ConstVerifier(0.9),
            max_steps=16, inner_steps=1, n_candidates=1,
            use_done_check=True,
        )
        self.assertEqual(trace.halt_reason, "done")
        self.assertEqual(trace.passes, 1)
        self.assertEqual(llm.counts["done"], 1)

    def test_hard_cap_halt(self):
        """No convergence, no done -> runs exactly max_steps passes and stops."""
        llm = MockLLM(answers=["a1", "a2", "a3"])           # all different -> never converge
        trace = rec.reason_trace(
            "q", llm=llm, verifier=ConstVerifier(0.9),
            max_steps=3, inner_steps=1, n_candidates=1,
            use_done_check=False,
        )
        self.assertEqual(trace.halt_reason, "max_steps")
        self.assertEqual(trace.passes, 3)


class TestSelfConsistency(unittest.TestCase):
    def test_majority_vote(self):
        """Three chains, answers Paris/Paris/London -> majority 'Paris' wins."""
        llm = MockLLM(answers=["Paris", "Paris", "London"])
        trace = rec.reason_trace(
            "Capital of France?", llm=llm, verifier=ConstVerifier(0.9),
            max_steps=1, inner_steps=1, n_candidates=1,
            use_judge=False, use_done_check=False, self_consistency=3,
        )
        self.assertEqual(trace.halt_reason, "self_consistency")
        self.assertEqual(len(trace.chains), 3)
        self.assertEqual(trace.votes.get("paris"), 2)
        self.assertEqual(trace.votes.get("london"), 1)
        self.assertEqual(rec._normalize_answer(trace.answer), "paris")


class TestVerifierHeuristics(unittest.TestCase):
    def test_progress_signal_prefers_on_topic(self):
        """With an empty scratchpad (novelty/consistency tied), the on-topic candidate
        must outscore the off-topic one -- the progress signal."""
        v = rec.Verifier(use_judge=False)
        goal = "What is the capital of France? The capital is Paris."
        scores = v.score_batch(
            ["The capital of France is the city Paris.",       # on-topic
             "Mitochondria are the powerhouse of the cell."],  # off-topic
            z_steps=[], goal=goal,
        )
        self.assertGreater(scores[0], scores[1])

    def test_novelty_signal_penalizes_repetition(self):
        """A verbatim repeat of an existing step must score below a step that adds new
        information -- the novelty signal (the loop-killer)."""
        v = rec.Verifier(use_judge=False)
        goal = "Compute the total cost of 3 items at $4 each."
        z = ["Three items at four dollars each."]
        scores = v.score_batch(
            ["Three items at four dollars each.",                 # verbatim repeat
             "Multiplying three by four dollars gives twelve."],  # new info
            z_steps=z, goal=goal, prev_step=z[-1],
        )
        self.assertLess(scores[0], scores[1])


class TestGracefulDegradation(unittest.TestCase):
    def test_judge_parse_failure_falls_back_to_heuristic(self):
        """When the LLM judge returns unparseable junk, the verifier must still return
        valid [0,1] scores from the heuristic -- no exception."""
        llm = MockLLM(judge="this is not json, sorry")
        v = rec.Verifier(llm=llm, use_judge=True)
        scores = v.score_batch(["step a", "step b"], z_steps=[],
                               goal="some goal about a and b")
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        self.assertGreaterEqual(llm.counts["judge"], 1)     # judge was attempted

    def test_judge_blends_when_valid(self):
        """A valid judge reply shifts the score toward the judge value (blend works)."""
        llm = MockLLM(judge='[{"i": 0, "score": 1.0}, {"i": 1, "score": 0.0}]')
        v = rec.Verifier(llm=llm, use_judge=True, judge_weight=0.5)
        scores = v.score_batch(["step a", "step b"], z_steps=[],
                               goal="goal")
        self.assertGreater(scores[0], scores[1])            # judge pushed a>b


class TestEmbeddingInjection(unittest.TestCase):
    def test_injected_embedder_is_used(self):
        """If an embed function is supplied, Similarity uses vector cosine, not text."""
        calls = {"n": 0}

        def fake_embed(text):
            calls["n"] += 1
            # map by token count -> identical-length strings look identical
            return [float(len(rec._tokenize(text))), 1.0]

        sim = rec.Similarity(embed=fake_embed)
        s = sim.sim("alpha beta", "gamma delta")            # same token count -> cosine 1.0
        self.assertGreater(calls["n"], 0)                   # embedder was called
        self.assertAlmostEqual(s, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
