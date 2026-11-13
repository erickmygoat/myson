"""Text-space recursive reasoning for the myson brain.

This is the runtime reasoning core. It reuses the *control algorithm* of
``reasoning/rat_urm.py`` (RAT-URM: generate -> verify -> backtrack -> re-draft,
with a 3-signal verifier and a bounded backtrack budget) but re-homes it from
LATENT space to TEXT space, because the base model (a downloaded open model) is a
black box that emits only tokens -- there are no hidden states to iterate.

TRM's state split maps to text:

    y = the current answer draft
    z = the reasoning scratchpad (a running chain-of-thought, a list of steps)

The loop (per query), capped so it always terminates:

    for each outer pass (<= max_steps=16):          # TRM's T / Nsup schedule
        for a few inner steps:                      # TRM's inner n
            sample N temperature-diverse candidate next-steps from the LLM
            score each with the verifier (3 signals, below)
            best-of-N: if best < tau -> BACKTRACK (drop a step, resample hotter)
                       else           -> APPEND the best step to z
        re-draft y from (question, z) at temperature 0
        halt if y converged, or an LLM self-check says it's done, or the cap hit
    optionally: run several chains and majority-vote the answer (self-consistency)

The verifier's three signals are lifted straight from ``RATVerifier``:

    consistency  - does the step follow from the scratchpad so far?
    progress     - does it move toward the goal/answer?
    novelty      - does it add new information rather than restate? (kills loops)

Cold-start (no trained aggregator yet, per the design doc): the signals are
computed as cheap text heuristics, optionally blended with an LLM-as-judge score.
If an embedding function is injected it is used for similarity; otherwise a
stdlib term-frequency cosine is used -- so the module runs with nothing but the
Python standard library and a chat LLM.

====================================================================
HONEST CAVEAT (read this before believing any accuracy claim)
====================================================================
This is GATED TEXT CHAIN-OF-THOUGHT, not latent reasoning. We do NOT iterate the
model's hidden states the way TRM / HRM / Huginn do -- a black-box API makes that
impossible. What recursion-over-a-9B can buy: it catches mid-chain arithmetic and
logic slips (verify + backtrack), and best-of-N / self-consistency are reliable,
well-established test-time-compute gains on multi-step tasks. What it CANNOT buy:
no new knowledge (recursion only amplifies what the 9B already knows); gains are
capped by verifier quality, and a 9B judging 9B output has correlated errors, so a
confidently-wrong verifier entrenches mistakes; and returns diminish while every
pass costs N x-k x more tokens and latency. It trades latency for accuracy -- it
does not make a 9B a frontier model. Invest in the verifier, not in more loops.

Safety note: reasoning runs UNDER the base system prompt (config.SYSTEM), which
carries the refusal boundary. This module never strips, rewrites, or bypasses that
prompt; it only adds reasoning scaffolding on top of it.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

# --- Imports, resilient to being run inside the package or standalone in tests ---
try:                                        # installed as a package
    from myson.core.llm import LLMClient
except Exception:                           # pragma: no cover - import shim
    try:                                    # relative, when imported as myson.reasoning.recursive
        from ..core.llm import LLMClient
    except Exception:
        LLMClient = None                    # must be injected via reason(..., llm=...)

try:
    from myson.config import SYSTEM as MYSON_SYSTEM
except Exception:                           # pragma: no cover - import shim
    try:
        from ..config import SYSTEM as MYSON_SYSTEM
    except Exception:
        MYSON_SYSTEM = (
            "You are MY SON, a private local assistant running on the user's own "
            "machine. You are direct, capable, and candid. Hard boundary: refuse "
            "anything that would harm the user or other people. Everything else: help."
        )


# ======================================================================================
# Defaults. These mirror the design doc and rat_urm.py (quality_threshold=0.3,
# max_steps=16, max_backtracks=3). Everything is overridable per call via **opts.
# ======================================================================================

DEFAULTS = {
    # --- loop shape (TRM (n, T) schedule, in text) ---
    "max_steps": 16,            # outer passes -- the hard cap. TRM Nsup=16.
    "inner_steps": 2,           # net scratchpad steps added per outer pass (TRM inner n)
    "n_candidates": 4,          # best-of-N candidates sampled per step
    # --- verifier gate ---
    "tau": 0.3,                 # accept if best candidate score > tau, else backtrack
    "weights": (0.35, 0.40, 0.25),  # (consistency, progress, novelty) cold-start weights
    "use_judge": True,          # blend an LLM-as-judge score into the verifier
    "judge_weight": 0.5,        # how much the judge counts vs. the text heuristic
    "repeat_floor": 0.2,        # novelty below this = a loop; the step is damped toward 0
    # --- backtracking (bounded, as in rat_urm.BacktrackController max_backtracks=3) ---
    "max_backtracks": 3,
    "backtrack_depth": 1,       # steps dropped from the scratchpad per backtrack
    # --- sampling temperatures ---
    "temperature": 0.7,         # candidate sampling (the text analog of noise injection)
    "hot_temperature": 1.0,     # resample hotter right after a backtrack
    "redraft_temperature": 0.0, # answer re-draft is deterministic
    # --- halting ---
    "conv_threshold": 0.98,     # sim(y, y_new) >= this -> converged (RecursiveRefinement)
    "use_done_check": True,     # also ask the LLM "complete and correct?"
    # --- self-consistency ---
    "self_consistency": 1,      # >1 runs that many chains and majority-votes the answer
    # --- budget / misc ---
    "max_scratchpad_chars": 6000,   # token ceiling: truncate the oldest CoT when rendering
    "system": None,             # base system prompt; None -> config.SYSTEM
    "embed": None,              # optional Callable[[str], Sequence[float]] for similarity
    "verifier": None,           # optional pre-built Verifier (dependency injection / tests)
}

CAVEAT = (
    "Gated text chain-of-thought, not latent reasoning: this recurses over the "
    "model's TEXT output (verify + backtrack + best-of-N + self-consistency), it "
    "does not iterate hidden states like TRM/HRM/Huginn. It catches mid-chain slips "
    "and gives reliable test-time-compute gains, but it adds no new knowledge, is "
    "capped by verifier quality (a 9B judging a 9B has correlated errors), and trades "
    "latency/tokens for accuracy. It will not turn a 9B into a frontier model."
)

# Stable markers embedded in each prompt so a caller (or a mock) can tell the four
# kinds of LLM call apart without brittle string matching.
TAG_NEXT = "[[MYSON:NEXT_STEP]]"
TAG_FINAL = "[[MYSON:FINAL_ANSWER]]"
TAG_DONE = "[[MYSON:DONE_CHECK]]"
TAG_JUDGE = "[[MYSON:SCORE_CANDIDATES]]"

_log = logging.getLogger(__name__)


# ======================================================================================
# Text similarity -- the embedding bridge, with a stdlib fallback.
# ======================================================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _tf_cosine(a: str, b: str) -> float:
    """Term-frequency cosine similarity in [0, 1]. Pure stdlib -- the offline fallback
    for when no embedding model is available."""
    ta, tb = Counter(_tokenize(a)), Counter(_tokenize(b))
    if not ta or not tb:
        return 0.0
    common = set(ta) & set(tb)
    dot = sum(ta[t] * tb[t] for t in common)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _vec_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    # cosine is in [-1, 1]; map to [0, 1] so it composes with the heuristic signals
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))


class Similarity:
    """Semantic-ish similarity with an optional embedding backend.

    If ``embed`` (a ``Callable[[str], Sequence[float]]``, e.g. an Ollama
    ``/api/embeddings`` wrapper) is supplied, similarity is embedding cosine;
    otherwise it falls back to a term-frequency cosine that needs only the stdlib.
    Embeddings are cached per string so a step is embedded at most once.
    """

    def __init__(self, embed: Optional[Callable[[str], Sequence[float]]] = None):
        self.embed = embed
        self._cache: dict = {}

    def _emb(self, text: str) -> Optional[Sequence[float]]:
        if self.embed is None:
            return None
        if text not in self._cache:
            try:
                self._cache[text] = list(self.embed(text))
            except Exception as e:                       # embedder failed -> degrade
                _log.warning("embed() failed (%s); falling back to text cosine", e)
                self.embed = None
                return None
        return self._cache[text]

    def sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        va, vb = self._emb(a), self._emb(b)
        if va is not None and vb is not None:
            return _vec_cosine(va, vb)
        return _tf_cosine(a, b)


# ======================================================================================
# The verifier -- RATVerifier's 3 signals, re-homed to text.
# ======================================================================================

class Verifier:
    """Three-signal step scorer: consistency, progress, novelty -> scalar in [0, 1].

    This is the text re-homing of ``rat_urm.RATVerifier``. The learned aggregator is
    cold-started as a fixed weighted blend of the three signals (train it later from
    logged ``(step, outcome)`` pairs, per the design doc). Each signal is a cheap text
    heuristic, optionally blended with an LLM-as-judge score when ``llm`` is supplied
    and ``use_judge`` is set. The judge grades all candidates of a step in ONE call.

    The gate (accept if score > tau, else backtrack) lives in the reasoner, not here --
    this class only scores, exactly like ``RATVerifier.forward``.
    """

    def __init__(
        self,
        sim: Optional[Similarity] = None,
        llm=None,
        use_judge: bool = True,
        judge_weight: float = 0.5,
        weights: Sequence[float] = (0.35, 0.40, 0.25),
        repeat_floor: float = 0.2,
        logger: Optional[logging.Logger] = None,
    ):
        self.sim = sim or Similarity()
        self.llm = llm
        self.use_judge = use_judge and llm is not None
        self.judge_weight = max(0.0, min(1.0, judge_weight))
        self.repeat_floor = max(0.0, min(1.0, repeat_floor))
        wc, wp, wn = weights
        total = (wc + wp + wn) or 1.0
        self.w = (wc / total, wp / total, wn / total)
        self.log = logger or _log

    # -- the three signals ----------------------------------------------------------

    def _signals(self, cand: str, z_steps: List[str], goal: str, prev_step: str):
        """Return (consistency, progress, novelty), each in [0, 1]."""
        # progress: how much the candidate moves toward the goal/question
        progress = self.sim.sim(cand, goal)
        if not z_steps:
            # the first step has nothing to follow from and nothing to repeat
            consistency = 0.6           # neutral prior
            novelty = 1.0
        else:
            consistency = self.sim.sim(cand, prev_step or "\n".join(z_steps))
            novelty = 1.0 - max(self.sim.sim(cand, s) for s in z_steps)
        return (max(0.0, min(1.0, consistency)),
                max(0.0, min(1.0, progress)),
                max(0.0, min(1.0, novelty)))

    def _heuristic(self, cand: str, z_steps: List[str], goal: str, prev_step: str) -> float:
        c, p, n = self._signals(cand, z_steps, goal, prev_step)
        wc, wp, wn = self.w
        raw = wc * c + wp * p + wn * n
        # Novelty gate ("novelty kills loops", design doc section 3): a near-duplicate
        # step (novelty below the floor) is damped toward zero no matter how consistent
        # or on-topic it looks -- otherwise a maxed consistency signal would let the loop
        # lock onto a restated step. This is the cold-start stand-in for the behavior the
        # learned aggregator should acquire from logged runs.
        if n < self.repeat_floor:
            raw *= (n / self.repeat_floor) if self.repeat_floor > 0 else 0.0
        return max(0.0, min(1.0, raw))

    # -- optional LLM-as-judge (one batched call for all candidates) -----------------

    def _judge(self, cands: List[str], z_text: str, goal: str) -> Optional[dict]:
        """Ask the LLM to score every candidate 0..1 in a single call. Returns a
        {index: score} map, or None if the model is unavailable or its reply doesn't
        parse -- in which case the caller falls back to the heuristic cleanly."""
        if not self.use_judge or self.llm is None:
            return None
        listing = "\n".join(f"{i}. {c}" for i, c in enumerate(cands))
        prompt = (
            f"GOAL / QUESTION:\n{goal}\n\n"
            f"REASONING SO FAR:\n{z_text or '(nothing yet)'}\n\n"
            f"CANDIDATE NEXT STEPS:\n{listing}\n\n"
            f"{TAG_JUDGE}: Score each candidate from 0.0 to 1.0 on whether it is a "
            "correct, relevant, and non-repetitive next reasoning step. Reply with ONLY "
            'a JSON list, e.g. [{"i": 0, "score": 0.8}, {"i": 1, "score": 0.3}].'
        )
        raw = _ask(self.llm, prompt, system="You are a strict reasoning-step grader. Output JSON only.",
                   temperature=0.0)
        data = _extract_json(raw)
        if not isinstance(data, list):
            return None
        out: dict = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            try:
                i = int(row.get("i", row.get("index")))
                s = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            out[i] = max(0.0, min(1.0, s))
        return out or None

    # -- public scoring API (mirrors RATVerifier.forward, batched) -------------------

    def score_batch(self, cands: List[str], z_steps: List[str], goal: str,
                    prev_step: str = "") -> List[float]:
        """Score a batch of candidate next-steps. Blends the text heuristic with the
        LLM judge (when enabled); degrades to pure heuristic if the judge fails."""
        heur = [self._heuristic(c, z_steps, goal, prev_step) for c in cands]
        if not self.use_judge:
            return heur
        judged = self._judge(cands, "\n".join(z_steps), goal)
        if not judged:
            return heur
        jw = self.judge_weight
        return [((1.0 - jw) * heur[i] + jw * judged[i]) if i in judged else heur[i]
                for i in range(len(cands))]


# ======================================================================================
# Trace objects -- observability (also what the aggregator is trained from later).
# ======================================================================================

@dataclass
class StepRecord:
    pass_idx: int
    kind: str                      # "accept" | "backtrack" | "forced-accept"
    best_score: float
    chosen: str = ""
    scores: List[float] = field(default_factory=list)
    scratchpad_len: int = 0


@dataclass
class PassRecord:
    pass_idx: int
    draft: str = ""
    converged_sim: Optional[float] = None
    done_check: Optional[bool] = None


@dataclass
class RunTrace:
    answer: str = ""
    passes: int = 0
    halt_reason: str = "max_steps"     # convergence | done | max_steps | self_consistency
    n_backtracks: int = 0
    scratchpad: List[str] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    pass_records: List[PassRecord] = field(default_factory=list)
    llm_calls: int = 0
    # self-consistency only:
    chains: List["RunTrace"] = field(default_factory=list)
    votes: dict = field(default_factory=dict)


# ======================================================================================
# The reasoner.
# ======================================================================================

class RecursiveReasoner:
    """RAT-URM control loop, in text, over an injected chat LLM.

    Parameters mirror ``DEFAULTS`` (see module top). The LLM must expose the myson
    ``LLMClient`` interface: ``ask(prompt, system=, temperature=)`` and/or
    ``chat(messages, system=, temperature=)``.
    """

    def __init__(self, llm=None, **opts):
        if llm is None:
            if LLMClient is None:
                raise RuntimeError("No LLMClient available and none injected. Pass llm=...")
            llm = LLMClient()
        self.llm = llm

        cfg = dict(DEFAULTS)
        cfg.update(opts)
        self.cfg = cfg
        self.system = cfg["system"] or MYSON_SYSTEM
        self.sim = Similarity(embed=cfg["embed"])
        self.verifier: Verifier = cfg["verifier"] or Verifier(
            sim=self.sim, llm=llm, use_judge=cfg["use_judge"],
            judge_weight=cfg["judge_weight"], weights=cfg["weights"],
            repeat_floor=cfg["repeat_floor"],
        )
        self._calls = 0

    # -- message parsing -------------------------------------------------------------

    @staticmethod
    def _parse_messages(messages):
        """Pull (system_override, goal, context_text) out of an OpenAI-style message
        list. ``goal`` is the last user turn; ``context`` is the prior conversation."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        sys_override = next((m["content"] for m in messages if m.get("role") == "system"), None)
        convo = [m for m in messages if m.get("role") != "system"]
        goal = next((m["content"] for m in reversed(convo) if m.get("role") == "user"), "")
        ctx_msgs = convo[:-1] if convo and convo[-1].get("role") == "user" else convo
        context = "\n".join(f"{m.get('role','user').upper()}: {m.get('content','')}" for m in ctx_msgs)
        return sys_override, goal, context

    # -- LLM-backed primitives -------------------------------------------------------

    def _ask(self, prompt: str, system: str, temperature: float) -> str:
        self._calls += 1
        return _ask(self.llm, prompt, system=system, temperature=temperature)

    def _render_z(self, z_steps: List[str]) -> str:
        """Render the scratchpad, truncating the OLDEST steps past the char ceiling
        (the token-budget guard from the design doc)."""
        if not z_steps:
            return "(nothing yet)"
        text = "\n".join(f"- {s}" for s in z_steps)
        cap = self.cfg["max_scratchpad_chars"]
        if len(text) > cap:
            text = "...(earlier steps elided)...\n" + text[-cap:]
        return text

    def _gen_candidates(self, goal, context, z_steps, n, temperature) -> List[str]:
        base = (
            (f"CONVERSATION SO FAR:\n{context}\n\n" if context else "")
            + f"QUESTION:\n{goal}\n\n"
            + f"REASONING SO FAR:\n{self._render_z(z_steps)}\n\n"
            + f"{TAG_NEXT}: Write the SINGLE next reasoning step that makes real progress "
              "toward the answer. One concise step -- not the final answer, not a restatement "
              "of earlier steps."
        )
        out = []
        for _ in range(n):
            out.append(self._ask(base, system=self.system, temperature=temperature).strip())
        return out

    def _redraft(self, goal, context, z_steps) -> str:
        prompt = (
            (f"CONVERSATION SO FAR:\n{context}\n\n" if context else "")
            + f"QUESTION:\n{goal}\n\n"
            + f"REASONING:\n{self._render_z(z_steps)}\n\n"
            + f"{TAG_FINAL}: Using the reasoning above, give the best final answer now."
        )
        return self._ask(prompt, system=self.system, temperature=self.cfg["redraft_temperature"]).strip()

    def _done_check(self, goal, draft) -> bool:
        prompt = (
            f"QUESTION:\n{goal}\n\nCANDIDATE ANSWER:\n{draft}\n\n"
            f"{TAG_DONE}: Is this answer complete and correct for the question? "
            "Reply with exactly one word: YES or NO."
        )
        resp = self._ask(prompt, system=self.system, temperature=0.0).strip().upper()
        return resp.startswith("YES")

    # -- one reasoning chain ---------------------------------------------------------

    def _run_once(self, messages) -> RunTrace:
        cfg = self.cfg
        sys_override, goal, context = self._parse_messages(messages)
        if sys_override:
            # a client-supplied system message wins (same rule as server.brain)
            self.system = sys_override

        z_steps: List[str] = []
        steps: List[StepRecord] = []
        pass_records: List[PassRecord] = []
        y_prev: Optional[str] = None
        n_backtracks = 0
        halt_reason = "max_steps"

        for t in range(cfg["max_steps"]):
            # ---- inner loop: expand the scratchpad by ~inner_steps accepted steps ----
            accepted = 0
            attempts = 0
            max_attempts = cfg["inner_steps"] + cfg["max_backtracks"] + 2
            just_backtracked = False
            while accepted < cfg["inner_steps"] and attempts < max_attempts:
                attempts += 1
                temp = cfg["hot_temperature"] if just_backtracked else cfg["temperature"]
                cands = self._gen_candidates(goal, context, z_steps, cfg["n_candidates"], temp)
                cands = [c for c in cands if c] or cands        # keep at least something
                prev_step = z_steps[-1] if z_steps else ""
                scores = self.verifier.score_batch(cands, z_steps, goal, prev_step)
                best_i = max(range(len(scores)), key=lambda i: scores[i])
                best_score = scores[best_i]

                can_backtrack = (best_score < cfg["tau"]
                                 and n_backtracks < cfg["max_backtracks"]
                                 and len(z_steps) > 0)
                if can_backtrack:
                    drop = min(cfg["backtrack_depth"], len(z_steps))
                    z_steps = z_steps[:-drop]
                    n_backtracks += 1
                    just_backtracked = True
                    steps.append(StepRecord(t, "backtrack", best_score,
                                            scores=scores, scratchpad_len=len(z_steps)))
                    continue
                # accept the best candidate (good enough, or forced: budget spent / z empty)
                kind = "accept" if best_score >= cfg["tau"] else "forced-accept"
                z_steps.append(cands[best_i])
                accepted += 1
                just_backtracked = False
                steps.append(StepRecord(t, kind, best_score, chosen=cands[best_i],
                                        scores=scores, scratchpad_len=len(z_steps)))

            # ---- re-draft the answer from the (updated) scratchpad ----
            y_new = self._redraft(goal, context, z_steps)
            rec = PassRecord(t, draft=y_new)

            # ---- halting: convergence, then done-check, then (implicitly) the cap ----
            if y_prev is not None:
                s = self.sim.sim(y_prev, y_new)
                rec.converged_sim = s
                if s >= cfg["conv_threshold"]:
                    halt_reason = "convergence"
                    pass_records.append(rec)
                    y_prev = y_new
                    break
            if cfg["use_done_check"]:
                done = self._done_check(goal, y_new)
                rec.done_check = done
                if done:
                    halt_reason = "done"
                    pass_records.append(rec)
                    y_prev = y_new
                    break
            pass_records.append(rec)
            y_prev = y_new

        return RunTrace(
            answer=y_prev or "",
            passes=len(pass_records),
            halt_reason=halt_reason,
            n_backtracks=n_backtracks,
            scratchpad=z_steps,
            steps=steps,
            pass_records=pass_records,
            llm_calls=self._calls,
        )

    # -- public entry: one chain, or self-consistency over several ------------------

    def run(self, messages) -> RunTrace:
        k = max(1, int(self.cfg["self_consistency"]))
        if k == 1:
            return self._run_once(messages)

        chains: List[RunTrace] = []
        for _ in range(k):
            self._calls = 0                      # count each chain's calls independently
            chains.append(self._run_once(messages))

        # majority-vote the normalized answers; tie -> the chain with fewer backtracks
        votes = Counter(_normalize_answer(c.answer) for c in chains)
        winner_norm, _ = votes.most_common(1)[0]
        representative = min(
            (c for c in chains if _normalize_answer(c.answer) == winner_norm),
            key=lambda c: c.n_backtracks,
        )
        return RunTrace(
            answer=representative.answer,
            passes=representative.passes,
            halt_reason="self_consistency",
            n_backtracks=sum(c.n_backtracks for c in chains),
            scratchpad=representative.scratchpad,
            steps=representative.steps,
            pass_records=representative.pass_records,
            llm_calls=sum(c.llm_calls for c in chains),
            chains=chains,
            votes=dict(votes),
        )


# ======================================================================================
# Public API.
# ======================================================================================

def reason(messages, llm=None, **opts) -> str:
    """Run text-space recursive reasoning and return the final answer string.

    Parameters
    ----------
    messages : an OpenAI-style ``[{"role", "content"}, ...]`` list (or a bare string,
        treated as one user turn). A system message, if present, overrides the persona.
    llm : an object with the myson ``LLMClient`` interface (``ask``/``chat``). If None,
        a default ``LLMClient`` is constructed (talks to local Ollama).
    **opts : any key in ``DEFAULTS`` -- e.g. ``max_steps``, ``n_candidates``, ``tau``,
        ``self_consistency``, ``use_judge``, ``embed=<text->vector>``, or a pre-built
        ``verifier=``.

    Returns
    -------
    str : the final answer draft (the majority answer when ``self_consistency > 1``).
    """
    return RecursiveReasoner(llm=llm, **opts).run(messages).answer


def reason_trace(messages, llm=None, **opts) -> RunTrace:
    """Like :func:`reason`, but returns the full :class:`RunTrace` (scratchpad, per-step
    scores, backtracks, halt reason, vote tally) -- for logging, debugging, and training
    the verifier's aggregator from real runs."""
    return RecursiveReasoner(llm=llm, **opts).run(messages)


def make_reasoner(llm=None, **opts) -> RecursiveReasoner:
    """Build a reusable reasoner (amortizes the embedding cache across calls)."""
    return RecursiveReasoner(llm=llm, **opts)


# ======================================================================================
# helpers
# ======================================================================================

def _ask(client, prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Single-turn call against the shared LLMClient interface, tolerant of clients that
    expose only ``chat`` or whose ``ask`` doesn't accept ``temperature``."""
    if client is None:
        return ""
    if hasattr(client, "ask"):
        try:
            return client.ask(prompt, system=system, temperature=temperature) or ""
        except TypeError:
            return client.ask(prompt, system=system) or ""
    # chat-only client
    msg = client.chat([{"role": "user", "content": prompt}], system=system,
                      temperature=temperature)
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return str(msg or "")


def _extract_json(text: str):
    """Best-effort JSON extraction from an LLM reply (strips code fences, finds the first
    balanced object/array). Returns a dict/list, or None on failure."""
    if not text:
        return None
    import json
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = s.find(open_c)
        end = s.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                continue
    return None


def _normalize_answer(text: str) -> str:
    """Normalize a free-text answer for self-consistency voting: if the model wrote an
    explicit 'answer: X' (or '=> X') marker, vote on X; otherwise vote on the last
    non-empty line, lowercased with punctuation/whitespace collapsed."""
    if not text:
        return ""
    t = text.strip()
    m = None
    for pat in (r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)$",
                r"=>\s*(.+)$", r"\\boxed\{([^}]*)\}"):
        found = list(re.finditer(pat, t, flags=re.IGNORECASE | re.MULTILINE))
        if found:
            m = found[-1].group(1)
            break
    if m is None:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        m = lines[-1] if lines else t
    m = m.lower().strip()
    m = re.sub(r"[\s]+", " ", m)
    m = m.strip(" .!?\"'`*")
    return m


__all__ = [
    "reason", "reason_trace", "make_reasoner",
    "RecursiveReasoner", "Verifier", "Similarity", "RunTrace",
    "DEFAULTS", "CAVEAT",
    "TAG_NEXT", "TAG_FINAL", "TAG_DONE", "TAG_JUDGE",
]
