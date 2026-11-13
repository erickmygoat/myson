> **Historical design document (October 2026).** Written when the model was Qwen3.5-9B served
> by Ollama. The model is now Gemma 4 26B-A4B through the GGUF backend; the design stands.
> Current plan and status: [ROADMAP.md](ROADMAP.md).

# Recursive Reasoning Core — design, LLM-loop, verifier gate, halting, honest limits

**Base:** Qwen3.5-9B via Ollama — text in / text out; weights and hidden states are inaccessible.
**Thesis:** The recursion runs in **text space**, not latent space. We reuse the RATURM *control algorithm* (`rat_urm.py`) and the 3-signal verifier; we drop the latent step-generator and latent refinement, because there are no hidden states to iterate.

## 1. The constraint that decides everything
TRM, HRM, and Huginn all recurse in **latent space** on their own hidden states (TRM refines tensors `z`,`y`; Huginn re-runs a transformer block N times). A black-box API emits only tokens, so `RATVerifier`, `RATURMStepGenerator`, and `RecursiveRefinement` **cannot be plugged into Qwen as written** — they expect `hidden_states (B,T,D)` we never get.

This is convenient: latent recursion *alone* underperforms explicit chain-of-thought. On GSM8K, raising Huginn's recurrence 4→32 gives only marginal gains and plateaus ("guess-freezing," not computation completing). The black-box constraint therefore pushes us onto the method that empirically wins — **explicit text CoT + a verifier gate + best-of-N** — while keeping TRM's *structure*.

Map TRM's state to text: **`y` = current answer draft**, **`z` = reasoning scratchpad (running CoT)**. The loop refines `z` a few times, then re-drafts `y` — TRM's inner/outer (`n`,`T`) schedule, in words.

## 2. The loop (per query)
```
y, z = None, ""                         # answer draft, scratchpad (both text)
goal  = embed(question)
for t in 1..T_max(=16):                 # TRM Nsup=16; our max_steps=16
    for i in 1..n:                      # expand reasoning (TRM n≈6 inner)
        cand[1..N] = Qwen(prompt(question, z, "next step"), temp=0.7)   # N diverse candidates
        s = [verifier(c, prev=z, goal) for c in cand]
        best = argmax(s)
        if s[best] < tau: z = backtrack(z)        # drop ≤3 steps, resample hotter
        else:             z += best
    y_new = Qwen(prompt(question, z, "final answer"), temp=0)           # re-draft answer
    if halt(y, y_new, s): return y_new            # §4
    y = y_new
return y
```
- **Candidate generation = temperature sampling** — the text analog of `RATURMStepGenerator`'s learned noise injection. N candidates per step → best-of-N; sample full CoTs instead and majority-vote the answer → self-consistency (same machinery, different granularity).
- **Beam (optional):** keep top-k scratchpads instead of top-1 (the `BeamManager` noted in the docstring) — more robust, k× cost.

## 3. Verifier gate — reuse `RATVerifier`, re-homed to text
`RATVerifier`'s three signals map directly; inputs become **embeddings** (Ollama `/api/embeddings`) instead of backbone latents (re-instantiate at the embedder's dim):
- **Consistency** — does the step follow from `z`? (entailment via embedding or LLM-judge)
- **Progress** — closer to the goal/answer? (PRM-style step score; embedding-to-goal or judge)
- **Novelty** — new information vs. restating? (embedding distance from `z`; kills loops)

Learned aggregator → scalar in [0,1]; **gate: accept if score > `tau` (code default 0.3), else backtrack** via `BacktrackController` (on a run of low scores, step back `d` and resample; max 3 backtracks, as coded).
**Cold-start honestly:** before the aggregator is trained, run the 3 signals as LLM-judge + embedding heuristics; log `(step, outcome)` from real runs, then train the aggregator (later the signal nets). PRM lesson: per-step scoring beats outcome-only.

## 4. Halting — three checks, cap always wins
- **Convergence** (Huginn / `RecursiveRefinement` idea): stop when the draft stops moving — `sim(y, y_new) > 0.98`. This is `RecursiveRefinement`'s `diff < threshold`, lifted to text.
- **Confidence** (TRM's ACT "am I done?"): a learned halt head over the score history (reuse `BacktrackController`'s head shape) + an explicit LLM self-check "complete and correct?".
- **Budget cap:** hard `T_max=16` and a token ceiling — guarantees termination. Easy inputs halt in 1–2 passes; hard inputs spend the budget. That *is* the test-time-compute trade.

## 5. Honest limits
**What recursion-over-9B CAN buy**
- Mitigates the "one wrong token invalidates the answer" failure (TRM's own motivation): verify + backtrack catches arithmetic/logic slips mid-chain.
- Best-of-N and self-consistency are reliable, well-established test-time-compute gains on multi-step tasks.
- A persistent scratchpad keeps long plans from drifting; compute is spent only on hard inputs.

**What it CANNOT buy**
- **No new knowledge.** Recursion amplifies what the 9B already knows; it cannot inject facts or skills the base lacks. A 9B will not become a frontier model.
- **Verifier-capped.** Gains are bounded by verifier quality, and a 9B judging 9B output has *correlated* errors — a confidently-wrong verifier entrenches mistakes. The verifier is the bottleneck; invest there, not in more loops.
- **Diminishing returns at real cost.** Gains plateau (Huginn 4→32 marginal); every pass is N×–k× more tokens and latency. This trades latency for accuracy — nothing is free.
- **Not latent reasoning.** We get gated text CoT, not TRM/Huginn latent depth. Name it honestly.

## 6. Reuse vs. build
| Piece | Decision |
|---|---|
| RATURM loop (`_run_reasoning_path`: generate→verify→backtrack→halt) | **Reuse** as the control algorithm, re-homed to text |
| `RATVerifier` (3 signals + learned aggregator) | **Reuse** architecture; feed embeddings; train aggregator from run logs |
| `BacktrackController` (halt/backtrack head over score history) | **Reuse** |
| `RecursiveRefinement` adaptive halt + TRM `(y,z)` split & `n`/`T` schedule | **Reuse the idea** (halting + state split); implement in text |
| `RATURMStepGenerator` latent noise injection | **Replace** with temperature sampling |
| `MySonModel` 400M backbone / latent `y`,`z` tensors | **Do not use** — we wrap Qwen, not a from-scratch backbone |
| text↔embedding bridge, candidate sampler, prompt templates, LLM-judge cold-start, self-consistency vote | **Build fresh** |

*Refs: TRM arXiv:2510.04871; HRM; Huginn/recurrent-depth arXiv:2502.05171; self-consistency (Wang 2022); PRMs (Lightman 2023); test-time compute (Snell 2024).*
