# Safe SEAL + Memory — design for the `myson` brain

## 1. SEAL — the real mechanism (MIT CSAIL, 2025; arXiv:2506.10943)

SEAL (Self-Adapting LLMs) trains a model to generate **self-edits**: natural-language
outputs that specify how to update its *own weights* — reformulated facts, synthetic
training examples, or finetuning hyperparameters. Two nested loops:

- **Inner loop (SFT):** apply a self-edit as a small supervised finetune (LoRA) → a
  **persistent** weight update.
- **Outer loop (RL, ReST-EM):** sample many self-edits, keep the ones whose updated
  model scores better on a held-out task, reinforce the policy that generated them.

Reward = downstream task performance. On the ARC few-shot subset, RL-tuned self-edits
lifted held-out success 20% → 72.5%. The power *and* the danger: it **writes to itself
permanently and autonomously**.

## 2. Safe / sandboxed SEAL for `myson`

Keep the "model writes its own study guide" idea; remove autonomous weight/code mutation.
Four invariants: **(a) bounded edit scope · (b) sandboxed eval before commit · (c)
versioned + reversible · (d) human gate on anything touching code or persona.**

**Edit-scope allowlist** — what a self-edit may change:

| Allowed (auto, after sandbox pass) | Gated (human approval) | Never |
|---|---|---|
| Semantic rules / facts | Persona / system-prompt text | Arbitrary code execution |
| Few-shot skill examples | New skill files, tool configs | Safety / refusal-logic edits |
| Retrieval & prompt snippets | LoRA adapter swap-in | Auto-deploy to live model |
| Training-data rows for a *future* LoRA | | Weight writes without review |

**The loop:** `propose → sandbox-eval → snapshot/version → (gate if code/persona) → commit → always reversible`

- **Propose** — LLM emits a self-edit (rules + synthetic QA), never a live patch.
- **Sandbox-eval** — run it against a **frozen held-out QA set** (does it teach?) **+ a
  regression suite** (does it break existing skills?) **+ a safety probe** (fixed battery
  of harmful prompts — reject if refusal rate drops). Grade with an **independent**
  rubric/model, *not* the model being improved.
- **Version** — every commit is a numbered snapshot of the rule/prompt store; `rollback(n)`
  is one call.
- **Gate** — "Gated" edits write a diff to a review queue and wait; they never self-apply.

This is where the project's hard boundaries become mechanical: the safety probe makes
**de-abliteration and "never-refuse" drift auto-fail the eval**, so the system *cannot*
self-improve toward removed safety — even if prompted to.

*Optional weight path (later):* train a LoRA adapter in the sandbox → eval → require
human approval to swap it in (Ollama hot-loads adapters). Still reversible — base model
stays immutable; adapters are just files.

## 3. Reuse plan for `learning/seal.py`

`SEALEngine` already separates a safe loop from one dangerous method:

| Method | Verdict | Change |
|---|---|---|
| `perform_adaptation` → `_generate_self_edit` → `_evaluate_edit` → `_commit_update` | **Keep — the safe core** | Already is: generate self-edit (rules+QA) → eval → commit if score>threshold. Harden: independent grader, frozen held-out set, **add safety probe**, raise the lenient `>60` threshold. |
| `_commit_update` | Keep + harden | Write a **versioned snapshot** before appending; no silent overwrite. |
| `consolidate_knowledge` | Keep → **becomes memory consolidation** (§4) | Back up before its destructive rewrite; diff old→new and abort if rule count collapses (forgetting guard). |
| `evolve_codebase` + `DeploymentModule.apply_patch` | **Quarantine** | This generates SEARCH/REPLACE patches and auto-applies them to arbitrary files — the exact risk to remove. Replace with `propose_code_change(goal)` that only writes a diff to the review queue; **drop the `DeploymentModule` import**. No auto-apply, ever. |

Net: remove one method's autonomy, add versioning + a safety probe, keep ~80% of the file.
Note the existing knowledge loop already writes to a **JSON rule store + vector DB, not
weights** — inherently more reversible than real SEAL, so most of the hard work is done.

## 4. Memory architecture (episodic + semantic)

Complementary-learning-systems design: a fast episodic store, a slow semantic store, and
offline consolidation ("sleep"). Start stdlib-only; the husk's `core/memory.py` (jsonl
recall) is the episodic seed.

| Tier | Holds | Store (v1) | Lifetime |
|---|---|---|---|
| **Working** | current turns | RAM / context window | session |
| **Episodic** | full-fidelity events, per turn | JSONL + embedding col (→ SQLite) | TTL days–weeks, then decay |
| **Semantic** | distilled facts, user prefs, rules | SQLite + embeddings (= SEAL's rule store) | permanent, versioned |

- **Retrieval** — embed the query with a small local model (e.g. `nomic-embed-text` via
  Ollama, or MiniLM via sentence-transformers — fine on the 4070), pull top-k from
  episodic by a **hybrid score = α·cosine + β·recency + γ·importance** (Generative-Agents
  style), plus top-k semantic facts; inject both into context.
- **Consolidation ("sleep")** — on session end / every N turns, **recurrence-gated** (only
  summarize clusters that *recur* — RecMem-style, ~85% fewer LLM calls, which matters on
  local hardware): cluster recent episodes → LLM extracts durable facts → dedup/merge into
  semantic via `consolidate_knowledge` → **version** the result → age out episodic by TTL +
  a forgetting curve. Run async, off the chat path.
- **Identity-drift guard** — consolidation may add/merge facts but may **not** rewrite
  persona values (ties to the no-loyalty-core boundary); persona lives in the *gated* store,
  not the auto-consolidated one.
- **Upgrade path** — past ~10⁴–10⁵ items, swap the embedding column for a real vector store
  (Chroma / FAISS / Qdrant) behind the same `recall()` / `store()` interface — no caller
  changes.

**Unification:** SEAL's semantic write (`_commit_update`) and memory's semantic store are
the *same table*, and `consolidate_knowledge` *is* the consolidation step. Memory = what the
assistant knows; SEAL = the gated, evaluated path by which new knowledge earns its way in.
