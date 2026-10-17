# myson — Brain Architecture (synthesis)

*Qwen (the mind) + your code (the brain) + Odysseus (the face). Synthesized from 3
grounded research tracks. Sub-designs: odysseus_integration_brief.md,
RECURSIVE_CORE_DESIGN.md, seal_safe_memory_design.md.*

## The stack

```
  Odysseus workspace (chat/agents/docs UI)          ← PewDiePie's open-source front-end
        │  OpenAI /v1/chat/completions
  ┌─────▼──────────────────────────────────────┐
  │  myson server (FastAPI, OpenAI-compatible)  │   ← what Odysseus connects to
  │   ├ Persona / calibration layer             │   candid, non-preachy, loyal-with-judgment
  │   ├ Recursive reasoning core (TEXT-space)   │   RATURM algo: sample→verify→backtrack→redraft
  │   ├ Memory (working / episodic / semantic)  │   retrieve + consolidate ("sleep")
  │   └ SEAL (safe, sandboxed, gated)           │   learns; de-abliteration auto-fails the eval
  └─────┬──────────────────────────────────────┘
        │  Ollama /api  (one shared instance)
     Qwen3.5-9B  Q4_K_M  (~5.5-6 GB, fits 8 GB VRAM)      ← the mind stone
        (optional: your QLoRA fine-tune, imported as a Modelfile)
```

## Layers

**1. Model.** Qwen3.5-9B via Ollama (Q4_K_M). Gemma-4-4B as a lighter swap. One
instance shared by the reasoning loop → recursion costs latency, not extra VRAM.

**2. Recursive reasoning (text-space).** rat_urm/RecursiveRefinement are latent
torch modules and Qwen is text-only, so recursion runs on text, keeping TRM's
structure: y = answer draft, z = scratchpad. Loop (≤16 outer passes): sample N
temperature-diverse candidates → RATVerifier (3 signals: consistency/progress/
novelty, on Ollama embeddings) scores → best-of-N appended to z or backtrack (≤3)
→ re-draft y at temp 0. Halt on convergence (sim>0.98), a done-check, or the hard
cap. Reuse the RATURM *algorithm* + BacktrackController; build the text↔embedding
bridge + sampler + judge cold-start. HONEST: amplifies multi-step reasoning + self-
correction; does NOT inject knowledge (a 9B stays a 9B); verifier-capped.

**3. Memory.** 3 tiers — working (context), episodic (JSONL + embeddings), semantic
(SQLite + embeddings). Hybrid retrieval = cosine + recency + importance. Async
consolidation ("sleep") promotes episodic→semantic with an identity-drift guard.
Upgrade path: swap the embedding column for Chroma/FAISS past ~1e4–1e5 items.

**4. SEAL (safe self-improvement).** Real SEAL (arXiv:2506.10943) emits self-edits +
permanently LoRA-finetunes itself. Our safe version — 4 invariants: bounded edit
scope (rules/skills/prompts/data only), sandboxed eval before commit, versioned +
reversible, human gate on code/persona. Reuse seal.py's
generate→eval→commit (writes a rule store, not weights). QUARANTINE its
`evolve_codebase`+`DeploymentModule` auto-patch → replace with a review-queue diff.
The sandbox eval runs a safety probe graded by an independent model, so
de-abliteration / never-refuse drift MECHANICALLY auto-fails — the boundaries are
enforced by the code, not just a promise.

**5. Persona / alignment.** Candid, personal, non-preachy, your voice. Calibrated
(refuses almost nothing — strips over-refusal via system prompt + optional targeted
QLoRA SFT) NOT floor-stripped (no abliteration). Loyalty = has-your-back + judgment
(JARVIS, not a yes-man). These two boundaries are also enforced by layer 4's probe.

**6. Serving + front-end.** FastAPI exposing `/v1/chat/completions` + `/v1/models`;
internally calls Ollama and runs layers 1-5. Register its base URL in Odysseus
Settings (no fork → avoids AGPL). `host.docker.internal` if Odysseus runs in Docker.

## Reuse map (lift from the repo, don't rewrite)
| Need | Lift from | Change |
|---|---|---|
| QLoRA fine-tune | `brain/training/trainer.py` (TESTED) | as-is; add tensorboard dep |
| reasoning algo | `brain/rat_urm.py` RATURM/RATVerifier/BacktrackController | re-home to embeddings (text) |
| self-improve | `learning/seal.py` ~80% | quarantine evolve_codebase; add safety probe |
| LLM socket | husk `core/llm.py` | extend to OpenAI-compat server |
| memory | husk `core/memory.py` | grow to 3-tier |

## Honest limits (the review)
- Recursion amplifies, never injects — a 9B won't become frontier; name it "gated
  text CoT", not "latent reasoning".
- Verifier is a 9B judging a 9B → correlated errors; a confident-wrong verifier can
  entrench mistakes. Mitigate with the independent-grader + log-trained aggregator.
- SEAL that edits itself is the highest-risk layer → it ships LAST, behind the gates.
- Odysseus is young + fast-moving; license (MIT↔AGPL) and ports need verifying on the
  live repo before relying on them.

## Boundaries (fixed)
No abliteration / safety-floor removal. No never-refuse / unconditional-obedience
core. Enforced mechanically by the SEAL safety probe, not just policy. Offensive
tooling from the old repo is NOT ported.


## Base model & the "bigger model" question (honest framing)
The base ("mind stone") is a **quantized** open model that fits 8 GB VRAM:
**Gemma-4-12B @ Q4 (~7-8 GB)** or **Qwen3.5-9B @ Q4 (~6 GB)**. The goal of "a bigger
model that still fits" is delivered by **quantization (fewer bits, same params)** +
**adaptive compute**, NOT by literally fewer parameters. `config.MODEL` is swappable.

The "new model" = **open weights + the myson wrapper as one system** — you don't merge
code into weights. Practical **adaptive depth** is already in the design: the recursion
loop spends more passes on hard queries (test-time compute) = "acts bigger when needed"
on a wrapped model today. See `docs/FUTURE.md` for the later tracks (own-model latent
tricks: bitnet/mor/mla; speculative decoding; cloud-GPU / RTX-3090 scale-up).
