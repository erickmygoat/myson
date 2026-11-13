# myson roadmap

Reconstructed 2026-11-13 from the project's own history — the original October planning
session, the two approved build plans, both local build chats — and from a review of the
precursor `my_son` research tree. Status is as of that date. `FUTURE.md` holds the longer-term
notes; `bench/` holds the model-selection evidence.

## The goal

A private assistant that runs entirely on one laptop (RTX 4070, 8 GB VRAM, 32 GB RAM), offline,
owned and editable by its owner. The open model is the "mind stone"; **the wrapper is the
point**: recursive reasoning, tools, memory, gated self-improvement and a loyal, candid persona
around that model. A bare model, however good, was never the goal.

## "Recursive Gemma": what that means here

Gemma 4 26B-A4B, running locally as a GGUF, driven by myson's text-space reasoning loop instead
of answering in one pass:

1. the **router** decides whether a request needs plain chat, tools, or recursion;
2. **recursion** samples candidate next steps, scores each with a three-signal verifier
   (consistency, progress, novelty), backtracks when a step scores badly, redrafts, and stops
   when the answer stops changing — more passes for hard questions, few for easy ones;
3. the **tool loop** runs the code the model writes rather than trusting the output it
   imagines (the model benchmarks showed every model size getting byte-level arithmetic wrong
   in its head);
4. **memory** carries context across sessions, and **SEAL** proposes improvements that wait for
   human approval.

All four exist in this repo and are unit-tested against a mocked model. **None has run against
the real Gemma yet.** That gap is the main thing this roadmap is about.

## Order of work

The owner set this order and it has not changed since October:

1. **Get the model onto the machine.** Done 2026-11-13: the 14.2 GB QAT build is in `models/`
   and runs on the 4070 at 21–25 tokens per second.
2. **Play with it by itself** (`python bench/play_gemma.py`) before anything else is built on
   it. **This is where things stand.**
3. **Then the model work**, in this order:
   1. first real end-to-end run of the wrapper on the model (server up, a real answer back);
   2. recursion on versus off, same model, same questions, real numbers;
   3. the execution sandbox on the default path, with real isolation;
   4. Gemma's thinking mode mapped onto the router, and its reasoning trace kept out of answers;
   5. persona calibration measured on a false-refusal set;
   6. fine-tuning to make it the owner's own.

## Tracks and status

| Track | Planned | Status |
|---|---|---|
| Model on the machine | open weights, on disk, offline | **done** — in `models/`; loads from disk with no network |
| Runtime | load a model bigger than VRAM | **built and measured** — GGUF backend splits layers between GPU and RAM automatically: 11–12 of the 26B's 30 layers on the GPU, 21–25 tokens per second |
| Play with it alone | standalone chat with the persona | **ready** — `bench/play_gemma.py`; the owner's turn |
| Router | auto-pick chat / tools / recursion, `/think` `/fast` `/tools` overrides | **built**, mock-tested. The "hard enough for recursion" threshold was reserved for the owner to tune and still has its default |
| Recursive reasoning | sample → verify → backtrack → redraft, adaptive depth | **built**, mock-tested (`myson/reasoning/recursive.py`). Never run on the real model |
| Latency caps | step cap plus a wall-clock budget | **half done** — the step cap is wired; `RECURSION_WALL_SECONDS` is defined but nothing enforces it. Needed before recursion runs on a slow local model |
| Tools and sandbox | run code, exact math, web; isolated execution | **built**, mock-tested. Not on the default path, and the isolation hardening (no inherited environment, no network, time and memory caps) is not done. **Top wrapper task** |
| Long context | map-reduce documents bigger than the window | **built** (`reasoning/longctx.py`) |
| Memory | working / episodic / semantic, recall across sessions | **built**. With the GGUF backend there is no embedding model, so recall uses the text-cosine fallback. The planned vector-store upgrade was not done |
| SEAL self-improvement | propose → quarantine → safety probe → human review → apply / rollback | **built** as a gated review queue with CLI. Not built: the "keep an edit only if it beats its parent" fitness archive from the old tree |
| Persona | loyal, candid, not a yes-man; answers ordinary edge cases | prompt **written**. The false-refusal evaluation (before / after) was never run |
| Autonomy / inner life | standing goals, reflection written to memory, initiative, opinions | **not started**. Design stance: build the functional version; self-modification and irreversible actions stay human-gated |
| Workspace UI | chat, live reasoning scratchpad, memory browser, SEAL queue | focused version **built**; the fuller multi-conversation workspace is deferred until real use |
| Benchmarks | pick the model; measure recursion on vs off | model selection **done** on cloud hardware. The recursion A/B on the real model — the claim that matters most — **not done** |
| Own-model compression | BitNet / MLA / MoR on the repo's own transformer | **built** and measured (`myson/compression`). It applies to weights we train, not to a downloaded Gemma |
| Fine-tuning | QLoRA → merge → GGUF | **not started**. Open question: target model and where it trains (the 26B cannot be tuned on 8 GB) |
| Model merging | same-family merge into one GGUF | **not started** |
| Subagents | coordinator and workers | basic `agents/` present; the fuller layer from the old tree was not ported |
| Docs and git | truthful setup docs, history pushed | this pass |

## Lessons carried over from the precursor project

The old tree's own research logs are blunt about what went wrong, and they apply here:

- **Never report a number the model did not produce.** The old "100%" benchmarks ran in 0.0
  seconds from hardcoded passes. Every myson number comes from a real run, with timing.
- **Measure the claim you care about.** Recursion needs a plain-versus-recursive A/B on held-out
  problems before anyone says it helps.
- **Run the thing before describing it.** The old autonomous loop was documented as a feature
  and never executed once. In this repo the same trap is "unit-tested against a mock" being
  read as "works".
- **Do not train a general model from scratch on one 4070.** Put the ideas in the wrapper, or in
  a fine-tune of a pretrained model.
- **One component at a time,** kept only if a measurement moves.
- **No silent failure** in the reasoning, tool or self-improvement paths.
- **Self-improvement stays bounded, sandboxed, reversible and human-gated.** The old auto-patching
  loop deployed failing patches until it was forced to wait.

## Boundaries that stand

No abliteration or other removal of the model's safety floor; no unconditional-obedience core;
no ungated self-rewrite or auto-deploy. Security work is defensive, authorized-testing and CTF.
Nothing from the old tree's offensive tooling or its insecure control plane enters this repo.
The not-over-refusing behaviour comes from persona calibration, which is why that evaluation is
on the list above.

## Open questions for the owner

- Is voice input part of the JARVIS experience? The 26B has none; the 12B does.
- Fine-tuning: which model, and trained where?
- Is model merging still wanted, and between which two same-family models?
- The router threshold reserved for you: tune it once recursion runs on the real model?

## Not yet reviewed

Four surveys of the precursor code (the reasoning core, agents / memory / self-improvement, the
own-model track, and miscellaneous scripts) were cut short and are still to do. The reuse map in
the October build plan covers them at a high level.
