# myson — HANDOFF (start here)

## Read first

1. [docs/ROADMAP.md](docs/ROADMAP.md) — the plan, the order of work, and what is and is not built.
2. [SETUP.md](SETUP.md) — how to install, test and run.
3. [bench/MODEL_REVIEW.md](bench/MODEL_REVIEW.md) — why the model is Gemma 4 26B-A4B and what is unproven.

The owner has said more than once: read the plans and the earlier chats before asking or
deciding. The roadmap exists so that is quick.

## What myson is

A local assistant = an open model (Gemma 4 26B-A4B, a 4-bit GGUF run through llama-cpp-python)
plus a "brain" wrapper (router, recursive reasoning, tools, memory, gated SEAL self-improvement,
persona), behind an OpenAI-compatible server with its own workspace UI. The "new model" is the
open weights and the wrapper **as one system**; nothing is merged into the weights.

## State (2026-11-13)

- **Built and unit-tested against a mocked model:** router, recursion, tool loop and sandbox,
  long-context reduction, 3-tier memory, SEAL review queue and CLI, server with streaming,
  workspace UI, own-model compression track.
- **Run for real on the 4070:** the GGUF backend (loading, automatic GPU/RAM split, chat,
  streaming, strict chat templates, concurrent requests), against two smaller local models.
- **Not done:** the 26B has not been loaded on this machine; the wrapper has never run end to
  end on a real model; no recursion on-versus-off measurement exists.

## Next, in the owner's order

1. Model on disk (`models/`), then **the owner plays with it by itself**: `python bench/play_gemma.py`.
2. First end-to-end run of the wrapper on the real model.
3. The rest of the list in `docs/ROADMAP.md`: recursion A/B, wall-clock cap, sandbox on the
   default path, thinking-mode mapping, persona calibration, fine-tuning.

## Things that bit us

- "Unit-tested against a mock" is not "works". Two backends in a row shipped that way and both
  had real faults the mocks could not show. Run it on a real model before describing it as done.
- The installed llama-cpp-python (0.3.36) has no expert-offload option; do not assume a
  llama.cpp CLI flag exists in the Python binding. Check the installed signature.
- A bare `import llama_cpp` fails on this Windows setup until torch's CUDA DLL folder is on the
  search path; the backend handles it, ad-hoc scripts must too.

## Boundaries (enforced in code — keep them)

No abliteration / safety-floor removal. No unconditional-obedience / never-refuse core. No
ungated self-rewrite or auto-deploy. `seal_safe.py`'s probe auto-fails de-abliteration drift.
