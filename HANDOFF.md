# myson — HANDOFF (continue here, ideally in Claude Code on your own machine)

## What myson is
A local AI assistant = an open LLM (Qwen3.5-9B or Gemma-4-12B @ Q4 via Ollama) + a
"brain" wrapper (recursion + tools + memory + safe SEAL self-improvement), exposed as an
OpenAI-compatible server, with its own workspace UI (to be built). The "new model" is the
quantized open weights + this wrapper **as one system** — you don't merge code into weights.

## DONE (built + unit-tested here with a MOCKED Ollama)
- Scaffold/husk: `myson/core/llm.py` (Ollama client), `config.py`, `cli.py`, `server.py` (OpenAI `/v1/chat/completions`, localhost-only).
- Lifted & verified: `myson/training/` (model.py, trainer.py = QLoRA), `myson/reasoning/rat_urm.py`.
- Phase-2 brain modules:
  - `reasoning/recursive.py` — text-space recursive reasoning (sample -> verify -> backtrack -> redraft). tests green.
  - `tools/` (safe_tools, sandbox, mathtool, web, agent_tools) — sandboxed code-exec + math + web + function-calling loop. tests green.
  - `memory/memory.py` — 3-tier memory (working/episodic/semantic) + hybrid retrieval + consolidation. tests green.
  - `learning/seal_safe.py` — sandboxed SEAL: quarantined evolve_codebase, independent-grader safety probe, review queue, versioned rollback. 5/5 tests.
- Docs: `docs/BRAIN_ARCHITECTURE.md`, `docs/RECURSIVE_CORE_DESIGN.md`, `docs/seal_safe_memory_design.md`, `docs/odysseus_integration_brief.md`, `docs/FUTURE.md`, `PORT_MANIFEST.md`.

## TODO (needs Ollama + your GPU — do it in Claude Code locally)
1. `ollama pull qwen3.5:9b` (or `gemma4:12b`).
2. Wire the layers into `server.py` `brain()`: persona -> recursion -> tools -> memory -> SEAL.
3. Run: `python -m myson.server`, then POST to http://127.0.0.1:8080/v1/chat/completions.
4. Persona + calibration: system prompt (candid, non-preachy, loyal-with-judgment), kill over-refusal; optional QLoRA SFT via `training/trainer.py`.
5. Build the workspace UI (talks to the endpoint).
6. End-to-end + honest coding/math/reasoning benchmark + README.

## BOUNDARIES (enforced in code — keep them)
No abliteration/safety-floor removal. No unconditional-obedience/never-refuse core. No
ungated self-rewrite or auto-deploy. `seal_safe.py`'s probe auto-fails de-abliteration drift.

## Plan
See the approved build plan JSON in the project. Phase 1 done; Phase 2 modules done; Phases 3-4 remain (local).
