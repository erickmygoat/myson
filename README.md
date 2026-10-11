# myson

A local-first AI assistant: an open LLM (Qwen3.5-9B via Ollama) wrapped with a recursive-reasoning brain, memory, tools, and safe self-improvement — all your own code.

- **Mind:** open weights (Qwen3.5-9B), runs locally.
- **Brain (this repo):** recursive reasoning (RATURM), 3-tier memory, tool-use (code/math/web), SEAL self-improvement.
- **Face:** its own local workspace UI.

Wake it: `ollama pull qwen3.5:9b` then `python -m myson.cli`. See `docs/BRAIN_ARCHITECTURE.md`.

## Boundaries (enforced in code)
No safety-floor removal, no unconditional-obedience core, no ungated self-rewrite. Candid, calibrated, loyal-with-judgment.

Inspired by PewDiePie's Ajax/Odysseus + the open TRM/HRM/SEAL research.
