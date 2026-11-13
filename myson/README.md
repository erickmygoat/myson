# myson (the package)

The brain. For what the project is and how to run it, see the [README](../README.md),
[SETUP](../SETUP.md) and [roadmap](../docs/ROADMAP.md) at the repo root.

| Module | What it holds |
|---|---|
| `config.py` | every knob: backend, model, placement, context, persona |
| `core/` | model backends behind one interface: `gguf_llm.py` (default), `local_llm.py` (transformers), `llm.py` (Ollama); `backend.py` picks one |
| `brain.py` | the one seam: recall → persona → router → chat / tools / recursion → save |
| `router.py` | decides which path a request takes; `/think`, `/fast`, `/tools` override it |
| `reasoning/` | recursive reasoning (`recursive.py`), long-context reduction (`longctx.py`) |
| `tools/` | the tool loop, code sandbox, exact math, web |
| `memory/` | 3-tier memory with hybrid recall |
| `learning/` | SEAL self-improvement behind a human review gate (`seal_safe.py`) |
| `agents/` | coordinator and worker personas |
| `compression/`, `training/` | the own-model track: BitNet / MLA / MoR and the trainer |
| `server.py`, `cli.py`, `ui/` | OpenAI-compatible API, terminal chat and SEAL queue, workspace UI |

Nothing from the precursor project's offensive tooling is ported in; see `PORT_MANIFEST.md`.
