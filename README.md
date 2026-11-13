# myson

A local-first AI assistant that runs **on your own machine**: an open model wrapped in a
recursive-reasoning brain with tools, memory, and gated self-improvement. The model is the
"mind stone"; the wrapper is the point.

- **Mind:** an open-weight model run locally as a GGUF through llama.cpp. Default: Gemma 4
  26B-A4B (4-bit), split automatically between an 8 GB GPU and system RAM.
- **Brain (this repo):** a router that picks plain chat, tools or recursion per request;
  recursive reasoning (sample, verify, backtrack, redraft); a tool loop (code, math, web);
  3-tier memory; SEAL self-improvement behind a human review gate; a candid, loyal persona.
- **Face:** a local workspace UI (chat, live reasoning scratchpad, memory, SEAL queue) and an
  OpenAI-compatible API, both bound to 127.0.0.1.
- **Own-model track:** BitNet / MLA / MoR compression on the repo's own small transformer.

## Status

The brain is unit-tested against a mocked model (132 tests on Windows + Python 3.13). The GGUF
backend has been run for real on an RTX 4070 laptop with smaller models. The full wrapper has
not yet been run end to end on the 26B. [docs/ROADMAP.md](docs/ROADMAP.md) says exactly what
is built, what is not, and in what order the rest happens.

## Quickstart

See **[SETUP.md](SETUP.md)**. Short version, once the dependencies are in:

```powershell
python bench/play_gemma.py      # chat with the model by itself (first run downloads 14.2 GB)
python -m myson.server          # the full assistant on http://127.0.0.1:8080
```

## Architecture

```
  you ─► workspace UI / cli ─► /v1/chat/completions
                                     │
                           Brain: recall ─► persona ─► router ─► {chat | tools | recursion} ─► save
                                     │
                     backend (make_llm) ─► GGUFLLM (llama.cpp, in-process)   ← the mind stone
                                     │
           tools (code/math/web)   memory (3-tier)   SEAL (gated self-improvement)
```

`docs/BRAIN_ARCHITECTURE.md` and `docs/RECURSIVE_CORE_DESIGN.md` are the original design
documents (written when the model was served by Ollama; the design is unchanged, the backend is
not). `bench/` holds the model-selection evidence and harnesses.

## Boundaries (enforced in code)

No safety-floor removal / abliteration, no unconditional-obedience core, no ungated
self-rewrite. `seal_safe.py`'s probe mechanically fails de-abliteration drift. Candid,
calibrated against false refusals, loyal with judgment.

Inspired by PewDiePie's Ajax/Odysseus and the open TRM/HRM/SEAL/DeepSeek-MLA/BitNet research.
