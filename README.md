# myson

A local-first AI assistant that runs the model **directly on your machine** — the mind
stone is yours to edit, fine-tune, merge, abliterate, or compress. An open LLM loaded
in-process (HuggingFace `transformers`), wrapped with a recursive-reasoning brain, memory,
tools, and safe self-improvement — all your own code.

- **Mind:** open weights loaded directly (no Ollama, no daemon, no black box), 4-bit on 8 GB.
- **Brain (this repo):** recursive reasoning (RATURM), 3-tier memory, tool-use
  (code/math/web), SEAL self-improvement, a candid loyal-with-judgment persona.
- **Face:** its own local workspace UI (chat + live reasoning scratchpad + memory + SEAL queue).
- **Own-model track:** BitNet / MLA / MoR compression on the repo's own transformer.

Quickstart: see **[SETUP.md](SETUP.md)**. Short version:

```powershell
pip install fastapi uvicorn pydantic sympy torch transformers accelerate bitsandbytes
python -m myson.server      # http://127.0.0.1:8080   (set MYSON_MODEL to your weights)
```

Verify the whole brain with no model (mocked): `python -m unittest myson.test_brain` and
the sibling suites — 79 tests green on Windows + Python 3.13.

## Why in-process weights (not Ollama)

Ollama is local, but it only *serves* a frozen model — you can't edit it, fine-tune it, run
Heretic, or apply the compression track through it. myson loads the weights in your own
process so all of that is possible. `MYSON_BACKEND=ollama` remains as an optional fallback.

## Architecture

```
  you ─► workspace UI / cli ─► /v1/chat/completions
                                     │
                           Brain: recall ─► persona ─► router ─► {chat | tools | recursion} ─► save
                                     │
                     backend (make_llm) ─► LocalLLM (transformers, in-process)  ← the mind stone
                                     │
           tools (code/math/web)   memory (3-tier)   SEAL (gated self-improvement)
```

See `docs/BRAIN_ARCHITECTURE.md` for the full design, `docs/FUTURE.md` for the own-model
compression track (built) and later tracks.

## Boundaries (enforced in code)

No safety-floor removal / abliteration, no unconditional-obedience core, no ungated
self-rewrite. `seal_safe.py`'s probe mechanically fails de-abliteration drift. Candid,
calibrated against false-refusals, loyal-with-judgment.

Inspired by PewDiePie's Ajax/Odysseus + the open TRM/HRM/SEAL/DeepSeek-MLA/BitNet research.
