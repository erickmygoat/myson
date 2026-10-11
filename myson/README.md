# MY SON — the husk (Vision's body)

*A clean, honest rebuild. The body is your code. The **mind stone** is an open
model's weights (Qwen 3.5-9B). Insert the stone → the body comes alive.*

Inspired by **PewDiePie's Ajax + Odysseus**: a small open model (Qwen3.5-9B),
fine-tuned + run **locally** as an always-on personal agent with tools (web,
files, memory) — not a from-scratch frontier model. Use it for the stuff you
don't want to do: grammar-check, research, draft/refactor code, self-improve.

## Architecture (Vision = body + mind stone)

```
          you  ──►  cli.py / app
                      │
             ┌────────▼─────────┐
             │  OrchestratorAgent│  decompose → delegate → verify → synthesize
             └───────┬──────────┘
            ┌─────────┼──────────┐
       Worker(A)  Worker(B)  Worker(C)      ← hive personas, isolated context
            └─────────┼──────────┘
                      ▼
              core/llm.py  ──►  OLLAMA  ──►  Qwen3.5-9B   ← THE MIND STONE
                      │
            tools/ (web, files, python)   memory/ (persistent)
```

## Bring it to life (3 steps)

```powershell
# 1. insert the mind stone (download the open base model)
ollama pull qwen3.5:9b              # or gemma4:4b for a smaller/faster stone

# 2. (later) your own fine-tune → import as your model
#    HF QLoRA → merge → GGUF → `ollama create myson -f Modelfile`

# 3. wake it up
python -m myson_husk.cli
```

## What's a husk vs a mind
- **Husk (this repo):** orchestrator, personas, tools, memory, CLI — your code.
- **Mind stone (not trained here):** open weights, legally downloaded, swapped in
  via Ollama. Start with stock Qwen3.5-9B; later replace with your fine-tune.

## Boundaries (kept on purpose)
Local + uncensored-by-default is fine for a personal tool. Ajax itself **kept a
harm boundary** (no harm to user/others). This husk does the same. No offensive
modules are ported in — see PORT_MANIFEST.md.
