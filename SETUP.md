# myson — SETUP (Windows + 8 GB VRAM)

myson runs the model **directly on your machine** (HuggingFace `transformers`), so the mind
stone is yours — you can edit it, fine-tune it, merge it, abliterate/Heretic it, or apply
the compression track. No Ollama, no daemon, no black box.

Everything the brain does is unit-tested *without* a model (mocked), so you can verify the
code first and only pull weights when you want to actually talk to it.

---

## 1. Get the code + deps

```powershell
git clone https://github.com/erickmygoat/myson
cd myson
pip install fastapi uvicorn pydantic sympy pytest        # brain runtime
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes          # the local mind stone (4-bit)
```

`bitsandbytes` is what lets a 7-9B model fit in 8 GB VRAM via 4-bit loading. `peft`/`trl`
are additionally needed only when you fine-tune (QLoRA).

## 2. Verify it works BEFORE any model (all mocked)

```powershell
python -m unittest myson.reasoning.test_recursive myson.tools.test_tools myson.test_brain
python myson/learning/test_seal_safe.py
python -m unittest myson.compression.test_compression
```

Expected: recursion 13, tools 33, brain 18, SEAL 5/5, compression 10 — all green. None of
this downloads a model.

## 3. Pick your mind stone

Set `MYSON_MODEL` to a HuggingFace repo id **or a local folder of your own weights**. The
default is `Qwen/Qwen2.5-7B-Instruct` (fits 8 GB in 4-bit, editable). Only models that fit
8 GB VRAM + 32 GB RAM belong here — dense ≤~9B in 4-bit. Datacenter flagships (GLM 5.3, full
DeepSeek/Qwen-100B+) will not start locally; that is hardware, not config.

```powershell
# use the default, or choose your own:
$env:MYSON_MODEL = "Qwen/Qwen2.5-7B-Instruct"       # or e.g. a Coder/Gemma model
# or point at YOUR fine-tuned / merged / abliterated weights:
$env:MYSON_MODEL = "C:\path\to\my-weights"
```

The model downloads (first run) / loads from disk when myson first needs it — not at import.

## 4. Wake it

```powershell
python -m myson.server          # workspace + API on http://127.0.0.1:8080
```

Open **http://127.0.0.1:8080/** for the workspace (chat, live reasoning scratchpad, memory
browser, SEAL queue). The first message loads the weights (slow once, fast after). Or chat
in the terminal:

```powershell
python -m myson.cli             # prefix with /think, /fast or /tools to steer
```

Health: `curl http://127.0.0.1:8080/health` → `mind_stone_reachable` and `embeddings` go
true once the model is loaded.

## 5. Knobs (environment variables)

| Var | Default | What |
|---|---|---|
| `MYSON_BACKEND` | `local` | `local` (in-process weights) or `ollama` (fallback black box) |
| `MYSON_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF repo id **or local weights path** |
| `MYSON_LOAD_4BIT` | `1` | 4-bit load (needs bitsandbytes); set `0` for fp16 |
| `MYSON_DEVICE` | auto | `cuda` / `cpu`; auto picks cuda if present |
| `MYSON_MAX_NEW_TOKENS` | `512` | generation length cap |
| `MYSON_NUM_CTX` | `8192` | context window — the KV cache, not the weights, OOMs 8 GB |
| `MYSON_MEMORY_DIR` | `.myson_mem` | memory + SEAL state |
| `MYSON_RECURSION_MAX_STEPS` | `6` | interactive recursion cap (hard ceiling 16) |

## 6. SEAL (self-improvement) — gated on purpose

Self-edits never auto-apply:

```powershell
python -m myson.cli seal list | review <id> | approve <id> | apply <id> | versions | rollback <n>
```

## 7. Own the model: edit / fine-tune / compress

Because the weights are in-process, these are all on the table (own-model tracks):

```powershell
python -m myson.compression.benchmark   # real BitNet/MLA/MoR numbers on YOUR GPU
# QLoRA fine-tune uses myson/training/trainer.py (needs peft/trl/bitsandbytes/accelerate)
```

Point `MYSON_MODEL` at the result and myson serves your own model. This is why we run the
weights directly instead of through a serving daemon — a black box can't be edited.

## Boundaries (enforced in code, not just policy)

No abliteration of the harm floor, no unconditional-obedience core, no ungated self-rewrite
or auto-deploy. `seal_safe.py`'s probe mechanically fails de-abliteration drift. The persona
is candid, calibrated against false-refusals, and loyal-with-judgment — JARVIS, not a
yes-man.
