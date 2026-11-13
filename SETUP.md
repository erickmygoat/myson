# myson — SETUP (Windows + 8 GB VRAM)

myson runs its model **on your machine**: no cloud, no daemon. The default path is GGUF through
`llama-cpp-python` with CUDA, because llama.cpp can split a model between the GPU and system
RAM, which is the only way a model bigger than 8 GB of VRAM runs on an RTX 4070 laptop.

The home model is **Gemma 4 26B-A4B (QAT, 4-bit)**: one 14.2 GB file. Why that model, and what
is still unproven about it, is in [bench/MODEL_REVIEW.md](bench/MODEL_REVIEW.md). What is built
and what is not is in [docs/ROADMAP.md](docs/ROADMAP.md).

> **Honest status.** The wrapper (router, recursion, tools, memory, SEAL) is unit-tested against
> a mocked model and has not yet been run end to end on a real one. The GGUF backend and the
> play script have been run for real on this laptop, on the 26B itself: 11–12 of its 30 layers
> on the GPU, 21–25 tokens per second, about 2 s to the first token once the file is cached
> (10–30 s on a cold start).

---

## 1. Code and dependencies

```powershell
git clone https://github.com/erickmygoat/myson
cd myson
pip install fastapi uvicorn pydantic sympy httpx huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

The last line installs a prebuilt CUDA wheel (this setup uses 0.3.36). If no wheel matches your
Python, build from source instead, which needs the CUDA Toolkit and a C++ compiler:
`$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --no-cache-dir`.

`torch` is needed even though the model runs through llama.cpp: on Windows, `llama.dll` needs
the CUDA runtime DLLs, and the backend finds them in torch's `lib` folder. That is also why a
bare `import llama_cpp` fails here. Check the GPU build like this:

```powershell
python -c "import os, torch; os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib')); import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

## 2. Verify the code (no model needed)

```powershell
python -m unittest discover -s myson -p "test_*.py"     # 133 tests
python myson/learning/test_seal_safe.py                 # the 5 SEAL safety checks
```

The SEAL checks are a script, so the first command does not collect them. Run both.

## 3. The model

Nothing to configure: `myson/config.py` already names it.

- **Repo / file:** `unsloth/gemma-4-26B-A4B-it-qat-GGUF`, quant `Q4_K_XL` (14.2 GB).
- **Where it lives:** `models/` at the repo root, one sub-folder per repo, ignored by git. The
  first run downloads it there; after that it loads from disk with no network at all.
- **How it fits:** the backend reads the model's layer count, measures free VRAM, and puts as
  many layers on the GPU as fit. The rest run from RAM. `/health` and the play script report the
  split it chose. Close GPU-heavy apps first for a better split; a bigger context window leaves
  room for fewer layers.

Other models, no code change:

```powershell
# the non-QAT build used in the cloud benchmark
$env:MYSON_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"; $env:MYSON_GGUF_QUANT = "Q4_K_M"
# the lighter 12B
$env:MYSON_MODEL = "unsloth/gemma-4-12b-it-GGUF"; $env:MYSON_GGUF_QUANT = "Q4_K_M"
# any .gguf you already have
$env:MYSON_MODEL = "D:\models\something.gguf"
```

## 4. Play with the model by itself

```powershell
python bench/play_gemma.py
```

A plain chat with the model and myson's persona: no server, no wrapper. It prints where the
weights were placed and the speed of every reply. `python bench/play_gemma.py "a question"`
answers once and exits.

## 5. Wake the full assistant

```powershell
python -m myson.server          # workspace + API on http://127.0.0.1:8080
python -m myson.cli             # or chat in the terminal; /think, /fast, /tools steer it
```

The workspace has chat, a live reasoning scratchpad, a memory browser and the SEAL queue. The
model loads on the first message. `curl http://127.0.0.1:8080/health` shows whether the runtime
is ready and how the weights were placed.

## 6. Knobs (environment variables)

| Var | Default | What |
|---|---|---|
| `MYSON_BACKEND` | `gguf` | `gguf`, `local` (transformers, for working on weights) or `ollama` |
| `MYSON_MODEL` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | GGUF repo id, `.gguf` file, or a folder holding one |
| `MYSON_GGUF_QUANT` | `Q4_K_XL` | which quant file to take from a repo; it must exist there |
| `MYSON_MODEL_DIR` | `models/` in the repo | where downloaded weights are kept |
| `MYSON_GGUF_N_GPU_LAYERS` | `-1` | `-1` decides automatically; any other number is used as given |
| `MYSON_GGUF_VRAM_RESERVE_MB` | `1280` | VRAM left free when deciding; raise it if a load fails |
| `MYSON_GGUF_N_CPU_MOE` | `0` | expert-only offload, for llama-cpp-python builds that support it (0.3.36 does not) |
| `MYSON_GGUF_FLASH_ATTN` | `1` | flash attention: smaller KV cache |
| `MYSON_GGUF_TOP_P` / `_TOP_K` | `0.95` / `64` | Gemma's recommended sampling |
| `MYSON_NUM_CTX` | `8192` | context window; bigger means fewer layers on the GPU |
| `MYSON_MAX_NEW_TOKENS` | `512` | longest reply |
| `MYSON_MEMORY_DIR` | `.myson_mem` | memory and SEAL state |
| `MYSON_RECURSION_MAX_STEPS` | `6` | recursion step cap (hard ceiling 16) |

With the GGUF backend there is no embedding model, so memory recall uses its built-in
text-cosine matching.

## 7. SEAL (self-improvement) — gated on purpose

Self-edits never apply themselves:

```powershell
python -m myson.cli seal list | review <id> | approve <id> | apply <id> | versions | rollback <n>
```

## 8. Working on weights

Fine-tuning, merging and the compression track use `MYSON_BACKEND=local` (transformers) and
need `transformers accelerate bitsandbytes peft trl`. The compression benchmark
(`python -m myson.compression.benchmark`) runs on the repo's own small transformer, not on
Gemma. Fine-tuning a model the size of the 26B needs a bigger GPU than this laptop; the route is
train elsewhere, merge, convert to GGUF, and point `MYSON_MODEL` at the result.

## Boundaries (enforced in code, not just policy)

No abliteration of the harm floor, no unconditional-obedience core, no ungated self-rewrite or
auto-deploy. `seal_safe.py`'s probe mechanically fails de-abliteration drift. The persona is
candid, calibrated against false refusals, and loyal with judgment.
