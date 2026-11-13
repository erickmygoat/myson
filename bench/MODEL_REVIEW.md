# myson model review — the full field

Written 2026-10-20, corrected 2026-11-13. The verdict across the benchmark runs, the
Qwen-vs-Gemma research and the feel test. Companion docs: [QWEN_VS_GEMMA.md](QWEN_VS_GEMMA.md),
[JARVIS_FEEL_TEST.md](JARVIS_FEEL_TEST.md).

**What this evidence is and is not.** Every model run so far happened on a cloud GPU (Colab
A100) or a hosted API, not on the target laptop (RTX 4070, 8 GB VRAM, 32 GB RAM). The raw
outputs of those runs are not in this repo; the harnesses are (`kaggle_benchmark.py` is the
second, harder version, which replaced the first in place; `kaggle_benchmark_v3.py` is the
security set). Grading rule throughout: compute the ground truth, run the code an answer
contains, and count confident-but-wrong as a fail.

## The finalists (open weights, small enough to attempt on this laptop)

| Model | What it is | Result | Verdict |
|---|---|---|---|
| **Gemma 4 26B-A4B** | MoE, 25.2B total / 3.8B active | feel test 11/12 on the hosted full-precision model; even across categories | **chosen** |
| Gemma 4 12B | dense 12B | best overall in the contest-math and code run, no bluffing; has audio input | fallback |
| Qwen3.5-9B | dense 9B | top small reasoner on published indexes, with a world-knowledge gap; never fairly run here | not chosen |

**Why the 26B:** the owner wanted a generalist, and on Google's published numbers it is strong
across the board (AIME 2026 88.3, LiveCodeBench v6 77.1, GPQA Diamond 82.3, MMLU-Pro 82.6, Tau2
agentic 68.2, 256K context, native function calling). In the feel test, with myson's persona,
it handled math, coding, a bug-fix, an authorized-security walkthrough without lecturing,
general knowledge, correcting a false premise, and tool planning. Its one miss was verbosity
on a trivial formatting task.

**Caveats on that choice.** The feel test ran against the hosted full-precision model, not the
4-bit build that runs locally, and the 12B was not put through it (the 12B column is empty).
The 12B-versus-26B comparison on this laptop has not been made.

## Eliminated (open weights)

| Model | Why |
|---|---|
| Gemma 4 31B | ~19 GB file and ~33 GB in use at 16K context on the benchmark GPU; slower than the 26B there; no better on our tasks |
| DeepSeek-R1-Distill-14B | its reasoning ran past the token budget and truncated about half the time |
| Phi-4 14B | in our run it stated wrong results confidently and fell into a loop |
| gpt-oss-20b | harmony channel markup leaked into answers under llama.cpp |

**Finding that cut across sizes (v3):** on the CTF crypto and reversing tasks all three Gemma
sizes knew the method and still produced wrong answers, because they could not execute
base64/XOR/byte arithmetic mentally. A bigger model did not fix it. Running the code the model
writes is the fix, which makes putting the execution sandbox on myson's default path the top
wrapper task.

## Closed models, tested for reference only

Three Gemini models were run through the same battery on a hosted API out of curiosity. They
cannot be myson (cloud-only, not downloadable). Where they returned answers they passed; the
run ended early on quota errors.

## Which build of the 26B

On 8 GB the model is split between the GPU and RAM whatever the quant, so a smaller file means
more of it on the GPU.

| Build | Size | Notes |
|---|---|---|
| **QAT UD-Q4_K_XL** | **14.2 GB** | quantization-aware trained, so 4-bit loses little; smallest sensible build. **Default.** |
| standard UD-Q4_K_M | 16.9 GB | the build used in the cloud benchmark |
| standard UD-Q5_K_XL / UD-Q6_K_XL | 21.2 / 23.3 GB | more precision, more of the model in RAM |
| standard UD-Q8_K_XL | 27.6 GB | leaves little of 32 GB RAM for anything else |

Other variants seen: a supervised fine-tune (Gemopus) with no published comparison against the
base model, and a writing-style tune. Neither is used. Formats such as AWQ / FP8 / NVFP4 target
GPU-only serving stacks, not a GPU/RAM split. Abliterated ("uncensored") builds are not used:
removing the safety floor is outside this project's boundaries.

## Where things stand

**Gemma 4 26B-A4B, QAT `Q4_K_XL` (`unsloth/gemma-4-26B-A4B-it-qat-GGUF`) is the default in
`myson/config.py`.**

Measured on the 4070 (2026-11-13, play script, persona as the system prompt): the backend puts
11–12 of the 30 layers on the GPU (about 6.5 GB of VRAM) and the rest in RAM, and the model
answers at **21–25 tokens per second**, at both 4096 and 8192 tokens of context. The first token
arrives after about 2 s once the file is in the disk cache, 10–30 s on a cold start, and in
under a second on follow-up turns. The installed llama-cpp-python (0.3.36) has no
expert-offload option (`n_cpu_moe`), so this is a plain split by whole layers.

What is not established yet:

- The 12B has not been run on this laptop, so the 12B-versus-26B comparison here is open.
- The verbosity noted above is not handled anywhere yet. Stripping the reasoning trace and a
  non-thinking mode for easy prompts are planned wrapper work.
- The wrapper (router, recursion, tools, memory) has only ever run against mocks.
