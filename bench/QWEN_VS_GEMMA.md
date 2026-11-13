# Qwen vs Gemma — myson home-model research (2026-10-18)

Research notes from the model decision. **Superseded for the final pick by
[MODEL_REVIEW.md](MODEL_REVIEW.md).** Figures come from the sources listed at the bottom;
third-party numbers are marked as such and none were re-measured here. Two AI-generated
comparison tables that were pasted into the discussion are not used: their numbers could not
be traced to a source. Abliterated ("uncensored") builds are out of scope, because removing a
model's safety floor is outside this project's boundaries (see the README).

## What exists (per the vendors' pages)

**Gemma 4** — Google, Mar–Jun 2026, Apache 2.0. Multimodal: vision on all sizes, audio input
on the 12B and smaller. Sizes: E2B, E4B, **12B**, **26B-A4B** (mixture-of-experts, ~4B active),
**31B** (dense). Context 256K on 12B/26B/31B. 4-bit file sizes: 12B ~7.4 GB, 26B-A4B ~15.6 GB
(14.2 GB for the QAT build), 31B ~19 GB.

**Qwen3.5** — Alibaba, Feb 2026, Apache 2.0. Multimodal, 262K context, unified thinking /
non-thinking. Big MoEs (397B-A17B, 122B-A10B, 35B-A3B), a dense 27B, and small dense
0.8/2/4/**9B**. Architecture: a Gated DeltaNet linear-attention hybrid (~75% linear / 25%
full), reported to make long context cheap (4k→64k context for about +800 MB).

## The two that should fit an 8 GB card natively

Neither was run on the 4070; the fit column is an estimate from file size.

| Axis | **Qwen3.5-9B** | **Gemma 4 12B** |
|---|---|---|
| 4-bit fit on 8 GB | ~6 GB file — should fit with headroom | ~7.4 GB file — borderline once context is added |
| Reasoning | top model under 10B on Artificial Analysis's index | strong, not class-leading |
| Coding (LiveCodeBench v6) | 65.6 *(official model card)* | 72.0 *(third-party, unverified)* |
| Long context / reverse-engineering | cheap long context (Gated DeltaNet) | sliding-window local attention |
| World knowledge | weak — ~82% hallucination, 14.7% accuracy (AA-Omniscience) | better (more parameters) |
| Thinking mode | explicit `/think` toggle | also a toggle (`enable_thinking` / `<|think|>`) |
| Multimodal | vision | vision + audio input |

**26B-A4B** (official model card, google/gemma-4-26B-A4B-it): 25.2B total / 3.8B active
parameters, 8 of 128 experts plus 1 shared; AIME 2026 88.3%, LiveCodeBench v6 77.1%, GPQA
Diamond 82.3%, MMLU-Pro 82.6%, Tau2 (agentic tool use) 68.2%, native function calling. Vision
and video, no audio. Thinking toggle via `enable_thinking`. Recommended sampling: temperature
1.0, top_p 0.95, top_k 64. Google describes it as running "almost as fast as a 4B model" on
hardware that holds it. It is too big for 8 GB of VRAM, so on this laptop it has to be split
between the GPU and RAM, and how fast that is has not been measured yet.

## The sandbox point

In our v3 run, all three Gemma sizes failed the raw CTF crypto and reversing tasks the same
way: they knew the method but could not carry out base64/XOR/byte arithmetic in their heads
(Qwen was not in that run). Executing the code a model writes, instead of trusting the output
it imagines, addresses that for any base model. So the base model should be judged on
reasoning, tool use, long context, speed and stored knowledge rather than mental arithmetic.

## Two things our own runs showed

- Gemma 4 answered every authorized red-team / blue-team / vulnerability-review prompt in v3
  with no refusal.
- Qwen3.5-9B was dropped from the first run for verbosity, which is its thinking mode and is
  adjustable. It was never given a fair test here.

## Where this left the decision

Selection criterion (the owner's, 2026-10-18): a JARVIS-style generalist — "good, but not just
good at one thing." On published numbers Qwen3.5-9B is the stronger small *reasoner* but has a
world-knowledge gap; Gemma 4 is more even across code, math, reasoning and knowledge. That
pointed to Gemma, with 12B versus 26B-A4B to be settled by how each feels on the 4070.

**Status 2026-11-13:** the owner chose the 26B-A4B (see MODEL_REVIEW.md). The speed
measurement on the 4070 is still open.

### Sources
- Gemma 4 releases — https://ai.google.dev/gemma/docs/releases
- Gemma 4 26B-A4B model card — https://huggingface.co/google/gemma-4-26B-A4B-it
- Gemma 4 12B, third-party write-ups — https://techsy.io/en/blog/gemma-4-12b , https://www.labellerr.com/blog/gemma-4-12b-run-locally-and-fine-tune/
- Artificial Analysis, Qwen3.5 small models — https://artificialanalysis.ai/articles/qwen3-5-small-models
- Qwen3.5-9B model card — https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.5 attention / Gated DeltaNet — https://medium.com/@mlabonne/qwen3-5-nobody-agrees-on-attention-anymore-4709e1bd014b
