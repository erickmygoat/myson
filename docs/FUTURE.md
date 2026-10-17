# myson — FUTURE (noted, not built now)

## Base-model framing (the honest version)
"Bigger model, fewer params but still big" is delivered by **quantization** — fewer
BITS per parameter, NOT fewer parameters — plus **adaptive compute**, not by literally
shrinking the param count. Picks that fit 8 GB VRAM:
- **Gemma-4-12B @ Q4** (~7-8 GB) — more parameters, quantized.
- **Qwen3.5-9B @ Q4** (~6 GB) — PewDiePie's Ajax base.
`config.MODEL` stays swappable, so you can change the mind stone without touching the brain.

## The "new model" is a SYSTEM, not merged weights
You do NOT merge your code into the weights. The quantized open model **+** the myson
wrapper (recursion + tools + memory + SEAL) *together* are the "new model." The wrapper
is where your ideas live; the weights stay stock (or your own QLoRA fine-tune, later).

## Adaptive depth we already have (today, on a wrapped model)
The recursion loop spends MORE passes on hard queries and fewer on easy ones
(test-time compute) — that IS "acts bigger when it needs to," achievable on a black-box
Ollama model right now. No weight access required.

## Latent tricks — ONLY if you train your OWN small model (later track)
These need control of the model's internals, so they do NOT apply to a black-box Ollama
wrap of Qwen/Gemma. Park them until you fine-tune/train your own small model:
- `bitnet.py` — 1-bit quantization (BitNet): extreme compression of YOUR weights.
- `mor.py` — Mixture-of-Recursions: adaptive per-token depth INSIDE your own net.
- `mla.py` — Multi-head Latent Attention: KV-cache compression (DeepSeek-style).

## Speed & scale (later)
- **Speculative decoding** — a small draft model proposes tokens, the big model verifies:
  faster generation. Future speed option.
- **More compute** — rent a cloud GPU (Vast.ai / RunPod) or get a used **RTX 3090 (24 GB)**:
  unlocks bigger models and genuinely parallel subagents.
