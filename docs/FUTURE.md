# myson — FUTURE (noted, not built now)

## Base-model framing (the honest version)
"Bigger model, fewer params but still big" is delivered by **quantization** — fewer
BITS per parameter, NOT fewer parameters — plus **adaptive compute**, not by literally
shrinking the param count. The current pick is **Gemma 4 26B-A4B @ 4-bit** (14.2 GB), which is
bigger than 8 GB of VRAM and so runs split between the GPU and RAM; Gemma 4 12B is the
smaller option that fits the card. `config.MODEL` stays swappable, so you can change
the mind stone without touching the brain. The current plan and status live in
[ROADMAP.md](ROADMAP.md); this file keeps the longer-term notes.

## The "new model" is a SYSTEM, not merged weights
You do NOT merge your code into the weights. The quantized open model **+** the myson
wrapper (recursion + tools + memory + SEAL) *together* are the "new model." The wrapper
is where your ideas live; the weights stay stock (or your own QLoRA fine-tune, later).

## Adaptive depth we already have (today, on a wrapped model)
The recursion loop spends MORE passes on hard queries and fewer on easy ones
(test-time compute) — that IS "acts bigger when it needs to," achievable on any downloaded
model. No weight access required.

## Latent tricks — BUILT (own-model track, in myson/compression/)
These need control of the model's internals, so they attach to the repo's OWN transformer
(myson/training/model.py), NOT to a downloaded Qwen/Gemma. Lifted from the
corpus, bugs fixed, tested, and measured on the 4070 (see myson/compression/benchmark.py):
- `compression/bitnet.py` — 1.58-bit ternary quantization. **9.24x smaller weights** on a
  4.49M MySonModel (17.1 → 1.85 MB). Memory win is real; wall-clock is not (int8 store,
  no packed kernel). Correct STE + int8 inference store.
- `compression/mla.py` — Multi-head Latent Attention (KV-cache compression). **16x smaller
  KV cache** at seq 512. Fixed the RoPE sin-slot bug + the cached-decode crash; correct for
  prefill and incremental decode.
- `compression/mor.py` — Mixture-of-Recursions (adaptive per-token depth). **~52% fewer
  inference token-steps**. Fixed: real gather/scatter (was masking → no saving) + a ponder
  aux loss so the depth predictor actually trains (argmax gave it zero gradient).

Known follow-up for the training path: when a real data loader is lifted into
`training/`, mask pad tokens to `-100` so padding stops leaking into the loss.

## Speed & scale (later)
- **Speculative decoding** — a small draft model proposes tokens, the big model verifies:
  faster generation. Future speed option.
- **More compute** — rent a cloud GPU (Vast.ai / RunPod) or get a used **RTX 3090 (24 GB)**:
  unlocks bigger models and genuinely parallel subagents.
