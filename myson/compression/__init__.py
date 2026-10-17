"""myson compression — own-model efficiency tricks (require torch + weight control).

These attach to a model whose architecture we define (the repo's own transformer in
myson/training), NOT to a black-box model served by Ollama. See docs/FUTURE.md.

  * bitnet : 1.58-bit ternary weight quantization (memory win; QAT + int8 inference store)
  * mla    : low-rank KV-cache compression (DeepSeek MLA), RoPE-correct for prefill + decode
  * mor    : adaptive per-token recursion depth (soft-trained, hard gather at inference)
"""
from .bitnet import (BitLinear, BitLinearInference, convert_to_bitnet,
                     prepare_for_inference, quantize_weights_158bit, memory_report)
from .mla import MultiHeadLatentAttention, MLATransformerBlock, LatentCompressor
from .mor import MixtureOfRecursions, MoRTransformerBlock, RecursiveBlock, DepthPredictor

__all__ = [
    "BitLinear", "BitLinearInference", "convert_to_bitnet", "prepare_for_inference",
    "quantize_weights_158bit", "memory_report",
    "MultiHeadLatentAttention", "MLATransformerBlock", "LatentCompressor",
    "MixtureOfRecursions", "MoRTransformerBlock", "RecursiveBlock", "DepthPredictor",
]
