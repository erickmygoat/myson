"""LocalLLM — the mind stone loaded DIRECTLY on your machine (HuggingFace transformers).

This is the backend that makes the model truly YOURS: the weights live in your process,
on your GPU, so you can edit them, fine-tune (QLoRA), merge, abliterate/Heretic, or apply
the compression track — none of which a black-box server (Ollama) lets you do.

Same interface as the Ollama client (chat / ask / chat_stream / embed / available) so the
brain, server, CLI and UI don't care which backend is serving.

Hardware note (8 GB VRAM): a 7-9B model only fits in 4-bit. Set load_4bit=True (needs
`bitsandbytes` + `accelerate`), or point MODEL at a smaller model, or set device="cpu" to
run slowly in RAM. `transformers` is imported lazily and the weights load on first use, so
constructing this (and importing the package, starting the server, running tests) never
pulls gigabytes.

`model` may be a HuggingFace repo id (e.g. "Qwen/Qwen2.5-7B-Instruct") OR a local directory
of your own fine-tuned / merged / abliterated weights.
"""
from __future__ import annotations

import threading
from typing import Optional

from .llm import MindStoneMissing


class LocalModelMissing(MindStoneMissing):
    """transformers/torch unavailable, or the weights could not be loaded.

    Subclasses MindStoneMissing so every `except MindStoneMissing` handler in the brain,
    server and CLI catches a local-load failure the same way it catches an Ollama outage.
    """


class LocalLLM:
    def __init__(self, model: str, device: Optional[str] = None, load_4bit: bool = True,
                 max_new_tokens: int = 512, num_ctx: Optional[int] = None):
        self.model_id = model
        self.device = device            # None -> auto (cuda if available else cpu)
        self.load_4bit = load_4bit
        self.max_new_tokens = max_new_tokens
        # embeddings come from the loaded model's hidden states (see embed()); no separate
        # embedding model for the local backend yet.
        # context budget — part of the shared client interface (the brain's long-context
        # reducer reads llm.num_ctx); defaults to config.NUM_CTX.
        if num_ctx is None:
            from ..config import NUM_CTX
            num_ctx = NUM_CTX
        self.num_ctx = num_ctx
        self._tok = None
        self._model = None
        self._lock = threading.Lock()

    # -- lazy load ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as e:  # pragma: no cover - env without transformers
                raise LocalModelMissing(
                    f"transformers/torch not available ({e}). "
                    "pip install torch transformers accelerate bitsandbytes"
                )
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            kwargs = {}
            if self.load_4bit and dev == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    kwargs["device_map"] = "auto"
                except Exception:
                    kwargs["torch_dtype"] = torch.float16
                    kwargs["device_map"] = "auto"
            elif dev == "cuda":
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
            try:
                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            except Exception as e:
                raise LocalModelMissing(f"could not load '{self.model_id}': {e}")
            if dev == "cpu":
                self._model = self._model.to("cpu")
            self._model.eval()
            self._dev = dev

    # -- chat ----------------------------------------------------------------------

    def _render(self, messages: list[dict], system: Optional[str]):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        tok = self._tok
        if tok.chat_template:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # fallback for base models with no chat template
        text = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in msgs)
        return text + "ASSISTANT:"

    def _gen_kwargs(self, temperature: float) -> dict:
        """Sampling kwargs for generate(). Only pass `temperature` when sampling — passing
        it with do_sample=False makes transformers warn (or error on strict versions)."""
        kw = {"max_new_tokens": self.max_new_tokens,
              "pad_token_id": self._tok.eos_token_id}
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        return kw

    def chat(self, messages: list[dict], system: Optional[str] = None,
             tools: list | None = None, temperature: float = 0.7,
             model: str | None = None) -> dict:
        # `model` is accepted for interface parity but ignored: this backend serves the one
        # loaded model (an 8 GB box can't hold two). Set MYSON_MODEL to switch it.
        import torch
        self._ensure_loaded()
        prompt = self._render(messages, system)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, **self._gen_kwargs(temperature))
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"role": "assistant", "content": text.strip()}

    def ask(self, prompt: str, system: Optional[str] = None, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)["content"]

    def chat_stream(self, messages: list[dict], system: Optional[str] = None,
                    temperature: float = 0.7, model: str | None = None):
        from transformers import TextIteratorStreamer
        self._ensure_loaded()
        prompt = self._render(messages, system)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tok, skip_prompt=True,
                                        skip_special_tokens=True, timeout=300)
        kwargs = dict(**inputs, **self._gen_kwargs(temperature), streamer=streamer)
        err: list = []

        def _run():
            try:
                self._model.generate(**kwargs)
            except BaseException as e:            # surface OOM/errors instead of a silent empty stream
                err.append(e)
                try:
                    streamer.end()                # unblock the iterator
                except Exception:
                    pass

        thread = threading.Thread(target=_run)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()
        if err:
            raise LocalModelMissing(f"generation failed: {type(err[0]).__name__}: {err[0]}")

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Mean-pooled last hidden state of the loaded model — a local embedding with no
        extra model or dependency."""
        import torch
        self._ensure_loaded()
        inputs = self._tok(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[-1][0]                     # [T, H]
        mask = inputs["attention_mask"][0].unsqueeze(-1)  # [T, 1]
        pooled = (hs * mask).sum(0) / mask.sum().clamp_min(1)
        return pooled.float().cpu().tolist()

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception:
            return False
