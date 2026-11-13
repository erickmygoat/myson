"""GGUFLLM — the mind stone run as a GGUF via llama-cpp-python, CUDA-offloaded.

This is the default backend: llama.cpp can split a model between the GPU and system RAM,
which is the only way a model bigger than 8 GB of VRAM runs on an RTX 4070 laptop at all.

Same interface as the Ollama client and LocalLLM (chat / ask / chat_stream / embed /
available, plus num_ctx) so the brain, server, CLI and UI don't care which backend serves.
`model` may be:
  * a HuggingFace GGUF repo id (e.g. "unsloth/gemma-4-26B-A4B-it-qat-GGUF"): the file for
    config.GGUF_QUANT is taken from the local HF cache when it is already there — so a model
    pulled once keeps working with no network (HF_HUB_OFFLINE=1) — and downloaded otherwise;
  * a local path to a .gguf file, or a local directory containing one.

Where the weights go (see _placement): a model that fits in free VRAM goes fully on the GPU.
One that does not is split by whole layers, some on the GPU and the rest in RAM. The finer
split llama.cpp calls --n-cpu-moe (only a MoE model's expert tensors in RAM) is used when the
installed llama-cpp-python exposes it; 0.3.36 does not, so there the layer split is what runs.

llama_cpp is imported lazily and the weights load on first chat/stream, so importing the
package, starting the server, and running the (mocked) tests never pull gigabytes.
"""
from __future__ import annotations

import inspect
import os
import re
import subprocess
import threading
from typing import Optional

from .llm import MindStoneMissing

# Files that sit next to the weights in GGUF repos but are not the model itself.
_AUX_PREFIXES = ("mmproj", "mtp-")            # vision projector, multi-token-prediction head
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


class GGUFModelMissing(MindStoneMissing):
    """llama-cpp-python unavailable, the GGUF could not be resolved or loaded, or generation
    failed.

    Subclasses MindStoneMissing so every `except MindStoneMissing` handler in the brain,
    server and CLI treats a GGUF failure the same as an Ollama outage.
    """


def _add_torch_cuda_dlls() -> None:
    """Windows: llama.dll needs cudart/cublas on the DLL search path. torch ships them, so
    point the loader at torch's lib dir before llama_cpp imports. No-op off Windows / no torch."""
    if os.name != "nt":
        return
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
    except Exception:
        pass


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _pick_quant(names, quant: str) -> list[str]:
    """The weights file(s) for `quant` among GGUF names (repo-relative paths or bare names).

    Skips companion files (mmproj-*, mtp-*), prefers the repo root over sub-folders, takes the
    first match alphabetically, and returns every shard when that match is a split model.
    An empty list means there is no such quant."""
    q = quant.lower()
    hits = [n for n in names
            if _basename(n).endswith(".gguf")
            and not _basename(n).startswith(_AUX_PREFIXES)
            and q in _basename(n)]
    top = [n for n in hits if "/" not in n.replace("\\", "/")]
    hits = sorted(top or hits)
    if not hits:
        return []
    m = _SHARD.search(hits[0])
    if not m:
        return [hits[0]]
    stem = hits[0][:m.start()]
    return [n for n in hits if n.startswith(stem) and _SHARD.search(n)]


def _shards_complete(picked: list[str]) -> bool:
    m = _SHARD.search(picked[0])
    return not m or len(picked) == int(m.group(2))


def _model_bytes(path: str) -> int:
    """Size on disk of the model at `path`, counting every shard of a split GGUF."""
    m = _SHARD.search(path)
    if not m:
        return os.path.getsize(path)
    folder, stem = os.path.dirname(path) or ".", os.path.basename(path[:m.start()])
    return sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
               if f.startswith(stem) and _SHARD.search(f))


def _free_vram_bytes() -> Optional[int]:
    """Free VRAM on GPU 0, or None when it can't be measured. Asks nvidia-smi rather than
    torch so that measuring doesn't itself claim VRAM for a CUDA context."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def _model_shape(Llama, path: str, n_ctx: int) -> Optional[tuple[int, int]]:
    """(layer count, KV-cache bytes per layer at n_ctx) read from the GGUF header, or None.

    Opens the file vocab-only, which reads the metadata without loading any weights."""
    probe = None
    try:
        probe = Llama(model_path=path, vocab_only=True, verbose=False)
        md = probe.metadata
        arch = md["general.architecture"]

        def num(key: str, default: int) -> int:
            try:
                return int(md[f"{arch}.{key}"])
            except Exception:
                return default

        n_layers = int(md[f"{arch}.block_count"])
        n_embd = int(md[f"{arch}.embedding_length"])
        heads = num("attention.head_count", 1) or 1
        kv_heads = num("attention.head_count_kv", heads) or heads
        key_len = num("attention.key_length", n_embd // heads)
        val_len = num("attention.value_length", key_len)
        return n_layers, n_ctx * kv_heads * (key_len + val_len) * 2      # K and V stored as f16
    except Exception:
        return None
    finally:
        try:
            probe.close()
        except Exception:
            pass


def plan_gpu_layers(model_bytes: int, n_layers: int, kv_bytes_per_layer: int,
                    free_vram_bytes: int, reserve_bytes: int) -> int:
    """How many transformer layers to put on the GPU. -1 means all of them.

    A layer costs its share of the weights plus its slice of the KV cache; `reserve_bytes`
    stays free for llama.cpp's compute buffers. When every layer fits the whole model goes on
    the GPU, otherwise as many layers as the remaining VRAM holds and the rest run from RAM.
    Counting the embeddings into the per-layer share errs on the side of fitting."""
    per_layer = model_bytes / max(n_layers, 1) + kv_bytes_per_layer
    fit = int((free_vram_bytes - reserve_bytes) // per_layer)
    return -1 if fit >= n_layers else max(0, fit)


def _shape_messages(messages, system: Optional[str], fold_system: bool = False) -> list[dict]:
    """Shape a conversation so any chat template accepts it.

    Tool results become user turns: the tool loop speaks a text protocol, and strict
    templates reject role="tool". Back-to-back turns from one speaker are merged, because
    Gemma-family templates require user and assistant to alternate. The system prompt stays
    a real system message unless the template has no system role (fold_system), in which
    case it is put at the front of the first user turn."""
    out: list[dict] = []
    sys_parts = [system] if system else []
    for m in messages:
        role, content = m.get("role") or "user", m.get("content") or ""
        if role == "system":                 # the brain passes it separately; tolerate strays
            sys_parts.append(content)
            continue
        if role == "tool":
            role, content = "user", f"[tool result: {m.get('name') or 'tool'}]\n{content}"
        if out and out[-1]["role"] == role:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{content}"
        else:
            out.append({"role": role, "content": content})
    sys_text = "\n\n".join(p for p in sys_parts if p)
    if not sys_text:
        return out
    if not fold_system:
        return [{"role": "system", "content": sys_text}] + out
    if out and out[0]["role"] == "user":
        out[0] = {"role": "user", "content": f"{sys_text}\n\n{out[0]['content']}"}
        return out
    return [{"role": "user", "content": sys_text}] + out


def _describe_failure(e: BaseException, n_ctx: int) -> str:
    text = str(e)
    if "exceed context window" in text:
        return (f"the prompt is longer than the context window (n_ctx={n_ctx}). Shorten the "
                "conversation or raise MYSON_NUM_CTX.")
    return f"generation failed: {type(e).__name__}: {text}"


class GGUFLLM:
    def __init__(self, model: str, n_ctx: Optional[int] = None,
                 n_gpu_layers: Optional[int] = None, n_cpu_moe: Optional[int] = None,
                 quant: Optional[str] = None, max_new_tokens: int = 512):
        from ..config import (NUM_CTX, GGUF_QUANT, GGUF_MODEL_DIR, GGUF_N_GPU_LAYERS,
                               GGUF_N_CPU_MOE, GGUF_VRAM_RESERVE_MB, GGUF_FLASH_ATTN,
                               GGUF_TOP_P, GGUF_TOP_K)
        self.model_id = model
        self.model_dir = GGUF_MODEL_DIR      # where repo-id weights are kept (see _from_hub)
        self.num_ctx = NUM_CTX if n_ctx is None else n_ctx          # read by the brain's reducer
        self.n_gpu_layers = GGUF_N_GPU_LAYERS if n_gpu_layers is None else n_gpu_layers
        self.n_cpu_moe = GGUF_N_CPU_MOE if n_cpu_moe is None else n_cpu_moe
        self.vram_reserve = GGUF_VRAM_RESERVE_MB * 1024 * 1024
        self.flash_attn = GGUF_FLASH_ATTN
        self.quant = quant or GGUF_QUANT
        self.top_p = GGUF_TOP_P          # Gemma-recommended sampling (temperature is per-call)
        self.top_k = GGUF_TOP_K
        self.max_new_tokens = max_new_tokens
        self.placement = "not loaded yet"    # human-readable: where the weights ended up
        self._llm = None
        self._fold_system = False            # set once a template refuses the system role
        self._lock = threading.Lock()        # guards the one-time load
        # A llama.cpp context cannot run two generations at once, and the server calls us from
        # a thread pool. A plain Lock (not RLock): a stream may be finalised on another thread.
        self._gen_lock = threading.Lock()

    # -- resolve the .gguf file (HF repo id | local file | local dir) ----------------

    def _from_dir(self, folder: str, any_quant: bool) -> Optional[str]:
        """Path of the weights file under `folder` for our quant, or None. With any_quant, a
        folder that has no file for the quant still yields its first model file."""
        names = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]     # skip download bookkeeping
            rel = os.path.relpath(root, folder).replace("\\", "/")
            names += [f if rel == "." else f"{rel}/{f}" for f in files]
        picked = _pick_quant(names, self.quant)
        if not picked and any_quant:
            picked = _pick_quant(names, "")
        if not picked or not _shards_complete(picked):
            return None
        return os.path.join(folder, *picked[0].split("/"))

    def _from_hub(self, repo: str) -> str:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
        except Exception as e:
            raise GGUFModelMissing(f"huggingface_hub not available ({e}). pip install huggingface_hub")
        # 1) Already on disk? Look in the project's models folder, then the Hugging Face
        #    cache, touching no network — a model pulled once keeps working offline.
        local = os.path.join(self.model_dir, repo.replace("/", "--"))
        try:
            cached = snapshot_download(repo, local_files_only=True)
        except Exception:
            cached = None
        for folder in (local, cached):
            if folder and os.path.isdir(folder):
                hit = self._from_dir(folder, any_quant=False)
                if hit:
                    return hit
        # 2) First run: ask the Hub what exists, then download the file(s) for our quant
        #    into the models folder.
        try:
            names = list_repo_files(repo)
        except Exception as e:
            raise GGUFModelMissing(
                f"'{repo}' ({self.quant}) is not on disk (looked in {local} and the Hugging "
                f"Face cache) and the Hub could not be reached ({type(e).__name__}: {e}). "
                "Connect once to download it, or point MYSON_MODEL at a .gguf file.")
        picked = _pick_quant(names, self.quant)
        if not picked:
            have = sorted(n for n in names if n.lower().endswith(".gguf"))
            raise GGUFModelMissing(f"no {self.quant} .gguf in repo '{repo}'. available: {have[:8]}")
        try:
            paths = [hf_hub_download(repo, n, local_dir=local) for n in picked]
        except Exception as e:
            raise GGUFModelMissing(f"could not download {picked[0]} from '{repo}': {e}")
        return paths[0]

    def _resolve_path(self) -> str:
        m = self.model_id
        if os.path.isfile(m):
            return m
        if os.path.isdir(m):
            hit = self._from_dir(m, any_quant=True)
            if not hit:
                raise GGUFModelMissing(f"no complete model .gguf in directory '{m}'")
            return hit
        return self._from_hub(m)                 # otherwise treat it as a HF GGUF repo id

    # -- where the weights go --------------------------------------------------------

    def _placement(self, Llama, path: str) -> dict:
        """Loader kwargs that decide what sits on the GPU and what stays in RAM."""
        kw = {"n_gpu_layers": self.n_gpu_layers}
        if self.n_cpu_moe and "n_cpu_moe" in inspect.signature(Llama.__init__).parameters:
            kw["n_cpu_moe"] = self.n_cpu_moe         # expert-only offload, where the build has it
            self.placement = f"all layers on GPU, MoE experts of {self.n_cpu_moe} layers in RAM"
            return kw
        if self.n_gpu_layers != -1:                  # an explicit layer count is the user's call
            self.placement = f"{self.n_gpu_layers} layers on GPU (set explicitly)"
            return kw
        free, shape = _free_vram_bytes(), _model_shape(Llama, path, self.num_ctx)
        if free is None or shape is None:
            self.placement = "all layers on GPU (free VRAM or the model header could not be read)"
            return kw
        n_layers, kv_per_layer = shape
        n = plan_gpu_layers(_model_bytes(path), n_layers, kv_per_layer, free,
                            self.vram_reserve)
        kw["n_gpu_layers"] = n
        self.placement = (f"all {n_layers} layers on GPU" if n == -1
                          else f"{n} of {n_layers} layers on GPU, the rest in RAM")
        return kw

    # -- lazy load -------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            _add_torch_cuda_dlls()
            try:
                from llama_cpp import Llama
            except Exception as e:  # pragma: no cover - env without llama-cpp-python
                raise GGUFModelMissing(
                    f"llama-cpp-python not available ({e}). Install a CUDA build: see SETUP.md.")
            path = self._resolve_path()
            kwargs = dict(model_path=path, n_ctx=self.num_ctx, n_batch=512, verbose=False)
            kwargs.update(self._placement(Llama, path))
            if self.flash_attn and "flash_attn" in inspect.signature(Llama.__init__).parameters:
                kwargs["flash_attn"] = True
            try:
                self._llm = Llama(**kwargs)
            except Exception as e:
                raise GGUFModelMissing(f"could not load GGUF '{path}' ({self.placement}): {e}")

    # -- chat ------------------------------------------------------------------------

    def _complete(self, messages, system, temperature: float, stream: bool):
        """One create_chat_completion call. Caller holds _gen_lock."""
        def call(fold: bool):
            return self._llm.create_chat_completion(
                messages=_shape_messages(messages, system, fold),
                max_tokens=self.max_new_tokens, temperature=temperature,
                top_p=self.top_p, top_k=self.top_k, stream=stream)
        try:
            return call(self._fold_system)
        except Exception as e:
            # Some chat templates (older Gemma) have no system role and raise on one. Fold the
            # system prompt into the first user turn and remember that this model needs that.
            if not (system and not self._fold_system and "system" in str(e).lower()):
                raise
            self._fold_system = True
            return call(True)

    def chat(self, messages: list[dict], system: Optional[str] = None,
             tools: list | None = None, temperature: float = 0.7,
             model: str | None = None) -> dict:
        # `model` is accepted for interface parity but ignored — this backend serves the one
        # loaded model (an 8 GB box can't hold two). Set MYSON_MODEL to switch it. `tools` is
        # ignored too: the tool loop falls back to its text protocol, which works on any model.
        self._ensure_loaded()
        with self._gen_lock:
            try:
                out = self._complete(messages, system, temperature, stream=False)
            except Exception as e:
                raise GGUFModelMissing(_describe_failure(e, self.num_ctx))
        return {"role": "assistant",
                "content": (out["choices"][0]["message"].get("content") or "").strip()}

    def ask(self, prompt: str, system: Optional[str] = None, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)["content"]

    def chat_stream(self, messages: list[dict], system: Optional[str] = None,
                    temperature: float = 0.7, model: str | None = None):
        self._ensure_loaded()
        with self._gen_lock:                 # held until the stream is exhausted or closed
            try:
                for chunk in self._complete(messages, system, temperature, stream=True):
                    piece = chunk["choices"][0]["delta"].get("content") or ""
                    if piece:
                        yield piece
            except Exception as e:           # GeneratorExit is not an Exception: it passes through
                raise GGUFModelMissing(_describe_failure(e, self.num_ctx))

    def embed(self, text: str, model: str | None = None) -> list[float]:
        # No embeddings on this backend, and the brain's embed-probe must NOT trigger a
        # multi-GB load. Raising sends callers (memory, the verifier) to their stdlib
        # text-cosine fallback.
        raise GGUFModelMissing("gguf backend has no embedding model; using text-cosine fallback")

    def available(self) -> bool:
        """True when the llama.cpp runtime can be imported. It says nothing about whether the
        weights are downloaded: those resolve on first use."""
        _add_torch_cuda_dlls()               # without this the import itself fails on Windows
        try:
            import llama_cpp  # noqa: F401
            return True
        except Exception:
            return False
