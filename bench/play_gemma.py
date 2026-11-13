"""Chat with myson's home model by itself, on your own GPU — no server, no wrapper.

    python bench/play_gemma.py                      # interactive chat
    python bench/play_gemma.py "what is 17 * 23?"   # answer one question and exit

Loads Gemma 4 26B-A4B (QAT, 14.2 GB) from the models/ folder at the repo root — the first run
downloads it there — and talks to it with myson's persona. It goes through the same backend
the app uses (myson/core/gguf_llm.py), so what you feel here is what the app gets: the loader
measures free VRAM, puts as many layers on the GPU as fit, and runs the rest from RAM. Every
reply prints its speed.

    exit      quit                /reset    forget the conversation
    Ctrl+C    stop the current reply

Knobs (environment variables):
    GEMMA_REPO     a Hugging Face GGUF repo id, or a path to a .gguf file or folder
    GEMMA_QUANT    which quant file in the repo (default Q4_K_XL)
    N_CTX          context window (default 4096; bigger costs memory, so fewer layers on GPU)
    N_GPU_LAYERS   -1 decides automatically (default); or an exact number of layers
    MAX_TOKENS     longest reply (default 1024)

The lighter 12B, for comparison (PowerShell):
    $env:GEMMA_REPO="unsloth/gemma-4-12b-it-GGUF"; $env:GEMMA_QUANT="Q4_K_M"; python bench/play_gemma.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myson import config                                    # noqa: E402
from myson.core.gguf_llm import GGUFLLM, GGUFModelMissing   # noqa: E402

REPO = os.environ.get("GEMMA_REPO", "unsloth/gemma-4-26B-A4B-it-qat-GGUF")
QUANT = os.environ.get("GEMMA_QUANT", "Q4_K_XL")
N_CTX = int(os.environ.get("N_CTX", "4096"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))


def reply_to(llm: GGUFLLM, history: list[dict]) -> str:
    """Stream one reply to the screen, then print how fast it came."""
    print("myson> ", end="", flush=True)
    reply, start, first = "", time.time(), None
    try:
        for piece in llm.chat_stream(history, system=config.SYSTEM, temperature=1.0):
            first = first or time.time()
            print(piece, end="", flush=True)
            reply += piece
    except KeyboardInterrupt:
        print("  (stopped)", end="")
    end = time.time()
    n = len(llm._llm.tokenize(reply.encode("utf-8"), add_bos=False)) if reply else 0
    speed = f"{(n - 1) / (end - first):.1f} tok/s" if n > 1 and end > first else "n/a"
    print(f"\n  [{n} tokens · first token after {(first or end) - start:.1f}s · {speed}]\n")
    return reply


def main() -> int:
    llm = GGUFLLM(model=REPO, quant=QUANT, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS,
                  max_new_tokens=MAX_TOKENS)
    print(f"model : {REPO} ({QUANT})")
    print("loading… the first run downloads the weights into models/, which takes a while")
    t0 = time.time()
    try:
        llm._ensure_loaded()
    except GGUFModelMissing as e:
        print(f"\ncould not load the model: {e}")
        return 1
    print(f"loaded in {time.time() - t0:.0f}s — {llm.placement}, context {N_CTX} tokens\n")

    history: list[dict] = []
    once = " ".join(sys.argv[1:]).strip()
    if once:                                    # one question from the command line
        history.append({"role": "user", "content": once})
        try:
            reply_to(llm, history)
        except GGUFModelMissing as e:
            print(f"\n[{e}]")
            return 1
        return 0

    print("chat below. 'exit' quits, '/reset' forgets the conversation, Ctrl+C stops a reply.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("exit", "quit"):
            break
        if user == "/reset":
            history.clear()
            print("(conversation cleared)\n")
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        try:
            history.append({"role": "assistant", "content": reply_to(llm, history)})
        except GGUFModelMissing as e:
            print(f"\n[{e}]\n")
            history.pop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
