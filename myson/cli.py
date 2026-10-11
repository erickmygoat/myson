"""Wake MY SON. Chat loop against the local mind stone (Ollama)."""
from .core.llm import LLMClient, MindStoneMissing
from .core.memory import Memory
from .config import MODEL


def main():
    llm, mem = LLMClient(), Memory()
    print(f"MY SON — mind stone: {MODEL}. type 'exit' to quit.\n")
    history = []
    while True:
        try:
            u = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u in ("exit", "quit"):
            break
        if not u:
            continue
        ctx = mem.recall(u)
        sysmsg = None
        if ctx:
            from .config import SYSTEM
            sysmsg = SYSTEM + "\n\nRELEVANT MEMORY:\n" + "\n".join(f"- {c}" for c in ctx)
        history.append({"role": "user", "content": u})
        mem.save("user", u)
        try:
            reply = llm.chat(history, system=sysmsg or None).get("content", "")
        except MindStoneMissing as e:
            print(f"\n[no mind stone] {e}\n"); continue
        history.append({"role": "assistant", "content": reply})
        mem.save("assistant", reply)
        print(f"\nmyson> {reply}\n")


if __name__ == "__main__":
    main()
