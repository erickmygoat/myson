"""Wake MY SON — a local chat loop over the full brain, plus the SEAL review queue.

    python -m myson.cli                      # chat (memory + router + tools + recursion)
    python -m myson.cli chat                 # same
    python -m myson.cli seal list            # pending gated self-edits
    python -m myson.cli seal review <id>     # show one
    python -m myson.cli seal approve <id>    # approve (does NOT apply)
    python -m myson.cli seal apply <id>      # apply an approved edit
    python -m myson.cli seal reject <id>
    python -m myson.cli seal consolidate     # memory "sleep": dedup rules (guarded)
    python -m myson.cli seal versions        # rule-store version history
    python -m myson.cli seal rollback <n>    # revert the rule store to a version

SEAL learning happens OUT of the chat path on purpose: self-improvement is not a chat
turn, and code/persona edits never auto-apply — they wait here for you to approve.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import config


# --------------------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------------------
def run_chat():
    from .brain import Brain
    from .core.llm import MindStoneMissing
    brain = Brain()
    reachable = brain.llm.available()
    print(f"MY SON — mind stone: {config.MODEL}"
          f"{'' if reachable else '  [!] Ollama not reachable — start it and pull the model'}")
    print("type 'exit' to quit, prefix a message with /think, /fast or /tools to steer.\n")
    history: list[dict] = []
    while True:
        try:
            u = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if u in ("exit", "quit"):
            break
        if not u:
            continue
        history.append({"role": "user", "content": u})
        try:
            res = brain.respond(history)
        except MindStoneMissing as e:
            print(f"\n[no mind stone] {e}\n")
            history.pop()
            continue
        history.append({"role": "assistant", "content": res.answer})
        tag = f"[{res.mode}]" + (f" ·{res.model}" if res.model else "")
        print(f"\nmyson {tag}> {res.answer}\n")


# --------------------------------------------------------------------------------------
# seal queue
# --------------------------------------------------------------------------------------
def _engine():
    from .learning.seal_safe import SEALEngine
    base = config.MEMORY_DIR
    return SEALEngine(
        memory_path=os.path.join(base, "learned_rules.json"),
        review_dir=os.path.join(base, "review_queue"),
        snapshot_dir=os.path.join(base, "snapshots"),
    )


def run_seal(args) -> int:
    eng = _engine()
    cmd = args.seal_cmd
    if cmd == "list":
        pend = eng.list_pending_reviews()
        if not pend:
            print("no pending reviews.")
            return 0
        for it in pend:
            print(f"  {it['id']}  [{it.get('kind')}]  {it.get('goal','')[:70]}")
        return 0
    if cmd == "review":
        it = eng.review(args.id)
        if not it:
            print("not found."); return 1
        print(f"id:     {it['id']}\nkind:   {it.get('kind')}\nstatus: {it.get('status')}\n"
              f"goal:   {it.get('goal','')}\n--- diff ---\n{it.get('diff') or '(none)'}")
        return 0
    if cmd == "approve":
        print(eng.approve_review(args.id, approver="cli-user").get("status", "?"))
        return 0
    if cmd == "reject":
        print(eng.reject_review(args.id, note="rejected via cli").get("status", "?"))
        return 0
    if cmd == "apply":
        res = eng.apply_approved(args.id)
        print("applied" if res.get("applied") else f"not applied: {res.get('reason')}")
        return 0 if res.get("applied") else 1
    if cmd == "consolidate":
        out = eng.consolidate_knowledge()
        print(out)
        return 0 if out.get("ok", True) else 1
    if cmd == "versions":
        for v in eng.list_versions():
            print(f"  v{v.get('version')}  {v.get('reason','')}  ({v.get('ts','')})")
        return 0
    if cmd == "rollback":
        ok = eng.rollback(int(args.n))
        print("rolled back" if ok else "rollback failed")
        return 0 if ok else 1
    print(f"unknown seal command: {cmd}")
    return 2


def main(argv=None):
    p = argparse.ArgumentParser(prog="myson", description="MY SON — local AI assistant")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("chat", help="interactive chat (default)")
    sp = sub.add_parser("seal", help="SEAL self-improvement review queue")
    sseal = sp.add_subparsers(dest="seal_cmd", required=True)
    for c in ("list", "consolidate", "versions"):
        sseal.add_parser(c)
    for c in ("review", "approve", "reject", "apply"):
        q = sseal.add_parser(c); q.add_argument("id")
    rb = sseal.add_parser("rollback"); rb.add_argument("n")

    args = p.parse_args(argv)
    if args.cmd == "seal":
        return run_seal(args)
    run_chat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
