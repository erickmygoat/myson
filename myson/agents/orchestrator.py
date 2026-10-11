"""OrchestratorAgent — coordinator -> workers -> synthesis.
The clean repurpose of the old 'orchestration' idea: coordinate AI work, not payloads.
Reuses ONLY clean primitives (hive + llm). Never imports tools.py/swarm.py."""
import re
from ..core.llm import LLMClient
from ..core.memory import Memory
from .hive import SubAgentManager
from ..config import PERSONAS


class OrchestratorAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.hive = SubAgentManager(self.llm)
        self.mem = Memory()

    def decompose(self, goal: str) -> list[dict]:
        plan = self.llm.ask(
            f"Break this goal into 2-5 concrete subtasks. One per line, prefixed "
            f"'[persona] '. personas: {list(PERSONAS)}.\n\nGOAL: {goal}",
            system=PERSONAS["planner"])
        tasks = []
        for line in plan.splitlines():
            m = re.match(r"\s*\[?(\w+)\]?[:\-\s]+(.+)", line.strip())
            if m and m.group(2):
                p = m.group(1).lower()
                tasks.append({"persona": p if p in PERSONAS else "research",
                              "task": m.group(2).strip()})
        return tasks or [{"persona": "research", "task": goal}]

    def run(self, goal: str) -> dict:
        self.mem.save("user", goal)
        subtasks = self.decompose(goal)
        for t in subtasks:
            agent = self.hive.spawn(t["persona"])
            t["result"] = self.hive.run(agent, t["task"])   # TODO: parallel via a thread pool
            # TODO: quality gate here — port rat_urm RATVerifier as the trust check
        combined = "\n\n".join(f"[{t['persona']}] {t['task']}\n{t['result']}" for t in subtasks)
        answer = self.llm.ask(
            f"GOAL: {goal}\n\nWORKER OUTPUTS:\n{combined}\n\nSynthesize the final answer.")
        self.mem.save("assistant", answer)
        return {"goal": goal, "subtasks": subtasks, "answer": answer}
