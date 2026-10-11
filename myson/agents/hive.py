"""Worker factory — spawn isolated persona agents, each its own context.
Clean port of the old hive.py SubAgentManager (offensive preset dropped)."""
from ..core.llm import LLMClient
from ..config import PERSONAS, SYSTEM


class SubAgentManager:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def spawn(self, persona: str, role: str | None = None) -> dict:
        system = role or PERSONAS.get(persona, SYSTEM)
        return {"name": persona, "system": system}

    def run(self, agent: dict, task: str) -> str:
        # each call = fresh context under the persona's system prompt
        return self.llm.ask(task, system=agent["system"])
