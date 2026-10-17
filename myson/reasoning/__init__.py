"""myson reasoning package.

- ``recursive.py`` : TEXT-space recursive reasoning (runtime; stdlib + a chat LLM).
- ``rat_urm.py``   : the latent RAT-URM model (training-time; requires torch).

The text reasoner is always importable. The latent model is imported tolerantly so a
torch-less runtime box (one that only runs Ollama for inference) can still do
``from myson.reasoning import reason`` without pulling in torch.
"""

from .recursive import reason, reason_trace, make_reasoner, RecursiveReasoner

try:  # training-time latent model; optional at runtime (requires torch)
    from .rat_urm import RATURM, RATVerifier, BacktrackController, create_rat_urm
except Exception:  # pragma: no cover - torch not installed in a runtime-only env
    RATURM = RATVerifier = BacktrackController = create_rat_urm = None

__all__ = [
    "reason", "reason_trace", "make_reasoner", "RecursiveReasoner",
    "RATURM", "RATVerifier", "BacktrackController", "create_rat_urm",
]
