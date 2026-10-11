# PORT MANIFEST — what moves into the husk, what stays behind

## PORT (clean, real, reusable)
| From old repo | Into husk | Note |
|---|---|---|
| `my_son/agent/hive.py` | `agents/hive.py` | ported — offensive preset dropped |
| `my_son/agent/llm.py` (Ollama path) | `core/llm.py` | rewritten clean, stdlib-only |
| `my_son/agent/architect.py` | `agents/architect.py` | TODO — RAG-over-codebase persona |
| `my_son/agent/scheduler.py` | `agents/scheduler.py` | TODO — recurring tasks |
| `my_son/brain/rat_urm.py` | `reasoning/rat_urm.py` | VERIFIED runs; use RATVerifier as quality gate |
| `my_son/brain/training/` + `model.py` | `training/` | VERIFIED: MySonModel forwards, TRM recursion on |
| `my_son/chess/reasoner/` | separate repo | already standalone |
| `docs/*_TRAINING_DATA.md` | `data/curricula/` | benign dataset guides |

## LEAVE BEHIND (never port)
- `my_son/agent/tools.py` (botnet/DDoS/slowloris), `my_son/sync/swarm.py`
  (`swarm_botnet_deploy`), `my_son/brain/offensive_ops.py`, `my_son/network/`,
  `my_son/osint/`, `cyber_arsenal_tools/`.
- Offensive docs: `FULL_CAPABILITIES`, `NETWORK_SECURITY`, `IMPLEMENTATION_GUIDE`,
  `COMPLETE_SECURITY_SUITE`, `SECURITY_TRAINING_DATA`, `VULNERABILITY_SCANNER`,
  `.agent/rules/arkis_v2_context.md`.
- Agent-manipulation: `.agent/rules/rules.md`, `SOVEREIGN_BRAIN.md`.

## UNVERIFIED (port only after it earns its place by running)
- `brain/urm_agi.py` (runs, but mixes real cores with "consciousness"/math theatre),
  `jepa_embeddings.py`, `active_inference.py`, `test_time_compute.py`,
  `learning/` (SEAL), `self_improvement/` — LLM-orchestration, inert without a backend.
