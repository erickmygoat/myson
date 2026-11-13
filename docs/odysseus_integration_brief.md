> **Historical design document (October 2026).** Written when the model was Qwen3.5-9B served
> by Ollama. The model is now Gemma 4 26B-A4B through the GGUF backend; the design stands.
> Current plan and status: [ROADMAP.md](ROADMAP.md).

# Odysseus: what it is, backend interface, integration options, install path

*Sourced from web search (news coverage + the odysseusai.dev setup docs), not direct repo inspection. Items flagged **[verify on repo]** shifted between sources and should be checked against the live `LICENSE`/README before you rely on them.*

## (a) What Odysseus is
- **A self-hosted AI workspace** by Felix Kjellberg (PewDiePie): one browser UI for chat, autonomous agents, deep research, a document editor, email/calendar/notes, side-by-side model comparison, and persistent memory. Local-first, privacy-first, no telemetry.
- **Repo:** currently `github.com/odysseus-dev/odysseus` (community org; moved there from the original `github.com/pewdiepie-archdaemon/odysseus` around Aug–Sep 2026).
- **License:** **AGPL-3.0-or-later** per the most recent, GitHub-API-cited sources (Sep–Oct 2026). **[verify on repo]** — several launch-week (June 2026) articles reported MIT; treat the current license as AGPL-3.0 until you read the repo's LICENSE file. This matters only if you *fork* (see Option 3).
- **Release state:** launched May 31 2026; ~87k stars and ~320 contributors by Sep 2026. Active and fast-moving, so docs/UI labels drift. Note: **Ajax** (PewDiePie's own Qwen3.5-9B fine-tune for Odysseus) was still "coming soon"/unreleased as of early Oct 2026 — the model slot myson targets is currently empty.

## (b) Architecture & the model-backend interface (the important part)
- **Stack:** Python + FastAPI backend, Docker deployment, responsive/PWA web UI (default `http://localhost:7000`). Memory via **ChromaDB**. Agents run through **opencode** + **MCP** for tools. A **Cookbook** feature scans your hardware, recommends models, downloads GGUF weights, and serves them via **llama.cpp** (or vLLM/SGLang on CUDA/ROCm).
- **Backend interface = OpenAI-compatible, uniformly.** Odysseus does **not** have a bespoke model protocol. Every backend is added in **Settings → add model provider** as a **base URL + API key**, and the server just has to expose **`/v1/chat/completions`**. This is true even for local engines:
  - **Ollama** is registered as an OpenAI-compatible endpoint: `http://localhost:11434/v1` (native) or `http://host.docker.internal:11434/v1` (Odysseus-in-Docker).
  - **vLLM / llama.cpp / LM Studio / OpenRouter / OpenAI / etc.** are all "any server exposing `/v1/chat/completions`."
  - Self-run llama.cpp servers are auto-detected; everything else is a manual base-URL entry.
- **Implication for myson:** you do **not** need to speak any Odysseus-specific API. If myson exposes a standard `/v1/chat/completions` (+ `/v1/models`) endpoint, Odysseus treats it like any other provider. Agent/tool features additionally expect OpenAI **function-calling** fields (`tools`, `tool_choice` → `tool_calls`), so pass those through if you want Odysseus agents to drive myson.

## (c) Running myson (Qwen + recursive-reasoning wrapper) behind the Odysseus UI — 3 options

| Option | What you build | Pros | Cons |
|---|---|---|---|
| **1. OpenAI-compatible shim *(recommended)*** | Wrap myson's brain in a FastAPI server exposing `POST /v1/chat/completions` + `GET /v1/models`. Internally it calls Ollama (Qwen3.5-9B) for generation and layers the recursive-reasoning loop + memory + SEAL. Add its base URL in Odysseus Settings. | No fork; fully decoupled; survives Odysseus upgrades; myson stays your own licensed code; works with any OpenAI-compatible client, not just Odysseus. | You must implement the chat-completions schema faithfully, including **SSE streaming** and (for agents) tool-call passthrough. |
| **2. Sit beside Ollama / thin proxy** | Same shim, but packaged to mimic the Ollama endpoint (or proxy to it), so Odysseus "sees" myson as just another local model next to Qwen. | Lowest friction if you want it to appear in the normal local-model list; minimal new surface. | It's really Option 1 with different framing; proxying Ollama's exact response shape adds a translation layer and little benefit over a clean `/v1` server. |
| **3. Fork Odysseus, add a native provider** | Add myson as a first-class backend inside the Odysseus Python/FastAPI codebase. | Deepest integration (agent loop, Cookbook, memory internals); UI can expose myson-specific controls. | Heaviest; you track a fast-moving upstream; **AGPL-3.0 copyleft obligations** attach to your distributed fork. Only worth it if the OpenAI API shape genuinely can't express what you need. |

**Recommendation: Option 1.** Because Odysseus's entire provider model is "any `/v1/chat/completions` endpoint," the shim is the path of least resistance and keeps myson's codebase independent. It also fits what myson already is — a wrapper around `qwen3.5:9b` via Ollama; you're just adding an HTTP layer in front of the existing brain. (Keeping myson a separate process across the network boundary also sidesteps the AGPL question that a fork raises — not legal advice; confirm the license.)

## (d) Install / run path — Windows + 8 GB VRAM
1. **Install prerequisites:** Git; **Docker Desktop** (easiest on Windows) *or* Python 3.x for the native route; and **Ollama** for Windows.
2. **Pull a fitting model:** `ollama pull qwen3.5:9b`. At ~9B, run a **Q4_K_M GGUF (~5.5–6 GB)** so it fits 8 GB VRAM with a few-K context (estimate — tune quant/context to your card). This is the same base myson already targets.
3. **Get Odysseus:**
   - *Docker:* `git clone https://github.com/odysseus-dev/odysseus` → `cd odysseus` → `cp .env.example .env` → `docker compose up -d --build`.
   - *Native:* `python -m venv venv` → activate → `pip install -r requirements.txt` → `python setup.py`.
   - Open **`http://localhost:7000`**.
4. **Wire up the backend in Settings → add model provider (base URL + key):**
   - Plain Qwen first, to confirm the loop works: `http://host.docker.internal:11434/v1` (Odysseus in Docker) or `http://localhost:11434/v1` (native). The `/v1` suffix is required.
   - **Then point it at myson:** run your shim on the host (e.g. `:8000`) and add `http://host.docker.internal:8000/v1` (Docker) or `http://localhost:8000/v1` (native). Select the myson model in the picker.
5. **VRAM reality:** 8 GB runs *one* 9B-class model at Q4 comfortably. Since myson and Ollama share the **same** Qwen weights via one Ollama instance, you're not doubling VRAM — the recursive-reasoning overhead is extra *compute/latency per turn*, not extra GPU memory. Keep context modest.

## Honest caveats & the myson boundary
- **Unverified specifics:** exact Settings UI labels, the current license (MIT→AGPL discrepancy), and default ports come from secondary docs, not repo inspection — confirm against the live repo.
- **Do not copy Ajax's approach wholesale.** Reporting consistently states Ajax uses **refusal ablation / abliteration (Heretic)** plus GRPO tool-use RL for a "less restricted" model. Copy the *integration pattern* (local 9B behind Odysseus via an OpenAI-compatible endpoint) — **not** the abliteration, which is exactly the boundary myson has chosen to stay inside. myson's edge is the reasoning + memory + SEAL wrapper and a candid persona, not stripped safety.
