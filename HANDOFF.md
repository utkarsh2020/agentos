# AgentOS — Project Handoff & Continuity Document

**For:** AI-assisted development continuation (Kilo Code / Claude)  
**Version at handoff:** v0.2.1  
**Test status:** 40/40 passing  
**Total source:** ~6,100 lines across 30 files, zero external dependencies

---

## 1. What AgentOS Is

AgentOS is an **Agentic AI Operating System** — a lightweight, self-hosted runtime that orchestrates LLM-powered agents to execute tasks automatically. It is designed to run on a **Raspberry Pi Zero W** (ARMv6, 512 MB RAM) as a primary constraint, which means:

- **Zero external Python dependencies** — pure `stdlib` only (`sqlite3`, `asyncio`, `http.server`, `urllib.request`, `hmac`, `hashlib`, `json`, `pathlib`, etc.)
- **Single asyncio process** — no Docker, no message brokers, no npm
- **SQLite** for all persistent state (WAL mode)
- **Hand-rolled JWT**, **HMAC-XOR vault encryption** (AES-256-GCM when `cryptography` available)

The system is NOT a chatbot. It is a **task automation pipeline** where:
- **Projects** are the top-level unit, defined in TOML files
- **Tasks** are DAG nodes within a project
- **Agents** are LLM calls that execute within a task
- **Skills** are tools agents call (web scrape, HTTP, file I/O, etc.)

---

## 2. Architecture — Exact Layer Map

```
┌─────────────────────────────────────────────────────────────────┐
│  INTERFACE LAYER                                                 │
│  bin/agent (CLI)  ·  agentd/api/server.py (REST :7777)          │
│  static/dashboard.html (served at /)                             │
├─────────────────────────────────────────────────────────────────┤
│  AI LAYER                                                        │
│  agentd/ai/llm.py          — Anthropic, OpenAI, Ollama via      │
│                               urllib.request (no httpx/requests) │
│  agentd/ai/memory.py       — ShortTermMemory (LRU FIFO) +       │
│                               LongTermMemory (64-dim cosine/SQLite)│
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER                                             │
│  agentd/core/agent.py      — AgentRunner: the agent executor    │
│  agentd/core/scheduler.py  — DAG resolver + CronScheduler       │
│  agentd/core/bus.py        — EventBus: asyncio pub/sub          │
│  agentd/core/loader.py     — TOML project file → DB import      │
├─────────────────────────────────────────────────────────────────┤
│  INTEGRATION LAYER                                               │
│  agentd/integrations/builtin_skills.py  — 7 built-in skills     │
│  skills/*/handler.py                    — custom plugin skills   │
├─────────────────────────────────────────────────────────────────┤
│  CORE RUNTIME                                                    │
│  agentd/core/store.py      — SQLite CRUD (WAL, per-thread conn)  │
│  agentd/core/config.py     — env + TOML config, provider setup  │
│  agentd/security/vault.py  — AES/XOR encryption, JWT, RBAC      │
├─────────────────────────────────────────────────────────────────┤
│  BOOT                                                            │
│  agentd/daemon.py          — asyncio boot: DB→skills→API→cron   │
└─────────────────────────────────────────────────────────────────┘
```

**Dependency direction:** each layer only imports from layers below it. `agent.py` imports from `bus.py`, `store.py`, `llm.py`, `memory.py`, `vault.py`. Nothing in `store.py` imports from `agent.py`. This is strict and must be maintained.

---

## 3. Every File — Purpose and Key Decisions

### `agentd/core/config.py`
- Singleton `AgentOSConfig` dataclass loaded from env vars + optional `agentd.toml`
- `BASE_DIR` is set from `AGENTD_VAULT_PASS` env or defaults to the source tree root
- `LLMProviderConfig` list built from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_MODEL` env vars
- Ollama is only enabled when `OLLAMA_MODEL` is explicitly set
- **Pi Zero constraint:** `max_concurrent_agents=3`, `agent_memory_limit_mb=64`, `short_term_capacity=50`

### `agentd/core/store.py`
- All state in a single SQLite file at `BASE_DIR/agentd.db`
- One connection per thread (`threading.local`), WAL mode, `cache_size=-4096` (4 MB cap)
- Tables: `projects`, `tasks`, `agents`, `agent_messages`, `events`, `memory`, `scheduled_jobs`, `users`, `skills`
- Key helper functions: `insert()`, `update()`, `fetch_one()`, `fetch_where()`, `raw_query()`, `append_event()`
- `insert()` auto-detects `updated_at` column via `PRAGMA table_info` before including it — important: not all tables have `updated_at`
- `scheduled_jobs` has no `created_at` column — must use `raw_query` for inserts into it

### `agentd/core/bus.py`
- `EventBus` class: `asyncio.Queue`-based pub/sub, wildcard `"*"` topic subscription
- `emit_nowait()` for fire-and-forget from sync code
- `request()` for request/reply pattern (private reply topic)
- Events also persisted to SQLite `events` table for audit log
- Global singleton via `get_bus()`
- Well-known topics in `Topics` class constants (e.g. `Topics.AGENT_DONE`, `Topics.TASK_FAILED`)

### `agentd/core/agent.py`
- `AgentConfig` dataclass: id, task_id, project_id, role, model, provider, system_prompt, skills, max_tokens, temperature, max_retries, timeout
- `AgentRunner.run()`: the full agent lifecycle — load secrets → recall memory → build prompt → LLM call → skill calls → persist output → emit events
- **Skill call protocol:** agent output is scanned for `{"action": "skill_call", "skill": "...", "params": {...}}` JSON blocks. Up to 5 skill calls per turn before forced termination.
- `_extract_action()`: parses action blocks from LLM output (handles inline JSON, code-fenced JSON)
- `_build_system_prompt()`: assembles system prompt from role template + skills instructions + memory recall results + task context
- Three roles: `executor` (default), `planner` (outputs JSON DAG), `critic` (outputs `{"approved": bool, "feedback": "..."}`)
- Retry: exponential backoff with `asyncio.wait_for` timeout
- **Important:** `call_llm` is imported at the top of this file. Mock it as `agentd.core.agent.call_llm` in tests, not `agentd.ai.llm.call_llm`

### `agentd/core/scheduler.py`
- `get_ready_tasks(project_id)`: returns tasks whose `depends_on` deps are all `done` and status is `pending`. Resolves deps by name OR id.
- `run_task(task)`: runs all agents in a task sequentially (Pi Zero: no parallelism). Collects outputs into `combined_output` dict.
- `run_project(project_id)`: iterates DAG in a while loop — find ready tasks → run them → check completion. Max iterations = `len(tasks) * 2 + 1`.
- `CronScheduler`: polls `scheduled_jobs` every 10 seconds. Fires `run_project()` as async tasks.
- `next_run_time()`: parses `@every Ns/m/h/d`, `@daily`, `@hourly`, `@weekly`, `@once`

### `agentd/core/loader.py`
- `import_project(path)`: parses TOML → upserts `projects` table → deletes+re-creates `tasks` → registers `scheduled_jobs`
- Requires Python 3.11+ for `tomllib` (stdlib). Python 3.10 users must create projects via API.
- Agent configs are resolved by `id` from the `[[agents]]` array, matched to `tasks.*.agents = [...]`

### `agentd/ai/llm.py`
- `call_llm(messages, provider, model, max_tokens, temperature, fallback)` — single entry point
- Provider routing: `provider="auto"` tries `["anthropic", "openai", "ollama"]` in order
- Model override: `"anthropic/claude-3-5-haiku-20241022"` format splits on `/` to override model string
- All HTTP via `urllib.request.urlopen` — no `requests` library
- `LLMMessage` dataclass: `role` (system/user/assistant), `content`
- `LLMResponse` dataclass: `content`, `model`, `provider`, `input_tokens`, `output_tokens`, `latency_ms`
- `LLMError` exception with `provider`, `message`, `status_code`
- Anthropic: separates `system` messages, uses `x-api-key` header, `anthropic-version: 2023-06-01`
- OpenAI: standard `Authorization: Bearer` header, `/v1/chat/completions`
- Ollama: `POST /api/chat`, `stream: false`, `options.num_predict` for max_tokens

### `agentd/ai/memory.py`
- `ShortTermMemory(agent_id, capacity)`: FIFO list, system messages always preserved, non-system trimmed to `capacity`
- `ShortTermMemory.add()`: best-effort SQLite persist — uses try/except because agent row may not exist in tests
- `LongTermMemory(project_id)`: cosine similarity over `memory` table in SQLite
- `_embed(text)` → 64-dim float list: character n-gram hashing (MD5), L2 normalised. Deterministic. No ML deps.
- `_cosine(a, b)`: standard dot product / (|a| * |b|)
- `LongTermMemory.recall(query, top_k, min_score)`: embeds query, loads last 500 memories from DB, scores all, returns top_k above threshold
- Upgrade path: replace `_embed()` with `POST /api/embeddings` to Ollama for real semantic embeddings

### `agentd/api/server.py`
- `AgentOSHandler(BaseHTTPRequestHandler)` — the HTTP handler
- `@route(method, path)` decorator registers handlers in `_routes` dict
- Parametric routes: `<id>` style segments matched by splitting path on `/`
- `_serve_static()`: serves `static/` with path traversal protection. Falls back to source-tree `static/` if `BASE_DIR/static/` doesn't exist (dev mode)
- `_run_async(coro)`: submits coroutine to the main event loop from the HTTP server thread via `asyncio.run_coroutine_threadsafe`
- `_get_arch()`: `platform.machine()` → human label, included in `/api/status`
- `/api/status` returns: `status`, `version`, `arch`, `providers`, `uptime_s`, `db`
- API server runs in a daemon `Thread`, not the asyncio loop

### `agentd/security/vault.py`
- `hash_password(password)` → `salt:pbkdf2_hex` (100k iterations). Legacy sha256 hashes also verified for backward compat.
- `issue_token(user_id, username, role)` → HS256 JWT string (base64url header.payload.sig)
- `verify_token(token)` → claims dict or `None`
- `_get_master_key()`: loads or generates 32-byte master key. Stored as `vault/master.key` = `salt(16) + XOR(master, pbkdf2(VAULT_PASS, salt))`
- `_encrypt(plaintext)` → `"aes:..."` (AES-256-GCM if `cryptography` installed) or `"xor:..."` (HMAC-XOR fallback)
- `set_secret(project_id, key, value)`: writes `vault/<project_id>/<key>.enc`
- `get_secret(project_id, key)`: reads vault file, or falls back to `AGENTD_SECRET_<PROJECT>_<KEY>` env var, or plain env var named `key`

### `agentd/daemon.py`
- `boot()` coroutine: `init_db()` → `register_builtin_skills()` → `_load_plugin_skills()` → `start_api_server()` → `CronScheduler` task → heartbeat task → `_shutdown_event.wait()`
- `_load_plugin_skills()`: scans `skills/*/handler.py`, imports each module, auto-registers if `SKILL_ID` and `handle()` defined
- `_arch_label()`: `platform.machine()` → human string (ARMv6/ARMv7/ARM64/x86_64)
- `_build_banner()`: dynamic banner with detected arch and Python version (printed at boot)
- Handles `SIGINT` and `SIGTERM` for graceful shutdown

### `agentd/integrations/builtin_skills.py`
Seven skills, all stdlib-only:
- `http.call`: GET/POST/etc with custom headers and body
- `web.scrape`: fetch URL, strip HTML to plain text (regex-based, no BS4)
- `fs.write`: write to `/tmp/` or `/var/lib/agentd/projects/` only
- `fs.read`: read file content with byte cap
- `system.shell`: whitelisted prefix check, then `subprocess.run` with timeout
- `memory.remember`: calls `LongTermMemory.remember()`
- `memory.recall`: calls `LongTermMemory.recall()`

### `bin/agent`
CLI with `argparse`. Commands: `start`, `login`, `status`, `project`, `task`, `agent`, `secret`, `skill`, `run`, `monitor`, `logs`, `user`.
- Auth token saved to `~/.agentd_token`
- `monitor` command: polls every 2s, clears screen (`\033[2J\033[H`), draws task/agent status table
- `agent start` imports and calls `agentd.daemon.main()` directly (no subprocess)
- `agent run <file.toml>`: calls `import_project()` directly (no daemon needed for loading), then POSTs to `/api/projects/<id>/run`

### `static/dashboard.html`
- 1024 lines, zero framework dependencies
- `IBM Plex Mono` + `IBM Plex Sans` from Google Fonts
- Five pages: Overview, Projects, Agents, Events, Skills
- State: `S = {token, user, currentProject, currentAgent, pollTimer}`
- Auto-polls `loadAll()` every 4 seconds via `setInterval`
- `#si-arch` element updated from `/api/status.arch`
- `buildLayers(tasks)`: topological DAG layering for visual display
- Token persisted to `localStorage` across page reloads

### `deploy/install.sh`
- Detects `uname -m` → sets `ARCH_LABEL`, `ARCH_WARN` (warn if ARMv6)
- Memory limits per arch: ARMv6 → 180M, ARMv7 → 320M, ARM64/x86_64 → 512M
- Agent count defaults: ARMv6 → 1, ARMv7 → 2, others → 4
- Generates `agentd.env` with arch header comment
- Writes systemd unit file dynamically with correct `$PYTHON` path and `MemoryMax`

---

## 4. Data Model (SQLite Schema)

```sql
projects        id, name, description, status, schedule, config_toml, created_at, updated_at
tasks           id, project_id, name, status, depends_on(JSON), config(JSON), output(JSON), error, started_at, finished_at, created_at
agents          id, task_id, project_id, role, model, provider, state, system_prompt, output, error, token_usage, started_at, finished_at, created_at
agent_messages  id, agent_id, role, content, created_at
events          id, topic, source_id, payload(JSON), created_at
memory          id, project_id, agent_id, content, embedding(JSON float[64]), importance, created_at
scheduled_jobs  id, project_id, schedule, last_run, next_run, enabled
users           id, username, password_hash, role, created_at
skills          id, name, version, enabled, config(JSON), installed_at
```

**Status enums:**
- `projects.status`: `idle | running | paused | archived`
- `tasks.status`: `pending | running | done | failed | skipped`
- `agents.state`: `pending | running | done | failed`
- `users.role`: `read_only | skill | agent | project_owner | admin`

---

## 5. Key Patterns and Conventions

### Skill call protocol
Agents signal tool use by emitting a JSON block in their response:
```
{"action": "skill_call", "skill": "web.scrape", "params": {"url": "..."}}
```
`_extract_action()` in `agent.py` finds this block anywhere in the response text. After execution, result is fed back as a `user` message: `"Skill result for web.scrape:\n{...}"`. Up to 5 skill calls per turn; after that the turn is forced to complete.

### Agent-to-agent communication
Agents **never call each other directly**. All cross-agent data flows through the DAG output chain: agent A's output is stored in `tasks.output`, which is collected by `_collect_context()` in `scheduler.py` and passed as the `context` string to the next task's agents.

### Error handling philosophy
- LLM errors: retry with exponential backoff (2^attempt seconds), up to `max_retries`
- Task failure: marks task `failed`, emits `task.failed` event, marks remaining pending tasks `skipped`
- Skill errors: returned as `{"error": "..."}` dict — agent receives this and decides what to do
- DB errors in `ShortTermMemory.add()`: silently ignored (best-effort persistence)

### Event bus usage
```python
bus = get_bus()
await bus.publish(Message("task.done", {"task_id": "...", "name": "..."}, from_id="scheduler"))
q = await bus.subscribe("agent.done")
msg = await asyncio.wait_for(q.get(), timeout=30)
```

### Config singleton reset in tests
```python
import agentd.core.config as cfg_mod
cfg_mod._config = None  # force re-read with new AGENTD_BASE
```
This pattern is used in every test that needs an isolated temp directory.

---

## 6. What Is Complete (v0.2.1)

| Component | Status | Notes |
|---|---|---|
| Core daemon (`agentd/daemon.py`) | ✅ Complete | Boot, signals, plugin loader |
| SQLite state store | ✅ Complete | All tables, WAL, thread-safe |
| Async event bus | ✅ Complete | pub/sub, wildcard, request/reply |
| LLM abstraction (Anthropic/OpenAI/Ollama) | ✅ Complete | Pure urllib, fallback chain |
| Short-term memory | ✅ Complete | LRU FIFO, system msg preserved |
| Long-term vector memory | ✅ Complete | 64-dim n-gram, cosine/SQLite |
| Agent executor | ✅ Complete | Lifecycle, skill calls, retry |
| Task DAG scheduler | ✅ Complete | Dependency resolution, cron |
| TOML project loader | ✅ Complete | Requires Python 3.11+ |
| REST API (20+ endpoints) | ✅ Complete | stdlib http.server |
| JWT auth + RBAC | ✅ Complete | HS256, 5 roles |
| Secrets vault | ✅ Complete | AES-GCM / HMAC-XOR, PBKDF2 |
| Built-in skills (7) | ✅ Complete | http, scrape, fs, shell, memory |
| Plugin skill loader | ✅ Complete | Auto-discover skills/*/handler.py |
| CLI tool | ✅ Complete | All commands, live monitor |
| Web dashboard | ✅ Complete | 5 screens, live data, single file |
| Static file serving | ✅ Complete | Path-traversal protected |
| Multi-arch support (32/64-bit) | ✅ Complete | ARMv6/v7/ARM64/x86_64 |
| systemd service + installer | ✅ Complete | Arch-aware memory limits |
| Test suite | ✅ Complete | 40 tests, unit + integration |
| README | ✅ Complete | 798 lines, full reference |
| .gitignore, Makefile, CONTRIBUTING | ✅ Complete | Standard repo scaffolding |

---

## 7. What Is NOT Yet Built (Roadmap)

This section is the primary guide for the next development phase. Items are ordered by dependency — build top-to-bottom.

### 7.1 Communication Skills (v2 Priority 1)

**Telegram skill** (`skills/telegram/handler.py`) — ✅ Complete
- Bot API: `POST https://api.telegram.org/bot{TOKEN}/sendMessage`
- Secrets needed: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `handle(params, secrets)`: Send text messages
- `handle_photo(params, secrets)`: Send photos
- `handle_get_updates(params, secrets)`: Long poll for incoming messages
- `handle_set_webhook(params, secrets)`: Register webhook URL
- `handle_delete_webhook(params, secrets)`: Remove webhook

**WhatsApp skill** (`skills/whatsapp/handler.py`) — ✅ Complete
- WhatsApp Business Cloud API (Meta): `POST https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages`
- Secrets: `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_ID`
- `handle(params, secrets)`: Send text, image, document, audio, video, sticker messages
- `handle_list(params, secrets)`: Send interactive list messages
- `handle_template(params, secrets)`: Send template messages
- `handle_mark_seen(params, secrets)`: Send read receipts

**Slack skill** (`skills/slack/handler.py`) — ✅ Complete
- `POST https://slack.com/api/chat.postMessage`
- Headers: `Authorization: Bearer {SLACK_TOKEN}`
- Secrets: `SLACK_TOKEN`, optionally `SLACK_DEFAULT_CHANNEL`
- `handle(params, secrets)`: Send messages with text, blocks, attachments
- `handle_conversations_list(params, secrets)`: List channels
- `handle_users_info(params, secrets)`: Get user info
- `handle_webhook(params, secrets)`: Send via incoming webhook
- `handle_update(params, secrets)`: Update existing message
- `handle_delete(params, secrets)`: Delete message

**Gmail skill** (`skills/gmail/handler.py`) — ✅ Complete
- OAuth2 flow: exchange refresh token for access token via `POST https://oauth2.googleapis.com/token`
- Secrets: `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`
- `handle(params, secrets)`: Route to sub-operations based on mode
- `handle_send(params, secrets)`: Send emails with text/HTML, attachments
- `handle_list(params, secrets)`: List emails with query support
- `handle_get(params, secrets)`: Get specific email details
- `handle_draft(params, secrets)`: Create draft emails

### 7.2 Multi-Agent Collaboration Patterns (v2 Priority 2)

The current system runs agents sequentially within a task. True multi-agent collaboration needs:

**Planner-Executor pattern** (partially done — `role: planner` exists but isn't wired to auto-spawn subtasks)
- Planner agent outputs: `[{"name": "subtask-1", "description": "...", "agent": {...}}, ...]`
- Scheduler should parse this and dynamically create child tasks in the DB
- Implement in `scheduler.py`: `run_planner_task()` that calls `run_task()` for each dynamic subtask

**Critic-feedback loop** (partially done — `role: critic` exists but output isn't fed back to executor)
- After executor runs, critic reviews output
- If `approved: false`, re-run executor with critic's feedback appended to the prompt
- Max feedback cycles: configurable, default 2
- Implement in `scheduler.py`: `run_task_with_critic()` that wraps `run_task()`

**Agent-to-agent messaging via bus** (bus exists, not used for agent↔agent yet)
- An agent should be able to `await bus.request("agent.<target_id>", {"question": "..."})` and receive a response
- Requires: agents need to be subscribed to their own topic while running
- The `AgentRunner._one_turn()` loop needs a concurrent listener

### 7.3 Web Dashboard Enhancements (v2 Priority 3)

**Missing screens:**
- `Settings` page: live-edit `AGENTD_MAX_AGENTS`, LLM provider status, vault passphrase change
- `Users` page: create/delete users, change roles (POST /api/users — endpoint not yet built)
- `Memory browser`: query long-term memory per project, delete individual memories

**Missing features on existing screens:**
- Projects screen: inline TOML editor (CodeMirror or plain `<textarea>`) for project definition
- Projects screen: `+ Add task` form within project detail view
- Agents screen: real-time token count streaming (SSE or WebSocket)
- Event log: auto-refresh without full reload; highlight new rows

**SSE (Server-Sent Events)** for real-time updates without polling:
- Add `GET /api/events/stream` endpoint in `server.py` that holds the connection open
- Emit events from the bus in real time
- Dashboard subscribes with `new EventSource('/api/events/stream')`
- This is the single biggest UX improvement — eliminates the 4-second poll lag

### 7.4 API Gaps (v2 Priority 3)

These endpoints are referenced in the README or dashboard but not yet implemented:

| Endpoint | Method | Description |
|---|---|---|
| `/api/users` | GET | List users | ✅ Complete |
| `/api/users` | POST | Create user | ✅ Complete |
| `/api/users/<id>` | DELETE | Delete user | ✅ Complete |
| `/api/users/<id>/role` | PUT | Change user role | ✅ Complete |
| `/api/users/<id>/password` | PUT | Change password | ✅ Complete |
| `/api/projects/<id>/memory` | GET | List project memories |
| `/api/projects/<id>/memory/<id>` | DELETE | Delete a memory |
| `/api/projects/<id>/tasks/<id>` | DELETE | Delete a task |
| `/api/projects/<id>/schedule` | PUT | Update schedule | ✅ Complete |
| `/api/events/stream` | GET | SSE stream |

### 7.5 Skill SDK & Registry (v2 Priority 4)

**Formal SDK** (`agentd/sdk.py`):
- Currently there is an informal convention (`SKILL_ID`, `def handle(params, secrets) -> dict`)
- A proper SDK should export: `SkillHandler` base class, `register_skill` decorator, `SkillResult` dataclass, `SkillError` exception
- Parameter schema validation: define params with types and defaults, validate on call
- Versioning: `SKILL_VERSION = "1.0.0"` in handler files

**Skill registry** (DB table `skills` exists but is unused):
- On startup, populate the `skills` table from loaded handlers
- Expose `GET /api/skills/<id>` for skill details
- Enable/disable skills without restart via `PUT /api/skills/<id>` with `{"enabled": bool}`

**Skill marketplace** (v3):
- A simple index at a known URL listing available skills
- `agent skill install <skill-id>` downloads handler.py into `skills/<id>/`
- Skills are just Python files, so installation is just a file copy

### 7.6 Improved Memory (v2 Priority 4)

**Ollama embeddings integration:**
- When `OLLAMA_BASE_URL` is set, replace `_embed()` in `memory.py` with a call to `POST {OLLAMA_BASE_URL}/api/embeddings`
- Model: `nomic-embed-text` (2GB but much better semantic search)
- Pi Zero W: Ollama not supported (ARMv6). Pi Zero 2W and up: use `all-minilm:l6-v2` (45MB)
- Pattern: feature-flag based on `AGENTD_EMBEDDING_MODEL` env var

**Memory importance decay:**
- Currently `importance` is static. Add time decay: `effective_score = cosine * importance * exp(-age_days / 30)`
- Implement in `LongTermMemory.recall()` when computing scores

**Memory summarisation:**
- When project memory count exceeds threshold (e.g. 1000 entries), run a summariser agent
- Summariser compresses related memories into single higher-importance memories
- Schedule as a background task every N project runs

### 7.7 Observability (v2 Priority 5)

**Structured logging to files:**
- Currently all logs go to stdout/journald
- Add `LOG_DIR/agentd.jsonl` append-only structured log with full context
- Log format: `{"ts": 1234567890.123, "level": "INFO", "component": "scheduler", "event": "task.done", "task_id": "...", "duration_ms": 4321}`

**Prometheus metrics endpoint:**
- `GET /api/metrics` returning Prometheus text format
- Counters: `agentd_agents_total{state="done"}`, `agentd_tokens_total{provider="anthropic"}`
- Gauges: `agentd_projects_running`, `agentd_memory_entries{project_id="..."}`
- Histograms: `agentd_agent_duration_seconds`, `agentd_llm_latency_seconds{provider="..."}`

**OpenTelemetry trace export:**
- Events already carry `trace_id`; add full span hierarchy
- Export to `jaeger` or `zipkin` via OTLP HTTP (optional, configured via env)

### 7.8 Security Hardening (v2 Priority 5)

**Token refresh:**
- JWTs expire in 1 hour. CLI needs `POST /api/auth/refresh` to get a new token without re-login
- Dashboard auto-refreshes token before expiry

**Rate limiting:**
- API server currently has no rate limiting
- Add a simple per-IP token bucket in `AgentOSHandler._dispatch()` using a `defaultdict(deque)` in memory

**Audit log integrity:**
- Currently audit log (SQLite `events` table) is mutable
- Add an append-only JSONL file at `LOG_DIR/audit.jsonl` that is never deleted
- Hash chain: each event includes SHA256 of the previous event's hash

**Agent sandboxing** (Linux only, v3):
- Currently agents run in the same process as the daemon
- For stronger isolation: `os.fork()` + `os.setuid()` to an unprivileged user
- For stronger isolation: Linux namespaces via `unshare` syscall before exec
- For Pi Zero (no Docker): `seccomp` filter via `ctypes` to limit syscalls

### 7.9 Boot Image / OS Distribution (v3)

**Buildroot or custom Pi OS image:**
- `debootstrap` Debian 12 Bookworm minimal (armhf for Pi Zero W)
- Install: Python 3.11, AgentOS source, systemd unit
- Pre-configured `agentd.toml.example` and env template
- First-boot script: prompt for WiFi credentials + API key, then start daemon
- Image size target: <2 GB (fits on 4 GB SD card)
- Build script: `deploy/build-image.sh` using `debootstrap` + `losetup` + `dd`

### 7.10 Self-Improving Agents (v3)

**Feedback loop:**
- After project completion, a "reviewer" agent analyses all agent outputs and critic feedback
- Identifies prompt patterns that consistently produce poor output
- Writes improvement suggestions to long-term memory with high importance
- Next run: system prompt injected with the improvement notes

**LoRA fine-tuning pipeline (Ollama path):**
- Collect (prompt, good_output, bad_output) triples from critic feedback
- Periodically export to JSONL training data format
- Trigger `ollama create` to fine-tune a model variant
- Switch the project to use the fine-tuned model

---

## 8. How to Resume Development with Kilo Code

### Step 1: Clone and verify baseline
```bash
git clone https://github.com/YOUR_USERNAME/agentos
cd agentos
export ANTHROPIC_API_KEY=sk-ant-...
python3 tests/test_suite.py    # must be 40/40
python3 agentd/daemon.py &
open http://localhost:7777
```

### Step 2: Pick a task from section 7

Each item in section 7 is designed to be built independently without touching existing code except for adding one registration call or one import.

**Easiest first items (warmup):**
1. `skills/telegram/handler.py` — 30 lines, just `urllib.request` + vault secrets
2. `skills/slack/handler.py` — same pattern
3. `GET /api/users` endpoint in `server.py` — 5 lines using existing `store.fetch_all("users")`
4. `GET /api/projects/<id>/memory` endpoint — 5 lines using `store.raw_query`

**Medium items:**
5. SSE stream endpoint — new pattern but contained in `server.py`
6. Planner→executor subtask spawning — modify `scheduler.py:run_task()`
7. Ollama embeddings — modify `memory.py:_embed()` behind a feature flag

### Step 3: Test discipline

For every new skill:
```python
# In tests/test_suite.py, add to AgentOSTestCase:
def test_skill_telegram_send(self):
    from agentd.integrations.builtin_skills import skill_telegram_send  # or from skills.telegram.handler
    # Test with missing token
    result = skill_telegram_send({"chat_id": "123", "text": "hello"}, {})
    self.assertIn("error", result)
    # Test response structure (mock the HTTP call)
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true, "result": {"message_id": 42}}'
        mock_url.return_value.__enter__ = lambda s: mock_resp
        mock_url.return_value.__exit__ = MagicMock(return_value=False)
        result = skill_telegram_send({"chat_id": "123", "text": "hello"}, {"TELEGRAM_BOT_TOKEN": "fake"})
        self.assertTrue(result.get("ok"))
```

### Step 4: Architecture rules to enforce

When Kilo Code generates new code, verify these invariants:

1. **No external imports.** Any `import requests`, `import httpx`, `import flask`, `import fastapi` is a violation.
2. **Layer direction.** Nothing in `store.py`, `bus.py`, `config.py` may import from `agent.py` or `scheduler.py`.
3. **Skill signature.** Every skill handler must be `def handle(params: dict, secrets: dict) -> dict` or `async def handle(params: dict, secrets: dict) -> dict`.
4. **Event emission.** Every significant state change must `append_event()` or `bus.publish()`. Check that new scheduler code emits to `Topics`.
5. **Pi Zero budget.** No new code should allocate >10 MB per operation. No loading large libraries.
6. **Test count.** After any change, `python3 tests/test_suite.py` must show ≥40 tests passing.

---

## 9. Environment Variable Quick Reference

```bash
# Required (at least one LLM provider)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-haiku-20241022     # default

OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com        # default; override for Groq/Together
OPENAI_MODEL=gpt-4o-mini                      # default

OLLAMA_BASE_URL=http://localhost:11434        # default
OLLAMA_MODEL=llama3                           # must set to enable Ollama

# Daemon
AGENTD_BASE=/var/lib/agentd                   # where DB, vault, logs live
AGENTD_HOST=0.0.0.0
AGENTD_PORT=7777
AGENTD_LOG_LEVEL=INFO
AGENTD_JWT_SECRET=<32-hex-chars>              # set for persistence across restarts
AGENTD_VAULT_PASS=<passphrase>               # MUST set before first run
AGENTD_MAX_AGENTS=3                           # 1-2 for Pi Zero W
```
