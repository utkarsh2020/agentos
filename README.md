# Kriya

> Agentic AI Operating System — lightweight, modular, Pi Zero native.

A self-hosted runtime for orchestrating AI agents, task pipelines, and multi-channel integrations. Runs on ARM 32-bit (Pi Zero W), ARM 64-bit (Pi 4/5), and x86_64 Linux. Zero external Python dependencies — pure stdlib only.


## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Installation](#installation) — see also **[INSTALL.md](INSTALL.md)** for comprehensive guide
  - [Development (any Linux/macOS)](#development-any-linuxmacos)
  - [Raspberry Pi (production)](#raspberry-pi-production)
  - [Systemd service](#systemd-service)
- [Configuration](#configuration)
- [Web dashboard](#web-dashboard)
- [CLI reference](#cli-reference)
- [Project files (TOML)](#project-files-toml)
- [Agent roles](#agent-roles)
- [Built-in skills](#built-in-skills)
- [Writing a custom skill](#writing-a-custom-skill)
- [Memory system](#memory-system)
- [Security](#security)
- [API reference](#api-reference)
- [Changelog](#changelog)

---

## What it does

Kriya lets you define **projects** — collections of **tasks** arranged as a dependency graph (DAG). Each task is executed by one or more **agents** (LLM-powered workers). Agents can use **skills** (tools) to interact with the outside world: scrape the web, read/write files, call APIs.

Everything runs in a single Python process. There is no Docker, no Node.js, no message broker to install.

**Example flow:**

```
Project: "weekly-newsletter"
  │
  ├─ Task: research          (agent: researcher — uses web.scrape)
  │
  ├─ Task: summarise         (agent: summariser — depends on research)
  │      depends_on: research
  │
  └─ Task: draft             (agents: writer + critic — depends on summarise)
         depends_on: summarise
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Interface layer   CLI (bin/agent)  ·  REST API :7777   │
│                    Web dashboard   (static/dashboard.html)│
├─────────────────────────────────────────────────────────┤
│  AI layer          LLM abstraction (Anthropic/OpenAI/    │
│                    Ollama)  ·  Prompt engine  ·  Memory  │
├─────────────────────────────────────────────────────────┤
│  Orchestration     Agent lifecycle  ·  Task DAG scheduler│
│                    Event bus (asyncio pub/sub)            │
├─────────────────────────────────────────────────────────┤
│  Integration       Skills / plugins  ·  Secrets vault    │
│                    http.call · web.scrape · fs · shell   │
├─────────────────────────────────────────────────────────┤
│  Core runtime      SQLite state  ·  RBAC  ·  Audit log   │
├─────────────────────────────────────────────────────────┤
│  OS                Python 3.11+  ·  stdlib only          │
│                    ARMv6/v7 32-bit · ARM64 · x86_64      │
└─────────────────────────────────────────────────────────┘
```

**Key design decisions:**

| Decision | Reason |
|---|---|
| Pure Python stdlib, no pip deps | Runs on any architecture Python 3.10+ supports — no compiled wheels |
| Single asyncio process | Fits in 512 MB RAM; no inter-process complexity |
| SQLite WAL-mode | Zero-setup persistent state, Pi SD-card friendly |
| 64-dim n-gram embeddings | No ML library needed for long-term memory; upgrade path to Ollama embeddings |
| `urllib.request` for LLM calls | No `requests`/`httpx` dependency |
| JWT HS256, hand-rolled | No `PyJWT` dependency; pure `hmac` + `base64` |

---

## Requirements

| Item | Minimum | Notes |
|---|---|---|
| Python | 3.11 | `tomllib` (stdlib) needed for TOML project files |
| RAM | 512 MB | Pi Zero W baseline; 1 GB+ recommended for parallel agents |
| Disk | 100 MB | Source + SQLite DB + logs |
| LLM provider | One of Anthropic / OpenAI / Ollama | At least one API key or local Ollama instance |
| OS | Any POSIX | Raspberry Pi OS Lite, Debian, Ubuntu, macOS |
| Architecture | ARMv6+ / x86_64 | See OS image table below for Pi board mapping |

Python 3.10 works if you do not use TOML project files (use the API/CLI to create projects instead).

---

## Quick start

```bash
# Clone / extract
git clone https://github.com/kriya/kriya   # or: tar xzf kriya-v0.2.0.tar.gz
cd kriya

# Set at least one LLM provider key
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY or OLLAMA_MODEL=llama3

# Start the daemon (foreground)
python3 kriya/daemon.py

# In a second terminal — run a project
python3 bin/agent run examples/newsletter.toml --follow

# Open the web dashboard
open http://localhost:7777      # or http://<pi-ip>:7777
```

On first boot, a random admin password is generated and saved to `first_run_credentials.txt` in the data directory. Read it, login, then change the password immediately:

```bash
cat first_run_credentials.txt                    # dev
sudo cat /var/lib/kriya/first_run_credentials.txt  # Pi / systemd install
agent login
agent user passwd admin   # file is deleted automatically after this
```

---

## Installation

### Development (any Linux/macOS)

No install step needed. Run directly from the source directory.

```bash
tar xzf kriya-v0.2.0.tar.gz
cd kriya

# Optional: add the CLI to your PATH
export PATH="$PATH:$(pwd)/bin"

# Start
export ANTHROPIC_API_KEY=sk-ant-...
python3 kriya/daemon.py
```

### Raspberry Pi (production)

#### Choose the right OS image

| Board | CPU | OS image to flash |
|---|---|---|
| Pi Zero W | ARMv6 32-bit | **Raspberry Pi OS Lite 32-bit** (Bookworm) |
| Pi Zero 2 W | ARMv7/ARMv8 | Raspberry Pi OS Lite 32-bit **or** 64-bit |
| Pi 3 | ARMv7/ARMv8 | Raspberry Pi OS Lite 32-bit or 64-bit |
| Pi 4 / Pi 5 | ARM64 | Raspberry Pi OS Lite 64-bit (recommended) |

> **Pi Zero W note:** the 64-bit OS image does **not** support ARMv6. You must use the 32-bit image. Kriya itself is architecture-agnostic — the same code runs on all of the above.

Python 3.11 is included in Raspberry Pi OS Lite Bookworm (32-bit and 64-bit).

```bash
# Transfer the archive to the Pi
scp kriya-v0.2.0.tar.gz pi@raspberrypi.local:~
ssh pi@raspberrypi.local

tar xzf kriya-v0.2.0.tar.gz
cd kriya

# Run the installer — auto-detects architecture and sets memory limits
sudo bash deploy/install.sh

# Edit the environment file — add your API key(s)
sudo nano /etc/kriya/kriya.env

# Start
sudo systemctl start kriya
sudo systemctl status kriya
```

The installer detects your architecture (`uname -m`) and automatically sets:
- `MemoryMax` in the systemd unit (180 MB for ARMv6, 320 MB for ARMv7, 512 MB for ARM64/x86_64)
- `KRIYA_MAX_AGENTS` default (1 for ARMv6, 2 for ARMv7, 4 for ARM64/x86_64)
- Ollama warning if ARMv6 is detected (Ollama does not support ARMv6)

**Per-device tuning:**

| Device | `KRIYA_MAX_AGENTS` | Recommended model | Ollama? |
|---|---|---|---|
| Pi Zero W (ARMv6) | 1 | `claude-3-5-haiku` or `gpt-4o-mini` | ✗ Not supported |
| Pi Zero 2 W (ARMv7) | 1–2 | `claude-3-5-haiku` or `gpt-4o-mini` | ✓ q4 models only |
| Pi 3 (ARMv7) | 2 | Any | ✓ q4 models only |
| Pi 4 / Pi 5 (ARM64) | 3–4 | Any | ✓ Full support |
| x86_64 server | 4–8 | Any | ✓ Full support |

### Systemd service

The installer places a unit file at `/etc/systemd/system/kriya.service`.

```bash
sudo systemctl enable kriya    # start on boot
sudo systemctl start  kriya
sudo systemctl stop   kriya
sudo systemctl restart kriya

# View live logs
journalctl -u kriya -f
```

The service runs as the `kriya` system user (no root). Data lives at `/var/lib/kriya/`.

---

## Configuration

Configuration is read from **environment variables** (highest priority) or `kriya.toml` in the base directory.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `KRIYA_BASE` | source directory | Base path for DB, vault, logs, projects |
| `KRIYA_HOST` | `0.0.0.0` | API server bind address |
| `KRIYA_PORT` | `7777` | API server port |
| `KRIYA_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `KRIYA_JWT_SECRET` | persisted to `.jwt_secret` | HS256 signing key — auto-persisted across restarts; set explicitly in production |
| `KRIYA_VAULT_PASS` | auto-generated (see `vault/.vault_passphrase`) | Master key for secrets vault — set explicitly in production for portability |
| `KRIYA_CORS_ORIGINS` | _(none — CORS disabled)_ | Comma-separated allowed origins, e.g. `https://app.example.com`. Use `*` only for public APIs. |
| `KRIYA_MAX_AGENTS` | `3` | Maximum concurrent agents |
| `ANTHROPIC_API_KEY` | — | Enables Anthropic Claude provider |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Default Claude model |
| `OPENAI_API_KEY` | — | Enables OpenAI provider |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Override for compatible APIs (e.g. Groq, Together) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | — | Set to enable Ollama (e.g. `llama3`, `mistral`). **Not supported on ARMv6.** |

### kriya.toml (optional)

```toml
[daemon]
host             = "0.0.0.0"
port             = 7777
log_level        = "INFO"
max_concurrent_agents = 2
agent_timeout_sec     = 300
short_term_capacity   = 50
```

---

## Web dashboard

The dashboard is served at `http://<host>:7777/` — no separate server needed.

**Screens:**

| Screen | Description |
|---|---|
| **Overview** | Live stats, project table, activity feed. Auto-refreshes every 4 seconds. |
| **Projects** | Create, run, delete projects. Per-project view: task DAG, agent list, secrets manager. |
| **Agents** | All agents across all projects. Inspect conversation history and raw output. |
| **Event log** | Full audit log, filterable by topic. |
| **Skills** | All loaded skills with status. |

**Login:** use the `admin` credentials printed at first boot. Change the password immediately via `agent user passwd admin`.

The dashboard is a single static HTML file (`static/dashboard.html`). It requires no build step and no JavaScript framework — works in any modern browser including on-device if you access via `localhost`.

---

## CLI reference

```bash
python3 bin/agent <command> [options]
# Or, after install: agent <command>
```

### Authentication

```bash
agent login                          # prompts for username + password, saves token
agent status                         # show daemon uptime, version, API URL
```

### Projects

```bash
agent project list
agent project create --name myproject --schedule "@every 1h"
agent project status myproject
agent project run myproject
agent project run myproject --follow  # stream live monitor until done
agent project load myproject.toml     # import a TOML definition
agent project delete myproject
```

### Tasks

```bash
agent task list   --project myproject
agent task add    --project myproject --name "fetch data" --model anthropic/claude-3-5-haiku-20241022
agent task status <task-id>
```

### Agents

```bash
agent agent list  --project myproject
agent agent inspect <agent-id>
```

### Secrets

Secrets are encrypted at rest (AES-256-GCM or HMAC-XOR fallback). They are injected as environment variables at runtime and never logged.

```bash
agent secret set    --project myproject --key SLACK_TOKEN
agent secret list   --project myproject
agent secret delete --project myproject --key SLACK_TOKEN
```

### Skills

```bash
agent skill list
agent skill install <skill-id>   # v2 — not yet implemented (drop handler.py manually)

#### Telegram skill (`telegram.send`, `telegram.get_updates`, `telegram.set_webhook`)

Send messages via Telegram Bot API, poll for incoming messages, or set webhooks.

```python
# Send a message
{"action": "skill_call", "skill": "telegram.send", "params": {
  "text": "Hello from Kriya!",
  "parse_mode": "Markdown"
}}

# Poll for incoming messages (long polling)
{"action": "skill_call", "skill": "telegram.get_updates", "params": {
  "timeout": 30,
  "offset": None
}}

# Agent calls this internally to get new messages
# Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (set via vault or env)
```

#### WhatsApp skill (`whatsapp.send`, `whatsapp.list`, `whatsapp.template`)

Send messages via WhatsApp Business Cloud API (Meta).

```python
# Send a text message
{"action": "skill_call", "skill": "whatsapp.send", "params": {
  "to": "+1234567890",
  "text": "Hello from Kriya!"
}}

# Send media (image, document, video)
{"action": "skill_call", "skill": "whatsapp.send", "params": {
  "to": "+1234567890",
  "type": "image",
  "media_url": "https://example.com/image.jpg",
  "caption": "Image caption"
}}

# Send interactive list
{"action": "skill_call", "skill": "whatsapp.list", "params": {
  "to": "+1234567890",
  "title": "Select Option",
  "message": "Choose from the list below",
  "button_text": "Select",
  "sections": [{"title": "Options", "rows": [{"id": "1", "title": "Option 1"}]}]
}}

# Send template message
{"action": "skill_call", "skill": "whatsapp.template", "params": {
  "to": "+1234567890",
  "template_name": "hello_world",
  "language": "en_US"
}}

# Secrets: WHATSAPP_TOKEN, WHATSAPP_PHONE_ID (set via vault or env)
```

#### Slack skill (`slack.send`, `slack.conversations_list`, `slack.webhook`)

Send messages via Slack API or incoming webhooks.

```python
# Send a message
{"action": "skill_call", "skill": "slack.send", "params": {
  "channel": "#general",
  "text": "Hello from Kriya!"
}}

# Send with Slack Block Kit
{"action": "skill_call", "skill": "slack.send", "params": {
  "channel": "#alerts",
  "text": "New alert!",
  "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "*Alert*"}}]
}}

# List channels
{"action": "skill_call", "skill": "slack.conversations_list", "params": {
  "types": "public_channel,private_channel"
}}

# Use incoming webhook (no token needed)
{"action": "skill_call", "skill": "slack.webhook", "params": {
  "webhook_url": "https://hooks.slack.com/services/XXX",
  "text": "Message via webhook"
}}

# Secrets: SLACK_TOKEN, SLACK_DEFAULT_CHANNEL (set via vault or env)
```

#### Gmail skill (`gmail.send`, `gmail.list`, `gmail.get`, `gmail.draft`)

Send and read emails via Gmail API. Requires OAuth2 setup (see below).

```python
# Send an email
{"action": "skill_call", "skill": "gmail.send", "params": {
  "mode": "send",
  "to": "user@example.com",
  "subject": "Hello from Kriya",
  "body": "Email body text"
}}

# Send HTML email
{"action": "skill_call", "skill": "gmail.send", "params": {
  "mode": "send",
  "to": "user@example.com",
  "subject": "HTML Email",
  "html": "<h1>Hello</h1><p>HTML content</p>"
}}

# List emails
{"action": "skill_call", "skill": "gmail.send", "params": {
  "mode": "list",
  "max_results": 10,
  "query": "is:unread"
}}

# Get email details
{"action": "skill_call", "skill": "gmail.send", "params": {
  "mode": "get",
  "message_id": "<message-id>"
}}

# Create draft
{"action": "skill_call", "skill": "gmail.send", "params": {
  "mode": "draft",
  "to": "user@example.com",
  "subject": "Draft Subject",
  "body": "Draft body"
}}

# Secrets: GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN (set via vault or env)
```

**Gmail OAuth2 Setup (Pi Zero compatible):**
1. Create OAuth2 credentials in Google Cloud Console
2. On a desktop machine, obtain a refresh_token via OAuth2 flow
3. Store the refresh_token in Kriya vault
4. Skill will automatically exchange for access_token on each call
```
```

### Monitoring & logs

```bash
agent monitor myproject           # live TUI monitor (polls every 2s)
agent logs --tail 50              # recent audit log entries
agent run examples/newsletter.toml --follow   # load + run + monitor
```

### Schedule syntax

| Expression | Meaning |
|---|---|
| `@every 30s` | Every 30 seconds |
| `@every 5m` | Every 5 minutes |
| `@every 2h` | Every 2 hours |
| `@daily` | Once every 24 hours |
| `@hourly` | Once every hour |
| `@weekly` | Once every 7 days |
| `@once` | Run once, then never again |

---

## Project files (TOML)

Projects are defined as TOML files. Load them with `agent project load <file>` or `agent run <file>`.

```toml
[project]
name        = "my-pipeline"
description = "What this project does"
schedule    = "@every 1h"        # optional — omit for manual-only

# ── Task DAG ──────────────────────────────────────────────────────────────

[tasks.step1]
name    = "Fetch data"
agents  = ["fetcher"]

[tasks.step2]
name       = "Process data"
depends_on = ["step1"]           # wait for step1 to be DONE
agents     = ["processor"]

[tasks.step3]
name       = "Publish results"
depends_on = ["step2"]
agents     = ["publisher"]

# ── Agent definitions ──────────────────────────────────────────────────────
# Referenced by ID inside tasks.agents = [...]

[[agents]]
id          = "fetcher"
role        = "executor"          # executor | planner | critic
model       = "auto"              # auto | anthropic/... | openai/... | ollama/...
provider    = "auto"              # auto | anthropic | openai | ollama
max_tokens  = 512
temperature = 0.3
skills      = ["web.scrape", "http.call"]
prompt      = """
You are a data fetching agent. Retrieve the latest data from the source.
Use web.scrape to fetch HTML pages.
Return results as JSON.
"""

[[agents]]
id          = "processor"
role        = "executor"
model       = "anthropic/claude-3-5-haiku-20241022"
provider    = "anthropic"
max_tokens  = 800
temperature = 0.5
skills      = ["fs.write"]
prompt      = "Process the fetched data and produce a summary."

[[agents]]
id          = "critic"
role        = "critic"            # critics review output from other agents
model       = "auto"
prompt      = """
Review the processor's output for accuracy and completeness.
Return JSON: {"approved": true/false, "feedback": "..."}
"""
```

### Agent model strings

| String | Provider | Model |
|---|---|---|
| `auto` | First available | Default model for that provider |
| `anthropic/claude-3-5-haiku-20241022` | Anthropic | Claude Haiku (fast, cheap) |
| `anthropic/claude-3-5-sonnet-20241022` | Anthropic | Claude Sonnet (balanced) |
| `openai/gpt-4o-mini` | OpenAI | GPT-4o mini |
| `openai/gpt-4o` | OpenAI | GPT-4o |
| `ollama/llama3` | Local Ollama | Llama 3 8B |
| `ollama/mistral` | Local Ollama | Mistral 7B |

---

## Agent roles

| Role | Behaviour |
|---|---|
| `executor` | Runs the task prompt, may call skills, returns output |
| `planner` | Receives a goal, outputs a JSON subtask decomposition |
| `critic` | Reviews another agent's output; returns `{"approved": bool, "feedback": "..."}` |

When multiple agents are assigned to a task, they run sequentially. The output of each agent is passed as context to the next.

---

## Built-in skills

Skills are tools agents can call by emitting a JSON action block in their response:

```
{"action": "skill_call", "skill": "<skill-id>", "params": {...}}
```

### `http.call`

Make an HTTP request.

```json
{
  "action": "skill_call",
  "skill":  "http.call",
  "params": {
    "url":     "https://api.example.com/data",
    "method":  "GET",
    "headers": {"Authorization": "Bearer token"},
    "timeout": 30
  }
}
```

Returns: `{"status": 200, "data": {...}}` or `{"status": 200, "text": "..."}`

### `web.scrape`

Fetch a URL and return cleaned plain text (HTML stripped). Static HTML only — no JavaScript rendering.

```json
{
  "action": "skill_call",
  "skill":  "web.scrape",
  "params": { "url": "https://news.ycombinator.com", "timeout": 20 }
}
```

Returns: `{"url": "...", "text": "...", "chars": 4821}`

### `fs.write`

Write content to a file. Restricted to `/tmp/` and `/var/lib/kriya/projects/`.

```json
{
  "action": "skill_call",
  "skill":  "fs.write",
  "params": { "path": "/tmp/output.md", "content": "# My report\n..." }
}
```

### `fs.read`

Read file content.

```json
{
  "action": "skill_call",
  "skill":  "fs.read",
  "params": { "path": "/tmp/output.md", "max_bytes": 8192 }
}
```

### `system.shell`

Run a whitelisted shell command (read-only system info only).

Allowed prefixes: `df `, `du `, `free`, `uptime`, `date`, `hostname`, `cat /proc/`, `ls `, `pwd`, `echo `

```json
{
  "action": "skill_call",
  "skill":  "system.shell",
  "params": { "command": "free -h" }
}
```

### `memory.remember`

Store text in the project's long-term vector memory.

```json
{
  "action": "skill_call",
  "skill":  "memory.remember",
  "params": { "project_id": "my-project", "content": "The API rate limit is 1000 req/min", "importance": 1.5 }
}
```

### `memory.recall`

Retrieve relevant memories by semantic similarity.

```json
{
  "action": "skill_call",
  "skill":  "memory.recall",
  "params": { "project_id": "my-project", "query": "API rate limits", "top_k": 3 }
}
```

### `telegram.send`

Send a message (or read updates) via the Telegram Bot API.

```json
{
  "action": "skill_call",
  "skill":  "telegram.send",
  "params": { "text": "Pipeline finished.", "chat_id": "-100123456789" }
}
```

| param | type | description |
|---|---|---|
| `action` | string | `"send"` (default) or `"get_updates"` |
| `text` | string | Message text (required for send) |
| `chat_id` | string | Overrides `TELEGRAM_CHAT_ID` secret |
| `parse_mode` | string | `"HTML"` \| `"Markdown"` \| `"MarkdownV2"` |

Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (default target).

### `slack.post`

Post a message to Slack or read channel history.

```json
{
  "action": "skill_call",
  "skill":  "slack.post",
  "params": { "channel": "#alerts", "text": "Build passed." }
}
```

| param | type | description |
|---|---|---|
| `action` | string | `"post"` (default), `"history"`, or `"channels"` |
| `text` | string | Message text (required for post) |
| `channel` | string | Overrides `SLACK_DEFAULT_CHANNEL` secret |
| `blocks` | list | Slack Block Kit JSON (optional rich layout) |
| `limit` | int | Max messages for history (default 10, max 200) |

Secrets: `SLACK_TOKEN` (xoxb-...), `SLACK_DEFAULT_CHANNEL` (optional default).

---

## Writing a custom skill

Drop a `handler.py` file into `skills/<skill-id>/handler.py`. The daemon auto-discovers it on startup.

```python
# skills/slack/handler.py
import urllib.request, json

SKILL_ID = "slack.post"

def handle(params: dict, secrets: dict) -> dict:
    """
    Post a message to a Slack channel.
    params:  { channel, text }
    secrets: { SLACK_TOKEN }  — set via: agent secret set --project <p> --key SLACK_TOKEN
    """
    token   = secrets.get("SLACK_TOKEN", "")
    channel = params.get("channel", "#general")
    text    = params.get("text", "")

    if not token:
        return {"error": "SLACK_TOKEN not set"}

    body = json.dumps({"channel": channel, "text": text}).encode()
    req  = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    return {"ok": data.get("ok"), "ts": data.get("ts")}
```

**Rules:**

- The file must define `SKILL_ID = "..."` and `def handle(params, secrets) -> dict`.
- Secrets are injected from the project vault at call time — never hard-code keys.
- Return a `dict`. On error, include `{"error": "..."}`.
- The function can be `async def handle(...)` — Kriya will await it.
- Restart the daemon after adding a new skill (or send `SIGHUP`).

---

## Memory system

Kriya implements a two-tier memory system for agents.

### Short-term memory

A bounded FIFO message buffer per agent. Holds the last N messages (default: 50) in the current conversation. System messages are always preserved. Persisted to SQLite so agents can resume across restarts.

**Configuration:** `KRIYA_SHORT_TERM_CAPACITY` (or `short_term_capacity` in `kriya.toml`)

### Long-term memory

A per-project vector store backed by SQLite. Uses 64-dimensional n-gram embeddings — no ML library required. Cosine similarity search returns the most relevant memories for a given query.

Agents automatically save important output to long-term memory at the end of each run. Agents can also explicitly call `memory.remember` and `memory.recall` skills.

**Upgrade path:** When running with Ollama, you can replace `_embed()` in `kriya/ai/memory.py` with calls to `ollama.embeddings` for true semantic search.

---

## Security

### Secrets vault

- Stored at `vault/<project-id>/<KEY>.enc`
- Encrypted with AES-256-GCM (if `cryptography` library present) or HMAC-XOR (stdlib fallback)
- Master key derived with PBKDF2 (100,000 iterations) from `KRIYA_VAULT_PASS`
- Secrets injected as environment variables at runtime — never written to logs

```bash
# Always set this before first run
export KRIYA_VAULT_PASS="my-long-random-passphrase"
```

### Authentication

- JWT HS256, 1-hour TTL
- Tokens saved to `~/.kriya_token` by the CLI
- JWT secret is auto-persisted to `$KRIYA_BASE/.jwt_secret` on first boot (tokens survive restarts); set `KRIYA_JWT_SECRET` in the environment for production deployments
- All API endpoints require `Authorization: Bearer <token>` except `/api/health` and `/api/auth/login`
- Login endpoint is rate-limited: 10 failed attempts per IP per 60 seconds

### RBAC

Five roles in ascending privilege order:

| Role | Can do |
|---|---|
| `read_only` | Read projects, tasks, agents |
| `skill` | Above + execute skills |
| `agent` | Above + write tasks, manage agents |
| `project_owner` | Above + write/delete projects, manage secrets |
| `admin` | Everything |

Default user `admin` has the `admin` role. Create restricted users via the API (`POST /api/users` — v2).

### Audit log

Every agent action, skill call, and project event is written to:
- `events` table in SQLite (queryable via API/dashboard)
- Structured log output (journald when running as a service)

Each event carries an OpenTelemetry-compatible trace ID for full execution tracing.

---

## API reference

All endpoints require `Authorization: Bearer <token>` unless noted.

### System

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | No | Health check |
| `GET` | `/api/status` | Yes (`project:read`) | Version, uptime |
| `POST` | `/api/auth/login` | No | Get JWT token |
| `GET` | `/api/auth/me` | Yes | Current user info |

### Projects

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects/<id>` | Get project + tasks |
| `POST` | `/api/projects/<id>/run` | Start project DAG |
| `PUT` | `/api/projects/<id>/schedule` | Update project schedule (cron expression) |
| `DELETE` | `/api/projects/<id>` | Delete project |

**Update schedule request:**
```json
{
  "schedule": "@every 1h"
}
```

**Update schedule response:**
```json
{
  "id": "project-id",
  "name": "my-project",
  "schedule": "@every 1h",
  "status": "idle"
}
```

Required capability: `project:write`

### Tasks

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/<id>/tasks` | List tasks |
| `POST` | `/api/projects/<id>/tasks` | Add task |
| `GET` | `/api/tasks/<id>` | Task detail + agents |
| `DELETE` | `/api/projects/<id>/tasks/<task_id>` | Delete a task |
| `PUT` | `/api/projects/<id>/schedule` | Update project schedule |

### Agents

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents` | List agents (`?project_id=` to filter) |
| `GET` | `/api/agents/<id>` | Agent detail + messages |

### Secrets

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/<id>/secrets` | List secret keys (values never returned) |
| `POST` | `/api/projects/<id>/secrets` | Set a secret |
| `DELETE` | `/api/projects/<id>/secrets/<key>` | Delete a secret |

### Users

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/users` | List users (admin only) |
| `POST` | `/api/users` | Create user (admin only) |
| `DELETE` | `/api/users/<id>` | Delete user (admin only) |
| `PUT` | `/api/users/<id>/role` | Change user role (admin only) |
| `PUT` | `/api/users/<id>/password` | Change password (self or admin) |

### Memory

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/<id>/memory` | List project long-term memories |
| `DELETE` | `/api/projects/<id>/memory/<mid>` | Delete a memory entry |

### Events

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/events` | Audit log (`?topic=task.done&limit=50`) |

### Skills

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/skills` | List loaded skills |

### Users

| Method | Path | Description | Required Capability |
|---|---|---|---|
| `GET` | `/api/users` | List all users | `project:read` |
| `POST` | `/api/users` | Create a new user | `admin` |
| `DELETE` | `/api/users/<id>` | Delete a user by ID | `admin` |
| `PUT` | `/api/users/<id>/role` | Change user role | `admin` |
| `PUT` | `/api/users/<id>/password` | Change user password | `admin` or own user |

**Create user request:**
```json
{
  "username": "string",
  "password": "string",
  "role": "read_only|skill|agent|project_owner|admin"
}
```

**Change role request:**
```json
{
  "role": "read_only|skill|agent|project_owner|admin"
}
```

**Change password request:**
```json
{
  "password": "string"
}
```

---

## Project structure

```
kriya/
├── kriya/
│   ├── ai/
│   │   ├── llm.py           # LLM abstraction — Anthropic, OpenAI, Ollama
│   │   └── memory.py        # Short-term + long-term memory
│   ├── api/
│   │   └── server.py        # REST API + static file serving
│   ├── core/
│   │   ├── agent.py         # Agent executor + skill call protocol
│   │   ├── bus.py           # Async event bus (pub/sub)
│   │   ├── config.py        # Configuration (env + TOML)
│   │   ├── loader.py        # TOML project file importer
│   │   ├── scheduler.py     # Task DAG scheduler + cron
│   │   └── store.py         # SQLite state store
│   ├── integrations/
│   │   └── builtin_skills.py  # http.call, web.scrape, fs.*, system.shell, memory.*
│   ├── security/
│   │   └── vault.py         # Secrets, JWT, RBAC, password hashing
│   └── daemon.py            # Main entrypoint + boot sequence
├── bin/
│   └── agent                # CLI tool
├── static/
│   └── dashboard.html       # Web dashboard (single file, zero deps)
├── skills/                  # Custom skill handler.py files
│   ├── telegram/            # telegram.send — send/receive via Telegram Bot API
│   ├── slack/               # slack.post — post messages / read channel history
│   ├── gmail/               # (stub — v2)
│   └── whatsapp/            # (stub — v2)
├── examples/
│   └── newsletter.toml      # Example multi-agent project
├── tests/
│   └── test_suite.py        # 40 unit + integration tests
├── deploy/
│   ├── kriya.service       # systemd unit file
│   └── install.sh           # Pi Zero / Debian installer
└── README.md
```

---

## Changelog

### v0.3.0
- **Communication skills** — `telegram.send` and `slack.post` plugin skills
  - `telegram.send`: send messages, send photos, poll updates via the Telegram Bot API (stdlib only)
  - `slack.post`: post messages, read channel history, list channels, user info via the Slack Web API
  - Both support project vault secrets (`TELEGRAM_BOT_TOKEN`, `SLACK_TOKEN`, etc.)
- **User management API** — `GET/POST /api/users`, `DELETE /api/users/<id>`, `PUT /api/users/<id>/role`, `PUT /api/users/<id>/password`
- **Memory API** — `GET /api/projects/<id>/memory`, `DELETE /api/projects/<id>/memory/<mid>`
- **Task / schedule API** — `DELETE /api/projects/<id>/tasks/<id>`, `PUT /api/projects/<id>/schedule`
- **Installation guide** — comprehensive `INSTALL.md` with platform-specific instructions
- **Bug fixes:**
  - Fixed `from __future__ import annotations` for Python 3.9 compatibility in `server.py`
  - Fixed `store.update()` calls using dict positional args instead of kwargs (schedule endpoint)
  - Fixed CLI `bin/agent` f-string syntax errors in monitor and agent inspect displays
  - Fixed `fs.write` skill path check on macOS (symlinked `/tmp` → `/private/tmp`)
  - Removed duplicate endpoint definitions from merge artifacts in `server.py`
  - Test suite: **41/41 passing** (up from 38)

### v0.2.0
- **Multi-architecture support** — explicit 32-bit (ARMv6, ARMv7) and 64-bit (ARM64, x86_64) coverage
  - Install script auto-detects `uname -m` and sets memory limits, agent count defaults, Ollama availability
  - Daemon banner shows detected architecture and Python version at startup
  - Web dashboard sidebar shows live architecture from `/api/status`
  - README OS image selection table for all Pi boards
- **Web dashboard** — single-file `static/dashboard.html` served at `/` by `kriya`
  - Five screens: Overview, Projects, Agents, Event log, Skills
  - Live stats and auto-refresh every 4 seconds
  - Task DAG visualiser with topological layer layout
  - Per-project secrets manager (add / delete; values never displayed)
  - Agent conversation inspector
  - Filterable audit event log
- **Static file serving** added to REST API server with path traversal protection

### v0.1.0
- Initial release
- Core daemon (`kriya`) — asyncio, single process, Pi Zero optimised
- SQLite state store with WAL mode
- Task DAG scheduler with `@every`, `@daily`, `@hourly`, `@weekly`, `@once` syntax
- Async event bus (pub/sub + request/reply)
- Agent executor with skill call protocol and exponential backoff retry
- LLM abstraction layer — Anthropic, OpenAI, Ollama via `urllib.request` (zero deps)
- Two-tier memory: short-term LRU buffer + long-term 64-dim vector store
- Secrets vault — AES-256-GCM / HMAC-XOR fallback, PBKDF2 master key
- JWT HS256 auth + five-role RBAC
- Built-in skills: `http.call`, `web.scrape`, `fs.read`, `fs.write`, `system.shell`, `memory.remember`, `memory.recall`
- Plugin skills: `telegram.send`, `telegram.get_updates`, `telegram.set_webhook` (via `skills/telegram/handler.py`)
- CLI tool with project, task, agent, secret, skill, monitor, logs commands
- TOML project definition format
- systemd service unit + Debian/Pi Zero install script
- 40-test suite (unit + integration)
