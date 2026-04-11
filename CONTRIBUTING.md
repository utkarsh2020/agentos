# Contributing to Kriya

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/kriya
cd kriya

# No pip install needed — zero external dependencies
# Just set at least one LLM provider key
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / OLLAMA_MODEL

# Run the daemon
python3 kriya/daemon.py

# Run tests (separate terminal)
python3 tests/test_suite.py
```

## Project layout

```
kriya/
  core/         Orchestration: store, bus, agent, scheduler, config, loader
  ai/           LLM abstraction, memory (short-term + long-term vector)
  api/          REST API server + static file serving
  security/     Vault, JWT, RBAC, password hashing
  integrations/ Built-in skills
  daemon.py     Boot entrypoint
bin/agent       CLI tool
static/         Web dashboard (single HTML file)
skills/         Drop custom skill handler.py files here
tests/          40-test suite (unit + integration)
deploy/         install.sh, kriya.service
examples/       Example TOML project definitions
```

## Adding a skill

1. Create `skills/<skill-id>/handler.py`
2. Define `SKILL_ID = "namespace.action"` and `def handle(params, secrets) -> dict`
3. Restart the daemon (or send `SIGHUP`)

See `kriya/integrations/builtin_skills.py` for examples.

## Adding an API endpoint

Register a route in `kriya/api/server.py`:

```python
@route("GET", "/api/my-resource")
def my_resource(h: KriyaHandler, *_):
    if not h._require("project:read"):
        return
    h._send({"data": "..."})
```

## Running tests

```bash
python3 tests/test_suite.py
```

Tests use a temp directory and never hit the network (LLM calls are mocked). All 41 tests must pass before merging.

## Code style

- Python 3.11+, stdlib only (no external pip dependencies)
- Type hints on all public functions
- Module-level docstring on every file
- Functional over OOP where possible; dataclasses for config/data structures

## Architecture constraints

- **No external dependencies.** If you think you need one, find the stdlib equivalent.
- **Pi Zero W safe.** Every code path must work in 512 MB RAM on ARMv6.
- **Single process.** No subprocesses, no threads beyond the API server thread.
- **Audit everything.** Every agent action must emit an event to the bus (which persists to SQLite).
