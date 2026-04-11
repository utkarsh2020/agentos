"""
Kriya – REST API server
Pure stdlib http.server + json. No Flask/FastAPI deps.
Async-compatible via thread bridge.
Endpoints cover: projects, tasks, agents, skills, secrets, events, auth.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread, Lock
from typing import Optional
from urllib.parse import urlparse, parse_qs

from kriya.core import store
from kriya.core.config import get_config, BASE_DIR
from kriya.core.scheduler import run_project, next_run_time, CronScheduler
from kriya.security.vault import (
    authenticate, verify_token, has_capability,
    set_secret, get_secret, list_secrets, delete_secret,
    hash_password, ROLES,
)


def _remove_credentials_file():
    """Delete the first-run credentials file after the password has been changed."""
    cred_file = BASE_DIR / "first_run_credentials.txt"
    cred_file.unlink(missing_ok=True)


def _get_arch() -> str:
    import platform, struct
    m = platform.machine()
    b = struct.calcsize("P") * 8
    return {"armv6l":"ARMv6 32-bit","armv7l":"ARMv7 32-bit",
            "aarch64":"ARM64 64-bit","x86_64":"x86_64 64-bit",
            "AMD64":"x86_64 64-bit"}.get(m, f"{m} {b}-bit")

log = logging.getLogger("kriya.api")


# ── Login rate limiter ─────────────────────────────────────────────────────
# Blocks brute-force attempts: max 10 failures per IP per 60 seconds.

_login_failures: dict[str, list[float]] = defaultdict(list)
_login_lock = Lock()
_LOGIN_WINDOW_SEC = 60
_LOGIN_MAX_FAILURES = 10


def _is_login_allowed(ip: str) -> bool:
    now = time.time()
    with _login_lock:
        window = [t for t in _login_failures[ip] if now - t < _LOGIN_WINDOW_SEC]
        _login_failures[ip] = window
        return len(window) < _LOGIN_MAX_FAILURES


def _record_login_failure(ip: str) -> None:
    with _login_lock:
        _login_failures[ip].append(time.time())


# ── Route registry ─────────────────────────────────────────────────────────

_routes: dict[tuple[str, str], callable] = {}


def route(method: str, path: str):
    def decorator(fn):
        _routes[(method.upper(), path)] = fn
        return fn
    return decorator


# ── Base handler ───────────────────────────────────────────────────────────

class KriyaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        log.debug(f"API {args[0]} {args[1]} {args[2]}")

    def _cors_origin(self) -> Optional[str]:
        """Return the value to use for Access-Control-Allow-Origin, or None."""
        cfg = get_config()
        allowed = cfg.cors_origins
        if not allowed:
            return None
        if "*" in allowed:
            return "*"
        origin = self.headers.get("Origin", "")
        return origin if origin in allowed else None

    def _send(self, data: dict | list, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        cors = self._cors_origin()
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg: str, status: int = 400):
        self._send({"error": msg}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _claims(self) -> Optional[dict]:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return verify_token(auth[7:])
        return None

    def _require(self, cap: str) -> Optional[dict]:
        claims = self._claims()
        if not claims:
            self._err("Unauthorized", 401)
            return None
        if not has_capability(claims.get("rol", ""), cap):
            self._err("Forbidden", 403)
            return None
        return claims

    def _parsed_path(self):
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    def do_OPTIONS(self):
        self.send_response(204)
        cors = self._cors_origin()
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def _dispatch(self, method: str):
        path, qs = self._parsed_path()

        # Serve dashboard static files at /
        if method == "GET" and not path.startswith("/api"):
            return self._serve_static(path)

        # Exact match
        fn = _routes.get((method, path))
        if fn:
            fn(self, path, qs)
            return
        # Prefix match for parametric routes (e.g. /api/projects/<id>)
        for (m, pattern), fn in _routes.items():
            if m != method:
                continue
            parts_path    = path.split("/")
            parts_pattern = pattern.split("/")
            if len(parts_path) != len(parts_pattern):
                continue
            params = {}
            ok = True
            for pp, pt in zip(parts_path, parts_pattern):
                if pt.startswith("<") and pt.endswith(">"):
                    params[pt[1:-1]] = pp
                elif pp != pt:
                    ok = False
                    break
            if ok:
                fn(self, path, qs, **params)
                return
        self._err(f"Not found: {path}", 404)


    def _serve_static(self, path: str):
        """Serve files from static/. Checks BASE_DIR/static first, then source-tree static/."""
        import mimetypes
        from pathlib import Path
        from kriya.core.config import BASE_DIR

        if path in ("/", ""):
            path = "/dashboard.html"

        rel = path.lstrip("/")

        # Check BASE_DIR/static first (production), then source-tree static/ (dev)
        _src_static = Path(__file__).parent.parent.parent / "static"
        for root in (BASE_DIR / "static", _src_static):
            candidate = (root / rel).resolve()
            try:
                if str(candidate).startswith(str(root.resolve())) and candidate.exists():
                    file_path = candidate
                    break
            except Exception:
                continue
        else:
            return self._err(f"Not found: {path}", 404)

        # Final traversal check
        if ".." in rel:
            return self._err("Forbidden", 403)

        mime, _ = mimetypes.guess_type(str(file_path))
        mime = mime or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):    self._dispatch("GET")
    def do_POST(self):   self._dispatch("POST")
    def do_PUT(self):    self._dispatch("PUT")
    def do_DELETE(self): self._dispatch("DELETE")


# ── Auth ───────────────────────────────────────────────────────────────────

@route("POST", "/api/auth/login")
def login(h: KriyaHandler, *_):
    ip = h.client_address[0]
    if not _is_login_allowed(ip):
        return h._err("Too many failed attempts – try again later", 429)
    body = h._body()
    token = authenticate(body.get("username", ""), body.get("password", ""))
    if not token:
        _record_login_failure(ip)
        return h._err("Invalid credentials", 401)
    h._send({"token": token})


@route("GET", "/api/auth/me")
def me(h: KriyaHandler, *_):
    claims = h._require("project:read")
    if claims:
        h._send({"user": claims})


# ── System ─────────────────────────────────────────────────────────────────

@route("GET", "/api/status")
def status(h: KriyaHandler, *_):
    if not h._require("project:read"):
        return
    h._send({
        "status": "running",
        "version": "0.3.0",
        "arch": _get_arch(),
        "providers": [p.name for p in get_config().providers if p.enabled],
        "uptime_s": int(time.time() - _start_time),
    })

@route("GET", "/api/health")
def health(h: KriyaHandler, *_):
    h._send({"ok": True})


# ── Projects ───────────────────────────────────────────────────────────────

@route("GET", "/api/projects")
def list_projects(h: KriyaHandler, *_):
    if not h._require("project:read"):
        return
    h._send(store.fetch_all("projects"))


@route("POST", "/api/projects")
def create_project(h: KriyaHandler, *_):
    if not h._require("project:write"):
        return
    body = h._body()
    name = body.get("name", "").strip()
    if not name:
        return h._err("name required")
    pid = store.insert("projects",
        name=name,
        description=body.get("description", ""),
        schedule=body.get("schedule"),
        config_toml=body.get("config_toml"),
        status="idle",
        created_at=time.time(),
        updated_at=time.time(),
    )
    # Register schedule if provided
    sched = body.get("schedule")
    if sched:
        store.insert("scheduled_jobs",
            project_id=pid,
            schedule=sched,
            next_run=next_run_time(sched),
        )
    h._send(store.fetch_one("projects", pid), 201)


@route("GET", "/api/projects/<id>")
def get_project(h: KriyaHandler, *_, **params):
    if not h._require("project:read"):
        return
    p = store.fetch_one("projects", params["id"])
    if not p:
        return h._err("Project not found", 404)
    p["tasks"] = store.fetch_where("tasks", project_id=params["id"])
    h._send(p)


@route("POST", "/api/projects/<id>/run")
def run_project_endpoint(h: KriyaHandler, *_, **params):
    if not h._require("project:write"):
        return
    pid = params["id"]
    p = store.fetch_one("projects", pid)
    if not p:
        return h._err("Not found", 404)
    # Fire async task in background
    _run_async(run_project(pid))
    h._send({"status": "started", "project_id": pid})


@route("DELETE", "/api/projects/<id>")
def delete_project(h: KriyaHandler, *_, **params):
    if not h._require("project:write"):
        return
    store.delete("projects", params["id"])
    h._send({"deleted": True})


@route("PUT", "/api/projects/<id>/schedule")
def update_project_schedule(h: KriyaHandler, *_, **params):
    claims = h._require("project:write")
    if not claims:
        return
    pid = params["id"]
    p = store.fetch_one("projects", pid)
    if not p:
        return h._err("Project not found", 404)
    body = h._body()
    sched = body.get("schedule", "").strip()
    if not sched:
        return h._err("schedule required")
    try:
        next_run = next_run_time(sched)
    except Exception:
        return h._err("Invalid schedule format", 400)
    store.update("projects", pid, schedule=sched, updated_at=time.time())
    jobs = store.raw_query("SELECT id FROM scheduled_jobs WHERE project_id=?", (pid,))
    if jobs:
        store.raw_query(
            "UPDATE scheduled_jobs SET schedule=?, next_run=?, enabled=1 WHERE project_id=?",
            (sched, next_run, pid)
        )
    else:
        store.insert("scheduled_jobs", project_id=pid, schedule=sched, next_run=next_run)
    h._send({"updated": True, "schedule": sched})


# ── Memory ────────────────────────────────────────────────────────────────

@route("GET", "/api/projects/<project_id>/memory")
def list_project_memory(h: KriyaHandler, path, qs, **params):
    if not h._require("project:read"):
        return
    limit = int(qs.get("limit", [50])[0])
    rows = store.raw_query(
        "SELECT id, agent_id, content, importance, created_at FROM memory "
        "WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
        (params["project_id"], limit)
    )
    h._send(rows)


@route("DELETE", "/api/projects/<project_id>/memory/<id>")
def delete_memory(h: KriyaHandler, *_, **params):
    if not h._require("project:write"):
        return
    rows = store.raw_query(
        "SELECT id FROM memory WHERE id=? AND project_id=?",
        (params["id"], params["project_id"])
    )
    if not rows:
        return h._err("Not found", 404)
    store.raw_query("DELETE FROM memory WHERE id=?", (params["id"],))
    h._send({"deleted": True})


# ── Tasks ──────────────────────────────────────────────────────────────────

@route("GET", "/api/projects/<project_id>/tasks")
def list_tasks(h: KriyaHandler, *_, **params):
    if not h._require("task:read"):
        return
    h._send(store.fetch_where("tasks", project_id=params["project_id"]))


@route("POST", "/api/projects/<project_id>/tasks")
def create_task(h: KriyaHandler, *_, **params):
    if not h._require("task:write"):
        return
    body = h._body()
    tid = store.insert("tasks",
        project_id=params["project_id"],
        name=body.get("name", "Unnamed task"),
        depends_on=json.dumps(body.get("depends_on", [])),
        config=json.dumps(body.get("config", {})),
        status="pending",
        created_at=time.time(),
    )
    h._send(store.fetch_one("tasks", tid), 201)


@route("DELETE", "/api/projects/<project_id>/tasks/<id>")
def delete_task(h: KriyaHandler, *_, **params):
    if not h._require("task:write"):
        return
    t = store.fetch_one("tasks", params["id"])
    if not t or t.get("project_id") != params["project_id"]:
        return h._err("Not found", 404)
    store.delete("tasks", params["id"])
    h._send({"deleted": True})


@route("GET", "/api/tasks/<id>")
def get_task(h: KriyaHandler, *_, **params):
    if not h._require("task:read"):
        return
    t = store.fetch_one("tasks", params["id"])
    if not t:
        return h._err("Not found", 404)
    t["agents"] = store.fetch_where("agents", task_id=params["id"])
    h._send(t)


# ── Agents ────────────────────────────────────────────────────────────────

@route("GET", "/api/agents")
def list_agents(h: KriyaHandler, path, qs):
    if not h._require("agent:read"):
        return
    project_id = qs.get("project_id", [None])[0]
    if project_id:
        h._send(store.fetch_where("agents", project_id=project_id))
    else:
        h._send(store.fetch_all("agents"))


@route("GET", "/api/agents/<id>")
def get_agent(h: KriyaHandler, *_, **params):
    claims = h._require("agent:read")
    if not claims:
        return
    a = store.fetch_one("agents", params["id"])
    if not a:
        return h._err("Not found", 404)
    # Only admins get full conversation history — it may contain injected secrets
    if has_capability(claims.get("rol", ""), "admin:read"):
        a["messages"] = store.fetch_where("agent_messages", agent_id=params["id"])
    h._send(a)


# ── Secrets ────────────────────────────────────────────────────────────────

@route("GET", "/api/projects/<project_id>/secrets")
def list_proj_secrets(h: KriyaHandler, *_, **params):
    if not h._require("project:read"):
        return
    h._send({"keys": list_secrets(params["project_id"])})


@route("POST", "/api/projects/<project_id>/secrets")
def set_proj_secret(h: KriyaHandler, *_, **params):
    if not h._require("project:write"):
        return
    body = h._body()
    key   = body.get("key", "").strip()
    value = body.get("value", "")
    if not key:
        return h._err("key required")
    set_secret(params["project_id"], key, value)
    h._send({"stored": True, "key": key})


@route("DELETE", "/api/projects/<project_id>/secrets/<key>")
def del_proj_secret(h: KriyaHandler, *_, **params):
    if not h._require("project:write"):
        return
    delete_secret(params["project_id"], params["key"])
    h._send({"deleted": True})


# ── Events (audit log) ────────────────────────────────────────────────────

@route("GET", "/api/events")
def list_events(h: KriyaHandler, path, qs):
    if not h._require("project:read"):
        return
    topic = qs.get("topic", [None])[0]
    limit = int(qs.get("limit", [50])[0])
    if topic:
        rows = store.raw_query(
            "SELECT * FROM events WHERE topic=? ORDER BY created_at DESC LIMIT ?",
            (topic, limit)
        )
    else:
        rows = store.raw_query(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    h._send(rows)


# ── Users ─────────────────────────────────────────────────────────────────

@route("GET", "/api/users")
def list_users(h: KriyaHandler, *_):
    if not h._require("admin:read"):
        return
    rows = store.fetch_all("users")
    # Never return password hashes
    for r in rows:
        r.pop("password_hash", None)
    h._send(rows)


@route("POST", "/api/users")
def create_user(h: KriyaHandler, *_):
    if not h._require("admin:write"):
        return
    body = h._body()
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    role     = body.get("role", "read_only")
    if not username or not password:
        return h._err("username and password required")
    if role not in ROLES:
        return h._err(f"role must be one of: {', '.join(ROLES)}")
    existing = store.raw_query("SELECT id FROM users WHERE username=?", (username,))
    if existing:
        return h._err("username already exists", 409)
    uid = store.insert("users",
        username=username,
        password_hash=hash_password(password),
        role=role,
        created_at=time.time(),
    )
    row = store.fetch_one("users", uid)
    row.pop("password_hash", None)
    h._send(row, 201)


@route("DELETE", "/api/users/<id>")
def delete_user(h: KriyaHandler, *_, **params):
    claims = h._require("admin:write")
    if not claims:
        return
    if params["id"] == claims.get("sub"):
        return h._err("Cannot delete your own account", 400)
    store.delete("users", params["id"])
    h._send({"deleted": True})


@route("PUT", "/api/users/<id>/role")
def change_user_role(h: KriyaHandler, *_, **params):
    if not h._require("admin:write"):
        return
    body = h._body()
    role = body.get("role", "")
    if role not in ROLES:
        return h._err(f"role must be one of: {', '.join(ROLES)}")
    store.update("users", params["id"], role=role)
    row = store.fetch_one("users", params["id"])
    if not row:
        return h._err("User not found", 404)
    row.pop("password_hash", None)
    h._send(row)


@route("PUT", "/api/users/<id>/password")
def change_user_password(h: KriyaHandler, *_, **params):
    claims = h._require("project:read")   # any authenticated user
    if not claims:
        return
    # Non-admins can only change their own password
    if claims.get("sub") != params["id"] and not has_capability(claims.get("rol", ""), "admin:write"):
        return h._err("Forbidden", 403)
    body = h._body()
    password = body.get("password", "").strip()
    if not password:
        return h._err("password required")
    store.update("users", params["id"], password_hash=hash_password(password))
    # Remove the first-run credentials file once any password is changed
    _remove_credentials_file()
    h._send({"updated": True})


# ── Skills ────────────────────────────────────────────────────────────────

@route("GET", "/api/skills")
def list_skills_endpoint(h: KriyaHandler, *_):
    if not h._require("project:read"):
        return
    from kriya.core.agent import _skill_handlers
    h._send([{"id": sid, "loaded": True} for sid in _skill_handlers])


# ── Async bridge ──────────────────────────────────────────────────────────

_loop: Optional[asyncio.AbstractEventLoop] = None
_start_time = time.time()


def _run_async(coro):
    """Submit a coroutine to the main event loop from a sync thread."""
    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(coro, _loop)
    else:
        log.warning("No event loop – cannot run async task")


# ── Server bootstrap ──────────────────────────────────────────────────────

def start_api_server(host: str, port: int, loop: asyncio.AbstractEventLoop):
    global _loop
    _loop = loop
    server = HTTPServer((host, port), KriyaHandler)
    log.info(f"[api] listening on http://{host}:{port}")
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
