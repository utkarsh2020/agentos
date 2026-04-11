"""
Kriya – SQLite state store
Single WAL-mode database for all persistent state.
Thread-safe via connection-per-thread pattern.
Pi Zero safe: stdlib sqlite3 only.
"""
import sqlite3
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Any

from kriya.core.config import DB_PATH, BASE_DIR


# Allowlist of valid table names — prevents SQL injection via table-name interpolation
_ALLOWED_TABLES = frozenset({
    "projects", "tasks", "agents", "agent_messages",
    "events", "memory", "scheduled_jobs", "users", "skills",
})

def _validate_table(table: str) -> None:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table!r}")


# One connection per thread
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA cache_size=-4096")   # 4 MB – Pi Zero friendly
        _local.conn = c
    return _local.conn


def init_db():
    """Create all tables. Idempotent."""
    c = _conn()
    c.executescript("""
    -- ── Projects ──────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS projects (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE,
        description TEXT,
        status      TEXT DEFAULT 'idle',   -- idle|running|paused|archived
        schedule    TEXT,                  -- cron expression or @every Xs
        config_toml TEXT,                  -- raw TOML source
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL
    );

    -- ── Tasks ─────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS tasks (
        id          TEXT PRIMARY KEY,
        project_id  TEXT NOT NULL,
        name        TEXT NOT NULL,
        status      TEXT DEFAULT 'pending',  -- pending|running|done|failed|skipped
        depends_on  TEXT DEFAULT '[]',       -- JSON array of task IDs
        config      TEXT DEFAULT '{}',       -- JSON: agents, schedule, etc.
        output      TEXT,                    -- JSON output from agents
        error       TEXT,
        started_at  REAL,
        finished_at REAL,
        created_at  REAL NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    );

    -- ── Agents ────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS agents (
        id           TEXT PRIMARY KEY,
        task_id      TEXT NOT NULL,
        project_id   TEXT NOT NULL,
        role         TEXT NOT NULL,          -- executor|planner|critic
        model        TEXT NOT NULL,
        provider     TEXT NOT NULL,
        state        TEXT DEFAULT 'pending', -- pending|running|done|failed
        system_prompt TEXT,
        output       TEXT,
        error        TEXT,
        token_usage  INTEGER DEFAULT 0,
        started_at   REAL,
        finished_at  REAL,
        created_at   REAL NOT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks(id)
    );

    -- ── Agent messages (conversation history) ────────────────────────────
    CREATE TABLE IF NOT EXISTS agent_messages (
        id         TEXT PRIMARY KEY,
        agent_id   TEXT NOT NULL,
        role       TEXT NOT NULL,    -- user|assistant|tool
        content    TEXT NOT NULL,
        created_at REAL NOT NULL,
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    );

    -- ── Events (audit log + event bus persistence) ────────────────────────
    CREATE TABLE IF NOT EXISTS events (
        id         TEXT PRIMARY KEY,
        topic      TEXT NOT NULL,
        source_id  TEXT,            -- agent/task/project ID
        payload    TEXT NOT NULL,   -- JSON
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);
    CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);

    -- ── Long-term memory (vector store – simplified cosine) ───────────────
    CREATE TABLE IF NOT EXISTS memory (
        id          TEXT PRIMARY KEY,
        project_id  TEXT NOT NULL,
        agent_id    TEXT,
        content     TEXT NOT NULL,
        embedding   TEXT NOT NULL,  -- JSON float array
        importance  REAL DEFAULT 1.0,
        created_at  REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_project ON memory(project_id);

    -- ── Scheduled jobs ────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS scheduled_jobs (
        id          TEXT PRIMARY KEY,
        project_id  TEXT NOT NULL,
        schedule    TEXT NOT NULL,
        last_run    REAL,
        next_run    REAL NOT NULL,
        enabled     INTEGER DEFAULT 1
    );

    -- ── Users / RBAC ──────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS users (
        id           TEXT PRIMARY KEY,
        username     TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role         TEXT DEFAULT 'read_only',  -- admin|project_owner|agent|skill|read_only
        created_at   REAL NOT NULL
    );

    -- ── Skills registry ───────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS skills (
        id          TEXT PRIMARY KEY,   -- e.g. "gmail.read"
        name        TEXT NOT NULL,
        version     TEXT DEFAULT '1.0.0',
        enabled     INTEGER DEFAULT 1,
        config      TEXT DEFAULT '{}',  -- JSON
        installed_at REAL NOT NULL
    );
    """)
    c.commit()
    _ensure_admin()


def _ensure_admin():
    """Create default admin user if no users exist."""
    import secrets as _sec
    c = _conn()
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        # Generate a random one-time password — never use a hardcoded default
        from kriya.security.vault import hash_password
        pw = _sec.token_urlsafe(16)
        ph = hash_password(pw)
        c.execute(
            "INSERT INTO users VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), "admin", ph, "admin", time.time())
        )
        c.commit()

        # Write credentials to a file so they survive across systemd journal rotations.
        # The file is deleted automatically when the admin password is first changed.
        cred_file = BASE_DIR / "first_run_credentials.txt"
        try:
            cred_file.write_text(
                f"Kriya first-run credentials\n"
                f"===========================\n"
                f"username : admin\n"
                f"password : {pw}\n\n"
                f"Change this password immediately:\n"
                f"  agent user passwd admin\n"
                f"  (or via the web dashboard at http://<host>:7777)\n\n"
                f"This file is deleted automatically after the password is changed.\n"
            )
            cred_file.chmod(0o600)
            print(f"[db] *** First-run credentials written to: {cred_file} ***")
        except Exception as e:
            # File write is best-effort; credentials are still printed to stdout
            print(f"[db] Warning: could not write credentials file: {e}")

        print("[db] *** Default admin created ***")
        print(f"[db]   username : admin")
        print(f"[db]   password : {pw}")
        print(f"[db]   Credentials also saved to: {cred_file}")


# ── Generic CRUD helpers ──────────────────────────────────────────────────

def insert(table: str, **kwargs) -> str:
    _validate_table(table)
    if "id" not in kwargs:
        kwargs["id"] = str(uuid.uuid4())
    now = time.time()
    if "created_at" not in kwargs:
        kwargs["created_at"] = now
    # Check if this table has an updated_at column (sqlite3.Row uses dict-style access)
    table_cols = {row["name"] for row in _conn().execute(f"PRAGMA table_info({table})").fetchall()}
    if "updated_at" in table_cols:
        kwargs.setdefault("updated_at", now)
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    _conn().execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", list(kwargs.values()))
    _conn().commit()
    return kwargs["id"]


def update(table: str, id: str, **kwargs):
    _validate_table(table)
    table_cols = {row["name"] for row in _conn().execute(f"PRAGMA table_info({table})").fetchall()}
    if "updated_at" in table_cols:
        kwargs["updated_at"] = time.time()
    pairs = ", ".join(f"{k}=?" for k in kwargs)
    _conn().execute(f"UPDATE {table} SET {pairs} WHERE id=?", [*kwargs.values(), id])
    _conn().commit()


def fetch_one(table: str, id: str) -> Optional[dict]:
    _validate_table(table)
    row = _conn().execute(f"SELECT * FROM {table} WHERE id=?", (id,)).fetchone()
    return dict(row) if row else None


def fetch_where(table: str, **kwargs) -> list[dict]:
    _validate_table(table)
    if not kwargs:
        rows = _conn().execute(f"SELECT * FROM {table}").fetchall()
    else:
        conds = " AND ".join(f"{k}=?" for k in kwargs)
        rows = _conn().execute(f"SELECT * FROM {table} WHERE {conds}", list(kwargs.values())).fetchall()
    return [dict(r) for r in rows]


def fetch_all(table: str) -> list[dict]:
    _validate_table(table)
    return [dict(r) for r in _conn().execute(f"SELECT * FROM {table}").fetchall()]


def delete(table: str, id: str):
    _validate_table(table)
    _conn().execute(f"DELETE FROM {table} WHERE id=?", (id,))
    _conn().commit()


def raw_query(sql: str, params=()) -> list[dict]:
    return [dict(r) for r in _conn().execute(sql, params).fetchall()]


def append_event(topic: str, payload: dict, source_id: str = None):
    insert("events",
        id=str(uuid.uuid4()),
        topic=topic,
        source_id=source_id,
        payload=json.dumps(payload),
        created_at=time.time()
    )
