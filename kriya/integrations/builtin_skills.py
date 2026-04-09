"""
Kriya – Built-in skills (stdlib only)
Registered at daemon startup.
Each skill: handler(params: dict, secrets: dict) -> dict
"""
import json
import logging
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

log = logging.getLogger("kriya.skills")


# ── http.call ──────────────────────────────────────────────────────────────

def skill_http_call(params: dict, secrets: dict) -> dict:
    """
    Generic HTTP request.
    params: {url, method?, headers?, body?, timeout?}
    """
    url     = params.get("url", "")
    method  = params.get("method", "GET").upper()
    headers = params.get("headers", {})
    body    = params.get("body")
    timeout = params.get("timeout", 30)

    if not url:
        return {"error": "url is required"}

    data = json.dumps(body).encode() if body else None
    if data and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "json" in ct:
                return {"status": resp.status, "data": json.loads(raw)}
            return {"status": resp.status, "text": raw.decode(errors="replace")[:4096]}
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode(errors="replace")[:512]}
    except Exception as e:
        return {"error": str(e)}


# ── web.scrape ─────────────────────────────────────────────────────────────

def skill_web_scrape(params: dict, secrets: dict) -> dict:
    """
    Fetch a URL and return cleaned text content.
    params: {url, timeout?}
    Note: No JS rendering on Pi Zero – static HTML only.
    """
    url     = params.get("url", "")
    timeout = params.get("timeout", 20)
    if not url:
        return {"error": "url required"}

    headers = {
        "User-Agent": "Kriya/0.3 (+https://github.com/kriya)",
        "Accept": "text/html,application/xhtml+xml",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode(errors="replace")
        # Minimal HTML → text extraction (no deps)
        text = _html_to_text(html)
        return {
            "url":   url,
            "text":  text[:8000],
            "chars": len(text),
        }
    except Exception as e:
        return {"error": str(e)}


def _html_to_text(html: str) -> str:
    import re
    # Remove scripts/styles
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html, flags=re.DOTALL | re.I)
    # Remove tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # Decode common entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
               .replace("&nbsp;", " ").replace("&quot;", '"')
    # Collapse whitespace
    html = re.sub(r'\s+', ' ', html).strip()
    return html


# ── fs.write ───────────────────────────────────────────────────────────────

def skill_fs_write(params: dict, secrets: dict) -> dict:
    """
    Write content to a file.
    params: {path, content, mode?}
    """
    path    = params.get("path", "")
    content = params.get("content", "")
    mode    = params.get("mode", "w")
    if not path:
        return {"error": "path required"}
    # Safety: only allow writes under /tmp or configured project dir
    p = Path(path).resolve()
    allowed = [Path("/tmp").resolve(), Path("/var/lib/kriya/projects").resolve()]
    if not any(str(p).startswith(str(a)) for a in allowed):
        return {"error": f"Write not allowed to {p} – use /tmp or project dirs"}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode) as f:
        f.write(content)
    return {"written": True, "path": str(p), "bytes": len(content)}


# ── fs.read ────────────────────────────────────────────────────────────────

def skill_fs_read(params: dict, secrets: dict) -> dict:
    """
    Read file content.
    params: {path, max_bytes?}
    """
    path      = params.get("path", "")
    max_bytes = params.get("max_bytes", 8192)
    if not path:
        return {"error": "path required"}
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read(max_bytes)
        return {"content": content, "path": path}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except Exception as e:
        return {"error": str(e)}


# ── system.shell ───────────────────────────────────────────────────────────

def skill_system_shell(params: dict, secrets: dict) -> dict:
    """
    Run a shell command (whitelist enforced).
    params: {command}
    """
    import subprocess
    cmd = params.get("command", "")
    if not cmd:
        return {"error": "command required"}

    # Safety: very minimal allowlist for Pi Zero system info commands
    ALLOWED_PREFIXES = ("df ", "du ", "free", "uptime", "date", "hostname",
                        "cat /proc/", "ls ", "pwd", "echo ")
    if not any(cmd.startswith(p) for p in ALLOWED_PREFIXES):
        return {"error": f"Command not in allowlist: {cmd!r}"}

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return {
            "stdout": result.stdout[:2048],
            "stderr": result.stderr[:512],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


# ── memory.remember ────────────────────────────────────────────────────────

def skill_memory_remember(params: dict, secrets: dict) -> dict:
    """Store something in project long-term memory."""
    # Lazy import to avoid circular at module level
    from kriya.ai.memory import get_long_term
    project_id = params.get("project_id", "default")
    content    = params.get("content", "")
    importance = float(params.get("importance", 1.0))
    if not content:
        return {"error": "content required"}
    ltm = get_long_term(project_id)
    mid = ltm.remember(content, importance=importance)
    return {"stored": True, "memory_id": mid}


def skill_memory_recall(params: dict, secrets: dict) -> dict:
    """Recall relevant memories for a query."""
    from kriya.ai.memory import get_long_term
    project_id = params.get("project_id", "default")
    query      = params.get("query", "")
    top_k      = int(params.get("top_k", 5))
    if not query:
        return {"error": "query required"}
    ltm = get_long_term(project_id)
    return {"memories": ltm.recall(query, top_k=top_k)}


# ── Registration ───────────────────────────────────────────────────────────

def register_builtin_skills():
    from kriya.core.agent import register_skill
    register_skill("http.call",        skill_http_call)
    register_skill("web.scrape",       skill_web_scrape)
    register_skill("fs.write",         skill_fs_write)
    register_skill("fs.read",          skill_fs_read)
    register_skill("system.shell",     skill_system_shell)
    register_skill("memory.remember",  skill_memory_remember)
    register_skill("memory.recall",    skill_memory_recall)
    log.info("[skills] built-in skills registered: http.call, web.scrape, fs.read/write, system.shell, memory.*")
