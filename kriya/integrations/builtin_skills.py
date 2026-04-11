"""
Kriya – Built-in skills (stdlib only)
Registered at daemon startup.
Each skill: handler(params: dict, secrets: dict) -> dict
"""
import ipaddress
import json
import logging
import os
import re
import shlex
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

log = logging.getLogger("kriya.skills")


# ── SSRF guard ─────────────────────────────────────────────────────────────

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local / EC2 metadata
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

def _is_safe_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Blocks private IPs, loopback, and non-http(s) schemes."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "invalid URL"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme {parsed.scheme!r} not allowed (use http or https)"
    host = parsed.hostname or ""
    if not host:
        return False, "missing host"
    # Block known internal hostnames by name
    lower = host.lower()
    if lower in ("localhost",) or lower.endswith(".local") or lower.endswith(".internal"):
        return False, f"host {host!r} resolves to an internal address"
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE_NETS:
            if addr in net:
                return False, f"IP {host} is in a private/reserved range"
    except ValueError:
        pass  # hostname, not a raw IP — allow (DNS resolution happens later)
    return True, ""


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

    ok, reason = _is_safe_url(url)
    if not ok:
        return {"error": f"URL blocked: {reason}"}

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

    ok, reason = _is_safe_url(url)
    if not ok:
        return {"error": f"URL blocked: {reason}"}

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

_FS_ALLOWED_READ = [Path("/tmp").resolve(), Path("/var/lib/kriya/projects").resolve()]

def skill_fs_read(params: dict, secrets: dict) -> dict:
    """
    Read file content.
    params: {path, max_bytes?}
    """
    path      = params.get("path", "")
    max_bytes = params.get("max_bytes", 8192)
    if not path:
        return {"error": "path required"}
    p = Path(path).resolve()
    if not any(str(p).startswith(str(a)) for a in _FS_ALLOWED_READ):
        return {"error": f"Read not allowed from {p} – use /tmp or project dirs"}
    try:
        with open(p, "r", errors="replace") as f:
            content = f.read(max_bytes)
        return {"content": content, "path": str(p)}
    except FileNotFoundError:
        return {"error": f"File not found: {p}"}
    except Exception as e:
        return {"error": str(e)}


# ── system.shell ───────────────────────────────────────────────────────────

# Exact set of allowed base commands (first token only).
_SHELL_ALLOWED_CMDS = frozenset({
    "df", "du", "free", "uptime", "date", "hostname", "ls", "pwd", "echo",
})
# Reject shell metacharacters that could be used for injection even without shell=True.
_SHELL_METACHAR = re.compile(r'[;&|`$()<>\\!]')

def skill_system_shell(params: dict, secrets: dict) -> dict:
    """
    Run a shell command (exact base-command allowlist enforced, shell=False).
    params: {command}
    """
    import subprocess
    cmd = params.get("command", "")
    if not cmd:
        return {"error": "command required"}

    # Reject shell metacharacters before any parsing
    if _SHELL_METACHAR.search(cmd):
        return {"error": "Shell metacharacters are not allowed"}

    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return {"error": f"Invalid command syntax: {e}"}

    if not parts:
        return {"error": "Empty command"}

    base_cmd = parts[0]
    if base_cmd not in _SHELL_ALLOWED_CMDS:
        return {"error": f"Command {base_cmd!r} not in allowlist: {sorted(_SHELL_ALLOWED_CMDS)}"}

    try:
        result = subprocess.run(
            parts, shell=False, capture_output=True, text=True, timeout=10
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
