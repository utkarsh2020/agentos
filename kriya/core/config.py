"""
Kriya – Core configuration
Reads from kriya.toml (stdlib tomllib) or environment variables.
Pi Zero safe: no external deps.
"""
import os
import sys
import json
import pathlib
import dataclasses
from typing import Optional

# ── Base paths ─────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(os.environ.get("KRIYA_BASE", pathlib.Path(__file__).parent.parent.parent))
VAULT_DIR  = BASE_DIR / "vault"
LOG_DIR    = BASE_DIR / "logs"
PROJ_DIR   = BASE_DIR / "projects"
SKILLS_DIR = BASE_DIR / "skills"
DB_PATH    = BASE_DIR / "kriya.db"
SOCKET_PATH = BASE_DIR / "kriya.sock"
PID_FILE   = BASE_DIR / "kriya.pid"

for _d in (VAULT_DIR, LOG_DIR, PROJ_DIR, SKILLS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class LLMProviderConfig:
    name: str                    # "anthropic" | "openai" | "ollama"
    api_key: Optional[str]
    base_url: str
    default_model: str
    enabled: bool = True
    timeout: int = 60            # seconds – Pi Zero is slow on network


@dataclasses.dataclass
class KriyaConfig:
    # Daemon
    host: str = "0.0.0.0"
    port: int = 7777             # REST API
    log_level: str = "INFO"
    max_concurrent_agents: int = 3   # Pi Zero: keep low
    agent_memory_limit_mb: int = 64  # per-agent soft limit
    agent_timeout_sec: int = 300

    # Memory
    short_term_capacity: int = 50    # messages per agent
    vector_dims: int = 64            # tiny embeddings for Pi Zero

    # Auth
    jwt_secret: str = ""             # generated at first boot if empty
    jwt_ttl_sec: int = 3600

    # CORS — list of allowed origins. Empty list = no CORS headers (safest default).
    # Set to ["*"] only for fully public, unauthenticated APIs.
    cors_origins: dataclasses.field(default_factory=list) = dataclasses.field(default_factory=list)

    # Providers – populated from env / config file
    providers: dataclasses.field(default_factory=list) = dataclasses.field(default_factory=list)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def load_config() -> KriyaConfig:
    cfg = KriyaConfig()

    # Override from env
    if v := _env("KRIYA_HOST"):        cfg.host = v
    if v := _env("KRIYA_PORT"):        cfg.port = int(v)
    if v := _env("KRIYA_LOG_LEVEL"):   cfg.log_level = v
    if v := _env("KRIYA_JWT_SECRET"):  cfg.jwt_secret = v
    if v := _env("KRIYA_MAX_AGENTS"):  cfg.max_concurrent_agents = int(v)
    if v := _env("KRIYA_CORS_ORIGINS"):
        cfg.cors_origins = [o.strip() for o in v.split(",") if o.strip()]

    # Try loading TOML config (Python 3.11+)
    toml_path = BASE_DIR / "kriya.toml"
    if toml_path.exists():
        try:
            if sys.version_info >= (3, 11):
                import tomllib
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                _apply_toml(cfg, data)
        except Exception as e:
            print(f"[config] TOML parse warning: {e}", file=sys.stderr)

    # Build provider list from environment
    cfg.providers = _load_providers()

    # Persist JWT secret so tokens survive restarts.
    # Priority: env var > kriya.toml > persisted file > generate new.
    if not cfg.jwt_secret:
        import secrets as _sec
        secret_file = BASE_DIR / ".jwt_secret"
        if secret_file.exists():
            cfg.jwt_secret = secret_file.read_text().strip()
        else:
            cfg.jwt_secret = _sec.token_hex(32)
            secret_file.write_text(cfg.jwt_secret)
            secret_file.chmod(0o600)

    return cfg


def _load_providers() -> list:
    providers = []

    if key := _env("ANTHROPIC_API_KEY"):
        providers.append(LLMProviderConfig(
            name="anthropic",
            api_key=key,
            base_url="https://api.anthropic.com",
            default_model=_env("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        ))

    if key := _env("OPENAI_API_KEY"):
        providers.append(LLMProviderConfig(
            name="openai",
            api_key=key,
            base_url=_env("OPENAI_BASE_URL", "https://api.openai.com"),
            default_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        ))

    # Ollama – no key needed, local only
    ollama_url = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    providers.append(LLMProviderConfig(
        name="ollama",
        api_key=None,
        base_url=ollama_url,
        default_model=_env("OLLAMA_MODEL", "llama3"),
        enabled=bool(_env("OLLAMA_MODEL")),   # only if explicitly set
    ))

    return providers


def _apply_toml(cfg: KriyaConfig, data: dict):
    d = data.get("daemon", {})
    for field in dataclasses.fields(cfg):
        if field.name in d:
            setattr(cfg, field.name, d[field.name])


# Singleton
_config: Optional[KriyaConfig] = None

def get_config() -> KriyaConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
