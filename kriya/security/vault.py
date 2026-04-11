"""
Kriya – Security layer
Vault: AES-256-GCM via Python's hazmat (stdlib only fallback: XOR+HMAC)
JWT:   HS256 hand-rolled (stdlib hmac + base64)
RBAC:  five roles with capability checks
Pi Zero safe: stdlib only.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import time
import uuid
from pathlib import Path
from typing import Optional

from kriya.core.config import VAULT_DIR, get_config
from kriya.core import store

# ── Role hierarchy ─────────────────────────────────────────────────────────
ROLES = ["read_only", "skill", "agent", "project_owner", "admin"]

CAPABILITIES = {
    "read_only":     {"project:read", "task:read", "agent:read"},
    "skill":         {"project:read", "task:read", "skill:execute"},
    "agent":         {"project:read", "task:read", "task:write", "agent:read", "agent:write", "skill:execute"},
    "project_owner": {"project:read", "project:write", "task:read", "task:write",
                      "agent:read", "agent:write", "skill:execute", "skill:read"},
    "admin":         {"*"},  # all capabilities
}


def has_capability(role: str, cap: str) -> bool:
    caps = CAPABILITIES.get(role, set())
    return "*" in caps or cap in caps


# ── Password hashing ───────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{h.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    # Support legacy plain sha256 hashes from DB init
    if ":" not in stored_hash:
        return hmac.compare_digest(
            hashlib.sha256(password.encode()).hexdigest(), stored_hash
        )
    salt, h = stored_hash.split(":", 1)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hmac.compare_digest(candidate.hex(), h)


# ── JWT (HS256, stdlib only) ───────────────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def issue_token(user_id: str, username: str, role: str) -> str:
    cfg = get_config()
    now = int(time.time())
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "sub": user_id,
        "usr": username,
        "rol": role,
        "iat": now,
        "exp": now + cfg.jwt_ttl_sec,
        "jti": str(uuid.uuid4()),
    }).encode())
    sig = _b64url(
        hmac.new(cfg.jwt_secret.encode(), f"{header}.{payload}".encode(), "sha256").digest()
    )
    return f"{header}.{payload}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    try:
        header, payload, sig = token.split(".")
        cfg = get_config()
        expected = _b64url(
            hmac.new(cfg.jwt_secret.encode(), f"{header}.{payload}".encode(), "sha256").digest()
        )
        if not hmac.compare_digest(expected, sig):
            return None
        claims = json.loads(_b64url_decode(payload))
        if claims.get("exp", 0) < time.time():
            return None
        return claims
    except Exception:
        return None


def authenticate(username: str, password: str) -> Optional[str]:
    """Returns JWT or None."""
    rows = store.raw_query("SELECT * FROM users WHERE username=?", (username,))
    if not rows:
        return None
    user = rows[0]
    if not verify_password(password, user["password_hash"]):
        return None
    return issue_token(user["id"], user["username"], user["role"])


# ── Vault (secrets store) ──────────────────────────────────────────────────
# Encryption: AES-256-GCM preferred, fallback to HMAC-XOR (Pi Zero may lack hazmat)

def _get_master_key() -> bytes:
    """Derive or load master key. Stored in vault/master.key (encrypted with env passphrase)."""
    key_file = VAULT_DIR / "master.key"
    passphrase = os.environ.get("KRIYA_VAULT_PASS")

    if not passphrase:
        # No env var set — use a machine-local auto-generated passphrase persisted to disk.
        # This is weaker than a user-supplied passphrase but far better than a hardcoded default.
        pass_file = VAULT_DIR / ".vault_passphrase"
        pass_file.parent.mkdir(parents=True, exist_ok=True)
        if pass_file.exists():
            passphrase = pass_file.read_text().strip()
        else:
            import logging as _log
            _log.getLogger("kriya.vault").warning(
                "KRIYA_VAULT_PASS not set — generating a machine-local vault passphrase. "
                "Set KRIYA_VAULT_PASS in the environment for production deployments."
            )
            passphrase = secrets.token_hex(32)
            pass_file.write_text(passphrase)
            pass_file.chmod(0o400)

    if key_file.exists():
        raw = key_file.read_bytes()
        # raw = salt(16) + encrypted_key(32)
        salt = raw[:16]
        encrypted = raw[16:]
        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 100_000)
        return _xor(encrypted, key)
    else:
        master = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 100_000)
        key_file.write_bytes(salt + _xor(master, key))
        key_file.chmod(0o600)
        return master


def _xor(data: bytes, key: bytes) -> bytes:
    return bytes(a ^ key[i % len(key)] for i, a in enumerate(data))


def _encrypt(plaintext: str) -> str:
    """Encrypt with AES-256-GCM if available, else HMAC-XOR."""
    master = _get_master_key()
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(12)
        ct = AESGCM(master).encrypt(nonce, plaintext.encode(), None)
        return "aes:" + base64.b64encode(nonce + ct).decode()
    except ImportError:
        # Pure stdlib fallback: XOR + HMAC-SHA256 tag
        nonce = secrets.token_bytes(16)
        stream_key = hashlib.pbkdf2_hmac("sha256", master, nonce, 1000)
        ct = _xor(plaintext.encode(), stream_key)
        tag = hmac.new(master, nonce + ct, "sha256").digest()
        return "xor:" + base64.b64encode(nonce + ct + tag).decode()


def _decrypt(ciphertext: str) -> str:
    master = _get_master_key()
    if ciphertext.startswith("aes:"):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            raw = base64.b64decode(ciphertext[4:])
            nonce, ct = raw[:12], raw[12:]
            return AESGCM(master).decrypt(nonce, ct, None).decode()
        except Exception:
            raise ValueError("Decryption failed")
    elif ciphertext.startswith("xor:"):
        raw = base64.b64decode(ciphertext[4:])
        nonce, ct, tag = raw[:16], raw[16:-32], raw[-32:]
        expected = hmac.new(master, nonce + ct, "sha256").digest()
        if not hmac.compare_digest(expected, tag):
            raise ValueError("HMAC verification failed")
        stream_key = hashlib.pbkdf2_hmac("sha256", master, nonce, 1000)
        return _xor(ct, stream_key).decode()
    else:
        raise ValueError(f"Unknown encryption scheme")


# Per-project secret files: vault/<project_id>/<key>.enc

def set_secret(project_id: str, key: str, value: str):
    d = VAULT_DIR / project_id
    d.mkdir(exist_ok=True)
    (d / f"{key}.enc").write_text(_encrypt(value))
    (d / f"{key}.enc").chmod(0o600)


def get_secret(project_id: str, key: str) -> Optional[str]:
    path = VAULT_DIR / project_id / f"{key}.enc"
    if not path.exists():
        # Fall back to the namespaced env var only — never fall back to arbitrary env keys
        # to prevent agents from extracting JWT secrets, API keys, etc.
        env_key = f"KRIYA_SECRET_{project_id.upper()}_{key.upper()}"
        return os.environ.get(env_key)
    return _decrypt(path.read_text())


def list_secrets(project_id: str) -> list[str]:
    d = VAULT_DIR / project_id
    if not d.exists():
        return []
    return [p.stem for p in d.glob("*.enc")]


def delete_secret(project_id: str, key: str):
    path = VAULT_DIR / project_id / f"{key}.enc"
    if path.exists():
        path.unlink()


def inject_secrets(project_id: str) -> dict[str, str]:
    """Return all secrets for a project as a plain dict (use transiently, never log)."""
    return {k: get_secret(project_id, k) for k in list_secrets(project_id)}
