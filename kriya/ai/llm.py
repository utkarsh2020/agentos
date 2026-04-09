"""
Kriya – LLM abstraction layer
Supports: Anthropic (Claude), OpenAI, Ollama
Pure urllib.request – zero external deps.
Pi Zero: keep max_tokens conservative to avoid OOM.
"""
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, Iterator

from kriya.core.config import get_config, LLMProviderConfig


@dataclass
class LLMMessage:
    role: str     # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


@dataclass
class LLMError(Exception):
    provider: str
    message: str
    status_code: int = 0


# ── HTTP helper ────────────────────────────────────────────────────────────

def _post(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        raise LLMError(
            provider="unknown",
            message=f"HTTP {e.code}: {body_text[:200]}",
            status_code=e.code,
        )
    except urllib.error.URLError as e:
        raise LLMError(provider="unknown", message=f"Connection error: {e.reason}")


# ── Provider implementations ───────────────────────────────────────────────

def _call_anthropic(cfg: LLMProviderConfig, messages: list[LLMMessage],
                    max_tokens: int, temperature: float) -> LLMResponse:
    # Separate system message
    system = next((m.content for m in messages if m.role == "system"), None)
    user_msgs = [{"role": m.role, "content": m.content}
                 for m in messages if m.role != "system"]

    body = {
        "model": cfg.default_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": user_msgs,
    }
    if system:
        body["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": cfg.api_key,
        "anthropic-version": "2023-06-01",
    }

    t0 = time.monotonic()
    resp = _post(f"{cfg.base_url}/v1/messages", headers, body, timeout=cfg.timeout)
    latency = int((time.monotonic() - t0) * 1000)

    return LLMResponse(
        content=resp["content"][0]["text"],
        model=resp.get("model", cfg.default_model),
        provider="anthropic",
        input_tokens=resp.get("usage", {}).get("input_tokens", 0),
        output_tokens=resp.get("usage", {}).get("output_tokens", 0),
        latency_ms=latency,
    )


def _call_openai(cfg: LLMProviderConfig, messages: list[LLMMessage],
                 max_tokens: int, temperature: float) -> LLMResponse:
    body = {
        "model": cfg.default_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    t0 = time.monotonic()
    resp = _post(f"{cfg.base_url}/v1/chat/completions", headers, body, timeout=cfg.timeout)
    latency = int((time.monotonic() - t0) * 1000)

    return LLMResponse(
        content=resp["choices"][0]["message"]["content"],
        model=resp.get("model", cfg.default_model),
        provider="openai",
        input_tokens=resp.get("usage", {}).get("prompt_tokens", 0),
        output_tokens=resp.get("usage", {}).get("completion_tokens", 0),
        latency_ms=latency,
    )


def _call_ollama(cfg: LLMProviderConfig, messages: list[LLMMessage],
                 max_tokens: int, temperature: float) -> LLMResponse:
    # Ollama uses OpenAI-compatible /api/chat endpoint
    body = {
        "model": cfg.default_model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    headers = {"Content-Type": "application/json"}

    t0 = time.monotonic()
    resp = _post(f"{cfg.base_url}/api/chat", headers, body, timeout=cfg.timeout)
    latency = int((time.monotonic() - t0) * 1000)

    content = resp.get("message", {}).get("content", "")
    usage = resp.get("eval_count", 0)
    return LLMResponse(
        content=content,
        model=cfg.default_model,
        provider="ollama",
        input_tokens=resp.get("prompt_eval_count", 0),
        output_tokens=usage,
        latency_ms=latency,
    )


# ── Router ─────────────────────────────────────────────────────────────────

_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai":    _call_openai,
    "ollama":    _call_ollama,
}


def call_llm(
    messages: list[LLMMessage],
    provider: str = "auto",
    model: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    fallback: bool = True,
) -> LLMResponse:
    """
    Call an LLM. provider="auto" tries providers in priority order.
    Pi Zero tip: use max_tokens=256-512 for fast local responses.
    """
    cfg_all = get_config()
    providers_cfg = {p.name: p for p in cfg_all.providers if p.enabled}

    if provider == "auto":
        order = ["anthropic", "openai", "ollama"]
    else:
        order = [provider]

    last_err = None
    for pname in order:
        pcfg = providers_cfg.get(pname)
        if not pcfg:
            continue

        # Override model if specified (e.g. "anthropic/claude-3-opus")
        if model and "/" in model:
            _, mname = model.split("/", 1)
            pcfg = LLMProviderConfig(**{**pcfg.__dict__, "default_model": mname})
        elif model:
            pcfg = LLMProviderConfig(**{**pcfg.__dict__, "default_model": model})

        fn = _PROVIDERS.get(pname)
        if not fn:
            continue
        try:
            return fn(pcfg, messages, max_tokens, temperature)
        except LLMError as e:
            last_err = e
            if not fallback:
                raise
            continue
        except Exception as e:
            last_err = LLMError(provider=pname, message=str(e))
            if not fallback:
                raise
            continue

    raise last_err or LLMError(provider="none", message="No LLM providers available")


def available_providers() -> list[str]:
    return [p.name for p in get_config().providers if p.enabled]
