"""
AgentOS – Agent executor
Runs one agent turn: load memory → build prompt → call LLM → parse output → persist.
Handles tool/skill calls embedded in LLM output (JSON action blocks).
Pi Zero: single-threaded async, no subprocess spawning.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

from agentd.core import store
from agentd.core.bus import get_bus, Message, Topics
from agentd.ai.llm import call_llm, LLMMessage, LLMResponse, LLMError
from agentd.ai.memory import get_short_term, get_long_term, evict_agent
from agentd.security.vault import inject_secrets

log = logging.getLogger("agentd.agent")


# ── Agent definition ───────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    id: str
    task_id: str
    project_id: str
    role: str               # executor | planner | critic
    model: str              # e.g. "anthropic/claude-3-5-haiku-20241022"
    provider: str           # anthropic | openai | ollama | auto
    system_prompt: str
    skills: list[str] = field(default_factory=list)
    max_tokens: int = 512   # Pi Zero: keep low
    temperature: float = 0.7
    max_retries: int = 3
    timeout: int = 120


# ── Skill call protocol ────────────────────────────────────────────────────
# LLM outputs a JSON action block when it wants to call a skill:
#   {"action": "skill_call", "skill": "gmail.read", "params": {...}}
# We detect this, execute the skill, and feed result back.

ACTION_PREFIX = '{"action":'

def _extract_action(text: str) -> Optional[dict]:
    """Find and parse a JSON action block in LLM output."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(ACTION_PREFIX) or line.startswith("```json"):
            try:
                clean = line.replace("```json", "").replace("```", "").strip()
                obj = json.loads(clean)
                if obj.get("action") == "skill_call":
                    return obj
            except (json.JSONDecodeError, AttributeError):
                pass
    # Try full text
    try:
        start = text.find(ACTION_PREFIX)
        if start != -1:
            bracket = 0
            for i, ch in enumerate(text[start:], start):
                if ch == '{': bracket += 1
                elif ch == '}':
                    bracket -= 1
                    if bracket == 0:
                        return json.loads(text[start:i+1])
    except Exception:
        pass
    return None


# ── Skill registry (lazy loaded) ──────────────────────────────────────────

_skill_handlers: dict[str, Callable] = {}


def register_skill(skill_id: str, handler: Callable):
    _skill_handlers[skill_id] = handler


async def _call_skill(skill_id: str, params: dict, secrets: dict) -> dict:
    handler = _skill_handlers.get(skill_id)
    if not handler:
        return {"error": f"Skill '{skill_id}' not installed"}
    try:
        if asyncio.iscoroutinefunction(handler):
            result = await handler(params, secrets)
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: handler(params, secrets)
            )
        return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


# ── Build system prompt ────────────────────────────────────────────────────

SKILL_INSTRUCTIONS = """
When you need to use a tool, output a JSON action block on its own line:
{"action": "skill_call", "skill": "<skill_id>", "params": {<params>}}

Available skills: {skills}
After the action block, wait – the result will be provided to you.
If no skills are needed, just respond normally.
""".strip()


def _build_system_prompt(config: AgentConfig, context: str = "", memories: list = None) -> str:
    parts = [config.system_prompt]

    if config.role == "planner":
        parts.append(
            "\nYou are a PLANNER agent. Decompose the task into a JSON array of subtasks:\n"
            '[{"name":"...", "description":"...", "depends_on":[]}, ...]'
        )
    elif config.role == "critic":
        parts.append(
            "\nYou are a CRITIC agent. Review the provided output for quality, "
            "accuracy, and completeness. Return JSON: {\"approved\": bool, \"feedback\": \"...\", \"corrected\": <corrected_output_or_null>}"
        )

    if config.skills:
        parts.append("\n" + SKILL_INSTRUCTIONS.format(skills=", ".join(config.skills)))

    if memories:
        mem_text = "\n".join(f"- {m['content']}" for m in memories[:5])
        parts.append(f"\nRelevant context from memory:\n{mem_text}")

    if context:
        parts.append(f"\nTask context:\n{context}")

    return "\n\n".join(parts)


# ── Agent runner ───────────────────────────────────────────────────────────

class AgentRunner:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.stm = get_short_term(config.id)
        self.ltm = get_long_term(config.project_id)
        self.bus = get_bus()
        self._secrets: dict = {}

    async def run(self, user_input: str, context: str = "") -> dict:
        cfg = self.config
        bus = self.bus

        # Mark running
        store.update("agents", cfg.id, state="running", started_at=time.time())
        await bus.publish(Message(Topics.AGENT_SPAWNED, {
            "agent_id": cfg.id, "task_id": cfg.task_id, "role": cfg.role
        }, from_id=cfg.id))

        # Load secrets
        self._secrets = inject_secrets(cfg.project_id)

        # Recall relevant long-term memories
        memories = self.ltm.recall(user_input, top_k=4)

        # Build system prompt
        system = _build_system_prompt(cfg, context, memories)
        self.stm.add("system", system)

        # Add user input
        self.stm.add("user", user_input)

        result = await self._execute_turn(retries=cfg.max_retries)

        # Persist important output to long-term memory
        if result.get("success") and result.get("output"):
            summary = result["output"][:500]
            self.ltm.remember(summary, agent_id=cfg.id, importance=0.8)

        # Finalise
        state  = "done" if result.get("success") else "failed"
        output = json.dumps(result)
        store.update("agents", cfg.id,
            state=state,
            output=output,
            finished_at=time.time(),
            token_usage=result.get("total_tokens", 0),
        )

        topic = Topics.AGENT_DONE if result.get("success") else Topics.AGENT_FAILED
        await bus.publish(Message(topic, {
            "agent_id": cfg.id, "task_id": cfg.task_id, "result": result,
        }, from_id=cfg.id))

        evict_agent(cfg.id)
        return result

    async def _execute_turn(self, retries: int) -> dict:
        cfg = self.config
        for attempt in range(retries):
            try:
                return await asyncio.wait_for(
                    self._one_turn(), timeout=cfg.timeout
                )
            except asyncio.TimeoutError:
                log.warning(f"[agent:{cfg.id}] timeout on attempt {attempt+1}")
                if attempt == retries - 1:
                    return {"success": False, "error": "Agent timed out", "total_tokens": 0}
                await asyncio.sleep(2 ** attempt)   # exponential backoff
            except LLMError as e:
                log.warning(f"[agent:{cfg.id}] LLM error: {e.message}")
                if attempt == retries - 1:
                    return {"success": False, "error": e.message, "total_tokens": 0}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                log.exception(f"[agent:{cfg.id}] unexpected error")
                return {"success": False, "error": str(e), "total_tokens": 0}
        return {"success": False, "error": "Max retries exceeded", "total_tokens": 0}

    async def _one_turn(self) -> dict:
        cfg = self.config
        total_tokens = 0
        max_skill_calls = 5  # prevent infinite loops

        for _ in range(max_skill_calls + 1):
            # Build message list for LLM
            msgs = [
                LLMMessage(role=m["role"], content=m["content"])
                for m in self.stm.get_messages()
                if m["role"] in ("system", "user", "assistant")
            ]

            # Call LLM (run in thread to not block asyncio)
            resp: LLMResponse = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: call_llm(
                    msgs,
                    provider=cfg.provider,
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
            )
            total_tokens += resp.input_tokens + resp.output_tokens

            # Persist assistant message
            self.stm.add("assistant", resp.content)

            # Publish message event
            await self.bus.publish(Message(Topics.AGENT_MESSAGE, {
                "agent_id": cfg.id,
                "content":  resp.content[:200],
                "tokens":   resp.output_tokens,
            }, from_id=cfg.id))

            # Check for skill call
            action = _extract_action(resp.content)
            if not action:
                # Final response
                return {
                    "success":      True,
                    "output":       resp.content,
                    "total_tokens": total_tokens,
                    "model":        resp.model,
                    "provider":     resp.provider,
                    "latency_ms":   resp.latency_ms,
                }

            # Execute skill call
            skill_id = action.get("skill", "")
            params   = action.get("params", {})
            log.info(f"[agent:{cfg.id}] calling skill {skill_id}")

            await self.bus.publish(Message(Topics.SKILL_CALLED, {
                "agent_id": cfg.id, "skill": skill_id, "params": params,
            }, from_id=cfg.id))

            skill_result = await _call_skill(skill_id, params, self._secrets)

            await self.bus.publish(Message(Topics.SKILL_RESULT, {
                "agent_id": cfg.id, "skill": skill_id, "result": skill_result,
            }, from_id=cfg.id))

            # Feed result back as user message
            self.stm.add("user", f"Skill result for {skill_id}:\n{json.dumps(skill_result, indent=2)}")

        return {"success": False, "error": "Max skill call iterations exceeded", "total_tokens": total_tokens}
