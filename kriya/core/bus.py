"""
Kriya – Event bus
Async pub/sub backed by asyncio queues.
All cross-agent communication routes through here.
Persists events to SQLite for audit log.
Pi Zero safe: pure asyncio, no external deps.
"""
import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Callable, Awaitable, Any, Optional

from kriya.core import store


# ── Message schema ─────────────────────────────────────────────────────────

class Message:
    __slots__ = ("id", "topic", "from_id", "to_id", "type", "payload", "trace_id", "created_at")

    def __init__(
        self,
        topic: str,
        payload: dict,
        from_id: str = "system",
        to_id: str = "*",
        type: str = "EVENT",
        trace_id: str = None,
    ):
        self.id         = str(uuid.uuid4())
        self.topic      = topic
        self.from_id    = from_id
        self.to_id      = to_id
        self.type       = type          # EVENT | REQUEST | RESPONSE | SIGNAL
        self.payload    = payload
        self.trace_id   = trace_id or str(uuid.uuid4())
        self.created_at = time.time()

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return f"<Message {self.type} topic={self.topic} from={self.from_id}>"


# ── Bus ────────────────────────────────────────────────────────────────────

class EventBus:
    def __init__(self):
        # topic → list of asyncio.Queue
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._persist = True   # write events to SQLite audit log

    async def publish(self, msg: Message):
        """Publish to all subscribers of msg.topic and wildcard '*'."""
        if self._persist:
            try:
                store.append_event(msg.topic, msg.to_dict(), msg.from_id)
            except Exception:
                pass  # never let audit logging crash the bus

        targets = []
        async with self._lock:
            targets = list(self._subscribers.get(msg.topic, []))
            targets += list(self._subscribers.get("*", []))

        for q in targets:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # drop if consumer is too slow

    async def subscribe(self, topic: str, maxsize: int = 256) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subscribers[topic].append(q)
        return q

    async def unsubscribe(self, topic: str, q: asyncio.Queue):
        async with self._lock:
            subs = self._subscribers.get(topic, [])
            if q in subs:
                subs.remove(q)

    async def request(
        self,
        topic: str,
        payload: dict,
        from_id: str,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """
        Publish a REQUEST and wait for a RESPONSE on a private reply topic.
        """
        reply_topic = f"reply.{uuid.uuid4()}"
        reply_q = await self.subscribe(reply_topic)
        msg = Message(topic, payload, from_id=from_id, type="REQUEST")
        msg.payload["_reply_to"] = reply_topic
        await self.publish(msg)
        try:
            return await asyncio.wait_for(reply_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            await self.unsubscribe(reply_topic, reply_q)

    def emit_nowait(self, topic: str, payload: dict, from_id: str = "system"):
        """Fire-and-forget from sync code."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda: loop.create_task(
                    self.publish(Message(topic, payload, from_id=from_id))
                )
            )
        except RuntimeError:
            # No running loop – store directly
            store.append_event(topic, payload, from_id)


# ── Well-known topics ──────────────────────────────────────────────────────
class Topics:
    # Project lifecycle
    PROJECT_STARTED  = "project.started"
    PROJECT_STOPPED  = "project.stopped"
    PROJECT_FAILED   = "project.failed"

    # Task lifecycle
    TASK_READY      = "task.ready"
    TASK_STARTED    = "task.started"
    TASK_DONE       = "task.done"
    TASK_FAILED     = "task.failed"
    TASK_PAUSED     = "task.paused"   # human-in-the-loop

    # Agent lifecycle
    AGENT_SPAWNED   = "agent.spawned"
    AGENT_DONE      = "agent.done"
    AGENT_FAILED    = "agent.failed"
    AGENT_MESSAGE   = "agent.message"  # inter-agent communication

    # Skill events
    SKILL_CALLED    = "skill.called"
    SKILL_RESULT    = "skill.result"
    SKILL_ERROR     = "skill.error"

    # System
    SHUTDOWN        = "system.shutdown"
    HEARTBEAT       = "system.heartbeat"


# ── Global singleton ───────────────────────────────────────────────────────
_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
