"""
Kriya – Memory system
Short-term: bounded LRU message buffer per agent (in-process)
Long-term:  cosine similarity search over SQLite JSON embeddings
            Uses tiny hash-based embeddings (no ML deps) - Pi Zero safe.
            Upgrade path: plug in a real embedding model via Ollama later.
"""
import hashlib
import json
import math
import time
import uuid
from collections import OrderedDict
from typing import Optional

from kriya.core import store
from kriya.core.config import get_config


# ── Tiny embeddings ────────────────────────────────────────────────────────
# On Pi Zero we can't run sentence-transformers. Instead we use a 64-dim
# hash-based embedding. It's not semantic but captures token overlap well
# enough for keyword-style retrieval. Replace _embed() when Ollama is
# available (ollama.embeddings endpoint).

DIMS = 64


def _embed(text: str) -> list[float]:
    """Deterministic 64-dim embedding from character n-grams."""
    text = text.lower().strip()
    vec = [0.0] * DIMS
    words = text.split()
    for token in words:
        for n in (1, 2, 3):
            for i in range(len(token) - n + 1):
                gram = token[i:i+n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx = h % DIMS
                vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a)) or 1.0
    nb  = math.sqrt(sum(x*x for x in b)) or 1.0
    return dot / (na * nb)


# ── Short-term memory ──────────────────────────────────────────────────────

class ShortTermMemory:
    """Bounded FIFO message buffer. One per agent."""

    def __init__(self, agent_id: str, capacity: int = None):
        self.agent_id = agent_id
        self.capacity = capacity or get_config().short_term_capacity
        self._messages: list[dict] = []

    def add(self, role: str, content: str):
        msg = {"id": str(uuid.uuid4()), "role": role, "content": content, "ts": time.time()}
        self._messages.append(msg)
        # Trim – keep system message + last (capacity) non-system messages
        system = [m for m in self._messages if m["role"] == "system"]
        non_system = [m for m in self._messages if m["role"] != "system"]
        if len(non_system) > self.capacity:
            non_system = non_system[-(self.capacity):]
        self._messages = system + non_system
        # Persist to DB (best-effort)
        try:
            store.insert("agent_messages",
                id=msg["id"], agent_id=self.agent_id,
                role=role, content=content, created_at=time.time(),
            )
        except Exception:
            pass

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def load_from_db(self):
        """Reload history from previous run."""
        rows = store.raw_query(
            "SELECT * FROM agent_messages WHERE agent_id=? ORDER BY created_at",
            (self.agent_id,)
        )
        self._messages = [dict(r) for r in rows]

    def clear(self):
        self._messages = []

    def __len__(self):
        return len(self._messages)


# ── Long-term memory ───────────────────────────────────────────────────────

class LongTermMemory:
    """Project-scoped vector memory backed by SQLite."""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def remember(
        self,
        content: str,
        agent_id: str = None,
        importance: float = 1.0,
    ) -> str:
        """Store a memory. Returns memory ID."""
        emb = _embed(content)
        mid = str(uuid.uuid4())
        store.insert("memory",
            id=mid,
            project_id=self.project_id,
            agent_id=agent_id,
            content=content,
            embedding=json.dumps(emb),
            importance=importance,
            created_at=time.time(),
        )
        return mid

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[dict]:
        """Return top_k most relevant memories for query."""
        q_emb = _embed(query)
        rows = store.raw_query(
            "SELECT id, content, embedding, importance, created_at "
            "FROM memory WHERE project_id=? ORDER BY created_at DESC LIMIT 500",
            (self.project_id,)
        )
        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            score = _cosine(q_emb, emb) * row["importance"]
            if score >= min_score:
                scored.append({
                    "id":       row["id"],
                    "content":  row["content"],
                    "score":    round(score, 4),
                    "created_at": row["created_at"],
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def forget(self, memory_id: str):
        store.delete("memory", memory_id)

    def forget_all(self):
        store.raw_query(
            "DELETE FROM memory WHERE project_id=?", (self.project_id,)
        )

    def count(self) -> int:
        return store.raw_query(
            "SELECT COUNT(*) as n FROM memory WHERE project_id=?",
            (self.project_id,)
        )[0]["n"]


# ── Memory manager (singleton cache of instances) ─────────────────────────

_short_term: dict[str, ShortTermMemory] = {}
_long_term:  dict[str, LongTermMemory]  = {}


def get_short_term(agent_id: str) -> ShortTermMemory:
    if agent_id not in _short_term:
        _short_term[agent_id] = ShortTermMemory(agent_id)
    return _short_term[agent_id]


def get_long_term(project_id: str) -> LongTermMemory:
    if project_id not in _long_term:
        _long_term[project_id] = LongTermMemory(project_id)
    return _long_term[project_id]


def evict_agent(agent_id: str):
    """Release short-term memory after agent terminates."""
    _short_term.pop(agent_id, None)
