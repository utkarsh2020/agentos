"""
Kriya – Task DAG scheduler
Resolves dependency graphs, dispatches tasks when all deps are DONE.
Handles cron / @every scheduling.
Pi Zero: pure asyncio, no external scheduler deps.
"""
import asyncio
import json
import logging
import re
import time
import uuid
from typing import Optional

from kriya.core import store
from kriya.core.bus import get_bus, Message, Topics
from kriya.core.agent import AgentRunner, AgentConfig

log = logging.getLogger("kriya.scheduler")


# ── Schedule parsing ───────────────────────────────────────────────────────

def next_run_time(schedule: str, last_run: float = None) -> float:
    """
    Parse schedule string and return next Unix timestamp.
    Supports:
      @every 30s | @every 5m | @every 2h
      @daily | @hourly | @weekly
      (cron is v2 - not yet implemented)
    """
    now = time.time()
    s = schedule.strip().lower()

    if s == "@daily":
        return now + 86400
    if s == "@hourly":
        return now + 3600
    if s == "@weekly":
        return now + 604800
    if s == "@once":
        return now if last_run is None else float("inf")

    m = re.match(r"@every\s+(\d+)(s|m|h|d)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        base = last_run if last_run else now
        return base + n * secs

    # Default: run once immediately
    return now


# ── DAG resolution ─────────────────────────────────────────────────────────

def get_ready_tasks(project_id: str) -> list[dict]:
    """Return tasks whose dependencies are all DONE and are still PENDING."""
    tasks = store.fetch_where("tasks", project_id=project_id)
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}

    ready = []
    for task in tasks:
        if task["status"] != "pending":
            continue
        deps = json.loads(task.get("depends_on") or "[]")
        # Resolve deps by name or id
        dep_ids = set()
        for dep in deps:
            for t in tasks:
                if t["id"] == dep or t["name"] == dep:
                    dep_ids.add(t["id"])
                    break
        if dep_ids.issubset(done_ids):
            ready.append(task)
    return ready


def has_failed_deps(task: dict, project_id: str) -> bool:
    tasks = store.fetch_where("tasks", project_id=project_id)
    failed_ids = {t["id"] for t in tasks if t["status"] == "failed"}
    name_to_id = {t["name"]: t["id"] for t in tasks}
    deps = json.loads(task.get("depends_on") or "[]")
    for dep in deps:
        dep_id = name_to_id.get(dep, dep)
        if dep_id in failed_ids:
            return True
    return False


# ── Task runner ────────────────────────────────────────────────────────────

async def run_task(task: dict) -> bool:
    """
    Execute all agents for a task sequentially (Pi Zero: no parallelism).
    Returns True if all agents succeeded.
    """
    bus = get_bus()
    task_id    = task["id"]
    project_id = task["project_id"]

    store.update("tasks", task_id, status="running", started_at=time.time())
    await bus.publish(Message(Topics.TASK_STARTED, {
        "task_id": task_id, "name": task["name"]
    }))
    log.info(f"[scheduler] task started: {task['name']}")

    config = json.loads(task.get("config") or "{}")
    agents_cfg = config.get("agents", [])

    if not agents_cfg:
        log.warning(f"[scheduler] task {task['name']} has no agents configured")
        store.update("tasks", task_id, status="done", finished_at=time.time(), output='{"note":"no agents"}')
        return True

    # Get previous task outputs for context
    prev_outputs = _collect_context(project_id)

    all_ok = True
    combined_output = {}

    for acfg in agents_cfg:
        agent_id = str(uuid.uuid4())
        # Create agent record
        store.insert("agents",
            id=agent_id,
            task_id=task_id,
            project_id=project_id,
            role=acfg.get("role", "executor"),
            model=acfg.get("model", "auto"),
            provider=acfg.get("provider", "auto"),
            system_prompt=acfg.get("prompt", "You are a helpful AI agent."),
            state="pending",
            created_at=time.time(),
        )

        runner = AgentRunner(AgentConfig(
            id=agent_id,
            task_id=task_id,
            project_id=project_id,
            role=acfg.get("role", "executor"),
            model=acfg.get("model", "auto"),
            provider=acfg.get("provider", "auto"),
            system_prompt=acfg.get("prompt", "You are a helpful AI agent."),
            skills=acfg.get("skills", []),
            max_tokens=acfg.get("max_tokens", 512),
            temperature=acfg.get("temperature", 0.7),
            max_retries=acfg.get("max_retries", 3),
            timeout=acfg.get("timeout", 120),
        ))

        user_input = acfg.get("input", task["name"])
        result = await runner.run(user_input, context=prev_outputs)

        combined_output[acfg.get("id", agent_id)] = result
        if not result.get("success"):
            all_ok = False
            log.error(f"[scheduler] agent {agent_id} failed: {result.get('error')}")
            break   # stop task on first agent failure

    status = "done" if all_ok else "failed"
    store.update("tasks", task_id,
        status=status,
        output=json.dumps(combined_output),
        finished_at=time.time(),
    )
    topic = Topics.TASK_DONE if all_ok else Topics.TASK_FAILED
    await bus.publish(Message(topic, {"task_id": task_id, "status": status}))
    log.info(f"[scheduler] task {task['name']} → {status}")
    return all_ok


def _collect_context(project_id: str) -> str:
    """Collect done task outputs to pass as context to subsequent tasks."""
    tasks = store.raw_query(
        "SELECT name, output FROM tasks WHERE project_id=? AND status='done' ORDER BY finished_at",
        (project_id,)
    )
    parts = []
    for t in tasks:
        if t["output"]:
            out = json.loads(t["output"])
            # Extract text outputs
            for agent_key, res in out.items():
                if isinstance(res, dict) and res.get("output"):
                    parts.append(f"[{t['name']} / {agent_key}]: {res['output'][:300]}")
    return "\n".join(parts) if parts else ""


# ── Project runner ─────────────────────────────────────────────────────────

async def run_project(project_id: str):
    """
    Execute a full project DAG until all tasks are done or one fails.
    """
    bus = get_bus()
    project = store.fetch_one("projects", project_id)
    if not project:
        log.error(f"[scheduler] project {project_id} not found")
        return

    log.info(f"[scheduler] project started: {project['name']}")
    store.update("projects", project_id, status="running", updated_at=time.time())
    await bus.publish(Message(Topics.PROJECT_STARTED, {"project_id": project_id}))

    # Reset task statuses for fresh run
    tasks = store.fetch_where("tasks", project_id=project_id)
    for t in tasks:
        if t["status"] not in ("running",):
            store.update("tasks", t["id"], status="pending", output=None, error=None)

    max_iterations = len(tasks) * 2 + 1  # safety valve
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        ready = get_ready_tasks(project_id)

        if not ready:
            # Check if all done
            tasks = store.fetch_where("tasks", project_id=project_id)
            all_done   = all(t["status"] == "done" for t in tasks)
            any_failed = any(t["status"] == "failed" for t in tasks)

            if all_done:
                store.update("projects", project_id, status="idle", updated_at=time.time())
                await bus.publish(Message(Topics.PROJECT_STOPPED, {
                    "project_id": project_id, "status": "done"
                }))
                log.info(f"[scheduler] project {project['name']} completed ✓")
                return

            if any_failed:
                store.update("projects", project_id, status="idle", updated_at=time.time())
                await bus.publish(Message(Topics.PROJECT_FAILED, {
                    "project_id": project_id
                }))
                log.error(f"[scheduler] project {project['name']} failed")
                return

            # Some tasks still running or blocked – wait
            await asyncio.sleep(1)
            continue

        # Run ready tasks (sequentially on Pi Zero)
        for task in ready:
            if has_failed_deps(task, project_id):
                store.update("tasks", task["id"], status="skipped")
                continue
            success = await run_task(task)
            if not success:
                # Mark remaining tasks as skipped
                remaining = store.fetch_where("tasks", project_id=project_id)
                for t in remaining:
                    if t["status"] == "pending":
                        store.update("tasks", t["id"], status="skipped")
                break

    store.update("projects", project_id, status="idle")
    log.warning(f"[scheduler] project {project['name']} hit max iterations")


# ── Cron scheduler ─────────────────────────────────────────────────────────

class CronScheduler:
    """Checks scheduled projects and triggers them at the right time."""

    def __init__(self):
        self._running = False

    async def start(self):
        self._running = True
        log.info("[cron] scheduler started")
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                log.exception(f"[cron] tick error: {e}")
            await asyncio.sleep(10)   # check every 10s – fine for Pi Zero

    async def _tick(self):
        now = time.time()
        jobs = store.raw_query(
            "SELECT * FROM scheduled_jobs WHERE enabled=1 AND next_run<=?", (now,)
        )
        for job in jobs:
            project_id = job["project_id"]
            schedule   = job["schedule"]

            # Fire project
            log.info(f"[cron] triggering project {project_id}")
            asyncio.get_event_loop().create_task(run_project(project_id))

            # Update next_run
            next_run = next_run_time(schedule, last_run=now)
            store.raw_query(
                "UPDATE scheduled_jobs SET last_run=?, next_run=? WHERE id=?",
                (now, next_run, job["id"])
            )
            store._conn().commit()

    def stop(self):
        self._running = False
