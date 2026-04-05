"""
AgentOS – TOML project loader
Parses project definition files and upserts into the DB.
"""
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from agentd.core import store
from agentd.core.scheduler import next_run_time


def load_toml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")
    if sys.version_info < (3, 11):
        raise RuntimeError("TOML loading requires Python 3.11+")
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def import_project(path: str | Path) -> str:
    """
    Parse a .toml project file and upsert project + tasks into DB.
    Returns project ID.
    """
    data = load_toml(path)
    proj_cfg = data.get("project", {})
    name = proj_cfg.get("name")
    if not name:
        raise ValueError("Project file must define [project] name")

    # Upsert project
    existing = store.raw_query("SELECT id FROM projects WHERE name=?", (name,))
    if existing:
        pid = existing[0]["id"]
        store.update("projects", pid,
            description=proj_cfg.get("description", ""),
            schedule=proj_cfg.get("schedule"),
            config_toml=open(path).read(),
            updated_at=time.time(),
        )
        # Reset tasks
        store.raw_query("DELETE FROM tasks WHERE project_id=?", (pid,))
    else:
        pid = store.insert("projects",
            name=name,
            description=proj_cfg.get("description", ""),
            schedule=proj_cfg.get("schedule"),
            config_toml=open(path).read(),
            status="idle",
            created_at=time.time(),
            updated_at=time.time(),
        )

    # Resolve agent configs by id
    agents_by_id = {}
    for ag in data.get("agents", []):
        agents_by_id[ag["id"]] = ag

    # Import tasks
    task_name_to_id = {}
    for task_key, task_cfg in data.get("tasks", {}).items():
        agent_ids = task_cfg.get("agents", [])
        agents_list = []
        for aid in agent_ids:
            ag = agents_by_id.get(aid, {})
            if ag:
                agents_list.append({
                    "id":          ag.get("id", aid),
                    "role":        ag.get("role", "executor"),
                    "model":       ag.get("model", "auto"),
                    "provider":    ag.get("provider", "auto"),
                    "prompt":      ag.get("prompt", "You are a helpful AI agent."),
                    "skills":      ag.get("skills", []),
                    "max_tokens":  ag.get("max_tokens", 512),
                    "temperature": ag.get("temperature", 0.7),
                    "max_retries": ag.get("max_retries", 3),
                    "timeout":     ag.get("timeout", 120),
                })

        tid = store.insert("tasks",
            project_id=pid,
            name=task_cfg.get("name", task_key),
            depends_on=json.dumps(task_cfg.get("depends_on", [])),
            config=json.dumps({"agents": agents_list}),
            status="pending",
            created_at=time.time(),
        )
        task_name_to_id[task_key] = tid

    # Register schedule
    sched = proj_cfg.get("schedule")
    if sched:
        existing_sched = store.raw_query(
            "SELECT id FROM scheduled_jobs WHERE project_id=?", (pid,)
        )
        if existing_sched:
            store.raw_query(
                "UPDATE scheduled_jobs SET schedule=?, next_run=? WHERE project_id=?",
                (sched, next_run_time(sched), pid)
            )
        else:
            store.raw_query(
                "INSERT INTO scheduled_jobs (id, project_id, schedule, next_run, enabled) VALUES (?,?,?,?,?)",
                (str(__import__("uuid").uuid4()), pid, sched, next_run_time(sched), 1)
            )
        store._conn().commit()

    return pid
