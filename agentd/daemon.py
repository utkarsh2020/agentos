"""
AgentOS – Main daemon (agentd)
Boots: DB → skills → event bus → API server → cron scheduler → heartbeat
Single process, asyncio event loop. Pi Zero optimised.
"""
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure the repo root is on the path when run directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agentd.core.config import get_config, BASE_DIR, PID_FILE
from agentd.core import store
from agentd.core.bus import get_bus, Message, Topics
from agentd.core.scheduler import CronScheduler
from agentd.api.server import start_api_server
from agentd.integrations.builtin_skills import register_builtin_skills

# ── Logging ────────────────────────────────────────────────────────────────

def _setup_logging(level: str):
    fmt = "%(asctime)s %(levelname)-5s %(name)s │ %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt)
    # Quiet noisy stdlib loggers
    for name in ("urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


log = logging.getLogger("agentd")

def _arch_label() -> str:
    import platform, struct
    machine = platform.machine()
    bits    = struct.calcsize("P") * 8
    return {
        "armv6l":  "ARMv6 32-bit (Pi Zero W)",
        "armv7l":  "ARMv7 32-bit (Pi Zero 2W / Pi 3)",
        "aarch64": "ARM64 64-bit (Pi 4 / Pi 5)",
        "x86_64":  "x86_64 64-bit",
        "AMD64":   "x86_64 64-bit",
    }.get(machine, f"{machine} {bits}-bit")


def _build_banner() -> str:
    import platform
    arch = _arch_label()
    py   = platform.python_version()
    return (
        "\n"
        "   _                    _    ___  ____ \n"
        "  /_\\   __ _  ___ _ __ | |_ / _ \\/ ___|\n"
        " //_\\\\ / _` |/ _ \\ \'_\'| __| | | \\___ \\\n"
        "/  _  \\ (_| |  __/ | | | |_| |_| |___) |\n"
        "\\_/ \\_/\\__, |\\___|_| |_|\\__|\\___/|____/\n"
        "       |___/   Agentic AI Operating System  v0.2.0\n"
        f"       Arch  : {arch}\n"
        f"       Python: {py}\n"
    )


BANNER = _build_banner()



# ── PID file ───────────────────────────────────────────────────────────────

def _write_pid():
    PID_FILE.write_text(str(os.getpid()))

def _clear_pid():
    PID_FILE.unlink(missing_ok=True)


# ── Heartbeat ──────────────────────────────────────────────────────────────

async def _heartbeat(bus, interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        await bus.publish(Message(Topics.HEARTBEAT, {
            "ts": time.time(),
            "pid": os.getpid(),
        }))


# ── Graceful shutdown ──────────────────────────────────────────────────────

_shutdown_event: asyncio.Event = None


def _handle_signal(sig, frame):
    log.info(f"[daemon] received signal {sig} – shutting down")
    if _shutdown_event:
        _shutdown_event.set()


# ── Boot sequence ──────────────────────────────────────────────────────────

async def boot():
    global _shutdown_event
    cfg = get_config()
    _setup_logging(cfg.log_level)
    print(BANNER)

    log.info(f"[daemon] base dir : {BASE_DIR}")
    log.info(f"[daemon] database  : {store.DB_PATH}")
    log.info(f"[daemon] api       : http://{cfg.host}:{cfg.port}")

    # 1. Initialise database
    log.info("[boot] initialising database …")
    store.init_db()

    # 2. Register built-in skills
    log.info("[boot] loading skills …")
    register_builtin_skills()

    # 3. Load any external skill plugins from /opt/skills or skills/
    _load_plugin_skills()

    # 4. Start event bus (it's pure async, no setup needed)
    bus = get_bus()
    log.info("[boot] event bus ready")

    # 5. Start REST API in background thread
    loop = asyncio.get_event_loop()
    start_api_server(cfg.host, cfg.port, loop)

    # 6. Start cron scheduler
    cron = CronScheduler()
    asyncio.create_task(cron.start())
    log.info("[boot] cron scheduler started")

    # 7. Heartbeat
    asyncio.create_task(_heartbeat(bus))

    # 8. Signal handlers
    _shutdown_event = asyncio.Event()
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _write_pid()
    log.info(f"[daemon] ✓ AgentOS running  (PID {os.getpid()})")
    log.info(f"[daemon]   API → http://{cfg.host}:{cfg.port}/api/status")
    log.info(f"[daemon]   CLI → agent --help")
    log.info(f"[daemon]   Press Ctrl+C to stop")

    # 9. Publish startup event
    await bus.publish(Message("system.startup", {
        "pid": os.getpid(),
        "host": cfg.host,
        "port": cfg.port,
        "ts": time.time(),
    }))

    # 10. Wait for shutdown signal
    await _shutdown_event.wait()

    log.info("[daemon] shutting down …")
    cron.stop()
    _clear_pid()
    log.info("[daemon] goodbye")


def _load_plugin_skills():
    """
    Scan skills/ directory for handler.py files and auto-register them.
    Each handler.py must call register_skill() at module level or define
    a SKILL_ID and a handle(params, secrets) -> dict function.
    """
    import importlib.util
    skills_dir = BASE_DIR / "skills"
    if not skills_dir.exists():
        return

    for handler_path in skills_dir.glob("*/handler.py"):
        skill_dir = handler_path.parent
        skill_id  = skill_dir.name
        try:
            spec = importlib.util.spec_from_file_location(
                f"skills.{skill_id}", handler_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Auto-register if module defines handle() and SKILL_ID
            if hasattr(mod, "handle") and hasattr(mod, "SKILL_ID"):
                from agentd.core.agent import register_skill
                register_skill(mod.SKILL_ID, mod.handle)
                log.info(f"[skills] loaded plugin: {mod.SKILL_ID}")
        except Exception as e:
            log.warning(f"[skills] failed to load {handler_path}: {e}")


def main():
    asyncio.run(boot())


if __name__ == "__main__":
    main()
