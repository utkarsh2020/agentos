#!/usr/bin/env python3
"""
Kriya test suite – runs without a daemon, in-process.
Tests: config, store, security, memory, LLM layer, scheduler, API, skills.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Point to our source
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use a temp DB for all tests
_TMP = tempfile.mkdtemp()
os.environ["KRIYA_BASE"] = _TMP
os.environ["KRIYA_JWT_SECRET"] = "test-secret-key-for-testing-only"
os.environ["KRIYA_VAULT_PASS"] = "test-vault-passphrase"

from kriya.core.config import get_config, BASE_DIR
from kriya.core import store
from kriya.security.vault import (
    hash_password, verify_password, issue_token, verify_token,
    set_secret, get_secret, list_secrets, delete_secret, has_capability,
)
from kriya.ai.memory import ShortTermMemory, LongTermMemory, _embed, _cosine
from kriya.core.bus import EventBus, Message, Topics


# ── Colours ───────────────────────────────────────────────────────────────

G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; D = "\033[2m"; X = "\033[0m"
PASS = f"{G}PASS{X}"; FAIL = f"{R}FAIL{X}"


class KriyaTestCase(unittest.TestCase):

    def setUp(self):
        store.init_db()

    # ── Config ──────────────────────────────────────────────────────────

    def test_config_loads(self):
        cfg = get_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.jwt_secret, "test-secret-key-for-testing-only")
        self.assertGreater(cfg.max_concurrent_agents, 0)

    def test_base_dir_is_temp(self):
        self.assertEqual(str(BASE_DIR), _TMP)

    # ── Store ────────────────────────────────────────────────────────────

    def test_store_init(self):
        # Tables should exist after init
        tables = store.raw_query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        names = {t["name"] for t in tables}
        for t in ("projects", "tasks", "agents", "events", "memory", "users", "skills"):
            self.assertIn(t, names, f"Missing table: {t}")

    def test_store_crud(self):
        pid = store.insert("projects",
            name="test-project-crud",
            description="test",
            status="idle",
            created_at=time.time(),
            updated_at=time.time(),
        )
        self.assertTrue(len(pid) > 8)

        p = store.fetch_one("projects", pid)
        self.assertEqual(p["name"], "test-project-crud")

        store.update("projects", pid, status="running", updated_at=time.time())
        p2 = store.fetch_one("projects", pid)
        self.assertEqual(p2["status"], "running")

        store.delete("projects", pid)
        self.assertIsNone(store.fetch_one("projects", pid))

    def test_store_fetch_where(self):
        store.insert("projects", name="proj-fw-1", status="idle",
                     created_at=time.time(), updated_at=time.time())
        store.insert("projects", name="proj-fw-2", status="running",
                     created_at=time.time(), updated_at=time.time())
        rows = store.fetch_where("projects", status="running")
        names = [r["name"] for r in rows]
        self.assertIn("proj-fw-2", names)

    # ── Security – passwords ─────────────────────────────────────────────

    def test_password_hash_verify(self):
        h = hash_password("mysecretpassword")
        self.assertTrue(verify_password("mysecretpassword", h))
        self.assertFalse(verify_password("wrongpassword", h))

    def test_password_legacy_sha256(self):
        import hashlib
        legacy = hashlib.sha256("legacypass".encode()).hexdigest()
        self.assertTrue(verify_password("legacypass", legacy))
        self.assertFalse(verify_password("wrong", legacy))

    # ── Security – JWT ───────────────────────────────────────────────────

    def test_jwt_issue_verify(self):
        token = issue_token("user-123", "alice", "admin")
        self.assertIsNotNone(token)
        self.assertEqual(token.count("."), 2)

        claims = verify_token(token)
        self.assertIsNotNone(claims)
        self.assertEqual(claims["sub"], "user-123")
        self.assertEqual(claims["usr"], "alice")
        self.assertEqual(claims["rol"], "admin")

    def test_jwt_invalid(self):
        bad_token = "invalid.token.value"
        self.assertIsNone(verify_token(bad_token))

    def test_jwt_tampered(self):
        token = issue_token("u1", "bob", "read_only")
        parts = token.split(".")
        # Tamper payload
        parts[1] = parts[1][:-2] + "AA"
        self.assertIsNone(verify_token(".".join(parts)))

    # ── Security – RBAC ──────────────────────────────────────────────────

    def test_rbac_capabilities(self):
        self.assertTrue(has_capability("admin", "project:write"))
        self.assertTrue(has_capability("admin", "anything"))
        self.assertTrue(has_capability("project_owner", "task:write"))
        self.assertFalse(has_capability("read_only", "task:write"))
        self.assertFalse(has_capability("read_only", "project:write"))
        self.assertTrue(has_capability("read_only", "project:read"))

    # ── Vault ────────────────────────────────────────────────────────────

    def test_vault_set_get(self):
        set_secret("proj-vault-test", "MY_API_KEY", "sk-secret-value-123")
        val = get_secret("proj-vault-test", "MY_API_KEY")
        self.assertEqual(val, "sk-secret-value-123")

    def test_vault_list_delete(self):
        set_secret("proj-vault-list", "KEY_A", "val-a")
        set_secret("proj-vault-list", "KEY_B", "val-b")
        keys = list_secrets("proj-vault-list")
        self.assertIn("KEY_A", keys)
        self.assertIn("KEY_B", keys)

        delete_secret("proj-vault-list", "KEY_A")
        keys2 = list_secrets("proj-vault-list")
        self.assertNotIn("KEY_A", keys2)
        self.assertIn("KEY_B", keys2)

    def test_vault_env_fallback(self):
        # Namespaced env var (KRIYA_SECRET_{PROJECT}_{KEY}) should be returned
        os.environ["KRIYA_SECRET_MYPROJ_MY_ENV_SECRET"] = "from-namespaced-env"
        val = get_secret("myproj", "MY_ENV_SECRET")
        self.assertEqual(val, "from-namespaced-env")
        del os.environ["KRIYA_SECRET_MYPROJ_MY_ENV_SECRET"]

        # Raw env key must NOT leak — prevents agents from extracting API keys etc.
        os.environ["MY_ENV_SECRET"] = "should-not-leak"
        val2 = get_secret("nonexistent-project", "MY_ENV_SECRET")
        self.assertIsNone(val2, "Raw env var fallback must not expose arbitrary env keys")
        del os.environ["MY_ENV_SECRET"]

    # ── Memory – embeddings ───────────────────────────────────────────────

    def test_embed_shape(self):
        emb = _embed("hello world")
        self.assertEqual(len(emb), 64)
        # Should be normalised
        import math
        norm = math.sqrt(sum(x*x for x in emb))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_embed_deterministic(self):
        e1 = _embed("the quick brown fox")
        e2 = _embed("the quick brown fox")
        self.assertEqual(e1, e2)

    def test_cosine_similarity(self):
        a = _embed("machine learning artificial intelligence")
        b = _embed("machine learning deep learning AI")
        c = _embed("recipe cooking pasta dinner")
        sim_related = _cosine(a, b)
        sim_unrelated = _cosine(a, c)
        self.assertGreater(sim_related, sim_unrelated)

    # ── Short-term memory ────────────────────────────────────────────────

    def test_short_term_capacity(self):
        stm = ShortTermMemory("test-agent-cap", capacity=5)
        for i in range(10):
            stm.add("user", f"message {i}")
        msgs = [m for m in stm.get_messages() if m["role"] != "system"]
        self.assertLessEqual(len(msgs), 5)
        # Most recent should be kept
        contents = [m["content"] for m in msgs]
        self.assertIn("message 9", contents)

    def test_short_term_system_preserved(self):
        stm = ShortTermMemory("test-agent-sys", capacity=3)
        stm.add("system", "You are a helpful assistant.")
        for i in range(10):
            stm.add("user", f"msg {i}")
        system_msgs = [m for m in stm.get_messages() if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 1)

    # ── Long-term memory ─────────────────────────────────────────────────

    def test_long_term_remember_recall(self):
        ltm = LongTermMemory("test-project-ltm")
        ltm.remember("Python is a popular programming language for AI and ML tasks")
        ltm.remember("The weather today is sunny and warm")
        ltm.remember("TensorFlow and PyTorch are deep learning frameworks")

        results = ltm.recall("What frameworks are used for deep learning?", top_k=2)
        self.assertGreater(len(results), 0)
        top = results[0]["content"]
        # Should match AI-related content
        self.assertTrue(
            "TensorFlow" in top or "Python" in top or "AI" in top,
            f"Unexpected top result: {top}"
        )

    def test_long_term_count(self):
        ltm = LongTermMemory("test-project-count")
        ltm.remember("First memory")
        ltm.remember("Second memory")
        self.assertEqual(ltm.count(), 2)

    # ── Event bus ────────────────────────────────────────────────────────

    def test_event_bus_publish_subscribe(self):
        async def _run():
            bus = EventBus()
            q = await bus.subscribe("test.topic")
            msg = Message("test.topic", {"hello": "world"})
            await bus.publish(msg)
            received = await asyncio.wait_for(q.get(), timeout=1.0)
            self.assertEqual(received.payload["hello"], "world")
            self.assertEqual(received.topic, "test.topic")

        asyncio.run(_run())

    def test_event_bus_wildcard(self):
        async def _run():
            bus = EventBus()
            bus._persist = False  # don't hit DB in this test
            q = await bus.subscribe("*")
            await bus.publish(Message("anything.here", {"x": 1}))
            received = await asyncio.wait_for(q.get(), timeout=1.0)
            self.assertEqual(received.payload["x"], 1)

        asyncio.run(_run())

    def test_event_bus_request_reply(self):
        async def _run():
            bus = EventBus()
            bus._persist = False
            # Set up a responder
            q = await bus.subscribe("ping")
            async def responder():
                req = await q.get()
                reply_topic = req.payload.get("_reply_to")
                if reply_topic:
                    await bus.publish(Message(reply_topic, {"pong": True}))
            asyncio.create_task(responder())
            result = await bus.request("ping", {}, from_id="tester", timeout=2.0)
            self.assertIsNotNone(result)
            self.assertTrue(result.payload["pong"])

        asyncio.run(_run())

    # ── Skills – builtin ─────────────────────────────────────────────────

    def test_skill_fs_write_read(self):
        from kriya.integrations.builtin_skills import skill_fs_write, skill_fs_read
        result = skill_fs_write({"path": "/tmp/kriya_test.txt", "content": "hello Kriya"}, {})
        self.assertTrue(result.get("written"))

        result2 = skill_fs_read({"path": "/tmp/kriya_test.txt"}, {})
        self.assertEqual(result2.get("content"), "hello Kriya")

    def test_skill_fs_write_blocked_path(self):
        from kriya.integrations.builtin_skills import skill_fs_write
        result = skill_fs_write({"path": "/etc/passwd", "content": "bad"}, {})
        self.assertIn("error", result)
        self.assertIn("not allowed", result["error"])

    def test_skill_system_shell_allowed(self):
        from kriya.integrations.builtin_skills import skill_system_shell
        result = skill_system_shell({"command": "echo hello"}, {})
        self.assertIn("hello", result.get("stdout", ""))

    def test_skill_system_shell_blocked(self):
        from kriya.integrations.builtin_skills import skill_system_shell
        result = skill_system_shell({"command": "rm -rf /"}, {})
        self.assertIn("error", result)
        self.assertIn("allowlist", result["error"])

    def test_skill_html_to_text(self):
        from kriya.integrations.builtin_skills import _html_to_text
        html = "<html><head><title>Test</title><script>alert(1)</script></head><body><h1>Hello</h1><p>World</p></body></html>"
        text = _html_to_text(html)
        self.assertIn("Hello", text)
        self.assertIn("World", text)
        self.assertNotIn("<", text)
        self.assertNotIn("alert", text)

    def test_skill_telegram_send(self):
        import sys
        sys.path.insert(0, "skills/telegram")
        import handler
        # Missing token
        result = handler.handle({"text": "hello"}, {})
        self.assertIn("error", result)
        # Missing text
        result = handler.handle({}, {"TELEGRAM_BOT_TOKEN": "fake", "TELEGRAM_CHAT_ID": "123"})
        self.assertIn("error", result)

    # ── Scheduler – schedule parsing ──────────────────────────────────────

    def test_schedule_every_seconds(self):
        from kriya.core.scheduler import next_run_time
        now = time.time()
        nxt = next_run_time("@every 30s", last_run=now)
        self.assertAlmostEqual(nxt - now, 30, delta=1)

    def test_schedule_every_minutes(self):
        from kriya.core.scheduler import next_run_time
        now = time.time()
        nxt = next_run_time("@every 5m", last_run=now)
        self.assertAlmostEqual(nxt - now, 300, delta=1)

    def test_schedule_daily(self):
        from kriya.core.scheduler import next_run_time
        now = time.time()
        nxt = next_run_time("@daily")
        self.assertAlmostEqual(nxt - now, 86400, delta=2)

    def test_schedule_once(self):
        from kriya.core.scheduler import next_run_time
        now = time.time()
        nxt = next_run_time("@once", last_run=None)
        self.assertAlmostEqual(nxt, now, delta=2)
        # Second run should be infinity
        nxt2 = next_run_time("@once", last_run=now)
        self.assertEqual(nxt2, float("inf"))

    # ── DAG resolution ────────────────────────────────────────────────────

    def test_dag_get_ready_tasks(self):
        from kriya.core.scheduler import get_ready_tasks
        pid = store.insert("projects", name="test-dag-proj",
                           status="idle", created_at=time.time(), updated_at=time.time())
        t1 = store.insert("tasks", project_id=pid, name="fetch",
                          depends_on="[]", status="done", created_at=time.time())
        t2 = store.insert("tasks", project_id=pid, name="process",
                          depends_on=json.dumps(["fetch"]), status="pending",
                          created_at=time.time())
        t3 = store.insert("tasks", project_id=pid, name="send",
                          depends_on=json.dumps(["process"]), status="pending",
                          created_at=time.time())

        ready = get_ready_tasks(pid)
        names = [t["name"] for t in ready]
        # fetch is done, process should be ready, send should not
        self.assertIn("process", names)
        self.assertNotIn("send", names)
        self.assertNotIn("fetch", names)

    # ── LLM layer – mocked ────────────────────────────────────────────────

    def test_llm_no_providers(self):
        from kriya.ai.llm import call_llm, LLMMessage, LLMError
        # Override config to have no providers
        with patch("kriya.ai.llm.get_config") as mock_cfg:
            mock_cfg.return_value.providers = []
            with self.assertRaises(LLMError):
                call_llm([LLMMessage("user", "hello")], fallback=False)

    def test_llm_action_extraction(self):
        from kriya.core.agent import _extract_action
        text = 'I will search for that.\n{"action": "skill_call", "skill": "web.scrape", "params": {"url": "https://example.com"}}\n'
        action = _extract_action(text)
        self.assertIsNotNone(action)
        self.assertEqual(action["skill"], "web.scrape")
        self.assertEqual(action["params"]["url"], "https://example.com")

    def test_llm_no_action(self):
        from kriya.core.agent import _extract_action
        text = "Here is my response without any tool calls."
        self.assertIsNone(_extract_action(text))

    # ── TOML loader ───────────────────────────────────────────────────────

    def test_toml_loader(self):
        if sys.version_info < (3, 11):
            self.skipTest("TOML requires Python 3.11+")
        import tempfile
        toml_content = '''
[project]
name = "test-toml-project"
description = "A test project loaded from TOML"
schedule = "@every 1h"

[tasks.step1]
name = "First step"
agents = ["worker"]

[tasks.step2]
name = "Second step"
depends_on = ["step1"]
agents = ["worker"]

[[agents]]
id = "worker"
role = "executor"
model = "auto"
provider = "auto"
prompt = "Do the work."
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            fpath = f.name

        from kriya.core.loader import import_project
        pid = import_project(fpath)
        p = store.fetch_one("projects", pid)
        self.assertEqual(p["name"], "test-toml-project")
        tasks = store.fetch_where("tasks", project_id=pid)
        self.assertEqual(len(tasks), 2)
        names = {t["name"] for t in tasks}
        self.assertIn("First step", names)
        self.assertIn("Second step", names)

        Path(fpath).unlink(missing_ok=True)

    # ── API server ───────────────────────────────────────────────────────

    def test_api_login_and_status(self):
        """Start API server, login, hit /api/status."""
        import threading, socket, time
        from http.server import HTTPServer
        from kriya.api.server import KriyaHandler, _start_time

        # Insert a test user with a known password (don't rely on generated admin creds)
        TEST_USER = "testuser-api"
        TEST_PASS = "TestPass123!"
        store.raw_query("DELETE FROM users WHERE username=?", (TEST_USER,))
        store.insert("users",
            username=TEST_USER,
            password_hash=hash_password(TEST_PASS),
            role="admin",
            created_at=time.time(),
        )

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        loop = asyncio.new_event_loop()
        server = HTTPServer(("127.0.0.1", port), KriyaHandler)

        import kriya.api.server as api_mod
        api_mod._loop = loop
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.2)   # let server start

        try:
            base = f"http://127.0.0.1:{port}"

            # Health check (no auth)
            req = __import__("urllib.request", fromlist=["urlopen"]).urlopen(
                f"{base}/api/health", timeout=5
            )
            data = json.loads(req.read())
            self.assertTrue(data["ok"])

            # Status without auth must be rejected
            import urllib.request as ur, urllib.error as ue
            with self.assertRaises(ue.HTTPError) as ctx:
                ur.urlopen(f"{base}/api/status", timeout=5)
            self.assertEqual(ctx.exception.code, 401)

            # Login
            r = ur.Request(f"{base}/api/auth/login",
                           data=json.dumps({"username": TEST_USER, "password": TEST_PASS}).encode(),
                           headers={"Content-Type": "application/json"})
            resp = ur.urlopen(r, timeout=5)
            body = json.loads(resp.read())
            self.assertIn("token", body)
            tok = body["token"]

            # Authenticated status — db path must not be exposed
            r2 = ur.Request(f"{base}/api/status",
                            headers={"Authorization": f"Bearer {tok}"})
            resp2 = ur.urlopen(r2, timeout=5)
            body2 = json.loads(resp2.read())
            self.assertEqual(body2["status"], "running")
            self.assertNotIn("db", body2)

        finally:
            server.shutdown()

    # ── End-to-end: project + task (mocked LLM) ───────────────────────────

    def test_e2e_project_run_mocked(self):
        """Run a full project with a mocked LLM call."""
        from kriya.core.scheduler import run_project

        # Create project
        pid = store.insert("projects", name="e2e-test",
                           status="idle", created_at=time.time(), updated_at=time.time())
        store.insert("tasks",
            project_id=pid,
            name="summarise",
            depends_on="[]",
            config=json.dumps({"agents": [{
                "id":       "sum",
                "role":     "executor",
                "model":    "auto",
                "provider": "auto",
                "prompt":   "Summarise the following: Pi Zero is a tiny computer.",
                "skills":   [],
                "max_tokens": 100,
                "temperature": 0.5,
                "max_retries": 1,
                "timeout": 10,
            }]}),
            status="pending",
            created_at=time.time(),
        )

        # Mock LLM call
        from kriya.ai import llm as llm_mod
        mock_resp = llm_mod.LLMResponse(
            content="The Pi Zero is a compact, low-cost single-board computer.",
            model="mock-model",
            provider="mock",
            input_tokens=20,
            output_tokens=15,
            latency_ms=100,
        )
        # Patch where agent.py imports call_llm from
        with patch("kriya.core.agent.call_llm", return_value=mock_resp):
            asyncio.run(run_project(pid))

        # Check results
        p = store.fetch_one("projects", pid)
        self.assertEqual(p["status"], "idle")
        tasks = store.fetch_where("tasks", project_id=pid)
        self.assertEqual(tasks[0]["status"], "done")
        output = json.loads(tasks[0]["output"])
        agent_key = list(output.keys())[0]
        self.assertTrue(output[agent_key]["success"])
        self.assertIn("Pi Zero", output[agent_key]["output"])


# ── Test runner ────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(KriyaTestCase)
    total  = suite.countTestCases()

    print(f"\n  {D}Kriya Test Suite{X}  ({total} tests)\n")
    print("  " + "─" * 60)

    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
    )
    result = runner.run(suite)

    print("\n  " + "─" * 60)
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n  {G}{passed}{X} passed  "
          f"{R}{len(result.failures)}{X} failed  "
          f"{Y}{len(result.errors)}{X} errors\n")

    if result.failures:
        print(f"  {R}Failures:{X}")
        for test, trace in result.failures:
            print(f"    • {test}: {trace.splitlines()[-1]}")
        print()

    if result.errors:
        print(f"  {Y}Errors:{X}")
        for test, trace in result.errors:
            print(f"    • {test}: {trace.splitlines()[-1]}")
        print()

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
