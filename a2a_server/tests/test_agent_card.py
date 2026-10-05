"""The agent card advertises what the node serves and has; the extended card and inventory are authenticated.

2026-10-05: both live cards advertised loopback URLs, omitted memory_prime, listed privileged skills as if
anyone could call them, and described one hard-coded backend for every device. The card is now built from the
skills the node actually serves (LOCI_A2A_SKILLS), the node's own profile, and a live read-only inventory.
"""

import asyncio
import importlib.util
import json
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_server_card", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)

from fastapi.testclient import TestClient  # noqa: E402

PRIVILEGED = {"memory_remember", "memory_sleep", "context_broadcast", "mnemosyne_triple_add"}
UNAVAILABLE = {"available": False, "reason": "not configured"}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class CardBase(unittest.TestCase):
    mock_probes = True          # the network probes are replaced by fakes unless a test class exercises them

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        a2a_server._inventory_cache.update(at=0.0, data=None)
        a2a_server._tasks.clear()
        self.bearer = {"Authorization": f"Bearer {a2a_server.A2A_TOKEN}"}
        patches = [
            patch.object(a2a_server, "TOTP_SEED", ""),
            patch.object(a2a_server, "SIGNATURE_REQUIRED", False),
            patch.object(a2a_server, "_PROFILE", {}),
            patch.object(a2a_server, "_ENABLED_SKILLS", list(a2a_server._SKILL_MAP)),
        ]
        if self.mock_probes:
            patches += [patch.object(a2a_server, "_probe_ollama", new=AsyncMock(return_value=dict(UNAVAILABLE))),
                        patch.object(a2a_server, "_probe_qdrant", new=AsyncMock(return_value=dict(UNAVAILABLE)))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(a2a_server.app, raise_server_exceptions=False)

    def card(self):
        r = self.client.get("/.well-known/agent.json")
        self.assertEqual(r.status_code, 200)
        return r.json()

    def write_profile(self, obj, name="profile.json"):
        path = self.dir / name
        path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return str(path)


class TestSkillsAdvertised(CardBase):
    def test_the_card_lists_exactly_the_skills_the_server_serves_including_memory_prime(self):
        ids = [s["id"] for s in self.card()["skills"]]
        self.assertEqual(ids, list(a2a_server._SKILL_MAP))
        self.assertIn("memory_prime", ids)
        self.assertIn("device_inventory", ids)

    def test_every_skill_has_a_name_and_a_description(self):
        for skill in self.card()["skills"]:
            with self.subTest(skill=skill["id"]):
                self.assertTrue(skill["name"] and skill["name"] != skill["id"])
                self.assertGreater(len(skill["description"]), 40)

    def test_privileged_skills_are_flagged_and_the_rest_are_not(self):
        flagged = {s["id"] for s in self.card()["skills"] if s["privileged"]}
        self.assertEqual(flagged, PRIVILEGED)
        self.assertEqual({s["id"] for s in self.card()["skills"] if not s["privileged"]},
                         set(a2a_server._SKILL_MAP) - PRIVILEGED)

    def test_card_version_matches_the_server_and_health(self):
        self.assertEqual(self.card()["version"], a2a_server.__version__)
        self.assertEqual(self.client.get("/health").json()["version"], a2a_server.__version__)


class TestSkillAllowlist(CardBase):
    def test_the_allowlist_selects_exactly_those_skills_in_dispatch_order(self):
        with patch.dict(os.environ, {"LOCI_A2A_SKILLS": "device_inventory, memory_stats ,memory_recall"}):
            self.assertEqual(a2a_server._enabled_skills(), ["memory_recall", "memory_stats", "device_inventory"])

    def test_unknown_names_are_reported_and_ignored(self):
        with patch.dict(os.environ, {"LOCI_A2A_SKILLS": "memory_stats,warp_drive"}), \
             self.assertLogs(a2a_server.log, level="ERROR") as logs:
            self.assertEqual(a2a_server._enabled_skills(), ["memory_stats"])
        self.assertIn("warp_drive", "\n".join(logs.output))

    def test_unset_means_every_skill(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCI_A2A_SKILLS", None)
            self.assertEqual(a2a_server._enabled_skills(), list(a2a_server._SKILL_MAP))

    def test_the_card_and_health_advertise_only_what_is_enabled(self):
        enabled = ["memory_recall", "memory_stats", "device_inventory"]
        with patch.object(a2a_server, "_ENABLED_SKILLS", enabled):
            self.assertEqual([s["id"] for s in self.card()["skills"]], enabled)
            self.assertEqual(self.client.get("/health").json()["skills"], enabled)

    def _send(self, skill):
        body = {"jsonrpc": "2.0", "id": "1", "method": "tasks/send", "params": {"skill_id": skill, "input": {}}}
        return self.client.post("/a2a", json=body, headers=self.bearer)

    def test_a_disabled_skill_is_refused_as_unknown_and_never_dispatched(self):
        with patch.object(a2a_server, "_ENABLED_SKILLS", ["memory_stats"]), \
             patch.object(a2a_server, "_dispatch", new=AsyncMock(return_value={"ok": True})) as dispatch:
            refused = self._send("gpu_inference")
            self.assertEqual(refused.status_code, 404)
            self.assertIn("Unknown skill", refused.text)
            self.assertEqual(dispatch.await_count, 0)
            allowed = self._send("memory_stats")                      # positive twin on the same fixture
            self.assertEqual(allowed.status_code, 200, allowed.text)
            self.assertEqual(dispatch.await_count, 1)


class TestAdvertisedUrl(CardBase):
    def test_loopback_url_on_a_non_loopback_listener_is_flagged(self):
        with patch.object(a2a_server, "AGENT_URL", "http://127.0.0.1:8201"), patch.object(a2a_server, "A2A_HOST", "0.0.0.0"):
            self.assertTrue(a2a_server._advertised_url_is_loopback())
            health = self.client.get("/health").json()
            self.assertEqual((health["advertised_url"], health["advertised_url_is_loopback"]), ("http://127.0.0.1:8201", True))

    def test_a_reachable_url_or_a_loopback_only_listener_is_not_flagged(self):
        cases = [("http://100.76.225.65:8201", "0.0.0.0"), ("http://127.0.0.1:8201", "127.0.0.1"),
                 ("http://localhost:8201", "localhost"), ("http://node.tailnet.example:8201", "0.0.0.0")]
        for url, host in cases:
            with self.subTest(url=url, host=host), patch.object(a2a_server, "AGENT_URL", url), patch.object(a2a_server, "A2A_HOST", host):
                self.assertFalse(a2a_server._advertised_url_is_loopback())

    def test_the_card_url_is_the_configured_one(self):
        with patch.object(a2a_server, "AGENT_URL", "http://100.115.69.88:8220"):
            self.assertEqual(self.card()["url"], "http://100.115.69.88:8220")


class TestProfileLoading(CardBase):
    GOOD = {
        "description": "Field laptop.", "summary": "i7 laptop, no GPU",
        "hardware": [{"name": "Intel i7-8550U", "detail": "4c/8t", "count": 1}, "bare string name"],
        "sensors": [{"name": "CubeCell", "kind": "radiation", "status": "intermittent", "secret_field": "dropped"}],
        "data": [{"name": "dama telemetry", "kind": "timeseries", "description": "InfluxDB", "where": "influxdb:8086"}],
        "notes": ["n1"], "unknown_key": {"a": 1},
    }

    def test_a_valid_profile_is_cleaned_to_known_fields(self):
        profile = a2a_server._load_profile(self.write_profile(self.GOOD))
        self.assertEqual(profile["description"], "Field laptop.")
        self.assertEqual(profile["summary"], "i7 laptop, no GPU")
        self.assertEqual(profile["hardware"], [{"name": "Intel i7-8550U", "detail": "4c/8t", "count": 1}, {"name": "bare string name"}])
        self.assertEqual(profile["sensors"], [{"name": "CubeCell", "kind": "radiation", "status": "intermittent"}])
        self.assertEqual(profile["data"], [{"name": "dama telemetry", "kind": "timeseries", "description": "InfluxDB", "where": "influxdb:8086"}])
        self.assertEqual(profile["notes"], ["n1"])
        self.assertNotIn("unknown_key", profile)

    def test_unusable_profiles_load_as_empty_and_say_why(self):
        big = self.write_profile("x" * (a2a_server._PROFILE_MAX_BYTES + 10), "big.json")
        cases = {"missing file": str(self.dir / "nope.json"), "not json": self.write_profile("{not json", "bad.json"),
                 "not an object": self.write_profile("[1, 2]", "list.json"), "too large": big}
        for label, path in cases.items():
            with self.subTest(label), self.assertLogs(a2a_server.log, level="ERROR"):
                self.assertEqual(a2a_server._load_profile(path), {})

    def test_no_path_means_no_profile_and_no_noise(self):
        self.assertEqual(a2a_server._load_profile(""), {})

    def test_strings_are_clipped_lists_capped_and_junk_rows_dropped(self):
        rows = [{"name": "n" * 500}] + [{"name": f"s{i}"} for i in range(80)] + [{"kind": "no name"}, 7, None, {"name": ""}]
        profile = a2a_server._load_profile(self.write_profile({"sensors": rows, "hardware": [{"name": "h", "detail": {"nested": 1}}]}))
        self.assertEqual(len(profile["sensors"]), 50)
        self.assertEqual(len(profile["sensors"][0]["name"]), 300)
        self.assertEqual(profile["hardware"], [{"name": "h"}])        # a nested value is not a scalar

    def test_the_public_card_carries_names_only_and_the_detail_stays_behind_auth(self):
        with patch.object(a2a_server, "_PROFILE", a2a_server._load_profile(self.write_profile(self.GOOD))):
            public = self.card()
            self.assertEqual(public["description"], "Field laptop.")
            self.assertEqual(public["resources"]["hardware"], ["Intel i7-8550U", "bare string name"])
            self.assertEqual(public["resources"]["sensors"], [{"name": "CubeCell", "kind": "radiation"}])
            self.assertEqual(public["resources"]["data"], [{"name": "dama telemetry", "kind": "timeseries"}])
            self.assertNotIn("influxdb:8086", json.dumps(public))     # the location is extended-card only
            self.assertNotIn("inventory", public)
            extended = self.client.get("/a2a/extended-card", headers=self.bearer).json()
            self.assertIn("influxdb:8086", json.dumps(extended["profile"]))

    def test_without_a_profile_the_card_has_no_resources_section(self):
        self.assertNotIn("resources", self.card())


class TestExtendedCard(CardBase):
    def test_it_needs_credentials(self):
        self.assertEqual(self.client.get("/a2a/extended-card").status_code, 401)
        self.assertEqual(self.client.get("/a2a/extended-card", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        ok = self.client.get("/a2a/extended-card", headers=self.bearer)
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_it_needs_totp_when_totp_is_on(self):
        with patch.object(a2a_server, "TOTP_SEED", "JBSWY3DPEHPK3PXP"):
            r = self.client.get("/a2a/extended-card", headers=self.bearer)
            self.assertEqual(r.status_code, 401)
            self.assertIn("TOTP", r.json()["detail"])

    def test_it_carries_the_inventory_and_the_extended_flag(self):
        card = self.client.get("/a2a/extended-card", headers=self.bearer).json()
        self.assertTrue(card["extended"])
        self.assertEqual(card["inventory"]["node"], a2a_server.AGENT_ID)
        self.assertEqual(card["inventory"]["ollama"], UNAVAILABLE)
        self.assertEqual([s["id"] for s in card["skills"]], list(a2a_server._SKILL_MAP))


class TestInventory(CardBase):
    def test_a_missing_tool_is_reported_unavailable_and_the_rest_still_reports(self):
        with patch.object(a2a_server, "_run", return_value=(None, "not installed")):
            inv = a2a_server._probe_local()
        self.assertEqual(inv["gpus"], {"available": False, "reason": "not installed"})
        self.assertEqual(inv["host"]["hostname"], a2a_server.platform.node())
        self.assertIn("storage", inv)
        self.assertIn("memory_store", inv)

    def test_run_reports_success_failure_timeout_and_a_missing_binary_without_raising(self):
        py = a2a_server.sys.executable
        self.assertEqual(a2a_server._run([py, "-c", "print('ok')"]), ("ok\n", ""))
        self.assertEqual(a2a_server._run([py, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]), (None, "boom"))
        self.assertEqual(a2a_server._run([py, "-c", "import time; time.sleep(30)"], timeout=1), (None, "timed out"))
        self.assertEqual(a2a_server._run(["definitely-not-a-real-binary-xyz"]), (None, "not installed"))

    def test_every_tool_being_absent_still_leaves_the_host_report(self):
        """Real _run, real probes: only the process launcher is faked, so a raised FileNotFoundError would show."""
        with patch.object(a2a_server.subprocess, "run", side_effect=FileNotFoundError("no such tool")):
            inv = a2a_server._probe_local()
        self.assertEqual(inv["gpus"], {"available": False, "reason": "not installed"})
        self.assertEqual(inv["host"]["hostname"], a2a_server.platform.node())
        self.assertGreaterEqual(len(inv["storage"]), 1)

    def test_gpu_output_is_reported_as_returned(self):
        smi = "NVIDIA GeForce RTX 2080 Ti, 11264 MiB, 512 MiB\nNVIDIA GeForce RTX 4070 Ti, 12282 MiB, 0 MiB\n"
        with patch.object(a2a_server, "_run", return_value=(smi, "")):
            gpus = a2a_server._probe_local()["gpus"]
        self.assertEqual(gpus, {"available": True, "devices": ["NVIDIA GeForce RTX 2080 Ti, 11264 MiB, 512 MiB",
                                                               "NVIDIA GeForce RTX 4070 Ti, 12282 MiB, 0 MiB"]})

    def test_memory_store_reports_size_and_row_count_without_a_path(self):
        db = self.dir / "mnemosyne.db"
        conn = sqlite3.connect(db)
        conn.execute("create table memories (id text)")
        conn.executemany("insert into memories values (?)", [("a",), ("b",), ("c",)])
        conn.commit()
        conn.close()
        with patch.object(a2a_server, "MNEMOSYNE_DB", str(db)):
            store = a2a_server._probe_local()["memory_store"]
        self.assertEqual((store["name"], store["present"], store["memories"]), ("mnemosyne.db", True, 3))
        self.assertNotIn(str(self.dir), json.dumps(store))

    def test_a_missing_store_is_reported_not_created(self):
        absent = self.dir / "sub" / "mnemosyne.db"
        with patch.object(a2a_server, "MNEMOSYNE_DB", str(absent)):
            store = a2a_server._probe_local()["memory_store"]
        self.assertEqual(store, {"name": "mnemosyne.db", "present": False})
        self.assertFalse(absent.exists())

    def test_the_inventory_is_cached_and_a_forced_refresh_probes_again(self):
        calls = []

        def spy():
            calls.append(1)
            return {"host": {}}
        with patch.object(a2a_server, "_probe_local", side_effect=spy):
            _run(a2a_server._inventory())
            _run(a2a_server._inventory())
            self.assertEqual(len(calls), 1)
            _run(a2a_server._inventory(force=True))
            self.assertEqual(len(calls), 2)
            with patch.object(a2a_server, "_INVENTORY_TTL_S", 0):
                _run(a2a_server._inventory())
            self.assertEqual(len(calls), 3)

    def test_the_skill_returns_the_inventory_and_the_skills_served(self):
        body = {"jsonrpc": "2.0", "id": "1", "method": "tasks/send", "params": {"skill_id": "device_inventory", "input": {}}}
        out = self.client.post("/a2a", json=body, headers=self.bearer).json()["result"]["output"]
        self.assertEqual(out["node"], a2a_server.AGENT_ID)
        self.assertEqual(out["skills_served"], list(a2a_server._SKILL_MAP))
        self.assertIn("gpus", out)


class _Resp:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        for suffix, (status, payload) in self.routes.items():
            if url.endswith(suffix):
                return _Resp(status, payload)
        return _Resp(404, {})


class TestNetworkProbes(CardBase):
    mock_probes = False

    def test_ollama_models_come_from_api_tags_with_the_v1_suffix_stripped(self):
        session = _FakeSession({"/api/tags": (200, {"models": [{"name": "zeta:1b"}, {"name": "alpha:7b"}]})})
        with patch.object(a2a_server, "OLLAMA_BASE", "http://gpu-host:11434/v1"), patch.object(a2a_server, "_get_http_session", return_value=session):
            out = _run(a2a_server._probe_ollama())
        self.assertEqual(out, {"available": True, "models": ["alpha:7b", "zeta:1b"], "model_count": 2})
        self.assertEqual(session.calls[0][0], "http://gpu-host:11434/api/tags")

    def test_ollama_failure_is_unavailable_with_a_reason(self):
        session = _FakeSession({"/api/tags": (503, {})})
        with patch.object(a2a_server, "OLLAMA_BASE", "http://gpu-host:11434/v1"), patch.object(a2a_server, "_get_http_session", return_value=session):
            self.assertEqual(_run(a2a_server._probe_ollama()), {"available": False, "reason": "HTTP 503"})

    def test_qdrant_lists_collections_with_counts_and_never_echoes_the_key(self):
        session = _FakeSession({
            "/collections": (200, {"result": {"collections": [{"name": "docs"}, {"name": "audio"}]}}),
            "/collections/docs": (200, {"result": {"points_count": 12}}),
            "/collections/audio": (200, {"result": {"points_count": 7}}),
        })
        with patch.object(a2a_server, "QDRANT_URL", "http://q:6333"), patch.object(a2a_server, "QDRANT_KEY", "very-secret-key"), \
             patch.object(a2a_server, "_get_http_session", return_value=session):
            out = _run(a2a_server._probe_qdrant())
        self.assertEqual(out, {"available": True, "collections": [{"name": "audio", "points": 7}, {"name": "docs", "points": 12}]})
        self.assertNotIn("very-secret-key", json.dumps(out))
        self.assertTrue(all(h == {"api-key": "very-secret-key"} for _, h in session.calls))   # it is sent, not returned

    def test_unconfigured_backends_are_reported_as_such(self):
        with patch.object(a2a_server, "QDRANT_URL", None), patch.object(a2a_server, "OLLAMA_BASE", ""):
            self.assertEqual(_run(a2a_server._probe_qdrant()), UNAVAILABLE)
            self.assertEqual(_run(a2a_server._probe_ollama()), UNAVAILABLE)


class TestNoSecretsAndFreshNodes(CardBase):
    def test_neither_card_carries_a_token_a_seed_or_a_key(self):
        secrets_ = {"A2A_TOKEN": "tok-should-not-leak", "TOTP_SEED": "SEEDSHOULDNOTLEAK", "BOOTSTRAP_KEY": "boot-should-not-leak",
                    "QDRANT_KEY": "qkey-should-not-leak"}
        patches = [patch.object(a2a_server, k, v) for k, v in secrets_.items()]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        public = json.dumps(self.client.get("/.well-known/agent.json").json())
        health = json.dumps(self.client.get("/health").json())
        for value in secrets_.values():
            self.assertNotIn(value, public)
            self.assertNotIn(value, health)

    def test_init_db_creates_the_table_and_never_disturbs_an_existing_store(self):
        db = self.dir / "node" / "mnemosyne.db"
        with patch.object(a2a_server, "MNEMOSYNE_DB", str(db)):
            a2a_server._init_memory_db()
            conn = sqlite3.connect(db)
            conn.execute("insert into memories (id, content) values ('keep', 'existing row')")
            conn.commit()
            conn.close()
            a2a_server._init_memory_db()                          # run again over data
        conn = sqlite3.connect(db)
        self.assertEqual(conn.execute("select id, content from memories").fetchall(), [("keep", "existing row")])
        conn.close()


class TestAudioCapture(CardBase):
    """Read from /proc/asound/pcm: `arecord -l` could not open the sound devices from a container."""

    PCM = ("00-00: USB Audio : USB Audio : playback 1 : capture 1\n"
           "01-00: HDA Analog : ALC285 Analog : playback 1\n"
           "02-03: ADMAIF4 : ADMAIF4 : capture 1\n")

    def _pcm(self, text):
        path = self.dir / "pcm"
        path.write_text(text, encoding="utf-8")
        return patch.object(a2a_server, "_ASOUND_PCM", str(path))

    def test_only_capture_capable_devices_are_listed_with_their_names(self):
        with self._pcm(self.PCM):
            self.assertEqual(a2a_server._probe_audio_capture(), ["00-00: USB Audio", "02-03: ADMAIF4"])

    def test_a_host_with_only_playback_devices_has_an_empty_list_not_an_error(self):
        with self._pcm("01-00: HDA Analog : ALC285 Analog : playback 1\n"):
            self.assertEqual(a2a_server._probe_audio_capture(), [])

    def test_the_setting_points_the_probe_at_a_mounted_copy_of_the_hosts_file(self):
        """Docker masks /proc/asound in containers, so the node is given the host's file under another path."""
        mounted = self.dir / "host-asound-pcm"
        mounted.write_text(self.PCM, encoding="utf-8")
        with patch.object(a2a_server, "_ASOUND_PCM", str(self.dir / "masked")), \
             patch.dict(os.environ, {"LOCI_A2A_ASOUND_PCM": str(mounted)}):
            self.assertEqual(a2a_server._probe_audio_capture(), ["00-00: USB Audio", "02-03: ADMAIF4"])
        with patch.object(a2a_server, "_ASOUND_PCM", str(self.dir / "masked")):     # same fixture, setting absent
            os.environ.pop("LOCI_A2A_ASOUND_PCM", None)
            self.assertEqual(a2a_server._probe_audio_capture()["available"], False)

    def test_a_missing_procfs_file_is_reported_unavailable_with_a_reason(self):
        with patch.object(a2a_server, "_ASOUND_PCM", str(self.dir / "absent")):
            self.assertEqual(a2a_server._probe_audio_capture(),
                             {"available": False, "reason": "no ALSA devices (/proc/asound/pcm is missing)"})


class TestSharedHttpSession(CardBase):
    """2026-10-05: mrpink's Qdrant 1.17 answers in Brotli; aiohttp advertised it and could not decode it."""

    def test_the_session_asks_for_gzip_and_deflate_only(self):
        async def echo_accept_encoding():
            from aiohttp import web

            async def handler(request):
                return web.Response(text=request.headers.get("Accept-Encoding", "<none>"))
            app = web.Application()
            app.router.add_get("/", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            session = a2a_server._get_http_session()
            try:
                async with session.get(f"http://127.0.0.1:{port}/") as resp:
                    return await resp.text(), session.headers.get("Accept-Encoding")
            finally:
                await session.close()
                await runner.cleanup()
        with patch.object(a2a_server, "_http_session", None):
            on_the_wire, configured = _run(echo_accept_encoding())
        self.assertEqual(on_the_wire, "gzip, deflate")
        # Set explicitly, not left to aiohttp's default: that default adds Brotli on any host that has the
        # brotli package, so the wire check alone would pass here and fail on such a machine.
        self.assertEqual(configured, "gzip, deflate")


class TestToolLookup(CardBase):
    def test_a_tool_on_the_path_is_used_as_found(self):
        with patch.object(a2a_server.shutil, "which", return_value="/somewhere/nvidia-smi"):
            self.assertEqual(a2a_server._tool_path("nvidia-smi"), "/somewhere/nvidia-smi")

    def test_a_tool_the_service_path_misses_is_found_in_the_fallback_directories(self):
        tool = self.dir / "fake-gpu-tool"
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        tool.chmod(0o755)
        with patch.object(a2a_server.shutil, "which", return_value=None), \
             patch.object(a2a_server, "_TOOL_FALLBACK_DIRS", (str(self.dir / "empty"), str(self.dir))):
            self.assertEqual(a2a_server._tool_path("fake-gpu-tool"), str(tool))

    def test_a_tool_found_nowhere_comes_back_as_its_bare_name_and_reports_not_installed(self):
        with patch.object(a2a_server.shutil, "which", return_value=None), \
             patch.object(a2a_server, "_TOOL_FALLBACK_DIRS", (str(self.dir),)):
            self.assertEqual(a2a_server._tool_path("no-such-tool-xyz"), "no-such-tool-xyz")
            self.assertEqual(a2a_server._run([a2a_server._tool_path("no-such-tool-xyz")]), (None, "not installed"))

    def test_the_inventory_runs_the_tool_it_found_not_the_bare_name(self):
        ran = []

        def fake_run(cmd, timeout=5):
            ran.append(cmd[0])
            return None, "stubbed"
        with patch.object(a2a_server, "_tool_path", side_effect=lambda n: f"/opt/wsl/lib/{n}"), \
             patch.object(a2a_server, "_run", side_effect=fake_run):
            a2a_server._probe_local()
        self.assertIn("/opt/wsl/lib/nvidia-smi", ran)
        self.assertNotIn("nvidia-smi", ran)


class TestBoardFile(CardBase):
    def test_the_board_model_comes_from_the_configured_file(self):
        model = self.dir / "board-model"
        model.write_bytes(b"NVIDIA Jetson Orin Nano Developer Kit\x00")
        with patch.dict(os.environ, {"LOCI_A2A_BOARD_FILE": str(model)}):
            self.assertEqual(a2a_server._probe_local()["host"]["board"], "NVIDIA Jetson Orin Nano Developer Kit")

    def test_a_missing_board_file_leaves_the_board_out_rather_than_guessing(self):
        with patch.dict(os.environ, {"LOCI_A2A_BOARD_FILE": str(self.dir / "absent")}):
            self.assertNotIn("board", a2a_server._probe_local()["host"])


if __name__ == "__main__":
    unittest.main()
