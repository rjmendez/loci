"""Ed25519 per-agent signature auth (v1 body-only, v2 bound to timestamp/nonce/method/path/body).

Real keys and real signatures throughout: a verifier that ignores the signature, checks the wrong
payload, or treats a header as proof is caught. The v1 vectors reproduce what the agent-mesh
clients send (agent-mesh/a2a/auth.py build_headers).
"""

import asyncio
import base64
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_server_sig", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

BEARER = {"Authorization": f"Bearer {a2a_server.A2A_TOKEN}"}   # whichever token this process loaded
TOTP_SEED = "JBSWY3DPEHPK3PXP"


def _pem_pub(priv) -> str:
    return priv.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def _b64(sig: bytes) -> str:
    return base64.urlsafe_b64encode(sig).decode()


def _body(method="tasks/list", params=None) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": "t-1", "method": method,
                       "params": params if params is not None else {}}, separators=(",", ":")).encode()


class SigBase(unittest.TestCase):
    def setUp(self):
        self.key_a = ed25519.Ed25519PrivateKey.generate()
        self.key_b = ed25519.Ed25519PrivateKey.generate()
        self.pubs = {"agent-a": self.key_a.public_key(), "agent-b": self.key_b.public_key()}
        a2a_server._sig_nonces.clear()
        a2a_server._totp_attempts.clear()
        a2a_server._tasks.clear()
        self.patches = [
            patch.object(a2a_server, "_PEER_PUBKEYS", self.pubs),
            patch.object(a2a_server, "SIGNATURE_REQUIRED", False),
            patch.object(a2a_server, "SIGNATURE_MIN_VERSION", 1),
            patch.object(a2a_server, "TOTP_SEED", ""),
            patch.object(a2a_server, "_PRIVILEGED_SENDERS", frozenset()),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(a2a_server.app, raise_server_exceptions=False)

    def tearDown(self):
        for p in self.patches:
            p.stop()

    # -- request builders -----------------------------------------------------------
    def v1(self, body=None, agent="agent-a", key=None, ts="now", nonce="fresh", sig=None):
        body = body if body is not None else _body()
        key = key or self.key_a
        h = {"Content-Type": "application/json", "X-Agent-ID": agent,
             "X-Signature": sig if sig is not None else _b64(key.sign(body))}
        if ts:
            h["X-Timestamp"] = str(int(time.time())) if ts == "now" else ts
        if nonce:
            h["X-Request-ID"] = uuid.uuid4().hex if nonce == "fresh" else nonce
        return body, h

    def v2(self, body=None, agent="agent-a", key=None, ts="now", nonce="fresh", path="/a2a", method="POST",
           sign_agent=None):
        body = body if body is not None else _body()
        key = key or self.key_a
        ts = str(int(time.time())) if ts == "now" else ts
        nonce = uuid.uuid4().hex if nonce == "fresh" else nonce
        payload = a2a_server._sig_payload_v2(sign_agent or agent, ts, nonce, method, path, body)
        return body, {"Content-Type": "application/json", "X-Agent-ID": agent, "X-Timestamp": ts,
                      "X-Request-ID": nonce, "X-Signature-Version": "2", "X-Signature": _b64(key.sign(payload))}

    def post(self, built, extra=None):
        body, headers = built
        return self.client.post("/a2a", content=body, headers={**headers, **(extra or {})})


class TestV1MeshCompatibility(SigBase):
    def test_a_mesh_style_signed_request_is_accepted(self):
        r = self.post(self.v1())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], {"tasks": []})

    def test_mesh_headers_without_timestamp_or_nonce_are_accepted(self):
        self.assertEqual(self.post(self.v1(ts=None, nonce=None)).status_code, 200)

    def test_signature_with_padding_stripped_or_standard_base64_is_accepted(self):
        body = _body()
        raw = self.key_a.sign(body)
        for text in (base64.urlsafe_b64encode(raw).decode().rstrip("="), base64.b64encode(raw).decode()):
            with self.subTest(text=text[:6]):
                self.assertEqual(self.post(self.v1(body=body, sig=text)).status_code, 200)

    def test_tampered_body_is_refused(self):
        body, h = self.v1()
        r = self.client.post("/a2a", content=body.replace(b"tasks/list", b"tasks/send"), headers=h)
        self.assertEqual(r.status_code, 401)

    def test_signature_from_another_agents_key_is_refused(self):
        self.assertEqual(self.post(self.v1(agent="agent-a", key=self.key_b)).status_code, 401)

    def test_unknown_agent_is_refused_and_looks_like_a_bad_signature(self):
        unknown = self.post(self.v1(agent="nobody", key=self.key_a))
        bad = self.post(self.v1(sig="AAAA"))
        self.assertEqual((unknown.status_code, bad.status_code), (401, 401))
        self.assertEqual(unknown.json()["detail"], bad.json()["detail"])

    def test_stale_and_future_timestamps_are_refused_when_present(self):
        skew = a2a_server.SIGNATURE_MAX_SKEW_S
        for label, ts in (("stale", str(int(time.time()) - skew - 30)), ("future", str(int(time.time()) + skew + 30)),
                          ("nan", "nan"), ("inf", "inf"), ("text", "yesterday")):
            with self.subTest(label):
                self.assertEqual(self.post(self.v1(ts=ts)).status_code, 401)

    def test_byte_identical_replay_is_refused(self):
        built = self.v1()
        self.assertEqual(self.post(built).status_code, 200)
        r = self.post(built)
        self.assertEqual(r.status_code, 401)
        self.assertIn("already used", r.json()["detail"])

    def test_v1_does_not_stop_a_replay_with_fresh_headers_which_is_why_v2_exists(self):
        """Documented limit of v1: the nonce and timestamp are not signed."""
        body, h = self.v1()
        self.assertEqual(self.client.post("/a2a", content=body, headers=h).status_code, 200)
        h2 = {**h, "X-Request-ID": uuid.uuid4().hex, "X-Timestamp": str(int(time.time()))}
        self.assertEqual(self.client.post("/a2a", content=body, headers=h2).status_code, 200)

    def test_version_header_must_be_known(self):
        body, h = self.v1()
        r = self.client.post("/a2a", content=body, headers={**h, "X-Signature-Version": "3"})
        self.assertEqual(r.status_code, 401)


class TestV2Binding(SigBase):
    def test_valid_v2_is_accepted(self):
        r = self.post(self.v2())
        self.assertEqual(r.status_code, 200, r.text)

    def test_replay_is_refused_even_with_the_same_headers(self):
        built = self.v2()
        self.assertEqual(self.post(built).status_code, 200)
        self.assertEqual(self.post(built).status_code, 401)

    def test_fresh_headers_cannot_be_swapped_in_for_a_captured_body(self):
        body, h = self.v2()
        for name, value in (("X-Request-ID", uuid.uuid4().hex), ("X-Timestamp", str(int(time.time()) + 1))):
            with self.subTest(name):
                r = self.client.post("/a2a", content=body, headers={**h, name: value})
                self.assertEqual(r.status_code, 401)

    def test_body_path_method_and_agent_are_all_bound(self):
        cases = {
            "other path": dict(path="/other"),
            "other method": dict(method="PUT"),
            "signed as another agent": dict(sign_agent="agent-b"),
        }
        for label, kwargs in cases.items():
            with self.subTest(label):
                self.assertEqual(self.post(self.v2(**kwargs)).status_code, 401)
        body, h = self.v2()
        self.assertEqual(self.client.post("/a2a", content=body + b" ", headers=h).status_code, 401)

    def test_timestamp_and_request_id_are_mandatory(self):
        self.assertEqual(self.post(self.v2(ts="")).status_code, 401)
        self.assertEqual(self.post(self.v2(nonce="")).status_code, 401)

    def test_stale_future_nan_timestamps_are_refused(self):
        skew = a2a_server.SIGNATURE_MAX_SKEW_S
        for ts in (str(int(time.time()) - skew - 30), str(int(time.time()) + skew + 30), "nan", "inf"):
            with self.subTest(ts=ts[:6]):
                self.assertEqual(self.post(self.v2(ts=ts)).status_code, 401)

    def test_min_version_2_refuses_v1_and_accepts_v2(self):
        with patch.object(a2a_server, "SIGNATURE_MIN_VERSION", 2):
            r = self.post(self.v1())
            self.assertEqual(r.status_code, 401)
            self.assertIn("minimum 2", r.json()["detail"])
            self.assertEqual(self.post(self.v2()).status_code, 200)

    def test_oversized_request_id_is_refused(self):
        self.assertEqual(self.post(self.v2(nonce="n" * 200)).status_code, 401)

    def test_nonce_cache_is_capped_per_agent_and_other_agents_are_unaffected(self):
        with patch.object(a2a_server, "_SIG_NONCE_PER_AGENT_CAP", 3):
            codes = [self.post(self.v2()).status_code for _ in range(4)]
            self.assertEqual(codes, [200, 200, 200, 429])
            self.assertEqual(self.post(self.v2(agent="agent-b", key=self.key_b)).status_code, 200)

    def test_expired_nonces_are_reclaimed(self):
        with patch.object(a2a_server, "_SIG_NONCE_PER_AGENT_CAP", 2):
            self.assertEqual(self.post(self.v2()).status_code, 200)
            self.assertEqual(self.post(self.v2()).status_code, 200)
            for nonce in a2a_server._sig_nonces["agent-a"]:
                a2a_server._sig_nonces["agent-a"][nonce] = time.monotonic() - 1
            self.assertEqual(self.post(self.v2()).status_code, 200)

    def test_unauthenticated_callers_cannot_fill_the_nonce_cache(self):
        for _ in range(5):
            self.post(self.v2(key=self.key_b))                       # wrong key for agent-a
        self.assertEqual(a2a_server._sig_nonces.get("agent-a", {}), {})


class TestNoDowngrade(SigBase):
    def test_bad_signature_never_falls_back_to_a_valid_bearer(self):
        r = self.post(self.v1(sig="AAAA"), extra=BEARER)
        self.assertEqual(r.status_code, 401)

    def test_unknown_agent_id_never_falls_back_to_a_valid_bearer(self):
        self.assertEqual(self.post(self.v1(agent="nobody"), extra=BEARER).status_code, 401)

    def test_bearer_without_signature_headers_still_works_when_not_required(self):
        r = self.client.post("/a2a", content=_body(), headers={**BEARER, "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 200)

    def test_agent_id_header_alone_is_not_a_credential(self):
        r = self.client.post("/a2a", content=_body(), headers={"Content-Type": "application/json", "X-Agent-ID": "agent-a"})
        self.assertEqual(r.status_code, 401)

    def test_required_mode_refuses_bearer_only_and_accepts_signed(self):
        with patch.object(a2a_server, "SIGNATURE_REQUIRED", True):
            r = self.client.post("/a2a", content=_body(), headers={**BEARER, "Content-Type": "application/json"})
            self.assertEqual(r.status_code, 401)
            self.assertIn("X-Signature", r.json()["detail"])
            self.assertEqual(self.post(self.v2()).status_code, 200)

    def test_without_the_cryptography_package_nothing_verifies(self):
        with patch.object(a2a_server, "_CRYPTO_OK", False):
            self.assertEqual(self.post(self.v1()).status_code, 401)


class TestTotpInteraction(SigBase):
    def setUp(self):
        super().setUp()
        self._totp = patch.object(a2a_server, "TOTP_SEED", TOTP_SEED)
        self._totp.start()

    def tearDown(self):
        self._totp.stop()
        super().tearDown()

    def test_a_verified_signature_needs_no_totp_code(self):
        self.assertEqual(self.post(self.v2()).status_code, 200)
        self.assertEqual(self.post(self.v1()).status_code, 200)

    def test_bearer_still_needs_totp(self):
        r = self.client.post("/a2a", content=_body(), headers={**BEARER, "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 401)
        self.assertIn("TOTP", r.json()["detail"])

    def test_a_forged_signature_header_does_not_skip_totp(self):
        """The bypass is earned by a verified signature, not by sending the header."""
        for built in (self.v1(sig="AAAA"), self.v1(agent="nobody"), self.v1(key=self.key_b)):
            r = self.post(built, extra=BEARER)
            self.assertEqual(r.status_code, 401)

    def test_a_stray_signature_or_agent_header_on_a_bearer_request_does_not_skip_totp(self):
        """A bearer request that merely carries one of the signature headers still takes the bearer path."""
        base = {**BEARER, "Content-Type": "application/json"}
        for label, extra in (("signature header only", {"X-Signature": "AAAA"}),
                             ("registered agent id only", {"X-Agent-ID": "agent-a"})):
            with self.subTest(label):
                r = self.client.post("/a2a", content=_body(), headers={**base, **extra})
                self.assertEqual(r.status_code, 401)
                self.assertIn("TOTP", r.json()["detail"])
        from pyotp import TOTP
        ok = self.client.post("/a2a", content=_body(), headers={**base, "X-Signature": "AAAA",
                                                                "X-TOTP": TOTP(TOTP_SEED).now()})
        self.assertEqual(ok.status_code, 200, ok.text)             # positive twin: with a real code it passes

    def test_signed_requests_do_not_spend_the_totp_attempt_budget(self):
        for _ in range(a2a_server._TOTP_MAX_ATTEMPTS * 2):
            self.assertEqual(self.post(self.v2()).status_code, 200)
        self.assertNotIn("testclient", a2a_server._totp_attempts)


class TestIdentityBinding(SigBase):
    def _send(self, built, skill="memory_remember", declared=None):
        body = json.loads(built[0])
        body["method"] = "tasks/send"
        body["params"] = {"skill_id": skill, "input": {"content": "x"}}
        if declared:
            body["params"]["sender"] = declared
        raw = json.dumps(body, separators=(",", ":")).encode()
        agent = built[1]["X-Agent-ID"]
        key = self.key_a if agent == "agent-a" else self.key_b
        return self.post(self.v2(body=raw, agent=agent, key=key))

    def test_a_signed_agent_cannot_declare_another_sender(self):
        with patch.object(a2a_server, "_dispatch", new=AsyncMock(return_value={"ok": True})) as dispatch:
            r = self._send(self.v2(), skill="memory_stats", declared="agent-b")
        self.assertEqual(r.status_code, 403)
        dispatch.assert_not_called()

    def test_a_signed_agent_that_is_not_privileged_cannot_call_a_destructive_skill(self):
        with patch.object(a2a_server, "_dispatch", new=AsyncMock(return_value={"ok": True})) as dispatch:
            r = self._send(self.v2())
        self.assertEqual(r.status_code, 403)
        self.assertIn("elevated privilege", r.text)
        dispatch.assert_not_called()

    def test_a_privileged_signed_agent_can(self):
        with patch.object(a2a_server, "_PRIVILEGED_SENDERS", frozenset({"agent-a"})), \
             patch.object(a2a_server, "_dispatch", new=AsyncMock(return_value={"ok": True})) as dispatch:
            r = self._send(self.v2())
            denied = self._send(self.v2(agent="agent-b", key=self.key_b))
        self.assertEqual(r.status_code, 200, r.text)
        dispatch.assert_called_once()
        self.assertEqual(denied.status_code, 403)

    def test_non_destructive_skill_is_open_to_any_signed_agent(self):
        with patch.object(a2a_server, "_dispatch", new=AsyncMock(return_value={"ok": True})):
            self.assertEqual(self._send(self.v2(), skill="memory_stats").status_code, 200)

    def test_the_signed_identity_is_bound_for_acl_checks(self):
        seen = []

        async def spy(skill_id, task):
            seen.append(a2a_server._caller_identity.bound_agent_id() if a2a_server._caller_identity else "no-helper")
            return {"ok": True}
        if a2a_server._caller_identity is None:
            self.skipTest("caller_identity helper not importable in this layout")
        with patch.object(a2a_server, "_dispatch", new=spy):
            self.assertEqual(self._send(self.v2(), skill="memory_stats").status_code, 200)
        self.assertEqual(seen, ["agent-a"])


class TestLoadingKeys(unittest.TestCase):
    def setUp(self):
        self.k1 = ed25519.Ed25519PrivateKey.generate()
        self.k2 = ed25519.Ed25519PrivateKey.generate()

    def test_json_and_directory_sources_merge_and_the_directory_wins(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "agent-b.pub").write_text(_pem_pub(self.k2))
            pathlib.Path(d, "notes.txt").write_text("ignored")
            keys = a2a_server._load_peer_pubkeys({
                "PEER_PUBKEYS_JSON": json.dumps({"agent-a": _pem_pub(self.k1), "agent-b": _pem_pub(self.k1)}),
                "PEER_PUBKEYS_DIR": d})
        self.assertEqual(sorted(keys), ["agent-a", "agent-b"])
        probe = b"probe"
        keys["agent-b"].verify(self.k2.sign(probe), probe)          # the directory's key, not the JSON one

    def test_bad_entries_are_skipped_without_losing_the_good_ones(self):
        rsa_pem = rsa.generate_private_key(65537, 2048).public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        with self.assertLogs(a2a_server.log, level="ERROR") as logs:
            keys = a2a_server._load_peer_pubkeys({"PEER_PUBKEYS_JSON": json.dumps({
                "good": _pem_pub(self.k1), "garbage": "not a key", "rsa": rsa_pem,
                "../evil": _pem_pub(self.k2), "": _pem_pub(self.k2)})})
        self.assertEqual(sorted(keys), ["good"])
        joined = "\n".join(logs.output)
        self.assertNotIn("not a key", joined)                       # key material is never logged
        self.assertNotIn("BEGIN PUBLIC KEY", joined)

    def test_bad_json_or_wrong_shape_is_ignored_loudly(self):
        for raw in ("{not json", "[1,2]", '"str"'):
            with self.subTest(raw=raw), self.assertLogs(a2a_server.log, level="ERROR"):
                self.assertEqual(a2a_server._load_peer_pubkeys({"PEER_PUBKEYS_JSON": raw}), {})

    def test_nothing_configured_means_no_keys_and_no_noise(self):
        self.assertEqual(a2a_server._load_peer_pubkeys({}), {})

    def test_required_mode_without_the_package_refuses_to_start(self):
        code = ("import sys, runpy; sys.modules['cryptography'] = None\n"
                f"runpy.run_path({str(_server_path)!r}, run_name='a2a_boot')")
        env = {**os.environ, "A2A_REQUIRE_SIGNATURE": "1", "LOCI_A2A_TOKEN": "t", "LOCI_ENV_FILE": os.devnull}
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1, r.stderr[-400:])
        self.assertIn("refusing to start", r.stderr)


class TestHealthAndCard(SigBase):
    def test_health_reports_registered_agents_and_never_key_material(self):
        h = self.client.get("/health")
        body = h.json()
        self.assertTrue(body["signature_auth"])
        self.assertEqual(body["registered_peers"], ["agent-a", "agent-b"])
        self.assertFalse(body["signature_required"])
        self.assertEqual(body["signature_min_version"], 1)
        self.assertNotIn("PUBLIC KEY", h.text)

    def test_health_reports_off_without_keys(self):
        with patch.object(a2a_server, "_PEER_PUBKEYS", {}):
            body = self.client.get("/health").json()
        self.assertFalse(body["signature_auth"])
        self.assertEqual(body["registered_peers"], [])


class TestOutboundSigning(SigBase):
    PEER = "http://peer.example:8201/a2a"

    def setUp(self):
        super().setUp()
        self.me = ed25519.Ed25519PrivateKey.generate()
        self.pubs["node-x"] = self.me.public_key()
        for p in (patch.object(a2a_server, "_SIGNING_KEY", self.me), patch.object(a2a_server, "AGENT_ID", "node-x"),
                  patch.dict(os.environ, {"PEER_A2A_SIGNED_URLS": "http://peer.example:8201"})):
            p.start()
            self.patches.append(p)

    def _roundtrip(self, version):
        payload = {"jsonrpc": "2.0", "id": "1", "method": "tasks/list", "params": {}}
        with patch.object(a2a_server, "SIGNING_VERSION", version):
            kwargs = a2a_server._peer_post_kwargs(self.PEER, {"Content-Type": "application/json"}, payload)
        self.assertIn("data", kwargs)
        self.assertNotIn("json", kwargs)
        return self.client.post("/a2a", content=kwargs["data"], headers=kwargs["headers"]), kwargs

    def test_v2_outbound_is_accepted_by_this_servers_verifier(self):
        r, kwargs = self._roundtrip(2)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(kwargs["headers"]["X-Signature-Version"], "2")

    def test_v1_outbound_is_accepted_too(self):
        r, kwargs = self._roundtrip(1)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("X-Signature-Version", kwargs["headers"])

    def test_every_outbound_call_has_its_own_request_id(self):
        _, a = self._roundtrip(2)
        _, b = self._roundtrip(2)
        self.assertNotEqual(a["headers"]["X-Request-ID"], b["headers"]["X-Request-ID"])

    def test_the_signature_is_bound_to_the_peer_path(self):
        payload = {"jsonrpc": "2.0", "id": "1", "method": "tasks/list", "params": {}}
        elsewhere = "http://peer.example:8201/elsewhere"
        with patch.dict(os.environ, {"PEER_A2A_SIGNED_URLS": elsewhere}):
            kwargs = a2a_server._peer_post_kwargs(elsewhere, {}, payload)
        self.assertIn("data", kwargs)                                  # it really was signed, for /elsewhere
        self.assertEqual(self.client.post("/a2a", content=kwargs["data"], headers=kwargs["headers"]).status_code, 401)
        with patch.dict(os.environ, {"PEER_A2A_SIGNED_URLS": self.PEER}):
            same = a2a_server._peer_post_kwargs(self.PEER, {}, payload)   # positive twin: the real path verifies
        self.assertEqual(self.client.post("/a2a", content=same["data"], headers=same["headers"]).status_code, 200)

    def test_unsigned_peers_keep_the_json_body_and_headers_untouched(self):
        headers = {"Authorization": "Bearer t"}
        payload = {"a": 1}
        kwargs = a2a_server._peer_post_kwargs("http://other.example:8201/a2a", headers, payload)
        self.assertEqual(kwargs, {"json": payload, "headers": headers})

    def test_a_signed_peer_gets_signature_headers_and_no_bearer_or_totp(self):
        headers, reason = a2a_server._peer_headers(self.PEER, {self.PEER: "tok"}, "tok", {}, "JBSWY3DPEHPK3PXP")
        self.assertIsNone(reason)
        self.assertEqual(headers, {"Content-Type": "application/json"})

    def test_a_signed_peer_without_a_signing_key_is_skipped_not_sent_unsigned(self):
        with patch.object(a2a_server, "_SIGNING_KEY", None):
            headers, reason = a2a_server._peer_headers(self.PEER, {}, "tok", {}, "")
        self.assertIsNone(headers)
        self.assertEqual(reason, "signing key not configured")


class _FakeResp:
    status = 200

    async def json(self):
        return {"result": {"output": {"ok": 1}}}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingSession:
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def post(self, url, **kw):
        _RecordingSession.calls.append((url, kw))
        return _FakeResp()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _run(coro):
    """Run a coroutine on a private loop; asyncio.run would also unset the current loop other tests use."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _load_client_module():
    spec = importlib.util.spec_from_file_location("loci_client_sig", _server_path.parent / "client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestClientSigning(SigBase):
    def setUp(self):
        super().setUp()
        import tempfile
        self.client_mod = _load_client_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keyfile = pathlib.Path(self.tmp.name, "agent-a.pem")
        self.keyfile.write_bytes(self.key_a.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        _RecordingSession.calls = []

    def _client(self, **kw):
        return self.client_mod.LociClient(endpoint="http://srv.example:8201", token="tok", sender="agent-a",
                                          signing_key_file=str(self.keyfile), **kw)

    def test_a_request_signed_by_the_client_is_accepted_by_the_server(self):
        for version in (2, 1):
            with self.subTest(version=version):
                c = self._client(signature_version=version)
                body, headers = c._signed_request("http://srv.example:8201/a2a",
                                                  json.loads(_body("tasks/list")))
                r = self.client.post("/a2a", content=body, headers=headers)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["result"], {"tasks": []})
                self.assertEqual(headers["X-Agent-ID"], "agent-a")
                self.assertEqual("X-Signature-Version" in headers, version == 2)

    def test_the_client_default_is_v2(self):
        self.assertEqual(self._client().signature_version, 2)

    def test_a_signed_call_sends_the_signed_bytes_and_no_bearer(self):
        c = self._client()
        with patch.object(self.client_mod.aiohttp, "ClientSession", _RecordingSession):
            out = _run(c._call("memory_stats"))
        self.assertEqual(out, {"ok": 1})
        ((url, kw),) = _RecordingSession.calls
        self.assertEqual(url, "http://srv.example:8201/a2a")
        self.assertEqual(sorted(kw), ["data", "headers"])
        self.assertIsInstance(kw["data"], bytes)
        self.assertNotIn("Authorization", kw["headers"])
        self.assertNotIn("X-TOTP", kw["headers"])
        self.assertEqual(kw["headers"]["X-Agent-ID"], "agent-a")
        self.assertEqual(self.client.post("/a2a", content=kw["data"], headers=kw["headers"]).status_code, 200)

    def test_an_unsigned_call_keeps_json_and_the_bearer_header(self):
        c = self.client_mod.LociClient(endpoint="http://srv.example:8201", token="tok", sender="agent-a")
        with patch.object(self.client_mod.aiohttp, "ClientSession", _RecordingSession):
            _run(c._call("memory_stats"))
        ((_, kw),) = _RecordingSession.calls
        self.assertEqual(sorted(kw), ["headers", "json"])
        self.assertEqual(kw["headers"]["Authorization"], "Bearer tok")
        self.assertNotIn("X-Signature", kw["headers"])

    def test_a_signing_key_that_cannot_be_used_is_an_error_not_a_silent_bearer_fallback(self):
        missing = pathlib.Path(self.tmp.name, "nope.pem")
        with self.assertRaisesRegex(RuntimeError, "signing key unusable"):
            self.client_mod.LociClient(endpoint="http://x", token="t", signing_key_file=str(missing))
        rsa_pem = pathlib.Path(self.tmp.name, "rsa.pem")
        rsa_pem.write_bytes(rsa.generate_private_key(65537, 2048).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        with self.assertRaisesRegex(RuntimeError, "not an Ed25519 private key"):
            self.client_mod.LociClient(endpoint="http://x", token="t", signing_key_file=str(rsa_pem))

    def test_an_unknown_signature_version_is_refused(self):
        with self.assertRaisesRegex(ValueError, "signature_version must be 1 or 2, got 3"):
            self._client(signature_version=3)

    def test_the_v2_signature_is_bound_to_the_endpoint_path(self):
        c = self._client()
        body, headers = c._signed_request("http://srv.example:8201/elsewhere", json.loads(_body("tasks/list")))
        self.assertEqual(self.client.post("/a2a", content=body, headers=headers).status_code, 401)


if __name__ == "__main__":
    unittest.main()
