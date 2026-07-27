"""Peer credential resolution tests for a2a_server/server.py.

Covers the outbound half of the mesh: how context_broadcast / memory_prime
build Authorization and X-TOTP headers for peer A2A endpoints. Pure header
construction — no network.
"""

import importlib.util
import os
import pathlib
import unittest
from unittest import mock

import pyotp

os.environ.setdefault("HERMES_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("HERMES_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_server_impl", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)

SEED = pyotp.random_base32()


class TestPeerBaseUrl(unittest.TestCase):
    def test_strips_a2a_suffix(self):
        self.assertEqual(
            a2a_server._peer_base_url("http://host:8201/a2a"), "http://host:8201"
        )

    def test_bare_base_url_unchanged(self):
        self.assertEqual(
            a2a_server._peer_base_url("http://host:8201"), "http://host:8201"
        )

    def test_trailing_slash_tolerated(self):
        self.assertEqual(
            a2a_server._peer_base_url("http://host:8201/a2a/"), "http://host:8201"
        )

    def test_does_not_eat_port_digits(self):
        # str.rstrip('/a2a') would return 'http://host:820' here.
        self.assertEqual(
            a2a_server._peer_base_url("http://host:8202/a2a"), "http://host:8202"
        )

    def test_does_not_eat_trailing_a_in_hostname(self):
        self.assertEqual(
            a2a_server._peer_base_url("http://alpha/a2a"), "http://alpha"
        )


class TestPeerCredentials(unittest.TestCase):
    def test_empty_env_yields_empty_config(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            token_map, token, seed_map, seed = a2a_server._peer_credentials()
        self.assertEqual((token_map, token, seed_map, seed), ({}, "", {}, ""))

    def test_reads_shared_token_and_seed(self):
        env = {"PEER_A2A_TOKEN": "tok", "PEER_A2A_TOTP_SEED": SEED}
        with mock.patch.dict(os.environ, env, clear=True):
            _, token, _, seed = a2a_server._peer_credentials()
        self.assertEqual(token, "tok")
        self.assertEqual(seed, SEED)

    def test_malformed_json_falls_back_to_empty_map(self):
        env = {
            "PEER_A2A_TOKENS_JSON": "{not json",
            "PEER_A2A_TOTP_SEEDS_JSON": "{also not json",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            token_map, _, seed_map, _ = a2a_server._peer_credentials()
        self.assertEqual(token_map, {})
        self.assertEqual(seed_map, {})


class TestPeerHeaders(unittest.TestCase):
    URL = "http://host:8201/a2a"
    BASE = "http://host:8201"

    def test_no_token_is_skipped(self):
        headers, reason = a2a_server._peer_headers(self.URL, {}, "", {}, "")
        self.assertIsNone(headers)
        self.assertEqual(reason, "no token configured")

    def test_bearer_only_when_no_seed(self):
        headers, reason = a2a_server._peer_headers(self.URL, {}, "tok", {}, "")
        self.assertIsNone(reason)
        self.assertEqual(headers["Authorization"], "Bearer tok")
        self.assertNotIn("X-TOTP", headers)

    def test_totp_attached_when_seed_configured(self):
        headers, reason = a2a_server._peer_headers(self.URL, {}, "tok", {}, SEED)
        self.assertIsNone(reason)
        self.assertTrue(pyotp.TOTP(SEED).verify(headers["X-TOTP"], valid_window=1))

    def test_per_peer_token_keyed_by_base_url(self):
        headers, _ = a2a_server._peer_headers(
            self.URL, {self.BASE: "base-tok"}, "shared", {}, ""
        )
        self.assertEqual(headers["Authorization"], "Bearer base-tok")

    def test_per_peer_token_keyed_by_full_url(self):
        headers, _ = a2a_server._peer_headers(
            self.URL, {self.URL: "full-tok"}, "shared", {}, ""
        )
        self.assertEqual(headers["Authorization"], "Bearer full-tok")

    def test_per_peer_seed_overrides_shared_seed(self):
        other = pyotp.random_base32()
        headers, _ = a2a_server._peer_headers(
            self.URL, {}, "tok", {self.BASE: other}, SEED
        )
        self.assertTrue(pyotp.TOTP(other).verify(headers["X-TOTP"], valid_window=1))

    def test_invalid_seed_reports_reason_instead_of_raising(self):
        headers, reason = a2a_server._peer_headers(
            self.URL, {}, "tok", {}, "not-a-valid-base32-seed!!"
        )
        self.assertIsNone(headers)
        self.assertIn("invalid TOTP seed", reason)

    def test_content_type_always_set(self):
        headers, _ = a2a_server._peer_headers(self.URL, {}, "tok", {}, "")
        self.assertEqual(headers["Content-Type"], "application/json")


if __name__ == "__main__":
    unittest.main()
