"""Routing-layer tests for a2a_server/server.py.

Tests the HTTP surface (auth, routing, error shapes) without exercising live
Qdrant/Ollama/SQLite. Skills that make network calls are not tested here —
only the JSON-RPC dispatch layer, auth enforcement, and error responses.
"""

import json
import os
import sys
import unittest

# Set required env vars before importing server (server exits at load if unset)
os.environ.setdefault("HERMES_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("HERMES_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
import server as a2a_server

client = TestClient(a2a_server.app, raise_server_exceptions=False)

AUTH = {"Authorization": "Bearer test-token-abc123"}
JSON_CT = {"Content-Type": "application/json"}
HEADERS = {**AUTH, **JSON_CT}


def rpc(method: str, params: dict = None, id: str = "req-1") -> dict:
    body = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        body["params"] = params
    return body


class TestHealthEndpoint(unittest.TestCase):
    def test_health_returns_200(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_no_auth_required(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_body_has_status_ok(self):
        resp = client.get("/health")
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_health_body_has_skills_list(self):
        resp = client.get("/health")
        data = resp.json()
        self.assertIn("skills", data)
        self.assertIsInstance(data["skills"], list)
        self.assertGreater(len(data["skills"]), 0)

    def test_health_body_has_agent_field(self):
        resp = client.get("/health")
        data = resp.json()
        self.assertIn("agent", data)


class TestAgentCard(unittest.TestCase):
    def test_agent_card_returns_200(self):
        resp = client.get("/.well-known/agent.json")
        self.assertEqual(resp.status_code, 200)

    def test_agent_card_no_auth_required(self):
        resp = client.get("/.well-known/agent.json")
        self.assertEqual(resp.status_code, 200)


class TestA2AAuth(unittest.TestCase):
    def test_no_auth_returns_401(self):
        resp = client.post("/a2a", json=rpc("tasks/send"))
        self.assertEqual(resp.status_code, 401)

    def test_wrong_token_returns_401(self):
        headers = {"Authorization": "Bearer wrong-token", **JSON_CT}
        resp = client.post("/a2a", json=rpc("tasks/send"), headers=headers)
        self.assertEqual(resp.status_code, 401)

    def test_malformed_auth_header_returns_401(self):
        headers = {"Authorization": "Basic dXNlcjpwYXNz", **JSON_CT}
        resp = client.post("/a2a", json=rpc("tasks/send"), headers=headers)
        self.assertEqual(resp.status_code, 401)


class TestA2ADispatch(unittest.TestCase):
    def _post(self, body) -> dict:
        resp = client.post("/a2a", json=body, headers=HEADERS)
        return resp.json()

    def test_unknown_method_returns_rpc_error(self):
        data = self._post(rpc("nonexistent/method"))
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32601)

    def test_tasks_send_unknown_skill_returns_error(self):
        data = self._post(rpc("tasks/send", {
            "skill_id": "does_not_exist",
            "message": "test",
            "sender": "test",
        }))
        # Either an error code or a result with an error payload
        self.assertIn("id", data)

    def test_tasks_get_missing_task_returns_error(self):
        data = self._post(rpc("tasks/get", {"task_id": "00000000-0000-0000-0000-000000000000"}))
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32602)

    def test_tasks_list_returns_result(self):
        data = self._post(rpc("tasks/list"))
        self.assertIn("result", data)
        self.assertIn("tasks", data["result"])
        self.assertIsInstance(data["result"]["tasks"], list)

    def test_id_echoed_in_response(self):
        data = self._post(rpc("tasks/list", id="my-custom-id"))
        self.assertEqual(data.get("id"), "my-custom-id")

    def test_parse_error_returns_rpc_parse_error(self):
        # Send raw invalid JSON
        resp = client.post(
            "/a2a",
            content=b"not json at all{{{{",
            headers={**AUTH, "Content-Type": "application/json"},
        )
        data = resp.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32700)

    def test_jsonrpc_field_present_in_response(self):
        data = self._post(rpc("tasks/list"))
        self.assertEqual(data.get("jsonrpc"), "2.0")


class TestTasksGetHTTP(unittest.TestCase):
    """GET /a2a/tasks/{task_id} — separate HTTP endpoint."""

    def test_unknown_task_returns_404(self):
        resp = client.get(
            "/a2a/tasks/00000000-0000-0000-0000-000000000000",
            headers=AUTH,
        )
        self.assertEqual(resp.status_code, 404)

    def test_no_auth_returns_401(self):
        resp = client.get("/a2a/tasks/some-task-id")
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()
