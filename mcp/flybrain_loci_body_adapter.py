"""Loci code-graph body adapter for the FlyBrain grow-to-spec framework.

Exposes the Loci code graph as a BodyAdapter so a grown circuit can learn to
route Loci MCP tool calls: given a code-graph node (CodeFile, CodeSymbol, or
Finding), pick the most useful next tool call.

Observation (obs_dim=20)
------------------------
A per-node feature vector extracted from code_graph_query results:
  [0]   node_kind: 0=file, 1=symbol, 2=finding, 3=investigation
  [1-5] symbol_kind one-hot: function, class, variable, method, other
  [6]   in_investigation: 1.0 if the node is referenced from an open investigation
  [7]   caller_count (log1p-scaled, /5 cap)
  [8]   callee_count (log1p-scaled, /5 cap)
  [9]   import_count (log1p-scaled, /5 cap)
  [10]  reference_count (log1p-scaled, /5 cap)
  [11]  finding_confidence: 0–1 if this is a Finding, else 0
  [12]  is_python: 1.0 if lang=python
  [13]  is_typescript: 1.0 if lang=typescript
  [14-19] reserved (zeros for now)

Action space (action_dim=4)
---------------------------
  0: code_graph_query — traverse the graph (find callers/callees/refs)
  1: classify_text    — classify a finding's text against known labels
  2: docs_search      — search documentation for the symbol
  3: no_op            — no further action needed for this node

Reward
------
  +1.0  action resolves a finding on the current node (changes it from open to
        resolved in the graph)
  +0.5  action returns useful results (non-empty, non-error)
  +0.0  action returns empty or error
  -0.1  no_op on a node that still has open findings

The grow-to-spec hook `reward(*, samples, realism_scores, ...)` returns the
mean task reward across CMA-ES sample batch (not used for live graph interaction
— that uses step()).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import numpy as np

# Add mcp directory to path when run standalone
_MCP_DIR = os.path.dirname(os.path.abspath(__file__))
if _MCP_DIR not in sys.path:
    sys.path.insert(0, _MCP_DIR)

from flybrain_body_adapter import FlyBrainBodyAdapter, StepResult

OBS_DIM = 20
ACTION_DIM = 4

_NODE_KINDS = {"file": 0, "symbol": 1, "finding": 2, "investigation": 3}
_SYMBOL_KINDS = {"function": 1, "class": 2, "variable": 3, "method": 4}
_ACTION_NAMES = {0: "code_graph_query", 1: "classify_text", 2: "docs_search", 3: "no_op"}
_CONFIDENCE_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3, "unknown": 0.0}


class LociCodeGraphAdapter(FlyBrainBodyAdapter):
    """Live adapter that queries the Loci code graph and rewards useful tool calls.

    Requires a running Loci MCP server with ``code_graph_query`` available.
    Falls back to a mock graph when ``mock=True`` (for testing).
    """

    def __init__(self, mock: bool = False, seed: int | None = None) -> None:
        self._mock = mock
        self._rng = np.random.default_rng(seed)
        self._current_node: dict[str, Any] = {}
        self._step_count: int = 0
        self._max_steps: int = 20
        self._episode_reward: float = 0.0
        if not mock:
            try:
                from graph_tools import code_graph_query
                self._query_fn = code_graph_query
            except ImportError:
                self._query_fn = None
        else:
            self._query_fn = None

    # ------------------------------------------------------------------
    # FlyBrainBodyAdapter interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "loci_code_graph"

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_node = self._sample_node()
        return self._encode_node(self._current_node)

    def step(self, action: np.ndarray) -> StepResult:
        action = self._coerce_action(action)  # validates shape and finite values
        action_idx = int(np.argmax(action))
        reward = self._execute_action(action_idx, self._current_node)
        self._episode_reward += reward
        self._step_count += 1
        done = self._step_count >= self._max_steps
        self._current_node = self._sample_node()
        obs = self._encode_node(self._current_node)
        return StepResult(obs=obs, reward=float(reward), done=done, info={"action": _ACTION_NAMES[action_idx]})

    def reward(self, *, samples: list, realism_scores: list, **kwargs: Any) -> float:
        """Grow-to-spec hook: reward = mean realism score (critic-aligned)."""
        if not realism_scores:
            return 0.0
        return float(np.mean(realism_scores))

    # ------------------------------------------------------------------
    # Node sampling
    # ------------------------------------------------------------------

    def _sample_node(self) -> dict[str, Any]:
        if self._mock:
            return self._mock_node()
        nodes = self._query_live_nodes()
        if not nodes:
            return self._mock_node()
        return nodes[int(self._rng.integers(0, len(nodes)))]

    def _query_live_nodes(self) -> list[dict[str, Any]]:
        if self._query_fn is None:
            return []
        try:
            result_str = self._query_fn(
                "MATCH (s:CodeSymbol) RETURN s.id AS id, s.name AS name, s.kind AS kind, "
                "s.file AS file, s.lang AS lang LIMIT 100"
            )
            result = json.loads(result_str)
            rows = result.get("rows", [])
            nodes = []
            for row in rows:
                nodes.append({
                    "node_kind": "symbol",
                    "symbol_kind": str(row.get("kind", "function")).lower(),
                    "id": str(row.get("id", "")),
                    "name": str(row.get("name", "")),
                    "lang": str(row.get("lang", "")),
                    "in_investigation": False,
                    "caller_count": 0,
                    "callee_count": 0,
                    "import_count": 0,
                    "reference_count": 0,
                    "finding_confidence": 0.0,
                })
            return nodes
        except Exception:
            return []

    def _mock_node(self) -> dict[str, Any]:
        kinds = ["function", "class", "variable", "method"]
        langs = ["python", "typescript", "python", "python"]
        idx = int(self._rng.integers(0, len(kinds)))
        return {
            "node_kind": "symbol",
            "symbol_kind": kinds[idx],
            "id": f"mock:symbol:{idx}",
            "name": f"mock_fn_{idx}",
            "lang": langs[idx],
            "in_investigation": bool(self._rng.random() > 0.7),
            "caller_count": int(self._rng.integers(0, 8)),
            "callee_count": int(self._rng.integers(0, 12)),
            "import_count": int(self._rng.integers(0, 5)),
            "reference_count": int(self._rng.integers(0, 3)),
            "finding_confidence": float(self._rng.uniform(0, 1) if self._rng.random() > 0.5 else 0.0),
        }

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_node(self, node: dict[str, Any]) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # [0] node kind
        obs[0] = float(_NODE_KINDS.get(node.get("node_kind", "symbol"), 1)) / 3.0
        # [1-5] symbol kind one-hot
        sk = _SYMBOL_KINDS.get(node.get("symbol_kind", ""), 0)
        if 1 <= sk <= 4:
            obs[sk] = 1.0
        # [6] in investigation
        obs[6] = 1.0 if node.get("in_investigation") else 0.0
        # [7-10] edge counts (log1p / 5)
        for i, key in enumerate(("caller_count", "callee_count", "import_count", "reference_count")):
            obs[7 + i] = min(math.log1p(float(node.get(key, 0))) / 5.0, 1.0)
        # [11] finding confidence
        obs[11] = float(node.get("finding_confidence", 0.0))
        # [12-13] language one-hot
        lang = str(node.get("lang", "")).lower()
        obs[12] = 1.0 if "python" in lang else 0.0
        obs[13] = 1.0 if "type" in lang or "ts" in lang else 0.0
        # [14-19] reserved
        return obs

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action_idx: int, node: dict[str, Any]) -> float:
        if action_idx == 3:  # no_op
            # penalize no_op if there are open findings
            return -0.1 if node.get("in_investigation") else 0.0

        if self._mock:
            # mock: 60% chance of useful result, 20% chance of resolving
            r = self._rng.random()
            if r < 0.20:
                return 1.0
            elif r < 0.60:
                return 0.5
            else:
                return 0.0

        # live execution
        try:
            result = self._run_action(action_idx, node)
            if result is None:
                return 0.0
            if isinstance(result, dict) and "error" in result:
                return 0.0
            rows = result.get("rows", []) if isinstance(result, dict) else []
            if not rows:
                return 0.0
            # Reward: +1.0 if we found a resolved finding, else +0.5 for useful results
            for row in rows:
                if isinstance(row, dict) and str(row.get("confidence", "")).lower() == "high":
                    return 1.0
            return 0.5
        except Exception:
            return 0.0

    def _run_action(self, action_idx: int, node: dict[str, Any]) -> dict[str, Any] | None:
        node_id = node.get("id", "")
        if action_idx == 0:  # code_graph_query
            if self._query_fn is None:
                return None
            result_str = self._query_fn(
                "MATCH (s:CodeSymbol {id: $id})-[:CALLS]->(t:CodeSymbol) "
                "RETURN t.id AS id, t.name AS name, t.kind AS kind LIMIT 20",
                {"id": node_id},
            )
            return json.loads(result_str)
        elif action_idx == 1:  # classify_text
            try:
                from llm_tools import classify_text
                result_str = classify_text(
                    text=node.get("name", ""),
                    labels=["security", "performance", "correctness", "documentation"],
                )
                return {"rows": [{"label": result_str}]}
            except Exception:
                return None
        elif action_idx == 2:  # docs_search
            return None  # stub — no live docs_search in sync mode
        return None
