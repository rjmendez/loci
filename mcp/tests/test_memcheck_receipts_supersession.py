"""memcheck: explicit receipt citation, comparable claim types, supersession.

Regressions from 2026-10-04: a long narrative observed finding backed by a real
receipt never cleared the lexical bar; an observed finding was promoted to a
hallucination candidate by a 'procedure' finding sharing negation words; a
SUPERSEDES pair left the old finding open and un-flagged.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402
from memcheck.checks import run_contradiction, run_provenance, run_supersession  # noqa: E402

NARRATIVE = (
    "Across the whole incident window the edge proxy fleet rotated certificates, "
    "restarted workers in waves, drained connection pools, replayed queued jobs and "
    "eventually settled after the upstream resolver cache expired on every node "
    "within the staging and production clusters, which explains the latency spike"
)


def _f(fid, text, ftype="observed", source="", **extra):
    return {"id": fid, "text": text, "type": ftype, "source": source, **extra}


class ReceiptCitationTest(unittest.TestCase):
    def test_lexical_alone_misses_long_narrative(self):
        receipt = {"tool": "kubectl_logs", "receipt_id": "rcpt-20261004-aaaaaaaaaaaa",
                   "output": "worker restart pool drain"}
        out = run_provenance([_f("f1", NARRATIVE, source="kubectl_logs")], [receipt])
        self.assertEqual(len(out), 1)

    def test_cited_receipt_supports_long_narrative(self):
        receipt = {"tool": "kubectl_logs", "receipt_id": "rcpt-20261004-aaaaaaaaaaaa",
                   "output": "worker restart pool drain"}
        f = _f("f1", NARRATIVE, source="kubectl_logs",
               metadata={"receipt_ids": ["rcpt-20261004-aaaaaaaaaaaa"]})
        self.assertEqual(run_provenance([f], [receipt]), [])

    def test_unknown_or_implausible_citation_does_not_support(self):
        receipt = {"tool": "nmap", "receipt_id": "r1", "output": "x", "ts": "2026-10-04T10:00:00+00:00"}
        unknown = _f("f1", NARRATIVE, source="nmap", metadata={"receipt_ids": ["nope"]})
        wrong_tool = _f("f2", NARRATIVE, source="kubectl_logs", metadata={"receipt_ids": ["r1"]})
        late = _f("f3", NARRATIVE, source="nmap", ts="2026-09-01T00:00:00+00:00",
                  metadata={"receipt_ids": ["r1"]})
        out = run_provenance([unknown, wrong_tool, late], [receipt])
        self.assertEqual(sorted(r for v in out for r in v.refs), ["f1", "f2", "f3"])

    def test_global_log_receipt_accepted(self):
        glob = {"tool": "nmap", "receipt_id": "g1", "output": "x"}
        f = _f("f1", NARRATIVE, source="nmap", metadata={"receipt_ids": ["g1"]})
        self.assertEqual(run_provenance([f], [], global_audit_entries=[glob]), [])

    def test_lexical_fallback_still_passes(self):
        text = "port 443 open on gateway-7 running nginx 1.25"
        out = run_provenance([_f("f1", text, source="nmap")],
                             [{"tool": "nmap", "output": text}])
        self.assertEqual(out, [])


class ComparableTypesTest(unittest.TestCase):
    OBS = "The deploy script avoids restarting the gateway during rollout windows"
    PROC = "Do not restart the gateway during rollout windows; deploy script rollout"

    def test_procedure_never_contradicts_observed(self):
        out = run_contradiction([_f("o", self.OBS), _f("p", self.PROC, ftype="procedure")])
        self.assertEqual(out, [])
        for t in ("gap", "assumed"):
            self.assertEqual(run_contradiction([_f("o", self.OBS), _f("x", self.PROC, ftype=t)]), [])

    def test_observed_pair_still_contradicts(self):
        out = run_contradiction([_f("o", self.OBS), _f("o2", self.PROC)])
        self.assertEqual(len(out), 1)

    def test_no_retract_hint_for_non_observed_counterpart(self):
        findings = [_f("o", self.OBS, record_type="observed"),
                    _f("p", self.PROC, ftype="procedure", record_type="procedure")]
        from memcheck.verdict import new_verdict
        unsup = [new_verdict(subject_kind="memory", subject_signature="o", subject_excerpt="",
                             verdict_type="unsupported_observed", decision="warn",
                             confidence=0.6, rationale="", source="rule", refs=["o"])]
        contra = [new_verdict(subject_kind="memory", subject_signature="o|p", subject_excerpt="",
                              verdict_type="contradiction", decision="flag",
                              confidence=0.5, rationale="", source="rule", refs=["o", "p"])]
        self.assertEqual(server._hallucination_candidates(findings, [], unsup, contra), [])


class SupersessionTest(unittest.TestCase):
    OLD = "The cache layer is not enabled on the staging gateway cluster"
    NEW = "SUPERSEDES old1 The cache layer is enabled on the staging gateway cluster"

    def test_supersession_pair_not_a_contradiction(self):
        new = _f("new1", self.NEW, derived_from=["old1"])
        self.assertEqual(run_contradiction([_f("old1", self.OLD), new]), [])
        # control: without the marker the same pair is flagged
        ctrl = _f("new1", "The cache layer is enabled on the staging gateway cluster")
        self.assertEqual(len(run_contradiction([_f("old1", self.OLD), ctrl])), 1)

    def test_old_finding_surfaced_as_info_superseded(self):
        out = run_supersession([_f("old1", self.OLD), _f("new1", self.NEW, derived_from=["old1"])])
        self.assertEqual(len(out), 1)
        v = out[0]
        self.assertEqual((v.verdict_type, v.decision), ("superseded", "info"))
        self.assertEqual(v.refs, ["old1", "new1"])

    def test_metadata_supersedes(self):
        new = _f("n", "cache enabled now", metadata={"supersedes": "old1"})
        self.assertEqual(len(run_supersession([_f("old1", self.OLD), new])), 1)

    def test_plain_derivation_is_not_supersession(self):
        new = _f("n", "cache is probably enabled", ftype="inferred", derived_from=["old1"])
        self.assertEqual(run_supersession([_f("old1", self.OLD), new]), [])


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name) / "mem"
        for p in (
            mock.patch.object(server, "_mnemo_remember", return_value=True),
            mock.patch.object(server, "_qdrant_upsert", return_value=None),
            mock.patch.object(server, "_get_qdrant", return_value=(None, None)),
            mock.patch.object(server, "_mirror_finding_to_ladybug", return_value=None),
            mock.patch.object(server, "_autolink_finding_to_ladybug", return_value=None),
        ):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_receipt_id_round_trip(self):
        inv = "mc-e2e"
        server.investigation_start(investigation_id=inv, title="t")
        rid = json.loads(server.audit_log(tool_name="kubectl_logs", inputs_json="{}",
                                          output="raw lines", investigation_id=inv))["receipt_id"]
        server.investigation_store(investigation_id=inv, finding_type="observed", text=NARRATIVE,
                                   source="kubectl_logs", metadata={"receipt_ids": [rid]})
        out = json.loads(server.memory_self_check(investigation_id=inv, record=False))
        self.assertEqual(out["counts"]["unsupported_observed"], 0, out)
        entries = server._read_jsonl(server._inv_dir(inv) / "audit.jsonl")
        self.assertEqual(entries[0]["receipt_id"], rid)

    def test_superseded_surfaced_in_self_check(self):
        inv = "mc-e2e-sup"
        server.investigation_start(investigation_id=inv, title="t")
        old = json.loads(server.investigation_store(
            investigation_id=inv, finding_type="inferred", text=SupersessionTest.OLD,
            source="x"))["finding_id"]
        server.investigation_store(
            investigation_id=inv, finding_type="inferred",
            text=f"SUPERSEDES {old} " + SupersessionTest.NEW.split(" ", 2)[2],
            source="x", derived_from=[old])
        out = json.loads(server.memory_self_check(investigation_id=inv, record=False))
        self.assertEqual(out["counts"]["superseded"], 1, out)
        self.assertEqual(out["counts"]["contradiction"], 0, out)


if __name__ == "__main__":
    unittest.main()
