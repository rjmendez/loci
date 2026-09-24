"""Provenance honesty: a tier nobody asserted must never pass as evidence.

Regression tests for the provenance-confidence audit (c22aac4). Each one pins a
place where Loci reported more evidentiary authority than it had:

- untagged rows (and Loci's own model/declaration writers) defaulted to
  tool_verified and counted as independent evidence in the firewall;
- the Mnemosyne round trip turned provenance_defaulted=True into False;
- verify_all / verify_finding passed the firewall on ANY other finding;
- loci_validated_knowledge_promotion blocked every confirmed finding, blamed
  the firewall, and ignored metadata tier aliases;
- audit_log receipts for model tools were stamped tool_verified;
- memory_self_check claimed a "receipted" counter-finding that had no receipt;
- investigation_load counted degraded (never-run) verifications as verified;
- FlyBrain experts with no validation data got an invented 0.82 confidence.

Backends are stubbed; storage is a temp dir.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import investigation_tools as IT  # noqa: E402
import llm_tools  # noqa: E402
import mnemo_ops  # noqa: E402
import server  # noqa: E402
import provenance_firewall as PF  # noqa: E402
import verify  # noqa: E402
from provenance_firewall import (  # noqa: E402
    assert_evidence_firewall,
    normalize_provenance_tier,
    provenance_fields,
)

CLAIM = "The payment service outage was caused by the redis connection pool exhaustion"


def _j(result: str) -> dict:
    return json.loads(result)


def _confirmed_gen(calls):
    def gen(prompt, fmt=None, max_tokens=256, **_kw):
        calls.append(prompt)
        return {"ok": True, "text": json.dumps(
            {"verdict": "confirmed", "refutation": "", "reasoning": "cannot refute", "confidence": 0.9})}
    return gen


class _Isolated(unittest.TestCase):
    """Temp MEMORY_DIR; Mnemosyne, Qdrant and LadybugDB stubbed out."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name) / "mem"
        self.mnemo_meta = []
        patches = [
            mock.patch.object(server, "_mnemo_remember",
                              side_effect=lambda content, importance=0.6, metadata=None:
                              self.mnemo_meta.append(metadata) or True),
            mock.patch.object(server, "_qdrant_upsert", return_value=None),
            mock.patch.object(server, "_mirror_finding_to_ladybug", return_value=None),
            mock.patch.object(server, "_autolink_finding_to_ladybug", return_value=None),
            mock.patch.object(verify, "_lazy_rag", return_value={}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_dir
        self._tmp.cleanup()

    def _start(self, inv):
        server.investigation_start(investigation_id=inv, title="t")

    def _rows(self, inv):
        return server._read_jsonl(server._inv_dir(inv) / "findings.jsonl")

    def _row(self, inv, fid):
        return next(r for r in self._rows(inv) if r.get("id") == fid)


# --- untagged-defaults-to-tool-verified / provenance-default-tool-verified-for-transcripts


class DefaultedTierIsNotIndependentTest(_Isolated):

    def test_firewall_does_not_count_defaulted_rows_as_independent(self):
        fw = assert_evidence_firewall(
            {"evidence_provenance_tier": "model_asserted"},
            [{"text": "untagged legacy row"},
             # Mnemosyne-shaped row: tier key present but flagged defaulted
             {"text": "[pre_compress] transcript", "evidence_provenance_tier": "tool_verified",
              "provenance_defaulted": True}],
        )
        self.assertFalse(fw["allowed"])
        self.assertEqual(fw["evidence_tiers"], ["tool_verified"])  # still displayed
        self.assertEqual(fw["independent_evidence_tiers"], [])
        self.assertEqual(fw["defaulted_evidence_count"], 2)

    def test_untagged_reasoned_finding_does_not_support_its_own_claim(self):
        # repro3: the untagged '[reasoned]' row used to give support_count=1.
        inv = "prov-pac-untagged"
        self._start(inv)
        server.investigation_store(
            investigation_id=inv, finding_type="inferred", text="[reasoned] " + CLAIM,
            source="investigation_reason", confidence="medium", tags="reasoned,investigation_reason")
        out = _j(server.investigation_pre_answer_check(investigation_id=inv, claims=[CLAIM], record=False))
        self.assertEqual(out["support_count"], 0)
        self.assertEqual(out["unsupported_claims"], [CLAIM])
        claim = out["claim_results"][0]
        self.assertEqual(claim["support_basis"], "provenance_blocked")
        self.assertFalse(claim["provenance_firewall"]["allowed"])

    def test_explicit_tool_verified_finding_still_supports(self):
        inv = "prov-pac-tool"
        self._start(inv)
        server.investigation_store(
            investigation_id=inv, finding_type="observed", text=CLAIM, source="redis-cli",
            confidence="high", evidence_provenance_tier="tool_verified")
        out = _j(server.investigation_pre_answer_check(investigation_id=inv, claims=[CLAIM], record=False))
        self.assertEqual(out["support_count"], 1)
        self.assertTrue(out["claim_results"][0]["provenance_firewall"]["allowed"])

    def test_investigation_reason_persists_model_asserted(self):
        inv = "prov-reason"
        self._start(inv)

        def call_llm(prompt, *a, json_mode=False, **kw):
            if json_mode:
                return json.dumps({"converged_claims": [CLAIM], "contested_areas": [],
                                   "final_answer": CLAIM})
            return "perspective: " + CLAIM

        from memcheck import llm as mc_llm
        with mock.patch.object(mc_llm, "llm_available", return_value=True), \
             mock.patch.object(mc_llm, "embed_texts", return_value=None), \
             mock.patch.object(mc_llm, "call_llm", side_effect=call_llm):
            out = _j(server.investigation_reason(investigation_id=inv, question="why?", persist=True))
        self.assertEqual(len(out["persisted_finding_ids"]), 1, out)
        row = self._row(inv, out["persisted_finding_ids"][0])
        self.assertEqual(provenance_fields(row),
                         {"evidence_provenance_tier": "model_asserted", "provenance_defaulted": False})

    def test_gap_and_assumed_findings_default_to_model_asserted(self):
        inv = "prov-gap"
        self._start(inv)
        for ftype in ("gap", "assumed"):
            fid = _j(server.investigation_store(
                investigation_id=inv, finding_type=ftype, text=f"{ftype} about host foo",
                source="agent", confidence="low"))["finding_id"]
            self.assertEqual(normalize_provenance_tier(self._row(inv, fid)), "model_asserted", ftype)
        # An explicit caller tier still wins.
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="gap", text="gap with a human tier",
            source="agent", confidence="low", evidence_provenance_tier="human_authored"))["finding_id"]
        self.assertEqual(normalize_provenance_tier(self._row(inv, fid)), "human_authored")
        # Observed rows from a caller keep the legacy (defaulted) display tier.
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="observed", text="observed untagged",
            source="agent", confidence="low"))["finding_id"]
        self.assertTrue(provenance_fields(self._row(inv, fid))["provenance_defaulted"])

    def test_declaration_and_reflection_writers_stamp_model_asserted(self):
        inv = "prov-writers"
        self._start(inv)
        server.wiring_obligation_declare(investigation_id=inv, class_name="Foo",
                                         method_name="bar", expected_effect="flush cache")
        server.contract_declare(investigation_id=inv, entity="Order", role="producer",
                                fields='{"id": "str", "total": "float"}')
        self.assertTrue(server._reflection_store_finding(
            investigation_id=inv, finding_type="observed", text="reflection tick summary",
            confidence="low", tags="self-reflection"))
        by_source = {r.get("source"): r for r in self._rows(inv) if r.get("text")}
        for source in ("wiring_obligation_declare", "contract_declare", "reflection_loop_tick"):
            self.assertIn(source, by_source)
            self.assertEqual(provenance_fields(by_source[source]),
                             {"evidence_provenance_tier": "model_asserted",
                              "provenance_defaulted": False}, source)


# --- mnemo-roundtrip-launders-defaulted-flag


class MnemoRoundTripTest(_Isolated):

    def test_defaulted_flag_survives_store_then_recall(self):
        inv = "prov-mnemo"
        self._start(inv)
        server.investigation_store(investigation_id=inv, finding_type="inferred",
                                   text="untagged claim", source="x", confidence="medium")
        meta = self.mnemo_meta[-1]
        self.assertTrue(meta["provenance_defaulted"])
        self.assertTrue(provenance_fields(meta)["provenance_defaulted"])
        raw = [{"content": "untagged claim", "metadata": meta, "score": 0.9}]
        with mock.patch.object(mnemo_ops, "_get_mnemo_funcs", return_value=(None, lambda **k: raw)):
            rows = mnemo_ops._mnemo_recall("untagged", investigation_id=inv)
        self.assertEqual(rows[0]["evidence_provenance_tier"], "tool_verified")
        self.assertTrue(rows[0]["provenance_defaulted"])


# --- firewall-satisfied-by-unrelated-findings


class FirewallRelevanceTest(_Isolated):

    def test_verify_all_does_not_pass_on_unrelated_findings(self):
        # repro2: an unrelated untagged row used to clear the gate -> 'confirmed'.
        inv = "prov-verify-all"
        self._start(inv)
        server.investigation_store(investigation_id=inv, finding_type="observed",
                                   text="Disk usage on host foo is 41 percent", source="df",
                                   confidence="high", evidence_provenance_tier="tool_verified")
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred",
            text="Service X leaks memory (the model said so)", source="llm",
            confidence="high", evidence_provenance_tier="model_asserted"))["finding_id"]
        calls = []
        with mock.patch.object(server, "_verify_gen_fn", _confirmed_gen(calls)):
            out = _j(server.investigation_verify_all(investigation_id=inv, limit=20))
        res = next(r for r in out["results"] if r["finding_id"] == fid)
        self.assertEqual(res["verdict"], "uncertain")
        self.assertEqual(res["confidence"], 0.0)

    def test_verify_all_passes_on_linked_independent_evidence(self):
        inv = "prov-verify-linked"
        self._start(inv)
        server.investigation_store(investigation_id=inv, finding_type="observed",
                                   text="Service X memory grows 2GB per hour in heap profile",
                                   source="pprof", confidence="high",
                                   evidence_provenance_tier="tool_verified")
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred",
            text="Service X memory grows per hour: heap leak", source="llm",
            confidence="high", evidence_provenance_tier="model_asserted"))["finding_id"]
        calls = []
        with mock.patch.object(server, "_verify_gen_fn", _confirmed_gen(calls)):
            out = _j(server.investigation_verify_all(investigation_id=inv, limit=20))
        res = next(r for r in out["results"] if r["finding_id"] == fid)
        self.assertEqual(res["verdict"], "confirmed")
        self.assertTrue(calls)

    def test_derived_from_parent_counts_as_linked(self):
        inv = "prov-verify-parent"
        self._start(inv)
        parent = _j(server.investigation_store(
            investigation_id=inv, finding_type="observed", text="nmap: 10.0.0.5 tcp/22 open",
            source="nmap", confidence="high", evidence_provenance_tier="tool_verified"))["finding_id"]
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred", text="The jump box accepts SSH",
            source="llm", confidence="high", evidence_provenance_tier="model_asserted",
            derived_from=[parent]))["finding_id"]
        linked = server._firewall_linked_evidence(inv, self._row(inv, fid))
        self.assertEqual([e["evidence_id"] for e in linked], [parent])

    def test_verify_finding_tool_ignores_unrelated_findings(self):
        inv = "prov-verify-tool"
        self._start(inv)
        server.wiring_obligation_declare(investigation_id=inv, class_name="Foo",
                                         method_name="bar", expected_effect="flush cache")
        server.investigation_store(investigation_id=inv, finding_type="observed",
                                   text="Disk usage on host foo is 41 percent", source="df",
                                   confidence="high", evidence_provenance_tier="tool_verified")
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred",
            text="Service X leaks memory (the model said so)", source="llm",
            confidence="high", evidence_provenance_tier="model_asserted"))["finding_id"]
        calls = []
        with mock.patch.object(verify, "_lazy_generate", _confirmed_gen(calls)):
            out = _j(llm_tools.verify_finding("Service X leaks memory (the model said so)",
                                              investigation_id=inv, finding_id=fid))
        self.assertEqual(out["verdict"], "uncertain")
        self.assertFalse(out["provenance_firewall"]["allowed"])
        self.assertEqual(calls, [])


# --- promotion-always-blocked-with-false-reason / promotion-ignores-metadata-tier-aliases


class PromotionGateTest(_Isolated):

    def test_confirmed_tool_verified_finding_is_promoted(self):
        # repro2: used to be blocked with reason 'provenance_firewall'.
        inv = "prov-promote"
        self._start(inv)
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="observed", text="Host foo runs nginx 1.24",
            source="nmap", confidence="high", evidence_provenance_tier="tool_verified"))["finding_id"]
        with mock.patch.object(verify, "_lazy_generate", _confirmed_gen([])):
            out = _j(server.loci_validated_knowledge_promotion(investigation_id=inv, finding_id=fid))
        self.assertEqual(out["status"], "promoted", out)
        self.assertEqual(out["reason"], "verified_promoted")

    def test_firewall_block_is_reported_as_firewall_not_verification(self):
        inv = "prov-promote-model"
        self._start(inv)
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred", text="Host foo is compromised",
            source="llm", confidence="high", evidence_provenance_tier="model_asserted"))["finding_id"]
        with mock.patch.object(verify, "_lazy_generate", _confirmed_gen([])):
            out = _j(server.loci_validated_knowledge_promotion(investigation_id=inv, finding_id=fid))
        self.assertEqual(out["status"], "blocked")
        self.assertEqual(out["reason"], "provenance_firewall")

    def test_metadata_tier_aliases_reach_the_firewall(self):
        # repro1 check 4: promotion read only the top-level field -> tool_verified.
        inv = "prov-promote-alias"
        self._start(inv)
        for md in ({"provenance_tier": "model_asserted"}, {"evidence_kind": "llm"}):
            fid = _j(server.investigation_store(
                investigation_id=inv, finding_type="inferred", text="model says Y " + str(md),
                source="x", confidence="high", metadata=md))["finding_id"]
            calls = []
            with mock.patch.object(verify, "_lazy_generate", _confirmed_gen(calls)):
                out = _j(server.loci_validated_knowledge_promotion(investigation_id=inv, finding_id=fid))
            self.assertEqual(out["status"], "blocked", md)
            self.assertEqual(out["reason"], "provenance_firewall", md)
            self.assertEqual(out["verification"]["provenance_firewall"]["candidate_tier"],
                             "model_asserted", md)
            self.assertEqual(calls, [], md)


# --- audit-log-caller-text-is-tool-verified


class AuditReceiptProvenanceTest(_Isolated):

    def test_tool_name_decides_the_receipt_tier(self):
        audit_provenance_fields = PF.audit_provenance_fields
        self.assertEqual(audit_provenance_fields("llm_local")["evidence_provenance_tier"], "model_asserted")
        self.assertEqual(audit_provenance_fields("mcp__loci__swarm_reason")["evidence_provenance_tier"],
                         "model_asserted")
        self.assertEqual(audit_provenance_fields("nmap"),
                         {"evidence_provenance_tier": "tool_verified", "provenance_defaulted": False})
        self.assertTrue(audit_provenance_fields("")["provenance_defaulted"])

    def test_model_tool_receipt_does_not_support_a_claim(self):
        # repro4: audit_log(tool_name='llm_local') used to give support_count=1.
        inv = "prov-audit-llm"
        self._start(inv)
        with mock.patch.object(server, "_get_qdrant", return_value=(None, None)):
            server.audit_log(tool_name="llm_local", inputs_json="{}", output="I think " + CLAIM,
                             investigation_id=inv)
            out = _j(server.investigation_pre_answer_check(investigation_id=inv, claims=[CLAIM],
                                                           record=False))
        self.assertEqual(out["support_count"], 0)
        fw = out["claim_results"][0]["provenance_firewall"]
        self.assertFalse(fw["allowed"])
        self.assertEqual(fw["evidence_tiers"], ["model_asserted"])

    def test_non_model_tool_receipt_still_supports(self):
        inv = "prov-audit-tool"
        self._start(inv)
        with mock.patch.object(server, "_get_qdrant", return_value=(None, None)):
            server.audit_log(tool_name="redis_cli_info", inputs_json="{}", output=CLAIM,
                             investigation_id=inv)
            out = _j(server.investigation_pre_answer_check(investigation_id=inv, claims=[CLAIM],
                                                           record=False))
        self.assertEqual(out["support_count"], 1)


# --- hallucination-candidate-false-receipt-claim


class HallucinationRationaleTest(unittest.TestCase):

    @staticmethod
    def _v(vtype, refs):
        from memcheck.verdict import Verdict
        return Verdict(id="v-" + "-".join(refs), subject_kind="memory", subject_signature=refs[0],
                       subject_excerpt="", verdict_type=vtype, decision="flag", confidence=1.0,
                       rationale="", source="rule", refs=list(refs))

    def test_blanket_unsupported_does_not_claim_a_receipt(self):
        findings = [{"id": "f-a", "record_type": "observed", "text": "anchor is calibrated"},
                    {"id": "f-b", "record_type": "observed", "text": "anchor is not calibrated"}]
        out = server._hallucination_candidates(
            findings, [], [self._v("unsupported_observed", ["f-a"]),
                           self._v("unsupported_observed", ["f-b"])],
            [self._v("contradiction", ["f-a", "f-b"])])
        self.assertTrue(out)
        for c in out:
            self.assertFalse(c["counter_receipted"])
            self.assertNotIn("receipted finding", c["rationale"])
            self.assertNotIn("memory_retract", c["hint"])

    def test_unchecked_inferred_finding_is_not_a_receipt(self):
        # f-c is observed and receipted, so this is not blanket; f-b is inferred and
        # was never checked for a receipt, so it cannot be receipted counter-evidence.
        findings = [{"id": "f-a", "record_type": "observed", "text": "relay is up"},
                    {"id": "f-b", "record_type": "inferred", "text": "relay is down"},
                    {"id": "f-c", "record_type": "observed", "text": "unrelated"}]
        out = server._hallucination_candidates(
            findings, [], [self._v("unsupported_observed", ["f-a"])],
            [self._v("contradiction", ["f-a", "f-b"])])
        self.assertEqual(out, [])


# --- verified-findings-count-includes-degraded


class VerificationCountTest(unittest.TestCase):

    def test_degraded_runs_are_not_verified_findings(self):
        rows = [{"record_type": "verification", "finding_id": f"f{i}", "verdict": "uncertain",
                 "confidence": 0.0, "degraded": True, "ts": "2026-09-01T00:00:00Z"} for i in range(20)]
        with mock.patch.object(IT, "_read_jsonl", return_value=rows), mock.patch.object(IT, "_inv_dir"):
            s = IT._verification_summary("inv")
        self.assertEqual(s["counts"]["degraded"], 20)
        self.assertEqual(s["verified_findings"], 0)
        self.assertEqual(s["verification_attempts"], 20)


# --- flybrain-invented-0.82-confidence


class FlyBrainUncalibratedConfidenceTest(unittest.TestCase):

    def _payload(self):
        import flybrain_brain_cluster_pipeline as p
        manifest = types.SimpleNamespace(region_counts={"r_noval": 5, "r_zero": 5, "r_ok": 5},
                                         manifest_id="m1")

        def res(val):
            return types.SimpleNamespace(metrics={"val_metrics": val},
                                         model_fingerprint="a" * 64, metrics_fingerprint="b" * 64)
        results = {
            "r_noval": res({"sample_count": 0, "accuracy": 0.0,
                            "warning": "no validation samples in selected split"}),
            "r_zero": res({"sample_count": 12, "accuracy": 0.0}),
            "r_ok": res({"sample_count": 12, "accuracy": 0.75}),
        }
        payload = p._build_experts_runtime_payload(manifest=manifest, expert_results=results,
                                                   shadow_overrides=None)
        return {row["region"]: row["shadow_replay"] for row in payload["experts"]}

    def test_no_invented_confidence(self):
        shadow = self._payload()
        self.assertEqual(shadow["r_noval"]["confidence"], 0.0)
        self.assertEqual(shadow["r_noval"]["confidence_calibration"], "uncalibrated")
        self.assertEqual(shadow["r_zero"]["confidence"], 0.0)  # measured, kept
        self.assertEqual(shadow["r_zero"]["confidence_calibration"], "validation_accuracy")
        self.assertEqual(shadow["r_ok"]["confidence"], 0.75)

    def test_uncalibrated_expert_fails_the_confidence_gate(self):
        import flybrain_brain_cluster as fbc
        gate = fbc.ConfidenceGate(min_confidence=0.65)
        task = types.SimpleNamespace(task_type="verification", cluster_id="c", request_id="r")
        prov = types.SimpleNamespace(replay_fingerprint="f" * 64)
        # No confidence at all: used to default to 0.82 and pass.
        out = fbc._ArtifactReplayExpert("e1", behavior={}).infer(task, prov)
        self.assertEqual(out.confidence, 0.0)
        result = gate.evaluate(task, prov, None, out)
        self.assertEqual(result.decision, fbc.ClusterDecision.FAIL_CLOSED)
        self.assertIn("uncalibrated", result.reason)
        # Uncalibrated stays failed even if a number was forced in.
        forced = fbc.ExpertOutput(expert_id="e2", confidence=0.99, claims=[],
                                  artifacts={"confidence_calibration": "uncalibrated"})
        self.assertEqual(gate.evaluate(task, prov, None, forced).decision,
                         fbc.ClusterDecision.FAIL_CLOSED)
        # A declared, measured confidence still passes.
        ok = fbc._ArtifactReplayExpert("e3", behavior={"confidence": 0.9}).infer(task, prov)
        self.assertEqual(gate.evaluate(task, prov, None, ok).decision, fbc.ClusterDecision.ACCEPT)


# --- review follow-ups: retracted evidence, defaulted candidates, unknown tier aliases


class ReviewFollowUpTest(_Isolated):

    def setUp(self):
        super().setUp()
        for name, value in {
            "_retract_quarantine_verdict": lambda *a, **k: False,
            "_forget_finding_verdicts": lambda *a, **k: 0,
            "_semantic_neighbor_ids": lambda *a, **k: [],
            "_event_log_append": lambda *a, **k: None,
        }.items():
            p = mock.patch.object(server, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)

    def _retract(self, inv, fid):
        r = _j(server.memory_retract(inv, fid, reason="hallucination", dry_run=False,
                                     scope_semantic=False))
        self.assertTrue(r.get("applied"), r)

    def test_retracted_finding_is_not_pre_answer_support(self):
        # Probe P1: the retracted row still gave support_count=1.
        inv = "prov-rev-p1"
        self._start(inv)
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="observed", text=CLAIM, source="redis-cli",
            confidence="high", evidence_provenance_tier="tool_verified"))["finding_id"]
        self._retract(inv, fid)
        out = _j(server.investigation_pre_answer_check(investigation_id=inv, claims=[CLAIM], record=False))
        self.assertEqual(out["support_count"], 0, out)
        self.assertNotIn(fid, json.dumps(out["claim_results"]))
        self.assertEqual(out["evidence_lanes"]["retraction"]["excluded_retracted"], 1)

    def test_retracted_finding_is_not_linked_firewall_evidence(self):
        # Probe P6: the retracted tool_verified row cleared the firewall for a model claim.
        inv = "prov-rev-p6"
        self._start(inv)
        bad = _j(server.investigation_store(
            investigation_id=inv, finding_type="observed",
            text="Service X memory grows 2GB per hour in heap profile", source="pprof",
            confidence="high", evidence_provenance_tier="tool_verified"))["finding_id"]
        self._retract(inv, bad)
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred",
            text="Service X memory grows per hour: heap leak", source="llm",
            confidence="high", evidence_provenance_tier="model_asserted"))["finding_id"]
        self.assertEqual(server._firewall_linked_evidence(inv, self._row(inv, fid)), [])
        calls = []
        with mock.patch.object(server, "_verify_gen_fn", _confirmed_gen(calls)):
            out = _j(server.investigation_verify_all(investigation_id=inv, limit=20))
        res = next(r for r in out["results"] if r["finding_id"] == fid)
        self.assertEqual(res["verdict"], "uncertain")
        self.assertEqual(calls, [])

    def test_untagged_candidate_needs_linked_evidence(self):
        # Probe P7: an untagged inferred finding reached the verifier as tool_verified
        # with zero linked evidence and came back 'confirmed'.
        inv = "prov-rev-p7"
        self._start(inv)
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred",
            text="Therefore host delta is compromised", source="agent"))["finding_id"]
        self.assertTrue(provenance_fields(self._row(inv, fid))["provenance_defaulted"])
        calls = []
        with mock.patch.object(server, "_verify_gen_fn", _confirmed_gen(calls)):
            out = _j(server.investigation_verify_all(investigation_id=inv, limit=20))
        res = next(r for r in out["results"] if r["finding_id"] == fid)
        self.assertEqual(res["verdict"], "uncertain")
        self.assertEqual(calls, [])
        with mock.patch.object(verify, "_lazy_generate", _confirmed_gen(calls)):
            promo = _j(server.loci_validated_knowledge_promotion(investigation_id=inv, finding_id=fid))
        self.assertEqual(promo["status"], "blocked", promo)
        self.assertEqual(promo["reason"], "provenance_firewall")
        self.assertEqual(calls, [])

    def test_unknown_tier_alias_is_still_defaulted(self):
        # Probes P4/P5: evidence_kind='log_excerpt'/'hunch' made an untagged row look asserted.
        f = provenance_fields({"metadata": {"evidence_kind": "log_excerpt"}})
        self.assertEqual(f, {"evidence_provenance_tier": "tool_verified", "provenance_defaulted": True})
        fw = assert_evidence_firewall({"evidence_provenance_tier": "model_asserted"},
                                      [{"text": "x", "metadata": {"evidence_kind": "log_excerpt"}}])
        self.assertFalse(fw["allowed"])
        # Known aliases still count as asserted.
        self.assertFalse(provenance_fields({"metadata": {"evidence_kind": "tool"}})["provenance_defaulted"])
        self.assertEqual(provenance_fields({"evidence_kind": "llm"}),
                         {"evidence_provenance_tier": "model_asserted", "provenance_defaulted": False})
        inv = "prov-rev-p5"
        self._start(inv)
        fid = _j(server.investigation_store(
            investigation_id=inv, finding_type="inferred", text="a hunch about foo",
            source="agent", metadata={"evidence_kind": "hunch"}))["finding_id"]
        self.assertTrue(provenance_fields(self._row(inv, fid))["provenance_defaulted"])


if __name__ == "__main__":
    unittest.main()
