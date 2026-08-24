"""loci_groom — the passive grooming passes.

The index pass is the one that can write, so most of these pin its arithmetic and
its refusal to write without --apply. The tags pass is asserted to stay in the
proposal lane: nothing it produces may reach findings.jsonl, and nothing outside
the vocabulary it was handed may reach a proposal.
"""
import importlib.util
import json
import pathlib
import sys
import unittest
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("loci_groom", _SCRIPTS / "loci_groom.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


groom = _load()


def _corpus(tmp, spec):
    """spec: {investigation: [finding-dict, ...]} -> writes findings.jsonl files."""
    for inv, rows in spec.items():
        d = tmp / inv
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "findings.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps({"investigation_id": inv, **r}) + "\n")
    return tmp


def _f(fid, text="a finding about the reranker", tags=None):
    row = {"id": fid, "text": text}
    if tags is not None:
        row["tags"] = tags
    return row


class _Point:
    def __init__(self, pid):
        self.id = pid


class _Client:
    """Scrolls in pages, like the real one."""

    def __init__(self, ids, page=2):
        self._ids = list(ids)
        self._page = page
        self.upserts = []

    def scroll(self, offset=None, **_kw):
        start = offset or 0
        chunk = self._ids[start:start + self._page]
        nxt = start + self._page
        return [_Point(i) for i in chunk], (nxt if nxt < len(self._ids) else None)


class TestIterFindings(unittest.TestCase):
    def test_it_reads_every_investigation_and_skips_junk_lines(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv-a": [_f("1"), _f("2")], "inv-b": [_f("3")]})
            with open(tmp / "inv-a" / "findings.jsonl", "a") as fh:
                fh.write("not json\n\n")
            ids = sorted(f["id"] for f in groom.iter_findings(tmp))
            self.assertEqual(ids, ["1", "2", "3"])


class TestIndexPass(unittest.TestCase):
    def _run(self, disk_ids, indexed_ids, apply=False, limit=None):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._td.name)
        _corpus(tmp, {"inv": [_f(i) for i in disk_ids]})
        client = _Client(indexed_ids)
        fake_ops = mock.Mock()
        fake_ops._get_qdrant.return_value = (client, "hermes_memory")
        fake_ops._qdrant_upsert.side_effect = lambda pid, text, payload: client.upserts.append(pid)
        with mock.patch.dict(sys.modules, {"qdrant_ops": fake_ops}), \
             mock.patch.object(groom, "MEMORY_DIR", tmp):
            return groom.pass_index(apply=apply, limit=limit), client

    def tearDown(self):
        if getattr(self, "_td", None):
            self._td.cleanup()

    def test_it_counts_the_drift_between_disk_and_the_index(self):
        report, _ = self._run(["a", "b", "c", "d", "e"], ["a", "b"])
        self.assertEqual(report["on_disk"], 5)
        self.assertEqual(report["indexed"], 2)
        self.assertEqual(report["missing"], 3)
        self.assertEqual(report["coverage"], 0.4)

    def test_a_full_index_reports_no_drift(self):
        report, _ = self._run(["a", "b"], ["a", "b"])
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["coverage"], 1.0)

    def test_without_apply_it_writes_nothing(self):
        report, client = self._run(["a", "b", "c"], [])
        self.assertEqual(client.upserts, [])
        self.assertEqual(report["applied"], 0)

    def test_with_apply_it_reupserts_exactly_the_missing_ones(self):
        report, client = self._run(["a", "b", "c"], ["b"])
        self.assertEqual(report["applied"], 0)      # sanity: default is report-only
        report, client = self._run(["a", "b", "c"], ["b"], apply=True)
        self.assertEqual(sorted(client.upserts), ["a", "c"])
        self.assertEqual(report["applied"], 2)

    def test_limit_caps_the_write(self):
        report, client = self._run(["a", "b", "c", "d"], [], apply=True, limit=2)
        self.assertEqual(len(client.upserts), 2)

    def test_an_unreachable_qdrant_degrades_instead_of_raising(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f("a")]})
            fake_ops = mock.Mock()
            fake_ops._get_qdrant.return_value = (None, None)
            with mock.patch.dict(sys.modules, {"qdrant_ops": fake_ops}), \
                 mock.patch.object(groom, "MEMORY_DIR", tmp):
                report = groom.pass_index()
        self.assertEqual(report["status"], "degraded")
        self.assertEqual(report["applied"], 0)


class TestVocabulary(unittest.TestCase):
    def test_single_use_tags_are_not_vocabulary(self):
        findings = [_f(str(i), tags=["mqtt"]) for i in range(6)]
        findings += [_f("x", tags=["one-off-tag-nobody-reuses"])]
        vocab = groom.build_vocabulary(findings, min_uses=5)
        self.assertIn("mqtt", vocab)
        self.assertNotIn("one-off-tag-nobody-reuses", vocab)

    def test_run_provenance_stamps_are_excluded(self):
        findings = [_f(str(i), tags=["dt_run:whatever", "dt_model:haiku", "acoustic"])
                    for i in range(6)]
        vocab = groom.build_vocabulary(findings, min_uses=5)
        self.assertEqual(vocab, ["acoustic"])


class TestTagsPass(unittest.TestCase):
    def _run(self, gen, untagged=3):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._td.name)
        rows = [_f(f"t{i}", tags=["mqtt", "acoustic"]) for i in range(6)]
        rows += [_f(f"u{i}", text="the broker dropped the subscription", tags=[])
                 for i in range(untagged)]
        _corpus(tmp, {"inv": rows})
        groomdir = tmp / "_groom"
        report = groom.pass_tags(gen_fn=gen, memory_dir=tmp, groom_dir=groomdir)
        proposals = []
        p = groomdir / "proposals.jsonl"
        if p.exists():
            proposals = [json.loads(l) for l in open(p) if l.strip()]
        return report, proposals, tmp

    def tearDown(self):
        if getattr(self, "_td", None):
            self._td.cleanup()

    def test_it_proposes_vocabulary_tags_for_untagged_findings(self):
        gen = lambda prompts: [{"text": '{"tags": ["mqtt"]}', "ok": True} for _ in prompts]  # noqa: E731
        report, proposals, _ = self._run(gen)
        self.assertEqual(report["candidates"], 3)
        self.assertEqual(report["proposed"], 3)
        self.assertTrue(all(p["value"] == ["mqtt"] for p in proposals))
        self.assertTrue(all(p["proposed_by"] == "loci_groom/tags" for p in proposals))

    def test_it_never_touches_findings_jsonl(self):
        gen = lambda prompts: [{"text": '{"tags": ["mqtt"]}', "ok": True} for _ in prompts]  # noqa: E731
        _, _, tmp = self._run(gen)
        rows = [json.loads(l) for l in open(tmp / "inv" / "findings.jsonl") if l.strip()]
        untagged = [r for r in rows if not r.get("tags")]
        self.assertEqual(len(untagged), 3, "the pass must not write tags into the corpus")

    def test_a_tag_outside_the_vocabulary_is_dropped(self):
        gen = lambda prompts: [{"text": '{"tags": ["invented-label"]}', "ok": True} for _ in prompts]  # noqa: E731
        report, proposals, _ = self._run(gen)
        self.assertEqual(proposals, [])
        self.assertEqual(report["proposed"], 0)

    def test_a_failed_generation_proposes_nothing(self):
        gen = lambda prompts: [{"text": "", "ok": False} for _ in prompts]  # noqa: E731
        report, proposals, _ = self._run(gen)
        self.assertEqual(report["proposed"], 0)
        self.assertEqual(proposals, [])

    def test_non_json_output_is_skipped_not_raised(self):
        gen = lambda prompts: [{"text": "sure thing boss", "ok": True} for _ in prompts]  # noqa: E731
        report, proposals, _ = self._run(gen)
        self.assertEqual(report["proposed"], 0)

    def test_a_second_run_is_idempotent(self):
        gen = lambda prompts: [{"text": '{"tags": ["mqtt"]}', "ok": True} for _ in prompts]  # noqa: E731
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f(f"t{i}", tags=["mqtt", "acoustic"]) for i in range(6)]
                                 + [_f("u0", text="broker dropped it", tags=[])]})
            gd = tmp / "_groom"
            first = groom.pass_tags(gen_fn=gen, memory_dir=tmp, groom_dir=gd)
            second = groom.pass_tags(gen_fn=gen, memory_dir=tmp, groom_dir=gd)
            self.assertEqual(first["proposed"], 1)
            self.assertEqual(second["proposed"], 0)
            self.assertEqual(second["generated"], 1)   # regenerated, then deduped

    def test_no_generation_tier_degrades(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f(f"t{i}", tags=["mqtt"]) for i in range(6)]
                                 + [_f("u0", text="x", tags=[])]})
            with mock.patch.dict(sys.modules, {"batched_gen": None}):
                report = groom.pass_tags(memory_dir=tmp, groom_dir=tmp / "_groom")
        self.assertEqual(report["status"], "degraded")


class TestCli(unittest.TestCase):
    def test_apply_is_refused_for_proposal_only_passes(self):
        calls = {}

        def fake_tags(**kw):
            calls.update(kw)
            return {"pass": "tags", "status": "ok"}

        with mock.patch.dict(groom.PASSES, {"tags": {"fn": fake_tags, "applyable": False}}):
            groom.main(["tags", "--apply"])
        self.assertFalse(calls["apply"])

    def test_an_exploding_pass_does_not_take_the_run_down(self):
        def boom(**kw):
            raise RuntimeError("nope")

        with mock.patch.dict(groom.PASSES, {"tags": {"fn": boom, "applyable": False}}):
            rc = groom.main(["tags", "--json"])
        self.assertEqual(rc, 1)

    def test_unknown_pass_is_rejected(self):
        self.assertEqual(groom.main(["nonsense"]), 2)


if __name__ == "__main__":
    unittest.main()
