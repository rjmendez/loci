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
        fake_ops._retention_days.return_value = 0  # connect() gate
        # pass_index refuses unless retention resolves to 0 — connecting runs the
        # startup purge, which is the damage this pass exists to repair.
        fake_ops._retention_days.return_value = 0
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
            fake_ops._retention_days.return_value = 0
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


class TestRecallPass(unittest.TestCase):
    """The two probes must stay separable: identity is a wiring test, paraphrase
    is a semantic one, and collapsing them is what makes an outage look like a
    quality dip."""

    def _run(self, *, ranks_for_identity=None, ranks_for_paraphrase=None,
             gen=None, paraphrase=True, sample=3, k=5):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._td.name)
        ids = ["a", "b", "c"]
        _corpus(tmp, {"inv": [_f(i, text=f"finding text {i}") for i in ids]})

        def search(query):
            # identity queries carry the finding text; paraphrase queries do not
            is_identity = query.startswith("finding text ")
            table = ranks_for_identity if is_identity else ranks_for_paraphrase
            if table is None:
                return {"ok": False, "reason": "boom", "results": []}
            fid = query.split()[-1] if is_identity else query.split(":")[-1].strip()
            rank = table.get(fid)
            rows = [{"id": f"filler{n}"} for n in range(k)]
            if rank:
                rows[rank - 1] = {"id": fid}
            return {"ok": True, "reason": "hybrid", "results": rows}

        client = _Client(ids)
        fake_ops = mock.Mock()
        fake_ops._retention_days.return_value = 0  # connect() gate
        fake_ops._get_qdrant.return_value = (client, "hermes_memory")
        with mock.patch.dict(sys.modules, {"qdrant_ops": fake_ops}):
            return groom.pass_recall(
                sample=sample, k=k, paraphrase=paraphrase, gen_fn=gen,
                memory_dir=tmp, groom_dir=tmp / "_groom", search_fn=search,
            )

    def tearDown(self):
        if getattr(self, "_td", None):
            self._td.cleanup()

    def test_perfect_identity_recall(self):
        r = self._run(ranks_for_identity={"a": 1, "b": 1, "c": 1}, paraphrase=False)
        self.assertEqual(r["identity"]["recall_at_1"], 1.0)
        self.assertEqual(r["identity"]["mrr"], 1.0)
        self.assertEqual(r["identity"]["attempted"], 3)

    def test_a_finding_the_retriever_cannot_return_is_a_miss_not_an_error(self):
        r = self._run(ranks_for_identity={"a": 1, "b": None, "c": None}, paraphrase=False)
        self.assertAlmostEqual(r["identity"]["recall_at_1"], 1 / 3, places=3)
        self.assertAlmostEqual(r["identity"]["recall_at_5"], 1 / 3, places=3)
        self.assertEqual(r["search_errors"], 0)

    def test_rank_position_shows_up_in_mrr_not_just_recall_at_1(self):
        r = self._run(ranks_for_identity={"a": 1, "b": 2, "c": 4}, paraphrase=False)
        self.assertAlmostEqual(r["identity"]["recall_at_1"], 1 / 3, places=3)
        self.assertEqual(r["identity"]["recall_at_5"], 1.0)
        self.assertAlmostEqual(r["identity"]["mrr"], (1 + 0.5 + 0.25) / 3, places=4)

    def test_identity_and_paraphrase_are_scored_apart(self):
        gen = lambda prompts: [{"text": f"question: {p.split('finding text ')[1][0]}", "ok": True}  # noqa: E731
                               for p in prompts]
        r = self._run(ranks_for_identity={"a": 1, "b": 1, "c": 1},
                      ranks_for_paraphrase={"a": 1, "b": None, "c": None}, gen=gen)
        self.assertEqual(r["identity"]["recall_at_1"], 1.0)
        self.assertAlmostEqual(r["paraphrase"]["recall_at_1"], 1 / 3, places=3)

    def test_a_search_that_reports_not_ok_is_counted_as_an_error(self):
        r = self._run(ranks_for_identity=None, paraphrase=False)
        self.assertEqual(r["search_errors"], 3)
        self.assertEqual(r["identity"]["attempted"], 0)

    def test_an_empty_generated_question_is_unusable_not_a_miss(self):
        gen = lambda prompts: [{"text": "  ", "ok": True} for _ in prompts]  # noqa: E731
        r = self._run(ranks_for_identity={"a": 1, "b": 1, "c": 1},
                      ranks_for_paraphrase={}, gen=gen)
        self.assertEqual(r["paraphrase"]["unusable"], 3)
        self.assertEqual(r["paraphrase"]["attempted"], 0)

    def test_it_only_probes_findings_that_are_actually_indexed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f(i, text=f"finding text {i}") for i in ["a", "b", "c", "d"]]})
            client = _Client(["a", "b"])          # only two of the four are indexed
            fake_ops = mock.Mock()
            fake_ops._retention_days.return_value = 0  # connect() gate
            fake_ops._get_qdrant.return_value = (client, "hermes_memory")
            seen = []

            def search(q):
                seen.append(q)
                return {"ok": True, "reason": "hybrid", "results": []}

            with mock.patch.dict(sys.modules, {"qdrant_ops": fake_ops}):
                r = groom.pass_recall(sample=10, paraphrase=False, memory_dir=tmp,
                                      groom_dir=tmp / "_groom", search_fn=search)
        self.assertEqual(r["indexed_with_text"], 2)
        self.assertEqual(len(seen), 2)

    def test_it_appends_one_row_per_run_for_trend(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f("a", text="finding text a")]})
            client = _Client(["a"])
            fake_ops = mock.Mock()
            fake_ops._retention_days.return_value = 0  # connect() gate
            fake_ops._get_qdrant.return_value = (client, "hermes_memory")
            gd = tmp / "_groom"
            with mock.patch.dict(sys.modules, {"qdrant_ops": fake_ops}):
                for _ in range(2):
                    groom.pass_recall(sample=1, paraphrase=False, memory_dir=tmp,
                                      groom_dir=gd,
                                      search_fn=lambda q: {"ok": True, "results": [{"id": "a"}]})
            rows = [json.loads(l) for l in open(gd / "recall.jsonl") if l.strip()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["identity"]["recall_at_1"] == 1.0 for r in rows))

    def test_an_unreachable_qdrant_degrades(self):
        fake_ops = mock.Mock()
        fake_ops._retention_days.return_value = 0  # connect() gate
        fake_ops._get_qdrant.return_value = (None, None)
        with mock.patch.dict(sys.modules, {"qdrant_ops": fake_ops}):
            r = groom.pass_recall()
        self.assertEqual(r["status"], "degraded")


class TestKnnTags(unittest.TestCase):
    """Label transfer over the embeddings. Deterministic, so unlike the generated
    variant it can be scored against the author's own tags before promotion."""

    def _corpus_and_search(self, tmp, neighbours):
        rows = [_f(f"t{i}", tags=["mqtt", "acoustic"]) for i in range(6)]
        rows += [_f("u0", text="the broker dropped the subscription", tags=[])]
        _corpus(tmp, {"inv": rows})

        def search(_query):
            return {"ok": True, "results": neighbours}
        return search

    def test_a_close_neighbour_outweighs_several_loose_ones(self):
        votes = groom._knn_vote(
            [{"id": "n1", "score": 0.95, "tags": ["mqtt"]},
             {"id": "n2", "score": 0.2, "tags": ["acoustic"]},
             {"id": "n3", "score": 0.2, "tags": ["acoustic"]},
             {"id": "n4", "score": 0.2, "tags": ["acoustic"]}],
            self_id="self", vocab={"mqtt", "acoustic"}, min_weight=0.5)
        self.assertEqual(votes[0][0], "mqtt")

    def test_the_subject_never_votes_for_itself(self):
        votes = groom._knn_vote(
            [{"id": "self", "score": 1.0, "tags": ["mqtt"]}],
            self_id="self", vocab={"mqtt"}, min_weight=0.1)
        self.assertEqual(votes, [])

    def test_tags_outside_the_vocabulary_do_not_vote(self):
        votes = groom._knn_vote(
            [{"id": "n1", "score": 0.9, "tags": ["not-in-vocab"]}],
            self_id="s", vocab={"mqtt"}, min_weight=0.1)
        self.assertEqual(votes, [])

    def test_comma_joined_payload_tags_are_understood(self):
        # one write path stores tags as ",".join(tags)
        self.assertEqual(groom._tags_of({"tags": "mqtt,acoustic"}), ["mqtt", "acoustic"])
        self.assertEqual(groom._tags_of({"tags": ["MQTT", "dt_run:x"]}), ["mqtt"])

    def test_it_proposes_for_untagged_findings_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            search = self._corpus_and_search(
                tmp, [{"id": "t1", "score": 0.9, "tags": ["mqtt"]},
                      {"id": "t2", "score": 0.8, "tags": ["mqtt"]}])
            gd = tmp / "_groom"
            report = groom.pass_knn_tags(memory_dir=tmp, groom_dir=gd,
                                         search_fn=search, min_weight=1.0)
            rows = [json.loads(l) for l in open(gd / "proposals.jsonl") if l.strip()]
        self.assertEqual(report["candidates"], 1)
        self.assertEqual(report["proposed"], 1)
        self.assertEqual(rows[0]["subject_id"], "u0")
        self.assertEqual(rows[0]["value"], ["mqtt"])
        self.assertIsNone(rows[0]["model"], "no model was involved")

    def test_weak_agreement_proposes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            search = self._corpus_and_search(
                tmp, [{"id": "t1", "score": 0.1, "tags": ["mqtt"]}])
            report = groom.pass_knn_tags(memory_dir=tmp, groom_dir=tmp / "_groom",
                                         search_fn=search, min_weight=1.0)
        self.assertEqual(report["proposed"], 0)

    def test_calibrate_scores_against_the_authors_own_tags(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            self._corpus_and_search(tmp, [])
            # every neighbour says "mqtt"; the held-out findings really are mqtt+acoustic
            search = lambda _q: {"ok": True, "results": [  # noqa: E731
                {"id": "other", "score": 0.9, "tags": ["mqtt"]}]}
            report = groom.pass_knn_tags(memory_dir=tmp, groom_dir=tmp / "_groom",
                                         search_fn=search, calibrate=True, min_weight=0.5)
        self.assertEqual(report["checked"], 6)
        self.assertEqual(report["mean_precision"], 1.0)
        self.assertEqual(report["exact_or_partial_hit_rate"], 1.0)

    def test_calibrate_reports_a_miss_as_zero_precision(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            self._corpus_and_search(tmp, [])
            search = lambda _q: {"ok": True, "results": [  # noqa: E731
                {"id": "other", "score": 0.9, "tags": ["build"]}]}
            rows = [_f(f"b{i}", tags=["build"]) for i in range(6)]
            _corpus(tmp, {"inv2": rows})
            report = groom.pass_knn_tags(memory_dir=tmp, groom_dir=tmp / "_groom",
                                         search_fn=search, calibrate=True, min_weight=0.5)
        self.assertGreater(report["checked"], 0)

    def test_calibrate_writes_no_proposals(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            self._corpus_and_search(tmp, [])
            search = lambda _q: {"ok": True, "results": [  # noqa: E731
                {"id": "other", "score": 0.9, "tags": ["mqtt"]}]}
            gd = tmp / "_groom"
            groom.pass_knn_tags(memory_dir=tmp, groom_dir=gd, search_fn=search,
                                calibrate=True, min_weight=0.5)
            self.assertFalse((gd / "proposals.jsonl").exists())


class TestCodelink(unittest.TestCase):
    """Finding -> CodeSymbol proposals. A wrong code link is worse than none —
    it survives as provenance — so the unique/ambiguous split is the contract."""

    SYMBOLS = [
        ("_qdrant_upsert", "sym1", "mcp/qdrant_ops.py"),
        ("parse_frame", "sym2", "a2a_server/server.py"),
        ("parse_frame", "sym3", "mcp/memcheck/daemon.py"),
        ("EskfFusion", "sym4", "android/EskfFusion.java"),
    ]

    def _run(self, text, *, gen=None, calibrate=False, linked=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f("f1", text=text)]})
            gd = tmp / "_groom"
            report = groom.pass_codelink(
                memory_dir=tmp, groom_dir=gd, gen_fn=gen, calibrate=calibrate,
                symbols_fn=lambda: list(self.SYMBOLS),
                linked_fn=lambda: list(linked or []),
            )
            rows = []
            if (gd / "proposals.jsonl").exists():
                rows = [json.loads(l) for l in open(gd / "proposals.jsonl") if l.strip()]
            return report, rows

    def test_a_token_naming_exactly_one_symbol_is_evidence(self):
        report, rows = self._run("the _qdrant_upsert path swallows the failure")
        self.assertEqual(report["proposed"], 1)
        self.assertEqual(rows[0]["value"][0]["symbol_id"], "sym1")
        self.assertEqual(rows[0]["value"][0]["file"], "mcp/qdrant_ops.py")
        self.assertIsNone(rows[0]["model"], "no model needed for an unambiguous token")

    def test_an_ambiguous_token_is_not_guessed_at_without_a_model(self):
        report, rows = self._run("the parse_frame path is wrong")
        self.assertEqual(report["ambiguous_tokens"], 1)
        self.assertEqual(report["proposed"], 0)

    def test_the_model_resolves_an_ambiguous_token(self):
        gen = lambda prompts: [{"text": "2", "ok": True} for _ in prompts]  # noqa: E731
        report, rows = self._run("the parse_frame path is wrong", gen=gen)
        self.assertEqual(report["proposed"], 1)
        self.assertEqual(rows[0]["value"][0]["symbol_id"], "sym3")

    def test_a_model_answer_of_zero_declines_rather_than_linking(self):
        gen = lambda prompts: [{"text": "0", "ok": True} for _ in prompts]  # noqa: E731
        report, _ = self._run("the parse_frame path is wrong", gen=gen)
        self.assertEqual(report["proposed"], 0)

    def test_an_out_of_range_choice_is_refused(self):
        gen = lambda prompts: [{"text": "99", "ok": True} for _ in prompts]  # noqa: E731
        report, _ = self._run("the parse_frame path is wrong", gen=gen)
        self.assertEqual(report["proposed"], 0)

    def test_prose_words_do_not_become_symbols(self):
        report, _ = self._run("there should always be a result under these values")
        self.assertEqual(report["proposed"], 0)
        self.assertEqual(report["ambiguous_tokens"], 0)

    def test_already_linked_findings_are_skipped(self):
        report, _ = self._run("the _qdrant_upsert path", linked=[("f1", "sym1")])
        self.assertEqual(report["candidates"], 0)
        self.assertEqual(report["proposed"], 0)

    def test_calibrate_scores_against_the_existing_edges(self):
        report, rows = self._run("the _qdrant_upsert path", calibrate=True,
                                 linked=[("f1", "sym1")])
        self.assertEqual(report["checked"], 1)
        self.assertEqual(report["mean_precision"], 1.0)
        self.assertEqual(report["any_correct_rate"], 1.0)
        self.assertEqual(rows, [], "calibration writes no proposals")

    def test_calibrate_counts_a_wrong_link_as_zero(self):
        report, _ = self._run("the EskfFusion update", calibrate=True,
                              linked=[("f1", "sym1")])
        self.assertEqual(report["checked"], 1)
        self.assertEqual(report["mean_precision"], 0.0)

    def test_an_unreadable_graph_degrades(self):
        def boom():
            raise RuntimeError("locked")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            _corpus(tmp, {"inv": [_f("f1", text="x")]})
            report = groom.pass_codelink(memory_dir=tmp, groom_dir=tmp / "_groom",
                                         symbols_fn=boom, linked_fn=lambda: [])
        self.assertEqual(report["status"], "degraded")


class TestDistinctiveness(unittest.TestCase):
    """A bare lowercase word that happens to name a symbol is not evidence that
    the author meant the code. The first live calibration linked `device`, `roll`
    and `train` to real symbols on exactly that mistake."""

    def test_bare_words_are_rejected(self):
        for tok in ("device", "position", "confirmed", "roll", "train", "report",
                    "baseline", "remaining", "consumer", "trigger", "refusal"):
            self.assertFalse(groom._is_distinctive(tok), tok)

    def test_structured_identifiers_are_accepted(self):
        for tok in ("_qdrant_upsert", "disarmBeam", "SensorCollectionService",
                    "schroederRt60Ms", "estimateRt60ApproxMs", "parse_frame"):
            self.assertTrue(groom._is_distinctive(tok), tok)

    def test_a_long_lowercase_identifier_still_counts(self):
        self.assertTrue(groom._is_distinctive("triangulationsolver"))
        self.assertFalse(groom._is_distinctive("triangulate"))
