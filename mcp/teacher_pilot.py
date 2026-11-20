"""Adapters that chain the teacher components for a live pilot.

``gold_labels`` writes ``query`` and ``label`` rows (query id, candidate id, grade 0..3, per-labeler grades).
The other components want other shapes; this module converts, nothing more. Pure functions take injected
text/judge/teacher callables; the CLI wires the live ones (Qdrant payload text, local judge, ``teacher_call``).

    python mcp/teacher_pilot.py distill-rows  --gold G --out rows.jsonl
    python mcp/teacher_pilot.py calibration   --gold G --judge-out judge.jsonl --gold-out gold.jsonl [--limit N]
    python mcp/teacher_pilot.py query-expand  --gold G --max-usd X [--limit N]

Relevance threshold: a candidate is "relevant" at grade >= ``THRESHOLD`` (2).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

THRESHOLD = 2
EVIDENCE_CHARS = 1500
COLLECTION = "loci_memory"

TextFn = Callable[[Sequence[str]], Dict[str, str]]


class PilotError(ValueError):
    """Unusable input."""


def read_gold(path: str) -> Tuple[Dict[str, str], List[dict]]:
    """({query_id: query text}, label rows). Rows that are not valid JSON objects are skipped."""
    queries: Dict[str, str] = {}
    labels: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                if r.get("type") == "query" and r.get("query_id") and r.get("query"):
                    queries[str(r["query_id"])] = str(r["query"])
                elif r.get("type") == "label" and r.get("query_id") and r.get("candidate_id") \
                        and isinstance(r.get("grade"), int) and not isinstance(r.get("grade"), bool):
                    labels.append(r)
    except OSError as e:
        raise PilotError(f"{path}: cannot read ({e.strerror or e})") from e
    return queries, labels


def pair_id(query_id: str, candidate_id: str) -> str:
    return f"{query_id}:{candidate_id}"


def _labeler_grade(label: Mapping, role: str) -> Optional[int]:
    v = (label.get("labels") or {}).get(role)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


# ------------------------------------------------------------------ distill
def to_distill_rows(queries: Mapping[str, str], labels: Sequence[dict], texts: Mapping[str, str],
                    teacher_role: str = "label_a") -> List[dict]:
    """``{id, query_id, claim, evidence, label, teacher_label}`` per labeled pair whose text is known.

    claim = the query, evidence = the candidate text (truncated), label = gold grade >= THRESHOLD,
    teacher_label = the single ``teacher_role`` labeler's verdict (absent when that labeler has no grade)."""
    out = []
    for lb in labels:
        q, c = str(lb["query_id"]), str(lb["candidate_id"])
        text = texts.get(c)
        if not text or q not in queries:
            continue
        row = {"id": pair_id(q, c), "query_id": q, "claim": queries[q], "evidence": text[:EVIDENCE_CHARS],
               "label": 1 if lb["grade"] >= THRESHOLD else 0}
        tg = _labeler_grade(lb, teacher_role)
        if tg is not None:
            row["teacher_label"] = 1 if tg >= THRESHOLD else 0
        out.append(row)
    return out


# ------------------------------------------------------------------ judge calibration
def relevance_prompt(query: str, text: str) -> str:
    return ("You are grading search results. Decide whether the document is relevant to the query "
            "(it helps answer it or is clearly about its subject).\n"
            f"Query:\n<<<\n{query[:600]}\n>>>\nDocument:\n<<<\n{text[:EVIDENCE_CHARS]}\n>>>\n"
            'Reply with JSON only: {"correct": true} or {"correct": false}.')


def parse_verdict(text: str) -> Optional[bool]:
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get("correct")
    except (ValueError, AttributeError):
        return None
    return v if isinstance(v, bool) else None


def to_calibration(queries: Mapping[str, str], labels: Sequence[dict], texts: Mapping[str, str],
                   judge: Callable[[str], str], model: str = "", limit: Optional[int] = None
                   ) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """(judge rows, gold rows, counts). ``judge`` maps a prompt to raw model text. Unparseable verdicts are
    counted, not guessed; those pairs appear in neither output file."""
    judge_rows, gold_rows = [], []
    counts = {"pairs": 0, "judged": 0, "no_text": 0, "unparseable": 0}
    for lb in labels:
        if limit is not None and counts["pairs"] >= limit:
            break
        q, c = str(lb["query_id"]), str(lb["candidate_id"])
        counts["pairs"] += 1
        text = texts.get(c)
        if not text or q not in queries:
            counts["no_text"] += 1
            continue
        verdict = parse_verdict(judge(relevance_prompt(queries[q], text)))
        if verdict is None:
            counts["unparseable"] += 1
            continue
        iid = pair_id(q, c)
        judge_rows.append({"id": iid, "verdict": verdict, "model": model})
        gold_rows.append({"id": iid, "grade": lb["grade"]})
        counts["judged"] += 1
    return judge_rows, gold_rows, counts


# ------------------------------------------------------------------ query_expand gold
def expect_prompt(query: str, text: str) -> str:
    return ("A search query must be expanded so it reaches the document below. List 3 to 5 key terms "
            "(single words or short names that literally appear in the document) that an expanded query "
            "should contain.\n"
            f"Query:\n<<<\n{query[:600]}\n>>>\nDocument:\n<<<\n{text[:EVIDENCE_CHARS]}\n>>>\n"
            'Reply with JSON only: {"terms": ["..."]}')


def parse_terms(text: str) -> List[str]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        terms = json.loads(m.group(0)).get("terms")
    except (ValueError, AttributeError):
        return []
    if not isinstance(terms, list):
        return []
    seen, out = set(), []
    for t in terms:
        t = str(t).strip()
        if t and t.lower() not in seen and 1 < len(t) <= 40:
            seen.add(t.lower())
            out.append(t)
    return out[:5]


def to_query_expand_gold(queries: Mapping[str, str], labels: Sequence[dict], texts: Mapping[str, str],
                         ask: Callable[[str], Optional[str]], limit: Optional[int] = None) -> Tuple[List[dict], Dict[str, int]]:
    """One ``{id, query, expect}`` row per query that has a relevant (grade >= 2) candidate with text; the
    top-graded candidate is the target. Terms must literally occur in that document (case-insensitive) and
    not already be in the query. ``ask`` returns None when the teacher refused (stops the run)."""
    best: Dict[str, dict] = {}
    for lb in labels:
        q = str(lb["query_id"])
        if lb["grade"] >= THRESHOLD and texts.get(str(lb["candidate_id"])) \
                and (q not in best or lb["grade"] > best[q]["grade"]):
            best[q] = lb
    rows, counts = [], {"queries": len(best), "kept": 0, "no_terms": 0, "refused": 0}
    for q, lb in sorted(best.items()):
        if limit is not None and counts["kept"] + counts["no_terms"] >= limit:
            break
        raw = ask(expect_prompt(queries[q], texts[str(lb["candidate_id"])]))
        if raw is None:
            counts["refused"] += 1
            break
        doc, qlow = texts[str(lb["candidate_id"])].lower(), queries[q].lower()
        terms = [t for t in parse_terms(raw) if t.lower() in doc and t.lower() not in qlow]
        if not terms:
            counts["no_terms"] += 1
            continue
        rows.append({"id": q, "query": queries[q], "expect": terms})
        counts["kept"] += 1
    return rows, counts


# ------------------------------------------------------------------ live wiring
def qdrant_texts(ids: Sequence[str], collection: str = COLLECTION) -> Dict[str, str]:
    """{point id: payload text} for the ids found."""
    import server  # noqa: F401 - binds the backends config
    import qdrant_ops
    client = qdrant_ops._qdrant_client_readonly()[0]
    out: Dict[str, str] = {}
    uniq = sorted(set(ids))
    for i in range(0, len(uniq), 64):
        for p in client.retrieve(collection, ids=uniq[i:i + 64], with_payload=True):
            t = (p.payload or {}).get("text")
            if isinstance(t, str) and t:
                out[str(p.id)] = t
    return out


def _write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def _local_judge() -> Tuple[Callable[[str], str], str]:
    """The local judge model straight through the resident Ollama gen endpoint (``backends.ollama_gen_url``).
    ``llm_local.generate`` is not used: it refuses the judge task unless a tmux offload session is running."""
    import urllib.request
    import backends
    import model_pool
    model = os.environ.get("LOCI_PILOT_JUDGE_MODEL") or model_pool.rank_role("verify").candidates[0].name
    url = backends.ollama_gen_url().rstrip("/") + "/api/generate"

    def call(prompt: str) -> str:
        body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": 0, "num_predict": 24}}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return str(json.loads(resp.read().decode("utf-8")).get("response") or "")
        except (OSError, ValueError):
            return ""
    return call, model


def _teacher_ask(max_usd: Optional[float]) -> Callable[[str], Optional[str]]:
    import teacher
    spent = {"usd": 0.0, "max": 0.0}

    def ask(prompt: str) -> Optional[str]:
        if max_usd is not None and spent["usd"] + spent["max"] > max_usd:
            return None
        r = teacher.teacher_call("label_a", prompt, 120, "query_expand_gold")
        usd = float(r.get("usd") or 0.0)
        spent["usd"] += usd
        spent["max"] = max(spent["max"], usd)
        if not r.get("ok"):
            return None if str(r.get("reason", "")).startswith(("over_", "disabled", "no_", "sanitizer")) else ""
        return str(r.get("text") or "")
    return ask


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="teacher_pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("distill-rows", "calibration", "query-expand"):
        p = sub.add_parser(name)
        p.add_argument("--gold", required=True)
        p.add_argument("--limit", type=int)
    sub.choices["distill-rows"].add_argument("--out", required=True)
    sub.choices["calibration"].add_argument("--judge-out", required=True)
    sub.choices["calibration"].add_argument("--gold-out", required=True)
    sub.choices["query-expand"].add_argument("--max-usd", type=float)
    sub.choices["query-expand"].add_argument("--out", help="default: <LOCI_TEACHER_GOLD_DIR or ~/.loci/teacher/gold>/query_expand.jsonl")
    a = ap.parse_args(argv)
    try:
        queries, labels = read_gold(a.gold)
        texts = qdrant_texts([str(lb["candidate_id"]) for lb in labels])
        if a.cmd == "distill-rows":
            n = _write_jsonl(a.out, to_distill_rows(queries, labels, texts))
            print(json.dumps({"rows": n, "labels": len(labels), "texts": len(texts)}))
            return 0
        if a.cmd == "calibration":
            judge, model = _local_judge()
            jr, gr, counts = to_calibration(queries, labels, texts, judge, model, a.limit)
            _write_jsonl(a.judge_out, jr)
            _write_jsonl(a.gold_out, gr)
            print(json.dumps({**counts, "model": model}))
            return 0
        rows, counts = to_query_expand_gold(queries, labels, texts, _teacher_ask(a.max_usd), a.limit)
        base = os.environ.get("LOCI_TEACHER_GOLD_DIR") or str(Path("~/.loci/teacher/gold").expanduser())
        out = a.out or str(Path(base) / "query_expand.jsonl")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out, rows)
        print(json.dumps({**counts, "out": out}))
        return 0 if counts["refused"] == 0 else 3
    except PilotError as e:
        print(f"teacher_pilot: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
