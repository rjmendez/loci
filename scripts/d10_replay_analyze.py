#!/usr/bin/env python3
"""Analyse hand labels from the D10 replay sheet, as pre-registered in docs/d10_replay_plan.md.

Inputs: the private key file written by ``d10_replay.py sample`` (which pair and which
gate category each opaque item id stands for, and the population of every category per
investigation) and a labels file: the JSON the HTML sheet exports, or the JSONL the
terminal labeller appends. Output: aggregates only (counts, rates, intervals, the
decision), never an id, a question or a finding. Stdlib only; reads two files, writes
at most ``--out``.

Estimators (the plan gives the reasoning):

* On a disagreement pair exactly one gate keeps the finding, so exactly one is right:
  the keeper if the label is relevant, the dropper if not. ``mlp_correct_share`` is the
  unweighted share of decisive labelled disagreements the MLP got right; the exact
  two-sided binomial sign test is on the same counts.
* Population rates use the stratified (post-stratified Horvitz-Thompson ratio)
  estimator. The strata are the four categories; each was sampled by simple random
  sampling, so a decisive label in stratum s stands for N_s / m_s pairs (N_s pairs in
  the replay, m_s decisive labels). Unsure labels are dropped, which assumes they are
  missing at random within their stratum. Grounded recall of a gate is (weighted
  relevant pairs it keeps) / (weighted relevant pairs); bleed rejection is (weighted
  not-relevant pairs it drops) / (weighted not-relevant pairs).
* Intervals: percentile bootstrap over investigations. Pairs within an investigation
  share a question set and a topic, so they are not independent; each replicate draws
  investigations with replacement and recomputes N_s and the labelled items from the
  drawn investigations. A replicate with N_s > 0 but no decisive label in s is
  undefined for the population rates; the count is reported.

Duplicated items (the sheet repeats a few under new ids) measure test-retest agreement
and are left out of every other number.

Usage: d10_replay_analyze.py --key KEY.json --labels LABELS.json|.jsonl [--out FILE] [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

LABELS = ("relevant", "not_relevant", "unsure")
CATEGORIES = ("cos_only", "mlp_only", "both_keep", "both_drop")
KEEPS = {"cosine": ("cos_only", "both_keep"), "mlp": ("mlp_only", "both_keep")}

# docs/d10_replay_plan.md, "Decision rule". Changing these needs a written reason there.
RULE = {"max_unsure_rate": 0.25, "min_mlp_correct": 0.60, "max_sign_p": 0.01,
        "min_mlp_recall": 0.95, "min_mlp_recall_lower": 0.90, "max_undefined_share": 0.05}


# --------------------------------------------------------------------------- inputs


def load_labels(path: Path) -> tuple[Optional[str], dict[str, str]]:
    """(sheet_id or None, {item id: label}). JSON export or JSONL; last row per id wins."""
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("labels"), dict):
        out = {}
        for k, v in obj["labels"].items():
            lab = v.get("label") if isinstance(v, dict) else v
            if lab in LABELS:
                out[str(k)] = lab
        return obj.get("sheet_id"), out
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") in LABELS:
            out[str(row["id"])] = row["label"]
        elif row.get("label") is None:
            out.pop(str(row.get("id")), None)
    return None, out


# --------------------------------------------------------------------------- statistics


def sign_test_p(k: int, n: int) -> Optional[float]:
    """Exact two-sided binomial test of k successes in n trials against p = 1/2."""
    if n <= 0:
        return None
    lo = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * lo)


def mlp_correct(category: str, label: str) -> bool:
    """On a disagreement: the MLP is right iff it kept a relevant or dropped a not-relevant finding."""
    return (category == "mlp_only") == (label == "relevant")


def stratified_rates(N: dict[str, float], rel: dict[str, int], dec: dict[str, int]) -> Optional[dict]:
    """Population estimates from stratum sizes N_s, relevant counts and decisive counts.

    None when a stratum with pairs has no decisive label (its rate is unknown).
    """
    p = {}
    for c in CATEGORIES:
        if N.get(c, 0) > 0:
            if dec.get(c, 0) == 0:
                return None
            p[c] = rel.get(c, 0) / dec[c]
        else:
            p[c] = 0.0
    R = {c: N.get(c, 0) * p[c] for c in CATEGORIES}          # estimated relevant pairs
    B = {c: N.get(c, 0) * (1 - p[c]) for c in CATEGORIES}    # estimated not-relevant pairs
    tot_r, tot_b = sum(R.values()), sum(B.values())
    out: dict[str, Optional[float]] = {}
    for g, keeps in KEEPS.items():
        out[f"{g}_grounded_recall"] = sum(R[c] for c in keeps) / tot_r if tot_r > 0 else None
        drops = [c for c in CATEGORIES if c not in keeps]
        out[f"{g}_bleed_rejection"] = sum(B[c] for c in drops) / tot_b if tot_b > 0 else None
    union = R["cos_only"] + R["mlp_only"] + R["both_keep"]
    for g, keeps in KEEPS.items():
        out[f"{g}_recall_of_either_kept"] = sum(R[c] for c in keeps) / union if union > 0 else None
    n_dis = N.get("cos_only", 0) + N.get("mlp_only", 0)
    out["mlp_correct_share_weighted"] = ((B["cos_only"] + R["mlp_only"]) / n_dis) if n_dis > 0 else None
    out["relevant_share"] = tot_r / (tot_r + tot_b) if tot_r + tot_b > 0 else None
    return out


def _counts(items: list[tuple[str, str, str]]) -> tuple[dict, dict]:
    rel, dec = Counter(), Counter()
    for _inv, cat, lab in items:
        if lab != "unsure":
            dec[cat] += 1
            rel[cat] += lab == "relevant"
    return rel, dec


def _pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def cluster_bootstrap(population: dict[str, dict[str, int]], items: list[tuple[str, str, str]],
                      reps: int, seed: int) -> dict:
    """Percentile 95 % intervals, resampling investigations with replacement."""
    rng = random.Random(seed)
    invs = sorted(population)
    by_inv: dict[str, list] = defaultdict(list)
    for it in items:
        by_inv[it[0]].append(it)
    draws: dict[str, list[float]] = defaultdict(list)
    undefined = 0
    for _ in range(reps):
        pick = [invs[rng.randrange(len(invs))] for _ in invs]
        N: Counter = Counter()
        sample: list = []
        for i in pick:
            N.update(population[i])
            sample.extend(by_inv.get(i, ()))
        rel, dec = _counts(sample)
        dis = [(c, lab) for _i, c, lab in sample if c in ("cos_only", "mlp_only") and lab != "unsure"]
        if dis:
            draws["mlp_correct_share"].append(sum(mlp_correct(c, lab) for c, lab in dis) / len(dis))
        est = stratified_rates(N, rel, dec)
        if est is None:
            undefined += 1
            continue
        for k, v in est.items():
            if v is not None:
                draws[k].append(v)
    ci = {k: [round(_pctl(v, 0.025), 4), round(_pctl(v, 0.975), 4)] for k, v in sorted(draws.items()) if v}
    return {"reps": reps, "seed": seed, "undefined_reps": undefined, "ci95": ci}


def kappa(pairs: list[tuple[str, str]]) -> Optional[float]:
    """Cohen's kappa of two labellings of the same items (here: one person, twice)."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return None if pe >= 1 else (po - pe) / (1 - pe)


# --------------------------------------------------------------------------- analysis


def analyse(key: dict, labels: dict[str, str], *, sheet_id: Optional[str] = None,
            reps: int = 2000, seed: int = 0) -> dict:
    if sheet_id is not None and sheet_id != key.get("sheet_id"):
        raise ValueError(f"labels belong to sheet {sheet_id}, the key to {key.get('sheet_id')}")
    items = key["items"]
    unknown = [i for i in labels if i not in items]
    if unknown:
        raise ValueError(f"{len(unknown)} labelled ids are not in the key")
    population = key["population"]
    N = Counter()
    for per in population.values():
        N.update(per)

    primary = [(v["investigation_id"], v["category"], labels[i], v["question_kind"])
               for i, v in items.items() if v.get("duplicate_of") is None and i in labels]
    n_primary = sum(1 for v in items.values() if v.get("duplicate_of") is None)
    triples = [(inv, c, lab) for inv, c, lab, _k in primary]
    rel, dec = _counts(triples)

    by_cat = {}
    for c in CATEGORIES:
        labs = Counter(lab for _i, cc, lab in triples if cc == c)
        by_cat[c] = {"population": N.get(c, 0), "labelled": sum(labs.values()),
                     "relevant": labs["relevant"], "not_relevant": labs["not_relevant"], "unsure": labs["unsure"]}

    dis_all = [(c, lab, k) for _i, c, lab, k in primary if c in ("cos_only", "mlp_only")]
    dis = [(c, lab, k) for c, lab, k in dis_all if lab != "unsure"]
    k_mlp = sum(mlp_correct(c, lab) for c, lab, _k in dis)
    n_dis = len(dis)
    unsure_dis = (len(dis_all) - n_dis) / len(dis_all) if dis_all else None
    by_kind = {}
    for kind in sorted({k for _c, _l, k in dis_all}):
        sub = [(c, lab) for c, lab, k in dis if k == kind]
        by_kind[kind] = {"decisive": len(sub), "mlp_correct": sum(mlp_correct(c, lab) for c, lab in sub)}

    est = stratified_rates(N, rel, dec)
    boot = cluster_bootstrap(population, triples, reps, seed)

    dups = [(labels[i], labels[v["duplicate_of"]]) for i, v in items.items()
            if v.get("duplicate_of") and i in labels and v["duplicate_of"] in labels]
    dec_dups = [(a, b) for a, b in dups if "unsure" not in (a, b)]
    ctrl = {c: (by_cat[c]["relevant"] / (by_cat[c]["relevant"] + by_cat[c]["not_relevant"])
                if by_cat[c]["relevant"] + by_cat[c]["not_relevant"] else None) for c in ("both_keep", "both_drop")}

    result = {
        "schema": "loci-d10-replay-analysis/v1",
        "sheet_id": key.get("sheet_id"),
        "plan_sha256": (key.get("plan") or {}).get("sha256"),
        "items_in_sheet": len(items),
        "primary_items": n_primary,
        "labelled_primary": len(primary),
        "unsure_rate": (sum(1 for t in triples if t[2] == "unsure") / len(triples)) if triples else None,
        "by_category": by_cat,
        "disagreements": {
            "labelled": len(dis_all), "decisive": n_dis, "unsure_rate": unsure_dis,
            "mlp_correct": k_mlp, "cosine_correct": n_dis - k_mlp,
            "mlp_correct_share": (k_mlp / n_dis) if n_dis else None,
            "sign_test_p_two_sided": sign_test_p(k_mlp, n_dis),
            "by_question_kind": by_kind,
        },
        "population_estimates": est,
        "bootstrap": boot,
        "consistency": {
            "duplicates_labelled": len(dups),
            "duplicates_agree": sum(a == b for a, b in dups),
            "duplicates_decisive": len(dec_dups),
            "duplicates_decisive_agree": sum(a == b for a, b in dec_dups),
            "kappa": kappa(dups),
            "control_relevant_share": ctrl,
            "controls_ordered": (ctrl["both_keep"] is not None and ctrl["both_drop"] is not None
                                 and ctrl["both_keep"] > ctrl["both_drop"]),
        },
    }
    c = result["consistency"]
    c["flag_unreliable"] = (not c["controls_ordered"]) or (c["kappa"] is not None and c["kappa"] < 0.4)
    result["decision"] = decide(result)
    return result


def decide(r: dict) -> dict:
    """The pre-registered rule. It can only ever say "candidate"; enforcement is the operator's call."""
    d = r["disagreements"]
    est = r["population_estimates"] or {}
    boot = r["bootstrap"]
    lower = (boot["ci95"].get("mlp_grounded_recall") or [None])[0]
    undefined_ok = boot["reps"] > 0 and boot["undefined_reps"] / boot["reps"] <= RULE["max_undefined_share"]
    checks = {
        "unsure_rate_below_0.25": d["unsure_rate"] is not None and d["unsure_rate"] < RULE["max_unsure_rate"],
        "mlp_correct_share_at_least_0.60": d["mlp_correct_share"] is not None and d["mlp_correct_share"] >= RULE["min_mlp_correct"],
        "mlp_correct_share_weighted_at_least_0.60": (est.get("mlp_correct_share_weighted") is not None
                                                     and est["mlp_correct_share_weighted"] >= RULE["min_mlp_correct"]),
        "sign_test_p_below_0.01": d["sign_test_p_two_sided"] is not None and d["sign_test_p_two_sided"] < RULE["max_sign_p"],
        "mlp_grounded_recall_at_least_0.95": (est.get("mlp_grounded_recall") is not None
                                              and est["mlp_grounded_recall"] >= RULE["min_mlp_recall"]),
        "mlp_recall_bootstrap_lower_above_0.90": lower is not None and lower > RULE["min_mlp_recall_lower"] and undefined_ok,
    }
    ok = all(checks.values())
    return {"checks": checks, "rule": RULE,
            "verdict": ("mlp_candidate_for_enforcement_pending_live_shadow" if ok else "cosine_remains_default"),
            "note": "This never authorises enforcement by itself; the operator decides after live shadow data agrees."}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    key = json.loads(args.key.read_text(encoding="utf-8"))
    sheet_id, labels = load_labels(args.labels)
    try:
        res = analyse(key, labels, sheet_id=sheet_id, reps=args.bootstrap, seed=args.seed)
    except ValueError as exc:
        print(f"d10_replay_analyze: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(res, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
