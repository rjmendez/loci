#!/usr/bin/env python3
"""Measure the adversarial skeptic's verdicts against a fixed, self-contained set.

Run:  ./mcp/.venv/bin/python eval/verify_skeptic_eval.py [trials] [votes]

WHY THESE CASES. An earlier probe of this same skeptic was invalidated because
its labels asserted things about the loci corpus that the corpus then changed —
past-tense-true labels against a present-tense codebase. Nothing here touches
loci: every case carries its own context, so no commit can falsify a label.

WHY THE SCORE IS ASYMMETRIC. Overall accuracy is the wrong objective. A verdict
is an advisory note and never changes a finding's lifecycle, so "uncertain" is
harmless — it leaves a finding unverified. "Refuted" on a claim that is actually
true is the damage, because it is what a later reader acts on. FALSE REFUTATION
is therefore the headline number; missed refutations are reported but mild.
"""
from __future__ import annotations

import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))

# (expected_verdict, claim, context)
CASES = [
    # CONFIRMED — the context positively supports the claim.
    ("confirmed", "The function add returns the sum of its two arguments.",
     "def add(a, b):\n    return a + b"),
    ("confirmed", "TIMEOUT is set to 30.", "TIMEOUT = 30"),
    ("confirmed", "48000 divided by 960 equals 50.", ""),
    ("confirmed", "The loop body runs exactly 3 times.",
     "for i in range(3):\n    print(i)"),
    ("confirmed", "connect() raises ValueError when url is empty.",
     'def connect(url):\n    if not url:\n        raise ValueError("url required")\n'
     "    return open_socket(url)"),
    # REFUTED — the context positively contradicts the claim.
    ("refuted", "The function add multiplies its two arguments.",
     "def add(a, b):\n    return a + b"),
    ("refuted", "TIMEOUT is set to 60.", "TIMEOUT = 30"),
    ("refuted", "48000 divided by 960 equals 60.", ""),
    ("refuted", "The loop body runs 10 times.", "for i in range(3):\n    print(i)"),
    ("refuted", "connect() returns None when url is empty.",
     'def connect(url):\n    if not url:\n        raise ValueError("url required")\n'
     "    return open_socket(url)"),
    # UNCERTAIN — nothing here decides it either way.
    ("uncertain", "The staging deployment completed at 14:02 UTC.", ""),
    ("uncertain", "The retry_count setting defaults to 5.", ""),
    ("uncertain", "The battery monitor is an INA226 at I2C address 0x40.", ""),
    ("uncertain", "The nightly job takes about 20 minutes to finish.", ""),
    ("uncertain", "The team decided to postpone the migration to Q3.", ""),
]


def run(trials: int, votes: int):
    from verify import verify_finding
    per = collections.defaultdict(collections.Counter)
    rows = []
    for want, claim, ctx in CASES:
        got = []
        for _ in range(trials):
            # votes= is only passed when asked for, so this harness can also
            # benchmark a build that predates consensus voting.
            kw = {"votes": votes} if votes > 1 else {}
            got.append(verify_finding(claim, context=ctx, **kw)
                       .get("verdict", "uncertain"))
            per[want][got[-1]] += 1
        rows.append((want, claim, got))
    return per, rows


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    votes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    per, rows = run(trials, votes)
    print(f"\n  {len(CASES)} cases x {trials} trials, votes={votes}, "
          f"{time.time() - t0:.0f}s\n")
    print(f"  {'expected':<11}{'confirmed':>10}{'refuted':>9}{'uncertain':>11}   accuracy")
    total = correct = 0
    for want in ("confirmed", "refuted", "uncertain"):
        c = per[want]
        n = sum(c.values())
        total += n
        correct += c[want]
        print(f"  {want:<11}{c['confirmed']:>10}{c['refuted']:>9}"
              f"{c['uncertain']:>11}   {c[want]}/{n}")

    harmful = per["confirmed"]["refuted"] + per["uncertain"]["refuted"]
    non_ref = sum(sum(per[w].values()) for w in ("confirmed", "uncertain"))
    missed = sum(per["refuted"].values()) - per["refuted"]["refuted"]
    print(f"\n  overall: {correct}/{total} = {correct * 100 // max(total, 1)}%")
    print(f"  FALSE REFUTATION (the harm): {harmful}/{non_ref} = "
          f"{harmful * 100 // max(non_ref, 1)}%")
    print(f"  missed refutations (mild)  : {missed}/{sum(per['refuted'].values())}")
    print("\n  --- per case ---")
    for want, claim, got in rows:
        flag = "  " if all(g == want for g in got) else "!!"
        print(f"  {flag} want={want:<10} got={','.join(got):<30} {claim[:52]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
