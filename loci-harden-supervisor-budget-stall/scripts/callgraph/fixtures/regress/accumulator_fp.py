"""REGRESSION FIXTURE -- the false-positive class that outranked BUG B.

Not copied from history; distilled from the three real shapes measured over
the corpus at rev d359e9a, where 16 of 23 `cg flags` rows were counters and
the one real bug sat 7th.

  n_drifted     -- mlops/embedding/drift.py::measure_drift  (pure `+=`)
  created       -- mcp/graph/linker.py::link_findings       (pure `+=`)
  lines_scanned -- mcp/server.py::_process_reflection_item  (reset THEN `+=`)

All three are correct code: a counter is SUPPOSED to be updated on only some
paths, so the guard-exit ratio scores it exactly like a conditionally-set
flag. `degraded` below is the genuine flag shape, kept in the same file so
the two are separated by classification and not by which file they live in.
"""


def measure_drift(rows, threshold):
    n_drifted = 0
    degraded = False
    out = []
    for r in rows:
        if not r:
            continue                 # guard exit, skips both
        try:
            score = float(r["score"])
        except Exception:
            degraded = True          # genuine guarded flag flip
            continue                 # guard exit
        if score > threshold:
            n_drifted += 1           # accumulate on SOME paths -- correct
        out.append(score)
    return {"n_drifted": n_drifted, "degraded": degraded, "scores": out}


def scan_log(path, max_lines):
    lines_scanned = 0
    kept = []
    with open(path) as fh:
        first = fh.readline()
        if not first:
            return {"lines_scanned": lines_scanned, "kept": kept}
        lines_scanned = 1            # a RESET to a constant, not a flag flip
        for line in fh:
            if not line.strip():
                continue             # guard exit
            if lines_scanned >= max_lines:
                break                # guard exit
            lines_scanned += 1
            kept.append(line)
    return {"lines_scanned": lines_scanned, "kept": kept}
