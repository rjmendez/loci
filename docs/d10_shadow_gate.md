# D10 evidence relevance gate — shadow mode

`investigation_reason` decides which findings a model reasons over with a cosine rule:
a finding is kept when the cosine of its `nomic-embed-text` embedding to the question is
at least `ground_threshold` (0.59), and the 12 highest kept findings go into the prompt.
This change adds a second gate, a small pair-interaction MLP, **in shadow mode only**: it
scores the same pairs, logs both gates' decisions side by side, and changes nothing the
model sees. It is off unless the operator turns it on.

| Piece | Where |
|---|---|
| Features, numpy inference, artifact loader, shadow logger | `mcp/d10_gate.py` |
| Hook (one call after the cosine gate) | `mcp/server.py`, `_d10_shadow_gate` |
| Trained artifact (no pickle) | `mcp/models/d10_pair_gate/weights.npz`, `manifest.json` |
| Trainer / exporter | `deep_think_loci/grounding/train_d10_pair_gate.py` |
| Read-only analysis | `scripts/d10_shadow_report.py` |
| Tests | `mcp/tests/test_d10_gate.py`, `scripts/tests/test_d10_shadow_report.py` |

## Why this model, and why only in shadow

The model was chosen in a pre-registered offline comparison (FlyBrain D10, private repo
`rjmendez/flybrain`, PR #21, `docs/tasks/loci_d10.md` §3) on the frozen dataset
`deep_think_loci/grounding/grounding_dataset.jsonl` at Loci commit `113efecb`: 5,418
labelled pairs over 143 findings, grouped into 4 recovered deep-think runs, evaluated
leave-one-run-out on 1,811 topical test pairs. M1 is bleed rejection (share of off-topic
pairs dropped) at grounded recall ≥ 0.95; intervals are 95% run-level bootstrap.

| Gate | M1 | M1 95% CI | F1 | Params | p95 / pair | Debiased ECE |
|---|---:|---|---:|---:|---:|---:|
| cosine rule (current) | 0.949 | [0.49, 0.985] | 0.665 | 0 | — | 0.060 |
| **pair MLP** | **0.987** | [0.84, 1.00] | 0.943 | 203,541 | 582 µs | 0.033 |

Paired M1 gain of the MLP over the cosine rule: **+0.038 [0.012, 0.377]**.

The MLP: features `[u, v, |u−v|, u·v]` of the unit-normalised embeddings (u = claim /
question, v = evidence / finding) plus ten tabular columns (cos, cos², whitespace-token
Jaccard, length ratio, log min/max length, negation any/xor, log shared / only-one-side
number counts) → StandardScaler → 64 ReLU units → logistic output (sklearn
`MLPClassifier`, alpha 1e-3, adam, 200 iterations, seed 0). The trainer reproduces the
offline per-run M1 exactly (MLP 1.000 / 0.995 / 0.829 / 1.000, cosine 0.980 / 0.979 /
0.632 / 0.231; pooled 0.987 and 0.949), and those numbers are in the manifest.

**Caveats. These are why it ships in shadow.**

- **The labels are structural proxies.** A topical pair is "grounded" when both findings
  carry the same `dt_target` tag and "bleed" when they do not. Nobody judged whether a
  piece of evidence actually grounds a claim.
- **Four runs.** Only four deep-think runs could be recovered, two of them small (23
  and 13 findings). The run-level interval is wide; its lower end for the cosine rule is
  0.49, and most of the spread between gates comes from the two small runs.
- **The models learned target-tag agreement.** Lineage recall (pairs where one finding
  was derived from the other) is only 0.42–0.50 for the strong models: when a derived
  finding crosses targets, they call it bleed.
- **Different pair type.** Training pairs are finding–finding; the live gate scores
  question–finding pairs. Whether the model transfers is exactly what shadow mode
  measures.
- At the live threshold (0.59) the cosine rule, on the same out-of-fold finding pairs,
  kept 98.5% of grounded pairs but dropped only 49.7% of bleed; the MLP at its chosen
  threshold kept 95.1% and dropped 97.7%. That is the gap shadow mode has to confirm or
  refute on real questions.

## The artifact

`manifest.json` records the schema, the weights' sha256, the training data (path, Loci
commit, sha256, row/finding/run counts), the embedder (model, normalisation,
truncation, digest, sha256 of the float32 embedding matrix), the recipe with sklearn
and numpy versions, the operating threshold, a threshold table, the reproduced
leave-one-run-out metrics, the offline results quoted above with their caveats, and 16
sklearn probabilities on a fixed, data-free check batch.

- **Threshold.** `tau_R95` = 0.2144, the largest threshold whose pooled
  leave-one-run-out out-of-fold grounded recall is ≥ 0.95 (as pre-registered). The
  table in the manifest gives the thresholds for recall floors 0.90, 0.95, 0.975, 0.99.
- **Integrity.** `load_gate` refuses unless sha256(`manifest.json`) equals
  `PINNED_MANIFEST_SHA256` in `d10_gate.py` and sha256(`weights.npz`) equals the
  manifest's. Weights load with `np.load(allow_pickle=False)`; shapes and finiteness are
  checked. Inference is numpy only; sklearn is needed to train, not to serve.
- **Hashes.** manifest `7f2c4c4ab20d57953c59d1811ece3af42eaaf7eb4977483517c68c1f1c82f2e4`,
  weights `61fd567fa06599309547d8a81087ab76bcac2f18a5d1ec666e930065d0bd284a`
  (1.56 MB, float64), training data
  `2da3b224ba4ca7682d0c9813fdad890ac0812ca8e010e7b16590d28135581676`.

Retrain (deterministic: the same inputs give the same weights sha256):

```bash
cd deep_think_loci/grounding
python3 train_d10_pair_gate.py --loci-commit "$(git rev-parse HEAD)" --embed-url http://127.0.0.1:11434
# or, with a precomputed embedding cache of the same sorted text list:
python3 train_d10_pair_gate.py --loci-commit <sha> --texts-json texts.json --embeddings-npy embeddings.npy
```

then set `PINNED_MANIFEST_SHA256` to the printed manifest sha256 in the same commit.
The committed artifact was trained from the FlyBrain D10 embedding cache (the same
`nomic-embed-text` vectors the offline comparison used), passed with `--texts-json` /
`--embeddings-npy`; the trainer checks that its text list equals the dataset's.

## What shadow mode does

With `LOCI_D10_SHADOW=1`, after the cosine gate has chosen its findings,
`investigation_reason` calls `_d10_shadow_gate`, which:

1. reuses the embeddings the cosine gate already fetched (question first, then one per
   finding): **no extra embedding call**;
2. builds the features, scores every (question, finding) pair and applies `tau_R95`;
3. appends one JSON row per pair to `<data home>/instrumentation/d10_shadow.jsonl`
   (beside `memory_use.jsonl`; with the default memory dir that is
   `~/.loci/instrumentation/d10_shadow.jsonl`) through `instrumentation_log.append_rows`:
   the same 4 MiB rotation with 3 generations (`LOCI_INSTRUMENTATION_LOG_MAX_BYTES`,
   `LOCI_INSTRUMENTATION_LOG_KEEP`) and the same non-blocking lock (a contended write is
   dropped).

Its return value is ignored. It runs after `gated` is final and only reads it. Any
exception, whether a tampered artifact, a shape mismatch or a failed write, is swallowed,
logged at debug level and counted (`d10_gate.shadow_error_count()`, also written into
each row as `shadow_errors_total`). A failure outside `record_shadow` (for example the
import) is caught by the server wrapper.

Cost: loading the artifact once per process takes about 20 ms. Scoring, features
included, on one CPU thread measured 50–130 µs per pair at the median and 70–250 µs at
p95, for calls of 1 to 200 findings (measured 2026-09-26 on mrpink; logging time not
included). This is small next to the pure-Python cosines the live gate already computes
and the N+1 LLM calls that follow.

Rows are capped at `LOCI_D10_SHADOW_MAX_PAIRS` (default 512) per call, highest cosine
first; `n_pairs` and `n_logged` say when a call was truncated.

### Log row

Ids, scores, enums and counts only. There is no finding text, question text or hash of
either.

| Field | Meaning |
|---|---|
| `schema`, `event`, `ts` | `1`, `"d10_shadow"`, UTC ISO time |
| `call_id` | one id per `investigation_reason` call (random, 16 hex) |
| `investigation_id`, `finding_id` | the pair |
| `n_pairs`, `n_logged` | findings scored in this call / rows written |
| `cos`, `cos_threshold`, `cos_keep`, `cos_rank` | live cosine, its threshold, `cos >= threshold`, rank (1 = highest) |
| `cos_in_context` | the live gate put this finding in the prompt (kept and in its top 12) |
| `mlp_score`, `mlp_tau`, `mlp_keep` | MLP probability, `tau_R95`, `score >= tau` |
| `mlp_in_context` | kept by the MLP and in the MLP's top 12 (what it *would* have shown) |
| `gate_version` | `d10-mlp-pair-v1+<manifest sha256[:12]>` |
| `latency_us_call`, `latency_us_per_pair` | shadow scoring time for the call (features + MLP) |
| `shadow_errors_total` | swallowed shadow failures in this process so far |

## Turning it on and off

Nothing here is enabled by merging. The service reads the variable at call time, but the
running process only has the hook after it is restarted on the new code.

```bash
# on (operator):
systemctl --user edit loci-mcp        # add under [Service]:  Environment=LOCI_D10_SHADOW=1
systemctl --user restart loci-mcp

# rollback: remove that line (or set LOCI_D10_SHADOW=0), then
systemctl --user restart loci-mcp
# the log can be deleted at any time; nothing reads it but the report
rm -f ~/.loci/instrumentation/d10_shadow.jsonl*
```

With the variable unset the hook does one environment lookup and returns: no import of
the model, no scoring, no file.

## Reading the log

```bash
python3 scripts/d10_shadow_report.py                # defaults follow the server's memory dir
python3 scripts/d10_shadow_report.py --memory-dir ~/.loci/memory-sessions
```

The report opens files read-only and prints JSON aggregates to stdout:

- `keep_vs_keep` and `prompt_vs_prompt`: 2×2 counts of cosine against MLP (both keep,
  both drop, cosine only, MLP only) and the agreement rate;
- `latency_us_per_pair` p50/p95/max, `shadow_errors_total_max`, `calls_truncated`,
  `gate_versions`;
- `judge`: a join with each investigation's `judge_verdicts.jsonl`. **Read this with
  care.** The contradiction judge rules on finding–finding pairs (`agree`,
  `contradict`, `same_topic_no_conflict`), never on whether a finding is relevant to
  the question. Every judged pair is a same-topic pair, so the join measures
  *coherence*: how often each gate keeps both halves of a same-topic pair or drops
  both, rather than splitting it. It is not an accuracy score. Only pairs whose two
  findings were scored in the same shadow call are joined. At the time of writing the
  live store holds one verdict log, and all of its verdicts are null (judge
  unavailable), so expect this section to stay thin.

## What would justify enforcement

Not decided here; the operator decides later. A reasonable bar, all of it measured on
shadow data from real use:

1. **Volume**: at least N real `investigation_reason` calls (for example 50) across at
   least 10 investigations, with `shadow_errors_total` at 0 and p95 per-pair latency
   under 1 ms.
2. **Disagreements checked by hand**: a sample of the pairs where the gates disagree
   (both directions, for example 100), labelled blind by a person reading question and
   finding. The MLP should be right more often than cosine on them. Because the judge
   never rules on question relevance, this labelled sample is the only direct evidence
   of accuracy.
3. **No coherence loss**: on judged same-topic pairs, the MLP splits pairs no more
   often than cosine does.
4. **Retrain on real pairs** if (2) shows the finding–finding model does not transfer
   to question–finding pairs. The shadow log gives the ids needed to build that set.

Enforcement would be a separate, reviewed change that swaps the decision in
`investigation_reason`, still behind a flag, with this shadow log as its baseline.
