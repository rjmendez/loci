# What the grounding corpus can and cannot buy

Measured 2026-08-29, on the first runs of `mlops/loop.py` that ever completed.
Written down because three of these numbers contradict what the tooling prints,
and the printed ones are the ones that get quoted later.

## The corpus is 303 findings, not 12,684 rows

`grounding_dataset.jsonl` holds pairs, and pairs are `C(n,2)` within a topic:

| findings | positives | |
|---|---|---|
| 303 (now) | 4,203 | |
| 454 (1.5x) | 9,560 | 2.3x the rows from 1.5x the evidence |
| 606 (2x) | 17,115 | 4.1x the rows from 2x the evidence |
| 1,515 (5x) | 108,105 | 25.7x the rows from 5x the evidence |

Row count grows quadratically while information grows linearly. Judge corpus
growth in findings; the row count will always look like faster progress than it
is.

Source: 9 `dt-loci-*` investigations, median 39 usable findings each. One more
investigation is roughly +39 findings, +13%.

## Two thirds of the model's apparent margin is a leak

`train.py` reports 10-fold CV over **pairs**, so a finding appears in train and
test on every fold. Splitting findings *before* building pairs (6 seeds, paired
so between-seed variance cancels):

| | pair-level (printed) | leak-free |
|---|---|---|
| GradientBoosting | 0.944 | 0.908 +/- 0.029 |
| cosine baseline | 0.864 | 0.878 +/- 0.028 |
| margin over cosine | +0.080 | **+0.030 +/- 0.010** |

The model is better than a dot product. By about a third of what the log claims.
The promotion gate itself is sound — it uses leave-one-run-out whenever
`--findings-glob` is passed, which `loop.py` always does.

## More findings help the promoted model, and only that one

Gain from 95 -> 213 findings (2.2x), leak-free, 6 seeds:

| model | gain | per seed |
|---|---|---|
| LogisticRegression | +0.003 +/- 0.005 | +0.011, -0.004, +0.003, +0.009, 0.000, 0.000 |
| GradientBoosting | **+0.026 +/- 0.015** | +0.028, +0.048, +0.012, +0.041, +0.004, +0.024 |

LogisticRegression is saturated: a linear model over `|a-b|`, `a*b`, `cos` can
only re-learn the dot product, and 7x the findings moves it +0.006 — inside the
seed spread. GradientBoosting gains in 6 of 6 seeds.

So investigations are worth commissioning, but only because the nonlinear model
is the one that gets promoted. Extrapolating the measured slope, ~8 more
investigations at the current median would be worth roughly another +0.03,
diminishing after that.

## The access records are not the free training data they look like

`findings.jsonl` is a mixed log: 493 of 795 rows are `record_type: "access"`
retrieval telemetry with a `query` field, and every one resolves to a real
finding. 64 distinct queries, 265 (query, finding) positives. It looks like a
free corpus in exactly the shape the task wants — "does this evidence answer this
query" rather than "do these two findings share a tag".

It is not. An access record means the finding was *retrieved*, and retrieval is
what the model is meant to improve:

    cosine AUC at predicting "was retrieved":  0.929
    retrieved findings in cosine's top 10:     65%
    median rank under cosine:                  7

At 0.929 the label is mostly the retriever's own output. Training on it teaches
the model to imitate the ranker it is supposed to beat — the circular-label trap.
The 35% that cosine does not rank highly is the reranker's contribution, which is
another model's judgement, not ground truth.

Useful for: evaluating retrieval, and mining hard negatives. Not useful as
supervision.

## What actually caps the score

Positives are "these two findings share a `dt_target` tag". The model is learning
tag agreement, not grounding. No quantity of the same kind of data moves that
ceiling — it needs a label that encodes whether evidence *grounds* a claim, which
nothing in the current pipeline records.

## Reproducing

The learning-curve and circularity scripts are not committed; they are throwaway
harnesses around `mlops/grounding/train.py`'s own `embed_texts` and feature
construction. The method is the part worth keeping: **split findings before
building pairs**, never split the pairs.
