# FlyBrain-style brains in Loci: where they could apply

**Status:** evaluation, 2026-10-04. Docs only. Nothing here changes Loci behaviour.
**Builds on:** `docs/FLYBRAIN.md` (Loci never imports FlyBrain code), `docs/d10_shadow_gate.md`, `docs/d10_replay_plan.md`, `mcp/instrumentation_log.py`.
**Where the research lives:** private repo `rjmendez/flybrain` (grower, baselines, realism critic, run logs). Loci keeps only guardrails, exporters and shadow gates.

**Data rules followed.** This repo is public, so the document carries no counts from local stores, no claim or query text, and no investigation names or ids. Sizes are given only as orders of magnitude.

## Summary

| Question | Answer |
|---|---|
| Is there a decision where a multi-input grown brain could plausibly earn its place? | **Maybe one: the contradiction-judge pre-filter (D11).** Labels now exist (persisted judge verdicts). Value is modest since #409 made the judge asynchronous and capped. |
| Has a grown brain already been tried on a Loci decision? | **Yes, D10. Negative.** Grown brains scored below the shipped pair MLP and below the current cosine rule, and missed the 1 ms budget (private FlyBrain repo, D10 task doc). |
| Does Loci have a conventional learned gate already? | **Yes.** D10 pair MLP, shadow only (#406), numpy inference, well under 1 ms per pair. |
| Overall fit | **Moderate as an evaluation target, weak as a replacement target.** Best use: a harness benchmark with real embeddings and real labels, and the shadow-mode instrumentation Loci already has. |
| Rule-based gates (D25 to D32) | **No fit, by design.** They must stay deterministic and auditable. |

## (a) Inputs and signals Loci already has

| Signal | Shape | Volume | Where | Status |
|---|---|---|---|---|
| Finding text + embedding | one embedding per finding (model and dimension: verify in `embed_ops.py`) | thousands of findings across hundreds of investigation dirs | `<memory dir>/<investigation>/findings.jsonl`; vectors in Qdrant | inferred from code |
| Finding metadata | type (observed / inferred / gap / procedure / assumed), confidence, tier, provenance tier, `derived_from`, `supersedes` | as above | `findings.jsonl` | read |
| Contradiction-judge verdicts | pair ids, similarity, `heuristic_rule`, `judge_ok`, `judge_skipped`, `verdict` in {contradict, agree, same_topic_no_conflict}, neighbour rank and type | thousands of rows from about a dozen investigations; a few thousand judged, a small minority `contradict`; a majority of rows are `judge_skipped` | `<investigation>/judge_verdicts.jsonl` (#401) | read |
| Human conflict resolutions | open and resolved conflicts, resolution kind | only a handful resolved | `conflicts.jsonl` | read |
| Skeptic verdicts | confirmed / refuted / uncertain | a hundred or so | `finding_verifications.jsonl` | read |
| D10 shadow log | cosine and MLP keep/drop per (question, finding) pair, scores, latency | dozens of rows so far | `<data home>/instrumentation/d10_shadow.jsonl` (#406) | read |
| Memory-use log | which surfaced memories a finding cited | dozens of rows so far | `<data home>/instrumentation/memory_use.jsonl` (#402) | read |
| Route traces | sampled `memory_route` traces (#403) | not counted | audit log | inferred |
| Grounding pairs | labelled question-finding pairs from recovered runs; labels are structural proxies (same `dt_target`) | thousands | `deep_think_loci/grounding/grounding_dataset.jsonl` | read in R6.1 |

Not a signal on its own: GPU load. Model-routing decisions and their outcomes are now logged (D23 below).

## (b) Candidate decisions, ranked

The ranking keeps the R6.1 rubric and updates it for what changed since 2026-09-25: judge verdicts persist (#401), the judge left the store path (#409), D10 runs in shadow (#406, #407), memory use and route traces are instrumented (#402, #403).

| Rank | Decision | Input spec | Output | Label that exists | Baseline to beat | Budget | Cost of a wrong call |
|---|---|---|---|---|---|---|---|
| 1 | **D11 judge pre-filter**: should this (new finding, neighbour) pair go to the contradiction judge? | `[u, v, |u-v|, u*v]` of unit embeddings + `similarity`, `heuristic_rule`, `new_type`, `neighbor_type`, `neighbor_rank`, cheap text flags (negation, number mismatch) | P(judge returns contradict) | `judge_verdicts.verdict` on the judged pairs. Precomputed by the judge, so a distillation target, not truth. Human gold: a handful of resolved conflicts | Current rule (cosine >= 0.82, top `max_pairs`=3); LR and HGB on tabular features; pair MLP; fixed random sparse expansion + linear readout | < 1 ms per pair; <= 0.25 M parameters; numpy-only loader | Low to moderate. The judge is advisory and async; a missed pair is a missed advisory flag. Fail open = send to judge |
| 2 | **D10 evidence relevance gate**, second attempt on hand labels | same pair view, question-finding pairs | keep / drop | Hand labels from the D10 replay (#407) once collected. Today only structural proxies | The shipped pair MLP (shadow), the cosine rule | 1 ms per call; missed by grown brains | Moderate (off-topic evidence reaches the reasoning prompt) |
| 3 | **D17 entailment-check trigger** | claim and evidence features from `pre_answer_check` | run verifier or not (add-only) | Skeptic verdicts; verdict history in the vector store not read | Current one-line rule; LR/HGB | microseconds | High if it removes checks. Only an add-only form is safe |
| 4 | **D13 retrieval-worthiness / tier admission** | finding embedding + metadata | admit / tier | `memory_use` is too small; access markers say "returned", not "used" | Heuristic tier, LR on metadata | microseconds | Low |
| 5 | **Memory routing and per-turn injection (D1, D2, D8)** | query embedding, recent context | which store, how many | `memory_use` and route traces just started | Current drive rules; kNN over embeddings | < 300 ms path | Moderate to high (irrelevant context injected every prompt) |
| 6 | **D12 store-time novelty / dedup** | finding embedding | similar exists? | **None.** Loci does no store-time dedup and logs no merges. Supersession is a weak proxy | Cosine threshold, SimHash/LSH, LR | microseconds | Low |
| 7 | **D21 keep searching vs answer** | transcript state | continue / stop | **None.** No episode logs | Hard budgets already in `offload_loop` | per step | Moderate |
| 8 | **D23 model selection per role** | role, the ranked candidates (rank, resident, eligible), GPU load | which model | `model_pool_outcomes.jsonl`: `ok`, latency and `deadline_exceeded` per call, joined to the decision row. Accumulating since the pool shipped (#435) and outcomes were added (#439); off until `LOCI_MODEL_POOL_SHADOW=1` | The ranked-pool rule; budget < 1 ms | < 1 ms | Low (the legacy resolver is the fallback) |
| n/a | D25 to D32 honesty and safety gates | | | | | | **No good fit.** Must stay rule-based |

Decision IDs (D1 to D32) come from the R6.1 decision-point survey in the private FlyBrain repo; each ID used in this document is named in the table above.

Notes on weak fits:

- **D12 / novelty.** The plan's first hypothesis. No labels, no logged merges, several read-time rules exist. A fixed random sparse expansion is the known fly result (FlyHash); a grown brain has nothing to learn from. Useful as a **benchmark**, not as a decision (section e, track B).
- **D13, D5 to D8.** The label that would matter ("was the injected memory used") exists in small numbers only. Revisit after `memory_use` has thousands of rows.
- **D21 / keep searching.** Blocked on instrumentation, not on modelling.
- **D23 / model selection.** Now a concrete decision with the integration shape of section (d) already in place (`mcp/model_pool.py`). The decision and outcome logs are off by default and start empty, so there is no volume yet. The input space is small and tabular (a handful of candidates per role), so a grown brain is unlikely to beat the ranked rule; the value is a second labelled benchmark with a hard latency budget.

## (c) Data and privacy constraints for training

| Constraint | Rule |
|---|---|
| Content | Findings are user investigation content. Train from embeddings, ids, enums and counts only; never put text in a dataset that leaves the store. |
| Embeddings | Treat as invertible. Frozen datasets go to a private directory outside the repo, mode 600, sha256-manifested. Only schemas and aggregate statistics are committed. |
| Repo is public | This document, PR text and committed manifests carry aggregates only: no investigation names, ids, queries or paths from a person's home directory. |
| Population | Reuse the D10 replay exclusions: internal dirs, no manifest, `flybrain-ops`, names matching `smoke|probe|test|recover-`, fewer than 8 or more than 400 findings. |
| Splits | By investigation (and time), never by pair. A pair split leaked much of the apparent D10 margin (`docs/grounding-corpus-limits.md`). |
| Labels | Precomputed offline. Training never calls a live language model. Judge verdicts record `model`; fix one model version per dataset. |
| Hermetic tests | Tests that touch stores run with `HOME` and `LOCI_CONFIG` isolated; the suite has written to live stores before. |
| Cloud tier | No training export through `_try_cloud_tier`. |

## (d) Integration shape

Follow the D10 precedent exactly.

| Piece | Shape |
|---|---|
| Artifact | `weights.npz` + `manifest.json` (sha256 of weights, dataset sha256 and Loci commit, embedder digest, recipe, threshold table, reproduced offline metrics). Loader refuses on hash mismatch. No pickle. |
| Code in Loci | numpy-only inference and feature definition (like `mcp/d10_gate.py`). Loci imports no FlyBrain code. |
| Mode | Shadow only, off by default (`LOCI_<GATE>_SHADOW=1`). Logs next to the current decision via `instrumentation_log.append_rows`: ids, scores, enums, latency; no text. |
| No LLM in the body | The brain scores features only. Labels come from an offline replay of the judge over frozen pairs, run once, outside the brain. |
| Fallback | Any load, shape or timeout error returns the current rule's decision. A pre-filter may only skip judge calls on pairs the brain scores below a threshold chosen for recall >= 0.95; it never removes a heuristic conflict. |
| Rollback | Unset the env var, or delete the artifact directory. One step. |
| Evidence to go beyond shadow | Pre-registered live metric (calls avoided at fixed recall) from a read-only report script, then an A/B. |
| Worked example in Loci | The model pool (`mcp/model_pool.py`) follows this shape without a brain behind it: the rule always decides; `LOCI_MODEL_POOL_SHADOW=1` logs each decision and each call outcome (names, ranks, enums, latency; no text); `LOCI_MODEL_POOL_SELECTOR=module:callable` names a selector whose choice is logged beside the rule's and never used; unsetting both is the rollback. A learned D23 selector plugs into that hook. Since 2026-10-05 every decision row has a `decision_id` shared with the outcome of the call it led to (the join was previously "the next outcome for that model", which four workers make wrong), `shadow_status` says why `chosen_shadow` is empty (`no_selector`, `abstained`, `invalid`, `error`), `mcp/pool_selectors.py:success_rate` is the baseline any brain has to beat, and `python mcp/model_pool.py report` says per role whether the logs can support learning. Shadow alone never produces an outcome for an arm the rule does not pick, so the report says "only one arm observed" until `LOCI_MODEL_POOL_EXPLORE` and `LOCI_MODEL_POOL_EXPLORE_MODELS` (opt-in, off by default, changes live routing for that fraction of calls, logs the propensity) are set. |

## (e) Recommended first experiment

Small enough for one PR on the FlyBrain side plus one exporter in Loci. Two tracks; only A is a Loci decision.

### Track A: D11 judge pre-filter, offline, pre-registered

| Item | Value |
|---|---|
| Question | Can a small model predict which candidate pairs the judge will call "contradict", so the judge can skip the rest at recall >= 0.95? |
| Dataset | The judged rows of `judge_verdicts.jsonl` (verdict != skipped), joined to cached embeddings. Frozen, sha256-manifested, private. |
| Split | Leave-one-investigation-out folds. No investigation in more than one of train, val, test. |
| Primary metric | **Judge calls avoided**: share of candidate pairs the gate drops while keeping >= 95% of `contradict` verdicts, pooled across test folds. |
| Secondary | AUROC, debiased ECE, per-investigation spread, parameter count, p95 latency per pair. |
| Baselines (same rows, splits, seeds) | Current rule; LR; HGB; pair MLP; fixed random sparse expansion + linear readout; majority and best single-feature stump. |
| Statistics | Paired bootstrap that resamples investigations, not pairs. |
| Acceptance bar | A brain is eligible for shadow only if (CI of brain minus best baseline on the primary metric excludes 0) **or** (non-inferior within 0.02 at <= 25% of the best baseline's parameters and p95 <= 1 ms). Otherwise it stays a research artifact. |
| Power gate | Report as **exploratory** unless the test folds hold >= 40 positives from >= 8 investigations. A confirmatory run waits for >= 500 positives from >= 30 investigations. **Status 2026-10-04: not met.** Across the local stores the positives number in the low hundreds at most, sit in fewer than 8 investigations, and one investigation holds the large majority, so a leave-one-investigation-out split is dominated by a single fold. Track A stays blocked until more investigations carry judged pairs. |
| Null path | If no brain qualifies, record the baselines and ship nothing. The best baseline may still be proposed as a shadow gate on its own merits. |

### Track B: generic familiarity and retrieval benchmark (FlyBrain repo only)

Real Loci finding embeddings as one corpus in a multi-input harness. Tasks: familiarity/novelty (planted paraphrase and perturbation duplicates, plus supersession pairs as weak real labels) and nearest-neighbour retrieval. Baselines: cosine threshold, SimHash/LSH, fixed random sparse expansion (FlyHash), LR/HGB. This gives the harness a real-embedding benchmark. It is not a Loci decision and needs no Loci change beyond a read-only exporter.

## (f) What changes where

| Location | Change |
|---|---|
| Loci, now | This document only: `docs/flybrain_brains_eval.md`, plus a one-line link from `docs/FLYBRAIN.md`. |
| Loci, if Track A qualifies | `scripts/d11_export_pairs.py` (read-only, refuses output under the memory dir, `~/.loci` or `~/.hermes`, like `d10_replay.py`); `mcp/d11_gate.py` and `mcp/models/d11_prefilter/` (shadow, mirrors `d10_gate.py`); `scripts/d11_shadow_report.py`; `docs/d11_shadow_gate.md`; tests under `mcp/tests/`. |
| FlyBrain (private) | Body spec, task doc, dataset builder, baselines, grower, realism critic, run logs, the multi-input harness, Track B. |
| Never in Loci | Grower, critic, genomes, FlyBrain imports, private datasets. |

## (g) Risks and open questions

| Risk | Note |
|---|---|
| Labels are a model's verdicts | D11 labels distil one local model. A pre-filter learns the judge's habits, including its errors. Only a handful of human resolutions exist as gold. |
| Skipped rows | Most verdict rows are `judge_skipped` (roughly two thirds in each local store), and selection into the judged set is not recorded in the row. Check the cause before trusting the population; the same rows are what limits the power gate above. |
| Small N | About a dozen investigations. Intervals will be wide, as in D10. |
| Prior from D10 | A grown brain lost there on size, latency and score. Expect a negative result; the harness value is the deliverable. |
| Lower stakes than R6.1 scored | #409 made the judge async with a cap of 3 pairs and a circuit breaker. The latency argument for a pre-filter is gone; the remaining gain is judge compute. |
| Training vs live store | History sits mostly in one local store while the live instance's store is small, so training and live data may differ in distribution. |
| Drift | Judge model, embedder or threshold changes invalidate the dataset; the manifest pins them. |

Open questions:

1. Which store is the training source, and may it be used for frozen datasets?
2. Is an LLM judge verdict acceptable as a training label, or should only human resolutions count?
3. Does D11 judge compute cost enough to justify any gate, now that it is async and capped at 3 pairs?
4. Should `memory_use` logging get a wider sample so D5 to D8 become testable?
5. Should hand-labelling for the D10 replay (#407) finish before any second D10 brain?
6. Is the D23 outcome log (#439) enough label for a learned selector, or is a per-role success rate over the ranked candidates already the ceiling? Enable the shadow flag for a few weeks and read `python mcp/model_pool.py report` before deciding: on the 2026-10-05 logs it would have said that only one arm (`gemma4-e4b-hermes:16k`) had ever been observed, so neither a selector nor the ranked rule could be tested against an alternative.
