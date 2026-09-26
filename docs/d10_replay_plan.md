# D10 offline replay: analysis plan and decision rule (pre-registered)

This plan is committed **before** the labelling sample is drawn. `scripts/d10_replay.py
sample` refuses to run unless this file is committed and unmodified, and it records the
file's sha256 and commit in the key file; `scripts/d10_replay_analyze.py` copies that
sha256 into its output. Any change after the sample is drawn needs a dated entry under
"Deviations" below saying what changed and why.

## Why a replay

`investigation_reason` keeps a finding for the reasoning prompt when the cosine of its
`nomic-embed-text` embedding to the question is at least 0.59, and puts the 12 highest
kept findings in the prompt. PR #406 added the D10 pair MLP in shadow mode
(`docs/d10_shadow_gate.md`). The MLP was trained on finding-finding pairs with
structural-proxy labels from four deep-think runs; the live gate scores
question-finding pairs. Whether it transfers can only be settled by labels on
question-finding pairs. Live shadow rows arrive one `investigation_reason` call at a
time (its first call: cosine kept 0 of 34 findings, the MLP 15), so this replays
historical investigations to get a labelled comparison in about a day.

The judge verdicts Loci records say whether two findings agree or contradict. They say
nothing about relevance to a question, so they are not ground truth here. Hand labels
are.

## Population

- Investigations: every directory under the memory dir (`~/.loci/memory-sessions`),
  read-only. Excluded, with the reason recorded in `run_meta.json`: internal
  directories (`_*`), no readable manifest, `flybrain-ops` (an operations log),
  names matching `smoke|probe|test|recover-` (case-insensitive substring), fewer than
  8 or more than 400 active findings, and no question text. Active findings are the
  ones `investigation_reason` would use: non-retracted (the store's own
  `retractions.jsonl` fold) with non-blank text.
- Sample of investigations: 80 (or all eligible if fewer), stratified 3 x 3 by
  finding-count tertile and `created_at` tertile, proportional allocation (largest
  remainder, at least one per non-empty stratum), seeded simple random sampling within
  strata. Seed 20260926. The design is recorded in `run_meta.json`.

## Questions (no language model)

For each sampled investigation, each of the manifest's `title`, `hypothesis` and
`next_step` that is non-empty becomes one question (whitespace collapsed; a text
repeated across fields is asked once; the kind is recorded).

**A title is a proxy for a real question.** Real `investigation_reason` questions are
typed by an operator or an agent and are usually narrower than a title. Titles are
broad, so a larger share of an investigation's findings will be relevant to them than
to a real question; hypotheses and next steps are closer to real questions. Results are
reported per question kind as well as pooled. **Where the replay and the live shadow
data disagree, the live data wins.**

## Gates (production code, not copies)

- Cosine: `grounding_gate.cosine_gate` and `grounding_gate.passes`, the functions
  `investigation_reason` calls (this PR moves the rule there verbatim; an
  unanswerable `None` cosine is now dropped instead of raising).
  `memcheck.llm.cosine` computes the cosine, as live.
- MLP: `d10_gate.load_gate()` (the hash-pinned artifact) and `d10_gate.mlp_decisions`
  (score >= tau = 0.2144; the shadow logger uses the same function), on
  `d10_gate.pair_features` with u = question, v = finding, as in `record_shadow`.
- Embeddings: `memcheck.llm.embed_texts` against the local Ollama relay
  (`/v1/embeddings`, `nomic-embed-text`, texts truncated to 2,000 characters), the
  client `investigation_reason` uses. Vectors are cached by sha256 of model and text.
  The served model digest is recorded.

Each (question, finding) pair gets: cosine, cosine keep (>= 0.59), cosine in-context
(top 12 of kept), MLP score, MLP keep, MLP in-context, and a category from the keep
flags: `cos_only`, `mlp_only`, `both_keep`, `both_drop`.

**Primary comparison: the keep decision, not the top-12 cap.** The cap is shared
plumbing that truncates both gates the same way; the question is which gate's
judgement of relevance is better. In-context categories are reported as population
counts (no labels needed) as a secondary view.

## Labelling sample

Drawn by `d10_replay.py sample` after this plan is committed, seed 20260927:

| Stratum | Target | Role |
|---|---:|---|
| `cos_only` | 100 (all if fewer) | disagreement |
| `mlp_only` | 100 (all if fewer) | disagreement |
| `both_keep` | 30 | control |
| `both_drop` | 30 | control |
| duplicates | 12 | test-retest: re-shown under a new id at least 30 items later |

Simple random sampling without replacement within each stratum. Everything is
shuffled. The sheet shows an opaque id, the question and the finding text (truncated to
700 characters with an expand control) and nothing else: no score, gate, category,
question kind or investigation. Choices: relevant, not relevant, unsure.

## Analysis (scripts/d10_replay_analyze.py)

Duplicated items count only for consistency. Unsure labels are excluded from every
rate and counted.

1. **Disagreements.** On a disagreement exactly one gate keeps the finding, so exactly
   one is right (the keeper if relevant, the dropper if not). Report the counts, the
   MLP's share correct, and the exact two-sided binomial sign test (p = 1/2) of
   MLP-correct against cosine-correct over decisive labelled disagreements. Per
   question kind as well.
2. **Population estimates.** Stratified (post-stratified Horvitz-Thompson ratio)
   estimator: stratum s has N_s pairs in the replay and m_s decisive labels, of which
   r_s relevant; p_s = r_s / m_s; estimated relevant pairs R_s = N_s p_s and
   not-relevant B_s = N_s (1 - p_s). Grounded recall of a gate = sum of R_s over the
   strata it keeps / sum of all R_s. Bleed rejection = sum of B_s over the strata it
   drops / sum of all B_s. Also the MLP's weighted share correct over disagreements,
   (B_cos_only + R_mlp_only) / (N_cos_only + N_mlp_only), and each gate's recall
   relative to relevant pairs kept by either gate.
   Assumptions: simple random sampling within strata (true by construction); unsure
   labels missing at random within a stratum; one labeller's judgement is the truth;
   the replay population (sampled investigations x manifest questions) stands for the
   live question distribution, which it does only approximately (see Questions).
   **Precision warning:** `both_drop` is by far the largest stratum and gets 30
   labels, so each relevant label there stands for many pairs; recall estimates for
   both gates are dominated by it and are wide. The recall-of-either-kept figure does
   not depend on `both_drop` and is reported alongside.
3. **Intervals.** 95 % percentile bootstrap, 2,000 replicates, seed 0, resampling
   investigations with replacement (pairs within an investigation share questions and
   a topic, so pairs are not independent). Each replicate recomputes N_s from the drawn
   investigations and uses only their labelled items. A replicate that has pairs in a
   stratum but no decisive label there is undefined for population rates; the count is
   reported.
4. **Consistency.** Test-retest agreement and Cohen's kappa on the duplicates. Control
   base rates: share relevant among `both_keep` and among `both_drop`; the controls are
   "ordered" when the first exceeds the second. A labelling that fails that ordering,
   or kappa below 0.4, is flagged in the report as unreliable.
5. **Unsure rate**, overall and among disagreements.

Committed outputs, if any, are aggregates only (the analysis script prints no id or
text). The sheet, key, labels, pairs and vectors stay in the private output directory.

## Decision rule

The MLP is a **candidate for enforcement** only if all of these hold:

1. among labelled disagreements, the unsure rate is below 25 %;
2. the MLP is correct on at least 60 % of decisive labelled disagreements, **and** its
   weighted share correct (item 2 above) is at least 60 %;
3. the sign test p is below 0.01;
4. its estimated grounded recall is at least 0.95, **and** the investigation-bootstrap
   95 % lower bound is above 0.90, with no more than 5 % undefined replicates.

Otherwise the cosine gate remains the safer default. The result never authorises
enforcement by itself: the operator decides, and only after live shadow data agrees.

Changes from the suggested rule, with reasons:

- The weighted share in (2) is added. The design fixes 100 + 100 disagreement labels
  whatever the sizes of the two disagreement strata, so the unweighted share (and the
  sign test on it) describes the sample, not the population of disagreements; requiring
  both keeps a lopsided stratum from carrying the result.
- The undefined-replicate cap in (4) is added so a lower bound computed from a
  minority of replicates cannot pass.

## Deviations

None yet.
