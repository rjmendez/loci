# Changelog — deep-think-loci

All notable changes to the deep-think-loci reasoning engine. Versions track the
engine iteration (v1 → v3.2); each was validated against real runs over the Loci
corpus, and the empirical finding behind each change is recorded.

## [3.2.0-beta] — 2026-06-19

**All-Claude. External uncensored tier removed.**

### Changed
- **Dropped the external (heretic/abliterated) adversarial tier entirely.** It never
  persisted across 4 runs, the opus tiers already produce strong adversarial output
  on their own (v3.1 opus found a cross-target fail-open pattern with zero external
  input), and removing it spends **zero** external-provider tokens.
- The opus half-syntheses and final now **own the adversarial red-team** (per-target
  security-nightmare analysis + cross-target pattern detection + integrity check).

### Result
- Shape: init → 5 haiku ideation → dedicated writer → verify → 2 opus halves → opus final.
- ~$5/run (cheaper than v3.1's $6.77; whole adversarial phase removed), 0 external tokens.

## [3.1.0] — 2026-06-19

**Grounding gate tuned. Persistence solid.**

### Fixed
- **Grounding gate false-drops.** v3 used a blended multi-target query that diluted
  cosines and dropped whole genuine targets. Offline sweep → **per-target queries +
  threshold 0.59** (on-target min 0.600 > off-target p90 0.590). Confirmed: 0 false-drops,
  governance-gate restored to top findings.

### Known issue
- The external adversarial tier still persisted 0 (root-caused: agents fail the gate's
  temp-file handoff and *fabricate* under a broken multi-step shim). Resolved in 3.2 by
  removing the tier.

## [3.0.0] — 2026-06-19

**Dedicated-writer persistence + cosine grounding gate.**

### Added
- **Dedicated-writer pattern.** Generation agents return schema-validated output only;
  one single-purpose writer does all `investigation_store` calls. Persistence 40% → 100%
  (5/5 targets, validated by benchmark dt-bench-001: 9/9).
- **Cosine grounding gate** (`grounding/ground_gate.py`) — filters RAG-bleed before any
  model reasons. Plus the grounding specialist: dataset builder, 519-example labeled
  dataset, trained bleed-detector (CV AUC 0.957).

## [2.0.0] — 2026-06-19

**Integrity fixes.**

### Added
- `investigation_load` truth-gates between phases (thread real finding_ids, never trust
  agent self-reports), investigation-scoped RAG for untrusted tiers, mandatory grounding
  gate, quarantine of untrusted-model output (`assumed`/`dt_trust:untrusted`).

### Result
- No hallucination/poison entered memory (the v1 confabulation is gone); opus refused to
  fabricate empty partitions. Persistence still flaky (fixed in 3.0).

## [1.0.0] — 2026-06-19

**First build.** Tiered models (haiku ideation → adversarial → synthesis → opus final)
over a shared Loci/qdrant corpus with position + lineage (`derived_from`) tagging.

### Failure modes found (drove v2/v3)
- Agents fabricate `investigation_store` confirmations (only 1/5 persisted).
- The external uncensored synthesis hallucinated a 15-finding "kill chain" from global
  RAG-bleed (ambient cross-context memory) — also a host-memory leak to the provider.
- RAG semantic bleed: cross-target findings at cosine 0.35–0.55 fool a non-verifying model.
- What worked: the opus tier verified via `investigation_load`, reconciled similarity vs
  tags, refused to fabricate, and caught the poison — the basis for the v2/v3 gates.
