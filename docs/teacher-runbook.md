# Teacher runbook: deploy order, config, first pilot, off switch

Components: core (`mcp/teacher.py`, `docs/teacher.md`), gold labels (`docs/gold-labels.md`),
judge calibration + oracle judging, hillclimb teacher proposer, memory adjudication / promote
verify gate (`mcp/memory_adjudicate.py`), distillation export (`docs/distill.md`).
Everything defaults OFF. The guard fails closed; the features fail open.

## Deploy order

1. Merge and deploy the code with teacher disabled (no config change). Nothing calls the cloud.
2. Add the LiteLLM groups for the roles to the LiteLLM config (owner step), then the `[teacher]`
   block below to `backends.toml`. Keep `enabled = false` until step 4.
3. `python mcp/teacher.py status` and `python mcp/teacher.py check judge`: confirm caps, prices,
   ledger path and that the ledger exists. Exit 1 from `check` means the guard refuses; read `reason`.
4. Set `enabled = true`. Run the pilot below with the low caps.
5. Only after the pilot spend matches the ledger and the Console: enable per-feature switches
   (judge oracle, hillclimb teacher proposer, promote verify gate) one at a time.
6. Raise caps only after a week of the ledger matching the Console.

## backends.toml

```toml
[teacher]
enabled = true
daily_usd_cap = 1.0
monthly_usd_cap = 20.0
max_usd_per_call = 0.10        # lower if a single prompt can be large
ledger_file = "~/.loci/litellm_cost_ledger.jsonl"
safety_margin_pct = 10
sanitize = true

[teacher.roles]                # role -> LiteLLM group name from your LiteLLM config
judge = "<litellm-group>"
label_a = "<litellm-group>"
label_b = "<litellm-group>"
propose = "<litellm-group>"
adjudicate = "<litellm-group>"
verify = "<litellm-group>"

[teacher.prices.<litellm-group>]   # one table per group named above; USD per million tokens
in_usd_per_mtok = 0.0          # fill from the provider price list; keep in sync with LiteLLM
out_usd_per_mtok = 0.0
```

A group with no price is refused (`no_price`); a zero cap allows no spend. Env `LOCI_TEACHER_*`
overrides the file. Gold file location: `[gold] file` / `LOCI_GOLD_FILE`.

## Monthly cap and billing

Provider billing is not readable by API. The LiteLLM cost ledger is the only spend record the
guard has, and `monthly_usd_cap` is enforced against it alone. Compare
`python mcp/teacher.py status` with the provider Console at least weekly, and treat any gap as a
stale price table or ledger problem: fix it before raising a cap. A cap set in the provider Console
is a second brake, set it at or just above `monthly_usd_cap`.

## First live pilot

Run from the repo root with `PYTHONPATH` set to it. The owner runs these, with the caps above.

```
python mcp/teacher.py status
python mcp/gold_labels.py plan --queries 50 --source audit
python mcp/gold_labels.py build --queries 50 --source audit --max-usd 5
python mcp/gold_labels.py stats
python mcp/judge_calibration.py report --judge <judge.jsonl> --gold <gold.jsonl> --json
python mcp/hillclimb.py run --suite query_expand --rounds 1
python mcp/teacher.py status
```

Notes:
- `build` stops before a query that could exceed `--max-usd`; the 1 USD/day cap will stop it first
  unless raised, in which case rerun the next day (build resumes per query).
- `judge_calibration report` is local (no spend). Both files are JSONL keyed by `id`/`item`
  (see mismatches below).
- The hillclimb has no `--dry-run`. The spend-free equivalent is the default local proposer
  (`--rounds 1`, no `--proposer teacher`). The teacher proposer needs `"allow_cloud": true` in the
  suite `config.json` and `--max-usd`; do that only after the local run is clean.
- Compare the last `status` with the Console.

## Cross-component mismatches (adapters needed before the pilot chains up)

- `gold_labels` writes `label` rows keyed `query_id` + `candidate_id` with `grade`; `judge_calibration`
  needs one id per item (`id`, `item` or `decision_id`), unique across the file. Join the judge
  file and the gold file on `query_id:candidate_id` yourself.
- `distill` wants `{claim, evidence, label}` rows; `gold_labels` does not emit claim/evidence text.
- The hillclimb `query_expand` suite reads `<memory parent>/teacher/gold/query_expand.jsonl`
  (`{id, query, expect: [terms]}`); nothing produces that file. Without it the run stops
  `suite_unavailable`.

## Turn it all off

- Fast: `enabled = false` in `[teacher]`, or `LOCI_TEACHER_ENABLED=0` in the environment; every
  `teacher_call` then returns `disabled` and each feature degrades to its local path.
- Per feature: unset `LOCI_JUDGE_ORACLE`, `LOCI_PROMOTE_TEACHER_VERIFY`, and use the default
  hillclimb proposer (omit `--proposer teacher`).
- Hard stop on spend: set the three caps to 0, or remove `[teacher.roles]`.
- Written data (`~/.loci/gold/`, `teacher_verify.jsonl`, hillclimb logs) stays; delete it by hand if wanted.
