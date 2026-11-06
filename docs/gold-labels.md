# Gold retrieval labels

`mcp/gold_labels.py` builds a graded (0-3) relevance set for `retrieval_eval`, replacing the
same-investigation proxy. Needs the teacher core enabled (`docs/teacher.md`); roles `label_a`,
`label_b`, `adjudicate` map to LiteLLM groups in config only.

    python mcp/gold_labels.py plan  [--queries N] [--top-k K] [--source audit|synthetic|both]
    python mcp/gold_labels.py build [--max-usd X] [--dry-run] [--out PATH]
    python mcp/gold_labels.py stats [--out PATH]

- Queries: real ones from `~/.loci/audit/*.jsonl` (search-tool rows with a non-empty `query`); if too few,
  investigation titles and finding text, marked `synthetic`. Deduplicated, stratified by collection.
- One grader call per (query, role) over the whole pool; candidates are aliased c1..cN, sanitized,
  truncated, and delimited with a per-call nonce as untrusted data.
- Agreement: equal or within 1 -> mean rounded half up; gap > 1 -> `adjudicate` once; it must match one
  labeler exactly, else the candidate is dropped and counted.
- Output `~/.loci/gold/retrieval_gold.jsonl` (`[gold] file`, env `LOCI_GOLD_FILE`; also `top_k`,
  `max_text_chars`): `query` rows and `label` rows with provenance. Resumable per query, atomic rewrite.
- `plan` / `--dry-run` call nothing. `--max-usd` stops before the query that could exceed it; a guard
  refusal stops the run (exit 3) and is counted in `refusals`; transport errors are counted in `errors`.
- Eval: `scripts/shadow_eval.py --gold [PATH]` uses `retrieval_eval.evaluate_gold` (recall@k and MRR on
  grade >= 1, nDCG@k with gain 2^grade - 1). Default behavior without the flag is unchanged.
