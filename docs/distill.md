# Distillation (`mcp/distill.py`)

Teacher labels -> splits -> D10 trainer input -> promote/hold gate -> label budget. Offline: no model
call, no GPU, writes only to the export dir. Promotion is manual; `decide` only reports.

```
python mcp/distill.py export --gold gold.jsonl [--teacher t.jsonl] --out DIR --seed N \
    --loci-commit SHA --embed-url URL --model-out DIR2     # prints manifest, then the trainer command
python mcp/distill.py gate --criteria docs/distill_criteria.example.json --test test.jsonl \
    --validation validation.jsonl --student-test s.json --incumbent-test i.json --student-validation v.json
python mcp/distill.py plan --cap-usd 5 --cost-per-label-usd 0.01 --curve curve.json \
    --min-gain-per-step 0.005 --step-labels 100
```

Rows: `{query_id|id, claim, evidence, label|gold_correct, [teacher_label]}`. Gold wins over teacher for
the same query id. Splits are by query id (hash of seed+id), so no query spans two splits.
`d10_train.jsonl` is the train split as `{claim, evidence, label, signal}`, the exact input of
`deep_think_loci/grounding/train_d10_pair_gate.py`. Train to a candidate dir; the pinned artifact
(`PINNED_MANIFEST_SHA256`) changes only by a manual re-pin.

Gate (criteria are data, see the example): promote only if the student beats the incumbent on test by
`min_margin_over_incumbent` AND its validation metric is within `max_teacher_gap_on_validation` of the
teacher's own metric vs gold on validation (`teacher_label` per row; gold validation rows must carry `teacher_label` too, else the gate holds). Score files map `row_id`
(`<query_id>#<k>`) to a score; predict 1 when score >= threshold.
