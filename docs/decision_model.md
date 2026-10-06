# Decision model (`mcp/decide.py`)

A decision model does not write text. Given a context and typed questions it returns probabilities over the options in
one forward pass: `choice` (pick one), `noul` (yes/no), `score` (a position on an ordered scale). Loci uses
OpenThai-SystemOne 0.8B (apache-2.0, Qwen3.5-0.8B base, Thai-heavy training, English supported).

## Use

```python
from decide import decide, yes_no, choose
yes_no("Passage: The bridge opened in 2002.", "Does the passage state the bridge's length?")      # P(yes) or None
choose("Charged twice, refund me", "Which team?", {"billing": "payments", "technical": None})     # {choice, probabilities, confidence}
decide(state, {"team": {"type": "choice", "instructions": "...", "criteria": {...}},
               "urgent": {"type": "noul", "instructions": "..."}})                                 # {ok, degraded, answers, error, ms}
```

`python mcp/decide.py check` runs 8 easy questions against the live model (exit 1 on any miss). Nothing raises: a problem
returns `{"ok": False, "degraded": True, "error": ...}`.

In `hillclimb`, `python mcp/hillclimb.py label --decide` shows the decision model's category and confidence **after** you
answer, tallied as `decide_compared` / `decide_agreed` beside the existing `--model` comparison.
`reflection_triage.classify_reflection_observation_decide` is the same classifier as a function.

## Setup

* Pool entry: `roles = ["decide"]`, `gpu = <card>` in `[[models.pool]]`; `LOCI_DECIDE_MODEL` overrides the model.
* Ollama < 0.35 has no `/v1/systemone`. Import the GGUF (`iapp/OpenThai-SystemOne-Ollama`, Q8_0 812 MB) with a Modelfile
  holding only `FROM`, the `SYSTEM` line from the repo's Modelfile and `PARAMETER num_ctx 8192`; **drop `REQUIRES 0.35.0`**.
  Check the file's sha256 against the Hugging Face listing first. Ollama 0.35+ can serve the native endpoint instead.
* `decide.py` ports ollama/ollama v0.35.1 `decision/systemone.go`: one chat prompt per question, the compact JSON of
  `{context, schema}` (the schema of every question) plus `Requested field: "<name>"`, scored on the answer-letter
  probabilities. The weights were fine-tuned on that exact prompt.

## What the port does that Ollama's scorer does not need to

* It sends `think: false`. Without it this server renders a thinking-mode prompt and the letter probabilities fall about five-fold.
* Only the top 20 logprobs come back, not every candidate's logit. A letter missing from them is scored at the 20th
  token's logprob (an upper bound) instead of 0, otherwise a single visible letter reads as confidence 1.0.
* `letter_mass` (probability on valid answer letters) is low by design: the model was trained on a loss over the candidate
  letters only. It is a diagnostic, not a validity gate.

## Measured (2026-10-06, Windows Ollama 0.34.2, RTX 2080 Ti)

| Test | Result |
|---|---|
| 12 easy cases (routing, intent, toxicity, grounded yes/no, NLI, regression-vs-flaky) | 12/12 |
| Harness refusal vs real tool failure, 116 distinct real error strings (labels = hand-adjudicated guard patterns) | 70% accuracy, guard recall 35/66, failure recall 46/50 |
| Same, calibration | wrong answers at confidence margin >= 0.9: 32/35 before the two fixes above, 1/35 after |
| Latency | 150-200 ms per question once loaded; 0 reloads (unlike the 9B qwen35 models) |

It is weak where a rule is exact: the worktree-isolation (17 misses), structured-output (10) and auto-mode
classifier (4) families. It called EPERM and EISDIR refusals. Use it to **find** candidates for rules (that is how
PR #464's guard patterns were found) and as a second opinion on labels, not as the classifier of record. The card's own
weak spots: summary-relevance scoring, English calibration (median ECE 0.15 vs 0.05 Thai). Confidence is a ranking
signal, not the chance of being right.
