# Teacher: budget-guarded cloud calls

`mcp/teacher.py` is the one door for learning loops (judge, labelers, proposer, adjudicator,
verifier) to the cloud. Off by default. The guard fails CLOSED; callers fail OPEN: `teacher_call`
never raises, a refusal or error is a result the feature degrades on.

Spend is read from the LiteLLM cost ledger. Role -> LiteLLM group names and the per-group price
table come from config only, read at call time. Nothing is defaulted in code.

**LiteLLM has no budget enforcement here. The Loci guard is the only brake.** The `[teacher.prices]`
table is operator-supplied and must be kept in sync with the LiteLLM config; a stale (too low) price
lets calls cost more than the guard thinks. Such calls are counted as overruns (see `status()`).

## Config

`backends.toml` `[teacher]` (see `backends.toml.example`); env `LOCI_TEACHER_*` wins.

| key | env | default |
|---|---|---|
| `enabled` | `LOCI_TEACHER_ENABLED` | false |
| `[teacher.roles]` judge, label_a, label_b, propose, adjudicate, verify | `LOCI_TEACHER_ROLE_<ROLE>` | none |
| `monthly_usd_cap` | `LOCI_TEACHER_MONTHLY_USD_CAP` | 0 (= no spend) |
| `daily_usd_cap` | `LOCI_TEACHER_DAILY_USD_CAP` | 0 (= no spend) |
| `max_usd_per_call` | `LOCI_TEACHER_MAX_USD_PER_CALL` | 0 (= no spend) |
| `ledger_file` | `LOCI_TEACHER_LEDGER_FILE` | `~/.loci/litellm_cost_ledger.jsonl` |
| `groups_counted` | `LOCI_TEACHER_GROUPS_COUNTED` (csv) | the configured role groups |
| `sanitize` | `LOCI_TEACHER_SANITIZE` | true |
| `[teacher.prices.<group>]` `in_usd_per_mtok`, `out_usd_per_mtok` | `LOCI_TEACHER_PRICES` (`group=in:out,group2=in:out`; replaces the toml table) | none (unpriced group => `no_price`) |
| `chars_per_token` | `LOCI_TEACHER_CHARS_PER_TOKEN` | 3.0 (conservative; real text is nearer 4) |
| `max_bad_lines` | `LOCI_TEACHER_MAX_BAD_LINES` | 0 (any corrupt ledger row refuses) |
| `safety_margin_pct` | `LOCI_TEACHER_SAFETY_MARGIN_PCT` | 10 (usable cap = cap * (100 - pct) / 100; applies to all three caps) |
| `allow_missing_ledger` | `LOCI_TEACHER_ALLOW_MISSING_LEDGER` | false (missing ledger refuses) |
| `settle_seconds` | `LOCI_TEACHER_SETTLE_SECONDS` | 120 |

## Public API (`import teacher`)

```python
ROLES = ("judge", "label_a", "label_b", "propose", "adjudicate", "verify")

def teacher_config() -> dict
def read_ledger(cfg: dict | None = None) -> LedgerTotals        # raises LedgerUnreadable
def check(role: str, est_usd: float) -> Decision                # no budget held
def reserve(role: str, est_usd: float) -> Decision              # allowed => holds est_usd
def release(decision: Decision) -> None                         # idempotent
def teacher_call(role: str, prompt_or_messages: str | list[dict], max_tokens: int,
                 purpose: str, transport=None) -> dict
def status() -> dict
def cache_clear() -> None
```

`Decision` (dataclass, `.to_dict()`): `allowed: bool`, `reason: str`, `role`, `group`, `est_usd`,
`month_spent`, `day_spent`, `in_flight`, `caps: {monthly_usd, daily_usd, per_call_usd}`,
`usable_caps` (caps after the safety margin; the guard compares against these), `ledger_missing: bool`,
`bad_lines: int`, `settling: float`, `reservation: int`.

`reason` is one of: `ok`, `disabled`, `unknown_role`, `role_unmapped`, `no_cap`, `no_price`, `bad_estimate`,
`ledger_missing`, `ledger_corrupt`, `ledger_unreadable`, `ledger_error`, `over_per_call_cap`, `over_daily_cap`, `over_monthly_cap`.

`LedgerTotals`: `month`, `day`, `by_group_month`, `by_group_day`, `missing`, `bad_lines`,
`rows_counted`, `rows_over_call_cap`, `files`.

### teacher_call

Returns `{ok, text, usd, usd_source, cost_unknown, reason, role, group, decision}`.

1. `reserve` (guard + in-process hold, so concurrent callers cannot share one remaining budget).
   The estimate is price-based: `chars/chars_per_token * in_price + max_tokens * out_price` (prices per
   million tokens, from `[teacher.prices]`). It is an upper bound on the call (output cannot exceed
   `max_tokens`), so `max_usd_per_call` is a real per-call bound as long as the prices are right. A
   role whose group has no price is refused (`no_price`); a call whose estimate exceeds the usable
   per-call cap is refused (`over_per_call_cap`).
2. When `sanitize`: redact credentials (`llm_local._sanitize_for_cloud`; unavailable => refused,
   `sanitizer_unavailable`) and emails, IPv4 addresses and home-directory user names. This is not a
   privacy filter: case, finding and memory text goes out as given, so callers choose what may leave.
3. Call the LiteLLM group with metadata `loci_internal_call=true`, `loci_teacher_purpose=<purpose>`,
   so the LiteLLM plugin skips grounding and memory write-back.
4. `usd` is the cost the response reports (`usage.cost`, `response_cost`), else the estimate;
   `usd_source` is `gateway`, `estimate` or `none`. The reservation is always released, but once the
   request was sent its cost (reported, else the estimate) stays counted by the guard for
   `settle_seconds`, covering the gap before the gateway writes its ledger row (conservative: it is
   double counted once the row lands, until the window ends). `transport_error` and `bad_response`
   set `cost_unknown=true`: the gateway may have billed; the estimate is held the same way.
5. A gateway-reported cost above the estimate increments the overrun counter (`status()["overruns"]`):
   the price table is out of date. Ledger rows costing more than `max_usd_per_call` are counted too
   (`ledger_rows_over_call_cap`).

`reason` on failure: a guard reason above, `sanitizer_unavailable`, `transport_error:<Type>`,
`bad_response`, `error:<Type>`.

`transport(group, messages, max_tokens, metadata) -> dict` returns an OpenAI-shaped response. The
default goes through `loci.litellm_client.call_litellm_proxy` (which now accepts `metadata=`).

## Ledger

Rows `{ts, group, cost_usd, ok}`, append-only jsonl, rotated to `<file>.1`, `.2`, ... Read backward
by `ts`, stopping before the start of the local month. Unparseable and negative-cost rows are skipped
but counted (`bad_lines`); when the count exceeds `max_bad_lines` (default 0) the guard refuses with
`ledger_corrupt` (a truncated row would otherwise silently under-count spend). A missing
file reads as zero spend with `ledger_missing=true` but the guard refuses (`ledger_missing`) unless
`allow_missing_ledger` is set; an existing but unreadable one refuses all spend.
Cached 5 s (cleared after each `teacher_call`).

Group names are matched case- and whitespace-insensitively (the gateway does not normalise them),
and a role's group is always counted whatever `groups_counted` says. One stray out-of-order row
(zero or missing `ts`) does not end the backward scan; `OLD_ROWS_TO_STOP` consecutive pre-month rows
do, and bad lines inside such a run are not counted. A `<file>.13` (beyond the 12 rotated siblings
that are read) counts as an unreadable line, so it refuses rather than hiding that history.

## Cost accounting

A gateway-reported cost of `0` is treated as unknown (LiteLLM reports 0 for unmapped or unpriced
models) and the pre-call estimate is held and charged instead; if several fields report a cost the
highest wins. The estimate counts UTF-8 bytes / `chars_per_token` and is taken AFTER sanitizing,
so it prices what is actually sent. Emoji and rare scripts can still cost more tokens than that.

## Limits: the ledger is trusted

The guard sees only what the gateway writes to the ledger. If that file is stalled, rotated away or
points somewhere else, spend beyond the process-local settling hold reads as zero. Check
`python mcp/teacher.py status` against the Console now and then, and keep `max_usd_per_call` and
the caps low until you have.

## Limits: several processes

Reservations and the settling hold live in the memory of one process, and the ledger is cached for
5 s. The CLI, a second Loci process and the MCP server each see only the ledger, so N processes can
jointly overshoot the daily or monthly cap by up to `N x max_usd_per_call` plus the spend that lands
inside the cache window. `safety_margin_pct` is the cushion for this; size it for the number of
processes you run. `status()` repeats this in its `limits` field and reports `overruns`,
`max_bad_lines`, `safety_margin_pct` and `priced_groups`.

## CLI

```
python mcp/teacher.py status          # caps, spent, remaining, by group (JSON)
python mcp/teacher.py check <role>    # guard decision at the usable per-call cap (readiness probe); exit 0 allowed, 1 refused
```

## Hillclimb proposer

`python mcp/hillclimb.py run --suite <s> --proposer teacher [--max-usd N]` proposes with
`teacher_call("propose", ...)`. Default stays the local proposer. Case text leaves the machine only for a
suite whose `<hillclimb dir>/<suite>/config.json` has `"allow_cloud": true` (literal true; `"proposer":
"teacher"` there selects it without the flag). Only failing TRAIN cases are sent, sanitized. A refusal or
error becomes an `aborted_infra` round (reason code only): not scored, not counted against patience,
counted in `aborted_rounds`; `patience` consecutive aborts stop the run (`infra_aborted`). `--max-usd`
stops the run (`budget`) once spend plus the last call's cost reaches it; overshoot is at most one call,
itself capped by the guard. The `query_expand` suite scores against
`<memory parent>/teacher/gold/query_expand.jsonl` (`LOCI_TEACHER_GOLD_DIR`), rows
`{id, query, expect: [terms]}`; no file => stop `suite_unavailable`, status "no gold set yet".
