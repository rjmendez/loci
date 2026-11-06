# Teacher: budget-guarded cloud calls

`mcp/teacher.py` is the one door for learning loops (judge, labelers, proposer, adjudicator,
verifier) to the cloud. Off by default. The guard fails CLOSED; callers fail OPEN: `teacher_call`
never raises, a refusal or error is a result the feature degrades on.

Prices are not in Loci. Spend is read from the LiteLLM cost ledger. Role -> LiteLLM group names
come from config only, read at call time.

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
| `est_ref_tokens` | `LOCI_TEACHER_EST_REF_TOKENS` | 4000 |

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
`ledger_missing: bool`, `reservation: int`.

`reason` is one of: `ok`, `disabled`, `unknown_role`, `role_unmapped`, `no_cap`, `bad_estimate`,
`ledger_unreadable`, `ledger_error`, `over_per_call_cap`, `over_daily_cap`, `over_monthly_cap`.

`LedgerTotals`: `month`, `day`, `by_group_month`, `by_group_day`, `missing`, `bad_lines`,
`rows_counted`, `files`.

### teacher_call

Returns `{ok, text, usd, usd_source, reason, role, group, decision}`.

1. `reserve` (guard + in-process hold, so concurrent callers cannot share one remaining budget).
   The estimate is `max_usd_per_call * (chars/3 + max_tokens) / est_ref_tokens`; a request bigger
   than the reference size therefore exceeds the per-call cap and is refused.
2. Sanitize each message with `llm_local._sanitize_for_cloud` when `sanitize` (unavailable => refused,
   `sanitizer_unavailable`).
3. Call the LiteLLM group with metadata `loci_internal_call=true`, `loci_teacher_purpose=<purpose>`,
   so the LiteLLM plugin skips grounding and memory write-back.
4. `usd` is the cost the response reports (`usage.cost`, `response_cost`), else the estimate;
   `usd_source` is `gateway`, `estimate` or `none`. The reservation is always released; the ledger
   carries real spend.

`reason` on failure: a guard reason above, `sanitizer_unavailable`, `transport_error:<Type>`,
`bad_response`, `error:<Type>`.

`transport(group, messages, max_tokens, metadata) -> dict` returns an OpenAI-shaped response. The
default goes through `loci.litellm_client.call_litellm_proxy` (which now accepts `metadata=`).

## Ledger

Rows `{ts, group, cost_usd, ok}`, append-only jsonl, rotated to `<file>.1`, `.2`, ... Read backward
by `ts`, stopping before the start of the local month; bad lines are skipped and counted. A missing
file is zero spend with `ledger_missing=true`; an existing but unreadable one refuses all spend.
Cached 5 s (cleared after each `teacher_call`).

## CLI

```
python mcp/teacher.py status          # caps, spent, remaining, by group (JSON)
python mcp/teacher.py check <role>    # guard decision at max_usd_per_call; exit 0 allowed, 1 refused
```
