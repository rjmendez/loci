# Hermes config hardening (Tripwire-Governor S8)

Status: PROPOSED. Nothing in this document has been applied. The live Hermes config
(`C:/Users/rjmendez-admin/AppData/Local/hermes/config.yaml`, Hermes v0.21.5, `_config_version: 49`) was read once to learn its
structure and is unchanged. Apply it later, by hand or by the assistant, with the backup and verify steps below.

The contract is enforced by `scripts/hermes_config_check.py` (read-only; exit 0 compliant, 1 violations, 2 unreadable or
unparsable) and `tests/test_hermes_config_contract.py`.

## What the contract requires

| Key path | Required | Live value (checked 2026-10-18) |
|---|---|---|
| `delegation.max_iterations` | `40` | `250` |
| `delegation.max_concurrent_children` | `2` | absent (commented-out template line, Hermes default 10) |
| `delegation.max_spawn_depth` | `1` | absent (Hermes default 1, but unpinned) |
| `delegation.worktree_isolation` | `true` | absent (Hermes default `false`; the key is real, see `tools/delegate_tool_config.py`) |
| `tool_loop_guardrails.hard_stop_enabled` | `true` | `false` |
| any provider entry or URL | none at `gateway.ai.cloudflare.com`, `api.anthropic.com` or a Copilot host, unless via the LiteLLM `base_url`; a Copilot entry only if `seat_billed: true` | `custom_providers[0]` points at `gateway.ai.cloudflare.com` |

Running the checker on the live file today prints exactly six violations (one per row above) and exits 1. The apply recipe
below, run on a scratch copy of the live file, makes the checker exit 0; the live file was not touched.

The model section already routes through LiteLLM (`model.base_url: https://litellm.lab.floppydicks.net:4443/v1`,
`provider: custom`, `default: auto-router`), so no change is needed there. No `anthropic` or Copilot provider entry exists today.

## Proposed diff

Unified diff against the live structure (line numbers are the live file's). Secrets are redacted: the removed `key_env` line
holds a literal token in the live file (see "Findings" below) and the Cloudflare account id in the URL is shown as
`<ACCOUNT_ID>`. The live file has mixed line endings (CRLF on most lines, LF in the regions edited here); this diff is shown
LF-normalised, so do not feed it to `git apply` or `patch` against the live file. Use the apply recipe below, which keeps
each edited line's own ending.

```diff
--- a/config.yaml
+++ b/config.yaml
@@ -637,7 +637,7 @@
   backend: browser-use
 tool_loop_guardrails:
   warnings_enabled: true
-  hard_stop_enabled: false
+  hard_stop_enabled: true
   non_interactive_hard_stop_enabled: true
   warn_after:
     exact_failure: 2
@@ -1835,7 +1835,10 @@
   # The delegate_task tool spawns child agents with isolated context.
   # Supports single tasks and batch mode (default 3 parallel, configurable).
 delegation:
-  max_iterations: 250 # Max tool-calling turns per child (default: 250)
+  max_iterations: 40 # Max tool-calling turns per child (default: 250)
+  max_concurrent_children: 2
+  max_spawn_depth: 1
+  worktree_isolation: true
   # max_concurrent_children: 10               # Max parallel child agents per batch (default: 10, floor: 1, no ceiling).
   # WARNING: values above 10 multiply API cost linearly.
   # max_spawn_depth: 1                        # Delegation tree depth cap (range: 1-3, default: 1 = flat).
@@ -2414,11 +2417,7 @@
       - C:\Users\rjmendez-admin\AppData\Local\hermes\tools\node-26.7.0-win32-x64\node_modules\rea-agents\scripts\rea.mjs
       - mcp
     enabled: true
-custom_providers:
-  - name: Gateway.ai.cloudflare.com
-    base_url: https://gateway.ai.cloudflare.com/v1/<ACCOUNT_ID>/oxalis-hermes/openai
-    key_env: <REDACTED>
-    model: clef-omni
+custom_providers: []
 approvals:
   mcp_reload_confirm: false
   destructive_slash_confirm: false
```

Why `custom_providers: []` rather than re-pointing the entry (decision 6, "re-point the direct Cloudflare entry through the
gateway"): the main model already goes through the LiteLLM auto-router, so the entry only serves explicit selection of the
`clef-omni` model. Removing it closes the bypass and removes the literal token from the config. If `clef-omni` is still wanted,
the equivalent is a LiteLLM deployment (a Cloudflare gateway arm, decision 8) and a `custom_providers` entry whose
`base_url` is the LiteLLM URL and whose `key_env` is an environment variable NAME, not a token.

## Preconditions (checked or to check before applying)

1. Checked: no other file under the Hermes directory (`*.yaml`, `*.yml`, `*.json`, `*.toml`) mentions `clef-omni` or the
   `Gateway.ai.cloudflare.com` provider name; only `config.yaml` does.
2. Checked: no `clef-omni` in any `*.yaml` under `F:/services/litellm`, so LiteLLM does not currently serve that model name.
   Decide whether the model is still wanted before removing the entry; if yes, add the LiteLLM deployment first.
3. To check: `delegation.max_iterations: 40` is the design's value (design section 4, "250->40"). Confirm one real
   `delegate_task` still completes within 40 turns during the verify step.
4. Hermes treats `delegation.worktree_isolation: true` as opt-in (default false). Each child then gets its own git worktree;
   the delegate presumably needs to run inside a git repository. Read `tools/subagent_worktree.py` for what happens outside
   one (not verified here) before relying on it.

## Apply (later, by the owner or the assistant; restart rule applies)

Back up first, verify after, restore on any failure (owner rule: restarts are fine with backup, verify and rollback).

```powershell
$cfg = 'C:\Users\rjmendez-admin\AppData\Local\hermes\config.yaml'
$bak = "$cfg.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
Copy-Item $cfg $bak            # dated backup; keep it, it still contains the literal token
python scripts\hermes_config_check.py --config $cfg    # expect the 6 violations above, exit 1
```

Then apply the edits with exact, counted replacements (run from a Python 3 shell). This branch ran the recipe on a scratch
copy of the live file only, never on the live file:

```python
import re
from pathlib import Path
p = Path(r"C:\Users\rjmendez-admin\AppData\Local\hermes\config.yaml")
b = p.read_bytes()

def swap(b, old, new):
    # the file has mixed line endings, so match the edit in whichever ending it uses; exactly one hit
    for eol in ("\r\n", "\n"):
        o, n = old.replace("\n", eol).encode(), new.replace("\n", eol).encode()
        if b.count(o) == 1:
            return b.replace(o, n)
    raise AssertionError(old)

b = swap(b, "  hard_stop_enabled: false\n", "  hard_stop_enabled: true\n")
b = swap(b, "delegation:\n  max_iterations: 250 # Max tool-calling turns per child (default: 250)\n",
            "delegation:\n  max_iterations: 40 # Max tool-calling turns per child (default: 250)\n"
            "  max_concurrent_children: 2\n  max_spawn_depth: 1\n  worktree_isolation: true\n")
b, n = re.subn(rb"custom_providers:(\r?\n)(?:(?!approvals:).)*?(?=approvals:)", rb"custom_providers: []\1", b, flags=re.S)
assert n == 1
p.write_bytes(b)
```

Verify before restarting:

```powershell
python scripts\hermes_config_check.py --config $cfg    # expect: Hermes config contract OK, exit 0
```

Restart the Hermes gateway, then confirm per the S8 rollout gate: one chat turn through LiteLLM (`/health` plus one grounded
call) and one `delegate_task` that completes. The `hard_stop_after` thresholds already exist (lines 646-649), so no
warning-only soak is required.

## Rollback

```powershell
Copy-Item $bak $cfg -Force     # restore the dated backup
python scripts\hermes_config_check.py --config $cfg    # confirms the 6 original violations are back (exit 1)
```

Restart the gateway and verify one chat turn.

## Findings that need an owner decision (not changed here)

- The live `config.yaml` stores credentials as literal values: `model.api_key` is a literal LiteLLM key, and
  `custom_providers[0].key_env` holds a literal token where Hermes expects an environment variable NAME. Removing the entry
  drops the second from the live file, but every dated backup keeps it. Treat the token as exposed and rotate it when
  convenient (the token's scope is the Cloudflare gateway account). Moving `model.api_key` to an env reference is a separate
  change.
- `mcp_servers.loci.headers.Authorization` already uses `${WSL_LOCI_MCP_TOKEN}`, which is the pattern to copy.

## Not covered by this change

- Terminal toolset deny list for `claude`, `copilot` and `gh copilot` (S8 acceptance test 3): a search of Hermes v0.21.5
  `tools/*.py` found no command deny-list setting for the terminal tool (only unrelated deny lists in `file_tools.py`,
  `skills_guard.py` and the browser tool), and `terminal:` in the live config has none. It cannot be expressed as a config
  contract today; it needs agentd's per-run permission config plus the vitals orphan detector (design section 5.4(c)).
- `loci/agentd_launch.py` parent Job Object wrapper and the orphan detector: they need agentd, which is a separate branch.
- Cloudflare gateway deployments as LiteLLM arms (S16) and any Copilot seat-billed delegate path (S10).
