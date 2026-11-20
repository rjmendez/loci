# Defense stack: what runs, where it lives, how it is kept

Scheduled task `Defense-Stack-Check`: at logon and every 30 minutes. Runs `check_defense.py`, writes
`status.json` (latest) and `status.log` (one line per run). Exit 1 and `"ok": false` when a layer is not live.

## Layers (all persistent, all checked)

| layer | code / config on disk | running copy checked against |
|---|---|---|
| LiteLLM caller policy (verified-hermes rules, recall forcing, act-don't-talk) | release repo `main`: `loci_litellm/caller_policy.py` (repo `F:\services\litellm\release\litellm-loci`); prod config `AppData\Local\litellm\config.yaml` callback list | proxy on :4000 started after `caller_policy.py` last changed |
| LiteLLM governor (spend caps, enforce) | envelope `F:\services\litellm\governor\envelope.toml` + `envelope.pin`; wrapper `F:\services\litellm\litellm_service.py` (`LOCI_GOVERNOR=enforce`) | governor audit: `F:\services\litellm\governor\audit` |
| Hermes pre-tool guard (path, malformed args, Linux-only commands) | `AppData\Local\hermes\hooks\guard_tool.py`; `config.yaml` `hooks:`; consent in `shell-hooks-allowlist.json` | config + allowlist entries present |
| Hermes persona rules | `AppData\Local\hermes\SOUL.md` (backup `SOUL.md.bak-20261031`) | soft, not enforced |
| Loci provenance write gate | `loci-deploy` branch `hardening-20261030`, `mcp/server.py` `_resolve_write_provenance` | Loci server (newest `loci_launch.py`) started after `server.py` last changed |
| Loci mnemosyne mirror | `mnemosyne-memory` 4.0.0 + `sqlite-vec` in `C:\Python314` | `memory_health` mnemo_mirror ok |

## Restart recipes

- LiteLLM: `Stop-ScheduledTask LiteLLM_Proxy`, kill `run_server()` / `litellm_service.py` processes, `Start-ScheduledTask LiteLLM_Proxy` (task runs at system startup).
- Loci: kill the newest `loci_launch.py` process; `Loci-Watchdog` respawns it in about 40-90 s. The watchdog runs only in an interactive session (logon trigger plus every minute), so Loci is down from boot until logon.
- Hermes guard: register consent with `hermes --accept-hooks` (scoped to the event and command).

## Known gaps (checked, not fixed here)

- Hermes does not send `x-loci-client` identity headers on real turns. The identity-gated rules (verified-hermes policy, recall forcing) are inactive for Hermes until that is fixed. Ungated rules (model/args checks are gated; promise rule is not) still apply.
- Loci full test suite: 26 pre-existing failures on the committed baseline (Windows-specific: daemon sockets, file locks, GPU/cuDNN). Tests that stored `tool_verified` findings now cite real receipts.
