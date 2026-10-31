# Defense stack: live path -> tracked file

Every file here is the tracked copy of a live file. `install.ps1` copies them back. The live files are the
ones that run; this repo is the record of what they should be.

| tracked file | live location |
|---|---|
| hermes/hooks/guard_tool.py | AppData\Local\hermes\hooks\guard_tool.py |
| hermes/hooks/claim_audit.py | AppData\Local\hermes\hooks\claim_audit.py |
| hermes/plugins/loci-rag/__init__.py | AppData\Local\hermes\plugins\loci-rag\__init__.py |
| hermes/plugins/loci-rag/test_selfecho.py | AppData\Local\hermes\plugins\loci-rag\test_selfecho.py |
| hermes/plugins/loci-rag/plugin.yaml | AppData\Local\hermes\plugins\loci-rag\plugin.yaml |
| hermes/SOUL.md | AppData\Local\hermes\SOUL.md |
| hermes/config.defense.yaml | template: hooks block + providers entry (secret = placeholder) inside AppData\Local\hermes\config.yaml |
| defense-check/check_defense.py | F:\services\defense-check\check_defense.py |
| defense-check/README.md | F:\services\defense-check\README.md |
| litellm/litellm_service.py | F:\services\litellm\litellm_service.py |
| litellm/governor/envelope.toml | F:\services\litellm\governor\envelope.toml (secret stored as sha256 only) |
| runtime/requirements-loci-server.txt | Loci server Python packages (C:\Python314) |
| tasks/*.xml | Windows scheduled tasks: Defense-Stack-Check, LiteLLM_Proxy, Loci-Watchdog, Ollama_Resident_Proxy, OllamaServe |

Tracked elsewhere (their own git repos, already committed):
- LiteLLM caller policy and memory writer: F:\services\litellm\release\litellm-loci, branch main, commit cef11f1
- Loci provenance write gate: C:\Users\rjmendez-admin\loci-deploy, branch hardening-20261030, commit 230640ba

Not tracked, on purpose (secrets):
- AppData\Local\litellm\.env (LITELLM_MASTER_KEY, provider keys, ABLITERATION_AI_KEY)
- AppData\Local\hermes\.env, and the full AppData\Local\hermes\config.yaml (contains api_key values)
- AppData\Local\hermes\shell-hooks-allowlist.json (consent record; regenerate with `hermes --accept-hooks`)
- The Hermes client secret (lcs_*): owner secret store, referenced as ${LOCI_CLIENT_SECRET}
