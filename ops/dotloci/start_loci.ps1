# Start the live Loci server (detached, hidden). Stops whatever listens on 127.0.0.1:8000 first.
# Usage: powershell -NoProfile -File start_loci.ps1 [-NoWait]
# Env below is the live configuration; keep it here so a restart cannot drop a setting
# (omitting LOCI_MCP_TRANSPORT starts a stdio server that never listens).
param([switch]$NoWait)
$d = 'C:\Users\rjmendez-admin\.loci'
$env:LOCI_MCP_TRANSPORT = 'streamable-http'
$env:PYTHONPATH = 'C:\Users\rjmendez-admin\loci-deploy'   # 2026-10-18: repo root, so the loci library package resolves to THIS checkout; a global editable install points it at another worktree
$env:LOCI_MCP_HOST = '127.0.0.1'
$env:LOCI_MCP_PORT = '8000'
$env:LOCI_TOOL_WORKERS = '4'
# 2026-10-14: a stray default Ollama (0 models) listens on :11434 and wins the local probe; embeddings need the 11435 resident proxy.
$env:LOCI_LOCAL_OLLAMA = 'http://127.0.0.1:11435'
$env:LOCI_LOG_FILE = "$d\loci-server.rot.log"
$env:HERMES_AGENT_ID = 'rjmendez'          # this server's identity: recorded as owner of new investigations, checked by the owner-only rules, stamped on new index points
$env:LOCI_MODEL_POOL_SHADOW = '1'     # log pool decisions and call outcomes (names, ranks, enums only)
# Exploration, gen role only (owner 2026-10-05): 10% of gen picks go to another eligible model that is ALREADY RESIDENT in Ollama
# (resident-only is the default since #463; keep a second model warm with `python mcp/model_pool.py warm`). Re-enabled 2026-10-06.
# No model list: the pool decides what is available. Propensity is logged per decision. Remove both lines to stop;
# `python mcp/model_pool.py report` shows the arms.
$env:LOCI_MODEL_POOL_EXPLORE = '0.10'
$env:LOCI_MODEL_POOL_EXPLORE_ROLES = 'gen'
$env:LOCI_MODEL_POOL_GRADE = '0.05'   # judge-grade ~5% of live classify/compress answers (pool_grader; judge is lenient, rank signal only)
$env:LOCI_D10_SHADOW = '1'            # score (question, finding) pairs with the trained pair gate beside the cosine gate; log only
$env:LOCI_ROUTE_TRACE_AUDIT_RATE = '0.05'   # sample 5% of memory_route calls into the audit log for counterfactual replay
$env:LOCI_SWARM_SAFETY_CHECK = '1'    # advisory Granite Guardian annotation on swarm summaries; never blocks or rewrites
$env:MEMCHECK_LLM_CONTRADICTION = '1' # local-LLM contradiction pass in memory_self_check; advisory, fails open
$env:LOCI_LAUNCH_DUMP_S = '3000'
# 2026-10-18: Loci's own LiteLLM calls (litellm_call, litellm_health) need a key the proxy accepts. Load the proxy's master key
# from the proxy's own env file straight into this process; the value is never echoed or copied anywhere else.
$lk = 'C:\Users\rjmendez-admin\AppData\Local\litellm\.env'
if (Test-Path $lk) {
    $m = Select-String -Path $lk -Pattern '^\s*LITELLM_MASTER_KEY\s*=\s*(.+?)\s*$' | Select-Object -First 1
    if ($m) { $env:LOCI_LITELLM_API_KEY = $m.Matches[0].Groups[1].Value.Trim('"', "'") }
}
# 2026-10-18: litellm_call must name models the LiteLLM proxy serves, i.e. its own model group names (config.yaml model_name),
# not the model pool's Ollama picks. This map is for litellm_call only; LOCI_TIER_MODEL_MAP (every other tier lookup) is untouched.
# 2026-11-17: owner 'simple models for simple tasks, use the credits': standard/medium go to the claude-direct-haiku group (Anthropic credits)
# instead of medium-tier, which LiteLLM load-balances onto a slow local Ollama model (ledger: 43 s at 15k tokens, intermittent empty replies).
$env:LOCI_LITELLM_TIER_MAP = '{"nano":["simple-tier"],"light":["simple-tier"],"standard":["claude-direct-haiku"],"medium":["claude-direct-haiku"],"turbo":["complex-tier"],"heavy":["complex-tier"],"research":["complex-tier"]}'
# 2026-11-17: litellm_call defaulted to a 30 s client timeout; Sonnet 5 (heavy/research tiers) needs ~50 s, so answers were cut off while the proxy still billed them.
$env:LOCI_LITELLM_TIMEOUT_S = '150'
# 2026-11-06: judge oracle on. Sampled pool answers are graded by the teacher 'judge' role (Haiku) through teacher.py under its caps;
# observe-only (gates nothing). Off switch: remove this line or set [teacher] enabled=false.
$env:LOCI_JUDGE_ORACLE = '1'
# 2026-10-20: rerank on the GPU. Loci's own torch is CPU-only (20 s budget cut 779 of 835 searches). Sidecar: task
# Loci_Rerank_Sidecar -> F:\venvs\rerank, scripts\rerank_server.py on the 2080 Ti. If it is down, mcp/reranker.py fails open.
$env:RERANK_HTTP_URL = 'http://127.0.0.1:8765/rerank'

$old = (Get-NetTCPConnection -LocalPort 8000 -LocalAddress 127.0.0.1 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($old) { Stop-Process -Id $old -Force; Start-Sleep 3 }
Start-Process -FilePath 'C:\Python314\python.exe' -ArgumentList "$d\loci_launch.py" `
    -WorkingDirectory 'C:\Users\rjmendez-admin\loci-deploy\mcp' -WindowStyle Hidden `
    -RedirectStandardOutput "$d\loci-server.out.log" -RedirectStandardError "$d\loci-server.err.log" | Out-Null
if ($NoWait) { return }
for ($i = 0; $i -lt 50; $i++) {
    Start-Sleep 3
    try { if ((Invoke-WebRequest http://127.0.0.1:8000/health -TimeoutSec 4 -UseBasicParsing).StatusCode -eq 200) { 'up'; return } } catch {}
}
'FAILED to answer /health in 150s'; exit 1
