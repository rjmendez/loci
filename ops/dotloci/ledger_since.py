import json,sys,collections,datetime as dt
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
start=dt.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00")); 
if start.tzinfo is None: start=start.replace(tzinfo=dt.timezone.utc)
R=[json.loads(l) for l in open(r"C:\Users\rjmendez-admin\.loci\litellm_cost_ledger.jsonl",encoding="utf-8",errors="replace") if l.strip()]
def w(r):
    v=r["ts"]; return dt.datetime.fromtimestamp(v,dt.timezone.utc) if isinstance(v,(int,float)) else dt.datetime.fromisoformat(str(v).replace("Z","+00:00"))
rec=[r for r in R if w(r)>=start and r.get("call_type")=="acompletion"]
print(f"since restart: {len(rec)} completions; now {dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}Z")
for r in rec[-14:]:
    print(f"  {w(r).strftime('%H:%M:%S')} {str(r.get('deployment'))[:20]:20} prompt={r.get('prompt_tokens'):>6} read={r.get('cache_read_tokens')} write={r.get('cache_write_tokens')} internal={r.get('internal')} stream={r.get('stream')} ground={r.get('grounding')} cost={r.get('cost_usd')}")
