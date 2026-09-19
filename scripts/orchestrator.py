#!/usr/bin/env python3
"""
GPU-aware batch orchestrator for local LLM (Ollama) jobs — an Airflow-lite scheduler.

Monitors per-GPU VRAM/util, admits queued jobs only when their target endpoint's
affine GPU(s) have headroom (accounting for models already resident = zero marginal
cost), dispatches concurrently, supports DAG dependencies + retries, captures each
job's output, and logs a status table + heartbeat every tick.

Design goal (restoring a lost Loci feature): keep both physical GPUs busy without
oversubscribing VRAM, across engines pinned to different cards.

Usage:
  python3 orchestrator.py --jobs jobs.jsonl --out out/ [--poll 4] [--once]

Job record (one JSON object per line in --jobs):
  {
    "id": "dsp-reflash",             # unique
    "endpoint": "multi",             # key into ENDPOINTS (or omit -> any that fits)
    "model": "heretic-gemma3-4b-it:latest",
    "est_vram_gb": 3.5,              # marginal VRAM if not already resident
    "prompt": "....",               # or "prompt_file": "path"
    "system": "optional system prompt",
    "options": {"temperature": 0.4, "num_predict": 1024, "num_ctx": 8192},
    "depends_on": ["other-id"],     # optional DAG edges
    "priority": 0                    # lower = sooner
  }
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request

# --- topology: endpoints and the physical GPU indices each one can place models on ---
# Nothing environment-specific is hardcoded. URLs, the nvidia-smi path, and the VRAM
# slack come from the environment with generic localhost defaults (see the README):
#   GPU_ENDPOINT_MULTI - an Ollama endpoint that may span several cards (auto-splits)
#   GPU_ENDPOINT_GPU0  - an Ollama endpoint pinned to GPU0 (CUDA_VISIBLE_DEVICES=...)
#   NVIDIA_SMI_PATH    - path to nvidia-smi (default: found on PATH)
#   VRAM_GUARD_GB      - VRAM kept free per GPU as slack (default 0.8)
ENDPOINTS = {
    "multi": {"url": os.environ.get("GPU_ENDPOINT_MULTI", "http://127.0.0.1:11434"),
              "gpus": [0, 1], "max_inflight": int(os.environ.get("GPU_MULTI_MAX_INFLIGHT", "3"))},
    "gpu0":  {"url": os.environ.get("GPU_ENDPOINT_GPU0", "http://127.0.0.1:11435"),
              "gpus": [0], "max_inflight": int(os.environ.get("GPU_GPU0_MAX_INFLIGHT", "2"))},
}
NVIDIA_SMI = os.environ.get("NVIDIA_SMI_PATH", "nvidia-smi")
VRAM_GUARD_GB = float(os.environ.get("VRAM_GUARD_GB", "0.8"))  # keep this much free per GPU


def gpu_free_gb():
    """Return {gpu_index: free_MiB/1024} from nvidia-smi."""
    try:
        out = subprocess.check_output(
            [NVIDIA_SMI, "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception as e:
        print(f"[warn] nvidia-smi failed: {e}", file=sys.stderr)
        return {}
    free = {}
    for line in out.strip().splitlines():
        idx, mib = [x.strip() for x in line.split(",")]
        free[int(idx)] = int(mib) / 1024.0
    return free


def resident_models(url):
    """Set of model names currently loaded on an endpoint (zero marginal VRAM)."""
    try:
        with urllib.request.urlopen(url + "/api/ps", timeout=6) as r:
            data = json.load(r)
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def unload(url, model):
    """Ask an endpoint to evict a model now (free VRAM)."""
    try:
        req = urllib.request.Request(
            url + "/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0, "prompt": ""}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
        return True
    except Exception:
        return False


def endpoint_reachable(url):
    try:
        with urllib.request.urlopen(url + "/api/version", timeout=4) as r:
            json.load(r)
        return True
    except Exception:
        return False


def run_job(job, url, results, lock):
    """Blocking HTTP call to /api/generate; store result."""
    body = {
        "model": job["model"],
        "prompt": job["_prompt"],
        "stream": False,
        "keep_alive": job.get("keep_alive", "5m"),
        "options": job.get("options", {}),
    }
    if job.get("system"):
        body["system"] = job["system"]
    t0 = time.time()
    try:
        req = urllib.request.Request(
            url + "/api/generate",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=job.get("timeout", 900)) as r:
            resp = json.load(r)
        with lock:
            results[job["id"]] = {
                "status": "done", "endpoint_url": url, "secs": round(time.time() - t0, 1),
                "response": resp.get("response", ""),
                "eval_count": resp.get("eval_count"),
                "prompt_eval_count": resp.get("prompt_eval_count"),
            }
    except Exception as e:
        with lock:
            results[job["id"]] = {"status": "failed", "endpoint_url": url,
                                  "secs": round(time.time() - t0, 1), "error": str(e)}


def load_jobs(path):
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            j = json.loads(line)
            if "prompt_file" in j and "prompt" not in j:
                with open(j["prompt_file"]) as pf:
                    j["_prompt"] = pf.read()
            else:
                j["_prompt"] = j.get("prompt", "")
            j.setdefault("priority", 0)
            j.setdefault("depends_on", [])
            j.setdefault("est_vram_gb", 5.0)
            jobs.append(j)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--poll", type=float, default=4.0)
    ap.add_argument("--heartbeat", default="orchestrator.heartbeat")
    ap.add_argument("--once", action="store_true", help="exit when queue drains")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = {j["id"]: j for j in load_jobs(args.jobs)}
    state = {jid: "queued" for jid in jobs}          # queued|running|done|failed
    results, lock = {}, threading.Lock()
    threads = {}                                      # jid -> Thread
    inflight = {ep: 0 for ep in ENDPOINTS}
    attempts = {jid: 0 for jid in jobs}

    # only consider endpoints that answer right now
    live_eps = {ep: c for ep, c in ENDPOINTS.items() if endpoint_reachable(c["url"])}
    print(f"[init] {len(jobs)} jobs; live endpoints: {list(live_eps) or 'NONE'}")
    if not live_eps:
        print("[fatal] no reachable Ollama endpoint", file=sys.stderr)
        sys.exit(1)

    def deps_done(j):
        return all(state.get(d) == "done" for d in j["depends_on"])

    tick = 0
    while True:
        tick += 1
        # reap finished threads
        for jid, th in list(threads.items()):
            if not th.is_alive():
                th.join()
                res = results.get(jid, {"status": "failed", "error": "no result"})
                if res["status"] == "done":
                    state[jid] = "done"
                    with open(os.path.join(args.out, jid + ".json"), "w") as f:
                        json.dump({**{k: v for k, v in jobs[jid].items() if k != "_prompt"}, **res}, f, indent=2)
                else:
                    if attempts[jid] < 2:
                        state[jid] = "queued"      # retry
                        print(f"[retry] {jid} (attempt {attempts[jid]+1}) {res.get('error','')[:80]}")
                    else:
                        state[jid] = "failed"
                        with open(os.path.join(args.out, jid + ".json"), "w") as f:
                            json.dump({**{k: v for k, v in jobs[jid].items() if k != "_prompt"}, **res}, f, indent=2)
                inflight[jobs[jid]["_ep"]] -= 1
                # reclaim VRAM: if jobs are still queued and none needs this model, evict it
                fin_model = jobs[jid]["model"]
                still_queued = [j for k, j in jobs.items() if state[k] == "queued"]
                if still_queued and not any(j["model"] == fin_model for j in still_queued):
                    ep_url = ENDPOINTS[jobs[jid]["_ep"]]["url"]
                    if unload(ep_url, fin_model):
                        print(f"[reclaim] unloaded {fin_model} on {jobs[jid]['_ep']} (queue waiting on VRAM)")
                del threads[jid]

        free = gpu_free_gb()
        res_cache = {ep: resident_models(c["url"]) for ep, c in live_eps.items()}

        # admission: schedule queued jobs whose deps are met and VRAM fits
        for jid, j in sorted(jobs.items(), key=lambda kv: kv[1]["priority"]):
            if state[jid] != "queued" or not deps_done(j):
                continue
            cands = [j["endpoint"]] if j.get("endpoint") in live_eps else list(live_eps)
            for ep in cands:
                cfg = live_eps[ep]
                if inflight[ep] >= cfg["max_inflight"]:
                    continue
                resident = j["model"] in res_cache.get(ep, set())
                need = 0.0 if resident else j["est_vram_gb"]
                # Ollama places a small model on the best-fitting single card and only
                # splits one too big for a card; admit against TOTAL free headroom.
                gpus = cfg["gpus"]
                total_free = sum(free.get(g, 0.0) for g in gpus)
                best = max(gpus, key=lambda g: free.get(g, 0.0)) if gpus else None
                if resident or need + VRAM_GUARD_GB <= total_free:
                    j["_ep"] = ep
                    attempts[jid] += 1
                    state[jid] = "running"
                    inflight[ep] += 1
                    th = threading.Thread(target=run_job, args=(j, cfg["url"], results, lock), daemon=True)
                    th.start(); threads[jid] = th
                    # reserve VRAM: single-card if it fits the best card, else spill
                    if not resident and best is not None:
                        if need <= free.get(best, 0.0):
                            free[best] -= need
                        else:
                            for g in gpus:
                                free[g] = max(0.0, free.get(g, 0.0) - need / len(gpus))
                    print(f"[dispatch] {jid} -> {ep}({j['model']}) resident={resident} need={need:.1f}GB")
                    break

        # status line + heartbeat
        counts = {s: sum(1 for v in state.values() if v == s) for s in ("queued", "running", "done", "failed")}
        gpustr = " ".join(f"g{g}:{free.get(g,0):.1f}GBfree" for g in sorted(set(sum((c['gpus'] for c in live_eps.values()), []))))
        line = f"[tick {tick}] {counts} | inflight={inflight} | {gpustr}"
        print(line, flush=True)
        with open(args.heartbeat, "w") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")

        if counts["queued"] == 0 and counts["running"] == 0:
            print(f"[done] all jobs settled: {counts}")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
