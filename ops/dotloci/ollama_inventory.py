#!/usr/bin/env python3
"""ollama_inventory.py -- durable record of every Ollama model on every reachable server,
what is loaded, and which specialist roles are covered or missing.

Everything is ON by default: every server found is included, every model on it is listed
(no allowlist), discovery runs each time, history is appended each run. READ-ONLY against
Ollama: only /api/version, /api/tags, /api/ps and /api/show. It never calls generate or
embeddings, so it cannot load a model or evict one.

Outputs, in %USERPROFILE%\\.loci\\ollama_inventory\\ :
  latest.json     full snapshot of this run
  history.jsonl   append-only: one line per run (loaded set + inventory hash + changes)
  MODELS.md       generated, human-readable (models, loaded, specialist coverage, gaps)
  endpoints.json  extra servers/labels you want included (created on first run, editable)
  show_cache.json cache of /api/show by digest (models rarely change; delete to refresh)

Specialist assignments carry their EVIDENCE: measured (a recorded benchmark), capability
(Ollama's own /api/show capabilities) or name (inferred from the model name only).

Usage: ollama_inventory.py [--no-discover] [--quiet]
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, ".loci", "ollama_inventory")
BACKENDS = os.path.join(HOME, ".loci", "backends.toml")
CATALOG_GLOBS = [os.path.join(HOME, "loci-deploy", "artifacts", "model_catalog")]
TAILSCALE = r"C:\Program Files\Tailscale\tailscale.exe"
PORT = 11434
TIMEOUT = 5

# Roles Loci uses (from its own catalog and backends.toml) plus capability roles.
ROLES = [
    ("embedding", "vector embeddings for memory/RAG (Loci embed_model)"),
    ("cheap_fanout", "bulk extraction/classification (Loci catalog role)"),
    ("escalation", "adversarial verification / hard cases (Loci catalog role)"),
    ("guardian", "safety classification (Loci guardian_model)"),
    ("synthesis", "long-context synthesis (Loci catalog role)"),
    ("coding", "code generation and review"),
    ("tool_calling", "function/tool-call specialist"),
    ("math", "mathematical reasoning"),
    ("vision", "image / document understanding"),
    ("reasoning", "thinking-mode reasoning"),
    ("uncensored_redteam", "uncensored / red-team (Loci redteam_model)"),
]
LOCI_MODEL_KEYS = ["gen_model", "verify_model", "compress_model", "guardian_model", "redteam_model"]


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def http(url, data=None, timeout=TIMEOUT):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ------------------------------------------------------------------ endpoints
def seed_endpoints():
    return {"extra": [
        {"label": "windows-ollama", "url": "http://127.0.0.1:11440"},
        {"label": "resident-proxy", "url": "http://127.0.0.1:11435"},
    ], "note": "Add {label,url} entries for servers discovery cannot find. All are included."}


def configured_urls():
    urls = []
    try:
        import tomllib
        with open(BACKENDS, "rb") as f:
            cfg = tomllib.load(f)
        for k in ("url", "gen_url"):
            u = cfg.get("ollama", {}).get(k)
            if u:
                urls.append(u)
    except Exception:
        pass
    if os.environ.get("OLLAMA_HOST"):
        h = os.environ["OLLAMA_HOST"]
        # OLLAMA_HOST is often a BIND address (0.0.0.0, ::), which a client cannot connect to
        h = re.sub(r"^(?:https?://)?(?:0\.0\.0\.0|\[?::\]?)", "127.0.0.1", h)
        urls.append(h if "://" in h else "http://" + h)
    return urls


def discover_peers():
    """Online tailnet peers (any OS) -> candidate urls. Empty if tailscale is unavailable."""
    try:
        out = subprocess.run([TAILSCALE, "status", "--json"], capture_output=True, text=True, timeout=15).stdout
        peers = json.loads(out).get("Peer", {})
    except Exception:
        return []
    res = []
    for p in peers.values():
        if p.get("Online") and p.get("TailscaleIPs"):
            res.append({"label": (p.get("HostName") or p["TailscaleIPs"][0]), "url": "http://%s:%d" % (p["TailscaleIPs"][0], PORT)})
    return res


def candidate_endpoints(discover):
    cands, seen = [], set()

    def add(label, url, explicit):
        url = url.rstrip("/")
        if url not in seen:
            seen.add(url)
            cands.append({"label": label, "url": url, "explicit": explicit})
    ep = load_json(os.path.join(BASE, "endpoints.json"), None)
    if ep is None:
        ep = seed_endpoints()
        atomic_write(os.path.join(BASE, "endpoints.json"), json.dumps(ep, indent=2))
    for e in ep.get("extra", []):
        add(e["label"], e["url"], True)
    for u in configured_urls():
        add("configured:" + u.split("//")[-1], u, True)
    add("localhost", "http://127.0.0.1:%d" % PORT, True)
    if discover:
        for e in discover_peers():
            add(e["label"], e["url"], False)
    return cands


# --------------------------------------------------------------------- probing
def probe(ep):
    out = {"label": ep["label"], "url": ep["url"], "reachable": False, "error": None,
           "version": None, "models": [], "loaded": []}
    try:
        out["version"] = http(ep["url"] + "/api/version").get("version")
        tags = http(ep["url"] + "/api/tags").get("models", [])
        ps = http(ep["url"] + "/api/ps").get("models", [])
    except Exception as e:  # unreachable, 403, not an Ollama: record why, keep going
        out["error"] = re.sub(r"\s+", " ", "%s: %s" % (type(e).__name__, e))[:140]
        return out
    out["reachable"] = True
    out["models"] = tags
    out["loaded"] = [{"name": m.get("name"), "digest": m.get("digest"), "size": m.get("size"),
                      "size_vram": m.get("size_vram"), "expires_at": m.get("expires_at"),
                      "context_length": m.get("context_length")} for m in ps]
    return out


def show(url, name, cache):
    key = name
    return cache.get(key)


def enrich(ep, cache):
    """Add /api/show details per model (cached by digest)."""
    models = []
    for m in ep["models"]:
        digest = m.get("digest", "")
        ent = cache.get(digest)
        if ent is None:
            try:
                s = http(ep["url"] + "/api/show", {"model": m["name"]}, timeout=30)
                mi = s.get("model_info", {}) or {}
                ctx = next((v for k, v in mi.items() if k.endswith(".context_length")), None)
                ent = {"capabilities": s.get("capabilities") or [], "details": s.get("details") or {},
                       "context_length": ctx, "fetched": now()}
                cache[digest] = ent
            except Exception as e:
                ent = {"capabilities": [], "details": {}, "context_length": None,
                       "show_error": str(e)[:100]}
        d = ent.get("details", {})
        models.append({
            "name": m["name"], "digest": digest, "size_bytes": m.get("size"),
            "modified_at": m.get("modified_at"), "family": d.get("family"),
            "parameter_size": d.get("parameter_size"), "quantization": d.get("quantization_level"),
            "context_length": ent.get("context_length"), "capabilities": ent.get("capabilities", []),
            "show_error": ent.get("show_error"),
        })
    ep["models"] = sorted(models, key=lambda x: x["name"])


# ------------------------------------------------------------------ specialists
def load_measured():
    """(model, role) -> {accuracy, winner, total, source} from Loci's recorded benchmark(s)."""
    res = {}
    for d in CATALOG_GLOBS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            data = load_json(os.path.join(d, fn), {})
            for s in data.get("summary", []):
                # quality_winners lists the role's winners on EVERY row; only this row's own
                # quality_rank says whether THIS model won.
                res[(s.get("model"), s.get("role"))] = {
                    "accuracy": s.get("accuracy"), "correct": s.get("correct"), "total": s.get("total"),
                    "winner": s.get("quality_rank") == "winner",
                    "source": fn}
    return res


def infer_roles(m, measured):
    n = m["name"].lower()
    caps = set(m.get("capabilities") or [])
    roles = {}

    def add(role, evidence, basis):
        if role not in roles or evidence == "measured":
            roles[role] = {"role": role, "evidence": evidence, "basis": basis}
    if "embedding" in caps or "embed" in n:
        add("embedding", "capability" if "embedding" in caps else "name", "embedding capability" if "embedding" in caps else "name has 'embed'")
    if "vision" in caps:
        add("vision", "capability", "Ollama capability: vision")
    if "thinking" in caps:
        add("reasoning", "capability", "Ollama capability: thinking")
    if "coder" in n or re.search(r"(^|[-_:/])code", n):
        add("coding", "name", "name has code/coder")     # 'insert' (fill-in-middle) alone is not a code specialist
    if "math" in n:
        add("math", "name", "name has 'math'")
    if "guard" in n:
        add("guardian", "name", "name has 'guard'")
    if "tool" in n or "agent" in n:
        add("tool_calling", "name", "name has tool/agent")
    if any(k in n for k in ("heretic", "uncensored", "abliterat")):
        add("uncensored_redteam", "name", "name has heretic/uncensored/abliterated")
    for (mm, role), r in measured.items():
        if mm == m["name"]:
            add(role, "measured", "%s: %s/%s correct%s" % (r["source"], r["correct"], r["total"], ", WINNER" if r["winner"] else ""))
            roles[role].update({"winner": r["winner"], "correct": r["correct"], "total": r["total"],
                                "accuracy": r["accuracy"]})
    if not roles and "embedding" not in caps:
        add("general", "name", "no specialist signal")
    return list(roles.values())


# --------------------------------------------------------------- config check
def _size_b(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", (s or "").lower())
    return float(m.group(1)) if m else None


def config_check(endpoints, measured):
    out = {"checked": [], "problems": []}
    try:
        import tomllib
        with open(BACKENDS, "rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:
        return {"checked": [], "problems": ["backends.toml unreadable: %s" % e]}
    ol = cfg.get("ollama", {})
    urls = {u.rstrip("/") for u in (ol.get("url"), ol.get("gen_url")) if u}
    target = next((e for e in endpoints if e["url"] in urls and e["reachable"]), None)
    if target is None:
        out["problems"].append("configured Ollama url(s) %s not reachable" % sorted(urls))
        return out
    installed = {m["name"] for m in target["models"]}
    for k in LOCI_MODEL_KEYS:
        name = ol.get(k)
        if not name:
            continue
        ok = name in installed
        out["checked"].append({"key": k, "model": name, "installed": ok, "endpoint": target["label"]})
        if not ok:
            fam = re.split(r"[:/-]", name)[0].lower()
            want = _size_b(name)
            near = []
            for m in target["models"]:
                base = m["name"].split(":")[0].split("/")[-1].lower()
                sz = _size_b(m.get("parameter_size"))
                if base.startswith(fam) and (want is None or (sz and abs(sz - want) <= 0.25 * want)):
                    near.append(m["name"])
            msg = "%s = %r is NOT installed on %s" % (k, name, target["label"])
            if near:
                msg += "; same family and size installed: " + ", ".join(sorted(near)[:4])
            # evidence-based alternative from the recorded benchmark (gen is closest to cheap_fanout)
            if k == "gen_model":
                wins = sorted(((r["accuracy"] or 0, mm) for (mm, role), r in measured.items()
                               if role == "cheap_fanout" and r["winner"] and mm in installed), reverse=True)
                if wins:
                    msg += "; measured cheap_fanout winners installed: " + ", ".join("%s (%.0f%%)" % (mm, a * 100) for a, mm in wins)
            out["problems"].append(msg)
    embed = "nomic-embed-text"
    if not any(m.startswith(embed) for m in installed):
        out["problems"].append("embedding model %s not installed on %s" % (embed, target["label"]))
    return out


# ------------------------------------------------------------------ history
def load_history():
    rows = []
    p = os.path.join(BASE, "history.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def resolve_prev(history):
    """Last history row with each endpoint's model list filled in from the most recent row
    that stored one (rows keep the full list only when it changed)."""
    if not history:
        return None
    res = {"endpoints": {}}
    for lab, e in history[-1].get("endpoints", {}).items():
        e2 = dict(e)
        if "models" not in e2:
            e2["models"] = next((r["endpoints"][lab]["models"] for r in reversed(history)
                                 if "models" in r.get("endpoints", {}).get(lab, {})), [])
        res["endpoints"][lab] = e2
    return res


def inv_hash(ep):
    s = "|".join(sorted("%s@%s" % (m["name"], m["digest"][:12]) for m in ep["models"]))
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def diff_against_previous(snapshot, prev):
    changes = []
    pe = (prev or {}).get("endpoints", {})
    for ep in snapshot["endpoints"]:
        lab = ep["label"]
        old = pe.get(lab)
        if old is None:
            changes.append({"endpoint": lab, "event": "endpoint_first_seen"})
            continue
        if bool(old.get("reachable")) != ep["reachable"]:
            changes.append({"endpoint": lab, "event": "reachable" if ep["reachable"] else "unreachable"})
        if ep["reachable"] and old.get("reachable"):
            oldm = set(old.get("models", []))
            newm = {"%s@%s" % (m["name"], m["digest"][:12]) for m in ep["models"]}
            for x in sorted(newm - oldm):
                changes.append({"endpoint": lab, "event": "model_added_or_changed", "model": x})
            for x in sorted(oldm - newm):
                changes.append({"endpoint": lab, "event": "model_removed_or_changed", "model": x})
            ol, nl = set(old.get("loaded", [])), {m["name"] for m in ep["loaded"]}
            for x in sorted(nl - ol):
                changes.append({"endpoint": lab, "event": "loaded", "model": x})
            for x in sorted(ol - nl):
                changes.append({"endpoint": lab, "event": "unloaded", "model": x})
    return changes


# ------------------------------------------------------------------- render
def gb(b):
    return "%.1f" % (b / 1e9) if b is not None else "?"


def role_txt(r):
    if r["evidence"] == "measured":
        return "%s (measured %s/%s%s)" % (r["role"], r.get("correct"), r.get("total"), ", WINNER" if r.get("winner") else "")
    return "%s (%s)" % (r["role"], r["evidence"])


def render(snapshot, history, cfgcheck, coverage):
    L = []
    L.append("# Ollama model inventory")
    L.append("")
    L.append("Generated %s by `ollama_inventory.py` (read-only; do not edit by hand). Append-only history: `history.jsonl`; full snapshot: `latest.json`." % snapshot["ts"])
    L.append("")
    L.append("## Servers")
    L.append("")
    L.append("| label | url | status | version | models | loaded now |")
    L.append("|---|---|---|---|---:|---|")
    for e in snapshot["endpoints"]:
        st = "up" if e["reachable"] else "no Ollama (%s)" % (e["error"] or "")[:60]
        L.append("| %s | %s | %s | %s | %d | %s |" % (e["label"], e["url"], st, e["version"] or "", len(e["models"]),
                                                  ", ".join(m["name"] for m in e["loaded"]) or "-"))
    L.append("")
    L.append("Note: on this Windows host `127.0.0.1:11434` can reach a DIFFERENT Ollama (WSL, via wslrelay) than the tailnet address. Compare the rows above, not just the port.")
    L.append("")
    seen_loaded = {}
    for row in history:
        for lab, e in row.get("endpoints", {}).items():
            for n in e.get("loaded", []):
                k = (lab, n)
                a = seen_loaded.setdefault(k, {"first": row["ts"], "last": row["ts"], "runs": 0})
                a["last"] = row["ts"]
                a["runs"] += 1
    for e in snapshot["endpoints"]:
        if not e["reachable"]:
            continue
        loaded = {m["name"]: m for m in e["loaded"]}
        L.append("## Models on %s (%d)" % (e["label"], len(e["models"])))
        L.append("")
        L.append("| model | GB | params | quant | ctx | capabilities | specialist roles (evidence) | loaded | last seen loaded |")
        L.append("|---|---:|---|---|---:|---|---|---|---|")
        for m in e["models"]:
            roles = ", ".join(role_txt(r) for r in m["roles"])
            ld = loaded.get(m["name"])
            ldt = "YES (ps size %s GB, vram %s GB)" % (gb(ld.get("size")), gb(ld.get("size_vram"))) if ld else "no"
            sl = seen_loaded.get((e["label"], m["name"]))
            L.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                m["name"], gb(m["size_bytes"]), m["parameter_size"] or "?", m["quantization"] or "?",
                m["context_length"] or "?", ",".join(m["capabilities"]) or "-", roles, ldt,
                (sl["last"][:16] + " (%d runs)" % sl["runs"]) if sl else "never in history"))
        L.append("")
    L.append("## Specialist coverage (all servers)")
    L.append("")
    L.append("| role | what it is for | models | status |")
    L.append("|---|---|---|---|")
    for role, desc in ROLES:
        c = coverage.get(role, [])
        c = sorted(c, key=lambda x: (x["evidence"] != "measured", -(x.get("accuracy") or 0), x["model"]))
        ms = ", ".join("`%s`@%s [%s]" % (x["model"], x["endpoint"],
                       ("measured %s/%s%s" % (x.get("correct"), x.get("total"), " WINNER" if x.get("winner") else ""))
                       if x["evidence"] == "measured" else x["evidence"]) for x in c)
        if any(x["evidence"] == "measured" for x in c):
            st = "covered, measured"
        elif c:
            st = "covered, UNMEASURED"
        else:
            st = "NONE installed"
        L.append("| %s | %s | %s | %s |" % (role, desc, ms or "-", st))
    L.append("")
    srcs = sorted({r["basis"].split(":")[0] for e in snapshot["endpoints"] for m in e["models"] for r in m["roles"] if r["evidence"] == "measured"})
    L.append("")
    L.append("Evidence: **measured** = a recorded Loci benchmark (%s; 15 deterministic cases per role, run against the Windows server on 2026-09-17, so it can be stale if models were re-pulled); **capability** = Ollama's own `/api/show` capabilities; **name** = guessed from the model name only." % (", ".join(srcs) or "none found"))
    L.append("")
    L.append("## Configuration check (Loci backends.toml vs installed)")
    L.append("")
    if cfgcheck["checked"]:
        for c in cfgcheck["checked"]:
            L.append("- %s `%s` on %s: %s" % (c["key"], c["model"], c["endpoint"], "installed" if c["installed"] else "**NOT INSTALLED**"))
    for p in cfgcheck["problems"]:
        L.append("- PROBLEM: " + p)
    L.append("")
    L.append("## Gaps and specialists you may need")
    L.append("")
    gaps = snapshot["gaps"]
    for g in gaps:
        L.append("- " + g)
    if not gaps:
        L.append("- none detected")
    L.append("")
    L.append("## Recent changes (from history)")
    L.append("")
    recent = []
    for row in history[-400:]:
        for c in row.get("changes", []):
            recent.append((row["ts"], c))
    for ts, c in recent[-20:]:
        L.append("- %s %s %s%s" % (ts[:16], c["endpoint"], c["event"], (" " + c["model"]) if c.get("model") else ""))
    if not recent:
        L.append("- none recorded yet")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-discover", action="store_true", help="skip tailnet peer discovery (default: discover)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    os.makedirs(BASE, exist_ok=True)
    cache = load_json(os.path.join(BASE, "show_cache.json"), {})
    history = load_history()
    known = {lab for row in history for lab in row.get("endpoints", {})}
    cands = candidate_endpoints(not a.no_discover)
    explicit = {c["url"]: c["explicit"] for c in cands}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        probed = list(ex.map(probe, cands))
    # collapse addresses that reach the SAME server (same version + same model digests)
    eps, fp = [], {}
    for p in probed:
        if not p["reachable"]:
            noise = bool(p["error"]) and "HTTPError" not in p["error"]
            if noise and not explicit.get(p["url"]) and p["label"] not in known:
                continue        # an unknown discovered peer with nothing listening: not a record
            eps.append(p)       # explicit or previously seen servers are ALWAYS recorded, up or down
            continue
        key = (p["version"], tuple(sorted(m["digest"] for m in p["models"])))
        if key in fp:
            fp[key].setdefault("aliases", []).append(p["url"])
            continue
        fp[key] = p
        eps.append(p)
    for p in eps:
        if p["reachable"]:
            enrich(p, cache)
    measured = load_measured()
    coverage = {}
    for p in eps:
        for m in p.get("models", []):
            m["roles"] = infer_roles(m, measured)
            for r in m["roles"]:
                coverage.setdefault(r["role"], []).append({
                    "model": m["name"], "endpoint": p["label"], "evidence": r["evidence"],
                    "winner": r.get("winner", False), "correct": r.get("correct"), "total": r.get("total"),
                    "accuracy": r.get("accuracy")})
    cfgcheck = config_check(eps, measured)
    gaps = []
    for pr in cfgcheck["problems"]:
        gaps.append("CONFIG: " + pr)
    for role, desc in ROLES:
        c = coverage.get(role, [])
        if not c:
            gaps.append("no installed model for role **%s** (%s)" % (role, desc))
        elif not any(x["evidence"] == "measured" for x in c):
            gaps.append("role **%s** has candidates (%s) but none MEASURED; run Loci's `scripts/bench_model_catalog_quality.py` with them" % (
                role, ", ".join(sorted({x["model"] for x in c}))[:120]))
    emb = sorted({x["model"] for x in coverage.get("embedding", [])})
    if len(emb) == 1:
        ctxs = {m.get("context_length") for p in eps for m in p.get("models", []) if m["name"] == emb[0]}
        gaps.append("only ONE embedding model exists anywhere (%s, context %s): no fallback and no measured baseline; "
                    "a second embedder would be needed to compare or fail over" % (emb[0], "/".join(str(c) for c in sorted(ctxs, key=str))))
    for p in eps:
        if not p["reachable"] and p.get("error"):
            gaps.append("server %s (%s) is not answering as Ollama: %s" % (p["label"], p["url"], p["error"]))
    gaps.append("not an Ollama capability, served elsewhere in this setup: reranker (BAAI/bge-reranker-v2-m3), speech-to-text; check those services separately")
    snapshot = {"ts": now(), "script": os.path.abspath(__file__), "endpoints": [
        {"label": p["label"], "url": p["url"], "aliases": p.get("aliases", []), "reachable": p["reachable"],
         "error": p["error"], "version": p["version"], "inventory_hash": inv_hash(p) if p["reachable"] else None,
         "models": p["models"], "loaded": p["loaded"]} for p in eps],
        "config_check": cfgcheck, "gaps": gaps}
    prev = resolve_prev(history)
    changes = diff_against_previous(snapshot, prev)
    row = {"ts": snapshot["ts"], "changes": changes, "endpoints": {}}
    for e in snapshot["endpoints"]:
        ent = {"reachable": e["reachable"], "version": e["version"], "inventory_hash": e["inventory_hash"],
               "loaded": [m["name"] for m in e["loaded"]]}
        old_hash = ((prev or {}).get("endpoints", {}).get(e["label"], {}) or {}).get("inventory_hash")
        if e["reachable"] and e["inventory_hash"] != old_hash:      # full list only when it changed
            ent["models"] = ["%s@%s" % (m["name"], m["digest"][:12]) for m in e["models"]]
        row["endpoints"][e["label"]] = ent
    with open(os.path.join(BASE, "history.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
    history.append(row)
    atomic_write(os.path.join(BASE, "latest.json"), json.dumps(snapshot, indent=2))
    atomic_write(os.path.join(BASE, "show_cache.json"), json.dumps(cache))
    atomic_write(os.path.join(BASE, "MODELS.md"), render(snapshot, history, cfgcheck, coverage))
    if not a.quiet:
        up = [e for e in snapshot["endpoints"] if e["reachable"]]
        print("%s servers up: %d (%s); models: %d; loaded: %d; changes: %d; gaps: %d" % (
            snapshot["ts"], len(up), ", ".join(e["label"] for e in up),
            sum(len(e["models"]) for e in up), sum(len(e["loaded"]) for e in up), len(changes), len(gaps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
