"""Isolated two-node A2A rig: starts node A (:18201) and node B (:18202) with temp state, runs the scenarios."""
import json, os, subprocess, sys, time, shutil, tempfile, uuid
import httpx, pyotp

PY = sys.executable
SRV = os.environ.get("A2A_SRV", r"C:\Users\rjmendez-admin\loci-deploy\a2a_server\server.py")
ROOT = tempfile.mkdtemp(prefix="a2a_rig_", dir=os.environ["CLAUDE_JOB_DIR"] + "/tmp")
SEED_B = pyotp.random_base32()
BOOT_B = "boot-" + uuid.uuid4().hex
TOK = {k: "t-" + uuid.uuid4().hex for k in ("a_primary", "b_primary", "a_on_b", "other_on_b")}
URL_A, URL_B = "http://127.0.0.1:18201", "http://127.0.0.1:18202"


def env_for(name, port, extra):
    d = os.path.join(ROOT, name); os.makedirs(d, exist_ok=True)
    e = {k: v for k, v in os.environ.items() if not k.startswith(("QDRANT", "PEER_A2A", "LOCI_A2A", "HERMES", "MNEMOSYNE"))}
    e.update(HOME=d, USERPROFILE=d, LOCI_ENV_FILE=os.path.join(d, "none.env"), LOCI_A2A_HOST="127.0.0.1",
             LOCI_A2A_PORT=str(port), LOCI_A2A_URL=f"http://127.0.0.1:{port}", HERMES_AGENT_ID=name,
             MNEMOSYNE_DATA_DIR=os.path.join(d, "mnem"), LOCI_MEMORY_DIR=os.path.join(d, "mem"),
             PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    e.update(extra)
    os.makedirs(e["MNEMOSYNE_DATA_DIR"], exist_ok=True)
    import sqlite3
    c = sqlite3.connect(os.path.join(e["MNEMOSYNE_DATA_DIR"], "mnemosyne.db"))
    c.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT, session_id TEXT, importance REAL, metadata_json TEXT, created_at TEXT)")
    c.commit(); c.close()
    return e, d


def start(name, port, extra):
    e, d = env_for(name, port, extra)
    log = open(os.path.join(d, "server.log"), "w")
    p = subprocess.Popen([PY, SRV], env=e, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(SRV))
    return p, os.path.join(d, "server.log")


def wait(url):
    for _ in range(80):
        try:
            r = httpx.get(url + "/health", timeout=3)
            if r.status_code == 200:
                return True
            print("health", url, r.status_code, r.text[:120])
        except Exception as e:
            print("health err", url, repr(e)[:100])
        time.sleep(0.5)
    return False


def rpc(url, token, skill, inp=None, sender=None, totp=None, message="", method="tasks/send"):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if totp:
        h["X-TOTP"] = totp
    params = {"skill_id": skill, "message": message, "input": inp or {}}
    if sender:
        params["sender"] = sender
    r = httpx.post(url + "/a2a", json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params}, headers=h, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:200]


results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def totp_b():
    return pyotp.TOTP(SEED_B).now()


def stored(url_token_totp, text):
    """True if `text` is in the returned memory contents (not merely echoed in the query field)."""
    url, tok, tp = url_token_totp
    st, b = rpc(url, tok, "memory_recall", {"query": text, "top_k": 10, "semantic": False}, totp=tp)
    assert st == 200, (st, b)
    return any(text in (m.get("content") or "") for m in ((b.get("result") or {}).get("output") or {}).get("memories", []))


def output(body):
    return (body.get("result") or {}).get("output") if isinstance(body, dict) else None


procs = []


VER = {"tok": None}


def verifier_token():
    r = httpx.post(URL_B + "/bootstrap", json={"bootstrap_key": BOOT_B, "agent_id": "verifier"}, timeout=10)
    VER["tok"] = r.json()["session_token"]


def restart_b():
    global pb
    pb.terminate(); pb.wait(10)
    pb, _ = start("node-b", 18202, B_ENV); procs.append(pb)
    assert wait(URL_B), "B did not come back"
    verifier_token()


def b_has(text, n_only=False):
    st, b = rpc(URL_B, VER["tok"], "memory_recall", {"query": text, "top_k": 20, "semantic": False}, sender="verifier")
    assert st == 200, (st, b)
    ms = [m for m in ((b.get("result") or {}).get("output") or {}).get("memories", []) if text in (m.get("content") or "")]
    return len(ms)


try:
    B_ENV = dict(LOCI_A2A_TOKEN=TOK["b_primary"], LOCI_A2A_AGENT_TOKENS=json.dumps({"node-a": TOK["a_on_b"], "someone-else": TOK["other_on_b"]}),
                 LOCI_A2A_TOTP_SEED=SEED_B, LOCI_A2A_BOOTSTRAP_KEY=BOOT_B)
    A_BASE = dict(LOCI_A2A_TOKEN=TOK["a_primary"], PEER_A2A_URLS=URL_B + "/a2a")
    pb, logb = start("node-b", 18202, B_ENV); procs.append(pb)
    pa, loga = start("node-a", 18201, A_BASE); procs.append(pa)
    check("both nodes come up and answer /health", wait(URL_A) and wait(URL_B))

    # ---- B's front door ----
    s, _ = rpc(URL_B, None, "memory_stats"); check("B: no bearer token is refused", s in (401, 403), s)
    s, _ = rpc(URL_B, "wrong-token", "memory_stats"); check("B: wrong token is refused", s == 401, s)
    s, _ = rpc(URL_B, TOK["b_primary"], "memory_stats"); check("B: valid bearer without TOTP is refused (TOTP is on)", s in (401, 403), s)
    s, b = rpc(URL_B, TOK["b_primary"], "memory_stats", totp=totp_b()); check("B: primary bearer + valid TOTP is accepted", s == 200 and "error" not in (b if isinstance(b, dict) else {}), f"{s} {str(b)[:100]}")
    s, _ = rpc(URL_B, TOK["b_primary"], "memory_stats", totp="000000"); check("B: valid bearer + wrong TOTP is refused", s in (401, 403), s)

    # ---- identity binding of per-agent tokens ----
    s, b = rpc(URL_B, TOK["a_on_b"], "memory_stats", sender="node-a", totp=totp_b()); check("B: per-agent token as its own agent is accepted", s == 200, s)
    s, b = rpc(URL_B, TOK["a_on_b"], "memory_stats", sender="someone-else", totp=totp_b()); check("B: per-agent token cannot claim another sender", s == 403, s)
    s, b = rpc(URL_B, TOK["other_on_b"], "memory_remember", {"content": "should-not-store"}, sender="someone-else", totp=totp_b())
    check("B: an unprivileged agent token cannot call a destructive skill (memory_remember)", (s in (401, 403)) or "error" in json.dumps(b).lower() or output(b) is None, f"{s} {str(b)[:140]}")

    # ---- bootstrap ----
    s, b = httpx.post(URL_B + "/bootstrap", json={"bootstrap_key": "nope", "agent_id": "node-x"}, timeout=10).status_code, None
    check("B: /bootstrap with a wrong key is refused", s in (401, 403), s)
    r = httpx.post(URL_B + "/bootstrap", json={"bootstrap_key": BOOT_B, "agent_id": "node-x"}, timeout=10)
    sess = None
    try:
        sess = r.json().get("session_token")
    except Exception:
        pass
    check("B: /bootstrap with the right key issues a session token", r.status_code == 200 and bool(sess), f"{r.status_code} totp_required={r.json().get("totp_required") if r.status_code==200 else r.text[:80]}")
    if sess:
        s, b = rpc(URL_B, sess, "memory_stats", sender="node-x"); check("B: the session token works as a standalone bearer", s == 200, s)
        s, b = rpc(URL_B, sess, "memory_stats", sender="node-a"); check("B: the session token is bound to its agent_id", s == 403, s)

    # ---- A -> B broadcast, escalating the config ----
    restart_b()
    marker = "a2a-rig-" + uuid.uuid4().hex[:10]
    s, b = rpc(URL_A, TOK["a_primary"], "context_broadcast", {"content": f"{marker} unprivileged"}, sender="node-a")
    check("A: context_broadcast is refused until the sender is in LOCI_A2A_PRIVILEGED_SENDERS", s == 403 and "elevated privilege" in json.dumps(b), f"{s} {str(b)[:120]}")
    pa.terminate(); pa.wait(10)
    A_BASE["LOCI_A2A_PRIVILEGED_SENDERS"] = "node-a"
    pa, loga = start("node-a", 18201, A_BASE); procs.append(pa); wait(URL_A)
    s, b = rpc(URL_A, TOK["a_primary"], "context_broadcast", {"content": f"{marker} no-creds"}, sender="node-a")
    o = output(b) or {}
    st = [x.get("status") for x in o.get("broadcast", [])]
    check("A->B with NO peer credentials is skipped, not sent", st == ["skipped"], f"{s} {st} {[x.get('error') for x in o.get('broadcast', [])]}")
    check("  (and A still stored the memory locally)", o.get("stored_locally") is True, o.get("stored_locally"))

    pa.terminate(); pa.wait(10)
    pa, loga = start("node-a", 18201, dict(A_BASE, PEER_A2A_TOKENS_JSON=json.dumps({URL_B + "/a2a": "wrong-token"}), PEER_A2A_TOTP_SEED=SEED_B)); procs.append(pa)
    wait(URL_A)
    s, b = rpc(URL_A, TOK["a_primary"], "context_broadcast", {"content": f"{marker} wrong-token"}, sender="node-a")
    st = [x.get("status") for x in (output(b) or {}).get("broadcast", [])]
    check("A->B with a WRONG peer token is rejected by B (http_401)", st == ["http_401"], st)

    pa.terminate(); pa.wait(10)
    pa, loga = start("node-a", 18201, dict(A_BASE, PEER_A2A_TOKENS_JSON=json.dumps({URL_B + "/a2a": TOK["a_on_b"]}))); procs.append(pa)
    wait(URL_A)
    s, b = rpc(URL_A, TOK["a_primary"], "context_broadcast", {"content": f"{marker} no-totp"}, sender="node-a")
    st = [x.get("status") for x in (output(b) or {}).get("broadcast", [])]
    check("A->B with the right token but no TOTP seed is rejected (B requires TOTP)", st and st[0].startswith("http_"), st)

    pa.terminate(); pa.wait(10)
    pa, loga = start("node-a", 18201, dict(A_BASE, PEER_A2A_TOKENS_JSON=json.dumps({URL_B + "/a2a": TOK["a_on_b"]}), PEER_A2A_TOTP_SEED=SEED_B)); procs.append(pa)
    wait(URL_A)
    restart_b()
    s, b = rpc(URL_A, TOK["a_primary"], "context_broadcast", {"content": f"{marker} before-privilege"}, sender="node-a")
    st = [x.get("status") for x in (output(b) or {}).get("broadcast", [])]
    check("A->B with token+TOTP but B has not made node-a a privileged sender: http_403 (the post-#394 failure mode)", st == ["http_403"], st)
    B_ENV["LOCI_A2A_PRIVILEGED_SENDERS"] = "node-a"
    restart_b()
    s, b = rpc(URL_A, TOK["a_primary"], "context_broadcast", {"content": f"{marker} full-creds", "importance": 0.9}, sender="node-a")
    bc = (output(b) or {}).get("broadcast", [])
    check("A->B with token + TOTP + privileged sender is delivered (status ok)", [x.get("status") for x in bc] == ["ok"], bc)
    BT = (URL_B, TOK["b_primary"], None)
    def on_b(text):
        return b_has(text) > 0
    check("  B really stored it (memory content on B, not the echoed query)", on_b(marker + " full-creds"))
    for tag in ("no-creds", "wrong-token", "no-totp", "unprivileged", "before-privilege"):
        check(f"  the {tag} broadcast did NOT reach B", not on_b(f"{marker} {tag}"))
    check("  A stored its own copy locally", stored((URL_A, TOK["a_primary"], None), marker + " full-creds"))

    # ---- replay of a broadcast envelope (same idempotency key twice) ----
    import importlib.util
    spec = importlib.util.spec_from_file_location("srv", SRV)
    env_backup = dict(os.environ)
    os.environ.update(HERMES_AGENT_ID="node-a", LOCI_ENV_FILE=os.path.join(ROOT, "none.env"), HOME=ROOT, USERPROFILE=ROOT,
                      MNEMOSYNE_DATA_DIR=os.path.join(ROOT, "mnem-local"), LOCI_MEMORY_DIR=os.path.join(ROOT, "mem-local"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    payload = mod._peer_task_payload("memory_remember", f"{marker} replay", {"content": f"{marker} replay", "source": "broadcast:node-a", "importance": 0.5, "bank": "default"})
    hdr = lambda: {"Authorization": f"Bearer {TOK['a_on_b']}", "X-TOTP": totp_b(), "Content-Type": "application/json"}
    restart_b()
    x1 = httpx.post(URL_B + "/a2a", json=payload, headers=hdr(), timeout=30); r1 = x1.json()
    x2 = httpx.post(URL_B + "/a2a", json=payload, headers=hdr(), timeout=30); r2 = x2.json()
    check("B: first delivery of an envelope succeeds (200, completed)", x1.status_code == 200 and (r1.get("result") or {}).get("status") == "completed", f"{x1.status_code} {str(r1)[:140]}")
    check("B: replaying the identical envelope is rejected (403 boundary validation failed)", x2.status_code == 403 and "boundary validation failed" in json.dumps(r2), f"{x2.status_code} {str(r2)[:200]}")
    tamper = json.loads(json.dumps(payload)); tamper["params"]["input"]["_boundary"]["idempotency_key"] = str(uuid.uuid4()); tamper["params"]["input"]["content"] = f"{marker} TAMPERED"
    x3 = httpx.post(URL_B + "/a2a", json=tamper, headers=hdr(), timeout=30); r3 = x3.json()
    check("B: an envelope whose content no longer matches its artifact hash is rejected (403)", x3.status_code == 403 and "boundary validation failed" in json.dumps(r3), f"{x3.status_code} {str(r3)[:200]}")
    s4, b4 = rpc(URL_B, TOK["b_primary"], "memory_recall", {"query": marker + " TAMPERED", "top_k": 5}, totp=totp_b())
    check("  the tampered content was not stored on B", b_has(marker + " TAMPERED") == 0)
    n = b_has(marker + " replay")
    check("  the replayed envelope was stored exactly once on B", n == 1, f"stored {n} times")
    restart_b()
    codes = [rpc(URL_B, TOK["b_primary"], "memory_stats", totp=totp_b())[0] for _ in range(12)]
    check("TOTP limiter: 12 VALID requests in a minute from one IP all pass (only failures count)", codes == [200] * 12, codes)
    bad = [rpc(URL_B, TOK["b_primary"], "memory_stats", totp="000000")[0] for _ in range(6)]
    check("TOTP limiter: 5 wrong codes -> 401 each, the 6th attempt is 429", bad == [401] * 5 + [429], bad)
    check("TOTP limiter: once locked, even a valid code from that IP gets 429 until the window passes", rpc(URL_B, TOK["b_primary"], "memory_stats", totp=totp_b())[0] == 429)
    check("TOTP limiter: a bootstrap session token still works while locked (documented bypass)", rpc(URL_B, VER["tok"], "memory_stats", sender="verifier")[0] == 200)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    results.append(("RIG CRASHED: " + repr(exc)[:200], False, ""))
finally:
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(1)
    for name in ("node-a", "node-b"):
        lp = os.path.join(ROOT, name, "server.log")
        if os.path.exists(lp):
            shutil.copy(lp, os.environ["CLAUDE_JOB_DIR"] + f"/tmp/a2a_{name}.log")
    shutil.rmtree(ROOT, ignore_errors=True)
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results)-len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)
