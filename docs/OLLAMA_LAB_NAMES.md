# Ollama over the lab names

Consumers reach Ollama through a DNS name over TLS, never an IP. This page is the
runbook for the one remaining compatibility shim and the rules that keep it that way.

## Path

```
consumer --https--> ollama.lab.floppydicks.net:4443   (Caddy front door, LE wildcard, allow-list)
                      -> Windows resident proxy 127.0.0.1:11435 (breaker, input guard)
                      -> Ollama 127.0.0.1:11440
```

Plain HTTP on the tailnet IP ports (`:11434/:11435/:11440/:11441`) is closed on purpose.
Auth model: source allow-list only (tailnet, loopback, private ranges). Ollama has no token.

## The shim

Some callers still say `http://127.0.0.1:11434` (Windows through `wslrelay`, WSL scripts, the
watchdog). `scripts/ollama_lab_relay.py` answers there and forwards to the name over TLS with
the Host rewritten and the response streamed. Loopback bind only; nothing new is exposed.

| Item | Where |
|---|---|
| Source of truth | `scripts/ollama_lab_relay.py` |
| Deployed copy | WSL `~/ollama_relay_daemon.py` (`cp` it; started by cron `@reboot`) |
| Keep-alive | WSL `~/relay_watchdog.py` checks `127.0.0.1:11434` every 60 s |
| Env | `OLLAMA_UPSTREAM_URL`, `OLLAMA_RELAY_BIND_HOST`, `OLLAMA_RELAY_BIND_PORT`, `OLLAMA_RELAY_TIMEOUT` |
| Tests | `scripts/tests/test_ollama_lab_relay.py` |

Deploy: `cp scripts/ollama_lab_relay.py ~/ollama_relay_daemon.py`, kill the pid in
`/tmp/ollama_relay.pid`, `nohup python3 ~/ollama_relay_daemon.py &`. Verify:
`curl -s http://127.0.0.1:11434/api/tags | head -c 80`.

## Failure signatures

| Symptom | Cause |
|---|---|
| every call to `127.0.0.1:11434` hangs, `wslrelay` still listens | an old raw-TCP forwarder (forwards to a tailnet IP port that is now closed) is the listener: `ss -ltnp` in WSL, look for `python3 ollama_relay_daemon.py` older than the shim |
| `502 {"error":"relay upstream ..."}` | the name does not resolve, the door is down (Caddy task `Lab-TLS-FrontDoor`), or the resident proxy is down |
| watchdog logs `ollama_relay DOWN` every minute and restarts fail with `Address already in use` | it checks an address nothing binds (`10.42.0.1`); it must check `127.0.0.1:11434` |

## Rules

- Consumer URLs are names. Loopback IPs are for listener binds and local probes only.
- The watchdog's `INFLUX_TOKEN` default is a hard-coded secret in a file outside this repo;
  move it to an environment variable before copying that file anywhere.
- The LiteLLM side of the same rules is enforced by `litellm-loci/deploy/check_deployment.py`
  in the LiteLLM fork (no IP-literal URLs, pinned embedding group, priced deployments).

## Rollback

Restore `~/ollama_relay_daemon.py.bak-lab-*` and `~/relay_watchdog.py.bak-lab-*`, restart as
above. The pre-shim forwarder only works if the plain tailnet port is re-opened, which is
not recommended.
