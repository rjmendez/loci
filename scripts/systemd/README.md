# systemd units

Templates. Copy to `/etc/systemd/system/`, adjust the marked block, then enable.

## loci-context-bridge

Runs `a2a_context_bridge.py` on a timer, pushing this node's locally-authored memories to
its mesh peers.

```bash
sudo install -o root -g root -m 644 \
    scripts/systemd/loci-context-bridge.service \
    scripts/systemd/loci-context-bridge.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start loci-context-bridge.service      # one run first, watch the journal
sudo systemctl enable --now loci-context-bridge.timer
```

### Before enabling it on a second node

The bridge relays through the local A2A server's `context_broadcast`. **That server must
support `store_local`** (loci #144). Against an older server the flag is silently ignored,
`context_broadcast` stores before it fans out, and every run re-inserts a copy of the memory
it just read — with a fresh id and `created_at`, so the copy looks new next tick and is
relayed again. Unbounded growth on a single node, no peer required.

Check first:

```bash
curl -s localhost:8201/health >/dev/null && \
  grep -c store_local /path/to/a2a_server/server.py   # must be > 0 on the RUNNING code
```

Echo suppression relies on `BRIDGE_EXCLUDE_SOURCES` (default `bridge:,broadcast:,context_broadcast`).
Memories arriving from a peer are stamped `broadcast:<agent>` by the receiving server, and
that is what stops them being relayed back. Bring both ends up on #144 or later before
scheduling the bridge on more than one node, and watch a couple of cycles with row counts
before and after.

### Requirements

- `aiohttp` — the script `sys.exit()`s at import without it, so a missing dependency looks
  like a unit that fails instantly rather than one that quietly does nothing.
- `HERMES_A2A_TOKEN` reachable via `EnvironmentFile`. The script's own loader checks
  `~/.hermes/.env`, which does not exist on every host; where the A2A server runs as a
  container the token usually lives in the checkout's `.env` instead.

### Tuning

| env | default | notes |
| --- | --- | --- |
| `BRIDGE_LOOKBACK_MIN` | 30 | only used on the first run; after that the state file drives the window |
| `BRIDGE_MIN_IMP` | 0.5 | memories below this importance are not propagated |
| `BRIDGE_MAX_ITEMS` | 20 | per run |
| `BRIDGE_EXCLUDE_SOURCES` | `bridge:,broadcast:,context_broadcast` | echo suppression — do not narrow without reading #144 |
| `BRIDGE_STATE_FILE` | `~/.hermes/bridge_state.json` | last clean-run timestamp, plus the bounded `sent_ids` delivered-set |
| `HERMES_A2A_TOTP_SEED` | unset | required where the LOCAL server enforces TOTP; unset = no header sent |

## mrpink-context-bridge (the oxalis-mrpink half)

`mrpink-context-bridge.{service,timer}` are the live units from the second node, kept here
for the same reason as the first pair: the mesh should be reproducible from the checkout
rather than from whoever last touched the box. They differ from the loci pair in three ways
that matter, and each of them cost debugging time:

- **User scope, not system.** The A2A server there is a `systemd --user` unit, so the bridge
  is one too. That only survives a reboot with `loginctl enable-linger rjmendez` — without
  lingering the whole user manager exits at logout and the timer goes with it.
- **Two `EnvironmentFile=` lines, and the order is load-bearing.** `.bridge.env` comes second
  so it can override `HERMES_A2A_URL` from `.env`, which points at that box's WSL `eth0`
  address — reassigned on reboot. Loopback is stable and the server binds `0.0.0.0`.
  Do **not** move these into a drop-in: on that host a drop-in `Environment=` silently loses
  to the main unit's `EnvironmentFile=`, which is the opposite of what the docs imply.
- **TOTP.** See below.

### Before enabling it on a second node, part 2: TOTP

If the local A2A server enforces TOTP on `/a2a` (oxalis-mrpink does, hugbot5000-jetson does
not — check `"totp_enabled"` in `/health`), the bridge needs `HERMES_A2A_TOTP_SEED` in its
environment. Without it the bearer alone returns a flat `401` on every send, and the symptom
is a bridge that runs cleanly on a timer forever while moving nothing:

```
Bridge complete — ok=0 fail=4 skipped=0
```

`pyotp` is then a hard requirement; the script exits with a message rather than sending
unauthenticated requests. An empty seed emits no header, so this is a no-op on nodes whose
server does not enforce TOTP.

### Failure handling

The watermark only advances on a clean run. If any send fails, it is held and those memories
retry on the next tick — it used to advance unconditionally, so a peer that was briefly down
silently dropped whatever was in flight. The state file therefore also carries `sent_ids`
(bounded to the last 1000) so that holding the watermark retries the failure without
re-sending everything newer than it.

A run that is holding back says so at WARNING level:

```
Holding watermark at <ts> — N send(s) failed and would otherwise be skipped permanently.
```

A bridge that is failing is now noisy by design. Silence means it is working.
