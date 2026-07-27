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
| `BRIDGE_STATE_FILE` | `~/.hermes/bridge_state.json` | last successful run timestamp |
