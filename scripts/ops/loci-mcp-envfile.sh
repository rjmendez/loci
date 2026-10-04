#!/usr/bin/env bash
# ops-qdrant-envfile.sh -- move secret Environment= lines out of the loci-mcp
# systemd user unit into a mode-600 EnvironmentFile.
#
#   ops-qdrant-envfile.sh             back up unit, move secrets, daemon-reload (no restart)
#   ops-qdrant-envfile.sh --dry-run   show what would change (secret values redacted); change nothing
#   ops-qdrant-envfile.sh --restart   as the default, then restart loci-mcp and check it is active
#
# A variable is moved when its NAME matches SECRET_RE (default: KEY, TOKEN,
# SECRET, PASSWORD/PASSWD, CREDENTIAL, AUTH, PRIVATE). Non-secret variables stay
# in the unit. Values already in the env file are replaced by the key's
# unit value; other keys already in the env file are kept. Safe to re-run.
#
# Overrides (mainly for testing): LOCI_UNIT_NAME, LOCI_UNIT_PATH, LOCI_ENV_FILE,
# SYSTEMCTL (e.g. SYSTEMCTL=true to skip systemd calls), SECRET_RE.
#
# Without --restart the running server keeps its current environment until its
# next restart. On that restart it reads the key from the env file.

set -euo pipefail

MODE=apply
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --restart) MODE=restart ;;
  -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
  *) echo "usage: $0 [--dry-run|--restart]" >&2; exit 2 ;;
esac

UNIT_NAME="${LOCI_UNIT_NAME:-loci-mcp}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
ENV_FILE="${LOCI_ENV_FILE:-$HOME/.config/loci/loci-mcp.env}"
SECRET_RE="${SECRET_RE:-(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|PRIVATE)}"

if [[ -n "${LOCI_UNIT_PATH:-}" ]]; then
  UNIT="$LOCI_UNIT_PATH"
else
  UNIT="$($SYSTEMCTL --user show -p FragmentPath --value "$UNIT_NAME" 2>/dev/null || true)"
  [[ -n "$UNIT" ]] || UNIT="$HOME/.config/systemd/user/$UNIT_NAME.service"
fi
[[ -f "$UNIT" ]] || { echo "unit file not found: $UNIT" >&2; exit 1; }
[[ -w "$UNIT" ]] || { echo "unit file not writable: $UNIT (system unit? this script is for --user units)" >&2; exit 1; }

umask 077
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Python does the parsing: Environment= may hold several quoted assignments.
# Writes $WORK/unit.new, $WORK/env.new, $WORK/moved (names only), $WORK/redacted.diff
python3 - "$UNIT" "$ENV_FILE" "$SECRET_RE" "$WORK" <<'PY'
import difflib, os, re, shlex, sys

unit_path, env_path, secret_re, work = sys.argv[1:5]
secret = re.compile(secret_re, re.I)
env_file_line = f"EnvironmentFile={env_path}"

def quote_env(v: str) -> str:
    # systemd EnvironmentFile syntax: bare word or double-quoted with \" and \\ escapes.
    if re.fullmatch(r"[A-Za-z0-9_./:@%+,=-]*", v):
        return v
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'

def quote_unit(assign: str) -> str:
    return assign if re.fullmatch(r"[A-Za-z0-9_./:@%+,=-]*", assign) else '"' + assign.replace("\\", "\\\\").replace('"', '\\"') + '"'

src = open(unit_path, encoding="utf-8").read().splitlines()
out, moved, section = [], {}, None
has_env_file = False
last_service_env_idx = None
service_idx = None
for line in src:
    s = line.strip()
    if s.startswith("[") and s.endswith("]"):
        section = s
        if s == "[Service]":
            service_idx = len(out)
    if section == "[Service]" and s.startswith("EnvironmentFile="):
        target = s.split("=", 1)[1].lstrip("-")
        if os.path.expanduser(target) == os.path.expanduser(env_path):
            has_env_file = True
    if section == "[Service]" and s.startswith("Environment=") and not line.rstrip().endswith("\\"):
        body = s.split("=", 1)[1]
        try:
            assigns = shlex.split(body, posix=True)
        except ValueError:
            out.append(line); continue
        keep = []
        for a in assigns:
            if "=" not in a:
                keep.append(a); continue
            k, v = a.split("=", 1)
            if secret.search(k):
                moved[k] = v
            else:
                keep.append(a)
        if keep:
            out.append("Environment=" + " ".join(quote_unit(a) for a in keep))
            last_service_env_idx = len(out) - 1
        # a line whose assignments were all secret is dropped
        continue
    if section == "[Service]" and s.startswith("Environment="):
        # continuation lines: leave untouched, flag for the operator
        print(f"WARNING: multi-line Environment= left untouched: {s[:40]}...", file=sys.stderr)
    out.append(line)
    if section == "[Service]" and s.startswith(("Environment=", "EnvironmentFile=")):
        last_service_env_idx = len(out) - 1

if not moved:
    open(os.path.join(work, "moved"), "w").close()
    sys.exit(0)

if not has_env_file:
    insert_at = (last_service_env_idx + 1) if last_service_env_idx is not None else (service_idx + 1 if service_idx is not None else None)
    if insert_at is None:
        sys.exit("no [Service] section in unit")
    out[insert_at:insert_at] = [
        "# Secrets (QDRANT_API_KEY etc.) live here, mode 600; see ops-qdrant-envfile.sh",
        env_file_line,
    ]

# merge into existing env file, replacing moved keys
existing = []
if os.path.exists(env_path):
    existing = open(env_path, encoding="utf-8").read().splitlines()
env_out = [l for l in existing if not (re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", l) and re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", l).group(1) in moved)]
if not existing:
    env_out.append("# loci-mcp secrets (EnvironmentFile= of loci-mcp.service). Keep mode 600.")
for k, v in moved.items():
    env_out.append(f"{k}={quote_env(v)}")

open(os.path.join(work, "unit.new"), "w", encoding="utf-8").write("\n".join(out) + "\n")
open(os.path.join(work, "env.new"), "w", encoding="utf-8").write("\n".join(env_out) + "\n")
open(os.path.join(work, "moved"), "w").write("\n".join(moved) + "\n")

def redact(lines):
    r = []
    for l in lines:
        for k, v in moved.items():
            if v:
                l = l.replace(v, "<redacted>")
            l = re.sub(rf"({re.escape(k)}=)\S+", r"\1<redacted>", l)
        r.append(l)
    return r
diff = difflib.unified_diff(redact(src), redact(out), unit_path, unit_path + " (new)", lineterm="")
open(os.path.join(work, "redacted.diff"), "w").write("\n".join(diff) + "\n")
PY

if [[ ! -s "$WORK/moved" ]]; then
  echo "no secret-looking Environment= assignments in $UNIT; nothing to do."
  if [[ "$MODE" == restart ]]; then echo "(--restart ignored: nothing changed)"; fi
  exit 0
fi

echo "unit:     $UNIT"
echo "env file: $ENV_FILE"
echo "moving:   $(tr '\n' ' ' < "$WORK/moved")"
cat "$WORK/redacted.diff"

if [[ "$MODE" == dry ]]; then
  echo "--dry-run: nothing changed."
  exit 0
fi

ts="$(date +%Y%m%d-%H%M%S)"
backup="$UNIT.bak-$ts"
cp -p "$UNIT" "$backup"
chmod 600 "$backup"          # the backup still contains the secret
echo "backup:   $backup (mode 600; contains the secret -- remove it once the new unit is verified)"

mkdir -p "$(dirname "$ENV_FILE")"
chmod 700 "$(dirname "$ENV_FILE")"
install -m 600 "$WORK/env.new" "$ENV_FILE.tmp.$$"
mv -f "$ENV_FILE.tmp.$$" "$ENV_FILE"
chmod 600 "$ENV_FILE"

# Replace the unit atomically, keeping its mode.
install -m "$(stat -c %a "$UNIT")" "$WORK/unit.new" "$UNIT.tmp.$$"
mv -f "$UNIT.tmp.$$" "$UNIT"

if command -v systemd-analyze >/dev/null 2>&1 && [[ "$SYSTEMCTL" == systemctl ]]; then
  systemd-analyze --user verify "$UNIT" 2>&1 | sed 's/^/[verify] /' || true
fi

$SYSTEMCTL --user daemon-reload
echo "daemon-reload done."

if [[ "$SYSTEMCTL" == systemctl ]]; then
  $SYSTEMCTL --user show "$UNIT_NAME" -p EnvironmentFiles
  if $SYSTEMCTL --user show "$UNIT_NAME" -p Environment --value | tr ' ' '\n' | grep -qE "^[A-Za-z0-9_]*${SECRET_RE}[A-Za-z0-9_]*="; then
    echo "WARNING: systemd still reports a secret-looking Environment= entry (drop-in?)." >&2
  fi
fi

if [[ "$MODE" == restart ]]; then
  $SYSTEMCTL --user restart "$UNIT_NAME"
  sleep 3
  if $SYSTEMCTL --user is-active --quiet "$UNIT_NAME"; then
    echo "$UNIT_NAME restarted and active."
  else
    echo "ERROR: $UNIT_NAME is not active after restart. Roll back with:" >&2
    echo "  cp -p '$backup' '$UNIT' && systemctl --user daemon-reload && systemctl --user restart $UNIT_NAME" >&2
    exit 1
  fi
else
  echo "Not restarted: the running $UNIT_NAME keeps its current environment until its next restart."
  echo "Run with --restart (or: systemctl --user restart $UNIT_NAME) to switch it to the env file now."
fi
echo "Rollback: cp -p '$backup' '$UNIT' && systemctl --user daemon-reload"
