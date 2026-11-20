"""Audit Hermes sessions for promised-but-unperformed actions and unevidenced completion claims.

Read-only over state.db. A flag is a text-only assistant turn that (a) promises an action with no tool result
after it before the next user message, or (b) claims completion with no tool result earlier in its turn.
Flags are candidates: precision is not measured here. Run: python claim_audit.py [limit]
"""
import datetime
import os
import re
import sqlite3
import sys

DB = os.path.expanduser(r"~/AppData/Local/hermes/state.db")
FUTURE = re.compile(r"\b(i'?ll|i will|let me|i'?m going to|i am going to)\b.{0,60}\b(pull|check|run|search|query|call|"
                    r"open|fetch|look|retrieve|verify|test|start|deploy|read|inspect|grep|patch|fix|write|create)\w*", re.I)
DONE = re.compile(r"\b(done:|i (have )?(created|deployed|fixed|verified|confirmed|started|saved|stored|ran|patched|"
                  r"wrote)|confirmed via|verified via|all (checks|tests) pass)\b", re.I)


def main(limit: int) -> None:
    con = sqlite3.connect(DB)
    rows = con.execute("select id,session_id,role,content,tool_calls,timestamp from messages "
                       "order by session_id,timestamp,id").fetchall()
    flagged = []
    for i, (mid, sid, role, content, tc, ts) in enumerate(rows):
        if role != "assistant" or not content or (tc and tc not in ("", "[]")):
            continue
        if FUTURE.search(content):
            kind, acted = "promised_action", False
            for nxt in rows[i + 1:i + 60]:
                if nxt[1] != sid or nxt[2] == "user":
                    break
                if nxt[2] == "tool":
                    acted = True
                    break
        elif DONE.search(content):
            kind, acted = "completion_claim", False
            for prev in reversed(rows[max(0, i - 60):i]):
                if prev[1] != sid or prev[2] == "user":
                    break
                if prev[2] == "tool" or (prev[2] == "assistant" and prev[4] not in (None, "", "[]")):
                    acted = True
                    break
        else:
            continue
        if not acted:
            flagged.append((ts, kind, sid, content))
    for ts, kind, sid, content in flagged[-limit:]:
        when = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        print(f"{when} {kind:17s} {sid[:22]} | {content[:150]!r}")
    print(f"total flagged: {len(flagged)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
