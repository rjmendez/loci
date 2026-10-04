# Loci RAG grounding hooks for GitHub Copilot CLI

Brings the Loci grounding enforcement the Claude/hermes agents use
(`../pre_tool_grounding.py`, `../pre_llm_grounding.py`) to the GitHub Copilot CLI,
plus a global instructions file. Both the main agent and its subagents are covered.

## Install
    scripts/hooks/copilot/install.sh            # install for the current user
    scripts/hooks/copilot/install.sh --check    # report drift, change nothing

The installer:
- copies `loci_rag_grounding.sh` to `~/.copilot/hooks/`,
- writes `~/.copilot/copilot-instructions.md` (backing up any existing one),
- renders `loci-rag-grounding.json` (with the wrapper path) into the Copilot hooks
  dir (`~/.copilot/hooks/` and, if a repo is given, its `.github/hooks/`).

Confirm it loaded with `/env` inside a Copilot CLI session (it lists hooks).

## Contract
The CLI passes each hook event as JSON on stdin (`PreToolUse`, `UserPromptSubmit`,
...). The wrapper forwards it to the Loci grounding hook and echoes its decision
(`{}` allow or `{"action":"block",...}`); on any error it emits a non-blocking
grounding reminder (`additionalContext`) and allows - it never breaks a session.

## Notes
- The Copilot CLI hook JSON schema is event-keyed (same shape as `.github/hooks/*.json`).
  If GitHub changes field names, adjust `loci-rag-grounding.json` and re-run install.
- Windows/WSL: from Windows the wrapper is invoked via
  `wsl.exe -e bash -lc "<path> tool"`; install.sh detects WSL and renders that form.
