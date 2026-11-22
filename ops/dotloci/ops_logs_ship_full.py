"""Full historical ingest wrapper — bypasses 40MB per-file limit."""
import sys
sys.path.insert(0, r"C:\Users\rjmendez-admin\.loci")

# Patch the constant before importing
import ops_logs_ship as m
m.MAX_FIRST_RUN_BYTES = 10 * 1024 * 1024 * 1024  # 10GB — effectively unlimited

# Clear state for copilot and claude sources so all files restart from offset 0
import json, os
state_file = r"C:\Users\rjmendez-admin\.loci\ops_logs_state.json"
if os.path.exists(state_file):
    with open(state_file) as f:
        state = json.load(f)
    # Zero out offsets for copilot and claude log files
    cleared = 0
    for key in list(state.keys()):
        if ".copilot" in key.lower() or ".claude" in key.lower() or "hermes" in key.lower():
            state[key] = 0
            cleared += 1
    print(f"Cleared {cleared} state entries for copilot/claude/hermes")
    with open(state_file, "w") as f:
        json.dump(state, f)

# Run the main function with full flag
sys.argv = ["ops_logs_ship_full.py", "--full"]
m.main()
