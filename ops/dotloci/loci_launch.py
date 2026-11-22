"""Launch the deployed Loci server (run with cwd = loci-deploy/mcp). Dumps thread stacks once after
LOCI_LAUNCH_DUMP_S seconds (default 90) so a startup hang shows where it is blocked."""
import faulthandler
import os
import runpy
import sys

faulthandler.dump_traceback_later(int(os.environ.get("LOCI_LAUNCH_DUMP_S", "90")), exit=False)
sys.argv = ["server.py"]
runpy.run_path("server.py", run_name="__main__")
