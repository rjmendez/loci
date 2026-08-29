"""A loop that reports success after every step errored is worse than one that crashes.

These cover the three ways the run on 2026-08-29 misreported itself: the monitor
step could not import its own package, the cadence gates blamed the schedule for
an unreachable backend, and the summary said "done" with an exit code of 0.
"""
import ast
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LOOP = REPO / "mlops" / "loop.py"


def _load():
    sys.path.insert(0, str(REPO))
    import importlib
    mod = importlib.import_module("mlops.loop")
    return importlib.reload(mod)


def test_repo_root_is_importable_when_run_as_a_script():
    """`python mlops/loop.py` puts mlops/ on sys.path, not the repo root, so the
    monitor step's `from mlops.grounding.canary import ...` raised
    ModuleNotFoundError on every run. Reproduce that exact sys.path, load loop.py
    the way the interpreter would, and require the import to work afterwards."""
    probe = (
        "import sys, importlib.util\n"
        "sys.path = [r'%s'] + [p for p in sys.path[1:] if p not in ('', r'%s')]\n"
        "spec = importlib.util.spec_from_file_location('loop', r'%s')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "import mlops.grounding.canary\n"
    ) % (str(REPO / "mlops"), str(REPO), str(LOOP))
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, timeout=60,
                         cwd=str(pathlib.Path.home()))
    assert out.returncode == 0, out.stderr


def test_skip_reason_names_the_backend_not_the_cadence():
    loop = _load()
    down = loop._skip_reason(False, 999, 7, "http://localhost:11434")
    assert "Ollama unreachable" in down and "http://localhost:11434" in down
    assert "cadence" not in down
    due = loop._skip_reason(True, 2, 7, "http://x")
    assert "not due" in due and "cadence 7d" in due


def test_failed_steps_are_collected_and_change_the_exit_code():
    loop = _load()
    loop.FAILED_STEPS.clear()
    assert not loop.FAILED_STEPS
    loop._fail("train.py", "boom")
    assert loop.FAILED_STEPS == ["train.py"]
    loop.FAILED_STEPS.clear()


def test_main_returns_an_int_and_is_wired_to_sys_exit():
    """Returning None from main() meant every run exited 0 regardless."""
    tree = ast.parse(LOOP.read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert isinstance(main.returns, ast.Name) and main.returns.id == "int"
    src = LOOP.read_text()
    assert "sys.exit(main())" in src, "main()'s return value must reach the shell"


def test_last_error_line_extracts_the_exception_not_a_row_of_carets():
    loop = _load()
    tb = (
        'Traceback (most recent call last):\n'
        '  File "/usr/lib/python3.12/urllib/request.py", line 1347, in do_open\n'
        '    raise URLError(err)\n'
        '    ^^^^^^^^^^^^^^^^^^^\n'
        'urllib.error.URLError: <urlopen error [Errno 111] Connection refused>\n'
    )
    got = loop._last_error_line(tb)
    assert got.startswith("urllib.error.URLError")
    assert "^" not in got
    assert loop._last_error_line("") == "no stderr"
    assert loop._last_error_line("   \n  \n") == "no stderr"
