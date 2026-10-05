"""The A2A image must carry the helper modules server.py imports for the investigation ACL.

2026-10-05: the Dockerfile copied only server.py and client.py into a build context that could not
reach mcp/, so caller_identity / inv_store never imported in the image: rag_search withheld every
investigation hit (it fails closed) and the ACL path never ran. No Docker daemon is needed here:
the Dockerfile's COPY / WORKDIR / ENV lines are executed into a scratch root filesystem and the
server is started from it with the image's PYTHONPATH.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "a2a_server" / "Dockerfile"
COMPOSE = REPO / "docker-compose.yml"


def _instructions(text: str):
    """Dockerfile instructions with line continuations joined."""
    joined = re.sub(r"\\\r?\n\s*", " ", text)
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, rest = line.partition(" ")
            yield keyword.upper(), rest.strip()


def _build_rootfs(root: pathlib.Path, skip_copy_containing: str = "") -> dict:
    """Execute COPY / WORKDIR / ENV from the real Dockerfile into `root`; return the ENV and workdir."""
    workdir, env = "/", {}
    for keyword, rest in _instructions(DOCKERFILE.read_text(encoding="utf-8")):
        if keyword == "WORKDIR":
            workdir = rest
        elif keyword == "ENV":
            name, _, value = rest.partition("=") if "=" in rest.split(" ")[0] else rest.partition(" ")
            env[name] = value
        elif keyword == "COPY":
            *sources, dest = rest.split()
            if skip_copy_containing and any(skip_copy_containing in s for s in sources):
                continue
            target = pathlib.PurePosixPath(dest if dest.startswith("/") else f"{workdir}/{dest}")
            out = root / str(target).lstrip("/")
            is_dir = dest.endswith("/") or len(sources) > 1
            if is_dir:
                out.mkdir(parents=True, exist_ok=True)
            for source in sources:
                src = REPO / source
                if not src.is_file():
                    raise AssertionError(f"COPY source {source!r} does not exist in the build context")
                shutil.copy(src, out / src.name if is_dir else out)
    return {"workdir": workdir, "env": env}


_PROBE = (
    "import json, runpy\n"
    "ns = runpy.run_path('server.py', run_name='a2a_probe')\n"
    "print(json.dumps({'caller_identity': ns['_caller_identity'] is not None,\n"
    "                  'inv_store': ns['_inv_store_acl'] is not None,\n"
    "                  'token': ns['A2A_TOKEN']}))\n"
)


def _probe(root: pathlib.Path, info: dict) -> dict:
    workdir = root / info["workdir"].lstrip("/")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    paths = [str(root / p.lstrip("/")) for p in info["env"].get("PYTHONPATH", "").split(":") if p]
    if sys.platform == "win32":
        # The image is Linux, where fcntl is stdlib; Windows has none, so give it the repo's shim
        # as a stand-in, last on the path (it must not shadow anything the image copies).
        shim = root / "win32-fcntl-standin"
        shim.mkdir()
        shutil.copy(REPO / "mcp" / "fcntl.py", shim / "fcntl.py")
        paths.append(str(shim))
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    env.pop("LOCI_A2A_TOKEN", None)
    # The token is given under its OLD name: only the image's copy of legacy_env.py turns it into LOCI_A2A_TOKEN.
    env.update(HERMES_A2A_TOKEN="legacy-name-token", LOCI_ENV_FILE=os.devnull, HOME=str(root), USERPROFILE=str(root))
    done = subprocess.run([sys.executable, "-c", _PROBE], cwd=workdir, env=env,
                          capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise AssertionError(done.stderr[-600:])
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestImageLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_the_images_layout_lets_server_py_load_both_acl_helpers(self):
        info = _build_rootfs(self.root)
        self.assertEqual(_probe(self.root, info),
                         {"caller_identity": True, "inv_store": True, "token": "legacy-name-token"})

    def test_the_same_layout_without_the_helpers_leaves_them_unavailable(self):
        """Twin of the test above: shows the probe tells a broken image from a good one."""
        info = _build_rootfs(self.root, skip_copy_containing="mcp/")
        self.assertEqual(_probe(self.root, info),
                         {"caller_identity": False, "inv_store": False, "token": ""})

    def test_only_what_the_image_needs_is_copied_from_mcp(self):
        """A wider mcp copy would put mcp/fcntl.py (the Windows shim) ahead of the stdlib module."""
        _build_rootfs(self.root)
        copied = sorted(p.name for p in (self.root / "app" / "mcp").iterdir())
        self.assertEqual(copied, ["caller_identity.py", "inv_store.py", "legacy_env.py", "provenance_firewall.py"])

    def test_the_helpers_need_nothing_beyond_the_standard_library_and_each_other(self):
        for name in ("caller_identity", "inv_store", "provenance_firewall", "legacy_env"):
            text = (REPO / "mcp" / f"{name}.py").read_text(encoding="utf-8")
            local = {m.stem for m in (REPO / "mcp").glob("*.py")}
            imported = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", text, re.M))
            imported -= set(sys.stdlib_module_names)               # fcntl etc. are the standard library on Linux
            stray = sorted(i for i in imported & local if i not in {"caller_identity", "inv_store", "provenance_firewall", "legacy_env"})
            self.assertEqual(stray, [], f"{name}.py imports other mcp modules the image does not carry")

    def test_compose_builds_from_a_context_that_contains_every_copy_source(self):
        text = COMPOSE.read_text(encoding="utf-8")
        block = re.search(r"^  hermes-a2a:\n(?P<body>(?:    .*\n|\n)+)", text, re.M)
        self.assertIsNotNone(block, "hermes-a2a service not found in docker-compose.yml")
        context = re.search(r"^\s+context:\s*(\S+)", block.group("body"), re.M).group(1)
        dockerfile = re.search(r"^\s+dockerfile:\s*(\S+)", block.group("body"), re.M).group(1)
        base = (REPO / context).resolve()
        self.assertTrue((base / dockerfile).resolve().samefile(DOCKERFILE), "compose points at another Dockerfile")
        for keyword, rest in _instructions(DOCKERFILE.read_text(encoding="utf-8")):
            if keyword == "COPY":
                for source in rest.split()[:-1]:
                    self.assertTrue((base / source).is_file(), f"{source} is outside the compose build context")


if __name__ == "__main__":
    unittest.main()
