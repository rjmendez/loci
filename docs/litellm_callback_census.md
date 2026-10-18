# LiteLLM callback resolution census

Slice S4 of the Tripwire-Governor plan. Blocker for the gateway slices (S5-S7): they must not start until
the answer to "which callback loads from where" is recorded. The recorded answer is in
[Run on 2026-10-18](#run-on-2026-10-18).

`scripts/litellm_callback_census.py` reads `litellm_settings.callbacks` from a LiteLLM `config.yaml` and, for
each dotted path, asks a given interpreter with a given set of extra `sys.path` entries whether the module
resolves, which file it would load, that file's sha256, and which `sys.path` entry provides it.

## What it does and does not do

- Nothing is imported. The target interpreter walks the dotted name one level at a time through its own
  `sys.meta_path` finders (what `importlib.util.find_spec` does for a single name; `find_spec("a.b.c")` itself
  would import `a` and `a.b`, which for `litellm` is heavy). No `__init__.py`, no callback module is executed,
  no model is loaded, no network. An empty stand-in module is put in the child's `sys.modules` so a namespace
  child can compute its parent path; no code runs for it.
- The child runs with `PYTHONDONTWRITEBYTECODE=1` in an empty temporary working directory, and `PYTHONPATH` is
  exactly the `--path` entries (the caller's own `PYTHONPATH` is dropped), as the proxy is launched.
- `a.b.c` means module `a.b`, attribute `c`, as LiteLLM's `get_instance_fn` does. The module source is parsed with
  `ast` (never run) for a top-level binding of the attribute. A callback is `resolved` when the module is found and
  the attribute is not provably missing; `__getattr__` or unparseable source gives `attr_defined: null`.
- An entry with no dot is a LiteLLM built-in name (`resolved: null`, not a failure).
- The config is not echoed. Only dotted paths appear in the output; an entry that is not a dotted identifier is shown
  as `<invalid entry #N>` and counts as unresolved.
- Exit status: non-zero (`... N unresolved: ...`) when a primary-census callback is unresolved or no callbacks were found.

## Usage

```
# what the live proxy sees (interpreter and PYTHONPATH as in F:/services/litellm/litellm_service.py)
python scripts/litellm_callback_census.py \
    --config C:/Users/rjmendez-admin/AppData/Local/litellm/config.yaml \
    --python F:/services/litellm/venv/Scripts/python.exe \
    --path F:/services/litellm/control --path F:/services/litellm/control/enterprise \
    --path F:/services/litellm/control/litellm-proxy-extras

# add --compare-path (same interpreter, second path set) to compare each callback's file by sha256,
# and --json for the machine-readable report
```

Comparison statuses: `match`, `mismatch`, `only_left`, `only_right`, `neither`, `unhashable` (a namespace package has no file).

## Run on 2026-10-18

Read-only: the script plus `find_spec`-style lookups in the proxy's own interpreter
(`F:/services/litellm/venv/Scripts/python.exe`, Python 3.14.3). It exists and answered. Nothing was imported or written.

### 1. As the proxy runs: PYTHONPATH = control worktree (+ `enterprise`, `litellm-proxy-extras`), compared with the dev checkout

Exit 0, 5 callbacks, 0 unresolved. Every callback resolves from `F:\services\litellm\control`:

| # | callback | file (all under `F:\services\litellm\control\litellm\integrations\loci\`) | sha256 | dev checkout |
|---|---|---|---|---|
| 1 | `...loci.grounding_hook.loci_grounding_hook` | `grounding_hook.py` | `fc7083d87b20...` | mismatch (`7df5a69dc05e...`) |
| 2 | `...loci.memory_writer.loci_memory_writer` | `memory_writer.py` | `ba1d7181b16b...` | mismatch (`99794adcce23...`) |
| 3 | `...loci.tier_router.loci_tier_router` | `tier_router.py` | `a9c8f7ea882b...` | mismatch (`d0fdb0793a21...`) |
| 4 | `...loci.cost_ledger.loci_cost_ledger` | `cost_ledger.py` | `27ad09b4db06...` | mismatch (`d3a5ec246cc6...`) |
| 5 | `...loci.capability_gate.loci_capability_gate` | `capability_gate.py` | `ab814a1fcc3f...` | mismatch (`b0e041ff155c...`) |

Full sha256 values (control): grounding_hook `fc7083d87b207f0f526e9cc4a1f3e3a44e3bd1c9a3e57fc5bffdd538e9229d7c`,
memory_writer `ba1d7181b16ba3d37b6ecf41d58c31eed1e49e815e8070d89ea58b36f138045f`,
tier_router `a9c8f7ea882b420b49f99097c2b57a9b0da3e235b35113cad137b8638150d3d2`,
cost_ledger `27ad09b4db062c929a835af7750717eea116dc36226eb1d053187fc0f33c0ab9`,
capability_gate `ab814a1fcc3f92d3bab55395cacc3b5398355180c7475584c0ae35afca54ce83`.

The comparison right-hand side was `F:/services/litellm/dev` (+ `enterprise`, `litellm-proxy-extras`): all five also resolve
there, from `F:\services\litellm\dev\litellm\integrations\loci\`, and all five files differ from control.

### 2. Release plugin tree alone: PYTHONPATH = `F:/services/litellm/release/litellm-loci`

Exit 1, 5 callbacks, 5 unresolved, every one with the reason
`top-level module 'litellm' not found on sys.path`. The venv has no `litellm` of its own, and the release tree is the
plugin distribution `loci_litellm/` (modules `grounding_hook`, `memory_writer`, `circuit_monitor`, `cost_ledger`,
`capability_gate`, ...), not a `litellm` checkout.

### What this answers

- `litellm.integrations.loci.tier_router.loci_tier_router` resolves, today, to
  `F:\services\litellm\control\litellm\integrations\loci\tier_router.py` (sha256 in the table above), found via the control
  worktree on `PYTHONPATH`. It does not come from the release plugin tree, which has no `tier_router.py` (and no mention of
  `tier_router` in any `.py`, `.md`, `.toml` or `.yaml` there); `circuit_monitor.py` exists only in the release tree. The running
  config is therefore not broken today, but it depends on the control checkout carrying `litellm/integrations/loci/`, which a
  release-plugin-only install does not provide.
- `cost_ledger` and `capability_gate` load from the control worktree
  (`...\control\litellm\integrations\loci\cost_ledger.py` and `capability_gate.py`), as do `grounding_hook` and `memory_writer`.
  The release tree's `loci_litellm/cost_ledger.py` and `capability_gate.py` have different sha256 values from the control
  copies (release `5a601bf17a31...` / `4d4371283cfc...` against control `27ad09b4db06...` / `ab814a1fcc3f...`, from `sha256sum`,
  not from the census), so the release tree is not a drop-in for what the proxy runs.
- The dev checkout differs from control on all five callback files (this is a hash difference, not a judgement on which is right).
- Not covered by a census: whether the loaded modules actually behave as intended, or what the running process has already imported
  (this reads files, it does not inspect the live process).
