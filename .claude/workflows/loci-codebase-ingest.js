export const meta = {
  name: 'loci-codebase-ingest',
  description: 'Ingest Loci codebase architecture into the loci-codebase investigation — living self-knowledge',
  whenToUse: 'Run on demand after substantial changes, to refresh the loci-codebase investigation. Not CI-triggered: it needs a live Loci MCP server and its Qdrant, which a hosted runner cannot reach.',
  phases: [
    { title: 'Read', detail: 'Read each subsystem in parallel — purpose, key functions, data flows, invariants' },
    { title: 'Store', detail: 'Retire the prior doc for each module, then write the new one via MCP' },
  ],
}

// Repo-relative by default. This was an absolute path on one machine
// (/mnt/c/Users/rjmendez-admin/development/loci), which is why the script could
// never have run anywhere else — including the CI job that referenced it for two
// months without it ever being committed (#210). Agents run from the repo root,
// so '.' is correct; pass args.repo to point at a different checkout.
const LOCI = (typeof args === 'object' && args && args.repo) ? args.repo : '.'
const INV  = (typeof args === 'object' && args && args.investigation) || 'loci-codebase'

const MODULE_SCHEMA = {
  type: 'object',
  properties: {
    module_id:     { type: 'string' },
    purpose:       { type: 'string' },
    key_symbols:   {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name:      { type: 'string' },
          kind:      { type: 'string', enum: ['class', 'function', 'constant', 'decorator'] },
          signature: { type: 'string' },
          does:      { type: 'string' },
        },
        required: ['name', 'kind', 'does'],
      },
    },
    data_flows:    { type: 'string' },
    dependencies:  { type: 'array', items: { type: 'string' } },
    invariants:    { type: 'string' },
    open_issues:   { type: 'string' },
    last_commit:   { type: 'string' },
  },
  required: ['module_id', 'purpose', 'key_symbols', 'data_flows', 'dependencies'],
}

const MODULES = [
  {
    id: 'mcp-server-tools',
    files: [`${LOCI}/mcp/server.py`],
    desc: 'All MCP tool implementations (8211 lines, 42 @mcp.tool() functions). The public interface that Claude Code calls.',
    model: 'opus', effort: 'high',
    focus: 'Document every @mcp.tool()-decorated function: name, parameters, what it reads/writes, MCP tool name. Group by category (investigation_*, memory_*, reflection_loop_*). Note any tools that share mutable state.',
  },
  {
    id: 'memcheck-engine-verdict',
    files: [`${LOCI}/mcp/memcheck/engine.py`, `${LOCI}/mcp/memcheck/verdict.py`],
    desc: 'Memory-check engine and verdict system — core hallucination detection logic.',
    model: 'sonnet', effort: 'low', // 319 lines, but semantically the core — small and dense
    focus: 'Document the check pipeline: what checks run, in what order, how verdicts are aggregated. Note the verdict confidence levels and how they map to pass/fail.',
  },
  {
    id: 'memcheck-storage',
    files: [
      `${LOCI}/mcp/memcheck/backend.py`,
      `${LOCI}/mcp/memcheck/qdrant.py`,
      `${LOCI}/mcp/memcheck/mnemosyne.py`,
    ],
    desc: 'Three-layer storage: JSONL (on-disk), Qdrant (vector search), Mnemosyne (episodic memory mirror).',
    model: 'sonnet', effort: 'medium', // 686 lines across three backends
    focus: 'Document which functions write to which layer, the consistency guarantees between layers, and what happens when Qdrant is unreachable. Note the embedding dimensions and collection names.',
  },
  {
    id: 'mlops-grounding',
    files: [
      `${LOCI}/mlops/grounding/train.py`,
      `${LOCI}/mlops/grounding/active_learn.py`,
      `${LOCI}/mlops/grounding/canary.py`,
      `${LOCI}/deep_think_loci/grounding/ground_gate.py`,
    ],
    desc: 'Hallucination grounding gate — the ML classifier that detects fabricated entities.',
    model: 'sonnet', effort: 'high', // 1381 lines, train/active-learn/canary/inference all need reconciling
    focus: 'Document the training pipeline, active learning loop, canary evaluation, and the inference path used at runtime. Note model format, input features, and output schema.',
  },
  {
    id: 'mlops-embedding-drift',
    files: [
      `${LOCI}/mlops/embedding/contrastive.py`,
      `${LOCI}/mlops/embedding/drift.py`,
    ],
    desc: 'Contrastive embedding training and embedding drift detection.',
    model: 'sonnet', effort: 'low', // 419 lines, two self-contained files
    focus: 'Document the contrastive loss formulation, what positive/negative pairs are, and how drift is measured. Note whether drift triggers retraining automatically.',
  },
  {
    id: 'mlops-memory-loop',
    files: [
      `${LOCI}/mlops/memory/decay.py`,
      `${LOCI}/mlops/memory/live_evo.py`,
      `${LOCI}/mlops/loop.py`,
    ],
    desc: 'Memory decay schedules, live evolution, and the main MLOps loop.',
    model: 'sonnet', effort: 'medium', // 1049 lines, mostly the orchestrator
    focus: 'Document the decay algorithm (Ebbinghaus?), what "live evolution" modifies, and what the main loop orchestrates. Note the schedule/cadence and what triggers each operation.',
  },
  {
    id: 'claude-hooks',
    files: [
      `${LOCI}/scripts/hooks/pre_llm_grounding.py`,
      `${LOCI}/scripts/hooks/pre_tool_grounding.py`,
      `${LOCI}/scripts/hooks/session_end_sync.py`,
    ],
    desc: 'Claude Code hook scripts — pre-LLM grounding, pre-tool grounding, session-end sync.',
    model: 'sonnet', effort: 'high', // 1717 lines; the latency budget and side effects are easy to get wrong
    focus: 'Document exactly when each hook fires, what it reads/writes, and the latency budget. Note any side effects on the MCP server state. These are the integration point between Claude Code and Loci.',
  },
  {
    id: 'a2a-server',
    files: [`${LOCI}/a2a_server/server.py`, `${LOCI}/a2a_server/client.py`],
    desc: 'Agent-to-agent (A2A) server — inter-agent communication protocol.',
    model: 'sonnet', effort: 'high', // 1862 lines incl. the auth path
    focus: 'Document the routing protocol, authentication (bearer token + TOTP from git log shows a fix was recently made), and message schema. Note whether this is used in production or is experimental.',
  },
]

// ── Phase 1: Read modules in parallel ─────────────────────────────────────────

phase('Read')
log(`Reading ${MODULES.length} Loci subsystems in parallel`)

const modDocs = await parallel(MODULES.map(mod => () =>
  agent(
    `Read the Loci source files listed below and produce a structured architecture document.
This document will be stored as permanent self-knowledge in the loci-codebase Loci investigation.

Files to read: ${mod.files.join(', ')}
Module ID: ${mod.id}
Description: ${mod.desc}
Focus: ${mod.focus}

Read all files completely. Also run: git -C ${LOCI} log -1 --format="%H %s" -- ${mod.files[0]}
to get the last commit SHA and message for the primary file.

Return a structured doc with:
- module_id: "${mod.id}"
- purpose: 2-3 sentence description of what this module does and why it exists
- key_symbols: the 6-12 most important public functions/classes/decorators, each with name, kind, signature (condensed), and does (one sentence)
- data_flows: paragraph describing what goes in, what comes out, what side effects occur, and how this module connects to the rest of Loci
- dependencies: list of imports/modules this depends on (both internal loci.* and external packages)
- invariants: critical invariants callers must know (e.g. "investigation must exist before storing findings")
- open_issues: any TODO/FIXME/HACK comments found, or "None found"
- last_commit: the git SHA from the command above`,
    {
      label: `read:${mod.id}`,
      phase: 'Read',
      schema: MODULE_SCHEMA,
      // Reading is the expensive tier: it has to hold a whole subsystem in head
      // and compress it without inventing symbols. Size drives the tier —
      // mcp/server.py alone is 5.5k lines.
      model: mod.model || 'sonnet',
      effort: mod.effort || 'medium',
    }
  ).then(r => r ? { ...r, files: mod.files } : null)
))

const validDocs = modDocs.filter(Boolean)
log(`Read ${validDocs.length}/${MODULES.length} modules successfully`)

// ── Phase 2: Store each module doc to Loci investigation ──────────────────────

phase('Store')
log(`Storing module docs to investigation: ${INV}`)

// Enumerate the priors ONCE per run, from disk rather than the vector index.
//
// This used to be a per-agent mcp__loci__investigation_search. That tool returns
// {results: []} for BOTH "no priors exist" and "Qdrant is unavailable"
// (mode: "rag_required", reason: "qdrant_unavailable") — an empty list on a
// *failed* lookup. A search-driven supersede therefore retires nothing and
// appends a duplicate during an outage, which is the exact doc pile-up the
// supersede exists to prevent. loci_health is not a usable proxy either: it
// reported qdrant_reachable:true while search returned rag_required.
//
// investigation_load folds findings.jsonl + finding_updates.jsonl off disk, so
// it stays correct while the index is down. One call also costs less than eight.
const PRIORS_SCHEMA = {
  type: 'object',
  properties: {
    ok:     { type: 'boolean' },
    reason: { type: 'string' },
    priors: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          module_id:   { type: 'string' },
          finding_ids: { type: 'array', items: { type: 'string' } },
        },
        required: ['module_id', 'finding_ids'],
      },
    },
  },
  required: ['ok', 'priors'],
}

const priorsDoc = await agent(
  `Enumerate the currently-open architecture docs in the Loci investigation "${INV}".

Use ToolSearch to load the mcp__loci__investigation_load schema, then call it with:
- investigation_id: "${INV}"
- last_n_findings: 500
- fidelity: "full"

Do NOT use investigation_search: it returns an empty result set when Qdrant is
unavailable, which is indistinguishable from "no priors exist".

From the returned findings, select every finding whose text's FIRST LINE is
exactly "MODULE: <id>" for one of these ids, and whose resolution is "open"
(a finding with no resolution field counts as open):
${MODULES.map(m => `  - ${m.id}`).join('\n')}

Match the first line EXACTLY. Substring matching wrongly retires neighbours —
"mlops-memory-loop" must not match "mlops-grounding", and "memcheck-storage"
must not match "memcheck-engine-verdict".

Return:
- ok: true only if the investigation_load call SUCCEEDED. If it errored, returned
  an error field, or you could not enumerate the findings, return ok: false and
  put the error in reason. Never return ok: true on a failed or empty-because-of-error
  lookup. An investigation that genuinely has zero prior docs is ok: true with an
  empty priors array.
- reason: the error text when ok is false, otherwise "".
- priors: one entry per module id above that has at least one OPEN doc, with the
  finding_ids to retire. Omit modules with no open doc.`,
  { label: 'enumerate:priors', phase: 'Store', schema: PRIORS_SCHEMA, model: 'sonnet', effort: 'low' }
)

// Fail CLOSED: without a trustworthy prior list, storing would append a second
// open doc per module and retire nothing. Better to write nothing this run.
if (!priorsDoc || priorsDoc.ok !== true) {
  const why = (priorsDoc && priorsDoc.reason) || 'enumeration agent returned nothing'
  log(`ABORT: could not enumerate prior docs (${why}). Storing now would duplicate every module doc; nothing was written.`)
  return {
    investigation: INV,
    aborted: 'prior_enumeration_failed',
    reason: why,
    modules_read: validDocs.length,
    modules_stored: 0,
    modules: validDocs.map(d => d.module_id),
  }
}

const priorsById = new Map((priorsDoc.priors || []).map(p => [p.module_id, p.finding_ids || []]))
log(`Prior open docs found for ${priorsById.size} module(s)`)

const stored = await parallel(validDocs.map(doc => () =>
  agent(
    `Store this Loci module architecture document to the investigation "${INV}".

Use ToolSearch to load the mcp__loci__investigation_store schema, then call it.

Store the following text as a single finding:

---
MODULE: ${doc.module_id}

PURPOSE:
${doc.purpose}

KEY SYMBOLS:
${doc.key_symbols.map(s => `- [${s.kind}] ${s.name}(${s.signature || ''}): ${s.does}`).join('\n')}

DATA FLOWS:
${doc.data_flows}

DEPENDENCIES: ${doc.dependencies.join(', ')}

INVARIANTS:
${doc.invariants || 'None documented'}

OPEN ISSUES:
${doc.open_issues || 'None found'}

LAST COMMIT: ${doc.last_commit || 'unknown'}
SOURCE FILES: ${doc.files.join(', ')}
---

FIRST, retire the previous doc for this module. This workflow appends a fresh
full doc every run; if the old one is not retired the corpus accumulates several
"MODULE: ${doc.module_id}" findings, all finding_type "observed" and all
confidence "high", so a RAG query gets multiple answers with nothing marking the
stale ones. The prior docs have already been enumerated from disk for you --
do NOT call investigation_search to look for more.

Steps:
1. For each finding id listed below, call mcp__loci__finding_resolve with
   investigation_id "${INV}", resolution "superseded", and note
   "superseded by loci-codebase-ingest run".
   Finding ids to retire: ${(priorsById.get(doc.module_id) || []).join(', ') || '(none -- nothing to retire)'}
2. If every resolve succeeded (or there were none to do), continue to the store.
3. If any resolve FAILED, do NOT store. Return
   {"finding_id": null, "superseded": 0, "error": "<what failed>"} instead --
   storing after a failed retire is what leaves two open docs for this module.

THEN call mcp__loci__investigation_store with:
- investigation_id: "${INV}"
- finding_type: "observed"
- text: (the full text above)
- source: "loci-codebase-ingest workflow"
- confidence: "high"
- tags: ["architecture", "module:${doc.module_id}", "auto-ingested"]

Return JSON: {"finding_id": "<new id>", "superseded": <count of prior docs retired>}.`,
    {
      label: `store:${doc.module_id}`,
      phase: 'Store',
      // Deterministic MCP calls (finding_resolve per prior, then one store)
      // against text already written upstream; the ids come from the enumeration.
      model: 'haiku',
      effort: 'low',
    }
  )
))

const ok = stored.filter(r => r && r.finding_id && !r.error)
const failed = stored.filter(r => !r || r.error || !r.finding_id)
const supersededCount = ok.reduce((n, r) => n + (Number(r.superseded) || 0), 0)
log(`Stored ${ok.length}/${validDocs.length} module docs to ${INV}; retired ${supersededCount} prior doc(s)`)
if (failed.length) log(`WARNING: ${failed.length} module(s) did not store -- their prior doc is still the live one`)

return {
  investigation: INV,
  modules_read: validDocs.length,
  modules_stored: ok.length,
  priors_superseded: supersededCount,
  store_failures: failed.length,
  modules: validDocs.map(d => d.module_id),
}
