export const meta = {
  name: 'loci-codebase-ingest',
  description: 'Ingest Loci codebase architecture into the loci-codebase investigation — living self-knowledge',
  whenToUse: 'Run on demand after substantial changes, to refresh the loci-codebase investigation. Not CI-triggered: it needs a live Loci MCP server and its Qdrant, which a hosted runner cannot reach.',
  phases: [
    { title: 'Read', detail: 'Read each subsystem in parallel — purpose, key functions, data flows, invariants' },
    { title: 'Store', detail: 'Write structured module docs to loci-codebase Loci investigation via MCP' },
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
    desc: 'All MCP tool implementations (5495 lines). The public interface that Claude Code calls.',
    focus: 'Document every @mcp.tool()-decorated function: name, parameters, what it reads/writes, MCP tool name. Group by category (investigation_*, memory_*, reflection_loop_*). Note any tools that share mutable state.',
  },
  {
    id: 'memcheck-engine-verdict',
    files: [`${LOCI}/mcp/memcheck/engine.py`, `${LOCI}/mcp/memcheck/verdict.py`],
    desc: 'Memory-check engine and verdict system — core hallucination detection logic.',
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
    focus: 'Document the training pipeline, active learning loop, canary evaluation, and the inference path used at runtime. Note model format, input features, and output schema.',
  },
  {
    id: 'mlops-embedding-drift',
    files: [
      `${LOCI}/mlops/embedding/contrastive.py`,
      `${LOCI}/mlops/embedding/drift.py`,
    ],
    desc: 'Contrastive embedding training and embedding drift detection.',
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
    focus: 'Document exactly when each hook fires, what it reads/writes, and the latency budget. Note any side effects on the MCP server state. These are the integration point between Claude Code and Loci.',
  },
  {
    id: 'a2a-server',
    files: [`${LOCI}/a2a_server/server.py`, `${LOCI}/a2a_server/client.py`],
    desc: 'Agent-to-agent (A2A) server — inter-agent communication protocol.',
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
    { label: `read:${mod.id}`, phase: 'Read', schema: MODULE_SCHEMA }
  ).then(r => r ? { ...r, files: mod.files } : null)
))

const validDocs = modDocs.filter(Boolean)
log(`Read ${validDocs.length}/${MODULES.length} modules successfully`)

// ── Phase 2: Store each module doc to Loci investigation ──────────────────────

phase('Store')
log(`Storing module docs to investigation: ${INV}`)

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

Call mcp__loci__investigation_store with:
- investigation_id: "${INV}"
- finding_type: "observed"
- text: (the full text above)
- source: "loci-codebase-ingest workflow"
- confidence: "high"
- tags: ["architecture", "module:${doc.module_id}", "auto-ingested"]

Return the finding_id from the response.`,
    { label: `store:${doc.module_id}`, phase: 'Store' }
  )
))

const successCount = stored.filter(Boolean).length
log(`Stored ${successCount}/${validDocs.length} module docs to ${INV}`)

return {
  investigation: INV,
  modules_read: validDocs.length,
  modules_stored: successCount,
  modules: validDocs.map(d => d.module_id),
}
