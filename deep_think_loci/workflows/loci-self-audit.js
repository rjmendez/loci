export const meta = {
  name: 'loci-self-audit',
  description: 'Apply deep-think to audit the Loci codebase itself: haiku generators read source and produce ideas (no store), a dedicated writer persists all ideas, Opus adversarial reviewers red-team by domain, Opus final synthesizer produces a ranked action list.',
  phases: [
    { title: 'Init', detail: 'Open Loci investigation for this audit run' },
    { title: 'Ideate', detail: '6 haiku agents read source files, generate ideas (no store)' },
    { title: 'Write', detail: 'Dedicated haiku writer persists all ideas to the investigation' },
    { title: 'Verify', detail: 'Ground-truth load: confirm persisted counts per target' },
    { title: 'Synthesize', detail: '3 Opus adversarial reviewers cover domain pairs' },
    { title: 'Final', detail: 'Opus final synthesis → ranked action list' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
// Required:
//   loci_root — absolute path to the Loci repository on disk
//               (e.g. '/home/user/loci' or '/workspace/loci')
// Optional:
//   run_id    — unique investigation ID; defaults to 'loci-self-audit'
//               Pass a fresh ID each run to avoid mixing findings
// ─────────────────────────────────────────────────────────────────────────────
const A    = (typeof args === 'string') ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })() : (args || {})
const LOCI = A.loci_root
if (!LOCI) {
  log('args.loci_root is required. Pass the absolute path to the Loci repository.')
  return { error: 'loci_root_required' }
}

const RUN = A.run_id || 'loci-self-audit'

const NO_FAB = `CRITICAL: every finding_id you return MUST come from an actual loci.investigation_store tool result — never invent one; return [] if a store did not happen.`
const SRC = (phase, agentId) => `dt://${RUN}/${phase}/${agentId}`

// ── Targets ───────────────────────────────────────────────────────────────────
// Grep-first: agents locate current line numbers instead of relying on hardcoded offsets.
const TARGETS = [
  {
    name: 'investigation-tools',
    files: [`${LOCI}/mcp/server.py`],
    grepTerms: ['investigation_store', 'investigation_note', 'investigation_load', 'investigation_reflect', 'investigation_list'],
    focus: 'investigation_store, investigation_note, investigation_load, investigation_reflect, investigation_list — finding lifecycle, gap resolution, hypothesis staleness, checked_sources mutability, missing edge cases',
  },
  {
    name: 'memory-system',
    files: [`${LOCI}/mcp/server.py`],
    grepTerms: ['memory_retract', 'memory_restore', 'memory_confidence', 'memory_consolidate', 'rag_context_search'],
    focus: 'memory_retract, memory_restore, memory_confidence, memory_consolidate, rag_context_search — retraction chain traversal, stale memory detection, consolidation correctness, decay logic',
  },
  {
    name: 'memcheck',
    files: [
      `${LOCI}/mcp/memcheck/engine.py`,
      `${LOCI}/mcp/memcheck/checks/code_hallucination.py`,
      `${LOCI}/mcp/memcheck/checks/contradiction.py`,
      `${LOCI}/mcp/memcheck/checks/provenance.py`,
      `${LOCI}/mcp/memcheck/verdict.py`,
    ],
    grepTerms: ['check', 'verdict', 'hallucination', 'contradiction', 'provenance'],
    focus: 'memcheck engine correctness, hallucination detection logic, contradiction check gaps, provenance verification, verdict aggregation — false positives, false negatives, edge cases',
  },
  {
    name: 'reflection-loop',
    files: [`${LOCI}/mcp/server.py`],
    grepTerms: ['reflection_loop_seed', 'reflection_loop_tick', 'reflection_loop_status'],
    extraFiles: [
      `${LOCI}/scripts/hooks/session_end_sync.py`,
      `${LOCI}/scripts/hooks/pre_llm_grounding.py`,
    ],
    focus: 'reflection_loop_seed/tick/status — queue management, state persistence, hook compatibility, path handling across different Claude Code environments',
  },
  {
    name: 'scripts-hooks',
    files: [
      `${LOCI}/scripts/hooks/pre_tool_grounding.py`,
      `${LOCI}/scripts/glymphatic_sweep.py`,
      `${LOCI}/scripts/ebbinghaus_consolidation.py`,
      `${LOCI}/scripts/spreading_activation.py`,
    ],
    grepTerms: [],
    focus: 'pre_tool_grounding IOC scanner, glymphatic sweep orphan detection, Ebbinghaus forgetting-curve consolidation, spreading activation SA-RAG — correctness, edge cases, config drift, failure modes',
  },
  {
    name: 'a2a-dtl',
    files: [
      `${LOCI}/a2a_server/server.py`,
      `${LOCI}/deep_think_loci/grounding/ground_gate.py`,
    ],
    grepTerms: ['route', 'auth', 'broadcast', 'cosine', 'gate'],
    focus: 'A2A server routing, authentication, peer broadcast, DTL workflow engine — cosine grounding gate, ideation-without-store pattern, writer reliability, adversarial synthesis patterns',
  },
]

// ── Schemas ───────────────────────────────────────────────────────────────────
const IDEA_SCHEMA = {
  type: 'object',
  required: ['target', 'ideas'],
  properties: {
    target: { type: 'string' },
    ideas:  { type: 'array', items: { type: 'string', minLength: 20 }, minItems: 5, maxItems: 12 },
  },
}

const WROTE_SCHEMA = {
  type: 'object',
  required: ['attempted', 'finding_ids'],
  properties: {
    attempted:   { type: 'integer' },
    finding_ids: { type: 'array', items: { type: 'string' } },
  },
}

const VER_SCHEMA = {
  type: 'object',
  required: ['total_findings', 'by_target'],
  properties: {
    total_findings: { type: 'integer' },
    by_target: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          target: { type: 'string' },
          count:  { type: 'integer' },
        },
      },
    },
  },
}

const SYNTH_SCHEMA = {
  type: 'object',
  required: ['domain', 'top_findings', 'red_team_refutals', 'action_items'],
  properties: {
    domain:            { type: 'string' },
    top_findings:      { type: 'array', items: { type: 'string' }, minItems: 3 },
    red_team_refutals: { type: 'array', items: { type: 'string' } },
    action_items: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'severity', 'file', 'description'],
        properties: {
          title:       { type: 'string' },
          severity:    { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          file:        { type: 'string' },
          description: { type: 'string' },
        },
      },
    },
  },
}

const FINAL_SCHEMA = {
  type: 'object',
  required: ['summary', 'ranked_actions', 'gaps_identified'],
  properties: {
    summary: { type: 'string' },
    ranked_actions: {
      type: 'array',
      items: {
        type: 'object',
        required: ['rank', 'title', 'severity', 'file', 'description', 'issue_title'],
        properties: {
          rank:        { type: 'integer' },
          title:       { type: 'string' },
          severity:    { type: 'string' },
          file:        { type: 'string' },
          description: { type: 'string' },
          issue_title: { type: 'string' },
        },
      },
    },
    gaps_identified: { type: 'array', items: { type: 'string' } },
  },
}

// ── Phase 0: Init ─────────────────────────────────────────────────────────────
phase('Init')
await agent(
  `Call mcp__loci__investigation_start(investigation_id="${RUN}", title="Loci self-audit: bugs and improvements", context="Deep-think workflow applied to the Loci codebase itself. Targets: ${TARGETS.map(t => t.name).join(', ')}. Run ${RUN}"). Return one line confirming.`,
  { label: 'init', phase: 'Init', model: 'haiku' }
)

// ── Phase 1: Ideate — generators read files but do NOT store ──────────────────
// Separating read and write prevents partial/interleaved writes to the corpus.
// A single dedicated writer in the next phase handles all stores atomically.
phase('Ideate')

const gens = (await parallel(TARGETS.map((t, i) => () => {
  const fileList  = t.files.join(', ')
  const extraNote = t.extraFiles ? `\nAlso read: ${t.extraFiles.join(', ')}` : ''
  const grepNote  = t.grepTerms.length
    ? `\nGrep for these terms first to locate current line numbers: ${t.grepTerms.map(s => `"${s}"`).join(', ')}\nThen Read the relevant line ranges (use limit=400 per Read call).`
    : '\nRead each file fully (use limit=300 per Read call if large).'

  return agent(
    `You are Loci ideation generator #${i + 1} for TARGET "${t.name}".

TASK: Read the source files below, find bugs and improvement opportunities, return 10 concrete ideas.
Do NOT call loci.investigation_store — just return your ideas.

Focus area: ${t.focus}

Files to read: ${fileList}${extraNote}${grepNote}

For each idea be specific: name the function/variable involved, describe the exact problem and the fix.
Cover: correctness bugs, missing edge cases, API design gaps, compatibility issues, performance, security, testability.

Return target="${t.name}" and exactly 10 ideas (one-liners, each ≥25 chars).`,
    { label: `gen:${t.name}`, phase: 'Ideate', model: 'haiku', schema: IDEA_SCHEMA }
  )
}))).filter(Boolean)

const allIdeas = gens.flatMap(g => (g.ideas || []).map(idea => ({ target: g.target, idea })))
log(`Ideate complete: ${gens.length}/${TARGETS.length} generators, ${allIdeas.length} total ideas`)

// ── Phase 2: Write — single dedicated writer persists all ideas ───────────────
phase('Write')

const wrote = await agent(
  `DEDICATED WRITER. Store each of the following ${allIdeas.length} ideas into Loci investigation "${RUN}".

For EACH idea call:
  mcp__loci__investigation_store(
    investigation_id="${RUN}",
    finding_type="inferred",
    text="<target>: <idea>",
    source="${SRC('ideate', 'writer')}",
    confidence="medium",
    tags="dt_run:${RUN},dt_target:<target>,loci-self-audit"
  )

Return attempted=${allIdeas.length} and the real finding_ids list.
${NO_FAB}

IDEAS:
${JSON.stringify(allIdeas, null, 1)}`,
  { label: 'writer', phase: 'Write', model: 'haiku', schema: WROTE_SCHEMA }
)

log(`Writer: attempted=${wrote?.attempted}, persisted=${(wrote?.finding_ids || []).length}`)

// ── Phase 3: Verify ───────────────────────────────────────────────────────────
phase('Verify')

const truth = await agent(
  `Call mcp__loci__investigation_load(investigation_id="${RUN}", last_n_findings=120).
Group persisted findings tagged dt_run:${RUN} by their dt_target tag value.
Return total_findings (count with dt_run:${RUN} tag) and by_target array.
Flag any of the ${TARGETS.length} targets with zero findings.`,
  { label: 'verify', phase: 'Verify', model: 'haiku', schema: VER_SCHEMA }
)

const live = Object.fromEntries((truth?.by_target || []).map(t => [t.target, t.count]))
const liveTargets = TARGETS.filter(t => (live[t.name] || 0) > 0)
log(`Verified: ${liveTargets.length}/${TARGETS.length} targets have findings: ${liveTargets.map(t => t.name).join(', ')}`)

// ── Phase 4: Synthesize — 3 Opus adversarial reviewers by domain ──────────────
phase('Synthesize')

const synthResults = await parallel([
  () => agent(
    `You are an adversarial Opus reviewer for Loci's INVESTIGATION TOOLS + MEMORY SYSTEM.

1. Call mcp__loci__investigation_search(query="investigation store note reflect gap resolution", investigation_id="${RUN}", limit=30)
2. Call mcp__loci__investigation_search(query="memory retract restore confidence consolidate staleness", investigation_id="${RUN}", limit=30)
3. For each idea: try to REFUTE it (wrong diagnosis, already handled, impractical).
   Only keep ideas that survive refutation.
4. Return domain="investigation+memory", the top findings that survived, red_team_refutals for ideas you killed,
   and action_items with severity/file/description.`,
    { label: 'synth:inv+mem', phase: 'Synthesize', model: 'opus', schema: SYNTH_SCHEMA }
  ),
  () => agent(
    `You are an adversarial Opus reviewer for Loci's MEMCHECK + REFLECTION LOOP.

1. Call mcp__loci__investigation_search(query="memcheck hallucination contradiction provenance verdict", investigation_id="${RUN}", limit=30)
2. Call mcp__loci__investigation_search(query="reflection loop session paths queue state persistence", investigation_id="${RUN}", limit=30)
3. For each idea: try to REFUTE it (wrong diagnosis, already handled, impractical).
   Only keep ideas that survive refutation.
4. Return domain="memcheck+reflection", surviving top findings, red_team_refutals, and action_items.`,
    { label: 'synth:memcheck+reflect', phase: 'Synthesize', model: 'opus', schema: SYNTH_SCHEMA }
  ),
  () => agent(
    `You are an adversarial Opus reviewer for Loci's SCRIPTS/HOOKS + A2A/DTL ENGINE.

1. Call mcp__loci__investigation_search(query="hooks pre_tool grounding IOC glymphatic ebbinghaus spreading activation", investigation_id="${RUN}", limit=30)
2. Call mcp__loci__investigation_search(query="A2A server routing auth deep-think DTL cosine gate writer reliability", investigation_id="${RUN}", limit=30)
3. For each idea: try to REFUTE it (wrong diagnosis, already handled, impractical).
   Only keep ideas that survive refutation.
4. Return domain="scripts+a2a+dtl", surviving top findings, red_team_refutals, and action_items.`,
    { label: 'synth:scripts+a2a+dtl', phase: 'Synthesize', model: 'opus', schema: SYNTH_SCHEMA }
  ),
])

const confirmedItems = synthResults
  .filter(Boolean)
  .flatMap(s => (s.action_items || []).map(item => ({ ...item, domain: s.domain })))

log(`Synthesis: ${synthResults.filter(Boolean).length}/3 completed, ${confirmedItems.length} confirmed action items`)

// ── Phase 5: Final ────────────────────────────────────────────────────────────
phase('Final')

const final = await agent(
  `You are the final Opus synthesizer for Loci self-audit run "${RUN}".

Adversarial synthesis results from 3 domain reviewers:
${JSON.stringify(confirmedItems, null, 2)}

Your job:
1. Merge and deduplicate across domains — same bug reported from multiple angles = one action item
2. Rank all action items by: (critical > high > medium > low) × impact on Loci's core reliability
3. Write a 3-sentence summary of the audit findings
4. Return ranked_actions (max 20, each with rank/title/severity/file/description/issue_title for GitHub)
5. List any gaps_identified — areas the generators missed or couldn't read

Store the final summary as a finding:
  mcp__loci__investigation_store(
    investigation_id="${RUN}",
    finding_type="observed",
    text="FINAL SYNTHESIS: <summary>",
    source="${SRC('final', 'opus')}",
    confidence="high",
    tags="dt_run:${RUN},final-synthesis"
  )`,
  { label: 'final', phase: 'Final', model: 'opus', schema: FINAL_SCHEMA }
)

log(`Final: ${(final?.ranked_actions || []).length} ranked action items`)

return {
  run_id:                 RUN,
  targets_covered:        liveTargets.length,
  ideas_generated:        allIdeas.length,
  ideas_persisted:        (wrote?.finding_ids || []).length,
  confirmed_action_items: confirmedItems.length,
  final_ranked_count:     (final?.ranked_actions || []).length,
  summary:                final?.summary,
  ranked_actions:         final?.ranked_actions || [],
  gaps:                   final?.gaps_identified || [],
}
