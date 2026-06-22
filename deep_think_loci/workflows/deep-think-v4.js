export const meta = {
  name: 'deep-think-v4',
  description: 'Generic deep-think audit: RAG-mode or direct-read-mode, parameterized targets, structured ranked output, optional auto-filing to GitHub. Replaces v3.2 for dama-gotchi and loci-self-audit.',
  phases: [
    { title: 'Init',       detail: 'open Loci investigation' },
    { title: 'Ideate',     detail: 'N Haiku generators per target — RAG or direct-read, no store' },
    { title: 'Write',      detail: 'dedicated Haiku writer persists all ideas' },
    { title: 'Verify',     detail: 'confirm persistence counts per target' },
    { title: 'Synthesize', detail: '3 Opus domain reviewers — adversarial red-team per group', model: 'opus' },
    { title: 'Final',      detail: 'Opus final synthesis → ranked action list', model: 'opus' },
    { title: 'File',       detail: 'gh issue create for each action ≥ floor (only if github_repo set)' },
  ],
}

// ── args normalization ──
// Workflow tool delivers args as a JSON STRING on some paths — coerce to object.
const A = (typeof args === 'string')
  ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })()
  : (args || {})

// ── parameters ──
const RUN          = A.run_id            || 'dt-v4-001'
const TITLE        = A.title             || 'deep-think v4 run'
const TARGETS      = A.targets           || []   // required: [{name, files:[], focus, read_range?:{offset,limit}}]
const RAG_COLLS    = A.rag_collections   || null // if set → RAG mode; if null → direct-read mode
const GATE         = A.ground_gate       || '/home/rjmendez/.hermes/specialists/grounding/ground_gate.py'
const THRESH       = A.ground_threshold  || 0.59
const N_IDEAS      = A.n_ideas || A.ideas_per_agent || 10
const GITHUB_REPO  = A.github_repo       || null // 'owner/repo' — enables File phase
const FILE_FLOOR   = A.file_severity_floor || 'medium'  // critical|high|medium|low
const BASE_LABELS  = A.github_base_labels || 'bug'

if (!TARGETS.length) throw new Error('deep-think-v4: args.targets is required and must be non-empty')

const RAG_MODE  = !!RAG_COLLS
const CODE_COLLS = JSON.stringify(RAG_COLLS || [])

const SRC    = (ph, ag) => `dt://${RUN}/${ph}/${ag}`
const NO_FAB = `CRITICAL: every finding_id you return MUST come from an actual mcp__loci__investigation_store tool call — never invent one; return [] if a store did not happen.`

// Severity ranking for File phase gate
const SEV_RANK = { critical: 4, high: 3, medium: 2, low: 1 }
const floorRank = SEV_RANK[FILE_FLOOR] || 2

// ── Schemas ──
const IDEA_SCHEMA = {
  type: 'object', required: ['target', 'ideas'],
  properties: { target: { type: 'string' }, ideas: { type: 'array', items: { type: 'string', minLength: 20 } } },
}
const WROTE_SCHEMA = {
  type: 'object', required: ['attempted', 'finding_ids'],
  properties: { attempted: { type: 'integer' }, finding_ids: { type: 'array', items: { type: 'string' } } },
}
const VER_SCHEMA = {
  type: 'object', required: ['total_findings', 'by_target'],
  properties: {
    total_findings: { type: 'integer' },
    by_target: { type: 'array', items: { type: 'object', properties: { target: { type: 'string' }, count: { type: 'integer' } } } },
  },
}
const SYNTH_SCHEMA = {
  type: 'object', required: ['domain', 'top_findings', 'red_team_refutals', 'action_items'],
  properties: {
    domain: { type: 'string' },
    top_findings:      { type: 'array', items: { type: 'string' }, minItems: 1 },
    red_team_refutals: { type: 'array', items: { type: 'string' } },
    action_items: {
      type: 'array', items: {
        type: 'object', required: ['title', 'severity', 'file', 'description'],
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
  type: 'object', required: ['summary', 'ranked_actions', 'gaps_identified'],
  properties: {
    summary: { type: 'string' },
    ranked_actions: {
      type: 'array', items: {
        type: 'object', required: ['rank', 'title', 'severity', 'file', 'description', 'issue_title'],
        properties: {
          rank:        { type: 'integer' },
          title:       { type: 'string' },
          severity:    { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          file:        { type: 'string' },
          description: { type: 'string' },
          issue_title: { type: 'string' },
        },
      },
    },
    gaps_identified: { type: 'array', items: { type: 'string' } },
  },
}

// ── Init ──
phase('Init')
await agent(
  `Call mcp__loci__investigation_start(investigation_id="${RUN}", title="${TITLE}", context="${RAG_MODE ? 'RAG-mode' : 'direct-read-mode'} v4 run. Targets: ${TARGETS.map(t => t.name).join(', ')}."). Return one line confirming.`,
  { label: 'init', phase: 'Init', model: 'haiku' }
)

// ── Ideate ──
// RAG mode: ground via rag_context_search + optional cosine gate
// Direct-read mode: agents read the cited files directly
phase('Ideate')

const gens = (await parallel(TARGETS.map((t, i) => () => {
  let prompt
  if (RAG_MODE) {
    const gateBlock = `GROUNDING GATE (run BEFORE reasoning): after mcp__loci__rag_context_search, write retrieved findings as JSON [{"id","text"}] to /tmp/dt_${RUN}_${t.name}_cand.json, then Bash: python3 ${GATE} --query ${JSON.stringify(t.focus)} --threshold ${THRESH} --in /tmp/dt_${RUN}_${t.name}_cand.json --out /tmp/dt_${RUN}_${t.name}_kept.json — use ONLY kept findings.`
    prompt = `Ideation generator #${i+1} for TARGET "${t.name}".\nFocus: ${t.focus}\n\n1. Ground: mcp__loci__rag_context_search(query="${t.focus}", collections=${CODE_COLLS}, limit=8)\n2. ${gateBlock}\n3. Return exactly ${N_IDEAS} concrete improvement ideas. DO NOT store anything.`
  } else {
    const fileList = (t.files || []).join(', ')
    const rangeNote = t.read_range
      ? `\nFor ${t.files[0]}: Read with offset=${t.read_range.offset}, limit=${t.read_range.limit}. Grep for key function names in the area.`
      : '\nRead each file (use limit=300 per Read call if large).'
    prompt = `Ideation generator #${i+1} for TARGET "${t.name}".\nFocus: ${t.focus}\n\nFiles: ${fileList}${rangeNote}\n\nRead the source, find bugs and improvements, return ${N_IDEAS} concrete ideas. Each must name the function/line involved, the problem, and the fix. DO NOT store anything.`
  }
  return agent(prompt, { label: `gen:${t.name}`, phase: 'Ideate', model: 'haiku', schema: IDEA_SCHEMA })
}))).filter(Boolean)

const allIdeas = gens.flatMap(g => (g.ideas || []).map(idea => ({ target: g.target, idea })))
log(`Ideate: ${gens.length}/${TARGETS.length} generators, ${allIdeas.length} ideas`)

// ── Write ──
phase('Write')
const wrote = await agent(
  `DEDICATED WRITER. Store each of the following ${allIdeas.length} ideas into Loci investigation "${RUN}".\n\nFor EACH idea:\n  mcp__loci__investigation_store(investigation_id="${RUN}", finding_type="inferred", text="<target>: <idea>", source="${SRC('ideate','writer')}", confidence="medium", tags="dt_run:${RUN},dt_target:<target>")\n\nReturn attempted=${allIdeas.length} and the real finding_ids.\n${NO_FAB}\n\nIDEAS:\n${JSON.stringify(allIdeas, null, 1)}`,
  { label: 'writer', phase: 'Write', model: 'haiku', schema: WROTE_SCHEMA }
)
log(`Write: attempted=${wrote?.attempted}, persisted=${(wrote?.finding_ids||[]).length}`)

// ── Verify ──
phase('Verify')
const truth = await agent(
  `Call mcp__loci__investigation_load(investigation_id="${RUN}", last_n_findings=120). Group findings tagged dt_run:${RUN} by dt_target. Return total_findings and by_target. Flag any target with zero findings.`,
  { label: 'verify', phase: 'Verify', model: 'haiku', schema: VER_SCHEMA }
)
const live = Object.fromEntries((truth?.by_target || []).map(t => [t.target, t.count]))
const liveTargets = TARGETS.filter(t => (live[t.name] || 0) > 0)
log(`Verify: ${liveTargets.length}/${TARGETS.length} live: ${liveTargets.map(t => t.name).join(', ')}`)

// ── Synthesize ──
// Split live targets across 3 Opus agents. Each agent red-teams its group.
phase('Synthesize')

const chunk3 = arr => {
  const n = arr.length
  if (n === 0) return [[], [], []]
  return [
    arr.filter((_, j) => j % 3 === 0),
    arr.filter((_, j) => j % 3 === 1),
    arr.filter((_, j) => j % 3 === 2),
  ]
}
const synthGroups = chunk3(liveTargets)

const makeSynthPrompt = (group, idx) => {
  const names = group.map(t => t.name).join(', ')
  const queries = group.map(t =>
    `mcp__loci__investigation_search(query="${t.focus}", investigation_id="${RUN}", limit=25)`
  ).join(' and ')
  const gateNote = RAG_MODE
    ? `Then filter each result set through the grounding gate at ${GATE} (threshold ${THRESH}) to kill cross-target bleed before reasoning.`
    : ''
  return `Adversarial Opus reviewer for group ${idx+1}: ${names || '(empty — return empty action_items)'}\n\n1. Search: ${queries}\n${gateNote}\n2. For each idea: try to REFUTE it (wrong diagnosis, already handled, impractical). Keep only ideas that survive.\n3. Return domain="${names}", top_findings that survived, red_team_refutals for killed ideas, and action_items with severity+file+description.\n\nIf group is empty, return domain="empty", empty arrays.`
}

const synthResults = (await parallel(
  synthGroups.map((group, idx) => () =>
    agent(makeSynthPrompt(group, idx), { label: `synth:g${idx}`, phase: 'Synthesize', model: 'opus', schema: SYNTH_SCHEMA })
  )
)).filter(Boolean)

const confirmedItems = synthResults.flatMap(s => (s.action_items || []).map(item => ({ ...item, domain: s.domain })))
log(`Synthesize: ${synthResults.length}/3 complete, ${confirmedItems.length} confirmed items`)

// ── Final ──
phase('Final')
const final = await agent(
  `Final Opus synthesis for "${RUN}".\n\nAll confirmed items from 3 domain reviewers:\n${JSON.stringify(confirmedItems, null, 2)}\n\n1. Merge and deduplicate (same bug from multiple angles = one item)\n2. Rank by (critical > high > medium > low) × impact on core reliability\n3. Write a 3-sentence summary\n4. Return ranked_actions (max 20) each with rank/title/severity/file/description/issue_title\n5. List gaps_identified (areas not covered)\n\nAlso store the summary:\n  mcp__loci__investigation_store(investigation_id="${RUN}", finding_type="observed", text="FINAL SYNTHESIS: <summary>", source="${SRC('final','opus')}", confidence="high", tags="dt_run:${RUN},final-synthesis")`,
  { label: 'final', phase: 'Final', model: 'opus', schema: FINAL_SCHEMA }
)
log(`Final: ${(final?.ranked_actions||[]).length} ranked actions`)

// ── File ──
// Only runs if github_repo is set.  Files issues for actions at or above file_severity_floor.
if (GITHUB_REPO) {
  phase('File')
  const toFile = (final?.ranked_actions || []).filter(a => (SEV_RANK[a.severity] || 0) >= floorRank)
  log(`File: ${toFile.length} issues to file in ${GITHUB_REPO} (floor=${FILE_FLOOR})`)

  if (toFile.length > 0) {
    // One filer agent handles all issues sequentially to avoid label race
    const FILED_SCHEMA = {
      type: 'object', required: ['filed', 'urls'],
      properties: { filed: { type: 'integer' }, urls: { type: 'array', items: { type: 'string' } } },
    }
    const filed = await agent(
      `File GitHub issues for the Loci audit run "${RUN}" in repo ${GITHUB_REPO}.\n\nFor EACH action below, run:\n  Bash: gh issue create --repo ${GITHUB_REPO} --title "<issue_title>" --body "<body>" --label "${BASE_LABELS},severity:<severity>"\n\nBody template per issue:\n  ## Summary\\n<description>\\n\\n## Location\\n\`<file>\`\\n\\n*Found by ${RUN} deep-think-v4 workflow*\n\nBefore filing, ensure severity labels exist:\n  gh label create "severity:critical" --color "b60205" --force 2>/dev/null || true\n  gh label create "severity:high"     --color "d93f0b" --force 2>/dev/null || true\n  gh label create "severity:medium"   --color "e4e669" --force 2>/dev/null || true\n  gh label create "severity:low"      --color "0075ca" --force 2>/dev/null || true\n\nReturn filed=<count> and urls=[<issue url>, ...].\n\nACTIONS:\n${JSON.stringify(toFile, null, 2)}`,
      { label: 'filer', phase: 'File', model: 'haiku', schema: FILED_SCHEMA }
    )
    log(`File: filed=${filed?.filed}, urls=${(filed?.urls||[]).join(', ')}`)
  }
}

return {
  run_id:              RUN,
  mode:                RAG_MODE ? 'rag' : 'direct-read',
  targets_covered:     liveTargets.length,
  ideas_generated:     allIdeas.length,
  ideas_persisted:     (wrote?.finding_ids||[]).length,
  confirmed_items:     confirmedItems.length,
  ranked_count:        (final?.ranked_actions||[]).length,
  summary:             final?.summary,
  ranked_actions:      final?.ranked_actions || [],
  gaps:                final?.gaps_identified || [],
  github_repo:         GITHUB_REPO,
  file_floor:          FILE_FLOOR,
}
