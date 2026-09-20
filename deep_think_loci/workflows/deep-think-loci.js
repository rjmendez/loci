export const meta = {
  name: 'deep-think-loci-v3.2',
  description: 'v3.2 all-Claude: haiku ideation + dedicated writer, per-target cosine grounding gate @0.59, opus half-syntheses + final that OWN the adversarial red-team. External heretic/abliterated tier dropped — it never persisted across 4 runs, opus covers adversarial, and this spends ZERO abliterated tokens.',
  phases: [
    { title: 'Init', detail: 'open the Loci investigation' },
    { title: 'Ideate', detail: '5 haiku generators (no store)' },
    { title: 'Write', detail: 'dedicated writer persists all ideas' },
    { title: 'VerifyIdeate', detail: 'load ground-truth finding_ids' },
    { title: 'Final', detail: '2 opus half-syntheses (per-target gated, red-team) + 1 opus final' },
  ],
}

const INVALID_INVESTIGATION_ID_SENTINELS = new Set(['undefined', 'null', 'none'])
const DEFAULT_TARGETS = [
  { name: 'rooted-canary', focus: 'training/rooted_canary_e2e.py rooted-device telemetry canary + DGC-26 gate' },
  { name: 'governance-gate', focus: 'training/governance_gate.py DGC checks, shadow-eval, fail-closed logic' },
  { name: 'telemetry-ingest', focus: 'realtime ingest + gotchi_mqtt_bridge.py telemetry path and schema' },
  { name: 'ant-training', focus: 'training/ant_trainer_base.py candidate export, publish, shadow-eval hook' },
  { name: 'sensor-fusion', focus: 'android EskfFusion + sensors/tdoa_triangulation.py fusion correctness' },
]
const MAX_TARGETS = 12
const TARGET_NAME_RE = /^[A-Za-z0-9_-]{1,64}$/
const COLLECTION_RE = /^[A-Za-z0-9_.:-]{1,128}$/
const SHELL_SAFE_PATH_RE = /^\/[A-Za-z0-9._\/-]{1,220}$/

function _invalidInvestigationId(fieldName, value) {
  const rendered = typeof value === 'string' ? JSON.stringify(value) : String(value)
  return new Error(`${meta.name}: invalid ${fieldName}: expected a non-empty [A-Za-z0-9_-]+ investigation id, got ${rendered}`)
}

function optionalInvestigationId(value, fieldName = 'loci_investigation') {
  if (value === undefined || value === null) return null
  if (typeof value !== 'string') throw _invalidInvestigationId(fieldName, value)
  const trimmed = value.trim()
  if (!trimmed) throw _invalidInvestigationId(fieldName, value)
  if (INVALID_INVESTIGATION_ID_SENTINELS.has(trimmed.toLowerCase())) {
    throw _invalidInvestigationId(fieldName, value)
  }
  if (!/^[A-Za-z0-9_-]+$/.test(trimmed)) throw _invalidInvestigationId(fieldName, value)
  return trimmed
}

function requiredInvestigationId(value, fieldName = 'investigation_id') {
  const resolved = optionalInvestigationId(value, fieldName)
  if (!resolved) throw _invalidInvestigationId(fieldName, value)
  return resolved
}

function investigationIdOrDefault(value, fallback, fieldName = 'run_id') {
  if (value === undefined || value === null) {
    return requiredInvestigationId(fallback, fieldName)
  }
  return requiredInvestigationId(value, fieldName)
}

function _invalidArg(fieldName, rule, value) {
  const rendered = typeof value === 'string' ? JSON.stringify(value) : String(value)
  return new Error(`${meta.name}: invalid ${fieldName}: expected ${rule}, got ${rendered}`)
}

function parseWorkflowArgs(rawArgs) {
  if (rawArgs === undefined || rawArgs === null) return {}
  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (!trimmed) return {}
    try {
      const parsed = JSON.parse(trimmed)
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
        throw _invalidArg('args', 'a JSON object', rawArgs)
      }
      return parsed
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(`${meta.name}: args must be valid JSON object text when provided as a string`)
      }
      throw error
    }
  }
  if (Array.isArray(rawArgs) || typeof rawArgs !== 'object') {
    throw _invalidArg('args', 'an object or JSON object string', rawArgs)
  }
  return rawArgs
}

function optionalText(value, fieldName, { maxLen = 180 } = {}) {
  if (value === undefined || value === null) return null
  if (typeof value !== 'string') throw _invalidArg(fieldName, `a string <= ${maxLen} chars`, value)
  const trimmed = value.trim()
  if (!trimmed) throw _invalidArg(fieldName, `a non-empty string <= ${maxLen} chars`, value)
  if (trimmed.length > maxLen) throw _invalidArg(fieldName, `a string <= ${maxLen} chars`, value)
  return trimmed
}

function intArg(value, fieldName, fallback, { min, max }) {
  const resolved = value ?? fallback
  if (!Number.isInteger(resolved) || resolved < min || resolved > max) {
    throw _invalidArg(fieldName, `an integer in [${min}, ${max}]`, value)
  }
  return resolved
}

function numberArg(value, fieldName, fallback, { min, max }) {
  const resolved = value ?? fallback
  if (typeof resolved !== 'number' || !Number.isFinite(resolved) || resolved < min || resolved > max) {
    throw _invalidArg(fieldName, `a finite number in [${min}, ${max}]`, value)
  }
  return resolved
}

function normalizeCollections(value) {
  const resolved = value ?? ['dama_gotchi_code']
  if (!Array.isArray(resolved) || resolved.length === 0 || resolved.length > 12) {
    throw _invalidArg('rag_collections', 'a non-empty array of collection names (max 12)', value)
  }
  return resolved.map((entry) => {
    if (typeof entry !== 'string') throw _invalidArg('rag_collections[]', 'a string collection name', entry)
    const trimmed = entry.trim()
    if (!trimmed || !COLLECTION_RE.test(trimmed)) {
      throw _invalidArg('rag_collections[]', `a name matching ${COLLECTION_RE}`, entry)
    }
    return trimmed
  })
}

function normalizeTargets(value) {
  const resolved = value ?? DEFAULT_TARGETS
  if (!Array.isArray(resolved) || resolved.length === 0 || resolved.length > MAX_TARGETS) {
    throw _invalidArg('targets', `a non-empty array of {name,focus} (max ${MAX_TARGETS})`, value)
  }
  return resolved.map((target, index) => {
    if (!target || Array.isArray(target) || typeof target !== 'object') {
      throw _invalidArg(`targets[${index}]`, 'an object with {name, focus}', target)
    }
    const name = optionalText(target.name, `targets[${index}].name`, { maxLen: 64 })
    const focus = optionalText(target.focus, `targets[${index}].focus`, { maxLen: 600 })
    if (!TARGET_NAME_RE.test(name)) {
      throw _invalidArg(`targets[${index}].name`, `a slug matching ${TARGET_NAME_RE}`, target.name)
    }
    return { name, focus }
  })
}

function normalizeScratchRoot(value) {
  const resolved = optionalText(value ?? '/tmp/deep_think_loci', 'scratch_root', { maxLen: 220 })
  if (!SHELL_SAFE_PATH_RE.test(resolved) || resolved.includes('..') || resolved.includes('//')) {
    throw _invalidArg('scratch_root', `an absolute UNIX-like path matching ${SHELL_SAFE_PATH_RE} with no '..'`, value)
  }
  return resolved.replace(/\/$/, '')
}

function shellQuote(value) {
  return JSON.stringify(String(value))
}

function gateLabel(prefix, targetName) {
  return `${prefix}_${targetName.replace(/[^A-Za-z0-9_-]/g, '_')}`
}

// ── parameters ──
const A = parseWorkflowArgs(args)
const RUN = investigationIdOrDefault(A.run_id, 'dt-loci-005', 'run_id')
const TITLE = optionalText(A.title, 'title', { maxLen: 180 }) || 'deep-think-loci v3 run'
const CODE_COLLS = JSON.stringify(normalizeCollections(A.rag_collections))
const N_IDEAS = intArg(A.ideas_per_agent, 'ideas_per_agent', 10, { min: 1, max: 25 })
const GATE = optionalText(A.ground_gate, 'ground_gate', { maxLen: 220 }) || '/home/rjmendez/.hermes/specialists/grounding/ground_gate.py'
const THRESH = numberArg(A.ground_threshold, 'ground_threshold', 0.59, { min: 0.5, max: 0.95 })
const TARGETS = normalizeTargets(A.targets)
const SCRATCH_ROOT = normalizeScratchRoot(A.scratch_root)

const tagStr = (phase, agent, model, extra) => `dt_run:${RUN},dt_phase:${phase},dt_agent:${agent},dt_model:${model}${extra ? ',' + extra : ''}`
const SRC = (phase, agent) => `dt://${RUN}/${phase}/${agent}`
const NO_FAB = `CRITICAL: every finding_id you return MUST come from an actual loci.investigation_store tool result — never invent one; return [] if a store did not happen.`
// the reusable GROUND block: filter retrieved evidence through the cosine gate before reasoning
const groundBlock = (focus, label) =>
  `GROUNDING GATE (run BEFORE reasoning): after loci.investigation_search, write the retrieved findings as JSON [{"id","text"}] to ${SCRATCH_ROOT}/${label}_cand.json, then Bash:\n` +
  `  mkdir -p ${shellQuote(SCRATCH_ROOT)} && python3 ${shellQuote(GATE)} --query ${JSON.stringify(focus)} --threshold ${THRESH} --in ${shellQuote(`${SCRATCH_ROOT}/${label}_cand.json`)} --out ${shellQuote(`${SCRATCH_ROOT}/${label}_kept.json`)}\n` +
  `Use ONLY the findings in ${SCRATCH_ROOT}/${label}_kept.json ("kept") as evidence — the gate drops cross-target similarity-bleed (the v1/v2 failure). Note in your output how many were dropped.`
const LOAD_GATE = `Then call loci.investigation_load(investigation_id="${RUN}", last_n_findings=80, include_retracted=true) and reconcile against tags; never synthesize a target that has no real findings.`

// ── Init ──
phase('Init')
await agent(`Call loci.investigation_start(investigation_id="${RUN}", title="${TITLE}", context="v3.2 all-Claude: dedicated-writer persistence; per-target cosine grounding gate filters RAG-bleed; opus halves+final own the adversarial red-team (external uncensored tier dropped)."). Return one line.`,
  { label: 'init', phase: 'Init', model: 'haiku' })

// ── Ideate: generators produce only (no store) ──
phase('Ideate')
const IDEA = { type: 'object', required: ['target', 'ideas'], properties: {
  target: { type: 'string' }, ideas: { type: 'array', items: { type: 'string' } } } }
const gens = (await parallel(TARGETS.map((t, i) => () =>
  agent(`Ideation generator a${i + 1} for TARGET "${t.name}" — ${t.focus}.\n` +
    `1) Ground via loci.rag_context_search(query="${t.focus}", collections=${CODE_COLLS}, limit=8).\n` +
    `2) ${groundBlock(t.focus, gateLabel(`ideate_a${i + 1}`, t.name))}\n` +
    `3) Return exactly ${N_IDEAS} concrete one-line improvement ideas. Do NOT store anything — just return them.`,
    { label: `gen:${t.name}`, phase: 'Ideate', model: 'haiku', schema: IDEA })
))).filter(Boolean)
const allIdeas = gens.flatMap(g => (g.ideas || []).map(idea => ({ target: g.target, idea })))

// ── Write: one dedicated writer persists everything (validated pattern) ──
phase('Write')
const WROTE = { type: 'object', required: ['attempted', 'finding_ids'], properties: {
  attempted: { type: 'integer' }, finding_ids: { type: 'array', items: { type: 'string' } } } }
await agent(
  `DEDICATED WRITER. Store each of these ${allIdeas.length} ideas into Loci "${RUN}" and return the REAL finding_ids.\n` +
  `For each: loci.investigation_store(investigation_id="${RUN}", finding_type="inferred", text="<target>: <idea>", source="${SRC('ideate', 'writer')}", confidence="medium", tags="${tagStr('ideate', 'writer', 'haiku')},dt_target:<target>"). ` +
  `Return attempted=${allIdeas.length} and the real finding_ids.\n${NO_FAB}\n\nIDEAS:\n${JSON.stringify(allIdeas, null, 1)}`,
  { label: 'writer', phase: 'Write', model: 'haiku', schema: WROTE })

// ── VerifyIdeate: ground truth ──
phase('VerifyIdeate')
const VER = { type: 'object', required: ['total_findings', 'by_target'], properties: {
  total_findings: { type: 'integer' },
  by_target: { type: 'array', items: { type: 'object', properties: { target: { type: 'string' }, count: { type: 'integer' } } } } } }
const ideTruth = await agent(
  `loci.investigation_load(investigation_id="${RUN}", last_n_findings=120). Group persisted dt_phase:ideate findings by dt_target; return total_findings and per-target counts. Note any of the ${TARGETS.length} targets with ZERO persisted (writer-reliability check).`,
  { label: 'verify:ideate', phase: 'VerifyIdeate', model: 'haiku', schema: VER })
const live = Object.fromEntries((ideTruth.by_target || []).map(t => [t.target, t.count]))
const advTargets = TARGETS.filter(t => (live[t.name] || 0) > 0)
log(`persisted ${advTargets.length}/${TARGETS.length} targets: ${advTargets.map(t => t.name).join(', ') || 'NONE'}`)

// (Adversarial external-model tier removed in v3.2 — opus halves+final own the red-team; conserves abliterated tokens.)

// ── Final: opus, grounding-gated, red-team ──
phase('Final')
const half = Math.ceil(TARGETS.length / 2)
const halves = [TARGETS.slice(0, half), TARGETS.slice(half)]
const SYN = { type: 'object', required: ['model', 'synthesis', 'finding_id'], properties: {
  model: { type: 'string' }, synthesis: { type: 'string' }, finding_id: { type: 'string' } } }
const halfSyn = (await parallel(halves.map((hn, h) => () =>
  {
    const perTargetGate = hn.map(target =>
      `For target "${target.name}": ` +
      `(a) loci.investigation_search(query=${JSON.stringify(target.focus)}, investigation_id="${RUN}", limit=15); ` +
      `(b) ${groundBlock(target.focus, gateLabel(`half${h}_${target.name}`, target.name))}`
    ).join('\n')
    return agent(`FINAL synthesis (opus 4.8) half-${h === 0 ? 'A' : 'B'} for "${RUN}", targets ${JSON.stringify(hn.map(t => t.name))}. ${LOAD_GATE}\n` +
      `Process EACH target SEPARATELY — NEVER blend targets into one gate query (a blended query diluted cosines and false-dropped a whole genuine target in v3).\n${perTargetGate}\n` +
      `You are ALSO the red-team for your targets: deliver the strongest SAFE ideas AND the genuine security nightmares (fail-open gates, bypasses, data-poisoning, etc.), grounded ONLY in the kept findings. Note any decision made in error from missing context. Store: loci.investigation_store(investigation_id="${RUN}", finding_type="inferred", text="<half synthesis>", source="${SRC('final', `half${h === 0 ? 'A' : 'B'}`)}", confidence="high", tags="${tagStr('final', `half${h === 0 ? 'A' : 'B'}`, 'opus')}", derived_from=<real ids>).\n${NO_FAB}`,
      { label: `final:opus:half${h === 0 ? 'A' : 'B'}`, phase: 'Final', model: 'opus', effort: 'high', schema: SYN })
  }
))).filter(Boolean)
const finalOut = await agent(
  `FINAL agent (opus 4.8), last word over "${RUN}". ${LOAD_GATE}\n` +
  `Half-A: ${JSON.stringify(halfSyn[0] || {})}\nHalf-B: ${JSON.stringify(halfSyn[1] || {})}\n` +
  `Deliver: (1) strongest safe ideas; (2) genuine security nightmares — INCLUDING cross-target patterns (the same anti-pattern recurring across targets, e.g. fail-open gates); (3) COHORT DYNAMICS — how the tiers used the corpus and any error-from-missing-context (the per-target grounding gate filtered bleed up front this run); (4) INTEGRITY CHECK — confirm every cited finding is real per investigation_load (all tiers are Claude now); flag anything ungrounded. Store: loci.investigation_store(investigation_id="${RUN}", finding_type="inferred", text="<final analysis>", source="${SRC('final', 'final')}", confidence="high", tags="${tagStr('final', 'final', 'opus')}", derived_from=<real ids>).\n${NO_FAB}`,
  { label: 'final:opus:definitive', phase: 'Final', model: 'opus', effort: 'high', schema: SYN })

return {
  investigation_id: RUN,
  ideation_truth: ideTruth,
  targets_with_corpus: advTargets.map(t => t.name),
  ground_gate: `${GATE} @ cosine>=${THRESH} (drop-in upgrade: --model grounding_bleed_clf.joblib once it beats cosine)`,
  final_analysis: finalOut,
  trace_hint: `All-Claude v3.2. Ground truth = investigation_load("${RUN}"); per-target grounding gate filtered RAG-bleed before each opus synthesis.`,
}
