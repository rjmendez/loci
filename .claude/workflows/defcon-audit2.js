export const meta = {
  name: 'defcon-audit2',
  description: 'Whole-codebase audit: fan out one grounded reviewer per subsystem, adversarially verify each candidate against live code, dedup + write confirmed findings. Read-only.',
  whenToUse: 'A precise second-look audit of a well-defended codebase against current origin/main. Each subsystem reviewer is grounded with a known-fixed/intentional/in-flight exclusion set (no re-reporting). Every candidate is refuted before keeping. args = {tree, investigation_id, known_state, units:[{id,title,scope}]}.',
  phases: [
    { title: 'Ground', detail: 'assemble shared grounding from prior cases' },
    { title: 'Review', detail: 'one grounded reviewer per subsystem (candidates)' },
    { title: 'Verify', detail: 'adversarial refute of each unit candidates against live code' },
    { title: 'Synthesize', detail: 'dedup (local GPU) + write confirmed findings to the investigation' },
  ],
}

const CAND_ITEM = {
  type: 'object', required: ['title', 'location', 'severity'],
  properties: {
    title: { type: 'string' }, location: { type: 'string' }, severity: { type: 'string' },
    dimension: { type: 'string' }, evidence: { type: 'string' }, impact: { type: 'string' },
  },
}
const REVIEW_SCHEMA = {
  type: 'object', required: ['candidates'],
  properties: { candidates: { type: 'array', items: CAND_ITEM } },
}
const CONF_ITEM = {
  type: 'object', required: ['title', 'location', 'severity'],
  properties: {
    title: { type: 'string' }, location: { type: 'string' }, severity: { type: 'string' },
    dimension: { type: 'string' }, impact: { type: 'string' }, how_confirmed: { type: 'string' },
  },
}
const VERIFY_SCHEMA = {
  type: 'object', required: ['confirmed'],
  properties: { confirmed: { type: 'array', items: CONF_ITEM } },
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const TREE = A.tree
const INV = A.investigation_id
const KNOWN = A.known_state || ''
const UNITS = Array.isArray(A.units) ? A.units : []
if (!TREE || !INV || !UNITS.length) { log('need tree + investigation_id + units[]'); return { error: 'missing args' } }

phase('Ground')
const grounding = await agent(
  'Assemble grounding for a whole-codebase audit. Call mcp__loci__ground with focus "security + correctness + data-integrity + robustness audit" and case_ids ["' + INV + '","defcon-defacement-audit-2026-06-24"]. Return the block VERBATIM as your final text.',
  { label: 'ground', phase: 'Ground' })
const GBLOCK = String(grounding).slice(0, 3500)

function reviewPrompt(u) {
  return 'GROUNDING (shared reference, do not re-derive):\n' + GBLOCK + '\n\n'
    + 'KNOWN STATE — do NOT report any of these (already fixed / intentional / not-built / in-flight):\n' + KNOWN + '\n\n'
    + 'You are a senior reviewer auditing ONE subsystem of a WELL-DEFENDED Next.js 16 app at ' + TREE + ' (this is current '
    + 'origin/main — authoritative; read files directly, use ripgrep for cross-refs; treat any code_graph result as possibly '
    + 'stale). SUBSYSTEM [' + u.id + '] ' + u.title + '.\nFILES: ' + u.scope + '\n\n'
    + 'Hunt HARD across dimensions: correctness bugs, security (authz/IDOR/injection/SSRF/XSS/secret-exposure), data '
    + 'integrity (round-trips, transactions, race conditions), robustness (error handling, fail-open/closed, unhandled '
    + 'rejections), concurrency. Read the real code. For EACH candidate give file:line, the deciding code, and a CONCRETE '
    + 'impact/exploit. Be strict — this codebase is well-defended, so only surface issues tied to a specific line; do NOT '
    + 'pad. Exclude everything in KNOWN STATE. Return candidates (may be empty).'
}

function verifyPrompt(u, candidates) {
  return 'You are an ADVERSARIAL verifier. For each candidate below, try to REFUTE it by reading the ACTUAL code at '
    + TREE + ' (' + u.title + '). CONFIRMED only if the defect is real, reachable, NOT mitigated elsewhere (guards, callers, '
    + 'validation, framework behaviour), and NOT in the known-fixed/intentional set. Default to refuted when uncertain. '
    + 'This app is well-defended — expect some candidates to be wrong.\n\n'
    + 'KNOWN STATE (auto-refute if it matches):\n' + KNOWN + '\n\nCANDIDATES:\n' + JSON.stringify(candidates).slice(0, 6000) + '\n\n'
    + 'Return only CONFIRMED ones: title, location (file:line), severity (H/M/L), dimension, a one-line verified impact, '
    + 'and how_confirmed (the mitigation you checked for and did NOT find).'
}

const perUnit = await pipeline(
  UNITS,
  (u) => agent(reviewPrompt(u), { label: 'review:' + u.id, phase: 'Review', schema: REVIEW_SCHEMA })
    .then((r) => ({ unit: u.id, title: u.title, candidates: (r && r.candidates) || [] }))
    .catch(() => ({ unit: u.id, title: u.title, candidates: [] })),
  (rev, u) => {
    const cands = (rev && rev.candidates) || []
    if (!cands.length) return { unit: u.id, confirmed: [] }
    return agent(verifyPrompt(u, cands), { label: 'verify:' + u.id, phase: 'Verify', schema: VERIFY_SCHEMA })
      .then((v) => ({ unit: u.id, confirmed: (v && v.confirmed) || [] }))
      .catch(() => ({ unit: u.id, confirmed: [] }))
  },
)

const confirmed = perUnit.filter(Boolean).flatMap((r) => (r.confirmed || []).map((c) => ({ ...c, unit: r.unit })))
log('confirmed findings across ' + UNITS.length + ' units: ' + confirmed.length)
if (!confirmed.length) return { confirmed: 0, note: 'no confirmed findings survived verification' }

phase('Synthesize')
const synth = await agent(
  'Here are ' + confirmed.length + ' adversarially-CONFIRMED audit findings across a whole-codebase pass:\n'
  + JSON.stringify(confirmed).slice(0, 12000) + '\n\n'
  + '1. Call mcp__loci__semantic_dedup on the finding titles (threshold 0.85) to collapse duplicates across subsystems; keep one representative each.\n'
  + '2. For each deduped finding, call mcp__loci__investigation_store with investigation_id="' + INV + '", '
  + 'finding_type="observed", source="workflow:defcon-audit2", confidence from severity (H=high/M=medium/L=low), '
  + 'tags=the dimension, and text = "[SEV] title — file:line — impact (confirmed: how)".\n'
  + '3. Return a RANKED markdown summary (severity, then subsystem) of the confirmed findings, noting how many candidates '
  + 'were dropped as duplicates.',
  { label: 'synthesize', phase: 'Synthesize' })

return { confirmed: confirmed.length, by_unit: perUnit.map((r) => ({ unit: r.unit, n: (r.confirmed || []).length })), synthesis: synth }
