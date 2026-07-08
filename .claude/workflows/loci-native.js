export const meta = {
  name: 'loci-native',
  description: 'Grounded, model/effort-tiered fan-out over a task list (Loci-native template)',
  whenToUse: 'Fan out N planning/impl/verify agents that all share ONE injected grounding block (produced up front by scripts/ground.py) instead of each re-querying Loci. Tier model/effort per task.',
  phases: [
    { title: 'Fanout', detail: 'one grounded agent per task, tiered by kind' },
    { title: 'Verify', detail: 'adversarial check of each non-trivial result' },
  ],
}

// -----------------------------------------------------------------------------
// CONTRACT
//   Workflow scripts cannot import local files or call MCP tools directly, so the
//   grounding block is produced ONCE in the main loop and passed in via args:
//
//     block=$(python scripts/ground.py '{"title":"...","focus":"...","caseIds":[...]}')
//     Workflow({ name: 'loci-native', args: { ground: block, tasks: [...] } })
//
//   args = {
//     ground:  string   // grounding.ground().block — injected verbatim into every prompt
//     tasks:   [{ id, title, focus, tier?, verify? }]
//               // tier ∈ 'mechanical' | 'impl' | 'reason' (default 'impl')
//               // verify=true adds an adversarial second-stage check
//     tiers?:  { <tier>: {model?, effort?} }   // override the table below
//   }
//   Returns { results:[{id, title, tier, output, verdict?}] }.
//
//   WRITE-OUT SAFETY: agents only READ Loci + RETURN structured text. Persisting the
//   rolled-up findings is done by the SINGLE main-loop writer after this returns — never
//   by parallel agents (that is what the _append_jsonl flock + single-writer rule buy us).
// -----------------------------------------------------------------------------

// args may arrive as an object OR a JSON string depending on how it was passed — coerce.
const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const GROUND = A.ground || ''
const TASKS = Array.isArray(A.tasks) ? A.tasks : []

// Model/effort tiers. Omitting model => inherit the session model (right default).
// Only the reasoning-heavy lane names a bigger model; mechanical work drops to low effort.
const DEFAULT_TIERS = {
  mechanical: { effort: 'low' },                 // mechanical edits/lookups — cheapest
  impl: { effort: 'high' },                       // default: implementation/planning
  reason: { model: 'opus', effort: 'high' },      // hardest analysis/synthesis
}
const TIERS = Object.assign({}, DEFAULT_TIERS, A.tiers || {})

if (!GROUND) log('warning: args.ground is empty — agents run UNGROUNDED (did you run scripts/ground.py?)')
if (!TASKS.length) { log('no tasks provided'); return { results: [] } }

const promptFor = (t) =>
  (GROUND ? GROUND + '\n\n' : '') +
  '=== YOUR TASK ===\n' + (t.title || t.id) +
  (t.focus ? '\n\nFocus:\n' + t.focus : '') +
  '\n\nGround your answer in the context above where relevant; cite the [tag] you rely on, ' +
  'and flag anything the grounding is silent on rather than inventing it.'

const tierOpts = (t) => TIERS[t.tier] || TIERS.impl

// pipeline: each task flows Fanout -> (optional) Verify independently, no barrier.
const results = await pipeline(
  TASKS,
  (t) => agent(promptFor(t), { label: t.id || t.title, phase: 'Fanout', ...tierOpts(t) })
    .then((output) => ({ id: t.id, title: t.title, tier: t.tier || 'impl', output, _verify: !!t.verify })),
  (r) => {
    if (!r || !r._verify) return r
    return agent(
      GROUND + '\n\nAdversarially check the claim below against the grounding + live code/data. ' +
      'State whether it holds, and name any unsupported or contradicted point.\n\n=== CLAIM ===\n' + r.output,
      { label: (r.id || r.title) + ':verify', phase: 'Verify', effort: 'high' },
    ).then((verdict) => Object.assign({}, r, { verdict, _verify: undefined }))
  },
)

return { results: results.filter(Boolean) }
