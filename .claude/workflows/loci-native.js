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
//   grounding block is produced ONCE in the main loop and passed in via args. Produce it
//   with the warm `ground` MCP tool (keeps the cross-encoder loaded, so the RAG lane is
//   reliable) — or the scripts/ground.py CLI when no server is running (cold-start may
//   drop the RAG lane; fail-open, the case lanes still carry it):
//
//     block = mcp__loci__ground({title, focus, case_ids:[...]}).block   // preferred (warm)
//     block = $(python scripts/ground.py '{"title":"...","caseIds":[...]}')  // fallback
//   Same idea for DETERMINISTIC code-graph facts — produce them once, inject via args.graphFacts:
//     facts = $(python scripts/graph_facts.py '[{"key":"callsites","impact":"foo"}]')
//     Workflow({ scriptPath: '.../loci-native.js',
//                args: { ground: block, graphFacts: facts, tasks: [...] } })
//
//   args = {
//     ground:     string   // grounding.ground().block — injected verbatim into every prompt
//     graphFacts: { <key>: {text, ...} }  // scripts/graph_facts.py output (exact, from the code graph)
//     tasks:   [{ id, title, focus, tier?, verify?, graphKey? }]
//               // tier ∈ 'graph' | 'cheap' | 'mechanical' | 'impl' | 'reason' (default 'impl')
//               // tier:'cheap'  -> resolved from cheapFacts[cheapKey||id] with NO agent;
//               //                  produce it with scripts/cheap_batch.py (vLLM -> Ollama
//               //                  -> OpenRouter). Degrades to a mechanical agent if absent.
//               // cheapKey      -> on ANY task: prepends that pre-computed answer to the prompt.
//               // tier:'graph'  -> resolved from graphFacts[graphKey||id] with NO agent (zero tokens);
//               //                  degrades to a mechanical agent if the fact is missing.
//               // graphKey      -> on ANY task: prepends that exact graph fact into the agent's prompt.
//               // verify=true   -> adds an adversarial second-stage check.
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
// Deterministic code-graph facts (from scripts/graph_facts.py), keyed by graphKey.
// A tier:'graph' task resolves straight from these — NO agent, zero tokens.
const FACTS = (A.graphFacts && typeof A.graphFacts === 'object') ? A.graphFacts : {}
// Answers pre-computed on the local/remote generation tiers by
// scripts/cheap_batch.py. Same shape as graphFacts, same zero-agent resolution:
// a Workflow agent() is always a Claude subagent, so the only way to offload is
// to do the work before the workflow and inject the result.
const CHEAP = (A.cheapFacts && typeof A.cheapFacts === 'object') ? A.cheapFacts : {}
const pick = (bag, key) => {
  const f = key && bag[key]
  if (!f) return ''
  if (typeof f === 'string') return f
  return f.ok === false ? '' : (f.text || '')
}
const factText = (key) => pick(FACTS, key)
const cheapText = (key) => pick(CHEAP, key)

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

const promptFor = (t) => {
  const gf = factText(t.graphKey)  // any task may pull a precise graph fact into its prompt for free
  const cf = cheapText(t.cheapKey || t.id)
  return (GROUND ? GROUND + '\n\n' : '') +
    (gf ? '## CODE-GRAPH FACTS (exact, from the code graph)\n' + gf + '\n\n' : '') +
    (cf ? '## PRE-COMPUTED (local/remote tier — verify before relying on it)\n' + cf + '\n\n' : '') +
    '=== YOUR TASK ===\n' + (t.title || t.id) +
    (t.focus ? '\n\nFocus:\n' + t.focus : '') +
    '\n\nGround your answer in the context above where relevant; cite the [tag] you rely on, ' +
    'and flag anything the grounding is silent on rather than inventing it.'
}

const tierOpts = (t) => TIERS[t.tier] || TIERS.impl

// pipeline: each task flows Fanout -> (optional) Verify independently, no barrier.
const results = await pipeline(
  TASKS,
  (t) => {
    // tier:'graph' -> resolve deterministically from injected facts, NO agent (zero tokens).
    // tier:'cheap' -> resolved on vLLM/Ollama/OpenRouter BEFORE the run, NO agent.
    if (t.tier === 'cheap') {
      const txt = cheapText(t.cheapKey || t.id)
      if (txt) {
        log('cheap task "' + (t.id || t.title) + '" resolved off-tier (0 Claude tokens)')
        return Promise.resolve({ id: t.id, title: t.title, tier: 'cheap', output: txt, _verify: !!t.verify })
      }
      log('cheap task "' + (t.id || t.title) + '" has no fact; falling back to a mechanical agent')
      return agent(promptFor(t), { label: (t.id || t.title) + ':fallback', phase: 'Fanout', ...TIERS.mechanical })
        .then((output) => ({ id: t.id, title: t.title, tier: 'cheap-fallback', output, _verify: !!t.verify }))
    }
    if (t.tier === 'graph') {
      const txt = factText(t.graphKey || t.id)
      if (txt) {
        log('graph task "' + (t.id || t.title) + '" resolved deterministically (0 tokens)')
        return Promise.resolve({ id: t.id, title: t.title, tier: 'graph', output: txt, _verify: false })
      }
      // no fact available -> degrade to a cheap mechanical agent rather than returning nothing.
      log('graph task "' + (t.id || t.title) + '" has no fact; falling back to a mechanical agent')
      return agent(promptFor(t), { label: (t.id || t.title) + ':fallback', phase: 'Fanout', ...TIERS.mechanical })
        .then((output) => ({ id: t.id, title: t.title, tier: 'graph-fallback', output, _verify: !!t.verify }))
    }
    return agent(promptFor(t), { label: t.id || t.title, phase: 'Fanout', ...tierOpts(t) })
      .then((output) => ({ id: t.id, title: t.title, tier: t.tier || 'impl', output, _verify: !!t.verify }))
  },
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
