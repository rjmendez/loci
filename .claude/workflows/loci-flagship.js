export const meta = {
  name: 'loci-flagship',
  description: 'Reference Loci-native workflow: all tiers (graph/ground/embed/gen/Claude) + telemetry',
  whenToUse: 'The canonical, self-instrumented composition proving the cost/quality-tiered design: deterministic graph inventory (0 tokens) + grounded Claude fan-out + adversarial verify + an embed-dedup/gen-compress synthesis, returning a cost/quality telemetry report. Copy it as the pattern for real grounded fan-outs.',
  phases: [
    { title: 'Fanout', detail: 'graph tasks resolve 0-token from facts; others are grounded, tiered Claude agents' },
    { title: 'Verify', detail: 'adversarial check of each non-graph finding' },
    { title: 'Synthesize', detail: 'one agent dedups (embed tier) + compresses (gen tier) via Loci MCP tools' },
  ],
}

// -----------------------------------------------------------------------------
// CONTRACT (same producers as loci-native, plus a synthesis + telemetry return):
//   block = mcp__loci__ground({...}).block ; facts = graph_facts.py '[...]'
//   Workflow({ scriptPath:'.../loci-flagship.js',
//              args:{ ground:block, graphFacts:facts, tasks:[...], dedupThreshold:0.86 } })
//   tasks: [{id,title,focus, tier:'graph'|'mechanical'|'impl'|'reason', graphKey?, verify?}]
//   Returns { results:[...], synthesis, telemetry }.
//   Synthesis needs the Loci MCP connected (the agent calls mcp__loci__semantic_dedup +
//   mcp__loci__compress_text via ToolSearch); it degrades to a note if they're unavailable.
// -----------------------------------------------------------------------------

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const GROUND = A.ground || ''
const TASKS = Array.isArray(A.tasks) ? A.tasks : []
const FACTS = (A.graphFacts && typeof A.graphFacts === 'object') ? A.graphFacts : {}
const DEDUP = typeof A.dedupThreshold === 'number' ? A.dedupThreshold : 0.86
const factText = (k) => { const f = k && FACTS[k]; return f ? (typeof f === 'string' ? f : (f.text || '')) : '' }

const DEFAULT_TIERS = { mechanical: { effort: 'low' }, impl: { effort: 'high' }, reason: { model: 'opus', effort: 'high' } }
const TIERS = Object.assign({}, DEFAULT_TIERS, A.tiers || {})
const tierOpts = (t) => TIERS[t.tier] || TIERS.impl

if (!TASKS.length) { log('no tasks provided'); return { results: [], telemetry: { note: 'no tasks' } } }

// --- telemetry accumulators (what the SCRIPT can measure; token totals come from the harness usage) ---
const tel = { tiers: {}, graph_resolved_0token: 0, graph_fallback_agents: 0, claude_agents: 0,
              verify_run: 0, verify_pass: 0, grounding_sources: (GROUND.match(/\n\[[a-z]/gi) || []).length,
              grounding_chars: GROUND.length, tasks: TASKS.length }
const bump = (tier) => { tel.tiers[tier] = (tel.tiers[tier] || 0) + 1 }

const promptFor = (t) => {
  const gf = factText(t.graphKey)
  return (GROUND ? GROUND + '\n\n' : '') +
    (gf ? '## CODE-GRAPH FACTS (exact)\n' + gf + '\n\n' : '') +
    '=== YOUR TASK ===\n' + (t.title || t.id) + (t.focus ? '\n\nFocus:\n' + t.focus : '') +
    '\n\nGround your answer in the context above; cite the [tag] you rely on; flag anything grounding is silent on.'
}

phase('Fanout')
const results = await pipeline(
  TASKS,
  (t) => {
    bump(t.tier || 'impl')
    if (t.tier === 'graph') {
      const txt = factText(t.graphKey || t.id)
      if (txt) { tel.graph_resolved_0token++; log('graph "' + (t.id || t.title) + '" resolved 0-token')
        return Promise.resolve({ id: t.id, title: t.title, tier: 'graph', output: txt, _verify: false }) }
      tel.graph_fallback_agents++; tel.claude_agents++
      return agent(promptFor(t), { label: (t.id || t.title) + ':fallback', phase: 'Fanout', ...TIERS.mechanical })
        .then((o) => ({ id: t.id, title: t.title, tier: 'graph-fallback', output: o, _verify: !!t.verify }))
    }
    tel.claude_agents++
    return agent(promptFor(t), { label: t.id || t.title, phase: 'Fanout', ...tierOpts(t) })
      .then((o) => ({ id: t.id, title: t.title, tier: t.tier || 'impl', output: o, _verify: !!t.verify }))
  },
  (r) => {
    if (!r || !r._verify) return r
    tel.verify_run++
    return agent(GROUND + '\n\nAdversarially check the claim below vs grounding + live code. State whether it HOLDS ' +
      'and name any unsupported point.\n\n=== CLAIM ===\n' + r.output,
      { label: (r.id || r.title) + ':verify', phase: 'Verify', effort: 'high' })
      .then((v) => { if (/\b(holds|confirmed)\b/i.test(v)) tel.verify_pass++
        return Object.assign({}, r, { verdict: v, _verify: undefined }) })
  },
)

// --- Synthesis: exercise the embed (dedup) + gen (compress) tiers via one agent calling Loci MCP tools ---
phase('Synthesize')
const findings = results.filter(Boolean).map((r) => ({ id: r.id, text: String(r.output || '').slice(0, 1500) }))
let synthesis = { note: 'no findings to synthesize' }
if (findings.length) {
  const raw = await agent(
    'You have ' + findings.length + ' findings from a grounded fan-out (JSON below). Do TWO things using Loci MCP tools ' +
    '(load them via ToolSearch): (1) call mcp__loci__semantic_dedup with these items and threshold ' + DEDUP +
    ' to cluster near-duplicates; (2) call mcp__loci__compress_text on the concatenated KEPT findings with max_chars 700 ' +
    'to produce a tight synthesis. Return ONLY JSON: {"kept_count":int,"dropped_count":int,"dedup_ratio":float,' +
    '"compressed_synthesis":str,"tools_ok":bool}. If the MCP tools are unavailable, set tools_ok=false and do the ' +
    'dedup/compression yourself, still returning the same JSON shape.\n\nFINDINGS:\n' + JSON.stringify(findings),
    { label: 'synthesis:dedup+compress', phase: 'Synthesize', effort: 'high',
      schema: { type: 'object', required: ['kept_count', 'dropped_count', 'dedup_ratio', 'compressed_synthesis', 'tools_ok'],
        properties: { kept_count: { type: 'integer' }, dropped_count: { type: 'integer' },
          dedup_ratio: { type: 'number' }, compressed_synthesis: { type: 'string' }, tools_ok: { type: 'boolean' } } } })
  synthesis = raw || { note: 'synthesis agent returned nothing' }
}

// --- assemble the telemetry report (script-side; combine with harness subagent_tokens after the run) ---
const telemetry = {
  ...tel,
  verify_pass_rate: tel.verify_run ? Math.round((tel.verify_pass / tel.verify_run) * 100) / 100 : null,
  dedup: findings.length ? { kept: synthesis.kept_count, dropped: synthesis.dropped_count,
    ratio: synthesis.dedup_ratio, embed_gen_tools_ok: synthesis.tools_ok } : null,
  cost_note: 'graph_resolved_0token tasks spent ZERO tokens; only claude_agents (+verify) incur Claude cost; ' +
    'dedup/compress ran on the local-GPU embed+gen tiers.',
}
log('telemetry: ' + JSON.stringify(telemetry))
return { results: results.filter(Boolean), synthesis, telemetry }
