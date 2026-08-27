export const meta = {
  name: 'audit-remediate',
  description: 'Grounded, local-GPU-offloaded remediation planning over confirmed audit findings',
  whenToUse: 'After a security/quality audit has recorded findings in a Loci investigation. Grounds once, fans out one remediation-planner per finding (grounded on the code graph + bge/expand RAG), offloads the mechanical tier (batch risk-statements via vLLM generate_batch, dedup via semantic_dedup) to the local GPUs through Loci, then synthesizes a ranked fix plan back into the investigation. args = {investigation_id, repo, findings:[{id,title,file,severity}]}.',
  phases: [
    { title: 'Ground', detail: 'assemble one grounding block from the investigation + code graph' },
    { title: 'Plan', detail: 'one grounded remediation-planner agent per finding' },
    { title: 'Offload', detail: 'vLLM generate_batch risk-statements + semantic_dedup merge — through Loci' },
    { title: 'Synthesize', detail: 'rank + write the remediation plan back to the investigation' },
  ],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const INV = A.investigation_id
const REPO = A.repo || ''
const FINDINGS = Array.isArray(A.findings) ? A.findings : []
if (!INV || !FINDINGS.length) { log('need investigation_id + findings[]'); return { error: 'missing args' } }

// ---- Phase 1: ground ONCE (injected into every planner). Exercises grounding + bge/expand RAG. ----
phase('Ground')
const grounding = await agent(
  'Assemble grounding for a remediation pass on investigation "' + INV + '" (repo ' + REPO + '). '
  + 'Call mcp__loci__ground with a focus of "security + robustness remediation" and case_ids including "' + INV + '". '
  + 'Return the grounding block VERBATIM as your final text (it is injected into every downstream agent).',
  { label: 'ground', phase: 'Ground' })

// ---- Phase 2 (Plan) + Phase 3 (Offload) pipelined per finding, no barrier. ----
// Each finding: a grounded planner drafts a concrete fix, THEN an offload step routes the
// mechanical write-up through the local GPUs via Loci (vLLM batched gen + local-embed dedup).
const planned = await pipeline(
  FINDINGS,
  (f) => agent(
    'GROUNDING (shared, do not re-derive):\n' + String(grounding).slice(0, 6000) + '\n\n'
    + 'You are a senior engineer writing a CONCRETE remediation for one audit finding in ' + REPO + '.\n'
    + 'Finding [' + (f.severity || '?') + '] ' + f.title + (f.file ? ' (' + f.file + ')' : '') + '.\n'
    + 'Use mcp__loci__rag_context_search (bge rerank + query_expand) and, if available, '
    + 'mcp__loci__code_graph_query to pull the EXACT current code + its callers. Then write: (1) root cause in '
    + 'one sentence, (2) the concrete fix as a minimal patch sketch tied to file:line, (3) any blast-radius/callers '
    + 'to update, (4) a one-line test that would catch a regression. Be specific to the real code, not generic.',
    { label: 'plan:' + String(f.id || f.title).slice(0, 8), phase: 'Plan',
      schema: { type: 'object', required: ['id', 'severity', 'fix'],
        properties: { id: { type: 'string' }, severity: { type: 'string' },
          fix: { type: 'string' }, patch_sketch: { type: 'string' },
          callers_to_update: { type: 'array', items: { type: 'string' } },
          regression_test: { type: 'string' } } } })
    .then((p) => p ? { ...p, id: p.id || f.id, title: f.title, file: f.file || '' } : null),
)
const plans = planned.filter(Boolean)

// ---- Offload the mechanical tier to the local GPUs THROUGH Loci (the point of this run). ----
phase('Offload')
const offload = await agent(
  'You coordinate LOCAL-GPU offload through Loci — do NOT reason about the fixes yourself; delegate.\n'
  + 'Here are the drafted remediation plans as JSON:\n' + JSON.stringify(plans.map(p => ({ id: p.id, severity: p.severity, title: p.title, fix: String(p.fix).slice(0, 400) }))).slice(0, 8000) + '\n\n'
  + '1) Call mcp__loci__generate_batch (this routes to vLLM batched-gen via backends, else Ollama) with one prompt '
  + 'PER plan, each: "In one sentence, state the concrete production RISK if this is left unfixed: <title> — <fix>". '
  + 'This is the continuous-batching win: submit all prompts in a single generate_batch call.\n'
  + '2) Call mcp__loci__semantic_dedup on the array of plan titles (threshold ~0.85) to find remediations that '
  + 'collapse into one fix.\n'
  + 'Return {risk_by_id: {id: risk_sentence}, dedup_clusters: [[ids]], backend_used: "<vllm|ollama, from the tool output>"}.',
  { label: 'offload:local-gpu', phase: 'Offload',
    schema: { type: 'object', properties: {
      risk_by_id: { type: 'object' }, dedup_clusters: { type: 'array' }, backend_used: { type: 'string' } } } })

// ---- Phase 4: synthesize a ranked plan and persist it to the investigation. ----
phase('Synthesize')
const synthesis = await agent(
  'Synthesize a RANKED remediation plan from these drafted fixes + local-GPU offload outputs.\n'
  + 'PLANS:\n' + JSON.stringify(plans).slice(0, 12000) + '\n\nOFFLOAD (risk statements + dedup clusters):\n' + JSON.stringify(offload).slice(0, 4000) + '\n\n'
  + 'Order by severity then blast radius. Merge any dedup_clusters into a single entry. For each: severity, '
  + 'file:line, the fix, the regression test. Then call mcp__loci__investigation_store with investigation_id="' + INV + '", '
  + 'finding_type="observed", source="workflow:audit-remediate", tags="remediation,plan", and text = the full ranked plan. '
  + 'Return a concise markdown summary of the ranked plan for the operator.',
  { label: 'synthesize', phase: 'Synthesize' })

return { findings_planned: plans.length, backend_used: offload && offload.backend_used, synthesis }
