export const meta = {
  name: 'loci-model-tier-design',
  description: 'Audit every local + low-cost-cloud generation path in Loci and design the tier that should serve each',
  whenToUse: 'After the memcheck/llm_local endpoint fixes. Every consumer is already written; the question is which tier serves it and whether anyone reads the output.',
  phases: [
    { title: 'Census',   detail: 'One agent per surface: who generates, on which tier, and does the output have a reader' },
    { title: 'Design',   detail: 'Route each consumer to a tier, on measured cost/latency/quality — or say do not run it' },
    { title: 'Refute',   detail: 'Adversarial pass: is the routing justified, or is it a preference dressed as a measurement' },
  ],
}

const GROUND = (typeof args === 'object' && args && args.ground) || ''
const REPO = (typeof args === 'object' && args && args.repo) || '.'

const RULES = `
DISCIPLINE — mistakes this codebase has already paid for, several of them today:

1. MEASURE, do not assert. Lead with the number and how you got it. Mark inference
   as inference and say what would falsify it.
2. RUN it. py_compile and ruff pass code that raises NameError at runtime. A probe
   that returns nothing may be testing the wrong endpoint with stale code — today a
   "27B produces empty output" result was entirely an artefact of exactly that.
3. AVAILABILITY IS NOT CAPABILITY. llm_available() returned True for weeks while
   every call returned None: a provider was selected on key-presence, the key had a
   newline from an 80-column paste, and behind that the account had no credit. Three
   stacked failures, all swallowed by one fail-open. Check that a thing WORKS, not
   that it is configured.
4. ZERO OUTPUT IS NOT AUTOMATICALLY A DEFECT. Check whether the producer ever RAN
   before calling it broken. An issue was filed on that error and had to be retracted.
5. A LANE WITH NO READER IS NOT WORTH FEEDING. finding_verifications.jsonl had 0
   readers; the audit lane had 667 writes and 0 reads. Before recommending that
   something generate more, establish who consumes it.
6. Comment economy: rationale goes in the commit message, not a wall of source
   comments.
`

const CENSUS_SCHEMA = {
  type: 'object',
  properties: {
    surface: { type: 'string' },
    entrypoints: { type: 'array', items: { type: 'string' } },
    current_tier: { type: 'string' },
    works_today: { type: 'boolean' },
    evidence: { type: 'string' },
    output_destination: { type: 'string' },
    reader_count: { type: 'integer' },
    reader_evidence: { type: 'string' },
    call_volume: { type: 'string' },
    quality_bar: { type: 'string' },
  },
  required: ['surface', 'entrypoints', 'current_tier', 'works_today', 'evidence',
             'output_destination', 'reader_count', 'reader_evidence'],
}

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    routing: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          surface: { type: 'string' },
          recommended_tier: { type: 'string' },
          why: { type: 'string' },
          measured_basis: { type: 'string' },
          cost_per_run: { type: 'string' },
          do_not_run: { type: 'boolean' },
        },
        required: ['surface', 'recommended_tier', 'why', 'measured_basis'],
      },
    },
    tier_table: { type: 'string' },
    unmeasured: { type: 'array', items: { type: 'string' } },
    cheapest_next_experiment: { type: 'string' },
  },
  required: ['routing', 'tier_table', 'unmeasured', 'cheapest_next_experiment'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    unjustified_routings: { type: 'array', items: { type: 'string' } },
    safe_to_adopt: { type: 'boolean' },
  },
  required: ['refuted', 'strongest_objection', 'safe_to_adopt'],
}

const SURFACES = [
  {
    id: 'memcheck-llm',
    brief: `mcp/memcheck/llm.py — call_llm(), the shared generation primitive. Just fixed:
provider fallthrough (anthropic -> copilot -> ollama), whitespace-key guard,
_ollama_gen_base() for generation. Its consumers: investigation_reflect summaries
(summary_l1/l2), investigation_reason, _run_causal_inference's LLM slow path, and
memcheck/checks/contradiction_llm.py behind the llm_verify flag.
For EACH consumer establish: does it run now, what does it write, and who reads
what it writes. summary_l1/l2 sit on the manifest — find out whether anything
consumes them (grounding.py is a candidate).`,
  },
  {
    id: 'llm-local-and-batched',
    brief: `mcp/llm_local.py (single-shot, Ollama + vLLM fallback) and mcp/batched_gen.py
(vLLM /v1/completions). Consumers include verify.verify_finding, query_expand /
rag_context_search expansion, classify_text, compress_text, semantic_relevance,
and scripts/loci_groom.py's tags pass.
Note the groom 'tags' pass is DELIBERATELY not scheduled: kNN tagging measured
better at a fraction of the cost. Do not recommend reviving it without new
measurement. Establish per-consumer call volume from the tool-audit log at
~/.hermes/logs/tool-audit.log.`,
  },
  {
    id: 'cloud-tiers',
    brief: `mcp/openrouter.py (FREE_LADDER / CHEAP_LADDER / DEFAULT_LADDER, key labelled
"oxalis loci" in ~/.loci/backends.toml) and the Anthropic path in memcheck/llm.py.
MEASURED TODAY: the Anthropic account returns HTTP 400 "credit balance is too low",
so that tier costs nothing because it cannot transact. Verify whether OpenRouter's
free ladder actually answers right now — earlier in this project the free models
returned 429 "Provider returned error" and the ladder fell through to a paid one.
Establish what is actually reachable and at what price, and whether anything in
Loci currently routes to either.`,
  },
]

phase('Census')
log(`Censusing ${SURFACES.length} generation surfaces`)

const census = await parallel(SURFACES.map(s => () =>
  agent(
    `${GROUND}\n${RULES}\n
Repo: ${REPO}. Census this generation surface — do NOT design anything yet.

SURFACE: ${s.id}
${s.brief}

For every entrypoint answer, with evidence:
  - does it WORK today? Run it. "Configured" is not "works".
  - where does its output go, and HOW MANY READERS does that destination have?
    Grep for consumers; a file nothing reads is a lane, not a feature.
  - what is the call volume? ~/.hermes/logs/tool-audit.log has real counts.
  - what quality bar does the consumer actually need?

Use ./mcp/.venv/bin/python for anything importing qdrant_client or the mcp sdk.
scripts/loci_groom.py load_env() sets QDRANT_URL — without it RAG returns nothing
and you will measure your own harness instead of the system.`,
    { label: `census:${s.id}`, phase: 'Census', schema: CENSUS_SCHEMA,
      model: 'opus', effort: 'high' }
  )
))

const facts = census.filter(Boolean)
log(`${facts.length}/${SURFACES.length} surfaces censused`)
for (const f of facts) log(`  ${f.surface}: works=${f.works_today} readers=${f.reader_count}`)

phase('Design')
const design = await agent(
  `${GROUND}\n${RULES}\n
Here is the measured census of every generation surface in Loci:

${JSON.stringify(facts, null, 2)}

Design the tier routing. Available tiers, with what is known today:
  - local Ollama on oxalis (100.73.200.19): 11 models incl. qwen2.5:3b,
    heretic-llama31-8b-instruct, heretic-phi4-mini-reasoning, and a newly pulled
    Qwen3.8-27B-Heretic-Abliterated Q4_K_M (17GB). GPUs: 2080 Ti (11.8GB, SM 7.5)
    + 4070 Ti (12.9GB, SM 8.9). Persistence is a SYSTEM startup Scheduled Task,
    but the live process is not that task's child.
  - local vLLM at 127.0.0.1:18000 serving Qwen/Qwen2.5-3B-Instruct — note this is
    ANOTHER PROJECT's forwarder (dama-vllm), not Loci's to depend on.
  - OpenRouter free/cheap ladders.
  - Anthropic — measured today as unable to transact (400, no credit).

For each consumer give a tier and the MEASURED basis. Where a consumer's output
has no reader, recommend do_not_run=true and say so plainly rather than routing it
to a cheap tier — a cheaper way to write to nobody is not an improvement.

List everything you could NOT measure in 'unmeasured', and give the single
cheapest experiment that would resolve the most important unknown.`,
  { label: 'design:tiers', phase: 'Design', schema: DESIGN_SCHEMA,
    model: 'opus', effort: 'high' }
)

if (!design) return { census: facts, design: null, verdict: null }

phase('Refute')
const verdict = await agent(
  `${RULES}\n
Adversarially review this tier routing. Try to REFUTE it. Default to refuted=true
when uncertain.

CENSUS: ${JSON.stringify(facts).slice(0, 12000)}

DESIGN: ${JSON.stringify(design, null, 2).slice(0, 14000)}

Attack specifically:
  - Is any routing a PREFERENCE dressed as a measurement? "bigger is better" and
    "local is cheaper" are not measurements. Name every routing whose
    measured_basis does not actually support it.
  - Does any recommendation feed a lane with no reader? That is the failure this
    project has hit three times.
  - Does anything depend on a service Loci does not own — specifically the
    dama-vllm forwarder at 127.0.0.1:18000, or an oxalis Ollama whose live process
    is not managed by its startup task?
  - Does any routing assume a model works without it having been RUN?
  - Is the cheapest_next_experiment actually the cheapest, and does it resolve the
    most important unknown, or merely the most interesting one?

safe_to_adopt=false unless you would defend this routing yourself.`,
  { label: 'refute:tiers', phase: 'Refute', schema: VERDICT_SCHEMA,
    model: 'opus', effort: 'xhigh' }
)

return { census: facts, design, verdict }
