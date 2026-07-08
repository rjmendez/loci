export const meta = {
  name: 'judge-eval',
  description: 'Judge-based relevance grading — one strict agent per query grades its candidate pool',
  whenToUse: 'Step 2 of the judge-based A/B retrieval eval. args = {file, query_ids, show}. Each agent reads its own query+pool from disk (no huge inline payload) and returns relevant ids -> {grades: {query_id: [relevant_ids]}}.',
  phases: [{ title: 'Judge', detail: 'one relevance-judge agent per query, in parallel' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const FILE = A.file || ''
const SHOW = A.show || ''
const IDS = Array.isArray(A.query_ids) ? A.query_ids : []
if (!FILE || !SHOW || !IDS.length) { log('need file + show + query_ids'); return { grades: {} } }

phase('Judge')
const results = await parallel(IDS.map((qid) => () =>
  agent(
    'You are a STRICT retrieval-relevance judge. First load your query and its candidate documents by running:\n'
    + '  python3 ' + SHOW + ' ' + FILE + ' ' + qid + '\n'
    + 'It prints QUERY then each candidate as [id] followed by its text. Decide which documents are '
    + 'genuinely RELEVANT to the query — ones the person issuing this query would actually want. Judge '
    + 'topical/semantic relevance, NOT mere keyword overlap; be strict; exclude a near-verbatim restatement '
    + 'of the query itself if present. Return ONLY the relevant document ids.',
    { label: 'judge:' + String(qid).slice(0, 8), phase: 'Judge', effort: 'low',
      schema: { type: 'object', required: ['relevant'],
        properties: { relevant: { type: 'array', items: { type: 'string' } } } } })
    .then((r) => ({ id: qid, relevant: (r && Array.isArray(r.relevant)) ? r.relevant : [] }))
    .catch(() => ({ id: qid, relevant: [] })),
))

const grades = {}
let total = 0
for (const r of results.filter(Boolean)) { grades[r.id] = r.relevant; total += r.relevant.length }
log('judged ' + Object.keys(grades).length + ' queries, ' + total + ' relevant marks')
return { grades }
