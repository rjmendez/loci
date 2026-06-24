export const meta = {
  name: 'investigation-review',
  description: 'Load a Loci investigation, run parallel adversarial code review per cluster, verify the highest-severity claims, and synthesize a prioritized findings report. Works with any codebase.',
  phases: [
    { title: 'Load', detail: 'Pull all findings from the Loci investigation' },
    { title: 'Review', detail: 'Parallel per-cluster code review against actual files' },
    { title: 'Verify', detail: 'Adversarial verification of critical and high-severity clusters' },
    { title: 'Synthesize', detail: 'Prioritized report with verdicts and next actions' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
// Required:
//   investigation_id — the Loci investigation to review
//   repo_path        — absolute path to the repository on disk
//
// Optional:
//   clusters — array of {label, title, focus} describing review clusters.
//              If omitted, the workflow auto-clusters by finding tags.
//              Each cluster's `focus` is the full instruction for the review agent —
//              include file paths (relative to repo_path) and specific questions.
//   last_n_findings — how many findings to load (default 70)
// ─────────────────────────────────────────────────────────────────────────────
const A      = (typeof args === 'string') ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })() : (args || {})
const INV_ID = A.investigation_id
const REPO   = A.repo_path

if (!INV_ID || !REPO) {
  log('args.investigation_id and args.repo_path are both required.')
  return { error: 'missing_required_args' }
}

const LAST_N   = A.last_n_findings || 70
const CLUSTERS = A.clusters || null

// ── Schemas ───────────────────────────────────────────────────────────────────
const FINDINGS_SCHEMA = {
  type: 'object',
  required: ['hypothesis', 'findings'],
  properties: {
    hypothesis: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'type', 'text'],
        properties: {
          id:         { type: 'string' },
          type:       { type: 'string' },
          text:       { type: 'string' },
          tags:       { type: 'array', items: { type: 'string' } },
          confidence: { type: 'string' },
        },
      },
    },
  },
}

const AUTO_CLUSTER_SCHEMA = {
  type: 'object',
  required: ['clusters'],
  properties: {
    clusters: {
      type: 'array',
      items: {
        type: 'object',
        required: ['label', 'title', 'finding_ids', 'summary'],
        properties: {
          label:       { type: 'string' },
          title:       { type: 'string' },
          finding_ids: { type: 'array', items: { type: 'string' } },
          summary:     { type: 'string' },
        },
      },
    },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['cluster', 'confirmed', 'refuted', 'partial', 'severity', 'fix_complexity', 'key_detail', 'recommended_action'],
  properties: {
    cluster:            { type: 'string' },
    confirmed:          { type: 'array', items: { type: 'string' } },
    refuted:            { type: 'array', items: { type: 'string' } },
    partial:            { type: 'array', items: { type: 'string' } },
    severity:           { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    fix_complexity:     { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
    key_detail:         { type: 'string' },
    recommended_action: { type: 'string' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  required: ['cluster', 'verdict', 'reasoning', 'adjusted_severity', 'residual_risk'],
  properties: {
    cluster:           { type: 'string' },
    verdict:           { type: 'string', enum: ['confirmed', 'refuted', 'overstated', 'understated'] },
    reasoning:         { type: 'string' },
    adjusted_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    residual_risk:     { type: 'string' },
  },
}

const REPORT_SCHEMA = {
  type: 'object',
  required: ['findings_by_priority', 'stale_findings', 'new_gaps'],
  properties: {
    findings_by_priority: {
      type: 'array',
      items: {
        type: 'object',
        required: ['rank', 'cluster', 'severity', 'fix_complexity', 'one_line_summary', 'exact_action'],
        properties: {
          rank:             { type: 'number' },
          cluster:          { type: 'string' },
          severity:         { type: 'string' },
          fix_complexity:   { type: 'string' },
          one_line_summary: { type: 'string' },
          exact_action:     { type: 'string' },
        },
      },
    },
    stale_findings: { type: 'array', items: { type: 'string' } },
    new_gaps:       { type: 'array', items: { type: 'string' } },
  },
}

// ── Phase 1: Load ─────────────────────────────────────────────────────────────
phase('Load')

const inv = await agent(
  `Load the Loci investigation "${INV_ID}" and return ALL findings as structured JSON.
   Use mcp__loci__investigation_load with investigation_id="${INV_ID}" and last_n_findings=${LAST_N}.
   Return: { hypothesis: string, findings: Array<{id, type, text, tags, confidence}> }`,
  { label: 'load:investigation', phase: 'Load', schema: FINDINGS_SCHEMA }
)

log(`Loaded ${inv.findings.length} findings. Hypothesis: ${(inv.hypothesis || '').slice(0, 120)}...`)

// ── Determine clusters ────────────────────────────────────────────────────────
let activeClusters = CLUSTERS

if (!activeClusters) {
  // Auto-cluster by unique tags in the findings
  const autoClusterResult = await agent(
    `You have ${inv.findings.length} investigation findings from "${INV_ID}".
Findings: ${JSON.stringify(inv.findings.slice(0, 60), null, 1)}

Group these findings into 3-8 meaningful review clusters based on their tags, types, and topics.
For each cluster provide:
- label: short kebab-case identifier
- title: human-readable cluster name
- finding_ids: array of finding ids in this cluster
- summary: one sentence describing what this cluster is about

Make clusters thematically coherent — group by subsystem, severity class, or fix area.`,
    { label: 'auto-cluster', phase: 'Load', schema: AUTO_CLUSTER_SCHEMA }
  )

  // Convert auto-clusters to the same format as user-provided clusters.
  // Auto-cluster agents summarize findings; the review agent gets that summary as focus.
  activeClusters = (autoClusterResult.clusters || []).map(c => ({
    label:   c.label,
    title:   c.title,
    focus:   `${c.summary}\n\nFinding IDs in this cluster: ${c.finding_ids.join(', ')}\n\nRead the relevant source files in ${REPO} to verify each finding. Check if the issues described actually exist in the current code.`,
  }))

  log(`Auto-clustered into ${activeClusters.length} clusters: ${activeClusters.map(c => c.label).join(', ')}`)
}

// ── Phase 2: Review ───────────────────────────────────────────────────────────
phase('Review')

const reviews = await parallel(
  activeClusters.map(c => () => agent(
    `You are reviewing the codebase at ${REPO} for a specific cluster of investigation findings.

Cluster: ${c.title}

Task: ${c.focus}

Read the actual source files to verify each claim. Return a structured verdict with:
- confirmed: list of findings confirmed accurate (quote the relevant code or observation)
- refuted: list of findings that are wrong or stale (explain what you found instead)
- partial: findings that are partially correct (with what's wrong or outdated)
- severity: critical/high/medium/low for this cluster overall
- fix_complexity: trivial/small/medium/large for the highest-priority fix
- key_detail: the single most important thing to know about this cluster
- recommended_action: concrete next step (file:line or exact command)`,
    {
      label: `review:${c.label}`,
      phase: 'Review',
      schema: REVIEW_SCHEMA,
    }
  ))
)

const validReviews = reviews.filter(Boolean)
log(`${validReviews.length}/${activeClusters.length} cluster reviews completed`)

// ── Phase 3: Adversarial verify highest-severity clusters ─────────────────────
phase('Verify')

const criticalAndHigh = validReviews.filter(r => r.severity === 'critical' || r.severity === 'high')
log(`Adversarially verifying ${criticalAndHigh.length} critical/high clusters`)

const verifications = await parallel(
  criticalAndHigh.map(review => () => agent(
    `You are an adversarial verifier. Your job is to CHALLENGE and attempt to REFUTE this finding cluster review.

Cluster: ${review.cluster}
Severity claimed: ${review.severity}
Key detail: ${review.key_detail}
Recommended action: ${review.recommended_action}
Confirmed findings: ${(review.confirmed || []).join('; ')}

Be skeptical. Check the actual files in ${REPO}. Try to find reasons this is WRONG or OVERSTATED.

Return:
- verdict: "confirmed" | "refuted" | "overstated" | "understated"
- reasoning: what you checked and what you found
- adjusted_severity: what severity it actually deserves
- residual_risk: what risk remains even after the fix is applied`,
    {
      label: `verify:${review.cluster}`,
      phase: 'Verify',
      schema: VERIFY_SCHEMA,
    }
  ))
)

const validVerifications = verifications.filter(Boolean)
log(`${validVerifications.length} adversarial verifications completed`)

// ── Phase 4: Synthesize ───────────────────────────────────────────────────────
phase('Synthesize')

const report = await agent(
  `You are synthesizing a final prioritized findings report for investigation "${INV_ID}".

INVESTIGATION HYPOTHESIS:
${inv.hypothesis}

CLUSTER REVIEWS (${validReviews.length} clusters):
${JSON.stringify(validReviews, null, 2)}

ADVERSARIAL VERIFICATIONS (${validVerifications.length} verified):
${JSON.stringify(validVerifications, null, 2)}

Produce a final report with:
1. findings_by_priority: rank all confirmed findings by (severity × urgency ÷ fix_complexity), each with:
   - rank, cluster, severity (post-verification), fix_complexity, one_line_summary, exact_action (file:line or command)
2. stale_findings: any investigation findings now confirmed resolved or stale (should be memory_retract'd)
3. new_gaps: anything discovered during review that should be added to the investigation via investigation_store`,
  {
    label: 'synthesize:report',
    phase: 'Synthesize',
    schema: REPORT_SCHEMA,
  }
)

return {
  investigation_id: INV_ID,
  findings_loaded:  inv.findings.length,
  clusters_reviewed: validReviews.length,
  verifications:    validVerifications.length,
  reviews:          validReviews,
  verifications_detail: validVerifications,
  report,
}
