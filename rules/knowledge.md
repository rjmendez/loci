# Knowledge and Skill Management Rules
# Distilled from: knowledge-skill-mgmt cluster (21 skills)
# Generated: 2026-06-13

## Audit First

Before modifying, merging, deleting, or recommending changes to any workspace
surface — skills, hooks, MCP servers, automations, or knowledge stores — produce
a read-only inventory first. Do not act on assumed state; act only on observed
state. The inventory is the evidence record that makes every subsequent change
reviewable and reversible. When in doubt, one more read before the first write.
See skill: automation-audit-ops, workspace-surface-audit, rules-distill.

## Live Sources

When a question's answer depends on the current state of files, configs, or
external services, read the relevant files or call a fresh search before
replying. Do not substitute a remembered or inferred answer for a live read when
the live read is cheap. This applies equally to ECC catalog counts, MCP
availability, and whether a specific automation is actually running.
See skill: ecc-guide, research-ops, automation-audit-ops.

## Evidence Separation

Every output that mixes different confidence levels must label each claim's
epistemic status: observed fact, user-supplied evidence, inferred, or
recommended. Never write a recommendation that looks like a confirmed state.
Never collapse "configured" and "verified working" into a single status. When
the distinction matters — research reports, audit outputs, decision ledgers —
use explicit headers or inline tags for each tier.
See skill: research-ops, automation-audit-ops, recursive-decision-ledger.

## Read-Only Default

Any skill that reads, audits, or plans must remain read-only until the user
explicitly requests implementation. Destructive actions — merging overlapping
surfaces, deleting duplicates, executing live deploys — must be gated by
explicit user approval after the evidence is presented, not implied by the audit
completing successfully. Default posture is dry-run or preview.
See skill: automation-audit-ops, recursive-decision-ledger, workspace-surface-audit.

## Search Before Build

Before creating a new skill, writing a custom utility, or installing a new
plugin, search local skills, marketplace skills, package registries, and GitHub
for an existing implementation. Prefer adopting or wrapping what already exists
over creating from scratch. Only build custom when the search is exhaustive and
no suitable match is found.
See skill: skill-scout, search-first.

## Context Budget

Keep each skill file under 4 KB and each plugin under 20 tool surfaces to
minimise per-turn overhead. Monitor context consumption actively; when a session
approaches window limits, compact at a logical task boundary rather than letting
auto-compaction trigger mid-operation. For subagent workflows, pass only the
context the subagent needs and retrieve additional context iteratively.
See skill: context-budget, strategic-compact, iterative-retrieval.

## External Vetting

Before adopting, installing, or forking any externally sourced skill, script, or
repository, read its full source. Check for unexpected shell commands, file
writes outside expected paths, network calls, credential access, or package
installs. Prefer reviewing from a fresh branch before merging. Never recommend
an external component for adoption without completing this check.
See skill: skill-scout, repo-scan.

## Persist To Durable

Any significant output — research findings, experiment metrics, architectural
decisions, audit results, or learned patterns — must be written to a durable
cross-session store before the session ends. Transient session context is not
persistence. Use the appropriate layer: MCP memory for queryable structured
knowledge, project memory files for quick-access context, and the knowledge base
for curated long-form documents.
See skill: knowledge-ops, mle-workflow, ck, research-ops.

## Phased Workflow

Structure audit, research, and analysis tasks as explicit phases: first collect
facts deterministically (run scripts, read files, enumerate surfaces), then apply
LLM judgment over the complete collected context, and only then produce
recommendations or take action. Never merge the collection and judgment phases
into a single speculative pass. Deterministic collection provides the evidence
base that makes LLM verdicts trustworthy.
See skill: rules-distill, skill-stocktake, context-budget.

## Context Before Questions

Before asking the user any clarifying question, read the repository, existing
docs, schemas, config files, and any other available context that could answer
the question without user input. Only ask questions whose answers cannot be
inferred from available artifacts and that materially change scope or approach.
Present auto-detected drafts for confirmation rather than open-ended questions.
See skill: intent-driven-development, deep-research, ck.

## Project Scope Isolation

When capturing learned patterns, instincts, or conventions, default to project
scope first. Promote a pattern to global scope only after it has been validated
across two or more distinct projects, or when it is explicitly domain-agnostic.
Treat project-level skills, instincts, and context files as isolated from the
global surface until the evidence supports generalisation.
See skill: continuous-learning-v2, skill-stocktake, knowledge-ops.

## Parallel Collection Merge

For tasks requiring broad collection across many files, sources, or candidates,
spawn parallel subagents for concurrent collection and then merge results in a
single cross-read pass before making verdicts or recommendations. The collection
phase is parallelisable; the judgment phase requires the full merged context.
After parallel collection, deduplicate and re-check cross-cutting criteria on the
combined evidence before finalising.
See skill: rules-distill, deep-research, search-first.
