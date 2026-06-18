# Agent Operations Rules
# Distilled from: agent-orchestration cluster (18 skills)
# Generated: 2026-06-13

## Human Gates

All multi-step orchestration pipelines must pause at two explicit human gates:
Gate 1 before implementation begins (approve the plan/scope) and Gate 2 before
any commit or deployment (confirm the diff). Do not collapse these gates for
speed; a user can explicitly waive them but the orchestrator must not skip them
silently. For destructive or security-sensitive operations, add a dedicated
security-review pass before Gate 2.
See skill: orch-pipeline, orch-add-feature, claude-devfleet.

## Parallel Isolation

Before dispatching parallel agents, build an explicit lane matrix that maps each
lane to its write surface (file paths, DB tables, branches, services). Only run
lanes in parallel when their write surfaces are disjoint. Gate destructive
operations (migrations, deploys, permission changes) as sequential steps with
explicit approval, never parallelized. Merge results back through a single
integration step after all lanes complete.
See skill: parallel-execution-optimizer, ralphinho-rfc-pipeline, dmux-workflows.

## Subagent Context

Every subagent invocation must supply a self-contained context bundle: goal,
relevant file paths or excerpts, constraints, and the exact toolsets permitted.
Do not rely on ambient session state or prior conversation history being visible
to the subagent — treat each delegation as a fresh process with zero inherited
memory. Use structured output shapes (status, summary, artifacts, next_actions)
so the orchestrator can verify what the subagent actually did.
See skill: team-agent-orchestration, plan-orchestrate, agent-harness-construction.

## Adversarial Eval

Any output destined for production, end users, or downstream agents must be
evaluated by a separate agent or pipeline step that did not produce it. Configure
the evaluator with a strict rubric before generation begins, not after. For
critical paths (compliance, security, customer-facing content), use two
independent reviewers whose verdicts must both pass before the output ships.
See skill: gan-style-harness, santa-method, continuous-agent-loop.

## Done Signal

Before any agent loop or delegated task begins, specify the done signal: a
concrete, machine-checkable condition that ends the loop. For multi-unit work,
each unit must carry its own acceptance tests. Every error path must have an
explicit stop condition that prevents unbounded retries on the same root cause.
Reuse this criteria as the evaluator rubric in any GAN-style or adversarial
review pass.
See skill: parallel-execution-optimizer, ralphinho-rfc-pipeline, agent-harness-construction.

## Recovery Contract

Every work unit, lane, or delegated task must declare its rollback plan before
execution starts: what to undo, where to snapshot current state, and what the
escalation path is if the unit exceeds its retry budget. When a unit stalls,
evict it from the active queue, preserve its findings, and re-plan with a
narrowed scope rather than retrying the same scope indefinitely.
See skill: ralphinho-rfc-pipeline, agent-harness-construction, continuous-agent-loop.

## Right-Sizing

Before orchestrating any task, classify its blast radius across at least three
signals: files or services touched, new dependencies or contracts introduced, and
design ambiguity. Run only the pipeline phases warranted by the highest tier
reached. Trivial one-shot tasks should be executed inline without spawning
harnesses; large-blast tasks require full research-plan-TDD-review-gate pipelines.
State the chosen tier explicitly so the user can override before work begins.
See skill: orch-pipeline, dynamic-workflow-mode, autonomous-loops.

## Autonomy Consent

Autonomous capabilities — scheduled jobs, remote agent dispatch, persistent
memory writes, computer control, external posting, and third-party resource
modification — must each be individually approved by the user before activation.
Default to dry-run plans and local queue files; only promote to live recurring
execution after explicit confirmation. Do not let any background process or
schedule outlive the current session unless the user has explicitly requested a
continuing service.
See skill: autonomous-agent-harness, parallel-execution-optimizer.

## Structured Handoffs

Every agent or tool invocation must produce a structured output containing at
minimum: a status (success/warning/error), a one-line summary of what was done,
a list of artifacts created or modified (file paths, IDs), and actionable next
steps. For multi-unit pipelines, each unit must also emit a handoff document
describing what is complete, what is blocked, and how to resume. Do not ship
opaque output that requires the orchestrator to infer state.
See skill: agent-harness-construction, dynamic-workflow-mode, claude-devfleet.

## Dependency Ordering

Every work unit must declare its dependencies explicitly before dispatch. A unit
may not begin execution until all units it depends on have passed their acceptance
criteria and been merged to the integration branch. After each merge, re-run
integration tests before dispatching the next dependent unit. If a lane discovers
a blocker, pause all dependent work and update the dependency graph before
retrying.
See skill: ralphinho-rfc-pipeline, claude-devfleet, parallel-execution-optimizer.

## Subagent Role Taxonomy

Heterogeneous role assignment (SEARCHER / EDITOR / VERIFIER) reduces redundant capability
overlap and improves throughput on decomposed orchestration tasks. Assign one role per
subagent invocation and restrict context to only the files and tools needed for that role.

| Role | Permitted tools | Use for |
|---|---|---|
| SEARCHER | code-search skills, read-only Bash (grep/find/cat), Read | Discovery, symbol lookup, dependency mapping |
| EDITOR | Read, Edit, Write, build/test Bash | Implementation, file changes, test execution |
| VERIFIER | test/lint/diff Bash, Read | Correctness checks, diff review, acceptance criteria |

A SEARCHER that produces findings does not run EDITOR work — synthesis and dispatch stay
in the orchestrator. Prefix each subagent prompt with its assigned role so tool access intent
is unambiguous. Source: research finding from Task-Driven Co-Design of Heterogeneous Multi-Robot Systems.
