# FlyBrain PR Slicing Guardrails

## Purpose

Every PR must be a safe, reviewable, independently operable slice. A PR is not considered ready to merge unless it can be deployed, tested, and reversed without depending on another in-flight PR, branch, or hidden sequencing assumption.

This policy is designed to reduce rollout coupling, rollback ambiguity, and implementation drift across FlyBrain workstreams.

## Required PR slice checklist

Each PR must meet all of the following before merge:

- [ ] Independently deployable: The PR can be deployed on its own, without another PR being merged first, and without relying on a branch-specific config or code path.
- [ ] Independently reversible: The PR can be rolled back cleanly and safely without needing a second PR to undo or repair the change.
- [ ] Bounded in blast radius: The change is scoped to a small, understandable surface area and has a clearly identified impact boundary.
- [ ] Test-complete for its slice: The PR includes the evidence required to validate its behavior, including relevant unit/integration tests or equivalent verification.
- [ ] Free of hidden cross-PR dependencies: No unmerged work, migration ordering, contract assumptions, feature flags, or branch logic assumptions are required for the slice to work in isolation.

A PR is not “small enough” if the reviewer has to reconstruct a hidden cross-branch sequence to understand how it works.

## Anti-pattern examples

These are not acceptable PR slices:

### 1. Chain-PR deployment coupling

- PR A adds the database/schema changes.
- PR B assumes PR A is already live.
- PR C depends on both A and B to work in production.
- The deploy sequence is implicit, not documented, and not independently testable.

Why this fails:
- No PR is independently deployable.
- Rollback is ambiguous when the sequence breaks.
- Production behavior depends on branch timing instead of code correctness.

### 2. Migration split across PRs

- One PR adds a new table/field.
- A second PR updates readers.
- A third PR removes the old path.
- The code is intentionally broken between merges.

Why this fails:
- Hidden sequencing dependency.
- Reversibility is impossible without a coordinated rollback across multiple PRs.
- A single slice is not test-complete or deploy-safe.

### 3. Partial contract changes

- API contracts or serialization formats are changed in one PR.
- Consumer updates are shipped in another PR.
- Feature gates or compatibility shims are left in a half-updated state.

Why this fails:
- The system is not self-consistent within one PR.
- Hidden dependency on the consumer PR.
- Hard to reason about blast radius and rollback.

### 4. Feature-flag dependency traps

- A PR adds a toggle but relies on another PR to set the flag, enable the route, or populate required config.
- Behavior only works when another branch is merged or some environment-specific condition is satisfied.

Why this fails:
- Not independently deployable.
- Not independently reversible.
- Hidden environment assumptions are being treated as normal behavior.

### 5. Shared “cleanup” refactors that silently change behavior

- A PR mixes a large refactor with a behavior change, then asks reviewers to infer what is safe.
- The refactor is not a true no-op slice; it changes execution paths in subtle ways.

Why this fails:
- Blast radius is too wide.
- Review quality degrades and hidden dependencies are harder to detect.
- Test coverage is rarely sufficient for the full slice.

### 6. “It works on my branch” assumptions

- A PR relies on unmerged local changes, stale artifacts, or a branch-specific data setup outside the repo boundary.

Why this fails:
- The code is not portable or independently verifiable.
- Hidden dependency on local developer state is equivalent to hidden cross-PR coupling.

## Safe slicing principles

A PR should be safe if all of the following are true:

1. The branch can be deployed by itself.
2. The branch can be reverted by itself.
3. The branch touches only the targeted subsystem or behavior.
4. The PR’s tests validate the changed behavior in isolation.
5. The difference between “merged as a slice” and “not merged” is obvious.
6. Reviewers can explain the operational impact without needing a second PR or teammate memory.

If a PR cannot be described without naming another PR, it is not yet sliced correctly.

## Acceptance criteria template for PR descriptions

Use this template in every PR description:

```md
## Slice definition
- Scope of this PR:
- Explicitly out of scope:
- Why this is a single deployable slice:

## Independently deployable
- This PR does not require another merged PR to work.
- This PR does not depend on unmerged code from another branch.
- Any config/schema/flag changes are included in this PR or explicitly documented as not required.
- Deployment steps (if any):

## Independently reversible
- Rollback plan:
- Rollback commands or operational steps:
- Risks during rollback:
- Data migration or cleanup required for rollback:

## Blast radius
- Affected systems/services:
- Affected data stores or interfaces:
- Expected user or operator impact:
- Reason the change is bounded:

## Test completeness
- Tests added/updated:
- Coverage for changed behavior:
- Verification performed locally or in CI:
- Any gaps or follow-up validation required:

## Hidden dependency checks
- Any required feature flag or config? If yes, explain why it is included in this PR.
- Any required database or schema migration? If yes, explain ordering and rollback.
- Any required follow-up PR? If yes, explain why the slice is not complete.
- Any shared contracts or interfaces changed? If yes, list impacted consumers and compatibility plan.

## Reviewer sign-off
- [ ] I confirm this PR is independently deployable.
- [ ] I confirm this PR is independently reversible.
- [ ] I confirm this PR is bounded in scope.
- [ ] I confirm the changed slice is test-complete.
- [ ] I confirm there are no hidden cross-PR dependencies.
```

If any item is marked “no” or “not applicable without explanation,” the PR is not ready.

## Reviewer checklist

Reviewers should explicitly check each item before approving.

- [ ] I can explain the slice boundary in one paragraph without referencing another PR.
- [ ] The PR can be deployed alone without depending on a branch, merge order, or hidden local state.
- [ ] The PR can be rolled back alone without requiring another PR to undo its effect.
- [ ] The blast radius is small and the changed behavior is easy to reason about.
- [ ] The PR includes the tests needed to validate the slice, not just adjacent tests.
- [ ] All schema, config, flag, migration, and contract changes needed for the slice are included in the same PR or intentionally declared out of scope.
- [ ] There are no untracked assumptions like “this will work once the other branch merges.”
- [ ] The PR is not a partial migration or partial rollout disguised as a small change.
- [ ] The rollback path is documented and operationally realistic.
- [ ] Any follow-up work is clearly separated and does not create hidden dependency chains.
- [ ] The change does not widen impact to unrelated components without strong justification and review.
- [ ] The PR description is honest about risks, limits, and dependencies.

If a reviewer cannot answer “yes” to these, the PR should be returned to the author for slicing or documentation changes.

## Merge gate

PRs should be rejected when they rely on any of the following:

- another PR being merged first
- a shared branch or local-only code state
- incomplete migrations or staged rollout assumptions
- hidden interface or contract changes outside the PR
- ambiguous rollback steps
- vague “follow-up will clean this up” language

The default stance is: single-slice, single-rollback, single-verification story.

## Short operational rule

If the PR cannot stand alone, it is not a safe PR. It is a dependency chain disguised as a patch.
