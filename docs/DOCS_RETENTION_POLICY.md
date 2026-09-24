# Docs Retention and Archival Policy

This policy defines how Loci keeps dense documentation available without making it the default path. The goal is to preserve operational knowledge, avoid dead ends, and keep the canonical reading path short.

## Scope

This policy applies to repository documentation under `docs/` and to any linked appendix material that is intentionally retained but de-emphasized.

It is the repo-wide analogue of the FlyBrain-specific prune plan: keep the active path short, keep dense material searchable, and require an explicit redirect or summary before anything is retired.

## Retention classes

### 1. Active canonical

Active canonical docs are the first stop for operators and maintainers.

Keep a document in this class when it is:

- the primary entry point for a subsystem
- required for day-to-day operations or deployment
- the shortest accurate description of a live workflow
- the authoritative reference for a currently supported API or tool family

These docs should stay concise, current, and link outward to deeper material instead of repeating it.

### 2. Active supporting

Supporting docs are still current and fully retained, but they are not the main landing page.

Use this class for:

- operational runbooks
- technical references with enough detail to support implementation
- broad architecture notes that are still consulted regularly

Supporting docs remain discoverable from the main index and should be updated when the underlying flow changes.

### 3. Archived dense

Archived dense docs are still valid and searchable, but they are no longer the default reading path.

Use this class for:

- comparative studies
- design notes with high historical value
- specialized references that are useful only after the canonical doc has been read
- dense appendices that would crowd the active path if left uncurated

Archived dense docs stay in the tree, stay linked from an archive index, and stay searchable. They are de-emphasized, not hidden.

### 4. Redirected legacy

Redirected legacy docs are superseded files that are kept only to preserve navigation.

These files should contain a short pointer to the canonical replacement and, if needed, the archive entry that holds the retained detail.

Do not leave an old path orphaned after a summary split or rename.

## Lifecycle

1. **Create or update the active canonical doc first.** This is the source of truth for the current workflow.
2. **Move dense detail to supporting or archive material when the canonical path gets too heavy.** Keep the summary short and link to the deeper page.
3. **Retire only after a redirect exists.** If a page is no longer the right front door, leave a pointer behind.
4. **Reclassify when demand changes.** If an archived doc becomes a frequent source of support or a workflow changes materially, refresh the canonical summary and move the doc back toward active support.

## Refresh and archive behavior

Archive decisions should follow the same idempotent, fail-open pattern used elsewhere in Loci:

- prefer summary-before-retire over deletion
- preserve links and inbound references
- keep the archive searchable
- avoid silent demotion of the only current explanation of a workflow

Dense docs should stay active when they are still used by:

- operator guidance
- deployment or recovery steps
- change-detection or review workflows
- backfill or reconciliation jobs
- search results that repeatedly land on the page as the best available answer

Dense docs should be archived when they are:

- stable but no longer a default landing page
- too detailed for the canonical path
- still valuable as historical or implementation reference
- sufficiently covered by a shorter active summary

## Operator guidance

Use this decision order:

1. **Is this the shortest accurate front door?** Keep it active canonical.
2. **Is it still needed for operations or implementation?** Keep it active supporting.
3. **Is it dense but still useful?** Move it behind the archive index.
4. **Is it superseded?** Leave a redirect and keep the replacement discoverable.

When in doubt, preserve availability and lower prominence rather than deleting content.

## Alignment with Loci maintenance flows

This policy matches the repo's existing maintenance shape:

- search paths should still find archived material when it is the best answer
- backfill and reconciliation should restore missing coverage without changing the canonical order of documents
- change-detection should trigger refresh of the active summary, not silent removal of the older dense source
- archive indexes should point readers back to canonical docs instead of creating dead ends

The practical rule is simple: keep dense material available, but make the shortest correct path the default path.

