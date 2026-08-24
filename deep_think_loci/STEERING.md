# Steerable deep-think — design note

Today a run is fire-and-forget: `Workflow({scriptPath})`, seven agents, one answer,
~$5, and the only way to change your mind is to kill it and start over. The goal
is to be able to redirect the thinking *while it thinks* — and to move model
routing from five string literals in the script to a decision that can be revised
mid-run.

This note is the design, not the implementation. It is written against what the
tools actually permit, which turns out to shape most of it.

## The one constraint that shapes everything

**A workflow script cannot read Loci.** It has `agent()`, `parallel()`,
`pipeline()`, `phase()` and `log()` — no filesystem, no MCP. `loci-native.js`
already documents this: grounding is produced *before* the workflow starts and
injected through `args`, precisely because the script itself cannot fetch it.

But **agents can**. deep-think's own prompts tell them to call
`mcp__loci__rag_context_search`. So the script cannot poll for guidance — an agent
can, and hand the result back as the return value of a stage.

That single fact settles the shape: **steering is a cheap agent between phases,
not a callback into the script.**

## The substrate already exists

Nothing here needs new storage. Three things are already in place:

| primitive | what it gives us |
|---|---|
| `investigation_note(id, field, value)` | an operator-writable manifest per investigation — `hypothesis`, `next_step`, `open_question_add`, `context`. This *is* a steering surface; deep-think simply never reads it. |
| `memory_hints(id, since_ts)` | incremental poll of what a run has produced since a timestamp — built for exactly this. |
| `record_type` | not validated against a fixed vocabulary, so a `directive` record is a convention, not a schema change. |

The investigation is already the run's shared state. Making it the *control plane*
costs a read.

## Design: the investigation is the control plane

### 1. Directives

An operator (or a supervising model) writes a directive into the same
investigation the run is writing to:

```
investigation_store(investigation_id=RUN, record_type="directive",
                    text="drop target 3, it is out of scope",
                    tags=["phase:ideate", "priority:high"])
```

Or, for the coarse steer, just `investigation_note(RUN, "next_step", "...")`.

Directives are scoped by tag: `phase:*` targets a stage, untagged applies to all
remaining stages. They are findings, so they are timestamped, attributable and
auditable — the run records not just what it concluded but what it was told.

### 2. The steering agent

Between phases, one cheap agent — haiku, single purpose:

```
read investigation RUN's manifest and every record_type="directive" since <ts>;
return {directives: [...], hypothesis, next_step, halt: bool}
```

Its return value is threaded into the next phase's prompts, the same way `args.ground`
is threaded today. Cost is one small agent per phase boundary — call it 4 extra
agents on a 7-agent run, well under the `~$0.15-per-agent` floor the README already
records.

### 3. Phase gates that cannot hang

After writing its checkpoint, a phase may wait for an acknowledging directive —
but **always with a deadline and a default of proceed**. An unattended run must
behave exactly as it does today; a watched run pauses long enough to be redirected.

```
gate(phase) := poll for directives up to T seconds
               → halt directive  : stop, write closed_summary
               → steer directive : fold into the next phase's prompt
               → nothing by T    : proceed unchanged
```

`halt` is the one directive that must be honoured immediately, because the reason
to want it is usually "this run is going the wrong way and costing money".

## Model routing as data

Routing is currently `model: 'haiku'` ×4 and `model: 'opus'` ×2, literal in the
script. Make it a policy the run carries and a directive can revise:

```
tiers: {
  ideate:   { tier: "cheap-parallel" },
  write:    { tier: "mechanical"     },
  verify:   { tier: "reason"         },
  synth:    { tier: "judgement"      },
}
```

and resolve `tier → concrete model` at dispatch. The resolver is the ladder built
for the passive tier, extended with the Claude tiers:

| tier | resolves to | because |
|---|---|---|
| `mechanical` | embeddings, or a free/cheap remote model | measured: kNN beat every model at tagging, 2× the precision at a tenth of the time |
| `cheap-parallel` | free ladder → cheap paid → haiku | high volume, low judgement; free availability is opportunistic and the ladder absorbs the 429s |
| `reason` | haiku / sonnet | needs to be right, not deep |
| `judgement` | **opus** | adversarial synthesis and red-teaming. v3.2 removed the external tier for good reason and nothing here revisits that — this is where Anthropic models are the correct tool, not a cost to optimise |

Two things this buys beyond flexibility. A directive can say *"you are going in
circles, escalate the next synthesis to opus"* — or the reverse, *"this target is
mechanical, stop spending judgement on it"*. And the run records which tier served
each finding, so "did the expensive tier actually earn it" becomes a measurable
question instead of an intuition.

## Merging Loci's memory into the thinking

Today each run re-derives its grounding from raw RAG. The corpus now has structure
the run never sees:

- **code links** — `Finding → CodeSymbol` edges, so a target named as a subsystem
  can pull the findings that touch its symbols, not just the ones that read similar
- **related prior investigations** — `investigation_related_cases` and
  `related_investigations_via_code` already answer "who has thought about this
  before"; a fresh run currently starts as though nobody had
- **conflicts** — `_detect_conflicts` has produced records since it started working;
  a contradiction found in an earlier run is exactly what a new one should open with
- **the manifest** — prior `hypothesis` / `open_question` entries are the cheapest
  possible seed for ideation

So phase 0 becomes *assemble*, not *retrieve*: build the run's opening context from
the groomed graph, and let RAG fill the gaps rather than carry the whole load. That
is the merge — the passive tier maintains the structure, deep-think consumes it, and
each run's findings feed back into what the passive tier grooms next.

## Two modes, one set of phases

The phases should be **separately invocable**. Then:

- **unattended** — a wrapper runs them back to back with `T=0` gates. Identical to
  today's behaviour.
- **driven** — the session invokes one phase at a time and the operator steers
  between them in conversation. No control plane needed at all; the gate is the
  human.

The second mode is worth building first *because it is nearly free*: it is the
current script split at its existing phase boundaries. The directive channel is
what makes the first mode as steerable as the second, and it can follow.

## Order of work

1. **Split the workflow at its phase boundaries** so each phase can be invoked
   alone. Buys interactive steering immediately, with no new machinery.
2. **Phase 0 assembles from the graph** — code links, related cases, prior
   manifest — instead of RAG-only grounding.
3. **Tier table**, replacing the literal model names; routing recorded per finding.
4. **Steering agent + directive records**, giving mode one the same steerability as
   mode two.
5. **Escalation directives** — the point at which a supervising model, not just a
   person, can redirect a run.

Steps 1 and 2 are worth doing regardless of whether the rest ever happens: one
makes the run interruptible, the other makes it start from what is already known.

## Open questions

- **What does a directive mean to work already dispatched?** A `parallel()` stage
  is in flight; a directive arriving mid-stage cannot reach it. Simplest honest
  answer: directives apply at the next boundary and the run says so.
- **How is a steering agent's read authenticated?** It reads the same investigation
  the run writes. If a directive can redirect an expensive run, writing one is a
  privileged act, and today anything with MCP access can.
- **Does steering actually improve outcomes?** Unmeasured, and it should not be
  assumed. The comparison to run is: same targets, steered against unsteered,
  judged on the resulting findings — the same discipline the grounding gate was
  held to when its in-sample numbers looked better than they were.
