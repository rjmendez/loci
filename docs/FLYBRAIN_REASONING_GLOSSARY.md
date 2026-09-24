# FlyBrain + Reasoning Glossary

Quick lookup for terms used across the FlyBrain docs and Loci reasoning docs.
Definitions are operational (how the term is used in this repo), not textbook biology.

## FlyBrain and dataset terms

| Term | Practical meaning in this repo |
|---|---|
| Anatomy scope | The structural boundary a claim applies to (for example optic lobe, Kenyon cell class, whole CNS). |
| Curated evidence | Manually classified/ontology-backed evidence, such as `get_known_neurotransmitters`; no confidence score is implied. |
| Dataset symbol | Short code for a connectome dataset used in tool calls (for example `fw`, `mc`, `ol`, `l1em`). |
| Dataset version | Version-bearing identifier for a specific release snapshot (for example `flywire783`, `male_cns_v1_0`). |
| Evidence family | The kind of result being stored (for example curated neurotransmitter, predicted neurotransmitter, connectivity). |
| `exclude_dbs` | Query filter that removes listed dataset symbols from a connectivity query. |
| FlyBrain provenance envelope | The minimum metadata stored with a FlyBrain claim: tool, resolved entity, dataset scope, query settings, result contract, and scope tags. |
| `group_by_class` | Connectivity query mode that rolls up results over class/subclass hierarchy instead of raw neuron-to-neuron rows. |
| Life stage scope | Stage boundary for a claim (adult vs larval); do not generalize across stages without direct evidence. |
| Predicted evidence | Model-predicted evidence with confidence values, such as `get_predicted_neurotransmitters`. |
| Query path | The exact tool + parameters used to produce a result. In Loci, a claim is scoped to this path unless replayed under broader settings. |
| Result contract | Output integrity fields needed for replay and interpretation (`count`, `count_status`, warnings, returned rows). |
| Scope tags | Explicit labels for sex, life stage, anatomy, and evidence family attached to stored findings. |
| Sex scope | Sex boundary for a claim (female vs male); do not generalize across sex without direct evidence. |
| `weight` (connectivity) | Minimum synapse count threshold used to include a connectivity edge in `query_connectivity`. |

## Loci reasoning and memory terms

| Term | Practical meaning in this repo |
|---|---|
| Assumed finding | Working hypothesis used to guide checks; not evidence until supported. |
| Confidence (`low`/`medium`/`high`) | Current belief level for a finding; separate from source authority and storage tier. |
| Contradiction pressure | Urgency signal raised when findings conflict and lowered as conflicts are resolved. |
| Degraded result | A fail-open output produced when a tier/tool is unavailable; still machine-readable but lower-trust. |
| `derived_from` chain | Lineage links showing which findings a later inferred claim was built on. |
| Evidence provenance tier | Source authority class (`human_authored`, `tool_verified`, `deterministic_derived`, `model_asserted`). |
| Fail-open | Reliability policy where the system returns usable structured output instead of crashing when a step fails. |
| Gap finding | Explicit record of missing evidence or an open check; it marks unknowns, not conclusions. |
| Grounding | Pulling relevant prior findings/context before answering, so responses are evidence-bounded. |
| Inferred finding | Claim reasoned from existing findings; should cite lineage using `derived_from`. |
| Observed finding | Directly witnessed evidence from tools/data/humans; should include source details/receipts. |
| Procedure finding | Reusable runbook step sequence, tracked separately from factual claims. |
| Resolution state | Lifecycle status for a finding (`open`, `fixed`, `intentional`, `wontfix`, `superseded`). |
| Seed (swarm) | One full independent swarm pipeline run; multi-seed means several full runs merged in synthesis. |
| Self-consistency sampling | Cheap-tier majority retry for low-confidence findings before escalation. |
| Storage tier (`hot`/`warm`/`cold`) | Retrieval/access tier; distinct from confidence and provenance authority. |
| Triage gate | Deterministic stage that decides which findings escalate (for example low confidence or contradiction markers). |

## Related references

- Guide entry point: [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md)
- Provenance requirements: [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md)
- Cross-dataset scope rules: [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md)
- Reasoning policy and escalation: [REASONING_POLICY_SPEC.md](./REASONING_POLICY_SPEC.md)
- Reasoning provenance model: [REASONING_LOOP_PROVENANCE.md](./REASONING_LOOP_PROVENANCE.md)
