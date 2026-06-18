# Cognitive Foundations

The hermes_memory architecture is grounded in cognitive science research on human memory.
This document maps each design decision to its empirical basis and notes where the
research implies future improvements.

---

## 1. Miller (1956) and Cowan (2001) — Working Memory Capacity

**The research:** Miller's "Magical Number Seven, Plus or Minus Two" (Psych Review, 1956)
proposed humans hold ~7 items in short-term memory. Cowan (2001) refined this to ~4 chunks,
controlling for rehearsal. The limit is on *chunks* (compressed meaning units), not raw items.
Skill shadowing research (arxiv 2605.24050, 2026) found ~21% agent pass-rate degradation at
202 skills — directly analogous to chunk overload in retrieval.

**Design implications:**

| Design decision | Empirical basis |
|---|---|
| `HOOK_RECALL_TOP_K = 3` (default) in `pre_llm_grounding.py` | Near Cowan's 4-chunk limit; K=3 vs K=7 shows negligible accuracy difference (Cambridge NLP 2024) while reducing context noise |
| MemGAS entropy weighting narrows effective results despite searching 3 levels | High-entropy (scattered) retrieval gets down-weighted, preserving chunk coherence |
| `skillops_maintenance.py` detects similarity > 0.92 as SHADOW_RISK | Semantically overlapping skills create interference — the retrieval analogue of chunk confusion |
| "NOT for" clauses in SKILL.md | Disambiguation reduces inter-skill interference at retrieval time |

**Env var:** `HOOK_RECALL_TOP_K` (default: `3`) — controls how many memory hits are injected
into context per turn. Increase cautiously; values above 5 reliably increase context noise.

**Remaining gap:** Cowan's model suggests 4 as a principled ceiling. The current default of
3 already sits within the empirically-validated range. Consider lowering to 3 per-collection
and capping the merged total at 5 if cross-collection diversity becomes valuable.

---

## 2. Ebbinghaus (1885) — Forgetting Curves and Spaced Repetition

**The research:** Ebbinghaus showed memory retention follows R = e^(-t/S) where t = time
since last encoding and S = memory stability. Stability increases with each successful
retrieval (the *spacing effect*). The FOREVER technique (ACL 2026, arxiv 2601.03938) applied
this to LLM agent memory consolidation, showing decay-triggered refresh outperforms fixed
cron intervals.

**Design implications:**

| Design decision | Empirical basis |
|---|---|
| `ebbinghaus_consolidation.py` triggers on R < 0.3 | Consolidate when 70%+ of memory trace has likely decayed |
| Stability: FSRS DSR model — `S = (1 + recall_count)^(1/D) * (D/2)` initialized; success/failure FSRS update rules applied thereafter | Empirically outperforms SM-2 and linear approximations (open-spaced-repetition/FSRS); maps directly to the Difficulty/Stability/Retrievability DSR model |
| Ebbinghaus cron at 47-minute intervals (prime) | Avoids resonance with other consolidation jobs; decay is checked per entry, not assumed |
| `working_memory.last_recalled` and `recall_count` drive Ebbinghaus stability updates | Recall history is load-bearing state — updated on every consolidation run |

**FSRS parameters (all tunable via env vars):**

| Env var | Default | Purpose |
|---|---|---|
| `FSRS_W1` | `0.4` | Success stability growth rate |
| `FSRS_W2` | `0.6` | Retrievability factor in success update |
| `FSRS_DECAY_FACTOR` | `0.5` | Stability penalty on failed recall |
| `FSRS_DIFF_INIT` | `5.0` | Neutral difficulty when no importance metadata is available |
| `FORGET_THRESH` | `0.3` | Retention probability below which an entry is refreshed |
| `MAX_PER_RUN` | `50` | Maximum entries consolidated per cron invocation |

**Implementation note:** MemGAS (`memgas_hierarchy.py`) fetches only `(id, content,
importance)` from SQLite — it does not use `last_recalled` or `recall_count`. Only the
Ebbinghaus consolidation script reads those fields. MemGAS entropy weighting is based on
Qdrant vector search scores, not recall history.

---

## 3. Tulving (1972) — Episodic vs. Semantic Memory

**The research:** Tulving distinguished episodic memory (temporally-tagged, contextually-bound
events) from semantic memory (context-free factual knowledge). The distinction predicts
different retrieval cues work for each: episodic retrieval is context-dependent; semantic
retrieval is cue-independent.

**Design implications:**

| Memory tier | Mnemosyne table | Retrieval character |
|---|---|---|
| L1 — Utterances (episodic) | `working_memory` | Temporally-tagged, session-scoped, decays fast |
| L2 — Summaries (episodic+) | `episodic_memory` | Distilled events, longer half-life |
| L3 — Topics (semantic) | `consolidated_facts` / `triples` | Context-free, high stability |

`memgas_hierarchy.py` searches all three levels simultaneously with entropy weighting.
Low-entropy (confident) levels contribute more to the final ranking. This mirrors how
human recall mixes episodic cues and semantic knowledge — neither dominates by default.

**Context-dependent memory (Godden & Baddeley, 1975):** Memories formed in one environment
are better retrieved in that environment. The `branch`, `project`, `cwd`, and `session_id`
fields in Qdrant payloads support context-gated retrieval — a future filter pass could
prefer results from the same project context before cross-project results.

---

## 4. Chunking — Chase & Simon (1973), Miller (1956)

**The research:** Chess masters don't memorize individual pieces — they recognize patterns
(chunks). Miller showed the effective memory capacity in chunks is constant regardless of
chunk information content. Chase & Simon (1973) verified experts perceive meaningful chunks
where novices see noise.

**Design implications:**

| Design decision | Empirical basis |
|---|---|
| `amem_consolidation.py` builds cross-links in `graph_edges` | Semantic links between memories create retrievable chunk structure |
| A-MEM conflict detection in `conflicts` table | Contradicting chunks need resolution before they degrade retrieval coherence |
| `agentHER_relabeler.py` compresses failure traces into "This trace shows how to..." | Generation of a chunk-level summary from raw event data — the same compression chess masters apply to board positions |
| Mnemosyne consolidation passes | Convert raw episodic entries into higher-level semantic chunks in `consolidated_facts` |

**Spreading activation (Collins & Loftus, 1975):** When one concept is activated, related
concepts become more available. This is implemented in `scripts/spreading_activation.py`
and integrated into `pre_llm_grounding.py`. After the primary Qdrant retrieval pass, SA
seeds BFS traversal from mnemosyne hits that carry a `mnemosyne_id` payload field,
discovers associatively-linked memories the vector search missed, and appends them to the
result set (labeled `mnemosyne_sa` collection).

**Spreading activation parameters:**

| Env var | Default | Purpose |
|---|---|---|
| `HOOK_SA_ENABLED` | `true` | Enable/disable spreading activation in the grounding hook |
| `HOOK_SA_TIMEOUT_MS` | `25` | Skip SA results if traversal exceeds this wall-clock limit (ms) |
| `SA_EDGE_FLOOR` | `0.4` | Minimum edge weight to traverse in BFS |
| `SA_ACTIVATION_THRESHOLD` | `0.5` | Minimum activation score to include a node in results |
| `SA_MAX_HOPS` | `2` | BFS depth limit |
| `SA_FAN_EFFECT` | `true` | Divide activation by out-degree (SYNAPSE hub dampening) |
| `SA_HYBRID_VECTOR_WEIGHT` | `0.7` | Vector score weight in hybrid SA scoring |
| `SA_HYBRID_ACTIVATION_WEIGHT` | `0.3` | Activation score weight in hybrid SA scoring |

The grounding hook calls `run_spreading_activation(max_results=2)` — up to 2 SA-discovered
nodes per turn are appended, keeping the total injected context within the `HOOK_RECALL_TOP_K`
ceiling.

---

## 5. QuorumGate — Rate-limiting via Quorum Sensing

**The biological analog:** Bacterial quorum sensing — cells only fire a collective action
when enough correlated signal has accumulated above a threshold, preventing per-event chatter.

**Implementation:** `scripts/quorum_gate.py` implements a persistent decaying accumulator
per topic cluster. Events deposit signal; exponential decay erodes it over time. When the
accumulator crosses a threshold, the caller fires and resets the accumulator.

`amem_consolidation.py` imports `QuorumGate` and checks the `amem_consolidation` topic
before running the expensive pairwise embedding pass. If insufficient signal has accumulated
since the last run, the script exits early.

**Env var:** `QUORUM_AMEM_THRESHOLD` (default: `0`) — set to a positive float (e.g. `5.0`)
to require that many accumulated signal units before A-MEM consolidation runs. `0` disables
the gate. `QUORUM_HALFLIFE_SECONDS` (default: `1800`) controls how fast accumulated signal
decays.

---

## 6. Stigmergic Pheromone Recall

**The biological analog:** Ant colony optimization — paths used more frequently accumulate
pheromone, making them more likely to be chosen again. Pheromone evaporates over time,
preventing monoculture lock-in and allowing cold paths to resurface.

**Implementation:** `pre_llm_grounding.py` maintains a pheromone level on each retrieved
Qdrant point. On selection, a `PHERO_DEPOSIT` amount is added to the point's payload
(fire-and-forget, 0.5s timeout). On scoring, effective pheromone after exponential
evaporation boosts the hit's multi-signal score by `PHERO_BETA * log1p(phero)`.

**Pheromone parameters:**

| Env var | Default | Purpose |
|---|---|---|
| `HOOK_PHERO_BETA` | `0.08` | Score boost coefficient per unit of effective pheromone |
| `HOOK_PHERO_HALFLIFE_H` | `24` | Pheromone half-life in hours (evaporation rate) |
| `HOOK_PHERO_DEPOSIT` | `1.0` | Amount deposited per retrieval event |
| `HOOK_PHERO_EPSILON` | `0.05` | ε-exploration probability for MMR's final slot |

Pheromone deposit is currently applied only to the `hermes_memory` collection, whose points
have mutable payloads owned by this system.

---

## 7. MMR — Maximal Marginal Relevance

**The research:** Carbonell & Goldstein (1998) defined MMR as a selection criterion that
balances relevance to the query against redundancy with already-selected results, producing
diverse output sets without sacrificing top-ranked precision.

**Implementation:** `_mmr_select()` in `pre_llm_grounding.py` selects the final `HOOK_RECALL_TOP_K`
hits from the scored candidate pool using token-overlap as a cheap text-similarity proxy
(no re-embedding needed). The MMR objective is:

```
score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim_to_selected
```

ε-exploration (`HOOK_PHERO_EPSILON`) causes the final slot to occasionally be filled with
a random non-selected hit rather than the MMR winner, preventing pheromone-reinforced recall
from permanently locking into the same top-K.

**Env var:** `HOOK_MMR_LAMBDA` (default: `0.75`) — `1.0` = pure relevance, `0.0` = pure diversity.

---

## 8. Multi-Signal Ranker

**Design:** Retrieved hits are ranked on four weighted axes before MMR selection.
This mirrors the multi-factor structure of human memory retrieval, where recency,
confidence, and source type all modulate recall strength independently of raw similarity.

| Signal axis | Weight env var | Default | Basis |
|---|---|---|---|
| Relevance (semantic cosine × importance) | `HOOK_RANKER_W_RELEVANCE` | `0.50` | Primary retrieval signal |
| Recency (exponential decay by age) | `HOOK_RANKER_W_RECENCY` | `0.20` | Temporal proximity bias |
| Trust (confidence tier: high/medium/low) | `HOOK_RANKER_W_TRUST` | `0.15` | Source credibility |
| Record type (observed > inferred > assumed > gap) | `HOOK_RANKER_W_TYPE` | `0.15` | Epistemic status |

Weights must sum to 1.0. Stigmergic pheromone boost is added on top of the weighted base score.

**Env var:** `HOOK_RECENCY_HALFLIFE_DAYS` (default: `7`) — age at which a hit scores 0.5 on
the recency axis.

---

## 9. Memory Reconsolidation — Nader et al. (2000)

**The research:** Memories are not fixed at encoding. Each time a memory is retrieved,
it re-enters a labile state and can be modified before re-stabilizing. This *reconsolidation*
window allows memories to be updated with new context or corrected.

**Design implications:**

| Design decision | Empirical basis |
|---|---|
| `agentHER_relabeler.py` — reads failure memories, runs Ollama relabeling, writes positive traces back | Computationally equivalent to reconsolidation: the memory is "recalled" (read), transformed (relabeled), and re-encoded (upserted) |
| AgentHER runs nightly (720m cron), not in real time | Reconsolidation requires a consolidation window; nightly batch mirrors the offline consolidation that occurs during sleep in biological systems |

**DB schema note:** `agentHER_relabeler.py` reads from the `working_memory` table directly.
`exif_skill_discovery.py` reads from a `memories` table (with a `bank` column filter for
`working_memory`, falling back to an unfiltered query if the column does not exist). These
two scripts target different schema variants; check which table exists in your deployment.

---

## 10. Reflexion — Shinn et al. (2023, NeurIPS)

**The research:** LLM agents improve by verbally reflecting on their own failures and storing
corrective traces in memory. Retrieved traces constrain future generation without any gradient
updates. Failure traces categorized by type (wrong_args, timeout, etc.) improve transfer
across similar future tasks more than uncategorized raw errors.

**Design implementations:**
- `post_tool_failure_reflection.sh` (Claude Code PostToolUseFailure hook) — categorizes and
  stores typed failure traces in real time
- `score_trace_collector.py` — aggregates traces into a labeled dataset for eventual SFT

---

## 11. The Generation Effect — Slamecka & Graf (1978)

**The research:** Information you generate yourself is remembered better than information
you passively read. The act of generation drives deeper encoding.

**Design implication:** `agentHER_relabeler.py` has the model *generate* a reinterpretation
of each failure trace ("This trace shows how to...") rather than storing the raw error.
The generated interpretation is what gets embedded and retrieved. This is intentional:
generated content produces better recall signals than verbatim copies.

---

## 12. The Skill Shadowing Problem — arxiv 2605.24050 (2026)

**The research:** At 202 skills, agent pass rates drop ~21% due to skill shadowing —
semantically similar skills compete at retrieval and the wrong one wins. Disambiguation
via description contrast ("NOT for" clauses) and retrieval reranking are the proven mitigations.

**Design implementations:**
- `skillops_maintenance.py` — computes pairwise SKILL.md embedding similarity, reports
  SHADOW_RISK pairs above 0.92 cosine threshold
- `_keyword_rerank()` in `pre_llm_grounding.py` — keyword overlap boost after semantic
  retrieval breaks ties toward the more literally-matching skill
- "## NOT for" sections in SKILL.md — description-level disambiguation

---

## 13. EXIF — Closed-Loop Skill Discovery (arxiv 2506.04287)

**The research:** EXIF (EXperience-driven Incremental Frontier) proposes a two-agent
closed loop: one agent identifies capability gaps from failure traces; a second generates
candidate skill scaffolds to fill them.

**Implementation:** `scripts/exif_skill_discovery.py` runs a three-phase pipeline:

1. **Alice (gap-finder):** Queries recent high-importance failure memories and asks Ollama
   to identify the most impactful missing skill, returning a JSON suggestion with a
   confidence score.
2. **Bob (generator):** If confidence > 0.6, generates a candidate SKILL.md scaffold in a
   staging directory.
3. **Validator:** Logs the candidate to `exif_discoveries.jsonl` for human review. No
   auto-promotion — a human must manually copy the candidate into the active skills directory.

**DB schema note:** EXIF reads from a `memories` table (not `working_memory`), filtering by
`bank = 'working_memory'` where available. This differs from agentHER, which reads from
`working_memory` directly. Ensure your schema matches before running both scripts against
the same database.

---

## 14. AgentRR — Session Trace Collection

AgentRR implements session-level reinforcement logging. At session end, a guard script
(`session_end_evaluate_guard.sh`) collects the session trace and stores it to the
`hermes_sessions` Qdrant collection. This provides a long-lived episodic record of
past sessions that the grounding hook can retrieve during future turns.

The `hermes_sessions` collection is included in the base collection list searched by
`pre_llm_grounding.py` on every turn.

---

## Design implications not yet implemented

Updated from deep literature search (2026-06-17). Each row includes the specific paper
that grounds the recommendation and the implementation detail.

### Quick wins (LOW difficulty)

| Research basis | Paper | Implied improvement | Evidence |
|---|---|---|---|
| Phonological loop (~2s window) | — | Grounding timeout ≤ 2s | Current 9s timeout allows late context arrival in the turn |
| Qdrant payload indexing | Permission-aware RAG (SNU 2024) | Index `project` and `session_id` as keyword payload indexes on the `mnemosyne` collection | Pre-filter is free; unindexed fields force brute-force scan. Note: `qdrant_payload_indexes.py` already creates indexes for `project` and `session_id` — verify they are applied to all relevant collections |

### Medium complexity

| Research basis | Paper | Implied improvement | Implementation note |
|---|---|---|---|
| Encoding Specificity Principle | Dual-Trace Encoding arXiv:2604.12948 | Store originating context (project+branch+task) in chunk text, not just as metadata | Embedding encodes context → cue-match retrieval improves; two-pass query: `must=[project=X, session_id=Y]` then `must=[project=X]`, prepend session-scoped hits |
| SCM multidimensional importance | SCM arXiv:2604.20943 | Replace single `importance` int with three axes: **novelty + severity + relevance** | Novelty = embedding distance from nearest existing memory; severity = error/failure→high, informational→low; relevance = cosine to current session embedding |

### High complexity / research-level

| Research basis | Paper | Implied improvement | Notes |
|---|---|---|---|
| NREM/REM dual-phase sleep consolidation | SCM arXiv:2604.20943 | Extend `agentHER_relabeler.py` (NREM) with an REM pass: stochastic random walks through `graph_edges` to surface novel cross-domain associations | Current agentHER is NREM-equivalent (structured replay + relabeling); REM phase would add temperature-boosted random walks |
| Global Workspace Theory | Theater of Mind arXiv:2604.08206 | Entropy-based intrinsic drive: if current session semantic diversity is low, surface diverse grounding results rather than highest-similarity | `entropy(softmax(cosine_scores))` already computed in MemGAS; route low-entropy sessions to diversity-boosted retrieval |
| Private working memory impossibility | arXiv:2601.06973 (impossibility theorem) | Theoretical validation: hook-based memory is necessary, not optional | Formal proof: agents restricted to public conversation history cannot maintain hidden state consistency across turns |
| Causal timeline linking | THEANINE arXiv:2406.10996 | Index `graph_edges` with a `causal_chain` edge type; retrieval traverses cause-effect links | Extends A-MEM's semantic links with directional causality — especially useful for failure→fix chains in agent traces |
| Spreading activation hybrid scoring | SA-RAG arXiv:2512.15922, SYNAPSE arXiv:2601.02744, HippoRAG arXiv:2405.14831 | Promote `run_spreading_activation_hybrid()` (already in `spreading_activation.py`) to the default path in `pre_llm_grounding.py` | Currently `run_spreading_activation()` (activation-only) is called; the hybrid wrapper adds a combined `0.7·vector + 0.3·activation` score for better precision |

---

## Literature research map (2026-06-17)

Topics researched and papers found per area:

### Spaced repetition for LLM agents
- MemoryBank (arXiv:2305.10250, AAAI 2024): uses discrete strength counter — functionally equivalent to a linear stability approximation
- "My agent understands me better" (arXiv:2404.00573): custom `p(t) = 1 - exp(-r·e^(-t/gn))` with cosine similarity as relevance
- **FSRS** (open-spaced-repetition GitHub): open-source DSR model outperforms SM-2 empirically; Rust + Python implementations available — **implemented in `ebbinghaus_consolidation.py`**
- ACT-R base-level learning `B_i = ln(Σ t_j^{-d})`: power-law alternative, decay param d≈0.5 for humans, tunable per memory type

### Spreading activation in graph-RAG
- **SA-RAG** (arXiv:2512.15922, Dec 2024): BFS spreading activation, +22% over naive RAG, +39% combined with chain-of-thought — **implemented in `spreading_activation.py`**
- **SYNAPSE** (arXiv:2601.02744, Jan 2026): episodic+semantic unified graph, fan-effect hub dampening, lateral inhibition (top-M=7 suppress competitors), +60.7% F1 vs vector-only on LoCoMo — fan-effect dampening implemented via `SA_FAN_EFFECT`
- **HippoRAG** (arXiv:2405.14831, NeurIPS 2024): PPR with damping=0.5 as spreading activation; code at github.com/OSU-NLP-Group/HippoRAG
- **GAAMA** (arXiv:2603.27910): edge-type-aware PPR on 4-node-type graph
- **MS GraphRAG** (arXiv:2404.16130): does NOT implement spreading activation — community summarization only

### Optimal top-K for RAG injection
- K=3 vs K=7 accuracy difference: 0.67 vs 0.68 — negligible (Cambridge NLP 2024)
- EMNLP 2024 Best Practices: K=5–10 upper bound for accuracy; K>10 reliably degrades
- Chroma eval: K=5 recall=88.5%; K=10 precision drops to 3.8%
- Lost in the Middle (arXiv:2307.03172, TACL 2024): U-curve; position 1 or last best; 30%+ drop at position 10 of 20
- Context Length Alone Hurts (arXiv:2510.05381): 13.9%–85% accuracy loss from growing context even with perfect retrieval
- Cognitive Workspace (arXiv:2508.13171): explicitly cites Miller 7±2 and Cowan 4±1; system naturally converges to 3 items

### Context-gated / session-aware retrieval
- No paper uses "context-gated RAG" as a formal term; distributed across personalized RAG, permission-aware RAG, episodic memory literatures
- PersonaRAG (arXiv:2407.09394, SIGIR 2024): user-centric agents with real-time profile gating
- O-Mem (arXiv:2511.13593): hierarchical retrieval gated by persona + topic context
- Dual-Trace Encoding (arXiv:2604.12948): embed context description in chunk text so embedding itself encodes context
- Rashomon Memory (arXiv:2604.03588): cites Tulving & Thomson Encoding Specificity as theoretical grounding
- Cross-user leakage: "57-71% of AI agents leak data between users" — isolation must be enforced at recall boundary

### Novel cognitive architectures (new in 2025-2026)
- **SCM: Sleep-Consolidated Memory** (arXiv:2604.20943): 5 components including Miller-bounded WM (7 eps), NREM/REM dual-phase, multidimensional importance tagging, computational self-model; 90.9% noise reduction
- **SleepGate** (arXiv:2603.14517): KV-cache-level sleep consolidation; reduces proactive interference O(n)→O(log n)
- **Private WM impossibility theorem** (arXiv:2601.06973): formal proof agents need private state separate from public context
- **Theater of Mind** (arXiv:2604.08206): Global Workspace Theory implementation with entropy-driven temperature regulation
- **THEANINE** (arXiv:2406.10996): causal timeline memory links; manages large-scale memories without deletion
- **ReadAgent** (arXiv:2402.09727): gist memory grounded in fuzzy-trace theory
- **Survey: "From Storage to Experience"** (arXiv:2605.06716, ACL 2026): three-stage evolutionary taxonomy
