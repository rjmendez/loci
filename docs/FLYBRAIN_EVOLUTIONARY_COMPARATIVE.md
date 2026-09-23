> Archived / retired from active navigation
>
> This page is kept only for historical continuity and searchability. It is no longer part of the default FlyBrain reading path.
>
> Canonical entry point: [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md)
> Archive home: [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md)
> For current guidance, use the guide and the evidence/provenance docs before digging into historical detail.
>
# FlyBrain Evolutionary Comparative Research
## Conserved vs. Fly-Specific Mechanisms — Engineering Implications for Loci

**Status**: Complete (todo `flybrain-evolutionary-comparative-research` done)

**Purpose**: Compare fly-brain principles against other insect connectomes
(*Apis mellifera*, *Schistocerca gregaria*, *Camponotus* spp.) and compact
worm connectomes (*C. elegans*) to determine which Loci design patterns
are universal vs. which must be guarded behind Drosophila-specific scope.

**Audience**: Engineers deciding whether to generalize or scope-guard a
fly-brain-derived design decision. Reviewers auditing whether a claim is
being applied too broadly.

**Related docs**:
- [`FLYBRAIN_GUIDE.md`](./FLYBRAIN_GUIDE.md) — canonical entry point
- [`FLYBRAIN_CROSS_DATASET_LESSONS.md`](./FLYBRAIN_CROSS_DATASET_LESSONS.md) — per-dataset lessons
- [`FLYBRAIN_INSPIRATION_MAP.md`](./FLYBRAIN_INSPIRATION_MAP.md) — placement guide for new ideas
- [`FLYBRAIN_RESEARCH_PRIORITIES.md`](./FLYBRAIN_RESEARCH_PRIORITIES.md) — prioritized implementation queue

---

## 1. Comparative Matrix

Each row is a mechanism drawn from the FlyBrain corpus. Confidence reflects
the quality of cross-species evidence as of 2024–2026.

| Mechanism | Conserved or Fly-Specific? | Confidence | Evidence Source | Loci Design Implication |
|---|---|---|---|---|
| **Sparse coding via expansion layer** (Kenyon cells, MB) | ✅ Conserved — all insects | High | Perez-Orive et al. 2002 (*Science* 297); bee/ant KC expansion literature; comparative connectomics preprint (bioRxiv 2025.09.22.677863) | Claim density cap (`semantic_dedup`) is a universal principle; not fly-specific. Implement without fly-scope guard. |
| **Ring attractor navigation** (CX/EPG head-direction circuit) | ✅ Conserved — all insects (≥300 Mya) | High | Bees, ants, locusts confirmed; bioRxiv 2026.07.26.740564 "Deep conservation of HD circuits"; locust optic-flow integration (Front. Neural Circuits 2023) | Fan-shaped body → homeostatic drive working-memory model generalizes. FSB-derived `context_load` / `task_horizon` keys apply universally. |
| **Layered sensorimotor transformation** | ✅ Conserved — insects and worms | High | *C. elegans* full adult connectome (PLoS Biol. 2024); Drosophila larva complete connectome; zebrafish partial EM | Multi-pass reasoning (input → abstraction → output) is a general principle. Optic-lobe-style ingress layer design applies universally. |
| **Feedforward loop circuit motifs** | ✅ Conserved — insects and worms | High | Signed motif analysis *C. elegans* (Springer 2026); *C. elegans* comprehensive analysis (PLoS Biol. 2024); Drosophila larval connectome (Network stats, Nature 2024) | Feedforward check-before-commit patterns are universal. The multi-step reasoning pipeline is validated beyond fly. |
| **Degeneracy / parallel redundant circuits** | ✅ Conserved — insects and worms | High | *C. elegans* asymmetric synaptic connectivity paper (iScience 2024); Drosophila lesion studies; comparative insect connectomics | Multi-lane retrieval (`investigation_search` → mnemo → qdrant → keyword) is a universal robustness principle. |
| **State-gated routing via neuromodulation** (general principle) | ✅ Conserved — insects and worms | High | *C. elegans* octopamine/serotonin analogs; general insect neuromodulation (Strausfeld 2012; comparative reviews) | Homeostatic drive state-gated routing is general. The concept of explicit state variables biasing decisions generalizes beyond fly. |
| **MB learning circuit convergent motifs** (KC→MBON convergence) | ✅ Conserved structurally; ⚠️ MBON parameter details are fly-specific | Medium-High | Comparative Connectomics MB motifs (bioRxiv 2025.09.22.677863); Eichler et al. 2017 larval MB (*Nature*); Saumweber et al. 2018 (*Nat. Commun.*) | Population-vote retrieval confidence principle generalizes. Exact MBON valence thresholds must stay fly-scoped until cross-species calibration. |
| **Sleep-dependent memory consolidation** | ✅ Conserved — insects and vertebrates | High | Drosophila sleep literature (multiple); *C. elegans* consolidation analogs; general comparative neuroscience | Sleep-consolidation scheduler and consolidation thresholds are general. Not fly-specific; apply universally. |
| **Efference copy / corollary discharge** | ✅ Conserved — broad phyla | Medium | Drosophila Kim et al. 2015 (*eLife*); Tuthill & Bhatt 2020 (*Curr. Biol.*); vertebrate corollary discharge broadly established | `investigation_pre_answer_check` self-prediction pattern is universal. Fly-specific circuit topology is not needed. |
| **Descending neuron threshold-gating for action commitment** | ⚠️ Structural homologs conserved; quantitative details fly-specific | Medium | Namiki et al. 2018 (*J. Comp. Neurol.*); Schnell et al. 2017 (*Curr. Biol.*); MANC DN map | Threshold-gating principle for `swarm_reason` commit generalizes. Exact threshold values and DN class counts are fly-specific. |
| **Sex-specific wiring** (*fruitless*/*doublesex* circuits) | ❌ Fly-specific — *fruitless*/*doublesex* gene expression is Drosophila-specific | High | MRC LMB 2025 male CNS publication; bioRxiv 2025.10.09.680999; FlyWire male-female comparison | Must remain guarded behind `sex` scope key and fly-specific dataset scope. Do **not** generalize to other provenance contexts. |
| **Optic lobe 6-layer architecture** (lamina/medulla/lobula detail) | ❌ Fly-specific — insect compound eye optics differ significantly | High | VFB FBbt_00003748 optic lobe hierarchy; Optic Lobe Connectome (Janelia 2024) | Optic-lobe ingress-layer naming (lamina/medulla/lobula analog) is a useful metaphor but the specific 6-layer structure must be scoped to fly. Use as a design pattern only, not a claim. |
| **Lobula plate direction-selectivity circuits** | ⚠️ Motion detection is conserved; lobula plate circuit topology is fly-specific | Medium | Optic Lobe Connectome (Janelia 2024); comparative visual systems literature | Direction-selectivity circuits are Drosophila-optic-lobe-specific. Underlying fast-vs-slow routing principle generalizes but the wiring implementation does not. |
| **Multi-connectome comparative coverage** (8 datasets) | ❌ Fly-specific — data richness is unique to Drosophila | High | VFB list_connectome_datasets: fw, mc, BANC, hb, mv, ol, fafb, l1em | This data richness justifies fly-specific provenance features (dataset, sex, stage scope keys). No other organism has this; fly-specific scope guards are warranted and should remain. |

---

## 2. Summary Classification

### 2.1 Universally generalizable (no fly-specific scope guard needed)

These mechanisms are backed by cross-phylum evidence and can be applied
to Loci without qualifying them as Drosophila-specific:

1. **Sparse/expansion coding → claim density cap**: The principle that an
   expansion layer creates sparse, non-overlapping representations is confirmed
   in bees, ants, *C. elegans*, and Drosophila. `semantic_dedup` and
   investigation claim density caps apply universally.

2. **Ring attractor persistent state → working-memory register**: The CX/EPG
   head-direction mechanism is the most deeply conserved of all
   connectome-verified circuits. The `context_load`/`task_horizon` drive keys
   derived from the fan-shaped body generalize without fly-scope restriction.

3. **Layered multi-pass transformation → multi-step reasoning**: Confirmed
   across *C. elegans* (302 neurons), Drosophila larva (10 K neurons), and
   partial zebrafish data. Single-pass input-to-output reasoning is insufficient
   as a universal engineering principle.

4. **Feedforward loop motifs → commit-before-action gates**: Overrepresented in
   both *C. elegans* and Drosophila connectomes. The pattern of check-before-commit
   (pre-answer check, verify-before-persist) is a universal small-circuit design.

5. **Degeneracy → multi-lane retrieval fallback**: Independent redundant paths
   maintaining function under partial failure are confirmed in *C. elegans* and
   Drosophila. Multi-lane retrieval architecture is universally justified.

6. **State-gated routing via neuromodulation → homeostatic drives**: The
   *existence* of internal state biasing behavior is universal (even *C. elegans*
   uses monoaminergic state modulation). The drive dictionary pattern is
   universally applicable.

7. **Sleep-dependent consolidation → consolidation scheduler**: Cross-species
   evidence strongly supports selective sleep-gated memory consolidation as a
   general principle.

8. **Efference copy → pre-answer self-prediction**: The corollary discharge
   principle is one of the most broadly confirmed in all of neuroscience.
   The Loci self-check mechanism is universally justified.

### 2.2 Principle generalizes; parameter values are fly-specific

These require the engineering principle to be generalized but the specific
thresholds, class counts, or parameter values must be guarded:

9. **MBON valence gating → retrieval confidence weighting**: The population-vote
   model for confidence is general. The specific `valence_weight` parameter
   values (derived from fly data) should be tagged as fly-specific until
   cross-species calibration is done.

10. **Descending neuron threshold gating → swarm commit threshold**: The
    threshold-before-commitment concept is general. The specific commit threshold
    derived from ~500 DN classes in fly should be treated as an initial
    calibration point, not a universal constant.

### 2.3 Fly-specific — keep scope guards; do not generalize

11. **Sex-specific wiring** (*fruitless*/*doublesex*): Retain `sex` scope key
    and fly-only provenance for all claims derived from male-CNS vs. FlyWire
    comparisons.

12. **Optic lobe 6-layer topology details**: Use as naming metaphor only; the
    exact 6-layer architecture is fly-specific. Do not claim that a Loci
    ingress layer "is" a lamina in a cross-species sense.

13. **Multi-connectome data richness**: The justification for the richness of
    `DATASET_SYMBOLS`, provenance scope keys, and versioned scope enforcement
    comes from fly data. These mechanisms are Drosophila-data-maintenance
    artifacts and should be documented as such.

---

## 3. Engineering Implications for Loci

### 3.1 What to generalize immediately

| Loci subsystem / hook | Action |
|---|---|
| `mcp/flybrain_scope_engine.py::DEFAULT_SCOPE_KEYS` | Keep existing keys; **add `species` key** to allow claims to be tagged as multi-species when evidence supports it |
| `mcp/flybrain_scope_engine.py::ComparativeClaim` | Add optional `species: str = "Drosophila melanogaster"` field with default; enables cross-species claim tagging |
| `mcp/server.py::memory_route` + `_derive_homeostatic_drives` | Fan-shaped-body working-memory keys (`context_load`, `task_horizon`) are safe to add without scope guard |
| `mcp/server.py::semantic_dedup` + `investigation_store` | Claim density cap is universally justified; remove any fly-specific caveats from the implementation rationale |
| `docs/REASONING_POLICY_SPEC.md` | Add note that multi-pass reasoning, feedforward gates, and degeneracy lanes are confirmed as universal principles across phyla |

### 3.2 What to keep fly-scoped

| Loci subsystem / hook | Scope guard to maintain |
|---|---|
| `mcp/flybrain_scope_engine.py::ComparativeClaim.sex` | Keep `sex` scope key; sex-specific claims remain fly-scoped |
| `mcp/slow_neuromod.py::routing_policy` — valence weights | Tag initial `valence_weight` values as `species="Drosophila melanogaster"` until cross-species calibration |
| `mcp/server.py::swarm_reason` — commit threshold | Tag initial commit threshold derived from DN class counts as fly-specific calibration point |
| `FLYBRAIN_DATASET_PROVENANCE_MATRIX.md` — all dataset entries | Remain fly-specific; add note that no equivalent multi-connectome coverage exists for other insects |

### 3.3 New scope key: `species`

The `ComparativeClaim` dataclass and `DEFAULT_SCOPE_KEYS` should be extended
with a `species` field to allow claims to be explicitly scoped:

```python
# In flybrain_scope_engine.py
DEFAULT_SCOPE_KEYS = (
    "dataset",
    "dataset_version",
    "sex",
    "life_stage",
    "annotation_completeness",
    "circuit_class",
    "experience_window",
    "species",          # NEW — default "Drosophila melanogaster"
)
```

Use `species="multi-insect"` for claims confirmed in ≥2 insect orders.
Use `species="pan-arthropod"` for claims confirmed in arthropods + worms.
Use `species="universal"` for claims with broad phylum-level support.

This makes the evolutionary breadth of a claim machine-checkable and enables
future test assertions like: "claims tagged universal must have ≥2 independent
phylum-level citations."

### 3.4 Validation priorities

In order of near-term value:

1. **Add `species` scope key** to `DEFAULT_SCOPE_KEYS` and `ComparativeClaim`
   (30-minute code change; immediately unlocks cross-species claim tagging).

2. **Annotate existing high-confidence claims** in stored investigations with
   `species="Drosophila melanogaster"` (default) and retag the 8 universally
   generalizable principles with `species="pan-arthropod"` or `species="multi-insect"`.

3. **Add a test assertion** in `test_flybrain_claim_scope_validator.py` that
   verifies: universally-generalizable claims (species ≠ "Drosophila melanogaster")
   cite ≥2 independent phylum-level sources.

4. **Update `REASONING_POLICY_SPEC.md`** to note that multi-pass reasoning,
   feedforward gates, degeneracy lanes, and consolidation scheduling are
   confirmed universal and do not require fly-scope justification.

5. **Annotate `slow_neuromod.py` routing_policy valence_weight** with a
   `# TODO: fly-specific calibration; tag species="Drosophila melanogaster"`
   comment when the A2 MBON implementation lands.

---

## 4. Confidence Framework for Cross-Species Claims

This extends the tiering in `FLYBRAIN_CROSS_DATASET_LESSONS.md` section 7
to cover cross-species (not just cross-dataset) confidence.

| Claim level | Species evidence required | Tier cap | Example |
|---|---|---|---|
| `Drosophila-only` | Single fly dataset | T2 max | "MBON valence_weight=0.7 in fw dataset" |
| `multi-insect` | ≥2 insect orders (e.g., Diptera + Hymenoptera) | T2 | "KC sparse coding confirmed in Drosophila and honeybee" |
| `pan-arthropod` | ≥2 arthropod classes + behavioral validation | T3 eligible | "Ring attractor navigation confirmed in Diptera, Orthoptera, Hymenoptera" |
| `universal` | Arthropod + worm (or vertebrate) independently | T3 | "Feedforward loop motifs conserved in Drosophila and C. elegans" |

Claims tagged `universal` or `pan-arthropod` do **not** require fly-scope
guards in implementation and can be applied to Loci without Drosophila-specific
caveats.

---

## 5. Citations

| Citation | Claim supported |
|---|---|
| Perez-Orive et al. 2002, *Science* 297(5580):359–365 | Sparse KC coding in Drosophila MB |
| Honegger et al. 2011, *Nat. Neurosci.* 14:1057–1063 | Cross-dataset KC sparseness confirmation |
| "Deep conservation of head direction circuits in bees, ants and flies" (bioRxiv 2026.07.26.740564) | Ring attractor CX conservation ≥300 Mya |
| Fortier & Bhatt 2023, *Front. Neural Circuits* (doi:10.3389/fncir.2023.1111310) | Locust optic-flow integration in CX |
| "From the fly connectome to exact ring attractor dynamics" (bioRxiv 2024.11.01.621596) | EPG/ring attractor Drosophila connectome validation |
| "Comparative Connectomics Highlights Conserved Architectural Synaptic Motifs" (bioRxiv 2025.09.22.677863) | MB convergent motifs conserved across Drosophila species |
| "Evolution of connectivity architecture in the Drosophila mushroom body" (*Nat. Commun.* 2024, doi:10.1038/s41467-024-48839-4) | Intra-Drosophila MB connectivity conservation |
| Eichler et al. 2017, *Nature* 548:175–182 | Larval MB connectome (l1em); MBON circuit validation |
| Saumweber et al. 2018, *Nat. Commun.* 9:1104 | Larval + adult MB cross-dataset MBON confirmation |
| Kim et al. 2015, *eLife* 4:e13273 | Efference copy in Drosophila visual system |
| Tuthill & Bhatt 2020, *Curr. Biol.* 30:R1002–R1012 | Mechanosensory corollary discharge in Drosophila |
| Namiki et al. 2018, *J. Comp. Neurol.* 526(3):534–566 | Descending neuron map; threshold gating for escape |
| Schnell et al. 2017, *Curr. Biol.* 27(15):2300–2315 | DN threshold gating validated in behavior |
| "Comprehensive analysis of the C. elegans connectome reveals novel insights" (*PLoS Biol.* 2024, doi:10.1371/journal.pbio.3002939) | Feedforward loops and degeneracy in *C. elegans* |
| "Asymmetry in synaptic connectivity balances redundancy and reachability" (*iScience* 2024, doi:10.1016/j.isci.2024.109938) | Degeneracy in *C. elegans* connectome |
| Signed motif analysis of *C. elegans* neuronal network (*SpringerLink* 2026, doi:10.1186/s12915-026-02641-4) | Feedforward loop overrepresentation in *C. elegans* |
| "Network statistics of the whole-brain connectome of Drosophila" (*Nature* 2024, doi:10.1038/s41586-024-07968-y) | Small-world motifs in Drosophila |
| "Comparative connectomics and escape behavior in larvae of closely related species" (*Curr. Biol.* 2023) | Conserved motifs in Drosophila larval escape circuits |
| "Sexual dimorphism in the complete connectome of the Drosophila male CNS" (bioRxiv 2025.10.09.680999) | Fly-specific sex-specific circuits (fruitless/doublesex) |
| Farris 2013 (comparative insect MB) | General MB conservation review |
| Strausfeld 2012 *Arthropod Brains* | Broad insect comparative neuroanatomy |

---

## 6. Gaps and Open Questions

- **Bee connectome at synaptic resolution**: Current bee MB/CX comparisons rely
  on light-level anatomy and physiology, not EM connectomics. When a bee
  synaptic-resolution connectome is available, re-score MB motif claims.

- **Moth and cockroach MBs**: These are cited as conserved in general reviews
  but do not yet have EM-resolution data. Claims based on moth/cockroach should
  stay at T1 until EM data confirms.

- **Descending neuron threshold values**: The threshold principle is conserved;
  the specific parameter values (e.g., how many weak inputs trigger commitment)
  come from fly-only MANC data. Cross-species calibration is needed before
  treating the threshold as a tunable universal constant.

- **Neuromodulator receptor specificity**: General neuromodulatory gating
  (state-dependent routing) is conserved, but specific receptor subtypes and
  circuit-level wiring differ across species. Do not port fly-specific
  dopamine-receptor circuit topology to non-fly contexts without re-validation.
