# FlyBrain Neuromodulatory Circuit Comparison

**Purpose**: Evidence-backed comparison of dopaminergic, serotonergic, and octopaminergic
circuit evidence across VFB/connectome datasets, with explicit confidence/scope notes
and actionable implications for Loci reasoning and routing design.

**Audience**: Loci engineers working on confidence calibration, routing, memory promotion,
or provenance-aware claim storage. Also the primary reference when a FlyBrain-derived
neuromodulatory claim needs to be evaluated or challenged.

**Related docs**:
- [`FLYBRAIN_GUIDE.md`](./FLYBRAIN_GUIDE.md) — canonical entry point and method rules
- [`FLYBRAIN_EVIDENCE_MAP.md`](./FLYBRAIN_EVIDENCE_MAP.md) — claim-to-evidence traceability
- [`FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) — dataset scope boundaries
- [`FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) — provenance envelope requirements

---

## Query provenance

All VFB data below was retrieved in a single session using the live `virtual-fly-brain-*`
MCP tool surface. Scope of the queries:

| Parameter | Value |
|---|---|
| Tool surface | `virtual-fly-brain-get_known_neurotransmitters`, `virtual-fly-brain-get_predicted_neurotransmitters` |
| Known NT source | Ontology-curated classifications; no confidence value |
| Predicted NT source | Per-instance connectome predictions aggregated by class × dataset |
| Prediction min confidence | 0.5 (applied) |
| Datasets included | All available: `fw` (flywire783), `mc` (male_cns_v1_0), `BANC` (BANC888), `hb` (neuprint_JRC_Hemibrain_1point2point1), `mv` (neuprint_JRC_Manc_1_2_1), `ol` (neuprint_JRC_OpticLobe_v1_0_1), `fafb` (catmaid_fafb), `l1em` (catmaid_l1em) |
| Split by dataset | Yes (`split_by_dataset=true`) |
| Session date | 2026-09-22 |

Important: `get_known_neurotransmitters` returns ontology-curated data without confidence
scores and should not be treated as equivalent to `get_predicted_neurotransmitters`.
These are different evidence families (see `FLYBRAIN_CROSS_DATASET_LESSONS.md`).

---

## System overview

| System | VFB class | Ontology ID | Known subclasses | Predicted instances (all datasets) |
|---|---|---|---|---|
| Dopaminergic | dopaminergic neuron | `FBbt_00005131` | 190 | 17,488 |
| Serotonergic | serotonergic neuron | `FBbt_00005133` | 141 | 2,188 |
| Octopaminergic | octopaminergic neuron | `FBbt_00007364` | 127 | 377 |

The population size difference is significant: dopaminergic neurons are nearly 5× more
numerous than serotonergic neurons in the reconstructed connectomes. This is a
dataset-local count, not a claim about fly biology in general.

---

## Dopaminergic circuits

### Known neurotransmitters (curated)
- **Primary transmitter**: dopamine secretion, neurotransmission (`GO_0061527`)
- **Broader category also curated**: catecholamine secretion, neurotransmission (`GO_0160043`) — dopamine is a catecholamine, so this is not contradictory
- **Scope note**: 385 known-NT entries across 190 cell type IDs. Some entries also include
  acetylcholine, GABA, or serotonin for individual subclasses — these indicate co-transmitter
  annotations for specific cell types and are not contradictions of the dopaminergic class label.

### Predicted neurotransmitters (connectome-derived)

**High-confidence, cross-dataset consistent entries:**

| Cell type | nt_label | Dataset | Instances | % of class | Mean confidence |
|---|---|---|---|---|---|
| adult dopaminergic neuron (`FBbt_00049526` subclass aggregate) | dopamine secretion | flywire783 | 4,576 | 100% | 0.64 |
| adult dopaminergic neuron | dopamine secretion | male_cns_v1_0 | 3,932 | 100% | 0.73 |
| adult dopaminergic neuron | dopamine secretion | BANC888 | 1,678 | 100% | 0.56 |
| adult dopaminergic neuron | dopamine secretion | neuprint_JRC_Hemibrain_1point2point1 | 653 | 100% | 0.61 |
| adult dopaminergic neuron | dopamine secretion | neuprint_JRC_OpticLobe_v1_0_1 | 15 | 100% | 0.61 |
| EB neuron of the dopaminergic PPM3 cluster | dopamine secretion | male_cns_v1_0 | 4 | 100% | 0.635 |
| mushroom body DPM cell (`FBbt_00005780`) | dopamine secretion | male_cns_v1_0 | 2 | 100% | 0.87 |
| mushroom body DPM cell | dopamine secretion | flywire783 | 2 | 100% | 0.615 |
| mushroom body DPM cell | dopamine secretion | neuprint_JRC_Hemibrain_1point2point1 | 2 | 100% | 0.67 |

**Confidence/scope assessment**: ROBUST. The main dopaminergic population shows 100%
instance agreement for dopamine across all five reconstructed-neuron datasets at
≥0.56 mean confidence. The DPM cell is confirmed across three independent datasets.
No cross-dataset discordances are observed for the broad adult dopaminergic class.

### Loci design implication — dopaminergic analog

Dopaminergic neurons in the fly brain are organized into named clusters with distinct
anatomical targets and behavioral roles (PAM cluster → appetitive; PPL1 cluster → aversive
/ punishment; DPM → memory consolidation). This multi-cluster, valence-segregated
architecture has a direct analog in Loci confidence routing:

- **PAM-analog**: routing paths that reinforce a finding (positive prediction error → increase confidence)
- **PPL1-analog**: routing paths that penalize a finding on failure (negative prediction error → decrease confidence or flag for review)
- **DPM-analog**: memory consolidation gate, where new evidence is integrated before promotion to long-term storage

Implementation note: this is an architecture analogy, not a claim that the Loci
system operates on biological dopamine circuits. See `FLYBRAIN_IO_TO_LOCI_MAPPING.md`
for the explicit framing of fly-brain patterns as engineering design analogies.

---

## Serotonergic circuits

### Known neurotransmitters (curated)
- **Primary transmitter**: serotonin secretion, neurotransmission (`GO_0060096`)
- **Also curated for some subclasses**: catecholamine secretion, dopamine secretion, GABA — these are co-transmitter or lineage-specific annotations, not contradictions
- **Scope note**: 147 known-NT entries across 141 cell type IDs.

### Predicted neurotransmitters (connectome-derived)

**High-confidence, cross-dataset consistent entries:**

| Cell type | nt_label | Dataset | Instances | % of class | Mean confidence |
|---|---|---|---|---|---|
| adult serotonergic neuron (`FBbt_00049526`) | serotonin secretion | flywire783 | 608 | 100% | 0.636 |
| adult serotonergic neuron | serotonin secretion | male_cns_v1_0 | 323 | 100% | 0.633 |
| adult serotonergic neuron | serotonin secretion | BANC888 | 195 | 100% | 0.715 |
| adult serotonergic neuron | serotonin secretion | neuprint_JRC_OpticLobe_v1_0_1 | 10 | 100% | 0.77 |
| adult serotonergic SE neuron (`FBbt_00047319`) | serotonin secretion | male_cns_v1_0 | 54 | 100% | 0.610 |
| adult serotonergic SE neuron | serotonin secretion | BANC888 | 14 | 100% | 0.655 |
| adult serotonergic SE neuron | serotonin secretion | flywire783 | 10 | 100% | 0.625 |
| adult CSD interneuron (`FBbt_00007405`) | serotonin secretion | flywire783 | 2 | 100% | 0.720 |
| adult CSD interneuron | serotonin secretion | male_cns_v1_0 | 2 | 100% | 0.725 |
| adult serotonergic IP neuron (`FBbt_00110937`) | serotonin secretion | male_cns_v1_0 | 4 | 100% | 0.790 |
| adult serotonergic IP neuron | serotonin secretion | flywire783 | 1 | 100% | 0.530 |
| adult CB0878 neuron (`FBbt_20004573`) | serotonin secretion | flywire783 | 8 | 100% | 0.745 |
| adult CB0878 neuron | serotonin secretion | male_cns_v1_0 | 6 | 100% | 0.593 |

**Discordant / uncertain entries — do not treat as high confidence without scope qualification:**

| Cell type | Dataset A prediction | Dataset B prediction | Notes |
|---|---|---|---|
| adult serotonergic PMPD neuron (`FBbt_00110955`) | dopamine (flywire783, 100%, 0.61) | serotonin (male_cns_v1_0, 100%, 0.715) | Cross-dataset identity discordance; sex difference (female vs male CNS) may explain |
| adult serotonergic PLP neuron (`FBbt_00110945`) | glutamate (male_cns_v1_0, 100%, 0.615) | glutamate (flywire783, 100%, 0.50) | Both datasets predict glutamate despite "serotonergic" name — possible co-transmitter or classification mismatch |
| adult serotonergic IP neuron (`FBbt_00110937`) | serotonin (male_cns_v1_0, 0.79) | dopamine (neuprint_JRC_OpticLobe_v1_0_1, 33%, 0.53) | Minor discordance; only 1 of 3 instances in OL predicts dopamine; majority prediction serotonin is consistent |
| adult CB0212 neuron (`FBbt_20004026`) | serotonin (male_cns_v1_0, 100%, 0.79) | — | Only one dataset; cross-check recommended before promoting claim |
| adult CB0364 neuron (`FBbt_20004147`) | acetylcholine (BANC888, 100%, 0.615) | serotonin (flywire783, 100%, 0.58) | Cross-dataset discordance; BANC female CNS vs FlyWire female brain may differ anatomically |

**Confidence/scope assessment**: ROBUST for the broad adult serotonergic SE population
(confirmed across three large adult datasets at moderate-to-high confidence). UNCERTAIN
for named subclasses that show cross-dataset discordances, especially PMPD and CB-prefixed
neurons.

### Loci design implication — serotonergic analog

Serotonergic neurons in the fly brain modulate global circuit state — they are numerically
fewer than dopaminergic neurons but project widely, controlling arousal, locomotion speed,
and sensory gating. The serotonergic system in Loci maps to **gain control and
speed-accuracy tradeoff**:

- A "high serotonin" state analog in Loci corresponds to: slow-but-careful reasoning,
  higher evidence thresholds before claim promotion, increased weighting of provenance checks
- A "low serotonin" state analog corresponds to: faster, more heuristic routing,
  lower thresholds (appropriate for time-critical or bandwidth-limited queries)
- The PMPD discordance (dopamine in female brain, serotonin in male CNS) is a direct
  example of how neuromodulatory identity can be dataset-dependent — exactly the pattern
  that Loci's provenance envelope is designed to handle

---

## Octopaminergic circuits

### Known neurotransmitters (curated)
- **Primary transmitters**: octopamine secretion (`GO_0061540`) and tyramine secretion (`GO_0061546`) — octopamine is synthesized from tyramine; the co-transmitter annotation is well-established in Drosophila literature
- **Also curated for some subclasses**: acetylcholine, GABA, glutamate — lineage-specific or co-transmitter annotations
- **Scope note**: 222 known-NT entries across 127 cell type IDs.

### Predicted neurotransmitters (connectome-derived)

**High-confidence, cross-dataset consistent entries:**

| Cell type | nt_label | Dataset | Instances | % of class | Mean confidence |
|---|---|---|---|---|---|
| adult octopaminergic neuron (`FBbt_00058203`) | octopamine secretion | male_cns_v1_0 | 36 | 100% | 0.703 |
| adult octopaminergic neuron | octopamine secretion | neuprint_JRC_OpticLobe_v1_0_1 | 20 | 100% | 0.613 |
| adult octopaminergic neuron | octopamine secretion | neuprint_JRC_Hemibrain_1point2point1 | 16 | 100% | 0.578 |
| adult octopaminergic neuron | octopamine secretion | flywire783 | 7 | 100% | 0.578 |
| octopaminergic AL2b1 (`FBbt_00110146`) | octopamine secretion | male_cns_v1_0 | 2 | 100% | 0.860 |
| octopaminergic AL2b1 | octopamine secretion | neuprint_JRC_Hemibrain_1point2point1 | 2 | 100% | 0.755 |
| octopaminergic AL2b1 | octopamine secretion | neuprint_JRC_OpticLobe_v1_0_1 | 2 | 100% | 0.750 |
| octopaminergic AL2b1 | octopamine secretion | flywire783 | 2 | 100% | 0.600 |
| octopaminergic VUMa1–VUMa8 (family) | octopamine secretion | male_cns_v1_0 + flywire783 + hb | 2 each | 100% | 0.55–0.87 |
| adult mandibular octopaminergic VM | octopamine secretion | male_cns_v1_0 | 5 | 100% | 0.814 |
| mushroom body octopaminergic neuron (`FBbt_00048137`) | octopamine secretion | neuprint_JRC_Hemibrain_1point2point1 | 3 | 100% | 0.540 |
| mushroom body octopaminergic neuron | octopamine secretion | neuprint_JRC_OpticLobe_v1_0_1 | 3 | 100% | 0.687 |

**Discordant / uncertain entries:**

| Cell type | Dataset A prediction | Dataset B/C prediction | Notes |
|---|---|---|---|
| octopaminergic AL2b2 (`FBbt_00053405`) | acetylcholine (flywire783, 100%, 0.858) | acetylcholine (ol, 100%, 0.898); acetylcholine (male_cns, 100%, 0.923) | **Three-dataset acetylcholine prediction at very high confidence** despite "octopaminergic" class name — possible co-transmitter identity or classification mismatch; treat as flagged |
| adult octopaminergic and glutamatergic neuron (`FBbt_00052346`) | octopamine (ol, 100%, 0.86) | glutamate (MANC, 100%, 0.785) | This type is ontologically labeled co-transmitter; discordance is expected — not a provenance error |
| octopaminergic VUMd4 (`FBbt_00110330`) | octopamine (male_cns, 100%, 0.82) | GABA (MANC, 100%, 0.565) | Male CNS vs male VNC scope difference may account for this; treat as uncertain |
| octopaminergic VPM2 (`FBbt_00110150`) | octopamine (male_cns, 0.735); octopamine (flywire783, 0.51) | GABA (MANC, 0.54) | MANC VNC vs whole-CNS scope; moderate confidence discordance |
| octopaminergic ASM2 (`FBbt_00110169`) | dopamine (flywire783, 100%, 0.57) | — | Only one dataset; not enough data to assess |
| octopaminergic ASM3 (`FBbt_00110170`) | dopamine (flywire783, 100%, 0.52) | serotonin (male_cns, 100%, 0.525) | Both very low confidence; treat as below-threshold |

**Confidence/scope assessment**: ROBUST for the main adult octopaminergic VUM/VPM neuron
families and the AL2b1 type (four-dataset agreement). UNCERTAIN for AL2b2 (three-dataset
acetylcholine disagreement with class name), VUMd4, and ASM-type neurons.

The AL2b2 case is particularly notable: the ontology assigns it to the octopaminergic class,
but three independent connectome datasets at high confidence predict acetylcholine. This
may indicate a co-transmitter scenario (octopamine + acetylcholine) not captured in the
current ontology, or a classification boundary issue. This should not be treated as settled
without further literature review.

### Loci design implication — octopaminergic analog

Octopamine in the fly brain is the functional analog of norepinephrine in mammals:
it mediates arousal, fight-or-flight urgency, and gain adjustment for sensory circuits.
The VUM neuron family (bilateral, widespread projection) is the primary arousal signal.
In Loci, the octopaminergic analog maps to **urgency-weighted routing**:

- High urgency signal: relax evidence thresholds, prefer speed over depth, allow lower-confidence
  claims to proceed through the routing pipeline
- Low urgency / baseline arousal: normal evidence gating, standard confidence thresholds

The AL2b2 discordance is a direct example of the "named class vs. predicted identity"
conflict that Loci's provenance model must distinguish: the ontology label is not the
same as the per-instance connectome prediction, and they require separate provenance tags.

---

## Cross-system comparison

### Robustness summary

| System | Cross-dataset consistency | Main caveats |
|---|---|---|
| Dopaminergic | **High** — 100% dopamine at ≥0.56 across 5 datasets for main class | Some subclass co-transmitter annotations in ontology; not contradictions |
| Serotonergic (broad) | **High** — 100% serotonin across 4 datasets for SE/adult serotonergic class | Specific subclass discordances (PMPD, PLP) require dataset scope |
| Octopaminergic (VUM family) | **High** — octopamine confirmed across 3–4 datasets for core VUM/VPM types | AL2b2 anomaly (3-dataset ACh prediction) stands out as flagged |

### Population scale and diffuseness

| System | Instances (all datasets) | Character |
|---|---|---|
| Dopaminergic | ~17,488 | Many fine-grained identified subclasses; topographic cluster organization; valence-coded |
| Serotonergic | ~2,188 | Fewer, more diffuse; broad arousal and sensory gating effects |
| Octopaminergic | ~377 | Smallest population; highly bilateral VUM/DUM pattern; urgency and gain modulation |

This scale difference has design implications: dopaminergic evidence is the richest and
most subdivided; octopaminergic evidence is the sparsest. Higher confidence in dopaminergic
predictions is partly a function of the larger population available for prediction training.

### Discordance types observed

| Discordance type | Example | Interpretation | Loci action |
|---|---|---|---|
| Curated-vs-predicted discordance | DPM cell: curated serotonin (under serotonergic class), predicted dopamine | Different evidence families; both may be correct (co-transmitter or historical classification) | Store both with separate source tags; do not flatten |
| Cross-dataset discordance (female-vs-male CNS) | PMPD: dopamine in flywire783 (female), serotonin in male_cns | Sex-dependent identity or different neuron populations despite same class label | Require sex scope on any claim about PMPD identity |
| Cross-dataset discordance (brain-vs-VNC) | VUMd4, VPM2: octopamine in brain datasets, GABA in MANC (VNC) | Anatomy-dependent identity; VNC-specific expression may differ | Require anatomy scope on any claim about VUM-type identity |
| High-confidence anomaly | AL2b2: acetylcholine at 0.858–0.923 across 3 datasets | Possible co-transmitter not in ontology, or classification mismatch; requires literature review | Flag as "prediction vs ontology mismatch" — do not promote as settled |
| Apparent co-transmitter (expected) | octopaminergic+glutamatergic neuron: octopamine in OL, glutamate in MANC | Both named transmitters present; anatomy-specific expression | Annotate as co-transmitter; no discordance to resolve |

---

## Actionable recommendations for Loci

### 1. Routing: three-register state model

Use the three neuromodulatory systems as inspiration for a three-register state model
for Loci's routing engine:

- **Dopamine register (valence/prediction error)**: route a claim through the PAM-analog
  (confidence boost) or PPL1-analog (confidence penalty) depending on whether the incoming
  evidence confirms or disconfirms a prior hypothesis.
- **Serotonin register (speed-accuracy tradeoff)**: adjust evidence depth requirements
  based on inferred session state — high-urgency queries reduce evidence depth requirements;
  low-urgency queries increase them.
- **Octopamine register (urgency/arousal)**: gate whether the routing pipeline should
  apply full provenance checks or use a lighter-weight fast path for time-critical queries.

This is a design analogy; it should be implemented as a configurable routing policy, not
as a direct biological simulation. See `FLYBRAIN_IO_TO_LOCI_MAPPING.md`.

### 2. Confidence calibration: use cross-dataset agreement as a proxy

The evidence above shows that cross-dataset agreement (same nt_label, high mean_confidence,
multiple datasets) is a strong proxy for biological robustness. Loci should track whether
a FlyBrain-derived claim has been verified across ≥2 datasets with matched scope.

Implementation pattern:
```
claim_confidence_tier = match (dataset_agreement_count, mean_confidence):
  case (≥3 datasets, ≥0.6): "high" — promote to long-term memory
  case (2 datasets, ≥0.55): "medium" — store as working hypothesis
  case (1 dataset, any) OR (discordant): "low" — flag, require manual review
```

This mirrors the existing `memory_confidence` and `memory_promote` MCP surface.

### 3. Provenance caveats: specific required fields for neuromodulatory claims

Any FlyBrain-derived claim about neuromodulatory identity must carry:

```json
{
  "dataset_symbol": "fw",
  "dataset_version": "flywire783",
  "sex_stage_scope": "adult female brain",
  "nt_evidence_family": "predicted",  // or "curated"
  "nt_confidence": 0.636,
  "instances_supporting": 608,
  "percent_of_class": 100,
  "cross_dataset_agreement": ["mc", "BANC"],  // other datasets with matching prediction
  "discordances_flagged": []  // or list known discordances with dataset labels
}
```

Do not store a bare "dopamine" or "serotonin" claim without this envelope.

### 4. Tests and docs updates

- **Test**: Add a test that checks whether a FlyBrain-derived neuromodulatory claim includes
  `dataset_symbol`, `nt_evidence_family`, and `cross_dataset_agreement` before storage.
  This is the minimum provenance gate for neuromodulatory evidence.
- **Glossary update**: Add `nt_evidence_family`, `cross_dataset_agreement`, and
  `discordances_flagged` to `FLYBRAIN_REASONING_GLOSSARY.md`.
- **Evidence map update**: See `FLYBRAIN_EVIDENCE_MAP.md` — the neuromodulatory comparison
  row is added there.

### 5. Specific flagged cases requiring human review

Do not treat these cases as settled. Flag them in memory as `requires_review`:

1. **Octopaminergic AL2b2** (`FBbt_00053405`): three datasets predict acetylcholine at
   0.858–0.923 confidence; ontology labels it octopaminergic. Possible co-transmitter
   scenario not yet reflected in the ontology.
2. **Serotonergic PMPD neuron** (`FBbt_00110955`): dopamine in female brain (flywire783),
   serotonin in male CNS (male_cns_v1_0). Sex-dependent identity or population mismatch.
3. **Serotonergic PLP neuron** (`FBbt_00110945`): both available datasets predict glutamate,
   not serotonin. Possible classification boundary or co-transmitter.

---

## What is robust vs uncertain

### Robust (safe to use in Loci reasoning without additional qualification)

- Adult dopaminergic neuron identity (dopamine) is confirmed across 5 datasets at 100% instance agreement.
- Adult serotonergic SE neuron identity (serotonin) is confirmed across 3 large adult datasets.
- Core octopaminergic VUM/VPM family identity (octopamine) is confirmed across 3–4 datasets.
- The octopamine+tyramine co-transmitter pattern is ontology-curated and consistent with the biology.
- Cross-dataset discordances in PMPD, AL2b2, and PLP neurons are real, not query artifacts.

### Uncertain / blocked (require scope qualification or do not promote without review)

- Serotonergic PMPD neuron transmitter identity: dataset-dependent.
- Serotonergic PLP neuron transmitter identity: likely glutamate but class name is misleading.
- Octopaminergic AL2b2 transmitter identity: strongly predicted non-octopaminergic but ontology says otherwise.
- Any single-dataset prediction for a CB-prefixed serotonergic neuron.
- Larval (l1em) and FAFB (fafb) datasets: no predictions returned at the class level for these systems
  — absence of predictions is absence of data, not absence of biology.

---

## Evidence map integration

This document is referenced from `FLYBRAIN_EVIDENCE_MAP.md` as supporting evidence for
the neuromodulatory comparison row. The live VFB tool surface that produced the data is
the `virtual-fly-brain-get_known_neurotransmitters` and `virtual-fly-brain-get_predicted_neurotransmitters`
calls, executed with `split_by_dataset=true`, `min_confidence=0.5`, and `exclude_dbs=[]`
(all datasets included).

---

Status: initial release, 2026-09-22. Cross-dataset predictions current as of session date.
Revisit if dataset versions change (new flywire, BANC, or male_cns releases).
