# Research notes: neurotransmitter (NT) prediction (F3, F2) — 2026-09-24

Verified = fetched page text read by agent. Unverified = search snippet only.

## Sources
1. Eckstein et al. 2024, Cell 187:2574, doi:10.1016/j.cell.2024.03.016 (PMC11106717) — VERIFIED.
   EM-image synapse classifier (6 NTs). 87% synapse / 94% neuron (FAFB), 78% / 91% (hemibrain); 91% cell-type agreement.
   GT: 356 cell types from 21 studies; 3,025 FAFB + 5,902 hemibrain neurons. Neuron split 70/10/20; also neuropil and
   hemilineage splits ("accuracy remained high", no numbers in main text). 183 hemilineages: "88% ... strongly biased
   toward a singular transmitter"; 19 dual, 3 triple. Serotonin 33-38%. GABA neurons more local/smaller than ACh;
   ACh tends to feed forward to higher layers; GABA no such bias. -> F3 METHOD/EXTENDS, F2 CONFIRMS.
2. Lacin et al. 2019, eLife 8:e43701 — VERIFIED. 34 VNC hemilineages: 14 ACh, 8 Glu, 12 GABA; "All neurons within a
   hemilineage use the same neurotransmitter"; ChAT mRNA in Glu/GABA neurons but not translated. -> F2/F3 core prior.
3. Takemura et al. 2024, MANC connectome, eLife RP 97769 — VERIFIED. NT classifier (ACh/GABA/Glu) trained on 187 GT
   neurons (67/55/65), 80/20 split; per-neuron = mean of presynaptic probs. -> F3 caveat: MANC NT labels are predictions.
4. Marin et al. 2024, MANC systematic annotation, eLife RP 97766 — VERIFIED. Hemilineage assigned from soma tract/morphology,
   clustering incl. cosine connectivity, and "Fast-acting neurotransmitter predictions helped to confirm or distinguish
   between identifications"; exceptions 09B (Glu w/ cholinergic subset), 19A (GABA w/ cholinergic early-born). -> F2
   CIRCULARITY risk (hemilineage labels partly from NT preds + connectivity clustering), F3 EXTENDS.
5. Schlegel et al. 2024, Nature 634:139 (PMC11446831) — VERIFIED. FlyWire hemilineages from hemilineage tracts / LM
   comparison; "neurons in each hemilineage usually express a single fast-acting transmitter"; sibling hemilineages can differ.
6. Dorkenwald et al. 2024, Nature 634:124 — VERIFIED (abstract). FlyWire includes "predictions of neurotransmitter identities".
7. Nern et al. 2025, Nature 641:1225 (PMC11042306; bioRxiv 2024.04.16.589741) — VERIFIED. Optic-lobe NT classifier trained
   on 59 cell types (~2M synapses); validated on 79 held-out types (numbers in Fig 4e/Suppl Tab 5, not in text);
   new EASI-FISH data for 66 types. Type-level NT is prediction + expert curation. -> F3: ol "ground truth" is partly
   prediction-derived; per-type NT is a curated consensus.
8. Berg et al. 2025/2026, male CNS, bioRxiv 10.1101/2025.10.09.680999 (Cell 2026) — PARTIALLY VERIFIED: 166,691 neurons,
   11,691 types; NT predictions included; no accuracy numbers found in fetched text.
9. BANC: Bates et al. 2025, bioRxiv 2025.07.31.667571 (Nature) + github htem/synister_banc — VERIFIED. 8 NTs predicted for
   synapses size>5; neuron-level = average; "largely agree with previous predictions"; no accuracy numbers in text.
   -> F3: BANC NT label = classifier output (our "pass" is distillation).
10. flyconnectome/drosophila_neurotransmitters (GitHub, 2024) — VERIFIED. >900 cell types / 71 studies, confidence 0-5
    (5 = protein, cell-type-specific; 0 = educated guess). Negative evidence sparse. -> F3 METHOD: filter GT by confidence.
11. Allen et al. 2020, eLife 9:e54074 — VERIFIED. VNC scRNA-seq ~26k cells; "40% cholinergic, 38% GABAergic, 18%
    glutamatergic"; 31% cells >=2 FAN markers (attributed to ambient RNA); "did not identify single TFs that globally
    defined FAN identity"; gene signatures reflect neuroblast origin and birth order. -> F2 CONFIRMS, F3 EXTENDS.
12. Konstantinides et al. 2018, Cell 174:622 (bioRxiv 243113) — VERIFIED (abstract). Optic lobe: "same terminal characters,
    including ... neurotransmitter identity, can be regulated by different transcription factors in different cell types".
    -> F3: NT is convergent, not tied to one developmental program in OL; weak expectation of wiring->NT mapping there.
13. Harris, Wytock, Kovacs 2022, Adv Sci 9:2104906 (PMC9165506) — VERIFIED. C. elegans polarity inference; with
    expression+rules 76% resolved; structure-only network methods recover only 24-112 polarities at 95% precision.
    -> F3 CONFIRMS: wiring alone weakly constrains sign.
14. Fenyves et al. 2020, PLoS Comput Biol 16:e1007974 — VERIFIED. Polarity for ~73% of C. elegans chemical synapses using
    gene expression (NT + receptor), not wiring alone.
15. Deng et al. 2019, Neuron 101:876 — UNVERIFIED (403). Snippet: "mutual exclusion of the major inhibitory and excitatory
    transmitters in the same neurons".
16. kostas52675/fly-connectome (GitHub) — VERIFIED, no results. Parallel attempt at NT-from-wiring on FlyWire; uses
    "fraction of inhibitory inputs" (derived from partner NT predictions) -> leakage example to avoid.

## Synthesis
- No published primary paper found that predicts fly NT from connectivity alone against experimental GT with grouped splits.
  Field solves NT from EM ultrastructure (Eckstein). Our F3 negative result is thus novel-ish and plausible.
- Lineage is the dominant determinant of fast NT (Lacin; Eckstein 88% of 183 hemilineages). Wiring predicts hemilineage only
  partly (our F2 0.49 MANC) so an upper bound on wiring->NT via lineage is limited; grouped splits that hold out hemilineages
  remove exactly the signal that makes NT predictable (explains F5 0.79 random vs 0.44 grouped).
- All dataset NT columns (FlyWire, hemibrain, MANC, BANC, male CNS, OL) are CNN predictions from partly overlapping GT
  (flyconnectome repo). "Ground truth" for mc/ol evaluation must be restricted to experimentally backed types (confidence >=4)
  and those types must not be used as training labels for the CNN we compare to.
- MANC hemilineage labels used NT predictions and connectivity clustering -> F2 target partly circular.
- Real but weak wiring correlates exist (GABA local, smaller; ACh feedforward) -> modest above-trivial accuracy is expected.
