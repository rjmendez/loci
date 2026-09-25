# Research notes: datasets-benchmarks (2026-09-24)

Topic: existing benchmarks, tooling and label provenance for ML on connectomes (F1, F3, F8).
All sources below were fetched (abstract / repo / methods at minimum) unless marked UNVERIFIED.

## Sources

1. Dorkenwald et al. 2024, "Neuronal wiring diagram of an adult brain", Nature, doi:10.1038/s41586-024-07558-y
   - 139,255 neurons; "incorporates annotations of cell classes and types, nerves, hemilineages and predictions of neurotransmitter identities". NT = model output.
   - Links: F3 (NT in FlyWire is predicted, not measured), F8.
2. Schlegel et al. 2024, "Whole-brain annotation and multi-connectome cell typing of Drosophila", Nature, doi:10.1038/s41586-024-07686-5 (PMC11446831)
   - Flow: "every neuron is either afferent, efferent or intrinsic"; 9 superclasses. Hemilineages from cell body fibre bundles vs light data: 183 hemilineages for 88% of central brain neurons.
   - Cell types: "NBLAST morphology clustering initially yielded 5,235 morphology types; multiple rounds of CBLAST connectivity clustering split some types" -> cell_type is partly CONNECTIVITY-DEFINED.
   - Cross-brain cosine connectivity similarity lower than within-brain, effect 0.045 +/- 0.096 -> cross-animal transfer should work (F9).
   - Links: F1 CONFIRMS/caveat, F8 CONFIRMS circularity, F2 METHOD (hemilineage is morphology-defined, not wiring -> good non-circular target), F9 EXTENDS.
3. flyconnectome/flywire_annotations (GitHub, Supplemental_file1_neuron_annotations.tsv)
   - top_nt = "top predicted neurotransmitter ... averaging confidences"; top_nt_conf; known_nt + known_nt_source (literature). status outlier_seg / outlier_bio.
   - REUSE: known_nt/known_nt_source as ground truth; drop top_nt as target (F3). Use tagged release matching v783.
4. Eckstein et al. 2024, "Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster", Cell, doi:10.1016/j.cell.2024.03.016
   - 87% synapse / 94% neuron (FAFB-Catmaid); 91% of 624 FlyWire types. GT: 356 cell types from 21 studies. Split by whole neuron (70/10/20); hemilineage-held-out tests "remained high"; "88% of hemilineages' predictions were strongly biased toward a singular transmitter".
   - Links: F3 (the "labels" our wiring model hits on BANC/ol are these image predictions; image model tops wiring) ; F2/F5 (NT strongly lineage-tied -> random vs grouped gap expected).
5. funkelab/drosophila_neurotransmitters (GitHub, CC-BY-4.0)
   - gt_data.csv: one row per cell type x study, confidence 1-5 ("5: evidence for protein expression in the given cell type, cell type specific labelling"), >900 cell types, 71 studies, maps to FlyWire, hemibrain, FANC, MANC, optic-lobe, maleCNS, BANC, L1.
   - REUSE as the NT ground-truth table with confidence filter (F3, F8). Male-CNS synister trained on confidence>=3 subset.
6. funkelab/synister_malecns (GitHub): male-CNS NT model trained on the above GT "filtered by confidence level 3", split by body id. -> any male-CNS NT GT neuron is also in the image model's training set; predictedNt vs GT agreement is not independent.
7. Takemura et al. 2024, "A Connectome of the Male Drosophila Ventral Nerve Cord", eLife reviewed preprint, doi:10.7554/eLife.97769.1
   - MANC NT: "Ground truth neurotransmitter labels for 187 neurons (67 acetylcholine, 55 GABA, 65 glutamate)", 80/20 neuron split, only ACh/GABA/Glu. MANC NT columns are predictions from a tiny GT.
8. Marin et al. 2024, "Systematic annotation of a complete adult male Drosophila nerve cord connectome...", eLife, doi:10.7554/eLife.97766.1
   - Hemilineage: soma-tract seeding + NBLAST; NT predictions "helped to confirm or distinguish" hemilineage IDs (so hemilineage label partly uses NT predictions). 97% thorax, 38% abdomen assigned.
   - Systematic types: "Typing is based on synaptic connectivity rather than morphology"; "Each annotated hemilineage is independently clustered" -> MANC type is connectivity-defined AND nested within hemilineage.
   - Links: F2 caveat, F8 CONFIRMS.
9. Bates et al. 2025, "Distributed control circuits across a brain-and-cord connectome" (BANC), bioRxiv 10.1101/2025.07.31.667571 (Nature version exists) + github htem/BANC-project (v888 meta parquet) + htem/synister_banc
   - Metadata from NBLAST transforms of FlyWire/MANC types, manual review, and some matching "based on connectivity". NT: "We used another CNN to predict the neurotransmitter"; 'peptide' class added from literature over monoamine prediction. synister_banc predicts 8 NTs for synapses size>5; GT stored at gs://leelab_fly_cns/files/banc_nt_ground_truth (not documented in README).
   - Links: F3 CONFIRMS circularity suspicion (BANC NT is a CNN output), F8.
10. Berg et al. 2025, "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system", bioRxiv 10.1101/2025.10.09.680999 (Cell 2026)
   - 166,691 neurons, 11,691 types; typing uses "quantitative metrics of morphology (NBLAST) and connectivity similarity"; 97.5% have a type match to FlyWire/hemibrain/MANC. Download: body-annotations (curated) separate from body-neurotransmitters ("Aggregate neurotransmitter predictions").
   - UNVERIFIED (search snippet only, PDF too large): consensusNt agrees with per-neuron predictedNt for 88.4%; median confidence 0.96 ACh / 0.86 GABA / 0.81 Glu.
11. Nern et al. 2025, "Connectome-driven neural inventory of a complete visual system", Nature, doi:10.1038/s41586-025-08746-0 — 53,000 neurons into 732 types "by integrating ... connectivity information, neurotransmitter identity and expert curation". ol types connectivity-informed (F8).
12. Matsliah et al. 2024, "Neuronal parts list and wiring diagram for a visual system", Nature, doi:10.1038/s41586-024-07981-1 — type consistency via "distances in high-dimensional feature space" (connectivity features). FlyWire optic-lobe types connectivity-defined (F8, F1 ol family).
13. Winding et al. 2023, "The connectome of an insect brain", Science, doi:10.1126/science.add9330 — (via search snippet of paper text) neurons spectrally embedded on graph structure and clustered to 93 types -> L1 types circular (F8 CONFIRMS).
14. Schwartzman et al. 2025, "Connectivity Is All You Need: Inferring Neuronal Types with NTAC", bioRxiv 10.1101/2025.06.11.659184 — seeded >95% on visual system with 2% labeled; unseeded ~70% visual, 52% central brain; morphology clustering <10%. Does not address that GT types used connectivity. Closest prior art to F1; METHOD baseline.
15. omkar-dhakane/flywire-gnn (GitHub, unreviewed) — FlyWire v783 PyG dataset, super_class 9-class, 174 features incl. NT profiles, stratified random 70/15/15: MLP 0.985 acc / 0.71 macro-F1, SAGE 0.981 / 0.756, GCN 0.917 / 0.47. Confirms F1 magnitude and that features-only MLP ~= GNN (F9), but random split and NT features leak.
16. Lu et al. 2026, "A Benchmark Analysis of Graph and Non-Graph Methods for C. elegans Neuron Classification", arXiv:2603.02241 — sensory/inter/motor; attention GNNs beat LR/MLP on spatial+connection features. Coarse 3-class target akin to our flow/super_class.
17. Dorkenwald et al. 2025, "CAVE: Connectome Annotation Versioning Engine", Nature Methods 22:1112 — "reproducible connectome analysis ... at arbitrary time points" (materialization versions). METHOD for pinning label versions.
18. Codex FAQ (codex.flywire.ai/faq) — shows "predicted neurotransmitter type and confidence" and "verified neurotransmitter types ... from curated sources"; "treat verified fields as curated annotations and predicted fields as model output"; v783 default; bulk downloads for FAFB and BANC.
19. sjcabs/fly_connectome_data_tutorial (GitHub) — BANC, FAFB, male CNS, MANC, hemibrain "harmonized to use the unified metadata schema we used in the BANC project" (data/meta_data_entries.csv). REUSE instead of per-dataset loaders.
20. Elabbady et al. 2025 (Nature, doi:10.1038/s41586-024-07765-7) — MICrONS perisomatic features, 1,619 manually labelled column cells, 91% CV, 82% dataset-wide; identifies connectivity-defined types from non-connectivity features (the non-circular direction). Schneider-Mizell et al. 2025 (Nature, doi:10.1038/s41586-024-07780-8) — 1,352-cell column, inhibitory classes defined by targeting of dendritic compartments (connectivity-defined).

## Synthesis
- No standard grouped-split benchmark for fly connectome node classification exists; public attempts (flywire-gnn, NTAC) use random/stratified or seeded splits and connectivity-defined type labels. Our grouped-split + trivial-baseline gate is stricter than prior art and is a publishable contribution.
- NT: every release's NT column is an image-CNN prediction (FlyWire top_nt, BANC synister, MANC 187-neuron GT, male-CNS synister). Only known_nt (FlyWire) and drosophila_neurotransmitters gt_data.csv are experimental. F3's "passes on predicted labels" is expected distillation; evaluate only on gt_data.csv confidence>=4, grouped by hemilineage, and report the image model's own accuracy (94% neuron) as the ceiling.
- Connectivity-defined labels: FlyWire cell_type (CBLAST splits), MANC systematic type (connectivity clustering within hemilineage), optic-lobe types (Matsliah, Nern), L1 types (spectral clustering), MICrONS inhibitory classes. Non-connectivity labels: hemilineage (soma tract/cell body fibre), nerve, side, flow for afferent/efferent (nerve entry), fru/dsx, known_nt.
- Reuse: sjcabs harmonized schema, CAVE/Codex versioned snapshots (v783, v888), navis/fafbseg, coconatfly for cross-dataset matching, drosophila_neurotransmitters GT, NTAC as a baseline.
