# Research notes: cell-type identification from connectivity (topic: celltype-from-connectivity)
Date: 2026-09-24. Read-only literature sweep for FlyBrain findings F1, F7, F8 (plus F2, F3, F9).
All numbers and quotes below come from pages fetched in this session. Where a number sits only in a figure or supplement, that is stated.

## Sources

1. Schlegel et al. 2024, Nature 634:139, doi:10.1038/s41586-024-07686-5 (PMC11446831). "Whole-brain annotation and multi-connectome cell typing of Drosophila"
   - Cell types were defined with NBLAST morphology, cosine connectivity similarity and combined embeddings. A cell type is "a group of neurons that are each more similar to a group in another brain than to any other cell in the same brain."
   - Of the hemibrain types, 56% were found unambiguously in FlyWire, 13% needed a merge or split, and 32% could not be re-identified. The result is 3,643 consensus types; 8,453 types are annotated in total.
   - Edge reproducibility: "53% of edges observed in hemibrain were found in FlyWire"; 61%/59% between hemispheres; edges with more than 10 synapses reproduce at >90%.
   - Circularity: "connectivity-based typing is typically used iteratively", and it "must be wielded with care".
   - Flow and super_class come from soma position and side, and from entry or exit nerve. Hemilineage comes from soma tract and bundle morphology, not from connectivity.
   - Links: F1 CONFIRMS, F8 CONFIRMS (types are partly connectivity-defined), F2 EXTENDS (hemilineage labels are independent of connectivity, so F2 is a clean test), F9 METHOD (cross-brain type definition).

2. Matsliah et al. 2024, Nature 634:166, doi:10.1038/s41586-024-07981-1 (PMC11446827; bioRxiv 10.1101/2023.10.12.562119). "Neuronal parts list and wiring diagram for a visual system"
   - Optic-lobe types were refined by hierarchical clustering of partner-type connectivity vectors (weighted Jaccard distance), seeded with morphological types. Result: 227 intrinsic types.
   - Membership "can be accurately predicted by a logical conjunction of on average five synaptic partner types". The clustering is "self-consistent" under nearest-cluster reassignment. The F-scores are in Extended Data Fig. 2 and were not transcribed.
   - The paper concedes it "used only connectivity at the final stage of cell typing".
   - Links: F1 CONFIRMS (and explains ol 0.99), F8 CONFIRMS (optic-lobe types are defined from partner-type connectivity, so this is circular for our ol cell_type/family targets), F7 EXTENDS (partner-type composition is the defining feature), METHOD (sparse predicates as an interpretable baseline).

3. Scheffer et al. 2020, eLife 9:e57443, doi:10.7554/eLife.57443. "A connectome and analysis of the adult Drosophila central brain"
   - 5,229 morphology types and 5,609 connectivity types. CBLAST, "a new tool to cluster neurons based on synaptic connectivity", splits morphology types into connectivity types.
   - Truncation at the volume boundary leaves many single-example or untyped neurons.
   - Links: F8 CONFIRMS (hemibrain type labels partly encode connectivity), F1 caveat (truncated neurons).

4. Winding et al. 2023, Science 379:eadd9330, doi:10.1126/science.add9330 (bioRxiv 10.1101/2022.11.28.516756)
   - Brain neurons were "spectrally embed[ded] ... and clustered" by connectivity, down to "90 fine-grained cell types". Mean within-cluster NBLAST is 0.82 ± 0.16. "clustering was based solely on connectivity".
   - Left-right homolog pairs came from "automated graph matching followed by manual review".
   - Links: F8 CONFIRMS strongly. Larval types are connectivity clusters, and the L/R pair groups used for grouped splits are also connectivity-derived. io_class needs a provenance check.

5. Marin et al. 2024, eLife reviewed preprint 97766, doi:10.7554/eLife.97766.1. MANC systematic annotation
   - Hemilineage was assigned from "soma tract entry position and gross morphology" plus NBLAST.
   - Systematic types: "Each annotated hemilineage is independently clustered based on how similar its neurons' connectivity is in this symmetrised network."
   - Secondary hemilineages share predicted NT.
   - Links: F2 CONFIRMS (hemilineage is a non-circular label), F8 EXTENDS (MANC type and possibly class labels are connectivity clusters nested inside hemilineage, which makes grouping by type or hemilineage important), F3 EXTENDS (NT is roughly constant within a hemilineage, so a model can reach NT through a lineage proxy; this fits F5's random-vs-grouped gap).

6. Berg et al. 2025/2026, bioRxiv 10.1101/2025.10.09.680999; Cell 2026 doi:10.1016/j.cell.2026.08.015 (DOI from the search result, not fetched). Male CNS connectome
   - 166,691 neurons and 11,691 cell types. "7,319 cross-matched cell types in the central brain"; 114 dimorphic, 262 male-specific, 69 female-specific.
   - The typing method is not described in the abstract. verified=partial.
   - Links: F9 METHOD (cross-matched types as transfer ground truth); dimorphic and sex-specific types need to be excluded when transferring to female BANC.

7. Bates, Phelps, Kim, Yang et al. 2026, Nature doi:10.1038/s41586-026-10735-w (bioRxiv 2025.07.31.667571). BANC; read via github.com/htem/BANC-project
   - NT annotations are the output of "the BANC fast-acting neurotransmitter classifier" (synister, Funke lab model). The cross-dataset match accuracy table uses NBLAST.
   - Links: F3 CONFIRMS the circularity suspicion. BANC NT "ground truth" is an image-classifier output, so a wiring model that passes on it is being scored against another model, not against biology.

8. Eckstein et al. 2024, Cell, doi:10.1016/j.cell.2024.03.016 (PMC11106717)
   - NT from EM images: 87% per synapse and 94% per neuron (FAFB); 91% of 624 FlyWire types.
   - Ground truth: 356 types from 21 studies. The split was "randomly assigning entire neurons (neuron split)", not by cell type.
   - "88% of our hemilineages' predictions were strongly biased toward a singular transmitter identity."
   - Links: F3 EXTENDS (these predictions are the labels in FlyWire, BANC and ol), F5 METHOD (their neuron split risks the same-type leakage that our grouped split removes).

9. Matsliah, Salmon, Bates, ... Lee, Saul et al. 2026, bioRxiv 10.64898/2026.06.14.732053 (PMC13307954). "Uncovering Sex Differences in the Drosophila Ventral Nerve Cord Through Connectome Alignment"
   - Topology-only ACDC graph alignment between MANC and BANC VNCs.
   - Reviewers judged connectivity matching better than NBLAST in 62% of cases and NBLAST better in 7%; connectivity was "as well as or better" in 88%.
   - MANC vs male-CNS alignment served as a same-sex baseline.
   - "network topology alone contains sufficient information to support large-scale cell-type alignment."
   - Links: F9 METHOD and CONFIRMS that cross-animal transfer by wiring is feasible. Unsupervised alignment is a strong baseline for the male-cns to BANC transfer.

10. Lee, Matsliah, Saul 2026, TMLR. "AC⊕DC Search: Behind the Winning Solution to the FlyWire Graph-Matching Challenge"
    - Winner of the FlyWire VNC matching challenge; runs "in less than 10 minutes on a laptop".
    - Links: F9 METHOD.

11. Pedigo, Winding, Priebe, Vogelstein 2023, Network Neuroscience 7(2):522, doi:10.1162/netn_a_00287. "Bisected graph matching improves automated pairing of bilaterally homologous neurons"
    - Using contralateral edges improves L/R pairing when edge correlation is sufficient. Only the abstract was readable (PMC was captcha-blocked).
    - Links: F8 (L/R pair labels themselves come from connectivity matching), F9 METHOD.

12. Elabbady et al. 2025, Nature, doi:10.1038/s41586-024-07765-7 (PMC11981918). MICrONS perisomatic ultrastructure
    - Trained on 1,619 expert-labelled column cells. Subclass accuracy: excitatory 90%, inhibitory 94% (with postsynaptic-shape features), non-neuronal 97.5%.
    - 91% on the column and 82% dataset-wide; tenfold CV.
    - Links: F1 EXTENDS (mammalian typing uses local ultrastructure rather than graph position; accuracy drops from column to dataset-wide, which is the analogue of our random-vs-grouped gap).

13. Schneider-Mizell et al. 2025, Nature, doi:10.1038/s41586-024-07780-8. MICrONS inhibitory census
    - 1,352 cells and more than 70,000 synapses. Inhibitory neurons were "classified ... based on targeting of dendritic compartments", i.e. connectivity-defined. Abstract only, read via the search snippet; PubMed was blocked. verified=partial.
    - Links: F8 CONFIRMS (a connectivity-defined label in mouse).

14. Witvliet et al. 2021, Nature (bioRxiv 10.1101/2020.04.30.066209)
    - "About 43% of all connections and 16% of all synapses are not conserved between animals"; modulatory neurons are the most variable and motor neurons the least.
    - Links: F7 and F9 context (weak edges are noisy across individuals; features should be synapse-weighted or thresholded).

15. Non-peer-reviewed parallel: github.com/omkar-dhakane/flywire-gnn
    - FlyWire v783, 9-class super_class, test accuracy MLP 0.9851, GraphSAGE 0.9812, GCN 0.9166.
    - Features include per-neuropil profiles and partner NT profiles. The split is a stratified random 70/15/15, not grouped.
    - Links: F1 parallel (same ~0.98 ceiling). F9: a plain MLP on hand-built features matched or beat GraphSAGE. It is leaky (random split plus neuropil features), so it is not evidence against our grouped results.

(Also found: Mehta et al. 2021, Network Neuroscience, doi:10.1162/netn_a_00195. ASE+GMM recovers blocks perfectly in simulated SBM graphs. Surrogate data only, so of limited relevance. Priebe et al. 2017, arXiv:1705.03297, ASE+GMM on the larval MB connectome.)

## Synthesis
- Across the field, connectivity-only typing is highly accurate. The field validates it mainly by self-consistency, left-right and cross-brain replication, and agreement with an independent modality (NBLAST morphology). It does not validate it with held-out supervised accuracy under grouped splits. Our grouped-split plus trivial-baseline gating is stricter than most published claims.
- Circularity is widespread and often acknowledged. Connectivity-defined labels:
  - hemibrain connectivity types (CBLAST)
  - optic-lobe types (final stage defined by connectivity)
  - larval types (pure spectral clustering)
  - MANC systematic types (connectivity clusters within hemilineage)
  - MICrONS inhibitory subclasses
  - L/R pairs in the larva (graph matching)
  Labels that are not connectivity-defined, and so are non-circular targets: hemilineage (soma tract), flow and super_class (soma, nerve), and literature-grounded NT.
- NT labels in FlyWire and BANC are image-classifier outputs (Eckstein; synister). F3's "pass" on predicted labels is therefore agreement between models. The training split in Eckstein was by neuron, not by type.
- Cross-animal: topology-only alignment (ACDC/ConSim) beats NBLAST in expert review. It is a strong unsupervised baseline for F9.

## Ideas to adopt
1. Add Matsliah-style sparse conjunctive predicates as an interpretable baseline for the cell_type and family targets.
2. Add an unsupervised ACDC or bisected graph matching baseline for male-cns to BANC transfer. Exclude dimorphic and sex-specific types (Berg).
3. Tag every label with its provenance class (connectivity-defined / morphology-defined / image-classifier / literature) and report gates separately for each class.
4. Report NT against Eckstein's literature ground-truth subset (356 types) instead of predicted NT.
5. Threshold or weight edges (more than 10 synapses reproduce at >90%) and test feature robustness to weak edges.
