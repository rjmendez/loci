# FlyBrain research synthesis: our findings against prior and parallel work

Run: research-20260924T2245Z, 2026-09-24. Inputs: 8 topic reports (celltype-from-connectivity, nt-prediction, gnn-connectome, stereotypy-transfer, graph-ml-leakage, calibration-hierarchical, moe-vs-global, datasets-benchmarks) and their citation checks. Our findings are provisional and some tracks (F9) are still running.

Citation handling:
- No source came back `not_found`, so none were dropped.
- Four were **misattributed** and are cited here in their corrected form: Mehta 2023, NeuNet 2024, Elabbady 2025 (the 82% figure), and Kratzert 2019 (baseline numbers).
- Claims marked **[unverifiable]** were not seen on any fetched primary page. They are there for context only and nothing here depends on them.

---

## 1) Bottom line

- **F1 matches the field, and our method is stricter.** Wiring alone recovers fly cell identity. Every group that looked finds this (Matsliah 2024: about 5 partner types define an optic-lobe type; NTAC >95% seeded; Matsliah 2026: topology alignment beats NBLAST). We found no published work that tests this with grouped splits and trivial-baseline gates, as our harness does.
- **Much of our high F1 accuracy is circular by construction.** The optic-lobe, larval, MANC-systematic and FlyWire CBLAST types were all defined partly from connectivity. super_class, flow, hemilineage and literature NT are the non-circular claims. Hemilineage has a caveat: MANC hemilineage assignment used NT predictions.
- **F3 looks new and plausible.** Wiring gives only weak, lineage-mediated NT signal. Nobody has published NT-from-wiring against ground truth with grouped splits. Every dataset NT column is CNN output (Eckstein 2024 family), so our BANC and optic-lobe "passes" measure how well we distil that CNN. They are not NT accuracy.
- **F5, F6 and F10 follow textbook patterns.** GBDT gains on random splits collapse on grouped splits (Kapoor L3.2; Bernett 2024). Many-class ECE is expected to be poor and is also biased upward by the estimator. Per-region MoE at our data scale is more likely to hurt than help.
- **We have one real gap to fill.** No peer-reviewed grouped/inductive GNN-vs-GBDT benchmark exists on any connectome. The same is true of cross-animal, wiring-only classifier transfer. F9 is publishable if we run it with care.

---

## 2) Finding by finding

### F1: Cell identity is predictable from wiring alone. **Expected, and partly circular.**
- **Confirms:**
  - [Matsliah 2024] 227 intrinsic optic-lobe types. Membership is "a logical conjunction of on average five synaptic partner types", and the final typing stage used connectivity only, seeded by morphological types.
  - [Nern 2025] ">99% of cell types can be distinguished by their top five connections" across 732 types (727 in the bioRxiv version).
  - [Schwartzman/NTAC 2025] >95% on the visual system with 2% of neurons labelled as seeds, >90% on the central brain. Unseeded: about 70% on the visual system and 52% on the full brain. NBLAST kNN stays below 50% even with 35% labelled.
  - [Matsliah 2026] In MANC-vs-BANC disagreements, reviewers preferred the connectivity match in 62% of 300 cases and NBLAST in 7%.
  - [Dhakane flywire-gnn, unreviewed] FlyWire super_class: MLP 0.985, GraphSAGE 0.981, GCN 0.917 accuracy.
- **Extends:** [Elabbady 2025] Mouse perisomatic (non-wiring) features reach 91% cross-validated. On an expert-checked validation set of 1,700 cells drawn from the full 94k-cell dataset they reach 82%. That drop is the analogue of our random-vs-grouped gap.
- **Where we differ:** Every published number uses a transductive, seeded or random split. None applies grouped hold-outs or trivial-baseline gates. The public ML parallel (flywire-gnn) uses a random stratified split plus neuropil and predicted-NT features, which is the leakage we guard against.
- **Suspicious parts:**
  - ol family 0.99 / 0.76 and l1em io_class 0.95: these labels are partly or wholly connectivity-defined [Matsliah 2024; Nern 2025; Winding 2023].
  - mc cell_class: male-CNS types used NBLAST plus connectivity similarity [Berg 2025].
  - Report these as "recovering connectivity-derived annotations".
- **Clean claims:** super_class (BANC 0.96 vs 0.38 trivial; 0.92 with neuropil features removed) and flow. Their provenance is soma position and nerve entry/exit [Schlegel 2024]. BANC's exact flow rule is still undocumented.
- **Missing control:** A size/degree-only baseline. [Subramonian 2024] shows high-degree nodes are easier to classify whatever the model.

### F2: Hemilineage is partly predictable (MANC 0.49 vs 0.14). **Plausible, but MANC labels are partly circular.**
- **Confirms that lineage is a real, non-wiring biological variable:**
  - [Lacin 2019] "All neurons within a hemilineage use the same neurotransmitter" (34 VNC hemilineages).
  - [Allen 2020] VNC transcriptomes track neuroblast origin.
- **Contradicts the cleanness of the label:** [Marin 2024] MANC hemilineage assignment used soma tract plus NBLAST plus "several clustering methods", including cosine connectivity clustering. NT predictions "helped to confirm or distinguish" identifications. Assignment rates are 97% thoracic and 38% abdominal.
- **Cleaner target:** FlyWire hemilineages. There are 183, covering 88% of the central brain, assigned from cell-body-fibre tracts and light-level comparison [Schlegel 2024]. The check found no sign that connectivity was used, but that is inferred from absence.
- **Status:** novel as a supervised, gated result. We need to re-run it on FlyWire hemilineages.

### F3: NT from wiring fails against ground truth and "passes" on classifier labels. **Novel negative result, consistent with the literature; the circularity is confirmed.**
- **Label provenance:**
  - BANC NT is output of "the BANC fast-acting neurotransmitter classifier" (synister_banc; 8 NTs; no accuracy reported) [Bates 2025/2026].
  - FlyWire top_nt comes from [Eckstein 2024]: 94% per neuron and 91% of 624 types, trained on 356 types from 21 studies with a neuron-level split, not a type-level one.
  - MANC NT was trained on only 187 GT neurons (ACh/GABA/Glu only) [Takemura 2024].
  - Male-CNS used synister_malecns, trained on GT at confidence ≥3 and split by body id.
  - The optic-lobe classifier was trained on 59 types and validated on 79 held-out types [Nern 2025]. Its type-level NT mixes predictions with curation, so even our "ground truth" for mc and ol may be partly prediction-derived.
- **Mechanism:**
  - "88% of our hemilineages' predictions were strongly biased toward a singular transmitter identity" [Eckstein 2024]. This is the lineage route that grouped splits remove, and it matches F5's BANC NT 0.79 on a random split vs 0.44 grouped.
  - Weak structural correlates exist. GABAergic neurons are more local and smaller. Correction from the check: cholinergic *and* glutamatergic neurons tend to target higher layers [Eckstein 2024]. So mc 0.52–0.56 vs 0.38 trivial is the kind of above-chance result we should expect.
  - In the optic lobe, the same NT comes from different transcription-factor programs [Konstantinides 2018].
- **Cross-species:** Structure-only sign inference in C. elegans recovers just 24–112 polarities at 95% precision. With expression data added, 76% of interactions are resolved [Harris 2022].
- **Prior art:** No peer-reviewed NT-from-wiring study in fly was found. The only parallel is an unreviewed hobby repo (kostas52675) with no results. It uses a "fraction of inhibitory inputs" feature derived from partner NT predictions, which is the leak to avoid.
- **Reframe:** "Wiring carries weak, lineage-mediated NT signal." Relabel the BANC and ol NT targets as "agreement with a synister CNN (distillation)".

### F4: Some targets are trivial or proxies. **Expected, and a textbook case.**
- [Kapoor & Narayanan 2022] Leakage type L2, "a feature is a proxy for the outcome". The BANC flow lookup (0.95) and the size-tracking "connectivity tier" fall here.
- [Bernett 2024] Once PPI leakage was removed, deep models "learn solely from sequence similarities and node degrees" and drop to random. Size/degree is the canonical shortcut, so a size/degree-only gate is required.
- [Subramonian 2024] Degree bias exists whatever the model. Report accuracy by degree decile.
- [Geirhos 2020; DeGrave 2021] Shortcuts can survive even external validation.

### F5: Naive Bayes fails; LR and GBDT pass; GBDT memorizes lineage (0.79 random vs 0.44 grouped). **Expected.**
- Random vs grouped gap: this is L3.2 non-independence [Kapoor 2022]. The same structure appears in [Park & Marcotte 2012] (C1/C2/C3 pair classes) and [Roberts 2017] (block CV; the "serious underestimation of predictive error" quote came from a search snippet).
- Once leakage is fixed, complex models are no better than LR [Kapoor 2022].
- Duplicate or near-duplicate nodes leak between train and test [Platonov 2023]. Our analogue is L/R homologs and same-type columns.
- Trees remain state of the art on tabular data of about 10K samples [Grinsztajn 2022].
- Naive Bayes scores are "typically too extreme" [Zadrozny & Elkan 2002; Niculescu-Mizil & Caruana 2005].
- Effective n is the number of groups, not neurons. CIs from fold standard errors are too narrow (±10% at n=100) [Varoquaux 2017].

### F6: Coarse classes calibrate well; many-class ECE is 0.17–0.43. **Expected; part of it may be estimator artifact.**
- Boosted trees push probabilities away from 0 and 1, and NB pushes them toward 0 and 1 [Niculescu-Mizil & Caruana 2005].
- Equal-width-bin ECE is biased when there are many small classes [Roelofs 2022; Kumar 2019: Platt and temperature scaling are "less calibrated than reported"]. Rankings flip depending on the metric [Nixon 2019].
- Grouped hold-out is a shift, and post-hoc calibration "falls short" under shift [Ovadia 2019].
- Remedies suited to many classes:
  - Dirichlet-ODIR [Kull 2019]
  - top-label M2B binning [Gupta & Ramdas 2022]
  - Top-versus-All [Le Coz 2024]
  - Venn-Abers or Beta calibration. Platt and isotonic can degrade strong tabular models [Manokhin 2026].
- Hierarchy: FlyWire defines flow > super_class > class > type [Schlegel 2024]. Conditional softmax per parent, as in the WordTree pattern [Redmon 2017], gives predictions that are consistent across levels and can back off to a parent.
- Label noise sets a floor: 32% of hemibrain morphology types could not be re-identified in FlyWire [Schlegel 2024].

### F7: 2-hop partner composition is the most informative feature family. **Expected.**
- Type definitions are partner-type vectors [Matsliah 2024; Nern 2025: top-5 connections], so partner composition should dominate, partly by construction.
- A 2-layer GraphSAGE is roughly a learned version of 2-hop composition, so expect small GNN gains [Hamilton 2017].
- Higher-order neighbourhoods help under heterophily [Zhu 2020].
- **Cross-animal robustness:**
  - Edges above 10 synapses reproduce more than 90% of the time. They are 16% of edges and carry about 79% of synapses [Schlegel 2024].
  - About 43% of C. elegans connections vary between animals [Witvliet 2021].
  - Build 2-hop features from strong, normalized edges.

### F8: Label provenance dominated the effort. **Strongly confirmed; this is what matters most in the literature.**
- Connectivity-defined labels:
  - hemibrain CBLAST types (5,609 connectivity vs 5,229 morphology types) [Scheffer 2020]
  - FlyWire cell_type (NBLAST, then CBLAST splits) [Schlegel 2024]
  - the optic-lobe final stage [Matsliah 2024]
  - larval types: spectral clustering, about 90 in bioRxiv v1; 93 per the Science version is [unverifiable] [Winding 2023]
  - MANC systematic types, "Typing is based on synaptic connectivity rather than morphology" [Marin 2024]
  - male-CNS types [Berg 2025]
  - MICrONS inhibitory subclasses [unverifiable: Schneider-Mizell 2025]
- **Even the grouping keys can be connectivity-derived.** Larval L/R pairs came from graph matching [Winding 2023; Pedigo 2023]. Cross-dataset matches use NBLAST plus connectivity co-clustering [Schlegel 2024; Berg 2025; Stürner 2025; Bates 2025].
- **Precedent for per-column provenance:** Codex tells users to "treat verified fields as curated annotations and predicted fields as model output". FlyWire keeps top_nt (predicted) separate from known_nt/known_nt_source (literature).
- REFORMS asks for a per-feature legitimacy justification [Kapoor 2023].

### F9: GNN vs GBDT and male-cns→BANC transfer (in progress). **A real gap; set baselines and ceilings first.**
- **No grouped GNN-vs-GBDT study on any connectome was found.** The closest:
  - flywire-gnn (random split): MLP ≈ SAGE on accuracy; SAGE better on macro-F1, 0.756 vs 0.712.
  - NeuNet (corrected): hemibrain connectome-branch-only 0.587 vs skeleton-branch 0.861; NeuNet on hemibrain 0.917. The 0.936 figure is H01 (human), not Drosophila.
  - Lu 2026 (C. elegans, 3 classes): attention GNNs beat LR, MLP, LOLCAT and NeuPRINT.
- **Evaluation pitfalls:**
  - Model rankings flip across splits [Shchur 2018; Errica 2020].
  - Simple model plus label propagation beats GNNs [Huang 2020, C&S].
  - GBDT+GNN hybrids help [Ivanov 2021, BGNN].
  - Transductive embeddings leak [Grover 2016].
- **Transfer ceilings:**
  - Only 56% of hemibrain morphology types were unambiguously found in FlyWire [Schlegel 2024].
  - 26% of BANC non-optic-lobe neurons are unmatched [Bates 2025].
  - 92–99% of DNs and ANs match across FAFB, FANC and MANC [Stürner 2025].
  - 97.5% of male-CNS neurons match other datasets [Berg 2025].
  - Out of 7,319 cross-matched central-brain types, 114 are dimorphic, 262 male-specific and 69 female-specific [Berg 2025].
- **Technical confounds:**
  - FIB-SEM detects more than 40% more synapses than serial-section TEM [Scheffer 2025].
  - DNa02 raw-weight slope across datasets is 0.42, rising to 0.69 once normalized [Stürner 2025].
  - FlyWire postsynapse attachment is only about 44–45% [unverifiable: Dorkenwald 2024].
- **Baseline to beat:** Topology-only graph alignment (ACDC, "under 10 minutes on a laptop") [Lee 2026; Matsliah 2026]; bisected graph matching gives about 22% better larval pairing [Pedigo 2023].
- **Larva→adult:** Do not expect transfer. No larval MBIN–MBON connection survives metamorphosis [Truman 2023].

### F10: A per-region expert "brain cluster" is planned. **The literature mostly contradicts this as a default.**
- Global models with entity descriptors match or beat local ones:
  - [Montero-Manso & Hyndman 2021]
  - [Kratzert 2019, corrected]: EA-LSTM mean NSE 0.674 single / 0.705 ensemble on 447 basins, vs best basin-calibrated HBV 0.631 and mHM 0.627
  - [Abdelaal 2019]: a general SVM was best for scRNA typing
  - [Eckstein 2024]: one global network held up on neuropil and hemilineage splits
- MoE overfits with little data per expert. Dense beats sparse on 250-example tasks and sparse wins at 138k [Zoph 2022]. It helps only with genuine cluster structure [Chen 2022].
- Top-down routing propagates errors [Silla & Freitas 2011]. Elabbady's hierarchy works because its first level is 97% accurate.
- **Validated alternatives:**
  - hierarchical routing by predicted super_class, which is already 0.96–0.99 [scHPL, Michielsen 2021; Elabbady 2025]
  - Laplacian-regularized stratified models [Tuck 2019]
  - confidence-gated escalation from a weak to a strong expert [Mowst, Zeng 2024]
  - parameter-shared ensembles [TabM, Gorishniy 2024]

---

## 3) What the field does that we don't (ranked by expected impact)

1. **Per-target label-provenance tags and gating by provenance.** Tag each target as measured, curated-morphology, connectivity-defined or model-predicted. Exclude predicted targets from pass/fail and report them as distillation checks. *Impact: very high.* This changes how F1, F3 and F8 are read.
   Evidence: Codex FAQ; flywire_annotations top_nt vs known_nt; REFORMS; Kapoor L2.
2. **Score NT only against literature ground truth.** Use funkelab/flyconnectome drosophila_neurotransmitters: >900 types from 71 studies, confidence 0–5, CC-BY-4.0. Filter to confidence ≥4 and deduplicate against CNN training types.
   Also add a **hemilineage→NT oracle baseline** and a two-stage wiring→lineage→NT test. *Very high for F3.*
3. **Size/degree-only baseline gate plus accuracy by degree decile.** *High, for F1 and F4* [Bernett 2024; Subramonian 2024].
4. **Group-level bootstrap and group-permutation nulls.** Add a restricted within-class feature permutation, which tests whether feature interactions carry signal. Also plot a graded split curve: random → type → hemilineage → hemisphere → animal. *High* [Varoquaux 2017; Ojala & Garriga 2010; OGB].
5. **Cross-animal validation the way the field does it.** Measure left→right hemisphere transfer inside one dataset as a ceiling, since Stürner reports L/R differences comparable to cross-dataset differences; the check did not re-fetch that quote. Run graph-matching baselines (ACDC, bisected GM). Use Park-Marcotte C1/C2/C3-style strata. *High for F9.*
6. **Robust wiring features.** Use normalized fractions and ranks instead of raw counts, and threshold edges at 5–10 synapses before building 2-hop features. *Medium-high:* this guards against the >40% FIB-SEM vs TEM synapse-detection gap [Scheffer 2025; Schlegel 2024; Witvliet 2021; Pedigo eLife 2023].
7. **Calibration protocol.** Use a grouped train/calibrate/test split. Measure with ECE_sweep, classwise-ECE, log-loss and Brier, with CIs. Try Dirichlet-ODIR, TvA, M2B and Venn-Abers. *Medium* [Roelofs 2022; Kull 2019; Ovadia 2019].
8. **Hierarchical outputs.** Use per-parent conditional probabilities, hF1 and LCA mistake severity, abstention, and hierarchical or clustered conformal sets. *Medium* [Redmon 2017; Kosmopoulos 2013; Bertinetto 2020; Ding 2023; Mortier 2025].
9. **Interpretable sparse-conjunction baseline.** Score "types = AND of about 5 partner types" [Matsliah 2024]. *Medium, mainly for explaining F7.*
10. **Reuse existing tooling instead of building our own.** *Medium effort-saver.*
    - sjcabs/fly_connectome_data_tutorial: a harmonized metadata schema across BANC, FAFB, male-CNS, MANC and hemibrain.
    - coconatfly and natverse for cross-dataset matching.
    - Pinned CAVE snapshots for versioning.

---

## 4) Parallel efforts to watch or collaborate with

- **Murthy/Seung labs, Princeton (FlyWire/Codex):**
  - Matsliah: optic-lobe typing, ACDC, VNC sex alignment.
  - NTAC [Schwartzman 2025], the closest ML competitor. Its seeded protocol leaks type members, so ask for, or run, a grouped NTAC evaluation.
  - Lee/Matsliah/Saul: ACDC, winner of the FlyWire VNC matching challenge.
- **Jefferis/Cambridge (flyconnectome; Schlegel, Bates, Stürner, Marin):** cross-dataset typing, coconatfly, flywire_annotations releases (v3.0.0 revised using male CNS), and the NT ground-truth repository. Their "unreviewed / connectivity-typed" flags would help us.
- **Funke lab, Janelia (Eckstein; synister_banc, synister_malecns):** the NT classifiers behind our predicted labels. Their neuropil- and hemilineage-split accuracies are the image-model analogue of our grouped gap.
- **Janelia FlyEM (Berg, Nern, Takemura, Scheffer):** male-CNS v1.0 with its dimorphic-type lists, the optic-lobe inventory, and synapse-detection efficiency work.
- **Bates/Wilson/Lee (BANC, Harvard/HMS):** BANC v888 and the harmonized schema tutorial (sjcabs). The BANC NT GT location (gs://leelab_fly_cns) and the flow rule still need to be requested.
- **Zlatic/Cardona and Vogelstein/Priebe (Pedigo):** larval connectome, bisected graph matching, network-symmetry tests.
- **Turaga/Macke (Lappalainen), Shiu/Scott, Litwin-Kumar:** connectome-constrained dynamics that *consume* predicted NT. Our F3 provenance point bears directly on their sign inputs: Shiu reports 1%→16% false positives when the glutamate sign is flipped.
- **Unreviewed parallels:** omkar-dhakane/flywire-gnn (a random-split PyG dataset, a good foil) and kostas52675/fly-connectome (an NT-from-wiring leak example).
- **Adjacent benchmarks:** Lu 2026, C. elegans graph vs non-graph neuron classification (arXiv 2603.02241). MICrONS (Elabbady; Schneider-Mizell [unverifiable]) for mouse cross-checks.
- **Opportunity:** there is no standard grouped-split fly-connectome node-classification benchmark. FlyBrain's gate suite plus provenance tags could become one.

---

## 5) Roadmap implications

**Add**
1. A `label_provenance` enum per target column. Harness gates apply only to measured and curated-morphology targets. *Evidence: F8 section; Codex FAQ; REFORMS.*
2. An NT-GT target from drosophila_neurotransmitters at confidence ≥4, grouped by hemilineage and type. Add a hemilineage→NT oracle, a two-stage route, and binary ACh vs GABA+Glu. *Evidence: Eckstein 88%; Lacin 2019; Harris 2022.*
3. A size/degree-only gate and degree-decile reporting. *Evidence: Bernett 2024; Subramonian 2024.*
4. Group bootstrap, group permutation, and a graded split curve. *Evidence: Varoquaux 2017; Ojala 2010; Shchur 2018.*
5. For F9:
   - Baselines: HGB+C&S (train labels only), a BGNN-style hybrid, and ACDC / bisected graph matching.
   - Run GraphSAGE fully inductively, with test-group partner categories masked during message passing.
   - Report macro-F1 and use multiple seeds.
   - Test on the left→right ceiling and on the matched ~74% BANC subset, with dimorphic and sex-specific types excluded.
   *Evidence: Huang 2020; Ivanov 2021; Matsliah 2026; Berg 2025; Bates 2025; Bouthillier 2021.*
6. Feature normalization and an edge threshold of 5–10 synapses, then re-run the F7 ablation. *Evidence: Schlegel 2024; Scheffer 2025; Stürner 2025.*
7. A grouped calibration fold and debiased ECE, then a hierarchical conditional model (super_class, then cell_class). *Evidence: Roelofs 2022; Ovadia 2019; Redmon 2017.*

**Change**
- Use FlyWire hemilineages instead of MANC for the headline F2 claim [Marin 2024; Schlegel 2024].
- Reword F3 to "weak lineage-mediated signal". Report the BANC and ol NT "passes" as CNN distillation.
- Report ol family, l1em type and mc/MANC cell_class results as "recovery of connectivity-derived annotations". Evaluate cell_type only in cross-dataset transfer.
- Add the size/degree gate to the F4 "connectivity tier" target, or retire the target.

**Cut or defer**
- Defer the per-region "brain cluster" (F10). First run a region×feature interaction test and compare against a global model with neuropil features on leave-one-region-out splits. If experts are wanted, use super_class-conditioned hierarchy or Laplacian-stratified LR instead of anatomical routing [Zoph 2022; Montero-Manso 2021; Tuck 2019].
- Drop Naive Bayes from the learner set, or keep it as a documented weak baseline only [F5; Zadrozny 2002].
- Keep partner-NT-derived features (e.g. fraction of inhibitory inputs) out of every NT model.
- Do not attempt larva→adult transfer [Truman 2023].

---

## 6) Open questions and suggested experiments

| # | Question | Experiment |
|---|---|---|
| 1 | How much of F1 is size/degree? | Size/degree-only HGB per target. Report the margin and accuracy by degree decile. |
| 2 | Is the wiring→NT signal only lineage? | Hemilineage-only NT oracle vs direct wiring→NT vs wiring→lineage→NT, all on literature GT (conf ≥4), grouped by hemilineage. |
| 3 | How much of the mc/ol NT "GT" overlaps CNN training types? | Join our targets to gt_data.csv and the CNN training lists. Re-score on the non-overlapping subset. |
| 4 | Is the cell_class ECE of 0.17–0.43 real? | Re-measure with ECE_sweep, classwise-ECE and a debiased estimator with group bootstrap CIs, plus a per-class count table. |
| 5 | Does hierarchy improve cell_class calibration? | P(class \| super_class)·P(super_class) vs flat, compared on ECE, hF1 and a coverage–accuracy curve. |
| 6 | Does the GNN add anything over 2-hop features under inductive masking? | GraphSAGE vs HGB+2-hop vs HGB+C&S vs BGNN, with ≥5 seeds, a grouped split, and macro-F1. |
| 7 | What is the cross-animal ceiling? | Left→right transfer within fw, mc and BANC, then male-cns→BANC on reviewed matches. Stratify morphology-only vs connectivity-assisted matches, and dimorphic vs shared types. |
| 8 | How much of the transfer gap is technical rather than biological? | Raw vs normalized/thresholded features, compared across FIB-SEM (mc) → TEM (BANC). |
| 9 | Is there region-specific structure worth routing on? | Region×feature interaction test and per-region SHAP, then per-region vs global vs Laplacian-stratified models on leave-one-region-out splits. |
| 10 | Does NTAC survive grouped hold-outs? | Run NTAC with no seeds from test types or L/R pairs. |
| 11 | Provenance gaps still open | Is larval io_class connectivity-derived? What is BANC's flow/super_class rule? Which FlyWire types were CBLAST-split? What is male-CNS consensusNt (the 88.4% agreement figure is **unverified**)? Where does BANC NT GT come from? |
| 12 | Numbers we could not retrieve | Matsliah 2024 ED Fig. 2 F-scores; Nern 2025 held-out type-level NT accuracy (Fig 4e / Suppl Table 5); Eckstein hemilineage-split accuracy; Park-Marcotte C1→C3 drop. |

---

## 6b) Errata: deferred questions for dataset teams, and numbers not retrieved

Deferred on 2026-09-24. None of these block current work. Each one would firm up a provenance tag or a published comparison, so pick them up when there is time.

**Provenance questions for the dataset teams (drafts not written yet):**

| # | Question | Who to ask | Affects |
|---|---|---|---|
| E1 | The exact rule behind BANC `super_class` and `flow`, and whether it uses connectivity | BANC team (Bates, Wilson, Lee) | Whether BANC super_class/flow stay `curated_morphology` or become `provenance_uncertain` |
| E2 | Where the BANC NT ground truth lives (`gs://leelab_fly_cns`?), and its size and confidence scheme | BANC team / Funke lab | Literature-NT target coverage for BANC |
| E3 | Whether the larval `io_class` labels (Winding 2023 S2) depend on connectivity. Our l1em agent and the research disagree | Zlatic / Cardona labs (Winding, Pedigo) | l1em io_class provenance tag |
| E4 | Which FlyWire `cell_type` labels were split using connectivity (CBLAST) rather than morphology | Jefferis lab (Schlegel) | Separating curated from connectivity-defined FlyWire types |
| E5 | How male-CNS `consensusNt` is computed, and whether the unverified 88.4% agreement figure is correct | Janelia FlyEM (Berg, Nern) / Funke lab | Whether mc NT "ground truth" is partly model-predicted |
| E6 | Which types were in each NT CNN's training set (FlyWire/Eckstein, synister_banc, synister_malecns, MANC, optic lobe) | Funke lab | Removing CNN-training types from literature-NT evaluation |

**Published numbers we could not retrieve** (these need supplementary tables or the authors):

| # | Number | Source | Why it matters |
|---|---|---|---|
| E7 | Per-type F-scores (Extended Data Fig. 2) | Matsliah et al. 2024 | Direct comparison for optic-lobe typing accuracy |
| E8 | Held-out type-level NT accuracy (Fig. 4e / Suppl. Table 5) | Nern et al. 2025 | Reference point for optic-lobe NT |
| E9 | Accuracy on hemilineage-split data | Eckstein et al. 2024 | The image-model analogue of our random-vs-grouped gap |
| E10 | Park-Marcotte C1→C3 accuracy drop | Park & Marcotte 2012 | Reference size for leakage-driven drops |

Status: open. When one is resolved, record the answer, the source and the date here, and update the provenance tag in the target registry.

## 7) Reference list

Status key: **V** = verified on a fetched page. **M** = misattributed (corrected text used). **U** = unverifiable (context only). Notes give the check's corrections.

### Fly connectomes, annotation and typing
1. Schlegel P, Yin Y, Bates AS, et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634:139–152. doi:10.1038/s41586-024-07686-5. https://pmc.ncbi.nlm.nih.gov/articles/PMC11446831/ — **V**
   - The 56/13/32% figures are for hemibrain *morphology* types (2,920/664/1,651 of 5,235).
   - Hemilineage was assigned without connectivity (inferred from absence).
2. Dorkenwald S, et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634:124–138. doi:10.1038/s41586-024-07558-y. https://www.nature.com/articles/s41586-024-07558-y — **V** for the abstract (139,255 neurons; NT predictions). **U** for the postsynapse-attachment figures, which depend on version (44.7% vs 43.9%).
3. Matsliah A, Yu S-c, et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634:166. doi:10.1038/s41586-024-07981-1. https://pmc.ncbi.nlm.nih.gov/articles/PMC11446827/ — **V** (connectivity stage seeded by morphological types).
4. Nern A, et al. (2025). Connectome-driven neural inventory of a complete visual system. *Nature* 641:1225–1237. doi:10.1038/s41586-025-08746-0. https://pmc.ncbi.nlm.nih.gov/articles/PMC12119369/ — **V** (727 types in bioRxiv, 732 in Nature; >99% of types distinguishable by top-5 connections). **U** for the >98% FlyWire match-rate claim.
5. Scheffer LK, et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife* 9:e57443. https://elifesciences.org/articles/57443 — **V**
6. Winding M, Pedigo BD, et al. (2023). The connectome of an insect brain. *Science* 379:eadd9330. https://www.biorxiv.org/content/10.1101/2022.11.28.516756v1.full — **V** for about 90 types (bioRxiv v1), 93% of neurons with homologs, and edge symmetry. **U** for the "93 types" wording in the Science version. The L/R-pairing quote is a paraphrase.
7. Takemura S, et al. (2024). A connectome of the male *Drosophila* ventral nerve cord. *eLife* RP 97769. https://elifesciences.org/reviewed-preprints/97769 — **V** (3 NT classes plus a non-synaptic class).
8. Marin EC, et al. (2024). Systematic annotation of a complete adult male *Drosophila* nerve cord connectome. *eLife* RP 97766. https://elifesciences.org/reviewed-preprints/97766 — **V** ("substantial fraction" for 09B is a paraphrase).
9. Berg S, Beckett IR, Costa M, Schlegel P, et al. (2025). Sexual dimorphism in the complete connectome of the *Drosophila* male CNS. bioRxiv 10.1101/2025.10.09.680999 (Cell 2026, not checked). https://www.biorxiv.org/content/10.1101/2025.10.09.680999v1 — **V**
   - The revision covered about 4% of FlyWire *neurons*, not types.
   - The 88.4% consensusNt agreement is **U**.
10. Bates AS, Phelps JS, Kim M, Yang HH, et al. (2025/2026). Distributed control circuits across a brain-and-cord connectome (BANC). bioRxiv 10.1101/2025.07.31.667571; *Nature* doi:10.1038/s41586-026-10735-w. https://www.biorxiv.org/content/10.1101/2025.07.31.667571v2.full ; https://github.com/htem/BANC-project ; https://github.com/htem/synister_banc — **V**
11. Stürner T, Brooks P, et al. (2025). Comparative connectomics of *Drosophila* descending and ascending neurons. *Nature* 643:158–172. https://pmc.ncbi.nlm.nih.gov/articles/PMC12222017/ — **V** (the preprint quote on L/R vs cross-dataset differences was not re-fetched).
12. Matsliah A, Salmon CK, Bates AS, et al. (2026). Uncovering sex differences in the *Drosophila* VNC through connectome alignment. bioRxiv; PMC13307954. https://pmc.ncbi.nlm.nih.gov/articles/PMC13307954/ — **V** (the 62/7/26% split is over 300 disagreement cases).
13. Lee D, Matsliah A, Saul LK (2026). AC⊕DC Search: behind the winning solution to the FlyWire graph-matching challenge. *TMLR*. https://mlanthology.org/tmlr/2026/lee2026tmlr-acdc/ — **V**
14. Pedigo BD, Winding M, Priebe CE, Vogelstein JT (2023). Bisected graph matching improves automated pairing of bilaterally homologous neurons. *Netw Neurosci* 7(2):522. https://www.biorxiv.org/content/10.1101/2022.05.19.492713v3 — **V** (about 22% improvement on the larva, from the PDF; venue not confirmed on the page).
15. Pedigo BD, Powell M, et al. (2023). Generative network modeling reveals quantitative definitions of bilateral symmetry. *eLife* 12:e83739. https://elifesciences.org/articles/83739 — **V**
16. Truman JW, Price J, Miyares RL, Lee T (2023). Metamorphosis of memory circuits in *Drosophila*. *eLife* 12:e80594. https://elifesciences.org/articles/80594 — **V**
17. Scheffer LK (2025). Synapse detection efficiency in EM *Drosophila* connectomics. bioRxiv 10.1101/2025.10.16.682869. https://www.biorxiv.org/content/10.1101/2025.10.16.682869v1 — **V**
18. Dorkenwald S, et al. (2025). CAVE: Connectome Annotation Versioning Engine. *Nat Methods* 22:1112–1120. doi:10.1038/s41592-024-02426-z — **V**
19. Schwartzman G, Jourdan B, García-Soriano D, Matsliah A (2025). Connectivity is all you need: inferring neuronal types with NTAC. bioRxiv 10.1101/2025.06.11.659184. https://pmc.ncbi.nlm.nih.gov/articles/PMC12259112/ — **V**
20. Mehta K, Goldin RF, Marchette D, Vogelstein JT, Priebe CE, Ascoli GA (2021). Neuronal classification from network connectivity via adjacency spectral embedding. *Netw Neurosci* 5(3):689–710. https://pmc.ncbi.nlm.nih.gov/articles/PMC8567830/ — **V**
21. Mehta K, Goldin RF, Ascoli GA (2023). Circuit analysis of the *Drosophila* brain using connectivity-based neuronal classification. *Netw Neurosci* 7(1):269–298. https://pmc.ncbi.nlm.nih.gov/articles/PMC10275213/ — **M**. Corrected: it uses 19,902 **FlyCircuit v1.2** neurons (a light-microscopy "potential connectome"), not hemibrain, and gives 54 connectivity-based classes.
22. Witvliet D, et al. (2021). Connectomes across development reveal principles of brain maturation. *Nature* 596:257. https://www.biorxiv.org/content/10.1101/2020.04.30.066209v2.full — **V**

### Neurotransmitter identity
23. Eckstein N, Bates AS, et al. (2024). Neurotransmitter classification from EM images at synaptic sites in *Drosophila*. *Cell* 187:2574–2594. https://pmc.ncbi.nlm.nih.gov/articles/PMC11106717/ — **V**
    - Correction: cholinergic *and* glutamatergic neurons target higher layers.
    - The exact 70/10/20 split is described in terms of presynapses.
24. Lacin H, et al. (2019). Neurotransmitter identity is acquired in a lineage-restricted manner in the *Drosophila* CNS. *eLife* 8:e43701. https://elifesciences.org/articles/43701 — **V**
25. Allen AM, Neville MC, et al. (2020). A single-cell transcriptomic atlas of the adult *Drosophila* VNC. *eLife* 9:e54074. https://elifesciences.org/articles/54074 — **V**
26. Konstantinides N, et al. (2018). Phenotypic convergence: distinct transcription factors regulate common terminal features. *Cell* 174:622–635. https://www.biorxiv.org/content/early/2018/01/04/243113 — **V** (the preprint title differs slightly).
27. Harris MR, Wytock TP, Kovács IA (2022). Computational inference of synaptic polarities in neuronal networks. *Adv Sci* 9:2104906. https://pmc.ncbi.nlm.nih.gov/articles/PMC9165506/ — **V**
28. Fenyves BG, et al. (2020). Synaptic polarity and sign-balance prediction using gene expression data in the *C. elegans* connectome. *PLoS Comput Biol* 16:e1007974. https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007974 — **V** (73% applies to ionotropic synapses).
29. Deng B, Li Q, et al. (2019). Chemoconnectomics: mapping chemical transmission in *Drosophila*. *Neuron* 101:876–893. https://www.cell.com/neuron/fulltext/S0896-6273(19)30072-8 — **U** (the paper is real; the quote comes from snippets only).
30. flyconnectome / funkelab (2024). drosophila_neurotransmitters (NT ground truth, gt_data.csv, CC-BY-4.0). https://github.com/funkelab/drosophila_neurotransmitters (also flyconnectome/drosophila_neurotransmitters) — **V** (maps to 8 datasets; the "~30% of CNS" figure appeared in one check but not the other).
31. Funke lab (2025). synister_malecns. https://github.com/funkelab/synister_malecns — **V**
32. Schlegel P / flyconnectome. flywire_annotations. https://github.com/flyconnectome/flywire_annotations — **V** (known_nt is not described in the supplemental README).
33. Codex FAQ (FlyWire Connectome Data Explorer). https://codex.flywire.ai/faq — **V**

### Connectome-constrained models and connectome ML
34. Lappalainen JK, et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634:1132–1140. https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/ — **V**
35. Shiu PK, et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634:210–219. https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/ — **V**
36. Beiran M, Litwin-Kumar A (2025). Prediction of neural activity in connectome-constrained recurrent networks. *Nat Neurosci*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12648571/ — **V**
37. Cowley BR, et al. (2024). Mapping model units to visual neurons reveals population code for social behaviour. *Nature* 629:1100–1108. https://pillowlab.princeton.edu/pubs/abs_Cowley2024knockouttraining.html — **V** (the 23 LC types figure comes from the Nature article; the model is not connectome-constrained).
38. Liao M, Wan G, Du B (2024). NeuNet: Joint learning of neuronal skeleton and brain circuit topology for neuron classification. AAAI 2024. https://arxiv.org/html/2312.14518 — **M**. Corrected: 0.9363 is H01 (human). NeuNet on hemibrain is **0.9169**. The other numbers are correct.
39. Dhakane O (2025–2026). flywire-gnn. https://github.com/omkar-dhakane/flywire-gnn — **V** (not peer reviewed).
40. kostas52675 (2025). fly-connectome. https://github.com/kostas52675/fly-connectome — **V** (not peer reviewed; no results).
41. Lu J, Han K, Wang Y, Mi L, Yang C (2026). A benchmark analysis of graph and non-graph methods for *C. elegans* neuron classification. arXiv:2603.02241. https://arxiv.org/abs/2603.02241 — **V**
42. Jefferis GSXE et al. coconatfly. https://natverse.org/coconatfly/ — **V** (year not confirmed).
43. sjcabs (2025). fly_connectome_data_tutorial. https://github.com/sjcabs/fly_connectome_data_tutorial — **V**
44. Elabbady L, et al. (2025). Perisomatic ultrastructure efficiently classifies cells in mouse cortex. *Nature* 640:478–486. https://pmc.ncbi.nlm.nih.gov/articles/PMC11981918/ — **M** (on one point). Corrected: 82% was measured on an expert-checked **1,700-cell validation set** sampled from the 94,010-cell dataset, not on all 94,010 cells. The 91% CV on 1,619 cells is correct.
45. Schneider-Mizell CM, et al. (2025). Inhibitory specificity from a connectomic census of mouse visual cortex. *Nature* 640:448. https://www.nature.com/articles/s41586-024-07780-8 — **U**

### Graph ML, leakage and evaluation
46. Hamilton WL, Ying R, Leskovec J (2017). Inductive representation learning on large graphs. NeurIPS. https://arxiv.org/abs/1706.02216 — **V**
47. Shchur O, et al. (2018). Pitfalls of graph neural network evaluation. arXiv:1811.05868 — **V**
48. Huang Q, et al. (2020/2021). Combining label propagation and simple models out-performs GNNs. ICLR 2021. https://arxiv.org/abs/2010.13993 — **V** (the 137× figure is for OGB-Products).
49. Ivanov S, Prokhorenkova L (2021). Boost then convolve (BGNN). ICLR. https://arxiv.org/abs/2101.08543 — **V**
50. Grinsztajn L, Oyallon E, Varoquaux G (2022). Why do tree-based models still outperform deep learning on tabular data? https://arxiv.org/abs/2207.08815 — **V**
51. Platonov O, et al. (2023). A critical look at the evaluation of GNNs under heterophily. ICLR. https://arxiv.org/abs/2302.11640 — **V** (the "GNNs almost always win" result is mainly on the authors' new datasets).
52. Grover A, Leskovec J (2016). node2vec. KDD. https://arxiv.org/abs/1607.00653 — **V**
53. Zhu J, et al. (2020). Beyond homophily in GNNs (H2GCN). NeurIPS. https://arxiv.org/abs/2006.11468 — **V**
54. Shi Y, et al. (2021). Masked label prediction (UniMP). IJCAI. https://arxiv.org/abs/2009.03509 — **V**
55. Hu W, et al. (2020). Open Graph Benchmark. NeurIPS. https://arxiv.org/abs/2005.00687 — **V**
56. Errica F, et al. (2020). A fair comparison of GNNs for graph classification. ICLR. https://arxiv.org/abs/1912.09893 — **V**
57. Subramonian A, Kang J, Sun Y (2024). Origins of degree bias in GNNs. NeurIPS. https://arxiv.org/abs/2404.03139 — **V**
58. Kapoor S, Narayanan A (2022/2023). Leakage and the reproducibility crisis in ML-based science. *Patterns*. https://arxiv.org/abs/2207.07048 — **V**
59. Kapoor S, et al. (2023/2024). REFORMS: reporting standards for ML-based science. *Sci Adv*. https://arxiv.org/abs/2308.07832 — **V**
60. Roberts DR, Bahn V, Ciuti S, et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929. https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881 — **V** (bibliographic record). One quote ("ample opportunity for overfitting…") comes from snippets only.
61. Park Y, Marcotte EM (2012). Flaws in evaluation schemes for pair-input computational predictions. *Nat Methods* 9:1134. https://pmc.ncbi.nlm.nih.gov/articles/PMC3531800/ — **V**
62. Bernett J, Blumenthal DB, List M (2024). Cracking the black box of deep sequence-based PPI prediction. *Brief Bioinform* 25:bbae076. https://academic.oup.com/bib/article/25/2/bbae076/7621029 — **V**
63. Geirhos R, et al. (2020). Shortcut learning in deep neural networks. *Nat Mach Intell*. https://arxiv.org/abs/2004.07780 — **V**
64. DeGrave AJ, Janizek JD, Lee SI (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nat Mach Intell*. https://www.nature.com/articles/s42256-021-00338-7 — **V**
65. Ojala M, Garriga GC (2010). Permutation tests for studying classifier performance. *JMLR* 11:1833–1863. https://www.jmlr.org/papers/v11/ojala10a.html — **V**
66. Varoquaux G (2017/2018). Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage*. https://arxiv.org/abs/1706.07581 — **V**
67. Bouthillier X, et al. (2021). Accounting for variance in ML benchmarks. MLSys. https://arxiv.org/abs/2103.03098 — **V**
68. Whalen S, Schreiber J, Noble WS, Pollard KS (2022). Navigating the pitfalls of applying ML in genomics. *Nat Rev Genet*. https://www.nature.com/articles/s41576-021-00434-9 — **V**
69. Walsh I, et al. (2021). DOME: recommendations for supervised ML validation in biology. *Nat Methods* 18:1122. https://www.nature.com/articles/s41592-021-01205-4 — **U**

### Calibration and hierarchical classification
70. Guo C, Pleiss G, Sun Y, Weinberger KQ (2017). On calibration of modern neural networks. ICML. https://arxiv.org/abs/1706.04599 — **V**
71. Kull M, et al. (2019). Dirichlet calibration. NeurIPS. https://arxiv.org/abs/1910.12656 — **V**
72. Niculescu-Mizil A, Caruana R (2005). Predicting good probabilities with supervised learning. ICML. https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf — **V**
73. Zadrozny B, Elkan C (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD. http://www.cs.columbia.edu/~djhsu/coms4771-f25/handouts/zadrozny2002kdd.pdf — **V**
74. Nixon J, et al. (2019). Measuring calibration in deep learning. CVPRW. https://arxiv.org/abs/1904.01685 — **V**
75. Kumar A, Liang P, Ma T (2019). Verified uncertainty calibration. NeurIPS. https://arxiv.org/abs/1909.10155 — **V**
76. Roelofs R, et al. (2022). Mitigating bias in calibration error estimation. AISTATS. https://arxiv.org/abs/2012.08668 — **V**
77. Gupta C, Ramdas A (2022). Top-label calibration and multiclass-to-binary reductions. ICLR. https://arxiv.org/abs/2107.08353 — **V**
78. Le Coz A, Herbin S, Adjed F (2024). Confidence calibration of classifiers with many classes. NeurIPS. https://arxiv.org/abs/2411.02988 — **V**
79. Resin J, Yang, Gneiting T (2026). Hierarchies of calibration: classification meets regression. arXiv:2606.03245. https://www.alphaxiv.org/abs/2606.03245 — **V** (the quote is an expanded paraphrase).
80. Manokhin V, Grønhaug D (2026). Classifier calibration at scale. arXiv:2601.19944 — **V**
81. Ovadia Y, et al. (2019). Can you trust your model's uncertainty? NeurIPS. https://arxiv.org/abs/1906.02530 — **V** (the second quote actually reads "some methods that marginalize over models…").
82. Redmon J, Farhadi A (2017). YOLO9000 (WordTree). CVPR. https://arxiv.org/abs/1612.08242 — **V**
83. Bertinetto L, et al. (2020). Making better mistakes. CVPR. https://arxiv.org/abs/1912.09393 — **V**
84. Kosmopoulos A, et al. (2013). Evaluation measures for hierarchical classification. arXiv:1306.6802 — **V**
85. Silla CN Jr, Freitas AA (2011). A survey of hierarchical classification across different application domains. *Data Min Knowl Disc* 22:31–72. https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/DMKD-J-2010-Silla.pdf — **V** via the Kent PDF (one topic's check could not reach the Springer page).
86. Michielsen L, Reinders MJT, Mahfouz A (2021). Hierarchical progressive learning of cell identities (scHPL). *Nat Commun* 12:2799. https://pmc.ncbi.nlm.nih.gov/articles/PMC8121839/ — **V** (flat 0.98 F1 with 19.8% of cells rejected vs hierarchical 0.94 with almost none rejected; the numbers are consistent).
87. Ergen C, et al. (2024). Consensus prediction of cell type labels with popV. *Nat Genet*. https://www.biorxiv.org/content/10.1101/2023.08.18.553912v1 — **V**
88. Ding T, et al. (2023). Class-conditional conformal prediction with many classes. NeurIPS. https://arxiv.org/abs/2306.09335 — **V**
89. Mortier T, et al. (2025). Conformal prediction in hierarchical classification. arXiv:2501.19038 — **V**
90. den Hengst F, et al. (2025). Hierarchical conformal classification. arXiv:2508.13288 — **V**

### Mixture of experts vs global models
91. Montero-Manso P, Hyndman RJ (2021). Principles and algorithms for forecasting groups of time series: locality and globality. *Int J Forecast*. https://arxiv.org/abs/2008.00444 — **V**
92. Kratzert F, et al. (2019). Towards learning universal, regional, and local hydrological behaviors. *HESS* 23:5089. https://hess.copernicus.org/articles/23/5089/2019/ — **M**. Corrected: comparisons are on 447 basins. EA-LSTM scores 0.674 single / 0.705 ensemble, HBV (upper) 0.631 and mHM 0.627. A plain LSTM with static inputs scores 0.685 / 0.718.
93. Tuck J, Barratt S, Boyd S (2019/2021). A distributed method for fitting Laplacian regularized stratified models. *JMLR*. https://arxiv.org/abs/1904.12017 — **V**
94. Chen Z, Deng Y, Wu Y, Gu Q, Li Y (2022). Towards understanding mixture of experts in deep learning. arXiv:2208.02813 — **V** (includes experiments as well as theory).
95. Zoph B, et al. (2022). ST-MoE. arXiv:2202.08906 — **V** (increasing expert dropout gave modest gains).
96. Shazeer N, et al. (2017). Outrageously large neural networks: the sparsely-gated MoE layer. ICLR. https://arxiv.org/abs/1701.06538 — **V**
97. Jacobs RA, Jordan MI, Nowlan SJ, Hinton GE (1991). Adaptive mixtures of local experts. *Neural Comput* 3:79–87. https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf — **U** (scanned PDF; metadata only).
98. Chernov A (2025). (GG) MoE vs. MLP on tabular data. arXiv:2502.03608 — **V**
99. Gorishniy Y, Kotelnikov A, Babenko A (2024/2025). TabM. ICLR 2025. https://arxiv.org/abs/2410.24210 — **V**
100. Wang H, et al. (2023). Graph mixture of experts (GMoE). NeurIPS. https://arxiv.org/abs/2304.02806 — **V**
101. Zeng H, et al. (2024). Mixture of weak & strong experts on graphs (Mowst). ICLR. https://arxiv.org/abs/2311.05185 — **V**
102. Abdelaal T, et al. (2019). A comparison of automatic cell identification methods for scRNA-seq data. *Genome Biol* 20:194. https://www.biorxiv.org/content/10.1101/644435v1 — **V**
103. Domínguez Conde C, et al. (2022). Cross-tissue immune cell analysis (CellTypist). *Science* 376:eabl5197. https://github.com/Teichlab/celltypist — **U** (repo confirmed; the ~0.9 F1 figure and the "no tissue experts" claim were not seen in the paper).

---
Structured data: `research.json` (same directory). Per-topic notes: `*.md` in this directory.
