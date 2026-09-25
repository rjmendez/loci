# graph-ml-leakage: research notes (2026-09-24)

Topic: leakage and evaluation pitfalls in graph/network ML and biology ML, and the gates that rigorous studies use. Maps to F4, F5 and F8, with side links to F1, F3, F6, F7 and F9.

## Sources
Each source was fetched, at least to its abstract page, unless marked partial.

1. Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", arXiv:2207.07048 (2022; Patterns 2023). https://arxiv.org/abs/2207.07048
   - Finds leakage in "17 fields and 329 papers".
   - Gives an 8-type taxonomy:
     - L1.1: no test set.
     - L1.2: pre-processing on train+test.
     - L1.3: feature selection on train+test.
     - L1.4: duplicates.
     - L2: illegitimate features ("a feature is a proxy for the outcome variable").
     - L3.1: temporal leakage.
     - L3.2: non-independence between train and test ("unless the scientific claim is about a distribution that has the same dependence structure").
     - L3.3: sampling bias in the test distribution.
   - In the civil-war case, once leakage was fixed, "complex ML models don't perform substantively better than decades-old LR models".
   - CONFIRMS F4/F8: the width lookup and body ids are L2 proxies, and connectivity-defined labels are L2-style circularity. CONFIRMS F5: LR is a strong baseline. METHOD: model info sheets.
2. Kapoor et al., "REFORMS: Reporting Standards for ML Based Science", arXiv:2308.07832 (2023; Sci Adv 2024). A checklist of "32 questions", from a consensus of "19 researchers". METHOD: an external checklist to compare our gates against.
3. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", Ecography 40:913-929 (2017), doi:10.1111/ecog.02881.
   - Ignoring the structure causes "serious underestimation of predictive error".
   - Structured data gives "ample opportunity for overfitting with non-causal predictors".
   - Recommended fix: "Block cross-validation, where data are split strategically rather than randomly".
   - The abstract text came from a search snippet (publisher page returned 403), so this entry is PARTIAL. CONFIRMS F5's grouped splits.
4. Park & Marcotte, "Flaws in evaluation schemes for pair-input computational predictions", Nat Methods 9:1134 (2012), doi:10.1038/nmeth.2259, PMC3531800.
   - Defines C1/C2/C3 test classes: the test pair shares both, one or neither component with training.
   - "pair-input methods tend to perform much better for test pairs that share components with a training set".
   - METHOD: graded split hierarchy.
5. Bernett, Blumenthal & List, "Cracking the black box of deep sequence-based PPI prediction", Brief Bioinform 25(2):bbae076 (2024), doi:10.1093/bib/bbae076.
   - "random splitting lead[s] to strongly overestimated performances".
   - Without leakage, "performances become random".
   - The models "learn solely from sequence similarities and node degrees".
   - CONFIRMS F4 (degree proxy) and F5 (the random-vs-grouped gap).
6. Shchur et al., "Pitfalls of Graph Neural Network Evaluation", arXiv:1811.05868 (2018). "different splits of the data leads to dramatically different rankings of models". METHOD for F9.
7. Errica et al., "A Fair Comparison of GNNs for Graph Classification", ICLR 2020, arXiv:1912.09893. Over 47,000 experiments; "by comparing GNNs with structure-agnostic baselines ... structural information has not been exploited yet". METHOD for F9.
8. Huang et al., "Combining Label Propagation and Simple Models Out-performs GNNs", arXiv:2010.13993 (ICLR 2021). A shallow model plus label propagation matches GNNs with "137 times fewer parameters". Use it as a baseline for F9, with train-only labels.
9. Shi et al., "Masked Label Prediction (UniMP)", arXiv:2009.03509 (IJCAI 2021). Masks input labels "to train the network without overfitting in self-loop input label information". CONFIRMS the F8 masking practice.
10. Zhu et al., "Beyond Homophily in GNNs (H2GCN)", arXiv:2006.11468 (NeurIPS 2020). GNNs "fail to generalize" under heterophily; higher-order neighbourhoods help. EXTENDS F7 and F9.
11. Platonov et al., "A critical look at the evaluation of GNNs under heterophily", arXiv:2302.11640 (ICLR 2023). "duplicate nodes ... leads to train-test data leakage". CONFIRMS F8: group near-duplicate neurons such as L/R homologs.
12. Hu et al., "Open Graph Benchmark", arXiv:2005.00687 (NeurIPS 2020). Calls for "meaningful application-specific data splits" and highlights "out-of-distribution generalization under realistic data splits". CONFIRMS grouped and cross-animal splits.
13. Subramonian, Kang & Sun, "Origins of Degree Bias in GNNs", arXiv:2404.03139 (NeurIPS 2024). "high-degree test nodes tend to have a lower probability of misclassification regardless of how GNNs are trained". EXTENDS F4.
14. Geirhos et al., "Shortcut Learning in Deep Neural Networks", arXiv:2004.07780 (Nat Mach Intell 2020). Shortcuts are "decision rules that perform well on standard benchmarks but fail to transfer". Frames F4 and F5.
15. DeGrave, Janizek & Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal", Nat Mach Intell (2021), doi:10.1038/s42256-021-00338-7 (abstract via the Semantic Scholar API). The models rely on confounders, and "evaluation of a model on external data is insufficient". EXTENDS F9.
16. Ojala & Garriga, "Permutation Tests for Studying Classifier Performance", JMLR 11:1833-1863 (2010). A label-permutation test plus a restricted within-class feature-permutation test. METHOD.
17. Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars", arXiv:1706.07581 (NeuroImage 2018). Error bars are "±10% for 100 samples", and "the standard error across folds strongly underestimates" them. METHOD: bootstrap over groups.
18. Bouthillier et al., "Accounting for Variance in ML Benchmarks", arXiv:2103.03098 (MLSys 2021). Data sampling, initialisation and hyperparameters "impact markedly the results". METHOD for F9.
19. Guo et al., "On Calibration of Modern Neural Networks", arXiv:1706.04599 (ICML 2017). Temperature scaling. METHOD for F6.
20. Whalen et al., "Navigating the pitfalls of applying ML in genomics", Nat Rev Genet (2022), doi:10.1038/s41576-021-00434-9 (abstract via Europe PMC). "the structure of genomics data can bias performance evaluations".
21. Walsh et al., "DOME", Nat Methods 18:1122 (2021), doi:10.1038/s41592-021-01205-4. Only the TLDR was read; PARTIAL.

## Synthesis
Our gates match what the literature recommends: trivial baselines, a shuffle control, the random-vs-grouped gap, grouped splits, masking of neighbour labels, and ECE. Nothing we found contradicts our findings.

Gaps to adopt:

a. Write down a per-target audit of feature legitimacy, as in L2 and REFORMS. It should cover size and degree proxies.

b. Add degree-only and size-only baselines, and report accuracy by degree bin.

c. Run the bootstrap and the permutation tests at group level, or permute within groups, rather than per neuron.

d. Report a graded split curve rather than one number: random, then by type, then by hemilineage, then cross-hemisphere, then cross-animal (the Park-Marcotte C1-C3 idea).

e. For F9, run many seeds and splits, give both models equal tuning, and add a Correct & Smooth baseline that uses train labels only.

f. Group near-duplicate neurons together in the splits.

g. Treat the F3 "passes" on classifier-predicted NT labels as distillation. The gate should require ground-truth labels.

h. Passing the cross-animal test is necessary but not sufficient, so add an attribution check (DeGrave).
