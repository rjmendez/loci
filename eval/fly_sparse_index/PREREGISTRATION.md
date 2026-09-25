# Fly sparse index: pre-registration

Written and committed before any evaluation run. The commit that adds this
file precedes the commit that adds results; `git log` is the timestamp.

## Question

Design-study case 1 (fly -> Loci). The mechanism is a mushroom-body-style
sparse index (`mcp/sparse_expansion_index.py`): input projection to G
"glomeruli", a sparse binary expansion to K "Kenyon cells", k-winners-take-all,
then Hamming search over bit-packed codes.

1. Do FlyWire-measured PN -> KC wiring statistics beat idealised random
   FlyHash when sparsity and memory are matched?
2. Does any fly variant beat LSH (SimHash) or ANN (HNSW) on any axis:
   recall, memory, latency, build time or novelty detection?

## Data

- **loci**: the Loci repository's own tracked `*.md` and `*.py` files at the
  worktree HEAD, chunked into pieces of at most 1,200 characters (chunks under
  200 characters dropped). Queries are 300 chunks drawn at random with split
  seed 12345 and held out of the index. This is the Loci-relevant corpus.
- **nfcorpus**: BEIR NFCorpus, all 3,633 documents and the 323 test queries.
- All vectors come from nomic-embed-text (768-d) via the local Ollama
  `/api/embed`, with no task prefixes (the way `mcp/embed_ops.py` calls it).

## Ground truth

For each query, the ground truth is the exact cosine top-10 over the index
(brute force).

## Methods at the operating point

G is the number of glomeruli with at least one KC connection in the primary
measured matrix (`fw_right`, FlyWire 783 right-hemisphere KCs, 5-synapse
threshold). Every method below uses the same G, K and budget:

- Expansion: K = 20 * G bits.
- Winner-take-all fraction: 5% (Dasgupta 2017; the "about 10%" APL figure is
  unverified).
- Projection: PCA fitted on the index vectors, not whitened, with components
  assigned to glomeruli in a seeded random order.
- Memory: every code method stores K bits per item. The one exception is
  noted under HNSW.

| id | description |
|---|---|
| `exact` | brute-force cosine (ground truth; reported for latency and memory) |
| `simhash` | sign of K Gaussian random projections of the centred vectors (LSH), K bits |
| `flyhash_random` | idealised FlyHash: uniform glomeruli, every KC has c claws, where c = round(mean claws of measured KCs that have at least 1 claw) |
| `flyhash_random_fullD` | idealised FlyHash on the raw 768 centred dimensions (identity projection, G = 768), same c and K |
| `fly_measured_matrix` | **primary fly variant**: KCs drawn from the measured `fw_right` binary matrix (`mode="matrix"`) |
| `fly_measured_sample` | claw counts drawn from the measured distribution; glomeruli drawn with the measured claw-share weights |
| `hnsw` | hnswlib, M=16, ef_construction=200, ef=64, on the normalised float32 vectors. **Not memory-matched**, so it is reported but excluded from the decision |

These controls are secondary. They separate the measured claw distribution
from the measured glomerulus weights:

- `ctrl_measured_claws_uniform_glom`
- `ctrl_fixed_claws_measured_glom`

Replicates (secondary):

- `fly_measured_matrix` on `fw_left`
- `fly_measured_matrix` on `hb` (hemibrain)

## Seeds and CIs

- Five seeds (0 to 4) at the operating point. Each seed changes the
  connectivity, the projection permutation and the SimHash/random planes. The
  query split is fixed.
- Per-query metrics are averaged over seeds.
- CIs are 95% percentile bootstrap over queries, with 2,000 resamples of the
  per-query differences.

## Pre-registered primary metric and decision rule

The **primary metric** is recall@10 against exact cosine for pure code search
(no re-ranking) on the **loci** corpus at the operating point.

- **D1: measured vs idealised.** Adopt measured fly statistics over
  idealised random FlyHash only if the mean per-query difference
  recall@10(`fly_measured_matrix`) - recall@10(`flyhash_random`) > 0 **and**
  its 95% CI excludes 0.
- **D2: adopt for Loci.** Adopt the fly index (for any later wiring
  proposal) only if recall@10(`fly_measured_matrix`) beats the **best
  memory-matched baseline** (the highest mean recall@10 among `simhash`,
  `flyhash_random` and `flyhash_random_fullD`) with the 95% CI of the
  per-query difference excluding 0. Otherwise: do not adopt.

A negative result is reported as such.

## Secondary (exploratory; no adoption on these alone)

These use the same CI construction and are flagged as multiple comparisons.

- The same comparisons on nfcorpus.
- recall@10 after exact re-ranking of the top-100 code candidates.
- MRR of the exact nearest neighbour within the method's top-100.
- Query latency (ms/query, single process), index memory (bytes) and build
  time (s).
- **Novelty AUROC**, leave-one-topic-out over the combined loci + nfcorpus
  pool. Topics are coarse repository areas (`mcp-core`, `mcp-tests`,
  `mcp-flybrain`, `scripts`, `docs`, `mlops`) plus `nfcorpus`; only topics
  with at least 150 items are used.
  - For each fold, 80% of the other topics' items are stored.
  - Negatives are up to 500 of the remaining 20%.
  - Positives are up to 500 items of the held-out topic.
  - Scores: the fly novelty (min normalised Hamming to stored codes), SimHash
    min Hamming, exact 1 - max cosine, and HNSW 1 - top-1 cosine.
  - Per-fold AUROC, then the mean over folds. The CI comes from a bootstrap
    of items within folds.
- Sweeps are one factor at a time around the operating point, 3 seeds, loci
  corpus:
  - expansion K/G in {2, 5, 10, 20, 50};
  - claws per KC in {2, 4, 6, 7, 10, 15}, applied to idealised and to
    measured-weight sampling;
  - WTA fraction in {0.01, 0.02, 0.05, 0.10, 0.20}.

## Fixed choices, made before running and not tuned on these corpora

- Whitening is off. On a synthetic clustered set in the unit tests, whitening
  lowered cluster purity. That was decided on synthetic data only.
- Tie-breaking is by the lower index. Hamming is the search metric; Jaccard
  gives the same ranking for fixed-weight codes.
