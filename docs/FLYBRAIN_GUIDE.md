# FlyBrain Guide for Loci

This is the short, user-facing version of the FlyBrain material in this repo.

If you are reading this because you want to understand why the FlyBrain work matters for Loci, the short answer is simple: the project is trying to teach Loci how to reason responsibly when evidence comes from multiple datasets, stages, and interpretations.

## The core message

The most important idea is not a biological fact. It is a method rule:

- Different FlyBrain datasets are not interchangeable.
- A claim is only as good as its dataset, version, sex, stage, anatomy, and query path.
- Loci should store provenance with each claim, not just the conclusion.

That rule is the reason the dense FlyBrain notes exist.

## What each dense document is trying to say

| Document | Core user-facing message | Best for |
|---|---|---|
| `FLYBRAIN_CROSS_DATASET_LESSONS.md` | Cross-dataset comparisons are useful, but every claim must be scoped to the dataset and context it was actually observed in. | Understanding why sex, stage, and anatomy matter. |
| `FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` | Every FlyBrain-derived claim should carry provenance: dataset, version, query settings, and result contract. | Provenance and replay requirements. |
| `FLYBRAIN_DATASET_PROVENANCE_MATRIX.md` | Here is the dataset map: which dataset is female/adult, which is male, which is larval, which is optic-lobe or whole-CNS specific. | Checking what a dataset can and cannot support. |
| `FLYBRAIN_IO_TO_LOCI_MAPPING.md` | This is the architecture translation: how fly-brain input, routing, and motor output map into Loci memory and reasoning patterns. | Design and implementation reference. |

## Read in this order

1. Start here: this page.
2. If you need the evidence and scope rules, read `FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`.
3. If you need the dataset boundaries, read `FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`.
4. If you need the design mapping to Loci behavior, read `FLYBRAIN_IO_TO_LOCI_MAPPING.md`.
5. If you want the comparative principles, read `FLYBRAIN_CROSS_DATASET_LESSONS.md`.

## Why it matters for Loci

The FlyBrain material is proving ground for a few Loci habits we want to keep:

- evidence-bounded claims;
- explicit dataset/version scope;
- typed memory with provenance;
- routing that responds to context and confidence rather than raw text alone.

The most important Loci takeaway is: a finding is only trustworthy when its evidence path is still traceable.

## Archive and appendix material

The deeper FlyBrain notes are intentionally dense and are kept as appendix/archive material for historical and technical reference. They are not the starting point for normal reading.

See [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) for the archive index and pointers to the retained deep notes.
