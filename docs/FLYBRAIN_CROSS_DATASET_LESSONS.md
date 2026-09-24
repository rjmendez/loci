> Archived / retired from active navigation
>
> This page is kept only for historical continuity and searchability. It is no longer part of the default FlyBrain reading path.
>
> Canonical entry point: [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md)
> Archive home: [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md)
> For current guidance, use the guide and the evidence/provenance docs before digging into historical detail.
>
# FlyBrain cross-dataset lessons

This page is the plain-English version of the cross-dataset guidance.

The main idea: different FlyBrain datasets are not interchangeable. A result is only meaningful when we also know which dataset, version, sex, stage, anatomy, and query settings produced it.

## The short version

If a claim comes from one dataset, say so. Do not speak as though it is a general rule for all FlyBrain data.

Example:
- Good: "In adult female FlyWire, this pathway was observed under the default connectivity query."
- Risky: "This pathway is how the fly brain works."

## What the repo is teaching

### 1) Dataset choice changes the answer

The FlyBrain tool surface includes different datasets with different coverage and scope.

Examples:
- `fw` / `flywire783` is an adult female FlyWire dataset.
- `mc` / `male_cns_v1_0` is a male whole-CNS dataset.
- `l1em` is larval (L1) scope.
- `ol` is optic-lobe specific.

A result in one of these is not a direct fact about the others unless the evidence is explicitly repeated.

### 2) Sex and stage matter

A statement from a female adult dataset is not automatically valid for males or larvae.

Example:
- "The pathway is present in adult female FlyWire" is a scoped statement.
- "The pathway is present in Drosophila" is broader and needs direct evidence across the claimed scopes.

### 3) Anatomy matters

A whole-brain dataset is not the same as an optic-lobe or VNC dataset.

Example:
- An optic-lobe result is not a whole-brain claim.
- A male VNC result is not a female brain claim.

### 4) Query mode matters

The same class can produce different outputs depending on how the query is run.

Examples:
- `group_by_class=true` changes how rows are aggregated.
- `exclude_dbs` can change the dataset set being compared.
- `weight` and paging can change what is considered a strong result.

So the output is not just a fact; it is a fact under a specific access path.

### 5) Evidence quality matters

Some FlyBrain evidence is structural (connectivity), while other evidence is curated or predicted.

Examples:
- `get_known_neurotransmitters` is curated and does not carry confidence values.
- `get_predicted_neurotransmitters` carries confidence and should not be mixed into a curated claim without saying so.

This matters because different evidence families are not interchangeable.

## The Loci rule

For every FlyBrain-derived finding, store the scope explicitly:

- dataset symbol
- dataset version / identifier
- sex
- life stage
- anatomy scope
- query settings that affect interpretation
- evidence family

A practical pattern is:

"In adult female FlyWire (`fw`, `flywire783`), this result appeared under a class-level connectivity query with `exclude_dbs=[hb, fafb]`."

That is a valid claim. It is not a general claim about all FlyBrain data.

## What Loci should do

- Treat provenance as part of the finding, not extra notes.
- Rewrite broad claims into scoped wording when evidence is partial.
- Keep dataset-local findings dataset-local.
- Require explicit evidence before claiming a cross-dataset, cross-sex, or cross-stage generalization.

## Common mistakes to avoid

- Saying one dataset proves a universal rule.
- Mixing adult and larval evidence without saying so.
- Mixing male and female evidence without saying so.
- Comparing grouped and ungrouped connectivity results as if they were the same measurement.
- Treating one dataset as a replacement for another without direct validation.

## Best practice for answers

Use this pattern:

- dataset-local: "In this dataset/version/query path..."
- dataset-version-specific: "In FlyWire v783..."
- cross-dataset matched-scope: "Observed in both adult female FlyWire and BANC under compatible conditions..."
- broad/generalized: only after direct evidence across the stated scopes

If the scope is unclear, prefer a narrower statement over a broader one.

---

Status: a plain-language summary of the cross-dataset guidance for FlyBrain and Loci.
