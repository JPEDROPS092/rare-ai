# RareAI

**Explainable AI for Rare Disease Variant Prioritization** — MVA Hackathon 2026 (Track 1)

> Research platform for hypothesis generation and prioritization. **Not a diagnostic
> medical device.** Every candidate is traceable: variant → feature → score → rank →
> evidence → explanation.

## Architecture

```
Genomic data (VCF, GRCh38)
        ↓  streaming parse + QC (cyvcf2 → Parquet)
Candidate funnel (cheap deterministic filters first)
        ↓  phenotype-matched gene panel (HPO semantic similarity)
Annotation (ClinVar local; Ensembl REST VEP for candidates only)
        ↓  features: rarity / pathogenicity / phenotype / clinical / inheritance / mosaicism
Transparent weighted ranking (configs/ranking.yaml)
        ↓  Top-N
Explainable report + ranked submission CSV
```

An LLM is never used to *produce* rankings or evidence. The scientific answer is
deterministic and auditable; language-model layers (Phase 8+) only summarize
retrieved evidence with mandatory source citations.

## Quickstart

```bash
cp .env.example .env          # set HF_TOKEN (dataset access must be granted first)
make setup                    # uv sync (Python 3.12)
make inspect                  # dataset discovery → data_catalog.json + DATASET.md
make download                 # Track 1 only: VCF + index + phenotype DOCX (~318 MB)
make resources                # HPO, ClinVar, Ensembl GTF (~150 MB)
make pipeline                 # end-to-end run → results/ + submission CSV
```

### Full dataset download (~85 GB, optional)

The FASTQs are only needed for re-calling (mosaicism-aware analysis). Set
`RAREAI_DATA_DIR` in `.env` to a disk with >= 110 GB free, then:

```bash
./scripts/download_full.sh                 # foreground
nohup ./scripts/download_full.sh &         # or background

make watch                                 # monitor: per-file + overall progress
uv run rareai status --watch 30            # same, explicit refresh interval
tail -f logs/download_full.log             # raw retry/backoff log
```

The script is fully re-runnable: completed files are skipped and partial
transfers resume automatically (12 retries with growing backoff per file).

Curated patient phenotype (HPO) is required before the first run: after `make
resources`, `rareai run` generates `data/phenotypes/patient_suggestions.yaml`
(label/synonym matches against the clinical text); approve terms into
`data/phenotypes/patient_phenotype.yaml` (kept out of git — private clinical data).

## Repository layout

```
configs/          dataset / ranking / annotation YAML (no hard-coded magic numbers)
rareai/
  data/           catalog, selective download, resources with sha256 manifests
  variants/       parser, normalization, QC, genotypes, compound-het, mosaicism
  phenotype/      HPO ontology, Resnik/Lin similarity, patient encoding
  annotation/     Ensembl GTF gene models, ClinVar join, REST VEP (cached)
  features/       evidence matrix with provenance + coverage accounting
  ranking/        transparent weighted baseline (ML ensemble: Phase 6)
  evaluation/     Top-K, MRR, MAP, F-max, challenge rank points (documented approx.)
  reporting/      explainable reports + official submission builder
  pipeline/       orchestration, funnel stats, timing, experiment log
data/             gitignored: raw / staging / annotations / features (private)
results/          gitignored: rankings, reports, submission, benchmarks
docs/             architecture, methodology, reproducibility, limitations
tests/            unit + integration (synthetic VCF, no network needed)
```

## Privacy

The dataset is a real child's genome, shared under an IRB-approved protocol.
`data/` and `results/` are gitignored; no tokens or genomic data are ever committed.
See the dataset access rules on the [dataset page](https://huggingface.co/datasets/SageBio/mva-hackathon-2026-data).

## Reproducibility

- Pinned dependency lock (`uv.lock`), Python 3.12, Docker image
- Resource manifests with sha256 (`data/annotations/resources_manifest.json`)
- Experiment log with config fingerprint (`results/benchmarks/experiments.jsonl`)
- Deterministic ranking: same inputs + configs → same outputs (tie-breakers fixed)

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Dataset discovery | done |
| 1 | VCF processing | done |
| 2 | Annotation (ClinVar + VEP) | done |
| 3 | Phenotype/HPO | done |
| 4 | Baseline ranking + submission | done |
| 5 | Evaluation harness (proxy) | done |
| 6 | ML ranking (LightGBM) | planned |
| 7 | Ablation study | planned |
| 8+ | Evidence retrieval, RAG, agents, Track 2 | planned |

## License

MIT for code. Challenge outputs are CC BY 4.0 per the hackathon rules.
