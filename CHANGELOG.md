# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- GitHub issue and pull request templates.
- Continuous integration workflow (ruff lint + pytest on Python 3.12).

### Changed
- Stopped tracking runtime logs; `logs/` is now gitignored.

## [0.1.0] - 2026-09-05

### Added
- Dataset discovery and catalog generation (`data_catalog.json`, `DATASET.md`).
- Streaming VCF processing: parse, normalization, QC, genotypes, compound-het,
  mosaicism detection.
- Phenotype/HPO layer: ontology, Resnik/Lin semantic similarity, patient encoding.
- Annotation: Ensembl GTF gene models, ClinVar join, cached Ensembl REST VEP.
- Feature matrix with provenance and coverage accounting.
- Transparent weighted baseline ranking driven by `configs/ranking.yaml`.
- Evaluation harness: Top-K, MRR, MAP, F-max, challenge rank points.
- Explainable reports and official submission builder.
- Pipeline orchestration with funnel stats, timing, and experiment log.

[Unreleased]: https://github.com/JPEDROPS092/rare-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JPEDROPS092/rare-ai/releases/tag/v0.1.0
