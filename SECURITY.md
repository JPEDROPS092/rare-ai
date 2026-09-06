# Security & Data Privacy Policy

RareAI processes real genomic and clinical data shared under an IRB-approved
protocol. Data privacy is treated as a security concern.

## Handling of sensitive data

- **Never commit** genomic data, phenotype files, tokens, or credentials.
  `data/`, `results/`, `.env`, and `*.token` are gitignored by default.
- Curated patient phenotype (`data/phenotypes/patient_phenotype.yaml`) is private
  clinical data and must stay out of version control.
- Do not attach real patient data to issues, pull requests, or discussions.
- Dataset access is governed by the
  [dataset page](https://huggingface.co/datasets/SageBio/mva-hackathon-2026-data);
  respect its access rules.

## Reporting a vulnerability

If you discover a security vulnerability or an accidental exposure of sensitive
data, please **do not** open a public issue. Instead, report it privately via
[GitHub Security Advisories](https://github.com/JPEDROPS092/rare-ai/security/advisories/new)
or by contacting the maintainers directly.

We aim to acknowledge reports within 72 hours and to provide a resolution
timeline after triage.

## Supported versions

This is a hackathon research project; security fixes are applied to the `main`
branch only.
