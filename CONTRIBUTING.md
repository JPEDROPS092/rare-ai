# Contributing to RareAI

Thanks for your interest in improving RareAI! This project is an explainable
variant-prioritization platform built for the MVA Hackathon 2026. Contributions
of all kinds are welcome — bug reports, documentation, tests, and code.

> **Reminder:** RareAI is a research platform for hypothesis generation, **not a
> diagnostic medical device**. Contributions must preserve the auditability of
> every result (variant → feature → score → rank → evidence → explanation).

## Getting started

```bash
git clone https://github.com/JPEDROPS092/rare-ai.git
cd rare-ai
cp .env.example .env          # set HF_TOKEN if you need dataset access
make setup                    # uv sync (Python 3.12)
make test                     # run the unit + integration suite
```

The test suite uses synthetic VCFs and needs no network access or private data.

## Branching model (git flow)

We follow [git flow](https://nvie.com/posts/a-successful-git-branching-model/):

- `main` — stable, released code.
- `develop` — integration branch for the next release.
- `feature/*` — new work, branched from and merged back into `develop`.
- `release/*` — release stabilization, branched from `develop`.
- `hotfix/*` — urgent fixes, branched from `main`.

Open pull requests against `develop` (or `main` for hotfixes).

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <description>

<optional body>
```

Common types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`.

## Code style

- Python 3.12, formatted and linted with [ruff](https://docs.astral.sh/ruff/).
- Type hints are expected; `mypy` runs in CI.
- Keep configuration in `configs/*.yaml` — no hard-coded magic numbers.

Before pushing:

```bash
make lint      # ruff check
make test      # pytest
```

## Reporting bugs & requesting features

Please use the issue templates under `.github/ISSUE_TEMPLATE/`. For anything
touching genomic or clinical data handling, see [SECURITY.md](SECURITY.md) first
and **never** attach real patient data to an issue.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
