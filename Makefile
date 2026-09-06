.PHONY: setup test lint fmt inspect download download-full watch resources pipeline submission report clean

setup: ## Create the virtualenv and install dependencies
	uv sync

test: ## Run the test suite
	uv run pytest

lint: ## Static checks (ruff)
	uv run ruff check rareai tests

fmt: ## Format code
	uv run ruff format rareai tests

inspect: ## Dataset discovery (catalog + DATASET.md)
	uv run rareai inspect

download: ## Download Track 1 files (~318 MB; requires HF_TOKEN in .env)
	uv run rareai download --track track1

download-full: ## Download the FULL dataset (~85 GB; uses RAREAI_DATA_DIR disk)
	./scripts/download_full.sh

watch: ## Live download/pipeline progress (refresh every 30 s)
	uv run rareai status --watch 30

resources: ## Download annotation resources (HPO, ClinVar, Ensembl)
	uv run rareai resources

pipeline: ## Run the full Track 1 pipeline
	uv run rareai run --track track1

submission: ## Show submission artifacts
	@ls -la results/submission/ 2>/dev/null || echo "no submission yet - run make pipeline"

report: ## Print report locations
	uv run rareai report

clean: ## Remove caches and derived data (keeps data/raw)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf data/staging data/normalized data/features
	rm -rf results
