"""Benchmark harness: real evaluation (truth file) and proxy sanity checks."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from rareai.evaluation.metrics import Metrics, TruthSet, evaluate_ranking, variant_key
from rareai.log import get_logger

logger = get_logger(__name__)


def load_truth_csv(path: Path | str) -> list[TruthSet]:
    """Load a truth/submission-style CSV into TruthSets (GRCh38, chr-prefixed)."""
    df = pl.read_csv(path, infer_schema_length=10000)
    required = {"chrom_1", "pos_1", "ref_1", "alt_1"}
    if not required.issubset(df.columns):
        raise ValueError(f"truth file missing columns: {required - set(df.columns)}")
    truths: list[TruthSet] = []
    for row in df.iter_rows(named=True):
        primary = variant_key(
            row.get("chrom_1"), row.get("pos_1"), row.get("ref_1"), row.get("alt_1")
        )
        partner = variant_key(
            row.get("chrom_2") or None,
            row.get("pos_2") or None,
            row.get("ref_2") or None,
            row.get("alt_2") or None,
        )
        if primary:
            truths.append(TruthSet(primary=primary, partner=partner))
    return truths


def proxy_truth_from_clinvar(ranked: pl.DataFrame) -> list[TruthSet]:
    """Proxy 'truth': ClinVar pathogenic/likely-pathogenic hits among candidates.

    This is a self-consistency sanity check (does the pipeline surface known
    pathogenic variants high?), NOT performance against the clinical answer.
    """
    hits = ranked.filter(pl.col("clinvar_category").is_in(["pathogenic", "likely_pathogenic"]))
    truths = []
    for row in hits.iter_rows(named=True):
        key = variant_key(row["chrom"], row["pos"], row["ref"], row["alt"])
        if key:
            truths.append(TruthSet(primary=key))
    return truths


def run_benchmark(
    ranked: pl.DataFrame,
    truth_csv: Path | str | None = None,
    proxy: bool = False,
    out_json: Path | str | None = None,
) -> tuple[Metrics, list[TruthSet]]:
    truths: list[TruthSet] = []
    mode = "none"
    if truth_csv:
        truths = load_truth_csv(truth_csv)
        mode = "ground_truth"
    elif proxy:
        truths = proxy_truth_from_clinvar(ranked)
        mode = "clinvar_proxy"

    if not truths:
        logger.warning("No truth available - metrics cannot be computed (reported as null).")
        metrics = Metrics()
        metrics.topk = {}
        metrics.best_ranks = []
    else:
        metrics = evaluate_ranking(ranked, truths)

    if out_json:
        payload = {
            "mode": mode,
            "n_truth": len(truths),
            "metrics": metrics.to_dict(),
            "note": (
                "clinvar_proxy is an internal sanity check, not ground-truth performance."
                if mode == "clinvar_proxy"
                else "Evaluate against the live leaderboard for official scoring."
            ),
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics, truths
