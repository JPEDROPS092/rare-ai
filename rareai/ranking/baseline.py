"""Interpretable weighted-sum baseline ranker.

Every contribution (weight x normalized feature) is stored per variant so any
rank can be audited: variant -> feature -> contribution -> score -> rank.
"""

from __future__ import annotations

import polars as pl

from rareai.config import RankingConfig
from rareai.features.build import BASELINE_FEATURES


def _normalize_minmax(series: pl.Series) -> pl.Series:
    low, high = series.min(), series.max()
    if low is None or high is None or high == low:
        return pl.Series(series.name, [0.5] * len(series), dtype=pl.Float64)
    return ((series - low) / (high - low)).cast(pl.Float64)


def rank_candidates(df: pl.DataFrame, cfg: RankingConfig) -> pl.DataFrame:
    """Score candidates with the configured baseline weights.

    Returns the input plus ``score``, ``rank``, ``contrib_<feature>`` and
    ``missing_<feature>`` flag columns. Deterministic: identical inputs and
    configs always produce identical rankings.
    """
    weights = cfg.baseline.weights.model_dump()
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"ranking.yaml baseline weights must sum to 1.0 (got {sum(weights.values())})"
        )

    neutral = {
        **{f: 0.5 for f in BASELINE_FEATURES},
        **cfg.baseline.neutral_values,
    }

    scored = df
    total = pl.lit(0.0)
    for feature, weight in weights.items():
        if feature not in scored.columns:
            raise ValueError(f"Feature '{feature}' missing from candidate matrix")
        norm = _normalize_minmax(scored[feature].cast(pl.Float64))
        scored = scored.with_columns(
            norm.alias(f"_norm_{feature}"),
            scored[feature].is_null().alias(f"missing_{feature}"),
        )
        # Missing data -> documented neutral value (flagged via missing_<feature>).
        contribution = (
            pl.col(f"_norm_{feature}").fill_null(float(neutral.get(feature, 0.5))) * weight
        )
        total = total + contribution
        scored = scored.with_columns(contribution.alias(f"contrib_{feature}"))

    tie_columns = [c for c in cfg.tie_breakers if c in scored.columns]
    scored = (
        scored.with_columns(total.round(6).alias("score"))
        .sort(by=["score", *tie_columns], descending=[True, *([False] * len(tie_columns))])
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("rank") + 1)
    )
    norm_cols = [c for c in scored.columns if c.startswith("_norm_")]
    return scored.drop(norm_cols)
