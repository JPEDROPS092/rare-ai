"""Quality control filters (deterministic, config-driven).

Note on allelic balance: variants outside the configured het AB window are
FLAGGED, not dropped - low allele fractions may represent mosaicism, which is
central to MVA. Hard drops are limited to PASS/DP/GQ/min-alt-AD criteria.
"""

from __future__ import annotations

import polars as pl

from rareai.config import QCConfig
from rareai.variants.genotype import ZYG_HET, ZYG_HOM_ALT


def quality_filter(df: pl.DataFrame, qc: QCConfig) -> pl.DataFrame:
    """Apply hard QC filters; add an ``ab_outlier`` flag for het calls."""
    if df.is_empty():
        return df.with_columns(pl.lit(False).alias("ab_outlier"))

    lo, hi = qc.het_ab_range
    nonref = df.filter(pl.col("zyg").is_in([ZYG_HET, ZYG_HOM_ALT]))
    conditions = [
        pl.col("is_pass").fill_null(False) if qc.require_pass else pl.lit(True),
        pl.col("dp") >= qc.min_dp,
        pl.col("gq") >= qc.min_gq,
        (pl.col("ad_alt") >= qc.min_alt_ad) | pl.col("ad_alt").is_null(),
    ]
    mask = conditions[0]
    for cond in conditions[1:]:
        mask = mask & cond
    kept = (
        nonref.with_columns(pl.lit(True).alias("ab_outlier"))
        .with_columns(
            pl.when((pl.col("af") < lo) | (pl.col("af") > hi))
            .then(True)
            .otherwise(False)
            .alias("ab_outlier")
        )
        .filter(mask)
    )
    return kept
