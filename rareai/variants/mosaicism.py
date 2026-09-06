"""Mosaicism features derived from read-backed allele fractions.

MVA (Mosaic Variegated Aneuploidy) makes mosaic alleles a first-class signal.
Only VCF AD fields are used here; true mosaic confirmation requires BAM-level
reanalysis (documented limitation) since raw reads are not processed in MVP.
"""

from __future__ import annotations

import polars as pl

from rareai.variants.genotype import ZYG_HET


def add_mosaic_features(
    df: pl.DataFrame,
    min_dp: int = 20,
    min_alt_ad: int = 5,
    low_af_threshold: float = 0.30,
) -> pl.DataFrame:
    """Add ``expected_af``, ``af_deviation``, ``mosaic_candidate`` and ``mosaic_score``.

    ``mosaic_score`` (0..1): deviation of the observed allele fraction below the
    heterozygous expectation (0.5), scaled by ``low_af_threshold``; null when
    depth/read support is insufficient to evaluate.
    """
    expected = (
        pl.when(pl.col("zyg") == ZYG_HET).then(0.5).otherwise(1.0)  # hom_alt
    )
    evaluable = (
        (pl.col("dp") >= min_dp) & (pl.col("ad_alt") >= min_alt_ad) & pl.col("af").is_not_null()
    )
    return df.with_columns(
        expected.alias("expected_af"),
        (pl.col("af") - expected).abs().alias("af_deviation"),
        ((pl.col("zyg") == ZYG_HET) & evaluable & (pl.col("af") < low_af_threshold)).alias(
            "mosaic_candidate"
        ),
        pl.when(evaluable & (pl.col("zyg") == ZYG_HET))
        .then(((low_af_threshold - pl.col("af")) / low_af_threshold).clip(0.0, 1.0).round(4))
        .otherwise(None)
        .alias("mosaic_score"),
    )
