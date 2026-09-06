"""Variant normalization: chromosome naming, deduplication, variant type."""

from __future__ import annotations

import polars as pl

_CHR_PREFIX = "chr"


def normalize_chrom(chrom: str) -> str:
    """Normalize a chromosome name to the UCSC ``chr``-prefixed convention.

    ``1`` -> ``chr1``; ``MT`` -> ``chrM``; ``chrX`` stays ``chrX``.
    Decoy/alt contigs keep their names with the prefix applied.
    """
    name = chrom.strip()
    if name.lower().startswith(_CHR_PREFIX):
        if name in ("chrMT",):
            return "chrM"
        return name
    if name == "MT":
        return "chrM"
    return f"{_CHR_PREFIX}{name}"


def dedup_variants(df: pl.DataFrame) -> pl.DataFrame:
    """Keep one row per (chrom, pos, ref, alt): highest QUAL, deterministic order."""
    if df.is_empty():
        return df
    return df.sort(
        by=["chrom", "pos", "ref", "alt", "qual"],
        descending=[False, False, False, False, True],
        nulls_last=True,
    ).unique(subset=["chrom", "pos", "ref", "alt"], keep="first", maintain_order=True)


def variant_type_expr() -> pl.Expr:
    """Classify SNV vs indel vs other as a Polars expression."""
    return (
        pl.when(
            (pl.col("ref").str.len_chars() == 1)
            & (pl.col("alt").str.len_chars() == 1)
            & pl.col("ref").str.contains("^[ACGTN]$")
            & pl.col("alt").str.contains("^[ACGTN]$")
        )
        .then(pl.lit("snv"))
        .when(pl.col("alt").str.contains("^[ACGTN]+$") & pl.col("ref").str.contains("^[ACGTN]+$"))
        .then(pl.lit("indel"))
        .otherwise(pl.lit("other"))
    )
