"""Candidate funnel filters with explicit stage accounting."""

from __future__ import annotations

import polars as pl

from rareai.pipeline.stats import Funnel
from rareai.variants.quality import quality_filter


def apply_qc(df: pl.DataFrame, qc, funnel: Funnel) -> pl.DataFrame:
    """Stage: raw parsed rows -> QC-passing non-reference rows."""
    n_in = df.height
    kept = quality_filter(df, qc)
    funnel.stage(
        "quality_control",
        input_count=n_in,
        output_count=kept.height,
        min_dp=qc.min_dp,
        min_gq=qc.min_gq,
        require_pass=qc.require_pass,
    )
    return kept


def filter_to_panel_genes(df: pl.DataFrame, panel_genes: list[str], funnel: Funnel) -> pl.DataFrame:
    """Stage: QC-passed variants -> variants inside phenotype-matched gene bodies."""
    n_in = df.height
    kept = df.filter(pl.col("gene").is_in(panel_genes))
    funnel.stage(
        "gene_panel_filter", input_count=n_in, output_count=kept.height, panel_size=len(panel_genes)
    )
    return kept


def filter_to_coding(df: pl.DataFrame, funnel: Funnel) -> pl.DataFrame:
    """Stage: panel-gene variants -> exon-overlapping (coding/splice) variants."""
    n_in = df.height
    kept = df.filter(pl.col("in_exon").fill_null(False))
    funnel.stage("exon_filter", input_count=n_in, output_count=kept.height)
    return kept
