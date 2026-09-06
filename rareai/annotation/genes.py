"""Gene model parsing (Ensembl GTF) and variant-to-gene interval assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from rareai.log import get_logger
from rareai.variants.normalizer import normalize_chrom

logger = get_logger(__name__)

GENE_SCHEMA = {
    "chrom": pl.String,
    "start": pl.Int64,  # 0-based, inclusive
    "end": pl.Int64,  # 0-based, exclusive
    "gene_id": pl.String,
    "gene_name": pl.String,
    "gene_type": pl.String,
}
EXON_SCHEMA = {
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "gene_id": pl.String,
    "gene_name": pl.String,
}


def _parse_attrs(field: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in field.split(";"):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition(" ")
        attrs[key] = value.strip('"')
    return attrs


def parse_gtf_to_parquet(gtf_path: Path | str, out_dir: Path | str) -> tuple[Path, Path]:
    """Extract gene bodies and exons (0-based half-open) into Parquet."""
    gtf_path = Path(gtf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genes_path = out_dir / "genes.parquet"
    exons_path = out_dir / "exons.parquet"

    genes: list[dict] = []
    exons: list[dict] = []
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _source, feature, start, end, *_rest, attrs_field = parts[:9]
            if feature == "gene":
                attrs = _parse_attrs(attrs_field)
                genes.append(
                    {
                        "chrom": normalize_chrom(chrom),
                        "start": int(start) - 1,
                        "end": int(end),
                        "gene_id": attrs.get("gene_id", ""),
                        "gene_name": attrs.get("gene_name", ""),
                        "gene_type": attrs.get("gene_type", ""),
                    }
                )
            elif feature == "exon":
                attrs = _parse_attrs(attrs_field)
                exons.append(
                    {
                        "chrom": normalize_chrom(chrom),
                        "start": int(start) - 1,
                        "end": int(end),
                        "gene_id": attrs.get("gene_id", ""),
                        "gene_name": attrs.get("gene_name", ""),
                    }
                )

    pl.DataFrame(genes, schema=GENE_SCHEMA).write_parquet(genes_path, compression="zstd")
    pl.DataFrame(exons, schema=EXON_SCHEMA).write_parquet(exons_path, compression="zstd")
    logger.info("GTF parsed: %d genes, %d exons", len(genes), len(exons))
    return genes_path, exons_path


def load_genes(genes_path: Path | str) -> pl.DataFrame:
    return pl.read_parquet(genes_path)


def load_exons(exons_path: Path | str) -> pl.DataFrame:
    return pl.read_parquet(exons_path)


def assign_gene_bodies(
    variants: pl.DataFrame, genes: pl.DataFrame, keep_unassigned: bool = False
) -> pl.DataFrame:
    """Assign each variant to overlapping gene bodies (explodes multi-gene overlaps).

    Positions are 1-based in VCF; gene intervals are 0-based half-open, so a
    variant at VCF pos P overlaps [start, end) when start <= P-1 < end.
    """
    if variants.is_empty():
        return variants.with_columns(
            pl.lit(None, dtype=pl.String).alias("gene"),
            pl.lit(None, dtype=pl.String).alias("gene_id"),
        )

    row_id = pl.int_range(pl.len(), dtype=pl.Int64)
    variants = variants.with_columns(row_id.alias("_row_id"))

    pairs_idx: list[int] = []
    pairs_gene_id: list[str] = []
    pairs_gene_name: list[str] = []
    for (chrom,), gene_chunk in genes.group_by(["chrom"], maintain_order=True):
        var_chunk = variants.filter(pl.col("chrom") == chrom)
        if var_chunk.is_empty():
            continue
        var_pos = var_chunk["pos"].to_numpy()
        order = np.argsort(var_pos)
        sorted_pos = var_pos[order]
        row_ids = var_chunk["_row_id"].to_numpy()[order]

        g = gene_chunk.sort("start")
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()
        gene_ids = g["gene_id"].to_list()
        gene_names = g["gene_name"].to_list()
        for i in range(len(g)):
            # VCF pos P is inside gene [start, end) when start+1 <= P <= end.
            lo = np.searchsorted(sorted_pos, starts[i] + 1, side="left")
            hi = np.searchsorted(sorted_pos, ends[i], side="right")
            if hi > lo:
                n_hits = hi - lo
                pairs_idx.extend(row_ids[lo:hi].tolist())
                pairs_gene_id.extend([gene_ids[i]] * n_hits)
                pairs_gene_name.extend([gene_names[i]] * n_hits)

    if not pairs_idx:
        return variants.with_columns(
            pl.lit(None, dtype=pl.String).alias("gene"),
            pl.lit(None, dtype=pl.String).alias("gene_id"),
        ).drop("_row_id")

    gene_cols = pl.DataFrame(
        {"_row_id": pairs_idx, "gene": pairs_gene_name, "gene_id": pairs_gene_id}
    )
    assigned = variants.join(
        gene_cols, on="_row_id", how="inner" if not keep_unassigned else "left"
    )
    return assigned.drop("_row_id").unique(subset=["chrom", "pos", "ref", "alt", "gene"])


def flag_exon_overlap(
    variants: pl.DataFrame, exons: pl.DataFrame, splice_window_bp: int = 2
) -> pl.DataFrame:
    """Add boolean ``in_exon`` (exon body or splice window +/- bp)."""
    if variants.is_empty():
        return variants.with_columns(pl.lit(False).alias("in_exon"))

    var_index: dict[str, np.ndarray] = {}
    for (chrom,), chunk in variants.group_by(["chrom"], maintain_order=True):
        pos = chunk["pos"].to_numpy()
        var_index[chrom] = np.sort(pos)

    hits: set[tuple[str, int]] = set()
    for (chrom,), exon_chunk in exons.group_by(["chrom"], maintain_order=True):
        sorted_pos = var_index.get(chrom)
        if sorted_pos is None or len(sorted_pos) == 0:
            continue
        starts = exon_chunk["start"].to_numpy()
        ends = exon_chunk["end"].to_numpy()
        for i in range(len(exon_chunk)):
            # Exon [start, end) with splice window w: start+1-w <= P <= end+w.
            lo = np.searchsorted(sorted_pos, starts[i] + 1 - splice_window_bp, side="left")
            hi = np.searchsorted(sorted_pos, ends[i] + splice_window_bp, side="right")
            if hi > lo:
                hits.update((chrom, int(p)) for p in sorted_pos[lo:hi])
    if not hits:
        return variants.with_columns(pl.lit(False).alias("in_exon"))
    hits_df = pl.DataFrame(
        {"chrom": [h[0] for h in hits], "pos": [h[1] for h in hits], "_in_exon": True}
    )
    return (
        variants.join(hits_df, on=["chrom", "pos"], how="left")
        .with_columns(pl.col("_in_exon").fill_null(False).alias("in_exon"))
        .drop("_in_exon")
    )


def read_parquet_count(path: Path | str) -> int:
    return pq.read_metadata(Path(path)).num_rows
