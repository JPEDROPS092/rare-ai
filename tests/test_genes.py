"""Gene model interval assignment tests (coordinate semantics)."""

from __future__ import annotations

import polars as pl

from rareai.annotation.genes import (
    assign_gene_bodies,
    flag_exon_overlap,
    parse_gtf_to_parquet,
)


def test_parse_gtf(mini_gtf, tmp_path):
    genes_path, exons_path = parse_gtf_to_parquet(mini_gtf, tmp_path)
    genes = pl.read_parquet(genes_path)
    exons = pl.read_parquet(exons_path)
    assert genes.height == 3
    assert exons.height == 4
    gene1 = genes.filter(pl.col("gene_name") == "GENE1")
    assert gene1["start"][0] == 999  # 0-based
    assert gene1["end"][0] == 1100
    assert gene1["chrom"][0] == "chr1"


def _variants_frame(positions: list[int], chrom: str = "chr1") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": [chrom] * len(positions),
            "pos": positions,
            "ref": ["A"] * len(positions),
            "alt": ["G"] * len(positions),
        }
    )


def test_gene_body_boundaries(mini_gtf, tmp_path):
    genes_path, _ = parse_gtf_to_parquet(mini_gtf, tmp_path)
    genes = pl.read_parquet(genes_path)
    # GENE1: 0-based [999, 1100) -> covers VCF pos 1000..1100 exactly
    variants = _variants_frame([999, 1000, 1100, 1101])
    out = assign_gene_bodies(variants, genes)
    assigned = out.filter(pl.col("gene") == "GENE1")["pos"].to_list()
    assert sorted(assigned) == [1000, 1100]


def test_exon_splice_window(mini_gtf, tmp_path):
    _, exons_path = parse_gtf_to_parquet(mini_gtf, tmp_path)
    exons = pl.read_parquet(exons_path)
    # GENE1 exon: 0-based [1019, 1060) -> pos 1020..1060; window +/-2
    variants = _variants_frame([1018, 1019, 1062, 1063])
    out = flag_exon_overlap(variants, exons, splice_window_bp=2)
    flags = dict(zip(out["pos"].to_list(), out["in_exon"].to_list()))
    assert flags[1018] is True   # 2 bp upstream of exon start-1... inside window
    assert flags[1019] is True
    assert flags[1062] is True
    assert flags[1063] is False


def test_assign_multi_gene_overlap(mini_gtf, tmp_path):
    genes_path, _ = parse_gtf_to_parquet(mini_gtf, tmp_path)
    genes = pl.read_parquet(genes_path)
    # Overlapping gene bodies: extend GENE2 over GENE1 territory
    extra = pl.DataFrame(
        {
            "chrom": ["chr1"], "start": [900], "end": [1500],
            "gene_id": ["GENE9"], "gene_name": ["GENE9"], "gene_type": ["protein_coding"],
        },
        schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64,
                "gene_id": pl.String, "gene_name": pl.String, "gene_type": pl.String},
    )
    genes = pl.concat([genes, extra])
    out = assign_gene_bodies(_variants_frame([1000]), genes)
    assert sorted(out["gene"].to_list()) == ["GENE1", "GENE9"]
