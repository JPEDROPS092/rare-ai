"""Variant normalization and genotype tests."""

from __future__ import annotations

import polars as pl

from rareai.variants.genotype import (
    ZYG_HET,
    ZYG_HOM_ALT,
    ZYG_HOM_REF,
    find_compound_hets,
    is_in_trans,
    zygosity_for_alt,
)
from rareai.variants.normalizer import dedup_variants, normalize_chrom, variant_type_expr


def test_normalize_chrom():
    assert normalize_chrom("1") == "chr1"
    assert normalize_chrom("chr1") == "chr1"
    assert normalize_chrom("MT") == "chrM"
    assert normalize_chrom("chrMT") == "chrM"
    assert normalize_chrom("X") == "chrX"
    assert normalize_chrom("1_KI270706v1_random") == "chr1_KI270706v1_random"


def test_zygosity_for_alt():
    assert zygosity_for_alt([0, 1], 1) == ZYG_HET
    assert zygosity_for_alt([1, 1], 1) == ZYG_HOM_ALT
    assert zygosity_for_alt([0, 0], 1) == ZYG_HOM_REF
    assert zygosity_for_alt([0, 1], 2) == ZYG_HOM_REF  # multiallelic: other allele
    assert zygosity_for_alt([1, 2], 2) == ZYG_HET
    assert zygosity_for_alt([None, 1], 1) == -1


def test_is_in_trans():
    assert is_in_trans("0|1", "1|0") is True
    assert is_in_trans("0|1", "0|1") is False
    assert is_in_trans(None, "1|0") is None
    assert is_in_trans("0/1", "1/0") is True  # unphased separators tolerated


def test_dedup_keeps_best_qual():
    df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [100, 100],
            "ref": ["A", "A"],
            "alt": ["G", "G"],
            "qual": [10.0, 99.0],
        }
    )
    out = dedup_variants(df)
    assert out.height == 1
    assert out["qual"][0] == 99.0


def test_variant_type_expr():
    df = pl.DataFrame(
        {"ref": ["A", "A", "AT"], "alt": ["G", "GT", "A"]}
    ).with_columns(variant_type_expr().alias("vtype"))
    assert df["vtype"].to_list() == ["snv", "indel", "indel"]


def test_find_compound_hets():
    df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "pos": [100, 200, 300],
            "ref": ["A", "A", "A"],
            "alt": ["G", "T", "C"],
            "gene": ["GENE1", "GENE1", "GENE1"],
            "zyg": [ZYG_HET, ZYG_HET, ZYG_HOM_ALT],
            "pgt": ["0|1", "1|0", None],
            "pid": [None, None, None],
        }
    )
    out = find_compound_hets(df)
    ch = out.filter(pl.col("compound_het"))
    assert ch.height == 2
    assert set(ch["compound_het_confidence"].to_list()) == {"phased_in_trans"}
    # hom_alt row is not part of a compound het
    assert out.filter(pl.col("zyg") == ZYG_HOM_ALT)["compound_het"][0] is False
