"""Genotype interpretation: zygosity per ALT, allele fraction, compound-het."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

ZYG_MISSING = -1
ZYG_HOM_REF = 0
ZYG_HET = 1
ZYG_HOM_ALT = 2


def zygosity_for_alt(genotype: Iterable[int], alt_index: int) -> int:
    """Zygosity of a specific ALT allele (1-based index) within a diploid GT.

    ``1/2`` split per-ALT gives: alt 1 -> het, alt 2 -> het.
    ``1/1`` gives hom_alt; ``0/1`` gives het; ``0/0`` gives hom_ref.
    """
    alleles = list(genotype)
    if not alleles or any(a is None for a in alleles):
        return ZYG_MISSING
    count = sum(1 for a in alleles if a == alt_index)
    if count == 0:
        return ZYG_HOM_REF
    if count == len(alleles):
        return ZYG_HOM_ALT
    return ZYG_HET


def zygosity_label(code: int) -> str:
    return {
        ZYG_MISSING: "missing",
        ZYG_HOM_REF: "hom_ref",
        ZYG_HET: "het",
        ZYG_HOM_ALT: "hom_alt",
    }.get(code, f"unknown({code})")


def is_in_trans(pgt_a: str | None, pgt_b: str | None) -> bool | None:
    """Whether two phased genotypes carry the ALT on opposite haplotypes.

    Returns None when phasing is unavailable for either variant.
    """
    if not pgt_a or not pgt_b:
        return None
    a = pgt_a.replace("|", "/")
    b = pgt_b.replace("|", "/")
    if a not in ("0/1", "1/0") or b not in ("0/1", "1/0"):
        return None
    return a != b


def find_compound_hets(df: pl.DataFrame) -> pl.DataFrame:
    """Detect compound-heterozygous pairs within genes.

    Input: variants with columns [chrom, pos, ref, alt, gene, zyg, pgt, pid].
    Output: same frame plus ``compound_het`` (bool) and ``compound_het_confidence``
    ("phased_in_trans" | "unphased_candidate" | null).
    """
    required = {"chrom", "pos", "ref", "alt", "gene", "zyg"}
    if not required.issubset(df.columns):
        raise ValueError(f"compound het input missing columns: {required - set(df.columns)}")
    if df.is_empty():
        return df.with_columns(
            pl.lit(False).alias("compound_het"),
            pl.lit(None, dtype=pl.String).alias("compound_het_confidence"),
        )

    het = df.filter(pl.col("zyg") == ZYG_HET)
    gene_counts = het.group_by("gene").len().filter(pl.col("len") >= 2)
    multi_genes = gene_counts["gene"].to_list()
    pairs = het.filter(pl.col("gene").is_in(multi_genes)).sort(["chrom", "pos"])

    pair_keys: dict[tuple[str, str, str, str], str] = {}
    by_gene: dict[str, list[dict]] = {}
    for row in pairs.iter_rows(named=True):
        by_gene.setdefault(row["gene"], []).append(row)
    for _gene, rows in by_gene.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                in_trans = is_in_trans(a.get("pgt"), b.get("pgt"))
                confidence = "phased_in_trans" if in_trans else "unphased_candidate"
                for r in (a, b):
                    key = (r["chrom"], str(r["pos"]), r["ref"], r["alt"])
                    existing = pair_keys.get(key)
                    if existing != "phased_in_trans":
                        pair_keys[key] = confidence

    if not pair_keys:
        return df.with_columns(
            pl.lit(False).alias("compound_het"),
            pl.lit(None, dtype=pl.String).alias("compound_het_confidence"),
        )

    keys_df = pl.DataFrame(
        {
            "chrom": [k[0] for k in pair_keys],
            "pos": [int(k[1]) for k in pair_keys],
            "ref": [k[2] for k in pair_keys],
            "alt": [k[3] for k in pair_keys],
            "compound_het_confidence": [pair_keys[k] for k in pair_keys],
        }
    )
    result = df.join(keys_df, on=["chrom", "pos", "ref", "alt"], how="left")
    return result.with_columns(
        pl.col("compound_het_confidence").is_not_null().alias("compound_het")
    )
