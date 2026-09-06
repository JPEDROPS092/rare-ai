"""VCF parser + QC + mosaicism tests (synthetic VCF)."""

from __future__ import annotations

import polars as pl

from rareai.config import QCConfig
from rareai.variants.genotype import ZYG_HET, ZYG_HOM_ALT, ZYG_HOM_REF
from rareai.variants.mosaicism import add_mosaic_features
from rareai.variants.parser import extract_header, parse_vcf_to_parquet
from rareai.variants.quality import quality_filter


def test_extract_header(mini_vcf):
    header = extract_header(mini_vcf)
    assert header["samples"] == ["SAMPLE1"]
    assert "1" in header["contigs"]
    assert "AD" in header["formats_defined"]


def test_parse_vcf_multiallelic_and_af(mini_vcf, tmp_path):
    out_parquet = tmp_path / "variants.parquet"
    stats = parse_vcf_to_parquet(mini_vcf, out_parquet, header_json=tmp_path / "hdr.json")
    df = pl.read_parquet(out_parquet)

    assert stats["n_records"] == 6
    assert stats["n_rows_per_alt"] == 7  # one record is multiallelic
    assert df.height == 7

    row_1000 = df.filter((pl.col("pos") == 1000) & (pl.col("alt") == "G"))
    assert row_1000["zyg"][0] == ZYG_HET
    assert abs(row_1000["af"][0] - 0.5) < 1e-6
    assert row_1000["chrom"][0] == "chr1"  # normalized
    assert row_1000["varid"][0] == "rs1"
    assert row_1000["is_pass"][0] is True

    # Multiallelic split: GT 0/1 -> alt T het, alt G hom_ref
    row_t = df.filter((pl.col("pos") == 3000) & (pl.col("alt") == "T"))
    row_g = df.filter((pl.col("pos") == 3000) & (pl.col("alt") == "G"))
    assert row_t["zyg"][0] == ZYG_HET
    assert row_g["zyg"][0] == ZYG_HOM_REF
    assert row_t["ad_alt"][0] == 10
    assert row_g["ad_alt"][0] == 3

    lowqual = df.filter(pl.col("pos") == 4000)
    assert lowqual["is_pass"][0] is False
    assert lowqual["filters"][0] == "LowQual"

    hom = df.filter(pl.col("pos") == 2000)
    assert hom["zyg"][0] == ZYG_HOM_ALT


def test_quality_filter(mini_vcf, tmp_path):
    out_parquet = tmp_path / "variants.parquet"
    parse_vcf_to_parquet(mini_vcf, out_parquet)
    df = pl.read_parquet(out_parquet)

    qc = QCConfig(require_pass=True, min_dp=10, min_gq=20, min_alt_ad=3)
    kept = quality_filter(df, qc)
    # Dropped: LowQual row (4000) and the hom_ref alt-G split of the
    # multiallelic record. The 2000 indel stays: ad_alt=28 passes thresholds.
    assert sorted(kept["pos"].to_list()) == [1000, 2000, 3000, 5000, 6000]
    zyg_3000 = kept.filter(pl.col("pos") == 3000)["zyg"].to_list()
    assert ZYG_HET in zyg_3000 and ZYG_HOM_REF not in zyg_3000


def test_mosaic_features():
    df = pl.DataFrame(
        {
            "zyg": [ZYG_HET, ZYG_HET, ZYG_HOM_ALT],
            "dp": [40, 40, 40],
            "ad_alt": [6, 16, 38],
            "af": [0.15, 0.40, 0.95],
        }
    )
    out = add_mosaic_features(df)
    assert out["mosaic_candidate"][0] is True   # AF 0.15 << het expectation
    assert out["mosaic_candidate"][1] is False
    assert out["mosaic_score"][0] == 0.5  # (0.30 - 0.15) / 0.30
    assert out["mosaic_score"][1] == 0.0
    assert out["expected_af"][2] == 1.0
    assert out["af_deviation"][0] > 0.3
