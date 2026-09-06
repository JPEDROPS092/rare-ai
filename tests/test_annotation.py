"""ClinVar parsing/mapping + VEP REST extraction tests."""

from __future__ import annotations

import polars as pl

from rareai.annotation.clinvar import clinvar_category, join_clinvar, review_stars
from rareai.config import ClinvarTerms


def test_review_stars():
    assert review_stars("reviewed_by_expert_panel") == 3
    assert review_stars("practice_guideline") == 4
    assert review_stars("criteria_provided,_single_submitter") == 1
    assert review_stars("criteria_provided,_multiple_submitters,_no_conflicts") == 2
    assert review_stars(None) is None
    assert review_stars("no_assertion_provided") == 0


def test_clinvar_category():
    terms = ClinvarTerms(
        pathogenic_terms=["pathogenic"],
        likely_pathogenic_terms=["likely_pathogenic"],
        vus_terms=["uncertain_significance"],
        benign_terms=["benign", "likely_benign"],
    )
    assert clinvar_category("Pathogenic", terms) == "pathogenic"
    assert clinvar_category("Pathogenic/Likely_pathogenic", terms) == "pathogenic"
    assert clinvar_category("Likely_pathogenic", terms) == "likely_pathogenic"
    assert clinvar_category("Uncertain_significance", terms) == "vus"
    assert clinvar_category("Benign", terms) == "benign"
    assert clinvar_category("Conflicting_classifications_of_pathogenicity", terms) == "conflicting"
    assert clinvar_category(None, terms) is None


def _terms() -> ClinvarTerms:
    return ClinvarTerms(
        pathogenic_terms=["pathogenic"],
        likely_pathogenic_terms=["likely_pathogenic"],
        vus_terms=["uncertain_significance"],
        benign_terms=["benign", "likely_benign"],
    )


def test_join_clinvar():
    variants = pl.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "pos": [1000, 2000],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
        }
    )
    clinvar = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [1000],
            "ref": ["A"],
            "alt": ["G"],
            "clnsig": ["Pathogenic"],
            "clnrevstat": ["reviewed_by_expert_panel"],
            "clndn": ["Disease A"],
            "clinvar_stars_raw": [3],
        }
    ).drop("clinvar_stars_raw").with_columns(pl.lit(3, dtype=pl.Int8).alias("clinvar_stars_raw"))
    clinvar = clinvar.rename({"clinvar_stars_raw": "_stars"})
    clinvar = clinvar.with_columns(pl.col("_stars").alias("clinvar_stars")).drop("_stars")

    out = join_clinvar(variants, clinvar, _terms())
    assert out["clinvar_category"][0] == "pathogenic"
    assert out["clinvar_category"][1] is None


def test_vep_extract_parsing():
    from rareai.annotation.rest import VEPAnnotator

    response = {
        "input": "1:1000:A:G",
        "most_severe_consequence": "missense_variant",
        "transcript_consequences": [
            {
                "gene_symbol": "GENE1",
                "transcript_id": "ENST00000001",
                "impact": "MODERATE",
                "hgvsc": "ENST00000001:c.1A>G",
                "hgvsp": "ENST00000001:p.Met1Val",
            },
            {
                "gene_symbol": "GENE1",
                "transcript_id": "ENST00000002",
                "impact": "MODIFIER",
                "canonical": 1,
                "hgvsc": "ENST00000002:c.1A>G",
            },
        ],
        "colocated_variants": [
            {
                "variant_id": "rs123",
                "gnomAD": {"AFR": 0.0001, "AMR": 0.0002},
                "gnomADg": {"AFR": 0.0004},
                "clin_sig": ["pathogenic", "likely_pathogenic"],
            }
        ],
    }
    parsed = VEPAnnotator._extract(response)
    assert parsed["csq"] == "missense_variant"
    assert parsed["vep_gene"] == "GENE1"
    assert parsed["transcript_id"] == "ENST00000002"  # canonical preferred
    assert abs(parsed["gnomad_popmax_af"] - 0.0004) < 1e-9
    assert parsed["colocated_clin_sig"] == "likely_pathogenic;pathogenic"
    assert parsed["rsid"] == "rs123"

    error = VEPAnnotator._extract({"input": "x", "error": "boom"})
    assert error["csq"] is None and error["vep_error"] == "boom"


def test_to_ensembl_chrom():
    from rareai.annotation.rest import to_ensembl_chrom

    assert to_ensembl_chrom("chr1") == "1"
    assert to_ensembl_chrom("chrM") == "MT"
    assert to_ensembl_chrom("chrX") == "X"
    assert to_ensembl_chrom("1") == "1"
