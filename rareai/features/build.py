"""Build the candidate feature matrix used by the ranking stage.

Every feature documents its data source; unavailable data becomes null and is
handled by the ranker's explicit missing-data policy.
"""

from __future__ import annotations

import polars as pl

from rareai.config import AnnotationConfig, RankingConfig
from rareai.variants.genotype import ZYG_HET, ZYG_HOM_ALT

#: Baseline feature groups (Section 13 of the study plan).
BASELINE_FEATURES = [
    "rarity",
    "pathogenicity",
    "phenotype_similarity",
    "clinical_evidence",
    "gene_disease",
    "inheritance",
    "mosaicism",
]

#: Feature -> human-readable data provenance (kept in reports for auditability).
FEATURE_SOURCES = {
    "rarity": "gnomAD allele frequency via Ensembl VEP colocated variants",
    "pathogenicity": "ClinVar CLNSIG + VEP consequence severity (consequence_severity map)",
    "phenotype_similarity": "HPO semantic similarity (Lin) patient vs gene, official annotations",
    "clinical_evidence": "ClinVar significance weighted by review-status stars",
    "gene_disease": "HPO semantic similarity patient vs associated diseases",
    "inheritance": "zygosity + phased compound-heterozygosity (single sample, no parents)",
    "mosaicism": "allele fraction deviation from zygosity expectation (VCF AD fields)",
}

_CLINVAR_SCORE = {
    "pathogenic": 1.0,
    "likely_pathogenic": 0.8,
    "conflicting": 0.5,
    "vus": 0.5,
    "other": 0.3,
    "benign": 0.0,
}


def clinvar_score_expr() -> pl.Expr:
    return (
        pl.when(pl.col("clinvar_category") == "pathogenic")
        .then(1.0)
        .when(pl.col("clinvar_category") == "likely_pathogenic")
        .then(0.8)
        .when(pl.col("clinvar_category") == "conflicting")
        .then(0.5)
        .when(pl.col("clinvar_category") == "vus")
        .then(0.5)
        .when(pl.col("clinvar_category") == "other")
        .then(0.3)
        .when(pl.col("clinvar_category") == "benign")
        .then(0.0)
        .otherwise(None)
        .alias("clinvar_score")
    )


def csq_severity_expr(severity: dict[str, float], default: float) -> pl.Expr:
    return (
        pl.col("csq")
        .replace_strict(severity, default=default, return_dtype=pl.Float64)
        .alias("csq_severity")
    )


def inheritance_expr() -> pl.Expr:
    """Inheritance-compatibility heuristic for a singleton proband.

    phased compound-het 1.0 > hom_alt 0.9 > unphased compound-het 0.7 > het 0.4.
    de_novo compatibility cannot be assessed without parental genotypes (null).
    """
    return (
        pl.when(pl.col("compound_het_confidence") == "phased_in_trans")
        .then(1.0)
        .when(pl.col("compound_het_confidence") == "unphased_candidate")
        .then(0.7)
        .when(pl.col("zyg") == ZYG_HOM_ALT)
        .then(0.9)
        .when(pl.col("zyg") == ZYG_HET)
        .then(0.4)
        .otherwise(None)
        .alias("inheritance")
    )


def build_features(
    candidates: pl.DataFrame,
    phenotype_scores: pl.DataFrame,
    ann_cfg: AnnotationConfig,
    _rank_cfg: RankingConfig,
) -> pl.DataFrame:
    """Compose the feature matrix from annotated candidates + phenotype scores."""
    severity = ann_cfg.consequence_severity
    default_severity = float(severity.get("default", 0.1))

    df = candidates
    df = df.join(
        phenotype_scores.select(
            ["gene", "phenotype_similarity", "gene_disease_score", "disease_id", "disease_name"]
        ),
        on="gene",
        how="left",
    )

    df = df.with_columns(
        clinvar_score_expr(),
        csq_severity_expr(severity, default_severity),
        (
            pl.when(pl.col("gnomad_popmax_af").is_not_null())
            .then(1.0 - pl.col("gnomad_popmax_af").clip(0.0, 1.0))
            .otherwise(None)
            .alias("rarity")
        ),
        inheritance_expr(),
    )
    df = df.with_columns(
        # Clinical evidence: category score scaled by review stars (0.5..1.0 factor).
        (
            pl.when(pl.col("clinvar_score").is_not_null() & pl.col("clinvar_stars").is_not_null())
            .then(pl.col("clinvar_score") * (0.5 + 0.125 * pl.col("clinvar_stars")))
            .otherwise(None)
            .alias("clinical_evidence")
        ),
        (
            pl.when(pl.col("clinvar_score").is_not_null() | pl.col("csq_severity").is_not_null())
            .then(
                pl.max_horizontal(
                    pl.col("clinvar_score").fill_null(0.0),
                    pl.col("csq_severity").fill_null(0.0),
                )
            )
            .otherwise(None)
            .alias("pathogenicity")
        ),
        # Ranking-facing mosaicism feature (null = unevaluable, documented gap).
        pl.col("mosaic_score").alias("mosaicism"),
        # Ranking-facing gene-disease feature (from the HPO best disease match).
        pl.col("gene_disease_score").alias("gene_disease"),
    )
    df = df.with_columns(
        # Evidence confidence (Section 25): fraction of feature groups with data.
        pl.mean_horizontal(
            [
                pl.col("rarity").is_not_null().cast(pl.Float64),
                pl.col("pathogenicity").is_not_null().cast(pl.Float64),
                pl.col("phenotype_similarity").is_not_null().cast(pl.Float64),
                pl.col("clinical_evidence").is_not_null().cast(pl.Float64),
                pl.col("gene_disease_score").is_not_null().cast(pl.Float64),
                pl.col("inheritance").is_not_null().cast(pl.Float64),
                pl.col("mosaic_score").is_not_null().cast(pl.Float64),
            ]
        ).alias("evidence_confidence"),
        pl.lit(None, dtype=pl.Float64).alias("de_novo_compatibility"),
    )
    return df


def feature_availability(df: pl.DataFrame, features: list[str] | None = None) -> dict[str, float]:
    """Coverage fraction (0..1) per feature - surfaces annotation gaps honestly."""
    features = features or BASELINE_FEATURES
    total = max(df.height, 1)
    return {
        feature: round(df[feature].is_not_null().sum() / total, 4) if feature in df.columns else 0.0
        for feature in features
    }
