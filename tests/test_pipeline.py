"""End-to-end integration test: synthetic VCF through the real funnel code."""

from __future__ import annotations

import polars as pl

from rareai.annotation.clinvar import join_clinvar
from rareai.annotation.genes import (
    assign_gene_bodies,
    flag_exon_overlap,
    parse_gtf_to_parquet,
)
from rareai.config import ClinvarTerms, QCConfig, get_annotation_config, get_ranking_config
from rareai.features.build import BASELINE_FEATURES, build_features, feature_availability
from rareai.pipeline.run import _null_vep_columns
from rareai.pipeline.stats import Funnel
from rareai.ranking.baseline import rank_candidates
from rareai.reporting.submission import build_submission
from rareai.variants.filter import apply_qc, filter_to_coding, filter_to_panel_genes
from rareai.variants.genotype import find_compound_hets
from rareai.variants.mosaicism import add_mosaic_features
from rareai.variants.normalizer import dedup_variants
from rareai.variants.parser import parse_vcf_to_parquet


def _clinvar_terms() -> ClinvarTerms:
    return ClinvarTerms(
        pathogenic_terms=["pathogenic"],
        likely_pathogenic_terms=["likely_pathogenic"],
        vus_terms=["uncertain_significance"],
        benign_terms=["benign", "likely_benign"],
    )


def test_full_funnel_no_network(mini_vcf, mini_gtf, tmp_path, phenotype_scores):
    funnel = Funnel()

    # 1. Parse
    staging = tmp_path / "variants.parquet"
    parse_vcf_to_parquet(mini_vcf, staging)
    variants = pl.read_parquet(staging)
    assert variants.height == 7

    # 2. QC (non-ref, PASS, DP/GQ/AD thresholds)
    qc = QCConfig(require_pass=True, min_dp=10, min_gq=20, min_alt_ad=3)
    kept = apply_qc(variants, qc, funnel)
    kept = dedup_variants(kept)
    assert kept.height == 5  # drops LowQual + hom-ref multiallelic + low-AD indel

    # 3. Gene assignment + panel filter
    genes_path, exons_path = parse_gtf_to_parquet(mini_gtf, tmp_path / "ann")
    genes = pl.read_parquet(genes_path)
    assigned = assign_gene_bodies(kept, genes)
    panel = ["GENE1", "GENE2", "GENE3"]
    in_panel = filter_to_panel_genes(assigned, panel, funnel)
    assert in_panel.height == 4  # pos 1000 (GENE1), 5000/6000 (GENE2), 3000-T (GENE3)

    exons = pl.read_parquet(exons_path)
    exon_flagged = flag_exon_overlap(in_panel, exons, splice_window_bp=2)
    coding = filter_to_coding(exon_flagged, funnel)
    # Only pos 6000 lies inside an exon body: exon1 spans pos 5050..5100
    # (5000 is 48 bp upstream, outside the 2 bp splice window), exon2 6000..6100.
    # pos 1000 is gene-body only (exon starts at 1020); pos 3000 is 48 bp
    # upstream of the GENE3 exon (3050..3100).
    assert coding["pos"].to_list() == [6000]

    # 4. Mosaic + ClinVar + null VEP (offline)
    candidates = add_mosaic_features(coding)
    clinvar = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [6000],
            "ref": ["A"],
            "alt": ["T"],
            "clnsig": ["Pathogenic"],
            "clnrevstat": ["reviewed_by_expert_panel"],
            "clndn": ["Disease A"],
            "clinvar_stars": [3],
        }
    )
    candidates = join_clinvar(candidates, clinvar, _clinvar_terms())
    candidates = _null_vep_columns(candidates)
    candidates = candidates.filter(pl.col("gnomad_popmax_af").is_null())

    # 5. Compound het
    candidates = find_compound_hets(candidates)
    assert "compound_het" in candidates.columns

    # 6. Features + ranking + submission
    features = build_features(
        candidates, phenotype_scores, get_annotation_config(), get_ranking_config()
    )
    assert all(f in features.columns for f in BASELINE_FEATURES)
    coverage = feature_availability(features)
    assert coverage["phenotype_similarity"] == 1.0
    assert coverage["rarity"] == 0.0  # no VEP data offline

    ranked = rank_candidates(features, get_ranking_config())
    assert ranked.height == candidates.height
    assert ranked["rank"].is_sorted()

    # ClinVar pathogenic variant must outrank the unannotated one
    top_row = ranked.head(1)
    assert top_row["clinvar_category"][0] == "pathogenic"

    sub = build_submission(ranked, proband_id="TESTPROB")
    assert sub.height >= 1
    assert sub["finding_type"][0] == "primary"

    # Funnel accounting exists for every stage
    names = [s.stage for s in funnel.stages]
    assert {"quality_control", "gene_panel_filter", "exon_filter"} <= set(names)
