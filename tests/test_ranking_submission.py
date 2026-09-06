"""Baseline ranker + evaluation metrics + submission builder tests."""

from __future__ import annotations

import polars as pl

from rareai.config import (
    BaselineConfig,
    BaselineWeights,
    RankingConfig,
    SubmissionConfig,
)
from rareai.evaluation.metrics import (
    Metrics,
    TruthSet,
    average_precision_at_10,
    evaluate_ranking,
    f_max,
    rank_points,
    reciprocal_rank,
    topk_accuracy,
    variant_key,
)
from rareai.ranking.baseline import rank_candidates
from rareai.reporting.submission import SUBMISSION_COLUMNS, build_submission


def _rank_cfg() -> RankingConfig:
    return RankingConfig(
        baseline=BaselineConfig(
            weights=BaselineWeights(
                rarity=0.2,
                pathogenicity=0.2,
                phenotype_similarity=0.2,
                clinical_evidence=0.1,
                gene_disease=0.1,
                inheritance=0.1,
                mosaicism=0.1,
            )
        ),
        submission=SubmissionConfig(team_name="test", model_name="unit"),
    )


def _candidate_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "pos": [100, 200, 300],
            "ref": ["A", "A", "A"],
            "alt": ["G", "T", "C"],
            "gene": ["G1", "G2", "G3"],
            "rarity": [1.0, 0.9, None],
            "pathogenicity": [0.9, 0.2, 0.3],
            "phenotype_similarity": [0.9, 0.1, 0.2],
            "clinical_evidence": [0.8, None, 0.1],
            "gene_disease": [0.7, 0.1, 0.0],
            "inheritance": [0.4, 1.0, 0.4],
            "mosaicism": [0.0, 0.0, 0.0],
        }
    )


def test_rank_candidates_order_and_contributions():
    df = _candidate_frame()
    ranked = rank_candidates(df, _rank_cfg())
    assert ranked.height == 3
    assert ranked["rank"].to_list() == [1, 2, 3]
    # Missing rarity on row 3 must be flagged and filled with the neutral value
    assert ranked["missing_rarity"][2] is True
    assert ranked["missing_rarity"][0] is False
    # Winner is the first variant (strongest overall evidence)
    assert ranked["gene"][0] == "G1"
    # Contribution math: contrib = normalized_feature * weight
    contrib_rarity = ranked["contrib_rarity"][0]
    assert 0.0 <= contrib_rarity <= 0.2


def test_rank_candidates_weight_sum_validation():
    bad = RankingConfig(
        baseline=BaselineConfig(weights=BaselineWeights(rarity=0.9, pathogenicity=0.9))
    )
    try:
        rank_candidates(_candidate_frame(), bad)
        raise AssertionError("expected ValueError for weights != 1.0")
    except ValueError:
        pass


def test_rank_candidates_deterministic():
    df = _candidate_frame()
    a = rank_candidates(df, _rank_cfg()).select(["chrom", "pos", "score"])
    b = rank_candidates(df, _rank_cfg()).select(["chrom", "pos", "score"])
    assert a.equals(b)


# ------------------------------------------------------------------- metrics
def test_variant_key_normalization():
    assert variant_key("chr1", 100, "a", "g") == "chr1:100:A>G"
    assert variant_key("1", 100, "A", "G") == "chr1:100:A>G"


def _ranked(variant_keys: list[str]) -> list[str | None]:
    return variant_keys


def test_topk_mrr_rank_points():
    keys = _ranked(["chr1:1:A>G", "chr2:2:C>T", "chr3:3:T>C", "chr4:4:G>A"])
    truth = TruthSet(primary="chr3:3:T>C")
    assert topk_accuracy(keys, truth, 1) == 0.0
    assert topk_accuracy(keys, truth, 5) == 1.0
    assert reciprocal_rank(keys, truth) == 1 / 3
    assert rank_points(keys, truth) == 80.0  # rank 3 -> 100 - 10*2
    assert rank_points(keys, TruthSet(primary="chr1:1:A>G")) == 100.0
    assert rank_points(keys, TruthSet(primary="chr9:9:A>T")) == 0.0


def test_rank_points_half_credit_for_pair():
    keys = _ranked(["chr1:1:A>G", "chr2:2:C>T"])
    truth = TruthSet(primary="chr1:1:A>G", partner="chr5:5:T>A")
    # Only one member of the heterozygous pair found at rank 1 -> half credit
    assert rank_points(keys, truth) == 50.0


def test_average_precision():
    keys = _ranked(["chr1:1:A>G", "chr2:2:C>T", "chr3:3:T>C"])
    truth = TruthSet(primary="chr3:3:T>C")
    assert average_precision_at_10(keys, truth) == 1 / 3


def test_f_max():
    df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "pos": [1, 2, 3],
            "ref": ["A", "C", "T"],
            "alt": ["G", "T", "C"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    truth = TruthSet(primary="chr3:3:T>C")
    # Predicted positives {1}: P=0, {1,2}: P=0, {1,2,3}: P=1/3, R=1 -> F=0.5
    assert abs(f_max(df, truth) - 0.5) < 1e-9


def test_evaluate_ranking_aggregates():
    df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "pos": [1, 2],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "score": [0.9, 0.8],
            "rank": [1, 2],
        }
    )
    metrics = evaluate_ranking(df, [TruthSet(primary="chr1:1:A>G")])
    assert isinstance(metrics, Metrics)
    assert metrics.topk["1"] == 1.0
    assert metrics.mrr == 1.0


# ---------------------------------------------------------------- submission
def _ranked_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rank": [1, 2, 3, 4],
            "chrom": ["chr1", "chr2", "chr3", "chr3"],
            "pos": [100, 200, 300, 400],
            "ref": ["A", "C", "T", "T"],
            "alt": ["G", "T", "C", "A"],
            "gene": ["G1", "G2", "G3", "G3"],
            "score": [0.9, 0.8, 0.7, 0.6],
            "csq": ["missense_variant", "stop_gained", "synonymous_variant", None],
            "clinvar_category": ["pathogenic", None, None, None],
            "gnomad_popmax_af": [0.00001, 0.001, None, None],
            "compound_het": [False, False, True, True],
            "compound_het_confidence": [None, None, "phased_in_trans", "phased_in_trans"],
        }
    )


def test_build_submission_pairs_compound_het():
    sub = build_submission(_ranked_df(), proband_id="PROB1", max_rows=10)
    assert sub.columns == SUBMISSION_COLUMNS
    # Ranks 3+4 form one paired row -> 3 rows total
    assert sub.height == 3
    paired = sub.filter(pl.col("chrom_1") == "chr3")
    assert paired.height == 1
    assert paired["chrom_2"][0] == "chr3"
    assert paired["pos_2"][0] == 400
    assert paired["finding_type"][0] == "secondary"
    # Top row is the primary finding; epcr within (0, 1]
    assert sub["finding_type"][0] == "primary"
    assert all(0 < e <= 1 for e in sub["epcr"].to_list())
    assert sub["proband_id"].unique().to_list() == ["PROB1"]


def test_build_submission_max_rows():
    sub = build_submission(_ranked_df(), proband_id="PROB1", max_rows=1)
    assert sub.height == 1
