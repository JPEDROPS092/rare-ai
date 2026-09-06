"""End-to-end Track 1 pipeline: VCF -> candidates -> features -> ranking -> submission.

Design rule: cheap deterministic filters first, expensive annotation only for
the filtered candidate set (cost control, Section 52 of the study plan).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from rareai.annotation.clinvar import join_clinvar, parse_clinvar_to_parquet
from rareai.annotation.genes import (
    assign_gene_bodies,
    flag_exon_overlap,
    load_exons,
    load_genes,
    parse_gtf_to_parquet,
)
from rareai.annotation.rest import VEPAnnotator
from rareai.config import (
    DatasetConfig,
    get_annotation_config,
    get_dataset_config,
    get_ranking_config,
)
from rareai.data.resources import ensure_all_resources, resource_path
from rareai.evaluation.benchmark import run_benchmark
from rareai.features.build import build_features, feature_availability
from rareai.log import get_logger
from rareai.mva.context import mva_diseases, mva_genes
from rareai.phenotype.docx import save_phenotype_text
from rareai.phenotype.mapping import suggest_terms
from rareai.phenotype.ontology import parse_hp_obo
from rareai.phenotype.patient import (
    load_patient_terms,
    prune_to_specific,
    write_patient_template,
)
from rareai.phenotype.similarity import HPOSimilarity, load_annotations
from rareai.pipeline.stats import Funnel
from rareai.pipeline.tracking import log_experiment
from rareai.ranking.baseline import rank_candidates
from rareai.reporting.report import (
    write_explainable_report,
    write_pipeline_report,
)
from rareai.reporting.submission import build_submission, write_submission_csv
from rareai.variants.filter import apply_qc, filter_to_coding, filter_to_panel_genes
from rareai.variants.genotype import find_compound_hets
from rareai.variants.mosaicism import add_mosaic_features
from rareai.variants.normalizer import dedup_variants
from rareai.variants.parser import parse_vcf_to_parquet

logger = get_logger(__name__)


@dataclass
class RunOptions:
    force: bool = False
    use_vep: bool = True
    top_n: int | None = None


@dataclass
class RunSummary:
    outputs: dict[str, str] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    proband_id: str | None = None
    submission_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": self.outputs,
            "candidate_counts": self.candidate_counts,
            "proband_id": self.proband_id,
            "submission_path": self.submission_path,
        }


class StageTimer:
    def __init__(self) -> None:
        self.timing: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timing[name] = round(time.perf_counter() - start, 2)


def _null_vep_columns(df: pl.DataFrame) -> pl.DataFrame:
    cols = {
        "csq": pl.String,
        "impact": pl.String,
        "vep_gene": pl.String,
        "transcript_id": pl.String,
        "hgvsc": pl.String,
        "hgvsp": pl.String,
        "gnomad_popmax_af": pl.Float64,
        "colocated_clin_sig": pl.String,
        "rsid": pl.String,
        "vep_error": pl.String,
    }
    return df.with_columns([pl.lit(None, dtype=dtype).alias(name) for name, dtype in cols.items()])


def _ensure_track1_raw(dcfg: DatasetConfig) -> Path:
    files = dcfg.tracks["track1"]["files"]
    vcf_name = next((f for f in files if f.endswith(".vcf.gz")), None)
    if not vcf_name:
        raise RuntimeError("No .vcf.gz configured for track1 in configs/dataset.yaml")
    vcf_path = dcfg.paths.raw / vcf_name
    missing = [f for f in files if not (dcfg.paths.raw / f).exists()]
    if not vcf_path.exists() or missing:
        raise RuntimeError(f"Missing dataset files {missing}. Run: rareai download --track track1")
    return vcf_path


def _phenotype_stage(dcfg: DatasetConfig, funnel: Funnel) -> tuple[pl.DataFrame, list[str], dict]:
    """Encode patient phenotype, build HPO similarity, select the gene panel."""
    phen_dir = dcfg.paths.phenotypes
    phen_dir.mkdir(parents=True, exist_ok=True)

    docx_name = next((f for f in dcfg.tracks["track1"]["files"] if f.endswith(".docx")), None)
    if not docx_name:
        raise RuntimeError("Clinical phenotype DOCX not configured for track1")
    text_path = phen_dir / "clinical_phenotype.txt"
    if not text_path.exists():
        save_phenotype_text(dcfg.paths.raw / docx_name, text_path)

    hp_obo = resource_path("hp_obo")
    p2g = resource_path("phenotype_to_genes")
    ontology = parse_hp_obo(hp_obo)

    curated = phen_dir / "patient_phenotype.yaml"
    if not curated.exists():
        text = text_path.read_text(encoding="utf-8")
        suggestions = suggest_terms(text, ontology)
        write_patient_template(
            phen_dir / "patient_suggestions.yaml",
            [{"id": s.id, "label": s.label, "source_text": s.source_text} for s in suggestions],
        )
        raise RuntimeError(
            "Curated phenotype missing. Review/approve terms in "
            f"{phen_dir / 'patient_suggestions.yaml'}, then save the approved "
            f"selection as {curated} (see configs template docs)."
        )

    patient_terms = prune_to_specific(load_patient_terms(curated, ontology), ontology)
    if not patient_terms:
        raise RuntimeError("No approved HPO terms in the curated phenotype file.")
    funnel.stage("phenotype_encoding", input_count=1, output_count=len(patient_terms))
    logger.info("Patient HPO terms: %d (after specificity pruning)", len(patient_terms))

    similarity = HPOSimilarity(ontology, load_annotations(p2g))
    gene_ranking = similarity.rank_genes(patient_terms, min_similarity=0.0)
    panel_cfg = get_ranking_config().gene_panel
    gene_score_map = dict(gene_ranking)
    panel = [
        g
        for g, _ in gene_ranking[: panel_cfg.size]
        if gene_score_map[g] >= panel_cfg.min_similarity
    ]

    rows = []
    for gene in panel:
        best = similarity.best_disease_for_gene(patient_terms, gene)
        rows.append(
            {
                "gene": gene,
                "phenotype_similarity": round(gene_score_map.get(gene, 0.0), 4),
                "gene_disease_score": round(best.similarity, 4) if best else None,
                "disease_id": best.disease_id if best else None,
                "disease_name": best.disease_name if best else None,
            }
        )
    scores_df = pl.DataFrame(rows)
    scores_df.write_parquet(phen_dir / "phenotype_scores.parquet", compression="zstd")

    mva_ctx = {
        "mva_diseases": [{"id": d, "name": n} for d, n in mva_diseases(similarity)],
        "mva_genes": mva_genes(similarity),
        "mva_genes_in_panel": sorted(set(panel) & set(mva_genes(similarity))),
        "note": "derived from official HPO annotations at runtime; not hard-coded",
    }
    (phen_dir / "mva_context.json").write_text(json.dumps(mva_ctx, indent=2))
    logger.info(
        "Gene panel: %d genes (MVA-associated genes in panel: %d)",
        len(panel),
        len(mva_ctx["mva_genes_in_panel"]),
    )
    return scores_df, panel, mva_ctx


def run_track1(options: RunOptions | None = None) -> RunSummary:
    """Execute the full Track 1 pipeline with caching and full accounting."""
    options = options or RunOptions()
    dcfg = get_dataset_config()
    rcfg = get_ranking_config()
    acfg = get_annotation_config()

    funnel = Funnel()
    timer = StageTimer()
    summary = RunSummary()

    # ---------------------------------------------------------------- raw data
    with timer.stage("ensure_raw"):
        vcf_path = _ensure_track1_raw(dcfg)

    # -------------------------------------------------------------- preprocess
    staging_parquet = dcfg.paths.staging / "variants.parquet"
    header_json = dcfg.paths.staging / "vcf_header.json"
    stats_json = dcfg.paths.staging / "parse_stats.json"
    with timer.stage("preprocess_vcf"):
        if options.force or not staging_parquet.exists():
            stats = parse_vcf_to_parquet(vcf_path, staging_parquet, header_json)
            stats_json.write_text(json.dumps(stats, indent=2))
        else:
            stats = json.loads(stats_json.read_text()) if stats_json.exists() else {}
            logger.info("Staging parquet cached: %s", staging_parquet)
        if stats:
            funnel.stage(
                "parse_vcf",
                input_count=stats.get("n_records", 0),
                output_count=stats.get("n_rows_per_alt", 0),
                errors=stats.get("errors", 0),
                samples=stats.get("samples"),
            )

    # ---------------------------------------------------------------- resources
    with timer.stage("resources"):
        ensure_all_resources()
        genes_path = dcfg.paths.annotations / "genes.parquet"
        exons_path = dcfg.paths.annotations / "exons.parquet"
        if options.force or not genes_path.exists():
            parse_gtf_to_parquet(resource_path("ensembl_gtf"), dcfg.paths.annotations)
        clinvar_parquet = dcfg.paths.annotations / "clinvar.parquet"
        if options.force or not clinvar_parquet.exists():
            parse_clinvar_to_parquet(resource_path("clinvar_vcf"), clinvar_parquet)

    # ---------------------------------------------------------------- phenotype
    with timer.stage("phenotype"):
        phenotype_scores, panel, _mva_ctx = _phenotype_stage(dcfg, funnel)

    # --------------------------------------------------------- variant funnel
    with timer.stage("variant_funnel"):
        variants = pl.scan_parquet(staging_parquet).collect()
        qc_kept = apply_qc(variants, dcfg.qc, funnel)
        qc_kept = dedup_variants(qc_kept)

        genes_df = load_genes(genes_path)
        assigned = assign_gene_bodies(qc_kept, genes_df)
        funnel.stage(
            "gene_assignment",
            input_count=qc_kept.height,
            output_count=assigned.height,
            note="multi-gene overlaps explode rows",
        )
        panel_kept = filter_to_panel_genes(assigned, panel, funnel)
        exons_df = load_exons(exons_path)
        panel_kept = flag_exon_overlap(panel_kept, exons_df, acfg.splice_window_bp)
        candidates = filter_to_coding(panel_kept, funnel)
        candidates = add_mosaic_features(candidates)
        summary.candidate_counts["pre_annotation"] = candidates.height

    # ---------------------------------------------------------------- ClinVar
    with timer.stage("clinvar_join"):
        clinvar_df = pl.read_parquet(clinvar_parquet)
        candidates = join_clinvar(candidates, clinvar_df, acfg.clinvar)
        n_annotated = candidates.filter(pl.col("clinvar_category").is_not_null()).height
        funnel.stage("clinvar_annotation", input_count=candidates.height, output_count=n_annotated)

    # -------------------------------------------------------------------- VEP
    with timer.stage("vep_annotation"):
        if options.use_vep and acfg.vep.enabled:
            if acfg.vep.max_variants and candidates.height > acfg.vep.max_variants:
                logger.warning(
                    "Candidate set (%d) exceeds VEP cap (%d) - annotating top slice only.",
                    candidates.height,
                    acfg.vep.max_variants,
                )
                candidates = candidates.head(acfg.vep.max_variants)
            annotator = VEPAnnotator(acfg.vep)
            candidates = annotator.annotate_variants(candidates)
        else:
            candidates = _null_vep_columns(candidates)
        n_vep = candidates.filter(pl.col("csq").is_not_null()).height
        funnel.stage("vep_annotation", input_count=candidates.height, output_count=n_vep)

    # ------------------------------------------------------------ rarity filter
    with timer.stage("rarity_filter"):
        max_af = float(rcfg.funnel.get("rarity_max_af", 0.02))
        keep_path = bool(rcfg.funnel.get("keep_pathogenic_above_af", True))
        before_rarity = candidates.height
        candidates = candidates.filter(
            pl.col("gnomad_popmax_af").is_null()
            | (pl.col("gnomad_popmax_af") <= max_af)
            | (
                keep_path
                & pl.col("clinvar_category")
                .is_in(["pathogenic", "likely_pathogenic"])
                .fill_null(False)
            )
        )
        funnel.stage(
            "rarity_filter",
            input_count=before_rarity,
            output_count=candidates.height,
            max_af=max_af,
        )

    # ----------------------------------------------------------- compound het
    with timer.stage("compound_het"):
        candidates = find_compound_hets(candidates)
        n_ch = candidates.filter(pl.col("compound_het")).height
        summary.candidate_counts["compound_het_variants"] = n_ch

    # ------------------------------------------------------ features & ranking
    with timer.stage("features"):
        features_df = build_features(candidates, phenotype_scores, acfg, rcfg)
        coverage = feature_availability(features_df)
        features_df.write_parquet(
            dcfg.paths.features / "candidates_features.parquet", compression="zstd"
        )

    with timer.stage("ranking"):
        ranked = rank_candidates(features_df, rcfg)
        top_n = options.top_n or rcfg.top_n
        top = ranked.head(top_n)
        rankings_dir = dcfg.paths.results / "rankings"
        rankings_dir.mkdir(parents=True, exist_ok=True)
        top.write_csv(rankings_dir / "top_candidates.csv")
        ranked.write_parquet(rankings_dir / "ranked_all.parquet", compression="zstd")
        summary.candidate_counts["ranked"] = ranked.height
        summary.candidate_counts["reported_top"] = top.height

    # --------------------------------------------------------------- submission
    with timer.stage("submission"):
        proband_id = dcfg.proband_id
        if not proband_id:
            raise RuntimeError(
                "proband_id is not configured. Set it in configs/dataset.yaml "
                "(identifier provided with the dataset / submission template)."
            )
        submission = build_submission(
            ranked, proband_id=proband_id, max_rows=rcfg.submission.max_rows
        )
        sub_path = write_submission_csv(
            submission,
            team_name=rcfg.submission.team_name,
            model_name=rcfg.submission.model_name,
            out_dir=dcfg.paths.results / "submission",
        )
        summary.proband_id = proband_id
        summary.submission_path = str(sub_path)

    # ------------------------------------------------------------------ reports
    with timer.stage("reports"):
        weights = rcfg.baseline.weights.model_dump()
        report_path = write_explainable_report(
            top, weights, dcfg.paths.results / "reports" / "explainable_report.md", top_n=10
        )
        write_pipeline_report(
            funnel, timer.timing, coverage, dcfg.paths.results / "reports" / "pipeline_report.md"
        )
        funnel.write(dcfg.paths.results / "funnel.json")
        (dcfg.paths.results / "timing.json").write_text(json.dumps(timer.timing, indent=2))

    # --------------------------------------------------------------- evaluation
    with timer.stage("evaluation_proxy"):
        run_benchmark(
            ranked,
            truth_csv=None,
            proxy=True,
            out_json=dcfg.paths.results / "evaluation.json",
        )

    with timer.stage("experiment_log"):
        log_experiment(
            {
                "run": "track1",
                "proband_id": proband_id,
                "funnel": funnel.to_dict(),
                "timing": timer.timing,
                "feature_coverage": coverage,
                "top10": top.head(10)
                .select(
                    [
                        "rank",
                        "gene",
                        "chrom",
                        "pos",
                        "ref",
                        "alt",
                        "score",
                        "csq",
                        "clinvar_category",
                        "gnomad_popmax_af",
                    ]
                )
                .to_dicts(),
                "submission_path": str(sub_path),
            }
        )

    summary.outputs = {
        "staging": str(staging_parquet),
        "features": str(dcfg.paths.features / "candidates_features.parquet"),
        "rankings": str(rankings_dir / "top_candidates.csv"),
        "report": str(report_path),
        "submission": str(sub_path),
        "funnel": str(dcfg.paths.results / "funnel.json"),
    }
    logger.info("Pipeline complete. Submission: %s", sub_path)
    return summary
