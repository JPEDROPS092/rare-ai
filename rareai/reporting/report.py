"""Explainable variant reports and the pipeline quality report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from rareai.features.build import BASELINE_FEATURES, FEATURE_SOURCES
from rareai.pipeline.stats import Funnel
from rareai.variants.genotype import zygosity_label

_EVIDENCE_LABELS = {
    "rarity": "Rare in population (gnomAD via VEP)",
    "pathogenicity": "Functional/clinical pathogenicity",
    "phenotype_similarity": "Phenotype compatibility (HPO similarity)",
    "clinical_evidence": "Clinical evidence (ClinVar)",
    "gene_disease": "Gene-disease association (HPO)",
    "inheritance": "Inheritance compatibility",
    "mosaicism": "Mosaicism evidence (allele fraction)",
}


def render_candidate_report(row: dict[str, Any], weights: dict[str, float]) -> str:
    """Markdown block for one ranked candidate (Section 24 structure)."""
    variant = f"{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']}"
    lines = [
        f"## Rank #{row['rank']} - {row.get('gene') or 'intergenic'}",
        "",
        f"- **Variant**: `{variant}`",
        f"- **Ranking score**: {row['score']:.4f}",
        f"- **Evidence confidence**: {float(row.get('evidence_confidence') or 0):.2f}"
        " (fraction of evidence groups with available data)",
        f"- **Zygosity**: {zygosity_label(int(row.get('zyg', -1)))}",
        f"- **Consequence**: {row.get('csq') or 'unavailable'}",
        f"- **ClinVar**: {row.get('clinvar_category') or 'not present'}"
        + (
            f" ({row.get('clnsig')}, {row.get('clinvar_stars')} stars)"
            if row.get("clinvar_category")
            else ""
        ),
        "",
        "### Why this variant? (weighted contributions)",
        "",
        "| Evidence | Weight | Contribution | Value available |",
        "|---|---:|---:|---|",
    ]
    contributions = {f: row.get(f"contrib_{f}") for f in BASELINE_FEATURES if f in weights}
    for feature, contribution in sorted(contributions.items(), key=lambda kv: -(kv[1] or 0.0)):
        missing = bool(row.get(f"missing_{feature}"))
        lines.append(
            f"| {_EVIDENCE_LABELS[feature]} | {weights[feature]:.2f} "
            f"| {contribution:+.3f} | {'no (neutral value used)' if missing else 'yes'} |"
        )

    lines += ["", "### Supporting evidence", ""]
    if row.get("clinvar_category") in ("pathogenic", "likely_pathogenic"):
        lines.append(f"- ClinVar classification: {row.get('clnsig')} (source: ClinVar VCF)")
    if row.get("gnomad_popmax_af") is not None:
        lines.append(
            f"- gnomAD popmax AF: {row.get('gnomad_popmax_af'):.5f} (source: Ensembl VEP colocated)"
        )
    if row.get("phenotype_similarity") is not None:
        lines.append(
            f"- HPO similarity patient vs {row.get('gene')}: "
            f"{float(row['phenotype_similarity']):.3f} (source: HPO phenotype_to_genes)"
        )
    if row.get("disease_name"):
        lines.append(
            f"- Best matching disease: {row.get('disease_name')} "
            f"(similarity {float(row.get('gene_disease_score') or 0):.3f})"
        )
    if row.get("compound_het_confidence"):
        lines.append(f"- Compound-heterozygous: {row.get('compound_het_confidence')}")
    if not any(line.startswith("- ") for line in lines[-6:]):
        lines.append("- No positive external evidence retrieved (see gaps).")

    lines += ["", "### Contradicting evidence", ""]
    if row.get("gnomad_popmax_af") is not None and float(row["gnomad_popmax_af"]) > 0.01:
        lines.append(
            f"- Common in population (AF {row['gnomad_popmax_af']:.4f}) - argues against causality"
        )
    if row.get("clinvar_category") in ("benign",):
        lines.append("- ClinVar classifies this variant as benign.")
    if not any(line.startswith("- ") for line in lines[-3:]):
        lines.append("- None identified from available sources.")

    lines += ["", "### Evidence gaps", ""]
    gaps = []
    for feature in BASELINE_FEATURES:
        if row.get(f"missing_{feature}"):
            gaps.append(f"- {FEATURE_SOURCES[feature]}: data unavailable")
    if row.get("de_novo_compatibility") is None:
        gaps.append("- De novo status cannot be assessed: no parental genotypes in dataset")
    if not row.get("mosaic_score") is not None:
        gaps.append("- Mosaicism score unavailable (insufficient depth or homozygous call)")
    if not gaps:
        gaps.append("- None recorded.")
    lines.extend(gaps)

    lines += [
        "",
        "### Sources",
        "",
        "- Ensembl REST VEP (consequence, frequencies)",
        "- ClinVar VCF (clinical significance)",
        "- HPO official releases (hp.obo, phenotype_to_genes.txt)",
        "- Dataset VCF: WGS_EX2312012_HGWCNDSX7 (GRCh38)",
        "",
    ]
    return "\n".join(lines)


def write_explainable_report(
    ranked: pl.DataFrame,
    weights: dict[str, float],
    out_path: Path | str,
    top_n: int = 10,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# RareAI - Explainable Candidate Report",
        "",
        "Research hypothesis-generation output. Not a clinical diagnosis.",
        "",
    ]
    for row in ranked.head(top_n).iter_rows(named=True):
        parts.append(render_candidate_report(row, weights))
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def write_pipeline_report(
    funnel: Funnel,
    timing: dict[str, float],
    feature_coverage: dict[str, float],
    out_path: Path | str,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# RareAI - Pipeline Report",
        "",
        funnel.render_markdown(),
        "## Stage timing",
        "",
        "| Stage | Seconds |",
        "|---|---:|",
    ]
    for stage, seconds in timing.items():
        parts.append(f"| {stage} | {seconds:.1f} |")
    parts += [
        "",
        "## Feature availability (candidate set)",
        "",
        "| Feature | Coverage |",
        "|---|---:|",
    ]
    for feature, coverage in feature_coverage.items():
        parts.append(f"| {feature} | {coverage:.1%} |")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out_path
