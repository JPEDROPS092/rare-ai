"""ClinVar evidence: parse the official VCF and join against candidate variants."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from cyvcf2 import VCF

from rareai.log import get_logger
from rareai.variants.normalizer import normalize_chrom

logger = get_logger(__name__)

CLINVAR_SCHEMA = pa.schema(
    [
        ("chrom", pa.string()),
        ("pos", pa.int64()),
        ("ref", pa.string()),
        ("alt", pa.string()),
        ("clnsig", pa.string()),
        ("clnrevstat", pa.string()),
        ("clnvc", pa.string()),
        ("origin", pa.string()),
        ("geneinfo", pa.string()),
        ("clndn", pa.string()),
    ]
)

#: Review-status string -> ClinVar star rating (official mapping).
_REVIEW_STARS = {
    "no_assertion_provided": 0,
    "no_assertion_criteria_provided": 0,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_classifications": 1,
    "reviewed_by_expert_panel": 3,
    "practice_guideline": 4,
}


def review_stars(clnrevstat: str | None) -> int | None:
    if not clnrevstat:
        return None
    return _REVIEW_STARS.get(clnrevstat.strip().lower())


def parse_clinvar_to_parquet(clinvar_vcf: Path | str, out_parquet: Path | str) -> Path:
    """Stream the ClinVar VCF into normalized per-ALT Parquet rows."""
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    batch: list[dict] = []
    n_rows = 0

    def _pick(info_value, index: int) -> str | None:
        """ClinVar per-allele INFO values may be pipe-separated across ALTs."""
        if info_value is None:
            return None
        parts = str(info_value).split("|")
        if len(parts) == 1:
            return parts[0] or None
        if index < len(parts):
            return parts[index] or None
        return str(info_value)

    with pq.ParquetWriter(out_parquet, CLINVAR_SCHEMA, compression="zstd") as writer:
        vcf = VCF(str(clinvar_vcf))
        try:
            for variant in vcf:
                if not variant.ALT:
                    continue
                info = variant.INFO
                for idx, alt in enumerate(variant.ALT, start=0):
                    row = {
                        "chrom": normalize_chrom(variant.CHROM),
                        "pos": int(variant.POS),
                        "ref": variant.REF,
                        "alt": alt,
                        "clnsig": _pick(info.get("CLNSIG"), idx),
                        "clnrevstat": _pick(info.get("CLNREVSTAT"), idx),
                        "clnvc": info.get("CLNVC"),
                        "origin": _pick(info.get("ORIGIN"), idx),
                        "geneinfo": info.get("GENEINFO"),
                        "clndn": info.get("CLNDN"),
                    }
                    row = {
                        k: (v if isinstance(v, str) or v is None else str(v))
                        for k, v in row.items()
                    }
                    batch.append(row)
                if len(batch) >= 100_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=CLINVAR_SCHEMA))
                    n_rows += len(batch)
                    batch = []
        finally:
            vcf.close()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CLINVAR_SCHEMA))
            n_rows += len(batch)
    logger.info("ClinVar parsed: %s rows -> %s", f"{n_rows:,}", out_parquet)
    return out_parquet


def clinvar_category(clnsig: str | None, terms) -> str | None:
    """Map a raw CLNSIG string to a coarse category."""
    if not clnsig:
        return None
    value = clnsig.strip().lower().replace(" ", "_")
    if "pathogenic/likely_pathogenic" in value:
        return "pathogenic"
    for term in terms.pathogenic_terms:
        if value == term:
            return "pathogenic"
    for term in terms.likely_pathogenic_terms:
        if value == term:
            return "likely_pathogenic"
    for term in terms.vus_terms:
        if value == term:
            return "vus"
    for term in terms.benign_terms:
        if value == term:
            return "benign"
    if "conflicting" in value:
        return "conflicting"
    if "pathogenic" in value:
        return "pathogenic"
    if "benign" in value:
        return "benign"
    return "other"


def join_clinvar(variants: pl.DataFrame, clinvar: pl.DataFrame, terms) -> pl.DataFrame:
    """Left-join ClinVar annotations onto variants; add derived columns."""
    if clinvar.is_empty():
        return variants.with_columns(
            pl.lit(None, dtype=pl.String).alias("clnsig"),
            pl.lit(None, dtype=pl.String).alias("clnrevstat"),
            pl.lit(None, dtype=pl.String).alias("clinvar_category"),
            pl.lit(None, dtype=pl.Int8).alias("clinvar_stars"),
        )
    key = ["chrom", "pos", "ref", "alt"]
    cv = clinvar.unique(subset=key, keep="first").select([*key, "clnsig", "clnrevstat", "clndn"])
    stars_map = pl.Series(
        "clinvar_stars",
        [review_stars(v) for v in cv["clnrevstat"].to_list()],
        dtype=pl.Int8,
    )
    cv = cv.with_columns(stars_map)
    result = variants.join(cv, on=key, how="left")
    return result.with_columns(
        pl.col("clnsig")
        .map_elements(lambda v: clinvar_category(v, terms), return_dtype=pl.String)
        .alias("clinvar_category")
    )
