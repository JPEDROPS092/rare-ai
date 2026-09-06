"""Streaming VCF parser (cyvcf2) writing normalized per-ALT rows to Parquet.

Staging layer: never loads the whole VCF into RAM - rows are flushed in batches.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from cyvcf2 import VCF, Variant

from rareai.log import get_logger
from rareai.variants.genotype import ZYG_HET, ZYG_HOM_ALT, zygosity_for_alt
from rareai.variants.normalizer import normalize_chrom

logger = get_logger(__name__)

#: Zygosity codes: 0 hom_ref, 1 het, 2 hom_alt, -1 missing/other.
STAGING_SCHEMA = pa.schema(
    [
        ("chrom", pa.string()),
        ("pos", pa.int64()),
        ("ref", pa.string()),
        ("alt", pa.string()),
        ("varid", pa.string()),
        ("qual", pa.float32()),
        ("filters", pa.string()),
        ("is_pass", pa.bool_()),
        ("gt", pa.string()),
        ("zyg", pa.int8()),
        ("phased", pa.bool_()),
        ("dp", pa.int32()),
        ("gq", pa.int32()),
        ("ad_ref", pa.int32()),
        ("ad_alt", pa.int32()),
        ("af", pa.float32()),
        ("pgt", pa.string()),
        ("pid", pa.string()),
    ]
)


def _format_int(variant: Variant, key: str, sample_idx: int = 0) -> int | None:
    try:
        val = variant.format(key)
    except Exception:
        return None
    if val is None:
        return None
    return int(val.reshape(-1)[sample_idx])


def _format_ad(variant: Variant) -> list[int] | None:
    try:
        ad = variant.format("AD")
    except Exception:
        return None
    if ad is None:
        return None
    return [int(x) for x in ad.reshape(-1)]


def _format_str(variant: Variant, key: str) -> str | None:
    try:
        val = variant.format(key)
    except Exception:
        return None
    if val is None:
        return None
    flat = val.reshape(-1)
    if len(flat) == 0:
        return None
    raw = flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _row_for_allele(
    variant: Variant,
    alt_idx: int,
    alt: str,
    genotype: list[int],
    phased: bool,
    dp: int | None,
    gq: int | None,
    ad: list[int] | None,
    pgt: str | None,
    pid: str | None,
) -> dict[str, Any]:
    zyg = zygosity_for_alt(genotype, alt_idx)
    # Flat AD layout: [ref_depth, alt1_depth, alt2_depth, ...]; alt_idx is 1-based.
    ad_ref = ad[0] if ad and len(ad) > 0 else None
    ad_alt = ad[alt_idx] if ad and len(ad) > alt_idx else None
    af: float | None = None
    if ad and sum(ad) > 0 and ad_alt is not None:
        af = ad_alt / sum(ad)
    filt = variant.FILTER if variant.FILTER not in (None, "PASS", ".") else "PASS"
    a1, a2 = genotype[0], genotype[1]
    sep = "|" if phased else "/"
    return {
        "chrom": normalize_chrom(variant.CHROM),
        "pos": int(variant.POS),
        "ref": variant.REF,
        "alt": alt,
        "varid": variant.ID if variant.ID not in (None, ".") else None,
        "qual": float(variant.QUAL) if variant.QUAL is not None else None,
        "filters": filt,
        "is_pass": variant.FILTER in (None, "PASS", "."),
        "gt": f"{a1}{sep}{a2}",
        "zyg": zyg,
        "phased": bool(phased),
        "dp": dp,
        "gq": gq,
        "ad_ref": ad_ref,
        "ad_alt": ad_alt,
        "af": af,
        "pgt": pgt,
        "pid": pid,
    }


def iter_variant_rows(vcf_path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield one row per (record, ALT allele) with genotype fields attached."""
    vcf = VCF(str(vcf_path))
    try:
        for variant in vcf:
            if not variant.ALT:
                continue
            genotype_data = variant.genotypes  # [[a1, a2, phased], ...]
            if not genotype_data:
                continue
            genotype, phased = genotype_data[0][0:2], genotype_data[0][2]
            dp = _format_int(variant, "DP")
            gq = _format_int(variant, "GQ")
            ad = _format_ad(variant)
            pgt = _format_str(variant, "PGT")
            pid = _format_str(variant, "PID")
            for alt_idx, alt in enumerate(variant.ALT, start=1):
                yield _row_for_allele(
                    variant, alt_idx, alt, genotype, bool(phased), dp, gq, ad, pgt, pid
                )
    finally:
        vcf.close()


def extract_header(vcf_path: Path | str) -> dict[str, Any]:
    """Capture header metadata (samples, contigs, filters, caller info)."""
    vcf = VCF(str(vcf_path))
    try:
        contigs = dict(zip(vcf.seqnames, vcf.seqlens, strict=False))
        buckets = {"FILTER": [], "FORMAT": [], "INFO": []}
        for item in vcf.header_iter():
            item_type = getattr(item, "type", None)
            if item_type in buckets:
                record_id = item.info().get("ID")
                if record_id:
                    buckets[item_type].append(record_id)
        fileformat = None
        first_line = vcf.raw_header.splitlines()[0] if vcf.raw_header else ""
        if first_line.startswith("##fileformat="):
            fileformat = first_line.split("=", 1)[1]
        return {
            "samples": list(vcf.samples),
            "fileformat": fileformat,
            "contigs": contigs,
            "filters_defined": buckets["FILTER"],
            "formats_defined": buckets["FORMAT"],
            "infos_defined": buckets["INFO"],
            "raw_header": vcf.raw_header,
        }
    finally:
        vcf.close()


def parse_vcf_to_parquet(
    vcf_path: Path | str,
    out_parquet: Path | str,
    header_json: Path | str | None = None,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Stream the VCF into a Parquet file; return parse statistics."""
    vcf_path = Path(vcf_path)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    header = extract_header(vcf_path)
    if header_json:
        Path(header_json).parent.mkdir(parents=True, exist_ok=True)
        Path(header_json).write_text(json.dumps(header, indent=2), encoding="utf-8")

    n_records = 0
    n_rows = 0
    n_pass_rows = 0
    n_nonref_rows = 0
    errors = 0
    batch: list[dict[str, Any]] = []

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal batch
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=STAGING_SCHEMA))
            batch = []

    with pq.ParquetWriter(out_parquet, STAGING_SCHEMA, compression="zstd") as writer:
        vcf = VCF(str(vcf_path))
        try:
            for variant in vcf:
                if not variant.ALT:
                    continue
                n_records += 1
                try:
                    genotype_data = variant.genotypes
                    if not genotype_data:
                        continue
                    genotype, phased = genotype_data[0][0:2], genotype_data[0][2]
                    dp = _format_int(variant, "DP")
                    gq = _format_int(variant, "GQ")
                    ad = _format_ad(variant)
                    pgt = _format_str(variant, "PGT")
                    pid = _format_str(variant, "PID")
                    for alt_idx, alt in enumerate(variant.ALT, start=1):
                        row = _row_for_allele(
                            variant, alt_idx, alt, genotype, bool(phased), dp, gq, ad, pgt, pid
                        )
                        n_rows += 1
                        if row["is_pass"]:
                            n_pass_rows += 1
                        if row["zyg"] in (ZYG_HET, ZYG_HOM_ALT):
                            n_nonref_rows += 1
                        batch.append(row)
                except Exception:
                    errors += 1
                    continue
                if len(batch) >= batch_size:
                    flush(writer)
            flush(writer)
        finally:
            vcf.close()

    stats = {
        "n_records": n_records,
        "n_rows_per_alt": n_rows,
        "n_pass_rows": n_pass_rows,
        "n_nonref_rows": n_nonref_rows,
        "errors": errors,
        "samples": header["samples"],
        "out_parquet": str(out_parquet),
    }
    logger.info(
        "Parsed %s records / %s per-ALT rows (%s PASS, %s non-reference, %s errors) -> %s",
        f"{n_records:,}",
        f"{n_rows:,}",
        f"{n_pass_rows:,}",
        f"{n_nonref_rows:,}",
        errors,
        out_parquet,
    )
    return stats
