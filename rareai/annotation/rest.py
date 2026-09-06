"""Ensembl REST VEP annotation with a local JSONL cache (cost control).

Only the filtered candidate set is annotated - never the full VCF.
Frequencies come from VEP colocated variants (gnomAD exomes/genomes where
available); absence is recorded as null, never fabricated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from rareai.config import VEPConfig
from rareai.log import get_logger

logger = get_logger(__name__)


def to_ensembl_chrom(chrom: str) -> str:
    """Convert UCSC-style chrom to Ensembl REST style (chr1 -> 1, chrM -> MT)."""
    name = chrom[3:] if chrom.lower().startswith("chr") else chrom
    if name == "M":
        return "MT"
    return name


class VEPAnnotator:
    """Batch annotator for candidate variants via Ensembl REST VEP."""

    def __init__(self, cfg: VEPConfig) -> None:
        self.cfg = cfg
        self.cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> None:
        path = Path(self.cfg.cache_file)
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.cache[record["key"]] = record["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _append_cache(self, key: str, response: dict[str, Any]) -> None:
        path = Path(self.cfg.cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "response": response}) + "\n")
        self.cache[key] = response

    # --------------------------------------------------------------- requests
    def _fetch_batch(self, identifiers: list[str]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        url = self.cfg.url
        delay = 1.0 / max(self.cfg.requests_per_second, 0.1)
        last_error: Exception | None = None
        for attempt in range(self.cfg.retries + 1):
            try:
                with httpx.Client(timeout=self.cfg.timeout_s) as client:
                    response = client.post(
                        url,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                        json={"variants": identifiers},
                    )
                    if response.status_code == 429:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    for item in payload:
                        key = item.get("input", "")
                        if key:
                            results[key] = item
                    time.sleep(delay)
                    return results
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        logger.error("VEP batch failed after %d retries: %s", self.cfg.retries, last_error)
        return {ident: {"input": ident, "error": str(last_error)} for ident in identifiers}

    def annotate_batch(self, identifiers: list[str]) -> dict[str, dict[str, Any]]:
        """Annotate identifiers using cache-first, batched REST calls."""
        out: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for ident in identifiers:
            if ident in self.cache:
                out[ident] = self.cache[ident]
            else:
                missing.append(ident)
        if not missing or not self.cfg.enabled:
            for ident in missing:
                out.setdefault(ident, {"input": ident, "error": "cache_miss_disabled"})
            return out

        total_batches = (len(missing) + self.cfg.batch_size - 1) // self.cfg.batch_size
        for batch_idx in range(total_batches):
            chunk = missing[batch_idx * self.cfg.batch_size : (batch_idx + 1) * self.cfg.batch_size]
            fetched = self._fetch_batch(chunk)
            for ident in chunk:
                record = fetched.get(ident, {"input": ident, "error": "not_returned"})
                self._append_cache(ident, record)
                out[ident] = record
            logger.info(
                "VEP batch %d/%d done (%d variants)", batch_idx + 1, total_batches, len(chunk)
            )
        return out

    # ---------------------------------------------------------------- parsing
    @staticmethod
    def _extract(response: dict[str, Any]) -> dict[str, Any]:
        """Extract the fields we need from one VEP response item."""
        if "error" in response:
            return {
                "csq": None,
                "impact": None,
                "vep_gene": None,
                "transcript_id": None,
                "hgvsc": None,
                "hgvsp": None,
                "gnomad_popmax_af": None,
                "colocated_clin_sig": None,
                "rsid": None,
                "vep_error": response["error"],
            }
        transcripts = response.get("transcript_consequences") or []
        canonical = next((t for t in transcripts if t.get("canonical")), None)
        chosen = canonical or (transcripts[0] if transcripts else None)

        popmax_af: float | None = None
        clin_sigs: list[str] = []
        rsid: str | None = None
        for colocated in response.get("colocated_variants") or []:
            rsid = rsid or colocated.get("variant_id")
            for freq_key in ("gnomADg", "gnomAD", "gnomADe", "gnomAD_alleles"):
                freq = colocated.get(freq_key)
                if isinstance(freq, dict):
                    for value in freq.values():
                        if isinstance(value, (int, float)) and value is not None:
                            popmax_af = max(popmax_af, float(value)) if popmax_af else float(value)
            for sig in colocated.get("clin_sig") or []:
                if isinstance(sig, str):
                    clin_sigs.append(sig)

        return {
            "csq": response.get("most_severe_consequence"),
            "impact": (chosen or {}).get("impact"),
            "vep_gene": (chosen or {}).get("gene_symbol"),
            "transcript_id": (chosen or {}).get("transcript_id"),
            "hgvsc": (chosen or {}).get("hgvsc"),
            "hgvsp": (chosen or {}).get("hgvsp"),
            "gnomad_popmax_af": popmax_af,
            "colocated_clin_sig": ";".join(sorted(set(clin_sigs))) or None,
            "rsid": rsid,
            "vep_error": None,
        }

    def annotate_variants(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add VEP-derived columns to a candidate variant DataFrame."""
        if df.is_empty():
            return df.with_columns(
                pl.lit(None, dtype=pl.String).alias("csq"),
                pl.lit(None, dtype=pl.String).alias("impact"),
                pl.lit(None, dtype=pl.String).alias("vep_gene"),
                pl.lit(None, dtype=pl.String).alias("transcript_id"),
                pl.lit(None, dtype=pl.String).alias("hgvsc"),
                pl.lit(None, dtype=pl.String).alias("hgvsp"),
                pl.lit(None, dtype=pl.Float64).alias("gnomad_popmax_af"),
                pl.lit(None, dtype=pl.String).alias("colocated_clin_sig"),
                pl.lit(None, dtype=pl.String).alias("rsid"),
                pl.lit(None, dtype=pl.String).alias("vep_error"),
            )

        identifiers: list[str] = []
        for row in df.iter_rows(named=True):
            chrom = to_ensembl_chrom(row["chrom"])
            identifiers.append(f"{chrom}:{row['pos']}:{row['ref']}:{row['alt']}")
        responses = self.annotate_batch(identifiers)

        extracted = [
            self._extract(responses.get(ident, {"error": "missing"})) for ident in identifiers
        ]
        cols = {name: [row[name] for row in extracted] for name in extracted[0]}
        dtypes = {
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
        add_df = pl.DataFrame(
            {name: pl.Series(name, values, dtype=dtypes[name]) for name, values in cols.items()}
        )
        return (
            df.with_row_index("_idx")
            .join(add_df.with_row_index("_idx"), on="_idx", how="left")
            .drop("_idx")
        )
