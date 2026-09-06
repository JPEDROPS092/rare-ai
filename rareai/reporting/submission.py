"""Hackathon submission builder (Track 1 ranked-prediction CSV, GRCh38)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rareai.evaluation.metrics import MAX_SUBMISSION_ROWS

SUBMISSION_COLUMNS = [
    "proband_id",
    "chrom_1",
    "pos_1",
    "ref_1",
    "alt_1",
    "chrom_2",
    "pos_2",
    "ref_2",
    "alt_2",
    "epcr",
    "finding_type",
    "notes",
]


def _pair_compound_hets(top: pl.DataFrame) -> pl.DataFrame:
    """Merge the two best compound-het partners of a gene into one paired row."""
    if top.is_empty() or "compound_het" not in top.columns:
        return top.with_columns(pl.lit(None, dtype=pl.String).alias("_partner_key"))
    ch_rows = top.filter(pl.col("compound_het") == True).sort("score", descending=True)  # noqa: E712
    used_ranks: set[int] = set()
    partner_for_row: dict[int, str] = {}
    by_gene: dict[str, list[dict]] = {}
    for row in ch_rows.iter_rows(named=True):
        by_gene.setdefault(row["gene"], []).append(row)
    for rows in by_gene.values():
        available = [r for r in rows if r["rank"] not in used_ranks]
        while len(available) >= 2:
            first, second = available[0], available[1]
            used_ranks.update({first["rank"], second["rank"]})
            partner_for_row[first["rank"]] = _variant_key(second)
            partner_for_row[second["rank"]] = _variant_key(first)
            available = available[2:]
    if not partner_for_row:
        return top.with_columns(pl.lit(None, dtype=pl.String).alias("_partner_key"))
    mapping = pl.DataFrame(
        {
            "rank": list(partner_for_row.keys()),
            "_partner_key": list(partner_for_row.values()),
        }
    )
    return top.join(mapping, on="rank", how="left")


def _variant_key(row: dict) -> str:
    return f"{row['chrom']}:{row['pos']}:{row['ref']}>{row['alt']}"


def _parse_partner(key: str | None) -> tuple:
    if not key:
        return (None, None, None, None)
    left, _, right = key.partition(":")
    pos, _, alleles = right.partition(":")
    ref, _, alt = alleles.partition(">")
    return (left, int(pos), ref, alt)


def build_submission(
    ranked: pl.DataFrame,
    proband_id: str,
    max_rows: int = MAX_SUBMISSION_ROWS,
) -> pl.DataFrame:
    """Build the ranked prediction table in the official submission format."""
    if proband_id in (None, "", "null"):
        raise ValueError("proband_id unknown - configure configs/dataset.yaml after inspection.")
    top = ranked.sort("score", descending=True).head(max_rows * 2)  # headroom for pairing
    top = _pair_compound_hets(top)

    max_score = top["score"].max()
    rows: list[dict] = []
    seen_pair_partners: set[str] = set()
    for row in top.sort("score", descending=True).iter_rows(named=True):
        key = _variant_key(row)
        partner_key = row.get("_partner_key")
        if partner_key and key in seen_pair_partners:
            continue  # already represented inside a paired row
        if len(rows) >= max_rows:
            break
        chrom_2, pos_2, ref_2, alt_2 = _parse_partner(partner_key)
        epcr = float(min(0.9999, max(0.0001, row["score"] / max_score))) if max_score else 0.5
        notes_bits = []
        if row.get("gene"):
            notes_bits.append(f"gene={row['gene']}")
        if row.get("csq"):
            notes_bits.append(f"csq={row['csq']}")
        if row.get("clinvar_category"):
            notes_bits.append(f"clinvar={row['clinvar_category']}")
        if row.get("gnomad_popmax_af") is not None:
            notes_bits.append(f"gnomAD_AF={row['gnomad_popmax_af']:.4g}")
        if row.get("compound_het_confidence"):
            notes_bits.append(f"compound_het={row['compound_het_confidence']}")
        rows.append(
            {
                "proband_id": proband_id,
                "chrom_1": row["chrom"],
                "pos_1": row["pos"],
                "ref_1": row["ref"],
                "alt_1": row["alt"],
                "chrom_2": chrom_2,
                "pos_2": pos_2,
                "ref_2": ref_2,
                "alt_2": alt_2,
                "epcr": round(epcr, 4),
                "finding_type": "primary" if len(rows) == 0 else "secondary",
                "notes": "; ".join(notes_bits),
            }
        )
        if partner_key:
            seen_pair_partners.add(partner_key)

    return pl.DataFrame(
        rows,
        schema={
            "proband_id": pl.String,
            "chrom_1": pl.String,
            "pos_1": pl.Int64,
            "ref_1": pl.String,
            "alt_1": pl.String,
            "chrom_2": pl.String,
            "pos_2": pl.Int64,
            "ref_2": pl.String,
            "alt_2": pl.String,
            "epcr": pl.Float64,
            "finding_type": pl.String,
            "notes": pl.String,
        },
    )


def write_submission_csv(
    submission: pl.DataFrame, team_name: str, model_name: str, out_dir: Path | str
) -> Path:
    """Write ``<team>_<model>.csv`` (filename convention required by the challenge)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{team_name}_{model_name}.csv"
    submission.write_csv(path)
    return path
