"""RareAI command-line interface."""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from rareai.config import (
    get_dataset_config,
    hf_token,
)

app = typer.Typer(
    name="rareai",
    help="Explainable AI for rare-disease variant prioritization (MVA Hackathon 2026).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_ERR_NOT_IMPLEMENTED = "Planned for phase {phase} - not implemented yet (no stubs shipped)."


@app.command()
def inspect() -> None:
    """Dataset discovery: build data_catalog.json and DATASET.md."""
    from rareai.data.catalog import inspect_dataset

    catalog = inspect_dataset()
    table = Table(title=f"Dataset: {catalog['repo_id']}")
    for column in ("File", "Size", "Type", "Track 1", "Track 2"):
        table.add_column(column)
    for entry in catalog["files"]:
        table.add_row(
            entry["filename"],
            entry["size_human"],
            entry["type"],
            entry["track1"],
            entry["track2"],
        )
    console.print(table)


@app.command()
def download(
    track: str = typer.Option("track1", help="track1 | track2"),
    file: str | None = typer.Option(None, help="Download a specific dataset file"),
    include_fastq: bool = typer.Option(
        False, help="Also download the ~85 GB FASTQ set (use RAREAI_DATA_DIR)"
    ),
    full: bool = typer.Option(False, help="Track files + all FASTQs (~85 GB)"),
) -> None:
    """Download dataset files (selective; never auto-downloads FASTQs)."""
    from rareai.data.download import download_fastqs, download_file, download_track

    if not hf_token():
        console.print("[red]HF_TOKEN missing. Set it in .env (see .env.example).[/red]")
        raise typer.Exit(1)
    if file:
        path = download_file(file)
    else:
        paths = download_track(track, include_fastq=include_fastq or full)
        path = paths[-1]
        if full:
            fastq_paths = download_fastqs()
            if fastq_paths:
                path = fastq_paths[-1]
    console.print(f"[green]Downloaded:[/green] {path}")


@app.command()
def resources() -> None:
    """Download annotation resources (HPO, ClinVar, Ensembl GTF)."""
    from rareai.data.resources import ensure_all_resources

    paths = ensure_all_resources()
    for name, path in paths.items():
        console.print(f"[green]{name}[/green]: {path}")


@app.command()
def preprocess() -> None:
    """Parse the VCF into normalized staging Parquet."""
    from rareai.variants.parser import parse_vcf_to_parquet

    cfg = get_dataset_config()
    vcf_name = next(f for f in cfg.tracks["track1"]["files"] if f.endswith(".vcf.gz"))
    vcf_path = cfg.paths.raw / vcf_name
    if not vcf_path.exists():
        console.print("[red]VCF missing. Run: rareai download --track track1[/red]")
        raise typer.Exit(1)
    stats = parse_vcf_to_parquet(
        vcf_path, cfg.paths.staging / "variants.parquet", cfg.paths.staging / "vcf_header.json"
    )
    (cfg.paths.staging / "parse_stats.json").write_text(json.dumps(stats, indent=2))
    console.print_json(json.dumps(stats))


@app.command()
def phenotype() -> None:
    """Encode patient phenotype and build the HPO gene panel."""
    from rareai.pipeline.stats import Funnel

    cfg = get_dataset_config()
    scores, _panel, _mva = _run_phenotype(cfg, Funnel())
    console.print(scores.head(20))


def _run_phenotype(cfg, funnel):
    from rareai.pipeline.run import _phenotype_stage

    return _phenotype_stage(cfg, funnel)


@app.command()
def features() -> None:
    """(Pipeline stage) Feature matrix is built as part of `rareai run`."""
    console.print(
        "Features are produced by `rareai run --track track1` "
        "(data/features/candidates_features.parquet)."
    )


@app.command()
def rank() -> None:
    """(Pipeline stage) Ranking is produced by `rareai run`."""
    console.print(
        "Ranking is produced by `rareai run --track track1` (results/rankings/top_candidates.csv)."
    )


@app.command()
def evaluate(
    truth: Path | None = typer.Option(None, help="Truth CSV for real evaluation"),
    proxy: bool = typer.Option(True, help="ClinVar-based sanity proxy"),
) -> None:
    """Evaluate the last ranking against truth (or the documented proxy)."""
    from rareai.evaluation.benchmark import run_benchmark

    cfg = get_dataset_config()
    ranked_path = cfg.paths.results / "rankings" / "ranked_all.parquet"
    if not ranked_path.exists():
        console.print("[red]No ranking found. Run: rareai run --track track1[/red]")
        raise typer.Exit(1)
    ranked = pl.read_parquet(ranked_path)
    metrics, _ = run_benchmark(
        ranked, truth_csv=truth, proxy=proxy, out_json=cfg.paths.results / "evaluation.json"
    )
    console.print_json(json.dumps(metrics.to_dict()))


@app.command()
def evidence() -> None:
    """Evidence retrieval + RAG layer (Phase 8 roadmap)."""
    console.print(_ERR_NOT_IMPLEMENTED.format(phase=8))
    raise typer.Exit(2)


@app.command()
def report() -> None:
    """Print the explainable report path summary."""
    cfg = get_dataset_config()
    path = cfg.paths.results / "reports" / "explainable_report.md"
    if path.exists():
        console.print(f"[green]Report:[/green] {path}")
    else:
        console.print("No report yet. Run: rareai run --track track1")


@app.command(name="run")
def run_cmd(
    track: str = typer.Option("track1", help="Pipeline track"),
    force: bool = typer.Option(False, help="Ignore stage caches"),
    no_vep: bool = typer.Option(False, help="Skip REST annotation (offline dev)"),
    top_n: int | None = typer.Option(None, help="Report top N candidates"),
) -> None:
    """Run the full pipeline end-to-end."""
    from rareai.pipeline.run import RunOptions, run_track1

    if track != "track1":
        console.print("[red]Only track1 is implemented (track2 is Phase 12).[/red]")
        raise typer.Exit(2)
    summary = run_track1(RunOptions(force=force, use_vep=not no_vep, top_n=top_n))
    console.print_json(json.dumps(summary.to_dict()))


@app.command()
def status(
    watch: int = typer.Option(
        0, help="Refresh every N seconds (0 = print once), e.g. --watch 30"
    ),
) -> None:
    """Show local data/pipeline status and full-dataset download progress."""
    cfg = get_dataset_config()
    while True:
        console.clear()
        table = Table(title="RareAI status")
        table.add_column("Item")
        table.add_column("Status")
        checks = {
            "HF_TOKEN": "set" if hf_token() else "missing",
            "RAREAI_DATA_DIR": str(_data_root()),
            "raw VCF": str((cfg.paths.raw / "WGS_EX2312012_HGWCNDSX7.vcf.gz").exists()),
            "staging parquet": str((cfg.paths.staging / "variants.parquet").exists()),
            "phenotype curated": str(
                (cfg.paths.phenotypes / "patient_phenotype.yaml").exists()
            ),
            "gene panel": str(
                (cfg.paths.phenotypes / "phenotype_scores.parquet").exists()
            ),
            "ranking": str(
                (cfg.paths.results / "rankings" / "top_candidates.csv").exists()
            ),
            "submission": str((cfg.paths.results / "submission").exists()),
        }
        for key, value in checks.items():
            table.add_row(key, value)
        console.print(table)

        _print_download_progress(console)
        if watch <= 0:
            break
        try:
            time.sleep(watch)
        except KeyboardInterrupt:
            break


def _data_root() -> Path:
    from rareai.config import data_root

    return data_root()


def _print_download_progress(console: Console) -> None:
    from rareai.data.download import download_progress

    progress = download_progress()
    if not progress:
        return
    table = Table(title="Dataset download progress")
    for column in ("File", "Present", "Expected", "%"):
        table.add_column(column, justify="right" if column != "File" else "left")
    total_present = 0
    total_expected = 0
    for filename, info in progress.items():
        total_present += info["present"]
        if filename != "_inflight":
            total_expected += info["expected"]
        if filename == "_inflight":
            if info["present"] > 0:
                table.add_row("(in-flight transfer)", f"{info['present'] / 1e9:.2f} GB", "-", "-")
            continue
        if info["complete"]:
            mark = "[green]100.0%[/green]"
        else:
            mark = f"{info['percent']:.1f}%"
        table.add_row(
            filename[:48],
            f"{info['present'] / 1e9:.2f} GB",
            f"{info['expected'] / 1e9:.2f} GB",
            mark,
        )
    overall = 100.0 * total_present / total_expected if total_expected else 100.0
    console.print(table)
    console.print(
        f"Overall: [bold]{total_present / 1e9:.2f} / {total_expected / 1e9:.2f} GB "
        f"({overall:.1f}%)[/bold]"
    )


if __name__ == "__main__":
    app()
