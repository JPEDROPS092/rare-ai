"""Configuration loading.

All thresholds, paths, weights and resource URLs live in ``configs/*.yaml``
and ``.env``. Nothing is hard-coded in pipeline code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"

load_dotenv(PROJECT_ROOT / ".env")

# Force the resumable CDN HTTP path for HF downloads unless overridden in .env.
# Must run before huggingface_hub is first imported (this module loads first).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class ResourceSpec(BaseModel):
    """External annotation resource specification."""

    url: str
    description: str = ""


class Paths(BaseModel):
    raw: Path
    staging: Path
    normalized: Path
    annotations: Path
    phenotypes: Path
    features: Path
    results: Path

    def resolve(self) -> Paths:
        """Anchor relative paths: data dirs at the data root, results at project.

        The data root is the project directory by default, or ``RAREAI_DATA_DIR``
        when set (e.g. an external disk for the ~85 GB full dataset). A configured
        path like ``data/raw`` becomes ``<data_root>/raw``.
        """
        root = data_root()
        fields = {}
        for name, value in self.model_dump().items():
            path = Path(value)
            if path.is_absolute():
                fields[name] = path
                continue
            parts = path.parts
            if parts and parts[0] == "data":
                fields[name] = root.joinpath(*parts[1:]) if len(parts) > 1 else root
            elif name == "results":
                fields[name] = PROJECT_ROOT / path
            else:
                fields[name] = PROJECT_ROOT / path
        return Paths(**fields)


def data_root() -> Path:
    """Root directory for all private data (override via RAREAI_DATA_DIR)."""
    override = os.environ.get("RAREAI_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "data"


class QCConfig(BaseModel):
    require_pass: bool = True
    min_dp: int = 10
    min_gq: int = 20
    min_alt_ad: int = 3
    het_ab_range: tuple[float, float] = (0.10, 0.90)


class DownloadConfig(BaseModel):
    retries: int = 12
    backoff_s: float = 20.0


class DatasetConfig(BaseModel):
    repo_id: str
    proband_id: str | None = None
    tracks: dict[str, Any]
    paths: Paths
    qc: QCConfig = QCConfig()
    resources: dict[str, ResourceSpec]
    genome_build: str = "GRCh38"
    download: DownloadConfig = DownloadConfig()


class BaselineWeights(BaseModel):
    rarity: float = 0.20
    pathogenicity: float = 0.20
    phenotype_similarity: float = 0.25
    clinical_evidence: float = 0.10
    gene_disease: float = 0.10
    inheritance: float = 0.05
    mosaicism: float = 0.10


class BaselineConfig(BaseModel):
    weights: BaselineWeights = BaselineWeights()
    neutral_values: dict[str, float] = Field(default_factory=dict)


class GenePanelConfig(BaseModel):
    size: int = 1500
    min_similarity: float = 0.05


class SubmissionConfig(BaseModel):
    max_rows: int = 10
    team_name: str = "rareai"
    model_name: str = "baseline-v1"


class RankingConfig(BaseModel):
    baseline: BaselineConfig = BaselineConfig()
    tie_breakers: list[str] = Field(default_factory=lambda: ["chrom", "pos", "ref", "alt"])
    top_n: int = 100
    gene_panel: GenePanelConfig = GenePanelConfig()
    funnel: dict[str, Any] = Field(
        default_factory=lambda: {"rarity_max_af": 0.02, "keep_pathogenic_above_af": True}
    )
    submission: SubmissionConfig = SubmissionConfig()
    ensemble: dict[str, Any] = Field(default_factory=dict)


class VEPConfig(BaseModel):
    enabled: bool = True
    url: str = "https://rest.ensembl.org/vep/homo_sapiens/region"
    batch_size: int = 200
    max_variants: int | None = None
    requests_per_second: float = 2.0
    timeout_s: float = 60.0
    retries: int = 3
    cache_file: Path = Path("data/annotations/rest_cache/vep_cache.jsonl")


class GnomadConfig(BaseModel):
    enabled: bool = False
    url: str = "https://gnomad.broadinstitute.org/api"
    cache_file: Path = Path("data/annotations/rest_cache/gnomad_cache.jsonl")


class ClinvarTerms(BaseModel):
    pathogenic_terms: list[str] = Field(default_factory=list)
    likely_pathogenic_terms: list[str] = Field(default_factory=list)
    vus_terms: list[str] = Field(default_factory=list)
    benign_terms: list[str] = Field(default_factory=list)


class AnnotationConfig(BaseModel):
    vep: VEPConfig = VEPConfig()
    gnomad_api: GnomadConfig = GnomadConfig()
    clinvar: ClinvarTerms = ClinvarTerms()
    consequence_severity: dict[str, float] = Field(default_factory=dict)
    splice_window_bp: int = 2


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_dataset_config() -> DatasetConfig:
    cfg = DatasetConfig(**_load_yaml("dataset.yaml"))
    return cfg.model_copy(update={"paths": cfg.paths.resolve()})


@lru_cache(maxsize=1)
def get_ranking_config() -> RankingConfig:
    return RankingConfig(**_load_yaml("ranking.yaml"))


@lru_cache(maxsize=1)
def get_annotation_config() -> AnnotationConfig:
    cfg = AnnotationConfig(**_load_yaml("annotation.yaml"))
    vep = cfg.vep.model_copy(update={"cache_file": _anchor(cfg.vep.cache_file)})
    gnomad = cfg.gnomad_api.model_copy(update={"cache_file": _anchor(cfg.gnomad_api.cache_file)})
    return cfg.model_copy(update={"vep": vep, "gnomad_api": gnomad})


def _anchor(path: Path) -> Path:
    """Anchor config-relative paths, honoring the data-root override."""
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        root = data_root()
        return root.joinpath(*parts[1:]) if len(parts) > 1 else root
    return PROJECT_ROOT / path


def hf_token() -> str | None:
    """Hugging Face access token from the environment (never hard-coded)."""
    token = os.environ.get("HF_TOKEN")
    return token if token else None


def results_dir() -> Path:
    return get_dataset_config().paths.results.resolve()
