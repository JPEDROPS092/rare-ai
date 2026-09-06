"""Experiment tracking: append-only JSONL run log (MLflow optional, Phase 6+)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rareai.config import (
    PROJECT_ROOT,
    get_annotation_config,
    get_dataset_config,
    get_ranking_config,
)
from rareai.log import get_logger

logger = get_logger(__name__)


def config_fingerprint() -> str:
    """Stable hash of all pipeline configuration (reproducibility audit)."""
    payload = {
        "dataset": get_dataset_config().model_dump(mode="json"),
        "ranking": get_ranking_config().model_dump(mode="json"),
        "annotation": get_annotation_config().model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def log_experiment(record: dict[str, Any], out_path: Path | str | None = None) -> Path:
    """Append one experiment record to ``results/benchmarks/experiments.jsonl``."""
    out_path = (
        Path(out_path)
        if out_path
        else PROJECT_ROOT / "results" / "benchmarks" / "experiments.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config_fingerprint": config_fingerprint(),
        **record,
    }
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    logger.info("Experiment logged: %s", out_path)
    return out_path
