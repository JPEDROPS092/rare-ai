"""Funnel statistics - every stage reports input/output/filtered/error counts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StageStats:
    """Counts for one pipeline stage."""

    stage: str
    input: int = 0
    output: int = 0
    filtered: int = 0
    errors: int = 0
    notes: dict[str, Any] = field(default_factory=dict)


class Funnel:
    """Accumulates StageStats across pipeline stages."""

    def __init__(self) -> None:
        self.started_at = datetime.now(UTC)
        self.stages: list[StageStats] = []

    def stage(self, name: str, *, input_count: int, output_count: int, **notes: Any) -> StageStats:
        stats = StageStats(
            stage=name,
            input=input_count,
            output=output_count,
            filtered=max(input_count - output_count, 0),
            notes=notes,
        )
        self.stages.append(stats)
        return stats

    def record(self, stats: StageStats) -> None:
        self.stages.append(stats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "stages": [asdict(s) for s in self.stages],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def render_markdown(self) -> str:
        lines = [
            "# Pipeline Quality Funnel",
            "",
            "| Stage | Input | Output | Filtered | Errors | Notes |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for s in self.stages:
            notes = "; ".join(f"{k}={v}" for k, v in s.notes.items()) if s.notes else ""
            lines.append(
                f"| {s.stage} | {s.input:,} | {s.output:,} | {s.filtered:,} "
                f"| {s.errors:,} | {notes} |"
            )
        lines.append("")
        return "\n".join(lines)
