from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class ProjectPaths:
    data_raw: Path
    data_processed: Path
    outputs: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "ProjectPaths":
        p = cfg.get("paths", {})
        return ProjectPaths(
            data_raw=Path(p.get("data_raw", "data/raw")),
            data_processed=Path(p.get("data_processed", "data/processed")),
            outputs=Path(p.get("outputs", "outputs")),
        )

    def ensure(self) -> None:
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
