"""Typed configuration loading.

The configuration is a YAML file holding every prespecified study parameter
(item identifiers, thresholds, drug lists, longitudinal windows). Paths support
environment-variable expansion so that the data location is not hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    """Recursively expand environment variables and user home in string leaves."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class Config:
    """Container around the parsed configuration dictionary."""

    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def data_root(self) -> Path:
        root = self.raw.get("data_root", "")
        if not root or root.startswith("${"):
            raise ValueError(
                "data_root is not set. Provide DATA_ROOT in the environment "
                "or set data_root in the configuration file."
            )
        return Path(root)

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", "outputs"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("random_seed", 0))

    def mimic_hosp(self) -> Path:
        return self.data_root / self.raw["cohorts"]["mimic"]["hosp_dir"]

    def mimic_icu(self) -> Path:
        return self.data_root / self.raw["cohorts"]["mimic"]["icu_dir"]

    def eicu_dir(self) -> Path:
        return self.data_root / self.raw["cohorts"]["eicu"]["dir"]


def load_config(path: str | Path) -> Config:
    """Read and validate the YAML configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(raw=_expand(raw), path=path)
