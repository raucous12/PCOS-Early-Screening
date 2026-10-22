"""Configuration loading: YAML defaults + CLI overrides.

Usage::

    cfg = load_config("config/default.yaml", overrides=["training.n_trials=10"])
    cfg.path("raw_dir")  # -> absolute Path
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised for missing or malformed configuration."""


def _coerce(value: str) -> Any:
    """Turn a CLI override string into a Python value using YAML rules."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_override(data: dict, dotted: str) -> None:
    if "=" not in dotted:
        raise ConfigError(f"Override '{dotted}' must look like key.sub=value")
    key, raw = dotted.split("=", 1)
    parts = key.strip().split(".")
    node = data
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = _coerce(raw)


@dataclass
class Config:
    data: dict
    root: Path = field(default=PROJECT_ROOT)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root / p).resolve()

    def path(self, name: str) -> Path:
        try:
            return self.resolve(self.data["paths"][name])
        except KeyError as exc:
            raise ConfigError(f"paths.{name} is not configured") from exc


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc
    data = copy.deepcopy(data)
    for item in overrides or []:
        _apply_override(data, item)
    for required in ("paths", "sources", "training"):
        if required not in data:
            raise ConfigError(f"Config is missing the '{required}' section")
    return Config(data=data)
