"""Configuration loading and validation.

Default files are located relative to the *project root* (the directory that
contains ``pyproject.toml``, ``config/`` and ``data/``), never relative to the
current working directory. Relative paths inside a configuration file are
resolved against ``base_dir``, which defaults to the project root.

Lookup order for the configuration file:

1. an explicit ``path`` argument (e.g. the CLI ``--config`` option);
2. the ``CHEST_HUNTER_CONFIG`` environment variable;
3. ``<project root>/config/settings.json``.

The project root itself can be overridden with ``CHEST_HUNTER_HOME`` (useful
for non-editable installs).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chest_hunter.exceptions import ConfigError, ValidationError
from chest_hunter.logging_config import VALID_LEVELS
from chest_hunter.models import (
    require_map_id,
    require_non_negative,
    require_probability,
)

CONFIG_ENV_VAR = "CHEST_HUNTER_CONFIG"
HOME_ENV_VAR = "CHEST_HUNTER_HOME"

# src/chest_hunter/config.py -> parents[2] is the project root in a checkout
# or an editable install.
_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SECTION_KEYS: dict[str, frozenset[str]] = {
    "database": frozenset({"path"}),
    "simulation": frozenset(
        {"default_map", "default_spawn_probability", "default_open_seconds"}
    ),
    "logging": frozenset({"level"}),
}


def project_root() -> Path:
    """Return the project root (``CHEST_HUNTER_HOME`` overrides the default)."""
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _PACKAGE_PROJECT_ROOT


def default_config_path() -> Path:
    """Return the configuration path used when none is given explicitly."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "config" / "settings.json"


@dataclass(frozen=True)
class DatabaseSettings:
    path: Path


@dataclass(frozen=True)
class SimulationSettings:
    default_map: str
    default_spawn_probability: float
    default_open_seconds: float


@dataclass(frozen=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True)
class Settings:
    """Validated application settings. All paths are absolute."""

    database: DatabaseSettings
    simulation: SimulationSettings
    logging: LoggingSettings
    config_path: Path
    base_dir: Path

    def with_database_path(self, path: str | os.PathLike[str]) -> Settings:
        """Return a copy that uses ``path`` (resolved against ``base_dir``) as the database."""
        resolved = _resolve_path(path, self.base_dir)
        return Settings(
            database=DatabaseSettings(path=resolved),
            simulation=self.simulation,
            logging=self.logging,
            config_path=self.config_path,
            base_dir=self.base_dir,
        )


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Settings:
    """Load and validate a JSON configuration file.

    :param path: configuration file; defaults to :func:`default_config_path`.
    :param base_dir: directory for resolving relative paths in the file;
        defaults to :func:`project_root`.
    :raises ConfigError: with a message naming the file and the bad field.
    """
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else project_root()

    if not config_path.is_file():
        raise ConfigError(
            f"Configuration file not found: {config_path}. Pass --config PATH, set "
            f"{CONFIG_ENV_VAR}, or install the project in editable mode (pip install -e .)."
        )
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file {config_path}: {exc}") from exc
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Configuration file {config_path} is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc

    try:
        return _parse_settings(data, config_path=config_path, base_dir=root)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}: {exc}") from exc


def _parse_settings(data: Any, *, config_path: Path, base_dir: Path) -> Settings:
    if not isinstance(data, dict):
        raise ValidationError("the top level must be a JSON object.")

    # Unknown top-level sections are tolerated so later phases can add their
    # own settings; unknown keys inside known sections are rejected (typos).
    sections: dict[str, dict[str, Any]] = {}
    for name, allowed in _SECTION_KEYS.items():
        section = data.get(name)
        if not isinstance(section, dict):
            raise ValidationError(f"missing or invalid '{name}' section (expected an object).")
        missing = sorted(allowed - section.keys())
        if missing:
            raise ValidationError(f"'{name}' is missing: {', '.join(missing)}.")
        unknown = sorted(section.keys() - allowed)
        if unknown:
            raise ValidationError(f"'{name}' has unknown keys: {', '.join(unknown)}.")
        sections[name] = section

    db_path = sections["database"]["path"]
    if not isinstance(db_path, str) or not db_path.strip():
        raise ValidationError("'database.path' must be a non-empty string.")

    sim = sections["simulation"]
    try:
        default_map = require_map_id(sim["default_map"])
        probability = require_probability(
            "default_spawn_probability", sim["default_spawn_probability"]
        )
        open_seconds = require_non_negative("default_open_seconds", sim["default_open_seconds"])
    except ValidationError as exc:
        raise ValidationError(f"'simulation': {exc}") from exc

    level = sections["logging"]["level"]
    if not isinstance(level, str) or level.strip().upper() not in VALID_LEVELS:
        raise ValidationError(
            f"'logging.level' must be one of {', '.join(VALID_LEVELS)}; got {level!r}."
        )

    return Settings(
        database=DatabaseSettings(path=_resolve_path(db_path.strip(), base_dir)),
        simulation=SimulationSettings(
            default_map=default_map,
            default_spawn_probability=probability,
            default_open_seconds=open_seconds,
        ),
        logging=LoggingSettings(level=level.strip().upper()),
        config_path=config_path,
        base_dir=base_dir,
    )


def _resolve_path(value: str | os.PathLike[str], base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
