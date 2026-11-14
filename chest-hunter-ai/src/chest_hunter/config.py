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

Phase 1 keys (``database.path``, ``simulation.default_map``,
``simulation.default_spawn_probability``, ``simulation.default_open_seconds``
and ``logging.level``) are required. Every Phase 2 key is optional and falls
back to the documented default, so Phase 1 configuration files stay valid.
Units are given in :data:`FIELD_UNITS`.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from chest_hunter.exceptions import ConfigError, ValidationError
from chest_hunter.logging_config import VALID_LEVELS
from chest_hunter.models import (
    STRATEGY_NAMES,
    require_map_id,
    require_non_negative,
    require_probability,
)

CONFIG_ENV_VAR = "CHEST_HUNTER_CONFIG"
HOME_ENV_VAR = "CHEST_HUNTER_HOME"

# src/chest_hunter/config.py -> parents[2] is the project root in a checkout
# or an editable install.
_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


# ---------------------------------------------------------------------------
# Settings objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseSettings:
    path: Path


@dataclass(frozen=True)
class SimulationSettings:
    """Phase 1 defaults plus the 2D simulator parameters (all with defaults)."""

    default_map: str
    default_spawn_probability: float
    default_open_seconds: float
    map_width: int = 100
    map_height: int = 100
    cell_size: int = 8
    world_units_per_cell: float = 10.0
    tick_rate: int = 30
    render_rate: int = 15
    default_seed: int = 42
    max_ticks: int = 100_000
    player_speed: float = 5.0
    player_acceleration: float = 12.0
    player_radius: float = 0.3
    agent_start: tuple[float, float] | None = None
    interaction_radius: float = 1.5
    chest_interaction_seconds: float = 2.0
    interaction_failure_probability: float = 0.0
    reward_probability: float = 0.0
    obstacle_density: float = 0.12
    decoration_density: float = 0.004

    @property
    def tick_seconds(self) -> float:
        """Fixed simulation timestep in simulated seconds."""
        return 1.0 / self.tick_rate


@dataclass(frozen=True)
class NavigationSettings:
    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True
    cache_paths: bool = True
    default_strategy: str = "weighted"
    stuck_timeout_seconds: float = 5.0
    smooth_paths: bool = True
    lookahead_depth: int = 2


@dataclass(frozen=True)
class AgentSettings:
    max_recovery_attempts: int = 3
    arrival_tolerance: float = 0.5
    observation_batch_size: int = 10


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool = True
    default_width: int = 640
    default_height: int = 360
    view_width_cells: int = 40
    detector: str = "hybrid"
    confidence_threshold: float = 0.85
    max_tracks: int = 100
    track_expiration_seconds: float = 3.0
    track_memory_seconds: float = 600.0
    use_regions_of_interest: bool = True
    noise_std: float = 6.0
    lighting_variation: float = 0.08
    frame_dropout: float = 0.0
    rate_travel: float = 2.0
    rate_explore: float = 5.0
    rate_tracking: float = 10.0
    rate_verify: float = 10.0
    rate_idle: float = 1.0
    template_dir: str = "assets/templates"


@dataclass(frozen=True)
class LocalizationSettings:
    smoothing_alpha: float = 0.4
    max_missing_observations: int = 5
    min_confidence: float = 0.5
    dead_reckoning_seconds: float = 2.0


@dataclass(frozen=True)
class VisualNavigationSettings:
    exploration_strategy: str = "frontier"
    replan_on_obstacle_discovery: bool = True
    maximum_recovery_attempts: int = 3
    use_spawn_priors: bool = True
    prior_weight: float = 300.0
    stop_when_spawn_points_resolved: bool = True
    unknown_cell_cost: float = 0.5
    empty_confirmations: int = 2


@dataclass(frozen=True)
class VisualDebugSettings:
    enabled: bool = True
    show_detections: bool = True
    show_object_tracks: bool = True
    show_occupancy_grid: bool = False
    show_localization: bool = True
    show_camera_view: bool = True


@dataclass(frozen=True)
class UnrealBridgeSettings:
    """Phase 4: how to reach the research server (Unreal plugin or reference server)."""

    host: str = "127.0.0.1"
    port: int = 47820
    timeout_seconds: float = 30.0
    connect_attempts: int = 5


@dataclass(frozen=True)
class Simulation3DSettings:
    """Phase 4: 3D episode and agent parameters."""

    default_map: str = "research_arena_3d"
    backend: str = "unreal"
    observation_width: int = 640
    observation_height: int = 360
    fov_deg: float = 90.0
    graphics_quality: str = "low"
    max_ticks: int = 18000
    decision_ticks: int = 2
    detector: str = "hybrid"
    odometry_drift: float = 0.005
    noise_std: float = 5.0
    reward_probability: float = 0.3
    interaction_failure_probability: float = 0.0
    template_dir: str = "assets/templates3d"


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
    navigation: NavigationSettings = field(default_factory=NavigationSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    localization: LocalizationSettings = field(default_factory=LocalizationSettings)
    visual_navigation: VisualNavigationSettings = field(default_factory=VisualNavigationSettings)
    visual_debug: VisualDebugSettings = field(default_factory=VisualDebugSettings)
    unreal_bridge: UnrealBridgeSettings = field(default_factory=UnrealBridgeSettings)
    simulation_3d: Simulation3DSettings = field(default_factory=Simulation3DSettings)

    def template_dir(self) -> Path:
        """Absolute directory of the perception templates."""
        return _resolve_path(self.vision.template_dir, self.base_dir)

    def template_dir_3d(self) -> Path:
        """Absolute directory of the Phase 4 multi-view chest templates."""
        return _resolve_path(self.simulation_3d.template_dir, self.base_dir)

    def with_database_path(self, path: str | os.PathLike[str]) -> Settings:
        """Return a copy that uses ``path`` (resolved against ``base_dir``) as the database."""
        return replace(self, database=DatabaseSettings(path=_resolve_path(path, self.base_dir)))


# ---------------------------------------------------------------------------
# Declarative schema
# ---------------------------------------------------------------------------

_REQUIRED: Final = object()
Parser = Callable[[str, Any], Any]


def _int_range(low: int, high: int) -> Parser:
    def parse(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"{name} must be an integer, got {value!r}.")
        if not low <= value <= high:
            raise ValidationError(f"{name} must be between {low} and {high}, got {value!r}.")
        return value

    return parse


def _float_range(low: float, high: float, *, low_open: bool = False,
                 high_open: bool = False) -> Parser:
    def parse(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number, got {value!r}.")
        number = float(value)
        too_low = number <= low if low_open else number < low
        too_high = number >= high if high_open else number > high
        if not math.isfinite(number) or too_low or too_high:
            left, right = ("(" if low_open else "["), (")" if high_open else "]")
            raise ValidationError(
                f"{name} must be in the range {left}{low:g}, {high:g}{right}, got {value!r}."
            )
        return number

    return parse


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be true or false, got {value!r}.")
    return value


def _choice(*choices: str) -> Parser:
    def parse(name: str, value: Any) -> str:
        if not isinstance(value, str) or value.strip().lower() not in choices:
            raise ValidationError(f"{name} must be one of {', '.join(choices)}; got {value!r}.")
        return value.strip().lower()

    return parse


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string, got {value!r}.")
    return value.strip()


def _optional_point(name: str, value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
        or not all(math.isfinite(float(v)) for v in value)
    ):
        raise ValidationError(f"{name} must be null or [x, y] with finite numbers, got {value!r}.")
    return (float(value[0]), float(value[1]))


def _map_id(name: str, value: Any) -> str:
    return require_map_id(value)


def _probability(name: str, value: Any) -> float:
    return require_probability(name, value)


def _non_negative(name: str, value: Any) -> float:
    return require_non_negative(name, value)


# (key, default or _REQUIRED, parser)
_SIMULATION_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("default_map", _REQUIRED, _map_id),
    ("default_spawn_probability", _REQUIRED, _probability),
    ("default_open_seconds", _REQUIRED, _non_negative),
    ("map_width", 100, _int_range(10, 1000)),
    ("map_height", 100, _int_range(10, 1000)),
    ("cell_size", 8, _int_range(2, 64)),
    ("world_units_per_cell", 10.0, _float_range(0.0, 1e6, low_open=True)),
    ("tick_rate", 30, _int_range(1, 240)),
    ("render_rate", 15, _int_range(1, 60)),
    ("default_seed", 42, _int_range(0, 2**63 - 1)),
    ("max_ticks", 100_000, _int_range(1, 10**9)),
    ("player_speed", 5.0, _float_range(0.0, 100.0, low_open=True)),
    ("player_acceleration", 12.0, _float_range(0.0, 1000.0, low_open=True)),
    ("player_radius", 0.3, _float_range(0.0, 0.5, low_open=True, high_open=True)),
    ("agent_start", None, _optional_point),
    ("interaction_radius", 1.5, _float_range(0.0, 50.0, low_open=True)),
    ("chest_interaction_seconds", 2.0, _float_range(0.0, 3600.0)),
    ("interaction_failure_probability", 0.0, _float_range(0.0, 1.0)),
    ("reward_probability", 0.0, _float_range(0.0, 1.0)),
    ("obstacle_density", 0.12, _float_range(0.0, 0.4)),
    ("decoration_density", 0.004, _float_range(0.0, 0.05)),
)

_NAVIGATION_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("allow_diagonal", True, _boolean),
    ("prevent_corner_cutting", True, _boolean),
    ("cache_paths", True, _boolean),
    ("default_strategy", "weighted", _choice(*STRATEGY_NAMES)),
    ("stuck_timeout_seconds", 5.0, _float_range(0.0, 3600.0, low_open=True)),
    ("smooth_paths", True, _boolean),
    ("lookahead_depth", 2, _int_range(1, 3)),
)

_AGENT_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("max_recovery_attempts", 3, _int_range(0, 20)),
    ("arrival_tolerance", 0.5, _float_range(0.0, 50.0, low_open=True)),
    ("observation_batch_size", 10, _int_range(1, 10_000)),
)

_VISION_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("enabled", True, _boolean),
    ("default_width", 640, _int_range(64, 1920)),
    ("default_height", 360, _int_range(36, 1080)),
    ("view_width_cells", 40, _int_range(10, 100)),
    ("detector", "hybrid", _choice("color", "template", "hybrid")),
    ("confidence_threshold", 0.85, _float_range(0.0, 1.0, low_open=True)),
    ("max_tracks", 100, _int_range(1, 10_000)),
    ("track_expiration_seconds", 3.0, _float_range(0.0, 3600.0, low_open=True)),
    ("track_memory_seconds", 600.0, _float_range(0.0, 86_400.0, low_open=True)),
    ("use_regions_of_interest", True, _boolean),
    ("noise_std", 6.0, _float_range(0.0, 50.0)),
    ("lighting_variation", 0.08, _float_range(0.0, 0.5)),
    ("frame_dropout", 0.0, _float_range(0.0, 0.9)),
    ("rate_travel", 2.0, _float_range(0.1, 60.0)),
    ("rate_explore", 5.0, _float_range(0.1, 60.0)),
    ("rate_tracking", 10.0, _float_range(0.1, 60.0)),
    ("rate_verify", 10.0, _float_range(0.1, 60.0)),
    ("rate_idle", 1.0, _float_range(0.0, 60.0)),
    ("template_dir", "assets/templates", lambda n, v: _non_empty_string(n, v)),
)

_LOCALIZATION_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("smoothing_alpha", 0.4, _float_range(0.0, 1.0, low_open=True)),
    ("max_missing_observations", 5, _int_range(1, 1000)),
    ("min_confidence", 0.5, _float_range(0.0, 1.0)),
    ("dead_reckoning_seconds", 2.0, _float_range(0.0, 60.0, low_open=True)),
)

_VISUAL_NAVIGATION_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("exploration_strategy", "frontier", _choice("frontier")),
    ("replan_on_obstacle_discovery", True, _boolean),
    ("maximum_recovery_attempts", 3, _int_range(0, 20)),
    ("use_spawn_priors", True, _boolean),
    ("prior_weight", 300.0, _float_range(0.0, 1e6)),
    ("stop_when_spawn_points_resolved", True, _boolean),
    ("unknown_cell_cost", 0.5, _float_range(0.0, 100.0)),
    ("empty_confirmations", 2, _int_range(1, 50)),
)

_VISUAL_DEBUG_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("enabled", True, _boolean),
    ("show_detections", True, _boolean),
    ("show_object_tracks", True, _boolean),
    ("show_occupancy_grid", False, _boolean),
    ("show_localization", True, _boolean),
    ("show_camera_view", True, _boolean),
)

_UNREAL_BRIDGE_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("host", "127.0.0.1", _non_empty_string),
    ("port", 47820, _int_range(1, 65535)),
    ("timeout_seconds", 30.0, _float_range(0.5, 600.0)),
    ("connect_attempts", 5, _int_range(1, 50)),
)

_SIMULATION_3D_FIELDS: tuple[tuple[str, Any, Parser], ...] = (
    ("default_map", "research_arena_3d", _map_id),
    ("backend", "unreal", _choice("unreal", "reference")),
    ("observation_width", 640, _int_range(320, 800)),
    ("observation_height", 360, _int_range(180, 450)),
    ("fov_deg", 90.0, _float_range(60.0, 110.0)),
    ("graphics_quality", "low", _choice("low", "medium", "high")),
    ("max_ticks", 18000, _int_range(30, 10_000_000)),
    ("decision_ticks", 2, _int_range(1, 30)),
    ("detector", "hybrid", _choice("color", "template", "hybrid")),
    ("odometry_drift", 0.005, _float_range(0.0, 0.2)),
    ("noise_std", 5.0, _float_range(0.0, 40.0)),
    ("reward_probability", 0.3, _probability),
    ("interaction_failure_probability", 0.0, _probability),
    ("template_dir", "assets/templates3d", _non_empty_string),
)

SUPPORTED_FRAME_SIZES: Final = ((320, 180), (480, 270), (640, 360), (800, 450))

FIELD_UNITS: Final[dict[str, str]] = {
    "simulation.map_width": "grid cells",
    "simulation.map_height": "grid cells",
    "simulation.cell_size": "screen pixels per cell (GUI only)",
    "simulation.world_units_per_cell": "database coordinate units per grid cell",
    "simulation.tick_rate": "fixed simulation ticks per simulated second",
    "simulation.render_rate": "GUI frames per real second",
    "simulation.max_ticks": "ticks per episode before it is stopped",
    "simulation.player_speed": "grid cells per simulated second",
    "simulation.player_acceleration": "grid cells per simulated second squared",
    "simulation.player_radius": "grid cells (collision circle)",
    "simulation.agent_start": "database coordinate units [x, y]; null = map centre",
    "simulation.interaction_radius": "grid cells",
    "simulation.chest_interaction_seconds": "simulated seconds to open a chest",
    "simulation.interaction_failure_probability": "probability 0-1 per attempt",
    "simulation.reward_probability": "probability 0-1 of a fictional token per opening",
    "simulation.obstacle_density": "target fraction 0-0.4 of blocked cells",
    "simulation.decoration_density": "walkable gold props per cell (visual distractors)",
    "navigation.stuck_timeout_seconds": "simulated seconds without progress",
    "navigation.lookahead_depth": "chests considered per decision (lookahead strategy)",
    "agent.max_recovery_attempts": "recoveries per target before abandoning it",
    "agent.arrival_tolerance": "grid cells",
    "agent.observation_batch_size": "observations per database transaction",
    "vision.default_width/default_height": "camera frame size in pixels (320x180 .. 800x450)",
    "vision.view_width_cells": "grid cells visible across the frame (sets pixels per cell)",
    "vision.confidence_threshold": "minimum detection score 0-1 (template-matching score)",
    "vision.track_expiration_seconds": "simulated seconds an in-view track may go unseen",
    "vision.track_memory_seconds": "simulated seconds an out-of-view track is remembered",
    "vision.noise_std": "camera pixel noise standard deviation (0-255 intensity levels)",
    "vision.lighting_variation": "relative per-frame brightness variation",
    "vision.frame_dropout": "probability 0-0.9 that a capture fails",
    "vision.rate_*": "visual analyses per simulated second in each situation",
    "localization.smoothing_alpha": "weight 0-1 of a new visual fix vs the prediction",
    "localization.dead_reckoning_seconds": "seconds after which pose confidence reaches 0",
    "visual_navigation.prior_weight": "information-gain units per unit of spawn probability",
    "visual_navigation.unknown_cell_cost": "extra A* cost for entering an unexplored cell",
    "visual_navigation.empty_confirmations": "analyses showing an empty spawn point before "
                                             "it is recorded as absent",
}


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


def _section(data: dict[str, Any], name: str, *, required: bool) -> dict[str, Any]:
    section = data.get(name)
    if section is None and not required:
        return {}
    if not isinstance(section, dict):
        raise ValidationError(f"missing or invalid '{name}' section (expected an object).")
    return section


def _parse_fields(
    name: str, section: dict[str, Any], fields: tuple[tuple[str, Any, Parser], ...]
) -> dict[str, Any]:
    known = {key for key, _, _ in fields}
    unknown = sorted(section.keys() - known)
    if unknown:
        raise ValidationError(f"'{name}' has unknown keys: {', '.join(unknown)}.")
    missing = sorted(key for key, default, _ in fields
                     if default is _REQUIRED and key not in section)
    if missing:
        raise ValidationError(f"'{name}' is missing: {', '.join(missing)}.")
    values: dict[str, Any] = {}
    for key, default, parse in fields:
        if key not in section:
            values[key] = default
            continue
        try:
            values[key] = parse(key, section[key])
        except ValidationError as exc:
            raise ValidationError(f"'{name}': {exc}") from exc
    return values


def _parse_settings(data: Any, *, config_path: Path, base_dir: Path) -> Settings:
    if not isinstance(data, dict):
        raise ValidationError("the top level must be a JSON object.")

    # Unknown top-level sections are tolerated so later phases can add their
    # own settings; unknown keys inside known sections are rejected (typos).
    database = _section(data, "database", required=True)
    simulation = _section(data, "simulation", required=True)
    logging_section = _section(data, "logging", required=True)
    navigation = _section(data, "navigation", required=False)
    agent = _section(data, "agent", required=False)
    vision_section = _section(data, "vision", required=False)
    localization_section = _section(data, "localization", required=False)
    visual_nav_section = _section(data, "visual_navigation", required=False)
    visual_debug_section = _section(data, "visual_debug", required=False)
    bridge_section = _section(data, "unreal_bridge", required=False)
    sim3d_section = _section(data, "simulation_3d", required=False)

    db_values = _parse_fields("database", database, (("path", _REQUIRED, lambda n, v: v),))
    db_path = db_values["path"]
    if not isinstance(db_path, str) or not db_path.strip():
        raise ValidationError("'database.path' must be a non-empty string.")

    level_values = _parse_fields("logging", logging_section,
                                 (("level", _REQUIRED, lambda n, v: v),))
    level = level_values["level"]
    if not isinstance(level, str) or level.strip().upper() not in VALID_LEVELS:
        raise ValidationError(
            f"'logging.level' must be one of {', '.join(VALID_LEVELS)}; got {level!r}."
        )

    sim = SimulationSettings(**_parse_fields("simulation", simulation, _SIMULATION_FIELDS))
    nav = NavigationSettings(**_parse_fields("navigation", navigation, _NAVIGATION_FIELDS))
    agent_settings = AgentSettings(**_parse_fields("agent", agent, _AGENT_FIELDS))

    vision = VisionSettings(**_parse_fields("vision", vision_section, _VISION_FIELDS))
    localization = LocalizationSettings(
        **_parse_fields("localization", localization_section, _LOCALIZATION_FIELDS))
    visual_navigation = VisualNavigationSettings(
        **_parse_fields("visual_navigation", visual_nav_section, _VISUAL_NAVIGATION_FIELDS))
    visual_debug = VisualDebugSettings(
        **_parse_fields("visual_debug", visual_debug_section, _VISUAL_DEBUG_FIELDS))
    unreal_bridge = UnrealBridgeSettings(
        **_parse_fields("unreal_bridge", bridge_section, _UNREAL_BRIDGE_FIELDS))
    if unreal_bridge.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValidationError("'unreal_bridge.host' must be a loopback address (127.0.0.1, "
                              "localhost or ::1): the research bridge never leaves this machine.")
    simulation_3d = Simulation3DSettings(
        **_parse_fields("simulation_3d", sim3d_section, _SIMULATION_3D_FIELDS))
    if (simulation_3d.observation_width, simulation_3d.observation_height) \
            not in SUPPORTED_FRAME_SIZES:
        sizes = ", ".join(f"{w}x{h}" for w, h in SUPPORTED_FRAME_SIZES)
        raise ValidationError(
            f"'simulation_3d' observation size {simulation_3d.observation_width}x"
            f"{simulation_3d.observation_height} is not supported; use one of {sizes}.")
    if (vision.default_width, vision.default_height) not in SUPPORTED_FRAME_SIZES:
        sizes = ", ".join(f"{w}x{h}" for w, h in SUPPORTED_FRAME_SIZES)
        raise ValidationError(
            f"'vision' frame size {vision.default_width}x{vision.default_height} is not "
            f"supported; use one of {sizes}.")
    if vision.default_width % vision.view_width_cells:
        raise ValidationError(
            "'vision.default_width' must be a multiple of 'vision.view_width_cells' "
            "(integer pixels per cell).")

    if agent_settings.arrival_tolerance > sim.interaction_radius:
        raise ValidationError(
            "'agent.arrival_tolerance' must not exceed 'simulation.interaction_radius' "
            f"({agent_settings.arrival_tolerance:g} > {sim.interaction_radius:g})."
        )

    return Settings(
        database=DatabaseSettings(path=_resolve_path(db_path.strip(), base_dir)),
        simulation=sim,
        logging=LoggingSettings(level=level.strip().upper()),
        config_path=config_path,
        base_dir=base_dir,
        navigation=nav,
        agent=agent_settings,
        vision=vision,
        localization=localization,
        visual_navigation=visual_navigation,
        visual_debug=visual_debug,
        unreal_bridge=unreal_bridge,
        simulation_3d=simulation_3d,
    )


def _resolve_path(value: str | os.PathLike[str], base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
