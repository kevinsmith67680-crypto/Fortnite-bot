"""The controlled 2D research environment.

Two layers:

* :class:`World` - everything that is static for a map: the obstacle grid,
  obstacle rectangles, chest spawn points (converted from database world
  coordinates to grid coordinates) and the agent start position. Built once
  by :func:`build_world` and shared by every episode on that map.
* :class:`Environment` - one *episode*: which chests spawned, which are
  opened, the agent, the fixed-timestep clock and fictional token rewards.

Randomness never uses the wall clock or global state. Every random decision
comes from a ``random.Random`` seeded with a string derived from
``(seed, episode, chest id)``, so the same inputs always produce the same
episode regardless of strategy, chest order or platform.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from chest_hunter.config import SimulationSettings
from chest_hunter.models import ChestLocation
from chest_hunter.simulation.collision import (
    Cell,
    GridMap,
    Point,
    circle_collides,
    segment_clear,
)
from chest_hunter.simulation.entities import Chest, Obstacle, PlayerAgent
from chest_hunter.simulation.map_generator import MapGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorldTransform:
    """Converts database world coordinates to grid units and back.

    ``grid = (world - origin) / units_per_cell``. ``z`` is ignored: the
    simulator is 2D, so chests stacked vertically share a grid position.
    """

    units_per_cell: float
    origin_x: float = 0.0
    origin_y: float = 0.0

    def to_grid(self, x: float, y: float) -> Point:
        return ((x - self.origin_x) / self.units_per_cell,
                (y - self.origin_y) / self.units_per_cell)

    def to_world(self, gx: float, gy: float) -> Point:
        return (gx * self.units_per_cell + self.origin_x,
                gy * self.units_per_cell + self.origin_y)


@dataclass(frozen=True)
class ChestSpec:
    """A chest spawn point placed on the grid (static part of a chest)."""

    location_id: int
    name: str
    x: float
    y: float
    spawn_probability: float
    estimated_open_seconds: float

    @property
    def cell(self) -> Cell:
        return (math.floor(self.x), math.floor(self.y))


@dataclass
class World:
    """Static map data shared by all episodes on one map layout."""

    map_id: str
    map_seed: int
    grid: GridMap
    obstacles: tuple[Obstacle, ...]
    chests: tuple[ChestSpec, ...]
    start: Point
    transform: WorldTransform
    excluded_location_ids: tuple[int, ...] = ()

    @property
    def start_cell(self) -> Cell:
        return GridMap.cell_of(*self.start)

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height


def build_world(
    locations: Sequence[ChestLocation],
    settings: SimulationSettings,
    *,
    map_id: str,
    map_seed: int,
) -> World:
    """Place database chest locations on a procedurally generated grid.

    Disabled locations and locations outside the grid are excluded (their IDs
    are kept in ``excluded_location_ids`` for reporting).
    """
    transform = WorldTransform(settings.world_units_per_cell)
    width, height = settings.map_width, settings.map_height
    specs: list[ChestSpec] = []
    excluded: list[int] = []
    for location in sorted(locations, key=lambda loc: loc.id or 0):
        assert location.id is not None
        gx, gy = transform.to_grid(location.x, location.y)
        if not location.enabled or not (0.0 <= gx < width and 0.0 <= gy < height):
            excluded.append(location.id)
            continue
        specs.append(ChestSpec(location.id, location.location_name, gx, gy,
                               location.spawn_probability, location.estimated_open_seconds))
    if excluded:
        logger.info("Excluded %d chest location(s) that are disabled or outside the "
                    "%dx%d grid: %s", len(excluded), width, height, excluded)

    if settings.agent_start is None:
        start_cell: Cell = (width // 2, height // 2)
    else:
        sx, sy = transform.to_grid(*settings.agent_start)
        start_cell = (min(max(math.floor(sx), 1), width - 2),
                      min(max(math.floor(sy), 1), height - 2))

    protected = [spec.cell for spec in specs] + [start_cell]
    generated = MapGenerator(width, height, density=settings.obstacle_density,
                             seed=map_seed).generate(protected)
    start = GridMap.cell_center(start_cell)
    return World(
        map_id=map_id,
        map_seed=map_seed,
        grid=generated.grid,
        obstacles=generated.obstacles,
        chests=tuple(specs),
        start=start,
        transform=transform,
        excluded_location_ids=tuple(excluded),
    )


class InteractionResult(StrEnum):
    OPENED = "opened"
    NOT_PRESENT = "not_present"
    ALREADY_OPENED = "already_opened"
    OUT_OF_RANGE = "out_of_range"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    UNKNOWN_CHEST = "unknown_chest"


class TokenRewardModel:
    """Fictional simulation reward: a "token" granted for some opened chests.

    Kept separate from chest-opening logic. This is a synthetic test signal
    for reward detection/reporting and says nothing about any real game.
    """

    def __init__(self, probability: float, *, seed: int, episode: int) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("reward probability must be between 0 and 1")
        self.probability = probability
        self.seed = seed
        self.episode = episode
        self.awarded: list[int] = []

    def roll(self, chest_id: int) -> bool:
        if self.probability <= 0.0:
            return False
        rng = random.Random(f"chest-hunter:reward:{self.seed}:{self.episode}:{chest_id}")
        granted = rng.random() < self.probability
        if granted:
            self.awarded.append(chest_id)
        return granted


@dataclass
class EpisodeEvent:
    tick: int
    kind: str
    chest_id: int | None = None


class Environment:
    """One deterministic episode on a :class:`World`.

    The environment owns the fixed timestep (``tick`` counts steps of
    ``dt = 1 / tick_rate`` simulated seconds) and never reads the wall clock.
    """

    def __init__(self, world: World, settings: SimulationSettings, *, seed: int,
                 episode: int = 0) -> None:
        self.world = world
        self.settings = settings
        self.seed = seed
        self.episode = episode
        self.dt = settings.tick_seconds
        self.grid: GridMap = world.grid
        self.obstacles: list[Obstacle] = list(world.obstacles)
        self.chests: dict[int, Chest] = {}
        self.agent = self._new_agent()
        self.tick = 0
        self.rewards = TokenRewardModel(settings.reward_probability, seed=seed, episode=episode)
        self.events: list[EpisodeEvent] = []
        self._interaction_rngs: dict[int, random.Random] = {}
        self.reset(episode)

    # -- lifecycle ----------------------------------------------------------

    def reset(self, episode: int | None = None) -> None:
        """Start (or restart) an episode: respawn chests, reposition the agent."""
        if episode is not None:
            self.episode = episode
        self.grid = self.world.grid
        self.obstacles = list(self.world.obstacles)
        self.tick = 0
        self.events = []
        self._interaction_rngs = {}
        self.rewards = TokenRewardModel(self.settings.reward_probability,
                                        seed=self.seed, episode=self.episode)
        self.chests = {}
        for spec in self.world.chests:
            chest = Chest(
                id=spec.location_id,
                position_x=spec.x,
                position_y=spec.y,
                spawn_probability=spec.spawn_probability,
                location_name=spec.name,
                estimated_open_seconds=spec.estimated_open_seconds,
            )
            chest.reset(self._spawn_roll(spec.location_id, spec.spawn_probability))
            self.chests[chest.id] = chest
        self.agent = self._new_agent()

    def _spawn_roll(self, chest_id: int, probability: float) -> bool:
        rng = random.Random(f"chest-hunter:spawn:{self.seed}:{self.episode}:{chest_id}")
        return rng.random() < probability

    def _new_agent(self) -> PlayerAgent:
        sx, sy = self.world.start
        if circle_collides(self.grid, sx, sy, self.settings.player_radius):
            raise ValueError(f"Agent start position {self.world.start} is not walkable.")
        return PlayerAgent(
            position_x=sx,
            position_y=sy,
            interaction_radius=self.settings.interaction_radius,
            max_speed=self.settings.player_speed,
            acceleration=self.settings.player_acceleration,
            radius=self.settings.player_radius,
        )

    @property
    def time(self) -> float:
        """Simulated seconds elapsed in this episode."""
        return self.tick * self.dt

    def advance(self) -> None:
        self.tick += 1

    # -- queries ------------------------------------------------------------

    @property
    def present_count(self) -> int:
        return sum(1 for chest in self.chests.values() if chest.present)

    @property
    def opened_count(self) -> int:
        return sum(1 for chest in self.chests.values() if chest.opened)

    def can_interact(self, chest_id: int) -> bool:
        """Agent is within interaction range of the spawn point *and* can see it."""
        chest = self.chests.get(chest_id)
        if chest is None or not self.agent.in_interaction_range(chest):
            return False
        return segment_clear(self.grid, self.agent.position, chest.position, 0.0)

    def observe_chest(self, chest_id: int) -> bool | None:
        """Presence of a chest as perceived by the agent.

        Returns ``None`` when the agent cannot perceive the spawn point (out of
        interaction range or no line of sight), otherwise whether a chest is
        there. The agent never learns presence from a distance.
        """
        chest = self.chests.get(chest_id)
        if chest is None or not self.can_interact(chest_id):
            return None
        return chest.present

    # -- interaction ----------------------------------------------------------

    def try_open(self, chest_id: int, interaction_elapsed: float) -> InteractionResult:
        """Attempt to open a chest after ``interaction_elapsed`` simulated seconds.

        A chest opens only if it exists this episode, is not already open, the
        agent is within interaction range and the configured interaction time
        has elapsed. An optional deterministic failure probability simulates
        unsuccessful interactions.
        """
        chest = self.chests.get(chest_id)
        if chest is None:
            return InteractionResult.UNKNOWN_CHEST
        if not chest.present:
            return InteractionResult.NOT_PRESENT
        if chest.opened:
            return InteractionResult.ALREADY_OPENED
        if not self.can_interact(chest_id):
            return InteractionResult.OUT_OF_RANGE
        if interaction_elapsed + 1e-9 < self.settings.chest_interaction_seconds:
            return InteractionResult.INCOMPLETE
        failure_p = self.settings.interaction_failure_probability
        if failure_p > 0.0:
            rng = self._interaction_rngs.get(chest_id)
            if rng is None:
                rng = random.Random(f"chest-hunter:interact:{self.seed}:{self.episode}:{chest_id}")
                self._interaction_rngs[chest_id] = rng
            if rng.random() < failure_p:
                self.events.append(EpisodeEvent(self.tick, "interaction_failed", chest_id))
                return InteractionResult.FAILED
        chest.open()
        self.events.append(EpisodeEvent(self.tick, "opened", chest_id))
        return InteractionResult.OPENED

    def grant_reward(self, chest_id: int) -> bool:
        """Roll the separate fictional token reward for an opened chest."""
        chest = self.chests.get(chest_id)
        if chest is None or not chest.opened:
            return False
        granted = self.rewards.roll(chest_id)
        if granted:
            self.events.append(EpisodeEvent(self.tick, "reward", chest_id))
        return granted

    # -- dynamic obstacles (tests, GUI experiments) ---------------------------

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Block more cells during an episode (copy-on-write; the World is unchanged)."""
        if self.grid is self.world.grid:
            self.grid = self.grid.copy()
        self.grid.fill_rect(int(obstacle.position_x), int(obstacle.position_y),
                            int(obstacle.width), int(obstacle.height))
        self.obstacles.append(obstacle)
        self.events.append(EpisodeEvent(self.tick, "obstacle_added"))
