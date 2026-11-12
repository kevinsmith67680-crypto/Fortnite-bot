"""Controlled 2D research environment (grid world, entities, collision, rendering).

The Tkinter renderer lives in :mod:`chest_hunter.simulation.renderer` and is
imported lazily, so headless simulation never needs Tkinter or a display.
"""

from chest_hunter.simulation.collision import GridMap
from chest_hunter.simulation.entities import Chest, Obstacle, PlayerAgent
from chest_hunter.simulation.environment import (
    Environment,
    InteractionResult,
    World,
    WorldTransform,
    build_world,
)
from chest_hunter.simulation.map_generator import GeneratedMap, MapGenerator

__all__ = [
    "Chest",
    "Environment",
    "GeneratedMap",
    "GridMap",
    "InteractionResult",
    "MapGenerator",
    "Obstacle",
    "PlayerAgent",
    "World",
    "WorldTransform",
    "build_world",
]
