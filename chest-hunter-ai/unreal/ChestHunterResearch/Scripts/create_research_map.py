"""Create the (nearly empty) research map inside the Unreal Editor.

Run once after the C++ module has been built:

    UnrealEditor-Cmd ChestHunterResearch.uproject -run=pythonscript \
        -script="<absolute path>/Scripts/create_research_map.py" -unattended -nosplash

(Requires the "Python Editor Script Plugin", enabled in ChestHunterResearch.uproject.)

The map only holds what cannot be spawned at runtime: a NavMeshBoundsVolume that covers the
arena (the navigation mesh itself is generated at runtime because RuntimeGeneration=Dynamic),
a PlayerStart and the lighting. All arena geometry and the chests are spawned by
AResearchGameMode from Content/ResearchData/research_arena_3d.json at RESET_EPISODE.

STATUS: written against the UE 5.4 Python API but NOT executed in this repository's CI
(no Unreal Engine installation is available there).
"""

from __future__ import annotations

import json
import os

import unreal  # type: ignore[import-not-found]  # only available inside the Unreal Editor

MAP_PATH = "/Game/Maps/ResearchArena3D"
CM = 100.0


def _layout() -> dict:
    content = unreal.Paths.project_content_dir()
    path = os.path.join(content, "ResearchData", "research_arena_3d.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    layout = _layout()
    x0, y0, x1, y1 = layout["bounds"]
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not level_editor.new_level(MAP_PATH):
        raise RuntimeError(f"could not create {MAP_PATH}")

    # Navigation bounds: the default NavMeshBoundsVolume brush is 200 x 200 x 200 cm.
    centre = unreal.Vector((x0 + x1) / 2 * CM, (y0 + y1) / 2 * CM, 5.0 * CM)
    volume = actors.spawn_actor_from_class(unreal.NavMeshBoundsVolume, centre)
    volume.set_actor_scale3d(unreal.Vector((x1 - x0 + 4) * CM / 200.0, (y1 - y0 + 4) * CM / 200.0,
                                           20.0 * CM / 200.0))
    volume.set_actor_label("ResearchArenaNavBounds")

    sx, sy, sz = layout["start"]["position"]
    start = actors.spawn_actor_from_class(unreal.PlayerStart,
                                          unreal.Vector(sx * CM, sy * CM, sz * CM + 100.0),
                                          unreal.Rotator(0.0, 0.0, layout["start"]["yaw_deg"]))
    start.set_actor_label("ResearchStart")

    sun = actors.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 5000),
                                        unreal.Rotator(0.0, -50.0, 30.0))
    sun.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sun.light_component.set_intensity(6.0)
    sun.light_component.set_atmosphere_sun_light(True)
    actors.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0))
    sky = actors.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 3000))
    sky.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sky.light_component.set_editor_property("real_time_capture", True)

    level_editor.save_current_level()
    unreal.log(f"Created {MAP_PATH}: nav bounds, player start and lighting. "
               "Arena geometry is spawned at runtime from ResearchData.")


if __name__ == "__main__":
    main()
