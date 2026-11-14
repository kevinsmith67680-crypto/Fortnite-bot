"""Turn controller intents into validated bridge actions.

The controller thinks in world terms ("walk on bearing 70 degrees at full
speed, look towards bearing 85 / pitch -15"). The research character moves
relative to its camera and turns its camera at bounded rates, so this module
converts intents into the normalised action fields and validates them with
the same rules as the server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from chest_hunter.unreal_bridge.protocol import validate_action
from chest_hunter.vision.perspective import angle_difference_deg


@dataclass(slots=True)
class ActionCommand:
    move_forward: float = 0.0
    move_right: float = 0.0
    look_yaw: float = 0.0
    look_pitch: float = 0.0
    jump: bool = False
    interact: bool = False
    sprint: bool = False
    crouch: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The wire representation, validated (raises ProtocolError if invalid)."""
        return validate_action({
            "move_forward": _clip(self.move_forward), "move_right": _clip(self.move_right),
            "look_yaw": _clip(self.look_yaw), "look_pitch": _clip(self.look_pitch),
            "jump": bool(self.jump), "interact": bool(self.interact),
            "sprint": bool(self.sprint), "crouch": bool(self.crouch),
        })


def _clip(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def move_towards(bearing_deg: float, camera_yaw_deg: float,
                 speed: float = 1.0) -> tuple[float, float]:
    """(move_forward, move_right) that walk along a world bearing, camera-relative."""
    rel = math.radians(angle_difference_deg(bearing_deg, camera_yaw_deg))
    s = max(0.0, min(1.0, speed))
    return s * math.cos(rel), s * math.sin(rel)


def look_towards(current_yaw: float, current_pitch: float, target_yaw: float,
                 target_pitch: float, *, step_seconds: float, yaw_rate: float,
                 pitch_rate: float, gain: float = 0.8) -> tuple[float, float]:
    """Rate commands that turn the camera towards a target without overshooting.

    Each axis covers at most ``gain`` of the remaining error in one step, so
    the camera converges smoothly instead of oscillating.
    """
    yaw_err = angle_difference_deg(target_yaw, current_yaw)
    pitch_err = target_pitch - current_pitch
    yaw_cmd = gain * yaw_err / max(1e-6, yaw_rate * step_seconds)
    pitch_cmd = gain * pitch_err / max(1e-6, pitch_rate * step_seconds)
    return _clip(yaw_cmd), _clip(pitch_cmd)
