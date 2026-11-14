"""Third-person research character: capsule physics and a spring-arm camera.

Mirrors the Unreal ``ACharacter`` + ``UCharacterMovementComponent`` +
``USpringArmComponent`` setup used by the research project:

* Movement is camera-relative: ``move_forward`` follows the camera's yaw on
  the ground plane, ``move_right`` is perpendicular to it.
* Continuous acceleration and braking up to ``max_speed``; the body turns
  smoothly towards its velocity ("orient rotation to movement").
* Collisions: a vertical capsule against axis-aligned boxes, resolved per
  axis so the character slides along walls. Steps up to ``step_height`` are
  climbed automatically, small drops are followed (stairs), larger drops fall
  under gravity. Jumping needs ground contact.
* The camera sits on a spring arm behind the character's head pivot. The arm
  shortens when geometry is in the way (a ray cast), like Unreal's
  ``bDoCollisionTest``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from chest_hunter.simulation3d.geometry import Box, World3D, circle_overlaps_box_xy
from chest_hunter.vision.perspective import CameraPose, angle_difference_deg, camera_basis


@dataclass(frozen=True, slots=True)
class CharacterParams:
    radius: float = 0.35
    height: float = 1.8
    step_height: float = 0.55          # = the navigation graph's max climb (Unreal: MaxStepHeight 55 cm)
    max_speed: float = 5.0             # m/s (Unreal: MaxWalkSpeed 500 cm/s)
    sprint_speed: float = 7.0
    crouch_speed: float = 2.5
    acceleration: float = 20.0         # m/s^2
    braking: float = 30.0              # m/s^2 when there is no input
    gravity: float = 9.81
    jump_speed: float = 4.2            # m/s (about 0.9 m jump height)
    turn_rate: float = 540.0           # deg/s, body yaw towards velocity
    pivot_height: float = 1.6          # camera pivot above the feet
    arm_length: float = 3.2
    shoulder_offset: float = 0.65      # camera to the right of the arm (over the shoulder)
    arm_probe: float = 0.2
    yaw_rate: float = 120.0            # deg/s at look_yaw = 1
    pitch_rate: float = 60.0           # deg/s at look_pitch = 1
    pitch_min: float = -55.0
    pitch_max: float = 25.0


class Character3D:
    def __init__(self, x: float, y: float, z: float, yaw_deg: float,
                 params: CharacterParams | None = None) -> None:
        self.p = params or CharacterParams()
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.body_yaw = yaw_deg % 360.0
        self.cam_yaw = yaw_deg % 360.0
        self.cam_pitch = -12.0
        self.grounded = True
        self.crouched = False
        self.distance_travelled = 0.0
        self.blocked_ticks = 0           # ticks where commanded motion was obstructed

    # -- control --------------------------------------------------------------------------

    def look(self, look_yaw: float, look_pitch: float, dt: float) -> None:
        self.cam_yaw = (self.cam_yaw + look_yaw * self.p.yaw_rate * dt) % 360.0
        self.cam_pitch = min(self.p.pitch_max, max(self.p.pitch_min,
                                                     self.cam_pitch + look_pitch * self.p.pitch_rate * dt))

    def step(self, world: World3D, dynamic: Sequence[Box], move_forward: float, move_right: float,
             *, jump: bool, sprint: bool, crouch: bool, dt: float) -> None:
        p = self.p
        self.crouched = crouch
        speed = p.crouch_speed if crouch else (p.sprint_speed if sprint else p.max_speed)
        f, r, _ = camera_basis(self.cam_yaw, 0.0)
        ix = move_forward * f[0] + move_right * r[0]
        iy = move_forward * f[1] + move_right * r[1]
        mag = math.hypot(ix, iy)
        if mag > 1.0:
            ix, iy, mag = ix / mag, iy / mag, 1.0
        want_x, want_y = ix * speed, iy * speed
        rate = p.acceleration if mag > 1e-3 else p.braking
        dvx, dvy = want_x - self.vx, want_y - self.vy
        dv = math.hypot(dvx, dvy)
        limit = rate * dt
        if dv > limit:
            dvx, dvy = dvx / dv * limit, dvy / dv * limit
        self.vx += dvx
        self.vy += dvy
        if math.hypot(self.vx, self.vy) > 0.05:
            target = math.degrees(math.atan2(self.vy, self.vx)) % 360.0
            diff = angle_difference_deg(target, self.body_yaw)
            turn = max(-p.turn_rate * dt, min(p.turn_rate * dt, diff))
            self.body_yaw = (self.body_yaw + turn) % 360.0

        if jump and self.grounded:
            self.vz = p.jump_speed
            self.grounded = False

        # Horizontal motion in sub-steps (never more than 0.15 m at a time).
        dx, dy = self.vx * dt, self.vy * dt
        steps = max(1, int(math.ceil(math.hypot(dx, dy) / 0.15)))
        obstructed = False
        for _ in range(steps):
            sx, sy = dx / steps, dy / steps
            if self._free(world, dynamic, self.x + sx, self.y + sy):
                self._advance(sx, sy)
                continue
            obstructed = True
            slid = self._slide(world, dynamic, self.x + sx, self.y + sy, sx, sy)
            if slid is not None:
                self._advance(slid[0] - self.x, slid[1] - self.y)
            elif sx and self._free(world, dynamic, self.x + sx, self.y):
                self._advance(sx, 0.0)
                self.vy = 0.0
            elif sy and self._free(world, dynamic, self.x, self.y + sy):
                self._advance(0.0, sy)
                self.vx = 0.0
            else:
                self.vx = self.vy = 0.0
                break
        self.blocked_ticks = self.blocked_ticks + 1 if obstructed and mag > 0.2 else 0

        # Vertical: step up, follow small drops, otherwise fall.
        support = world.support_height(self.x, self.y, self.z, radius=0.2,
                                       step_height=p.step_height)
        support = max(support, self._dynamic_support(dynamic))
        if self.grounded:
            if support >= self.z - p.step_height - 0.05:
                self.z = max(support, self.z) if support > self.z else support
            else:
                self.grounded = False
        if not self.grounded:
            self.vz -= p.gravity * dt
            self.z += self.vz * dt
            ceiling = self._ceiling(world)
            if self.vz > 0 and self.z + p.height > ceiling:
                self.z = ceiling - p.height
                self.vz = 0.0
            if self.z <= support:
                self.z = support
                self.vz = 0.0
                self.grounded = True
            if self.z < -20.0:                 # fell out of the world: should never happen
                self.z = support if support > -1e9 else 0.0
                self.vz = 0.0
                self.grounded = True

    def _advance(self, sx: float, sy: float) -> None:
        self.x += sx
        self.y += sy
        self.distance_travelled += math.hypot(sx, sy)

    def _free(self, world: World3D, dynamic: Sequence[Box], x: float, y: float) -> bool:
        p = self.p
        height = p.height * (0.6 if self.crouched else 1.0)
        if world.blocked(x, y, self.z, radius=p.radius, height=height,
                         step_height=p.step_height):
            return False
        lo, hi = self.z + p.step_height, self.z + height
        return not any(b.z1 > lo and b.z0 < hi and circle_overlaps_box_xy(x, y, p.radius, b)
                       for b in dynamic)

    def _slide(self, world: World3D, dynamic: Sequence[Box], x: float, y: float,
               sx: float, sy: float) -> tuple[float, float] | None:
        """Capsule push-out: move the circle out of the boxes it penetrates along the
        contact normals, so it slides along walls and around corners (door frames) like
        Unreal's character movement. Accepted only if it still makes forward progress."""
        p = self.p
        height = p.height * (0.6 if self.crouched else 1.0)
        lo, hi = self.z + p.step_height, self.z + height
        boxes = [b for b in list(world.nearby(x, y, p.radius + 0.3)) + list(dynamic)
                 if b.blocks and b.z1 > lo and b.z0 < hi]
        for _ in range(4):
            moved = False
            for b in boxes:
                nx = min(max(x, b.x0), b.x1)
                ny = min(max(y, b.y0), b.y1)
                dx, dy = x - nx, y - ny
                d = math.hypot(dx, dy)
                if d >= p.radius:
                    continue
                if d < 1e-9:                            # centre inside the box: shortest axis
                    pushes = ((b.x0 - p.radius - x, 0.0), (b.x1 + p.radius - x, 0.0),
                              (0.0, b.y0 - p.radius - y), (0.0, b.y1 + p.radius - y))
                    ox, oy = min(pushes, key=lambda v: abs(v[0]) + abs(v[1]))
                    x, y = x + ox, y + oy
                else:
                    push = p.radius - d + 1e-4
                    x, y = x + dx / d * push, y + dy / d * push
                moved = True
            if not moved:
                break
        progress = (x - self.x) * sx + (y - self.y) * sy
        if progress <= 0.2 * (sx * sx + sy * sy) or math.hypot(x - self.x, y - self.y) > 0.3:
            return None
        return (x, y) if self._free(world, dynamic, x, y) else None

    def _dynamic_support(self, dynamic: Sequence[Box]) -> float:
        best = -math.inf
        for b in dynamic:
            if b.z1 <= self.z + self.p.step_height and circle_overlaps_box_xy(self.x, self.y, 0.2, b):
                best = max(best, b.z1)
        return best

    def _ceiling(self, world: World3D) -> float:
        top = math.inf
        for b in world.nearby(self.x, self.y, self.p.radius):
            if b.blocks and b.z0 >= self.z + self.p.step_height and \
                    circle_overlaps_box_xy(self.x, self.y, self.p.radius, b):
                top = min(top, b.z0)
        return top

    # -- camera ----------------------------------------------------------------------------

    def pivot(self) -> tuple[float, float, float]:
        h = self.p.pivot_height * (0.7 if self.crouched else 1.0)
        return (self.x, self.y, self.z + h)

    def camera(self, world: World3D) -> tuple[CameraPose, float]:
        """Camera pose after spring-arm collision, and the resulting arm length."""
        px, py, pz = self.pivot()
        fwd, right, _ = camera_basis(self.cam_yaw, self.cam_pitch)
        arm = self.p.arm_length
        off = self.p.shoulder_offset
        # Desired camera point: behind the pivot along the view direction, shifted right.
        dx = -fwd[0] * arm + right[0] * off
        dy = -fwd[1] * arm + right[1] * off
        dz = -fwd[2] * arm
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        scale = 1.0
        hit = world.raycast((px, py, pz), (dx, dy, dz), length + self.p.arm_probe)
        if hit is not None:
            scale = max(0.1, min(1.0, (hit - self.p.arm_probe) / length))
        cam = CameraPose(px + dx * scale, py + dy * scale, pz + dz * scale,
                         self.cam_yaw, self.cam_pitch)
        return cam, arm * scale

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)
