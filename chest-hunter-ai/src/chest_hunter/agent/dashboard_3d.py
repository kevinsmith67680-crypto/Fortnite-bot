"""Tkinter debug dashboard for 3D episodes (``simulate-3d --render-debug``).

Shows, after every decision step:

* the agent's latest camera frame with overlays: chest detection boxes with
  confidence and estimated distance, the current target, the planned route
  projected onto the image, and the estimated heading;
* a top-down map of the agent's *estimated* world: spawn points coloured by
  status, chest tracks, its own position estimate and the planned route;
* live statistics: state, target, camera orientation, route progress, chests
  visited/opened, failed interactions, chests per minute, inference latency.

Overlays are drawn on copies; nothing drawn here is ever fed back to the
agent. The window only displays what the agent itself holds (plus, when the
runner evaluates, nothing more: ground truth is never shown here).
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from chest_hunter.localization.world_model_3d import PriorStatus3D
from chest_hunter.vision.perspective import project

STATUS_BGR = {
    PriorStatus3D.UNKNOWN: (140, 140, 140), PriorStatus3D.EMPTY: (90, 90, 90),
    PriorStatus3D.CHEST_SEEN: (0, 215, 255), PriorStatus3D.OPENED: (60, 200, 60),
    PriorStatus3D.FAILED: (40, 40, 220), PriorStatus3D.UNREACHABLE: (40, 40, 160),
    PriorStatus3D.ABANDONED: (120, 60, 160),
}
MAP_PX = 300


def draw_camera_overlay(controller: Any) -> np.ndarray | None:
    """Annotated copy of the agent's latest frame (or ``None`` before the first one)."""
    obs = getattr(controller, "obs", None)
    if obs is None:
        return None
    img = obs.frame.copy()
    for det, est in zip(controller.last_detections, controller.last_estimates, strict=False):
        x, y, w, h = det.bbox
        colour = (60, 200, 60) if det.category == "chest_opened" else (0, 215, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 1)
        label = f"{det.confidence:.2f}" + (f" {est.distance:.1f}m" if est is not None else "")
        cv2.putText(img, label, (x, max(10, y - 3)), cv2.FONT_HERSHEY_PLAIN, 0.9, colour, 1)
    pose = obs.camera_pose()
    follower = getattr(controller, "follower", None)
    if follower is not None and not follower.done:
        pts = np.array([w for w in follower.waypoints[follower.index:]], dtype=float)
        if len(pts):
            u, v, d = project(pts + np.array([0, 0, 0.1]), pose, obs.intrinsics)
            prev = None
            for uu, vv, dd in zip(u, v, d, strict=True):
                if dd > 0.3 and np.isfinite(uu) and np.isfinite(vv):
                    p = (int(uu), int(vv))
                    if prev is not None:
                        cv2.line(img, prev, p, (255, 220, 0), 2)
                    cv2.circle(img, p, 3, (255, 220, 0), -1)
                    prev = p
    goal = getattr(controller, "goal", None)
    if goal is not None:
        u, v, d = project(np.array([goal.position[0], goal.position[1], goal.position[2] + 0.33]),
                          pose, obs.intrinsics)
        if float(d) > 0.3:
            cv2.drawMarker(img, (int(u), int(v)), (0, 140, 255), cv2.MARKER_CROSS, 18, 2)
    # heading arrow (camera yaw) in the corner
    cx, cy, r = 30, 30, 20
    yaw = math.radians(obs.camera.yaw_deg)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)
    cv2.arrowedLine(img, (cx, cy), (int(cx + r * math.sin(yaw)), int(cy - r * math.cos(yaw))),
                    (255, 255, 255), 2)
    return img


def draw_map(controller: Any, bounds: tuple[float, float, float, float] = (0, 0, 100, 100)
             ) -> np.ndarray:
    """Top-down map of the agent's estimated world (north = +X up, east = +Y right)."""
    img = np.full((MAP_PX, MAP_PX, 3), 30, np.uint8)
    x0, y0, x1, y1 = bounds
    scale = MAP_PX / max(x1 - x0, y1 - y0)

    def px(x: float, y: float) -> tuple[int, int]:
        return int((y - y0) * scale), int(MAP_PX - 1 - (x - x0) * scale)

    world = controller.world
    for p in world.priors.values():
        cv2.circle(img, px(p.x, p.y), 4 if p.z < 1.0 else 3, STATUS_BGR[p.status], -1)
        if p.z >= 1.0:
            cv2.circle(img, px(p.x, p.y), 6, STATUS_BGR[p.status], 1)    # upper floor
    for t in world.tracker.tracks:
        colour = (60, 200, 60) if t.category == "chest_opened" else (0, 215, 255)
        cv2.drawMarker(img, px(t.position[0], t.position[1]), colour, cv2.MARKER_TILTED_CROSS, 7, 1)
    follower = getattr(controller, "follower", None)
    if follower is not None and controller.telemetry is not None:
        pts = [controller.position] + list(follower.waypoints[follower.index:])
        for a, b in zip(pts, pts[1:], strict=False):
            cv2.line(img, px(a[0], a[1]), px(b[0], b[1]), (255, 220, 0), 1)
    if controller.telemetry is not None:
        x, y, _ = controller.position
        yaw = math.radians(controller.telemetry.camera.yaw_deg)
        cv2.circle(img, px(x, y), 4, (246, 130, 59), -1)
        tip = px(x + 4 * math.cos(yaw), y + 4 * math.sin(yaw))
        cv2.line(img, px(x, y), tip, (255, 255, 255), 1)
    return img


def stats_text(controller: Any, result: Any = None) -> str:
    m = controller.metrics
    tel = controller.telemetry
    sim = tel.sim_time if tel is not None else 0.0
    rate = m.chests_opened / (sim / 60.0) if sim > 0 else 0.0
    visited = sum(1 for p in controller.world.priors.values() if p.status is not PriorStatus3D.UNKNOWN) \
        if hasattr(controller, "world") else 0
    follower = getattr(controller, "follower", None)
    progress = "-" if follower is None else f"{follower.index}/{len(follower.waypoints)} waypoints"
    cam = f"yaw {tel.camera.yaw_deg:.0f} pitch {tel.camera.pitch_deg:.0f}" if tel else "-"
    goal = getattr(controller, "goal", None)
    return "\n".join([
        f"State: {controller.state}",
        f"Target: {goal.kind + ' T' + str(goal.track_id) if goal and goal.track_id else (goal.kind if goal else '-')}",
        f"Objective: {getattr(getattr(controller, 'world', None), 'objective', '-')}",
        f"Camera: {cam}",
        f"Route: {progress}",
        f"Position estimate: {tuple(round(c, 1) for c in controller.position) if tel else '-'}",
        f"Simulated time: {sim:.1f} s",
        f"Spawn points resolved/visited: {visited}/{len(getattr(controller.world, 'priors', {}))}"
        if hasattr(controller, "world") else "",
        f"Chests opened: {m.chests_opened}   failed interactions: {m.failed_interactions}",
        f"Chests per minute: {rate:.2f}",
        f"Inference: {m.perception_ms_mean:.1f} ms mean ({m.analyses} analyses)",
        f"Recoveries: {m.recoveries}  navigation failures: {m.navigation_failures}",
    ])


def _photo(tk_module: Any, bgr: np.ndarray) -> Any:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return tk_module.PhotoImage(data=b"P6 %d %d 255\n" % (w, h) + rgb.tobytes(), format="PPM")


class Dashboard3D:
    """A small window refreshed from the runner's ``on_decision`` hook (~10 Hz real time)."""

    def __init__(self, *, refresh_seconds: float = 0.1) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Chest Hunter 3D - debug dashboard")
        self.root.configure(bg="#1f1f1f")
        self.camera = tk.Label(self.root, bg="#000000")
        self.camera.grid(row=0, column=0, rowspan=2, padx=6, pady=6)
        self.map = tk.Label(self.root, bg="#000000")
        self.map.grid(row=0, column=1, padx=6, pady=6, sticky="n")
        self.info = tk.StringVar(value="waiting for the first observation...")
        tk.Label(self.root, textvariable=self.info, justify="left", anchor="nw", bg="#1f1f1f",
                 fg="#e6e6e6", font=("TkFixedFont", 9)).grid(row=1, column=1, sticky="nw", padx=6)
        self.refresh_seconds = refresh_seconds
        self._last = 0.0
        self._images: list[Any] = []
        self.frames_shown = 0

    def update(self, runner: Any, result: Any = None) -> None:
        now = time.perf_counter()
        if now - self._last < self.refresh_seconds:
            return
        self._last = now
        controller = runner.controller
        if controller is None or not hasattr(controller, "world"):
            self.root.update()
            return
        cam = draw_camera_overlay(controller)
        images = []
        if cam is not None:
            images.append(_photo(self.tk, cam))
            self.camera.configure(image=images[-1])
        images.append(_photo(self.tk, draw_map(controller)))
        self.map.configure(image=images[-1])
        self.info.set(stats_text(controller, result))
        self._images = images
        self.frames_shown += 1
        self.root.update()

    def close(self) -> None:
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass
