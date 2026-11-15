"""Human teleoperation for recording demonstrations (``record-demo --controller human``).

A small Tk window shows the agent's camera image and turns the keys pressed *in this
window* into bridge actions, which are sent to the research environment through its own
control interface (the same ``STEP`` command the agent uses). Nothing is injected into the
operating system or any other application, and nothing outside this window is recorded.

Keys: W/S forward/back, A/D strafe, Left/Right turn, Up/Down look, Space jump, E interact
(hold), Shift sprint, P pause/resume, Escape stop and save.

While paused the environment is not stepped at all, so a recording with pauses still
replays exactly.
"""

from __future__ import annotations

import contextlib
import gc
import time
from typing import Any

import cv2
import numpy as np

from chest_hunter.learning.demonstration_recorder import SOURCE_HUMAN, DemonstrationRecorder
from chest_hunter.unreal_bridge.client import BridgeClient
from chest_hunter.unreal_bridge.observation_adapter import Observation3D

KEYS = {"w": "forward", "s": "back", "a": "left", "d": "right", "Left": "turn_left",
        "Right": "turn_right", "Up": "look_up", "Down": "look_down", "space": "jump",
        "e": "interact", "Shift_L": "sprint", "Shift_R": "sprint"}


def action_from_keys(pressed: set[str]) -> dict[str, Any]:
    """Pressed logical keys -> validated bridge action."""
    fwd = float("forward" in pressed) - float("back" in pressed)
    right = float("right" in pressed) - float("left" in pressed)
    return {"move_forward": fwd, "move_right": right,
            "look_yaw": float("turn_right" in pressed) - float("turn_left" in pressed),
            "look_pitch": float("look_up" in pressed) - float("look_down" in pressed),
            "jump": "jump" in pressed, "interact": "interact" in pressed,
            "sprint": "sprint" in pressed, "crouch": False}


class TeleopSession:
    """Drives one recorded episode from keyboard input (real-time pacing)."""

    def __init__(self, client: BridgeClient, recorder: DemonstrationRecorder, *, episode_id: int,
                 map_id: str, seed: int, episode: int = 0, width: int = 640, height: int = 360,
                 decision_ticks: int = 2, max_ticks: int = 18000, realtime: bool = True,
                 tk_root: Any = None) -> None:
        import tkinter as tk

        self.tk = tk
        self.client = client
        self.recorder = recorder
        self.decision_ticks = decision_ticks
        self.realtime = realtime
        self.pressed: set[str] = set()
        self.paused = False
        self.finished = False
        self.steps = 0
        telemetry = client.reset_episode(map_id, seed=seed, episode=episode, width=width,
                                         height=height, max_ticks=max_ticks)
        self.time = telemetry.sim_time
        self.obs: Observation3D | None = client.get_observation()
        recorder.start(episode_id=episode_id, map_id=map_id, seed=seed, episode=episode,
                       source=SOURCE_HUMAN, telemetry=telemetry,
                       settings={"width": width, "height": height, "max_ticks": max_ticks,
                                 "navigation": "navmesh", "decision_ticks": decision_ticks,
                                 "controller": "human-teleop"}, initial_observation=True)
        self._fresh_obs: Observation3D | None = self.obs
        self._owns_root = tk_root is None
        self.root: Any = tk_root or tk.Tk()
        self.root.title("Chest Hunter - demonstration recording (keys act on this window only)")
        self.view: Any = tk.Label(self.root, bg="#000000")
        self.view.pack()
        self.status: Any = tk.StringVar(value="recording")
        self._status_label: Any = tk.Label(self.root, textvariable=self.status, anchor="w")
        self._status_label.pack(fill="x")
        self.root.bind("<KeyPress>", self._down)
        self.root.bind("<KeyRelease>", self._up)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self._photo: Any = None
        self._show()

    # -- input -------------------------------------------------------------------------------

    def _down(self, event: Any) -> None:
        key = KEYS.get(event.keysym) or KEYS.get(str(event.keysym).lower())
        if event.keysym in ("p", "P"):
            self.toggle_pause()
        elif event.keysym == "Escape":
            self.stop()
        elif key:
            self.pressed.add(key)

    def _up(self, event: Any) -> None:
        key = KEYS.get(event.keysym) or KEYS.get(str(event.keysym).lower())
        if key:
            self.pressed.discard(key)

    def toggle_pause(self) -> None:
        if self.finished:
            return
        self.paused = not self.paused
        if self.paused:
            self.recorder.pause()
        else:
            self.recorder.resume()

    # -- stepping ---------------------------------------------------------------------------

    def step(self) -> None:
        """One control step (also callable directly, e.g. from tests)."""
        if self.finished or self.paused:
            return
        action = action_from_keys(self.pressed)
        result = self.client.step(action, ticks=self.decision_ticks, observe=True)
        self.recorder.record_step(action=action, ticks=self.decision_ticks, action_time=self.time,
                                  telemetry_after=result.telemetry, observation=self._fresh_obs,
                                  observed=True, phase=-1)
        self.time = result.telemetry.sim_time
        self.obs = result.observation
        self._fresh_obs = result.observation
        self.steps += 1
        opened = sum(1 for e in result.telemetry.events
                     if e.get("type") == "chest_opened_notification")
        self.status.set(f"{'PAUSED' if self.paused else 'recording'}  t={self.time:.1f} s  "
                        f"steps={self.steps}" + ("  chest opened!" if opened else ""))
        self._show()
        if result.ended:
            self.stop()

    def _show(self) -> None:
        if self.obs is None:
            return
        rgb = cv2.cvtColor(self.obs.frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._photo = self.tk.PhotoImage(data=b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(
            rgb).tobytes(), format="PPM")
        self.view.configure(image=self._photo)

    def stop(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.recorder.active:
            if self.recorder.state == "paused":
                self.recorder.resume()
            self.recorder.stop(outcome={"outcome": "stopped_by_operator", "steps": self.steps})
        with contextlib.suppress(self.tk.TclError):
            self.root.quit()

    def close(self) -> None:
        """Stop (the recording is saved) and destroy the window, releasing every Tk object
        on this thread: Tk must not be torn down by the garbage collector on another one."""
        if self.root is None:
            return
        self.stop()
        self._photo = None
        with contextlib.suppress(self.tk.TclError):
            if self._owns_root:
                self.root.destroy()
            else:
                self.view.destroy()
                self._status_label.destroy()
                self.root.unbind("<KeyPress>")
                self.root.unbind("<KeyRelease>")
        self.view = self.status = self._status_label = self.root = None
        gc.collect()

    def run(self) -> None:
        period_ms = int(1000 * self.decision_ticks / 30)

        def loop() -> None:
            if self.finished:
                return
            started = time.perf_counter()
            self.step()
            spent = int((time.perf_counter() - started) * 1000)
            self.root.after(max(1, period_ms - spent) if self.realtime else 1, loop)

        self.root.after(period_ms, loop)
        self.root.mainloop()
