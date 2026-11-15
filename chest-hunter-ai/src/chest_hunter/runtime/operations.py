"""Operator controls for running experiments (Phase 6).

:class:`OperationsControl` is the only channel between an operator interface (the
operations panel, a signal handler) and an experiment loop. The interface sets flags; the
loop reads them between decision steps, so the interface never blocks or re-enters the
simulation loop:

* ``pause()`` / ``resume()`` - the loop stops stepping the simulator and only keeps the
  interface responsive (``wait_while_paused``);
* ``request_stop()`` - the current episode ends safely (outcome ``cancelled``), results and
  the checkpoint are written, and the run ends;
* ``select_policy`` / ``select_planner`` / ``select_vision_profile`` - applied at the next
  safe point (policy and vision profile at the next decision, planner at the next episode);
* ``request_export()`` - the loop writes its current metrics to a file.

:func:`snapshot` collects what the panel shows from a running :class:`Runner3D`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class OperationsControl:
    def __init__(self, *, start_paused: bool = False) -> None:
        self._lock = threading.Lock()
        self.paused = start_paused
        self.stop_requested = False
        self.stop_reason = ""
        self.started = not start_paused
        self.policy: str | None = None
        self.planner: str | None = None
        self.vision_profile: str | None = None
        self.export_requested = False
        self.messages: list[str] = []

    # -- interface side ---------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self.started = True
            self.paused = False

    def pause(self) -> None:
        with self._lock:
            self.paused = True

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self.started = True

    def request_stop(self, reason: str = "stopped by the operator") -> None:
        with self._lock:
            self.stop_requested = True
            self.stop_reason = reason
            self.paused = False

    def select_policy(self, policy: str) -> None:
        with self._lock:
            self.policy = policy

    def select_planner(self, planner: str) -> None:
        with self._lock:
            self.planner = planner

    def select_vision_profile(self, profile: str) -> None:
        with self._lock:
            self.vision_profile = profile

    def request_export(self) -> None:
        with self._lock:
            self.export_requested = True

    # -- loop side ------------------------------------------------------------------------------

    def take_export(self) -> bool:
        with self._lock:
            requested, self.export_requested = self.export_requested, False
            return requested

    def wait_while_paused(self, keep_alive: Callable[[], None] | None = None,
                          poll_seconds: float = 0.05) -> None:
        """Block the experiment loop (only) while paused or not yet started."""
        while (self.paused or not self.started) and not self.stop_requested:
            if keep_alive is not None:
                keep_alive()
            time.sleep(poll_seconds)

    def apply_to_runner(self, runner: Any) -> None:
        """Apply pending policy / vision-profile selections to a running episode."""
        controller = runner.controller
        if controller is None:
            return
        agent = getattr(controller, "agent", controller)
        if self.policy is not None and runner.selector is not None:
            try:
                runner.selector.set_mode(self.policy)
                self.messages.append(f"policy -> {self.policy}")
            except ValueError as exc:
                self.messages.append(f"policy not changed: {exc}")
            self.policy = None
        if self.vision_profile is not None and hasattr(agent, "cfg"):
            from chest_hunter.agent.controller_3d import apply_vision_profile

            agent.cfg = apply_vision_profile(agent.cfg, self.vision_profile)
            self.messages.append(f"vision profile -> {self.vision_profile}")
            self.vision_profile = None


def snapshot(runner: Any, *, episode: int | None = None, monitor: Any = None,
             sup_events: int = 6, transitions: int = 8) -> dict[str, Any]:
    """Operator-facing state of a running episode (the agent's own view only)."""
    controller = runner.controller
    agent = getattr(controller, "agent", controller)
    m = agent.metrics
    minutes = agent.now / 60.0 if agent.now > 0 else 0.0
    goal = agent.goal
    priors = list(agent.world.priors.values())
    resolved = sum(1 for p in priors if p.resolved)
    sup = runner.supervisor
    out: dict[str, Any] = {
        "episode": episode if episode is not None else runner.episode,
        "map_id": runner.map_id, "state": str(agent.state),
        "policy": runner.selector.mode if runner.selector is not None else "deterministic",
        "planner": type(runner.planner).__name__ if runner.planner is not None else "phase4",
        "target": (f"{goal.kind} at spawn point {goal.location_id}" if goal is not None
                   else "-"),
        "sim_seconds": round(agent.now, 1), "chests_opened": m.chests_opened,
        "chests_per_minute": round(m.chests_opened / minutes, 3) if minutes else None,
        "coverage": round(resolved / len(priors), 3) if priors else None,
        "recoveries": m.recoveries, "frames_analysed": m.analyses,
        "frames_skipped": m.frames_skipped_unchanged,
        "frames_per_sim_second": round(m.analyses / agent.now, 2) if agent.now > 0 else None,
        "inference_ms": (runner.selector.stats.latencies_ms[-1]
                         if runner.selector is not None and runner.selector.stats.latencies_ms
                         else None),
        "supervisor_warnings": [f"{e.kind}: {e.detail}" + (f" -> {e.action}" if e.action
                                                             else "")
                                for e in sup.events.recent(sup_events)] if sup else [],
        "transitions": [f"{t.source}->{t.target}" for t in list(agent.history)[-transitions:]],
    }
    if monitor is not None and monitor.samples:
        last = monitor.samples[-1]
        out["cpu_percent"] = last["process_cpu_percent"]
        out["memory_mb"] = last["process_rss_mb"]
    return out
