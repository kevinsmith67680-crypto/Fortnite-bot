"""Operations panel for running experiments (Phase 6, optional Tkinter window).

Shows the current episode, agent state, policy, planner, target, chests opened, chests per
simulated minute, exploration coverage, recovery attempts, recent supervisor warnings,
frame-processing rate, process CPU and memory, the latest inference latency and the recent
state transitions. Buttons start, pause, resume and safely stop the experiment, select the
policy, the route strategy and the visual-processing profile, and export the metrics.

The panel never runs the simulation and is never waited on: the experiment loop calls
:meth:`on_decision` once per decision step, and the panel redraws at most every
``refresh_seconds`` (wall clock). Buttons only set flags on the shared
:class:`~chest_hunter.runtime.operations.OperationsControl`, which the loop reads at its
next safe point. Headless runs simply do not create a panel.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from chest_hunter.runtime.operations import OperationsControl, snapshot

FIELDS = (
    ("episode", "Episode"), ("map_id", "Map"), ("state", "Agent state"), ("policy", "Policy"),
    ("planner", "Route strategy"), ("target", "Current target"),
    ("chests_opened", "Chests opened"), ("chests_per_minute", "Chests / simulated min"),
    ("coverage", "Exploration coverage"), ("recoveries", "Recovery attempts"),
    ("frames_per_sim_second", "Frames analysed / sim s"), ("frames_skipped", "Frames skipped"),
    ("cpu_percent", "Process CPU %"), ("memory_mb", "Process memory MB"),
    ("inference_ms", "Inference latency ms"),
)


class OperationsPanel:
    def __init__(self, control: OperationsControl, *, title: str = "Chest hunter operations",
                 refresh_seconds: float = 0.25) -> None:
        import tkinter as tk

        self.tk = tk
        self.control = control
        self.refresh_seconds = refresh_seconds
        self.root: Any = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", lambda: control.request_stop("panel closed"))
        self.values: dict[str, Any] = {}
        grid = tk.Frame(self.root)
        grid.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        for row, (key, label) in enumerate(FIELDS):
            tk.Label(grid, text=label, anchor="w").grid(row=row, column=0, sticky="w")
            var = tk.StringVar(self.root, "-")
            tk.Label(grid, textvariable=var, anchor="w", width=46).grid(row=row, column=1,
                                                                        sticky="w")
            self.values[key] = var
        self.status = tk.StringVar(self.root, "waiting for Start")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=6)
        buttons = tk.Frame(self.root)
        buttons.pack(fill=tk.X, padx=6, pady=4)
        for text, action in (("Start", control.start), ("Pause", control.pause),
                             ("Resume", control.resume),
                             ("Stop safely", lambda: control.request_stop("stopped from the "
                                                                          "operations panel")),
                             ("Export metrics", control.request_export)):
            tk.Button(buttons, text=text, command=action).pack(side=tk.LEFT, padx=2)
        choices = tk.Frame(self.root)
        choices.pack(fill=tk.X, padx=6, pady=4)
        self.policy_choice = tk.StringVar(self.root, "deterministic")
        self.planner_choice = tk.StringVar(self.root, "short_horizon")
        self.profile_choice = tk.StringVar(self.root, "efficient")
        for label, var, opts, setter in (
                ("Policy", self.policy_choice, ("deterministic", "learned", "hybrid"),
                 control.select_policy),
                ("Route strategy (next episode)", self.planner_choice,
                 ("none", "adaptive", "greedy", "short_horizon"), control.select_planner),
                ("Vision profile", self.profile_choice, ("phase5", "efficient", "dense"),
                 control.select_vision_profile)):
            tk.Label(choices, text=label).pack(side=tk.LEFT)
            tk.OptionMenu(choices, var, *opts, command=setter).pack(side=tk.LEFT, padx=2)
        tk.Label(self.root, text="Supervisor warnings", anchor="w").pack(fill=tk.X, padx=6)
        self.warnings = tk.Listbox(self.root, height=6, width=90)
        self.warnings.pack(fill=tk.X, padx=6)
        tk.Label(self.root, text="Recent state transitions", anchor="w").pack(fill=tk.X, padx=6)
        self.transitions = tk.Listbox(self.root, height=6, width=90)
        self.transitions.pack(fill=tk.X, padx=6, pady=(0, 6))
        self._last = -float("inf")
        self.updates = 0
        self.endurance: Any = None
        self.last_episode: dict[str, Any] | None = None

    def attach(self, endurance_runner: Any) -> None:
        self.endurance = endurance_runner

    # -- called by the experiment loop ---------------------------------------------------------

    def pump(self) -> None:
        """Process window events (used while the loop waits for Start or Resume)."""
        if self.root is None:
            return
        try:
            self.root.update()
        except self.tk.TclError:
            self.root = None
            self.control.request_stop("panel closed")
            return
        self.status.set("paused" if self.control.paused else (
            "running" if self.control.started else "waiting for Start"))

    def on_decision(self, runner: Any, _result: Any) -> None:
        now = time.perf_counter()
        if now - self._last < self.refresh_seconds or self.root is None:
            return
        self._last = now
        monitor = getattr(self.endurance, "monitor", None)
        if monitor is not None:
            monitor.maybe_sample()
        snap = snapshot(runner, monitor=monitor)
        self.show(snap)
        self.pump()

    def on_episode(self, record: dict[str, Any]) -> None:
        self.last_episode = record

    def show(self, snap: dict[str, Any]) -> None:
        self.updates += 1
        for key, var in self.values.items():
            value = snap.get(key)
            var.set("-" if value is None else str(value))
        self.warnings.delete(0, self.tk.END)
        for w in snap.get("supervisor_warnings", []):
            self.warnings.insert(self.tk.END, w)
        self.transitions.delete(0, self.tk.END)
        for t in snap.get("transitions", []):
            self.transitions.insert(self.tk.END, t)

    def close(self) -> None:
        if self.root is not None:
            with contextlib.suppress(self.tk.TclError):
                self.root.destroy()
            self.root = None
