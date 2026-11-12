"""Lightweight Tkinter viewer and control panel for the simulator.

The GUI never runs simulation logic itself. It asks a *runner factory* for a
:class:`~chest_hunter.agent.runner.SimulationRunner`, advances it from
Tkinter's ``after()`` loop (no threads) and redraws at ``render_rate`` frames
per second. At speed 1x it advances ``tick_rate`` ticks per real second;
higher speeds run more fixed ticks per frame, so results are identical at any
speed.

Static items (obstacles, chest markers) are created once per episode; each
frame only moves the agent, updates the path line and recolours chests whose
state changed.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from chest_hunter.agent.controller import AgentController

if TYPE_CHECKING:  # pragma: no cover
    from chest_hunter.agent.runner import SimulationRunner
    from chest_hunter.config import VisualDebugSettings

COLOURS = {
    "background": "#2b2b2b",
    "panel": "#1f1f1f",
    "text": "#e6e6e6",
    "muted_text": "#9a9a9a",
    "agent": "#3b82f6",
    "heading": "#bfdbfe",
    "chest_available": "#f5c518",   # gold: unopened chest
    "chest_opened": "#22c55e",      # green: opened chest
    "chest_absent": "#6b7280",      # muted grey: verified empty spawn point
    "chest_unknown": "#8a8a5c",     # spawn point not checked yet and empty
    "chest_unreachable": "#b91c1c",
    "obstacle": "#64748b",          # slate grey
    "path": "#22d3ee",              # cyan
    "target": "#fb923c",            # orange
    "estimate": "#67e8f9",          # estimated (perceived) agent position
    "track": "#fde68a",             # tracked chest label
}
DEBUG_EVERY_FRAMES = 3   # refresh the camera/occupancy debug images at ~5 Hz

SPEEDS: tuple[int, ...] = (1, 2, 5, 10)

RunnerFactory = Callable[[str], "SimulationRunner"]


class SimulationApp:
    """Main window: map canvas plus an information/control panel."""

    def __init__(
        self,
        root: tk.Tk,
        runner_factory: RunnerFactory,
        *,
        strategies: Sequence[str],
        strategy: str,
        cell_size: int,
        tick_rate: int,
        render_rate: int,
        speed: int = 1,
        autostart: bool = True,
        on_finish: Callable[[SimulationRunner], None] | None = None,
        debug: VisualDebugSettings | None = None,
    ) -> None:
        self.root = root
        self.factory = runner_factory
        self.cell_size = cell_size
        self.tick_rate = tick_rate
        self.render_rate = render_rate
        self.frame_ms = max(1, round(1000 / render_rate))
        self.on_finish = on_finish
        self.runner: SimulationRunner | None = None
        self.running = False
        self.paused = False
        self.frames_rendered = 0
        self._tick_budget = 0.0
        self._after_id: str | None = None
        self._chest_items: dict[int, int] = {}
        self._chest_colours: dict[int, str] = {}
        self._agent_item = self._heading_item = self._path_item = self._target_item = 0

        root.title("Chest Hunter AI - 2D research simulator")
        root.configure(bg=COLOURS["panel"])
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.strategy_var = tk.StringVar(value=strategy)
        self.speed_var = tk.StringVar(value=f"{speed}x")
        self.show_paths_var = tk.BooleanVar(value=True)
        self.info_vars: dict[str, tk.StringVar] = {}
        self.debug_settings = debug
        self.debug_vars: dict[str, tk.BooleanVar] = {}
        self._camera_photo: tk.PhotoImage | None = None
        self._occupancy_photo: tk.PhotoImage | None = None
        self._debug_items: list[int] = []
        self._build_layout(strategies)
        if debug is not None:
            self._build_debug_panel(debug)
        self._bind_keys()
        self.reset()
        if autostart:
            self.start()
        self._schedule()

    # -- layout -------------------------------------------------------------------

    def _build_layout(self, strategies: Sequence[str]) -> None:
        self.canvas = tk.Canvas(self.root, width=1, height=1, bg=COLOURS["background"],
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        panel = tk.Frame(self.root, bg=COLOURS["panel"])
        panel.grid(row=0, column=1, padx=(0, 10), pady=8, sticky="n")

        label_opts: dict[str, Any] = {"bg": COLOURS["panel"], "fg": COLOURS["text"],
                                      "anchor": "w"}
        tk.Label(panel, text="Chest Hunter AI", font=("TkDefaultFont", 12, "bold"),
                 **label_opts).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        rows = [
            ("simulation", "Simulation"), ("agent_state", "Agent state"),
            ("target", "Current target"), ("time", "Elapsed (sim)"),
            ("visited", "Visited locations"), ("found", "Chests discovered"),
            ("opened", "Chests opened"), ("rate", "Chests / minute"),
            ("recoveries", "Recoveries"), ("unreachable", "Unreachable"),
            ("rewards", "Tokens (fictional)"), ("strategy", "Strategy"),
            ("session", "Session / seed"),
        ]
        for i, (key, title) in enumerate(rows, start=1):
            tk.Label(panel, text=title, fg=COLOURS["muted_text"], bg=COLOURS["panel"],
                     anchor="w").grid(row=i, column=0, sticky="w", padx=(0, 8))
            var = tk.StringVar(value="-")
            self.info_vars[key] = var
            tk.Label(panel, textvariable=var, width=22, **label_opts).grid(
                row=i, column=1, sticky="w")

        controls = tk.Frame(panel, bg=COLOURS["panel"])
        controls.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="w", pady=(10, 4))
        self.start_button = tk.Button(controls, text="Start", width=7, command=self.start)
        self.pause_button = tk.Button(controls, text="Pause", width=7, command=self.pause)
        self.resume_button = tk.Button(controls, text="Resume", width=7, command=self.resume)
        self.reset_button = tk.Button(controls, text="Reset", width=7, command=self.reset)
        for col, button in enumerate((self.start_button, self.pause_button,
                                      self.resume_button, self.reset_button)):
            button.grid(row=0, column=col, padx=1)

        options = tk.Frame(panel, bg=COLOURS["panel"])
        options.grid(row=len(rows) + 2, column=0, columnspan=2, sticky="w", pady=4)
        tk.Label(options, text="Speed", **label_opts).grid(row=0, column=0, sticky="w")
        tk.OptionMenu(options, self.speed_var, *[f"{s}x" for s in SPEEDS]).grid(
            row=0, column=1, sticky="w")
        tk.Label(options, text="Strategy", **label_opts).grid(row=1, column=0, sticky="w")
        tk.OptionMenu(options, self.strategy_var, *strategies).grid(row=1, column=1, sticky="w")
        tk.Checkbutton(options, text="Show paths", variable=self.show_paths_var,
                       bg=COLOURS["panel"], fg=COLOURS["text"], selectcolor=COLOURS["panel"],
                       activebackground=COLOURS["panel"]).grid(row=2, column=0, columnspan=2,
                                                               sticky="w")
        tk.Label(options, text="Strategy changes apply on Start/Reset.\n"
                               "Keys: Space pause/resume, S start, R reset, Q quit",
                 fg=COLOURS["muted_text"], bg=COLOURS["panel"], justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w")

        legend = tk.Frame(panel, bg=COLOURS["panel"])
        legend.grid(row=len(rows) + 3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        for i, (key, text) in enumerate([
            ("agent", "Agent"), ("chest_available", "Unopened chest"),
            ("chest_opened", "Opened chest"), ("chest_absent", "Absent (checked)"),
            ("chest_unknown", "Empty, not yet checked"), ("obstacle", "Obstacle"),
            ("path", "Navigation path"), ("target", "Current target"),
        ]):
            swatch = tk.Canvas(legend, width=12, height=12, bg=COLOURS["panel"],
                               highlightthickness=0)
            swatch.create_rectangle(1, 1, 11, 11, fill=COLOURS[key], outline=COLOURS[key])
            swatch.grid(row=i, column=0, padx=(0, 4), pady=1)
            tk.Label(legend, text=text, **label_opts).grid(row=i, column=1, sticky="w")

    def _bind_keys(self) -> None:
        """Keyboard shortcuts: Space pause/resume, S start, R reset, Q/Escape quit."""
        self.root.bind("<space>", lambda _e: self.resume() if self.paused else self.pause())
        self.root.bind("<Key-s>", lambda _e: self.start())
        self.root.bind("<Key-r>", lambda _e: self.reset())
        self.root.bind("<Key-q>", lambda _e: self.close())
        self.root.bind("<Escape>", lambda _e: self.close())

    # -- visual debugging (Phase 3) ------------------------------------------------

    def _build_debug_panel(self, debug: VisualDebugSettings) -> None:
        """Third column: camera view with detections, occupancy minimap, toggles.

        Overlays are drawn on *copies* for display only. Nothing drawn here is
        ever fed back to perception.
        """
        panel = tk.Frame(self.root, bg=COLOURS["panel"])
        panel.grid(row=0, column=2, padx=(0, 10), pady=8, sticky="n")
        opts: dict[str, Any] = {"bg": COLOURS["panel"], "fg": COLOURS["text"], "anchor": "w"}
        tk.Label(panel, text="Visual debugging", font=("TkDefaultFont", 11, "bold"),
                 **opts).grid(row=0, column=0, sticky="w")
        toggles = [("enabled", "Enable visual debugging", debug.enabled),
                   ("detections", "Show chest detections", debug.show_detections),
                   ("tracks", "Show object tracks", debug.show_object_tracks),
                   ("occupancy", "Show occupancy map", debug.show_occupancy_grid),
                   ("localization", "Show localisation estimate", debug.show_localization),
                   ("camera", "Show camera view", debug.show_camera_view)]
        for i, (key, text, value) in enumerate(toggles, start=1):
            var = tk.BooleanVar(value=value)
            self.debug_vars[key] = var
            tk.Checkbutton(panel, text=text, variable=var, bg=COLOURS["panel"],
                           fg=COLOURS["text"], selectcolor=COLOURS["panel"],
                           activebackground=COLOURS["panel"]).grid(row=i, column=0, sticky="w")
        self.camera_label = tk.Label(panel, bg=COLOURS["background"], text="camera view",
                                     fg=COLOURS["muted_text"])
        self.camera_label.grid(row=len(toggles) + 1, column=0, pady=(6, 2), sticky="w")
        self.occupancy_label = tk.Label(panel, bg=COLOURS["background"], text="occupancy",
                                        fg=COLOURS["muted_text"])
        self.occupancy_label.grid(row=len(toggles) + 2, column=0, pady=2, sticky="w")
        self.debug_info = tk.StringVar(value="-")
        tk.Label(panel, textvariable=self.debug_info, justify="left", **opts).grid(
            row=len(toggles) + 3, column=0, sticky="w")

    def _debug_on(self, key: str) -> bool:
        if not self.debug_vars or not self.debug_vars["enabled"].get():
            return False
        return self.debug_vars[key].get()

    @staticmethod
    def _photo(rgb: np.ndarray) -> tk.PhotoImage:
        height, width = rgb.shape[:2]
        data = b"P6 %d %d 255\n" % (width, height) + rgb.tobytes()
        return tk.PhotoImage(data=data, format="PPM")

    def _render_debug(self) -> None:
        runner = self.runner
        if runner is None or not self.debug_vars or not runner.is_visual \
                or runner.controller is None:
            return
        import cv2

        controller = runner.visual_controller
        result = controller.last_result
        canvas = self.canvas
        for item in self._debug_items:
            canvas.delete(item)
        self._debug_items = []
        world = controller.world
        if self._debug_on("localization") and world.pose.position is not None:
            ex, ey = self._xy(*world.pose.position)
            r = self._agent_radius + 3
            self._debug_items.append(canvas.create_oval(ex - r, ey - r, ex + r, ey + r,
                                                        outline=COLOURS["estimate"], width=2))
        if self._debug_on("tracks"):
            for track in world.tracker.tracks:
                tx, ty = self._xy(*track.estimated_position)
                colour = COLOURS["chest_opened"] if track.category == "chest_opened" \
                    else COLOURS["track"]
                self._debug_items.append(canvas.create_text(
                    tx + 7, ty - 7, text=f"T{track.track_id}", fill=colour, anchor="sw",
                    font=("TkDefaultFont", 8)))
        if self.frames_rendered % DEBUG_EVERY_FRAMES:
            return
        if self._debug_on("camera") and result is not None:
            frame = result.observation.frame.copy()   # display copy only
            if self._debug_on("detections"):
                for det in result.detections:
                    x, y, w, h = det.bbox
                    bgr = (94, 197, 34) if det.category == "chest_opened" else (0, 215, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 1)
                    cv2.putText(frame, f"{det.confidence:.2f}", (x, max(8, y - 2)),
                                cv2.FONT_HERSHEY_PLAIN, 0.7, bgr, 1)
            if self._debug_on("tracks"):
                cam = result.observation.camera
                for track in world.tracker.tracks:
                    u, v = cam.world_to_pixel(*track.estimated_position)
                    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                        cv2.putText(frame, f"T{track.track_id}", (int(u) + 8, int(v) + 4),
                                    cv2.FONT_HERSHEY_PLAIN, 0.8, (138, 230, 253), 1)
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            self._camera_photo = self._photo(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            self.camera_label.configure(image=self._camera_photo, text="")
        if self._debug_on("occupancy"):
            occ = world.occupancy.cells[::-1]
            rgb = np.empty(occ.shape + (3,), np.uint8)
            rgb[occ == -1] = (12, 12, 12)
            rgb[occ == 0] = (55, 55, 55)
            rgb[occ == 1] = (100, 116, 139)
            for track in world.tracker.tracks:
                tx, ty = int(track.estimated_position[0]), int(track.estimated_position[1])
                if 0 <= ty < occ.shape[0] and 0 <= tx < occ.shape[1]:
                    rgb[occ.shape[0] - 1 - ty, tx] = (245, 197, 24)
            if world.pose.position is not None:
                px, py = int(world.pose.position[0]), int(world.pose.position[1])
                if 0 <= py < occ.shape[0] and 0 <= px < occ.shape[1]:
                    rgb[occ.shape[0] - 1 - py, px] = (103, 232, 249)
            scale = max(1, 200 // max(occ.shape))
            big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
            self._occupancy_photo = self._photo(np.ascontiguousarray(big))
            self.occupancy_label.configure(image=self._occupancy_photo, text="")
        env_time = max(runner.env.time, 1e-9)
        confidence = world.pose.confidence()
        self.debug_info.set(
            f"Visual analyses: {controller.analyses} ({controller.analyses / env_time:.1f}/sim s)\n"
            f"Perception: {1000 * controller.pipeline.mean_seconds:.1f} ms/frame\n"
            f"Pose confidence: {confidence:.2f}\n"
            f"Explored: {100 * world.occupancy.known_fraction:.0f}%   "
            f"Tracks: {len(world.tracker.tracks)}\n"
            f"Objective: {world.objective or '-'}")

    # -- controls -------------------------------------------------------------------

    @property
    def speed(self) -> int:
        return int(self.speed_var.get().rstrip("x"))

    def start(self) -> None:
        if self.runner is None or self.runner.finished:
            self._new_runner()
        assert self.runner is not None
        if not self.runner.started:
            self.runner.start()
        self.running = True
        self.paused = False
        self._update_buttons()

    def pause(self) -> None:
        if self.running:
            self.paused = True
            self._update_buttons()

    def resume(self) -> None:
        if self.running:
            self.paused = False
            self._update_buttons()

    def reset(self) -> None:
        """Abandon the current episode (its session is aborted) and load a fresh one."""
        if self.runner is not None:
            self.runner.cancel("reset from the GUI")
        self._new_runner()
        self.running = False
        self.paused = False
        self._update_buttons()
        self.render()

    def close(self) -> None:
        if self.runner is not None:
            self.runner.cancel("GUI window closed")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.root.destroy()

    def _new_runner(self) -> None:
        self.runner = self.factory(self.strategy_var.get())
        self._draw_static()

    def _update_buttons(self) -> None:
        finished = self.runner is not None and self.runner.finished
        self.start_button.configure(state="normal" if not self.running or finished else "disabled")
        self.pause_button.configure(state="normal" if self.running and not self.paused
                                    else "disabled")
        self.resume_button.configure(state="normal" if self.running and self.paused
                                     else "disabled")

    # -- frame loop -------------------------------------------------------------------

    def _schedule(self) -> None:
        self._after_id = self.root.after(self.frame_ms, self._frame)

    def _frame(self) -> None:
        self._after_id = None
        try:
            self.advance_frame()
            self.render()
        finally:   # an exception is still reported by Tk, but never stops the loop
            self._schedule()

    def advance_frame(self) -> None:
        """Advance the simulation by one frame's worth of fixed ticks."""
        runner = self.runner
        if runner is None or not self.running or self.paused or runner.finished:
            return
        self._tick_budget += self.speed * self.tick_rate / self.render_rate
        ticks = int(self._tick_budget)
        self._tick_budget -= ticks
        if ticks and not runner.step(ticks):
            self.running = False
            self._update_buttons()
            if self.on_finish is not None:
                self.on_finish(runner)

    # -- drawing ------------------------------------------------------------------

    def _xy(self, gx: float, gy: float) -> tuple[float, float]:
        assert self.runner is not None
        return gx * self.cell_size, (self.runner.world.height - gy) * self.cell_size

    def _draw_static(self) -> None:
        runner = self.runner
        assert runner is not None
        world = runner.world
        cs = self.cell_size
        self.canvas.delete("all")
        self.canvas.configure(width=world.width * cs, height=world.height * cs)
        for obstacle in runner.env.obstacles:
            x0, y1 = self._xy(obstacle.position_x, obstacle.position_y)
            x1, y0 = self._xy(obstacle.position_x + obstacle.width,
                              obstacle.position_y + obstacle.height)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=COLOURS["obstacle"],
                                         outline=COLOURS["obstacle"], tags="obstacle")
        self._path_item = self.canvas.create_line(0, 0, 0, 0, fill=COLOURS["path"], width=2,
                                                  state="hidden")
        radius = max(3.0, 0.45 * cs)
        self._chest_items = {}
        self._chest_colours = {}
        for chest in runner.env.chests.values():
            cx, cy = self._xy(chest.position_x, chest.position_y)
            self._chest_items[chest.id] = self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius, width=2,
                outline=COLOURS["chest_unknown"], fill="")
        ring = runner.settings.simulation.interaction_radius * cs
        self._target_item = self.canvas.create_oval(-ring, -ring, ring, ring,
                                                    outline=COLOURS["target"], width=2,
                                                    state="hidden")
        agent_r = max(4.0, runner.env.agent.radius * cs * 1.6)
        self._agent_radius = agent_r
        self._agent_item = self.canvas.create_oval(0, 0, 0, 0, fill=COLOURS["agent"],
                                                   outline="#dbeafe")
        self._heading_item = self.canvas.create_line(0, 0, 0, 0, fill=COLOURS["heading"],
                                                     width=2)

    def _chest_colour(self, chest_id: int) -> tuple[str, str]:
        """(fill, outline) for a chest marker."""
        runner = self.runner
        assert runner is not None
        chest = runner.env.chests[chest_id]
        controller = runner.controller
        status: str | None = None
        if controller is not None and runner.is_visual:
            prior = runner.visual_controller.world.priors.get(chest_id)
            status = None if prior is None else prior.status.value
            visited = status in ("empty", "opened", "failed")
        elif isinstance(controller, AgentController):
            try:
                status = controller.tracker.status(chest_id).value
            except KeyError:  # not a candidate in this session
                status = None
            visited = status == "visited"
        else:
            visited = False
        unreachable = status in ("unreachable", "abandoned")
        if chest.opened:
            return COLOURS["chest_opened"], COLOURS["chest_opened"]
        if chest.present:
            return COLOURS["chest_available"], COLOURS["chest_available"]
        if unreachable:
            return "", COLOURS["chest_unreachable"]
        if visited:
            return COLOURS["chest_absent"], COLOURS["chest_absent"]
        return "", COLOURS["chest_unknown"]

    def render(self) -> None:
        runner = self.runner
        if runner is None:
            return
        env = runner.env
        agent = env.agent
        canvas = self.canvas

        for chest_id, item in self._chest_items.items():
            fill, outline = self._chest_colour(chest_id)
            key = fill + outline
            if self._chest_colours.get(chest_id) != key:
                canvas.itemconfigure(item, fill=fill, outline=outline)
                self._chest_colours[chest_id] = key

        ax, ay = self._xy(agent.position_x, agent.position_y)
        r = self._agent_radius
        canvas.coords(self._agent_item, ax - r, ay - r, ax + r, ay + r)
        hx = ax + math.cos(agent.heading) * r * 2.2
        hy = ay - math.sin(agent.heading) * r * 2.2
        canvas.coords(self._heading_item, ax, ay, hx, hy)

        controller = runner.controller
        path = controller.path if controller is not None else []
        if self.show_paths_var.get() and path:
            coords: list[float] = [ax, ay]
            for px, py in path:
                coords.extend(self._xy(px, py))
            canvas.coords(self._path_item, *coords)
            canvas.itemconfigure(self._path_item, state="normal")
        else:
            canvas.itemconfigure(self._path_item, state="hidden")

        target_id = agent.current_target
        goal = getattr(controller, "goal", None) if runner.is_visual else None
        if goal is not None and goal.kind == "chest":   # visual agent: its estimated target
            tx, ty = self._xy(*goal.position)
            ring = runner.settings.simulation.interaction_radius * self.cell_size
            canvas.coords(self._target_item, tx - ring, ty - ring, tx + ring, ty + ring)
            canvas.itemconfigure(self._target_item, state="normal")
        elif target_id is not None and target_id in env.chests:
            chest = env.chests[target_id]
            tx, ty = self._xy(chest.position_x, chest.position_y)
            ring = runner.settings.simulation.interaction_radius * self.cell_size
            canvas.coords(self._target_item, tx - ring, ty - ring, tx + ring, ty + ring)
            canvas.itemconfigure(self._target_item, state="normal")
        else:
            canvas.itemconfigure(self._target_item, state="hidden")
        canvas.tag_raise(self._agent_item)
        canvas.tag_raise(self._heading_item)

        self._update_info()
        if self.debug_settings is not None:
            self._render_debug()
        self.frames_rendered += 1

    def _update_info(self) -> None:
        runner = self.runner
        assert runner is not None
        m = runner.metrics
        env = runner.env
        if runner.finished:
            sim_state = f"finished: {m.outcome}"
        elif self.running:
            sim_state = "paused" if self.paused else f"running {self.speed}x"
        else:
            sim_state = "ready (press Start)"
        target = env.agent.current_target
        target_text = "-"
        goal = getattr(runner.controller, "goal", None) if runner.is_visual else None
        if goal is not None:
            target_text = (f"chest track T{goal.track_id}" if goal.kind == "chest"
                           else f"{goal.kind} {goal.cell}")
        elif target is not None and target in env.chests:
            target_text = f"#{target} {env.chests[target].location_name}"[:26]
        minutes = env.time / 60.0
        rate = f"{m.chests_opened / minutes:.2f}" if minutes > 0 else "-"
        total = m.candidate_locations or len(env.chests)
        values = {
            "simulation": sim_state,
            "agent_state": runner.state.value,
            "target": target_text,
            "time": f"{int(env.time // 60):02d}:{env.time % 60:04.1f} ({env.tick} ticks)",
            "visited": f"{m.locations_visited} / {total}",
            "found": str(m.chests_found),
            "opened": str(m.chests_opened),
            "rate": rate,
            "recoveries": str(m.recoveries),
            "unreachable": str(m.unreachable_targets + m.abandoned_targets),
            "rewards": str(m.rewards),
            "strategy": runner.strategy,
            "session": f"{runner.session_id or '-'} / {runner.seed}",
        }
        for key, value in values.items():
            if self.info_vars[key].get() != value:
                self.info_vars[key].set(value)


def run_gui(
    runner_factory: RunnerFactory,
    *,
    strategies: Sequence[str],
    strategy: str,
    cell_size: int,
    tick_rate: int,
    render_rate: int,
    speed: int = 1,
    on_finish: Callable[[SimulationRunner], None] | None = None,
    debug: VisualDebugSettings | None = None,
) -> SimulationApp:
    """Open the window and block in Tk's main loop until it is closed."""
    root = tk.Tk()
    app = SimulationApp(root, runner_factory, strategies=strategies, strategy=strategy,
                        cell_size=cell_size, tick_rate=tick_rate, render_rate=render_rate,
                        speed=speed, autostart=True, on_finish=on_finish, debug=debug)
    root.mainloop()
    return app
