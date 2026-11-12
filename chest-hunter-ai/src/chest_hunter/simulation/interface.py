"""The only gateway between the visual agent and the simulator.

:class:`VisualSimulatorInterface` exposes exactly what a visual policy may
use:

* ``capture()``: a rendered camera frame (pixels, timestamp, camera geometry);
* ``move(vx, vy)``: a velocity command (the simulator applies physics and
  collisions);
* ``hold_interact()``: hold the interact key for one tick. It returns
  nothing: success must be verified visually;
* static map metadata: grid size, tick length, current simulated time.

The environment itself is held in a private attribute. The
:attr:`evaluation_environment` property is for evaluation code (metrics,
benchmarks, tests) and must never be used by the policy; an import audit in
the test suite enforces this.
"""

from __future__ import annotations

from chest_hunter.simulation.camera import SensorModel
from chest_hunter.simulation.environment import Environment, InteractionResult
from chest_hunter.vision.observation import SimulatorObservationProvider, VisualObservation


class VisualSimulatorInterface:
    def __init__(self, env: Environment, *, width: int = 640, height: int = 360,
                 view_width_cells: int = 40, sensor: SensorModel | None = None,
                 seed: int = 0) -> None:
        self.__env = env
        self.provider = SimulatorObservationProvider(
            env, width=width, height=height, view_width_cells=view_width_cells,
            sensor=sensor, seed=seed)
        self._moved = False
        self.commands = 0
        self.interact_ticks = 0
        self.last_interaction: InteractionResult | None = None  # for tests/metrics only

    # -- permitted metadata ----------------------------------------------------------

    @property
    def map_width(self) -> int:
        return self.__env.grid.width

    @property
    def map_height(self) -> int:
        return self.__env.grid.height

    @property
    def dt(self) -> float:
        return self.__env.dt

    @property
    def time(self) -> float:
        return self.__env.time

    # -- observation ------------------------------------------------------------------

    def capture(self) -> VisualObservation | None:
        return self.provider.capture()

    # -- actions ----------------------------------------------------------------------

    def move(self, vx: float, vy: float) -> None:
        """Command a velocity for this tick (speed and acceleration limits apply)."""
        env = self.__env
        env.agent.steer(vx, vy, env.dt)
        env.agent.step(env.grid, env.dt)
        self._moved = True
        self.commands += 1

    def hold_interact(self) -> None:
        """Hold the interact key for this tick. The outcome is *not* returned."""
        self.last_interaction = self.__env.hold_interact()
        self.interact_ticks += 1

    def end_tick(self) -> None:
        """Called by the runner once per tick: a tick without a command brakes."""
        if not self._moved:
            env = self.__env
            env.agent.brake(env.dt)
            env.agent.step(env.grid, env.dt)
        self._moved = False

    # -- evaluation only ----------------------------------------------------------------

    @property
    def evaluation_environment(self) -> Environment:
        """Ground truth for metrics and tests. Never use this in a policy."""
        return self.__env
