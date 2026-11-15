"""Controlled failure injection for robustness testing (Phase 7).

A :class:`FaultInjector` is attached to one research episode (``Runner3D(faults=...)``) and
injects one planned failure at a planned simulated time, for a planned duration. It acts on
the *runner's* side of the bridge or, for faults that belong to the world, on the reference
environment itself; it never gives the agent extra information.

| Kind | Where it acts | What the agent sees |
|---|---|---|
| ``frame_loss`` | bridge reply | requested frames do not arrive |
| ``corrupted_frame`` | bridge reply | frames with zeroed bands |
| ``false_detection`` | detector output | a chest detection where there is none |
| ``missed_detection`` | detector output | no detections at all |
| ``localisation_drift`` | bridge reply | odometry jumps by a few metres and stays offset |
| ``unexpected_obstacle`` | environment | a crate appears on the path, 3 m ahead |
| ``interaction_failure`` | environment | the next ``intensity`` interactions fail |
| ``bridge_disconnect`` | bridge connection | the connection drops (the client re-sends) |
| ``slow_inference`` | learned policy | the next 10 inferences take ``intensity`` ms longer |
| ``unreachable_target`` | navigation service | the next goal area asked for becomes unreachable |

Every injection is a structured record (kind, start, end, details). After the episode,
:meth:`FaultInjector.assess` decides from the agent's and supervisor's own records whether the
failure was **detected** (a related event or counter changed during or soon after the fault),
whether the agent **recovered** (it made progress after the fault ended, or finished the
episode normally) and whether an unrecovered episode **terminated cleanly** (an explicit
reason, no exception).
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

FAULT_KINDS: Final = ("frame_loss", "corrupted_frame", "false_detection", "missed_detection",
                      "localisation_drift", "unexpected_obstacle", "interaction_failure",
                      "bridge_disconnect", "slow_inference", "unreachable_target")

SLOW_INFERENCES: Final = 10
#: seconds after a fault ends within which a related signal counts as detection
DETECTION_GRACE_S: Final = 15.0
#: seconds after a fault ends within which the agent must make progress to count as recovered
RECOVERY_WINDOW_S: Final = 30.0

#: the agent's / supervisor's own signals that show a fault was noticed (None: the fault is
#: by nature indistinguishable from the world - nothing to see, or a plausible chest)
DETECTION_SIGNALS: Final[dict[str, tuple[str, ...] | None]] = {
    "frame_loss": ("sup:invalid_observation", "sup:invalid_observations"),
    "corrupted_frame": ("sup:invalid_observation", "sup:invalid_observations"),
    "false_detection": None,
    "missed_detection": None,
    "localisation_drift": ("localisation_jumps", "sup:localisation_jump"),
    "unexpected_obstacle": ("navigation_failures", "recoveries", "sup:stationary",
                            "sup:no_progress"),
    "interaction_failure": ("failed_interactions",),
    "bridge_disconnect": ("resends",),
    "slow_inference": ("fallback:timeout", "sup:inference_latency"),
    "unreachable_target": ("unreachable", "navigation_failures"),
}

#: default (start window seconds, duration seconds, intensity) per kind
DEFAULTS: Final[dict[str, tuple[tuple[float, float], float, float]]] = {
    "frame_loss": ((40.0, 120.0), 3.0, 1.0),
    "corrupted_frame": ((40.0, 120.0), 3.0, 0.3),
    "false_detection": ((40.0, 120.0), 2.0, 1.0),
    "missed_detection": ((40.0, 120.0), 8.0, 1.0),
    "localisation_drift": ((40.0, 120.0), 0.0, 2.5),
    "unexpected_obstacle": ((40.0, 120.0), 0.0, 1.0),
    "interaction_failure": ((30.0, 100.0), 90.0, 2.0),
    "bridge_disconnect": ((40.0, 120.0), 0.0, 1.0),
    "slow_inference": ((40.0, 120.0), 60.0, 35.0),
    "unreachable_target": ((40.0, 120.0), 0.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class FaultPlan:
    kind: str
    start_s: float
    duration_s: float
    intensity: float
    #: start only once the agent is in this state (e.g. ``VERIFYING_OPEN``), at or after
    #: ``start_s``; ``None``: at ``start_s``
    trigger_state: str | None = None

    @classmethod
    def random(cls, kind: str, seed: int) -> FaultPlan:
        if kind not in FAULT_KINDS:
            raise ValueError(f"fault kind must be one of {', '.join(FAULT_KINDS)}")
        (lo, hi), duration, intensity = DEFAULTS[kind]
        rng = random.Random(f"fault:{kind}:{seed}")
        return cls(kind, round(rng.uniform(lo, hi), 2), duration, intensity)


@dataclass(slots=True)
class Injection:
    kind: str
    start: float
    end: float
    details: dict[str, Any] = field(default_factory=dict)
    applied: int = 0


class FaultInjector:
    def __init__(self, plan: FaultPlan, *, seed: int = 0) -> None:
        self.plan = plan
        self.rng = random.Random(f"faults:{plan.kind}:{seed}")
        self.runner: Any = None
        self.injection: Injection | None = None
        self.finished = False
        self._offset = np.zeros(3)
        self._unreachable: tuple[float, float] | None = None
        self._saved_failure_p: float | None = None
        self._failed_base = 0
        self._counters_before: dict[str, float] = {}
        self._progress_before: tuple[int, int, int] | None = None
        self._progress_after_end: tuple[int, int, int] | None = None
        self._counters_detect: dict[str, float] | None = None
        self._progress_window: tuple[int, int, int] | None = None
        self._restore: list[tuple[Any, str, Any]] = []

    # -- wiring (Runner3D calls attach after the agent and client exist) ------------------------

    def attach(self, runner: Any) -> None:
        self.runner = runner
        client = runner.client
        original_step = client.step
        original_nav = client.nav_query

        def step(action: dict[str, Any], *, ticks: int = 1, observe: bool = False) -> Any:
            self._tick_before_step()
            result = original_step(action, ticks=ticks, observe=observe)
            return self._filter(result)

        def nav_query(start: Any, goal: Any, *, goal_tolerance: float = 0.5) -> Any:
            if self.injection is not None and self.plan.kind == "unreachable_target" \
                    and self._unreachable is None:
                # the goal area the agent asks for next becomes unreachable
                self._unreachable = (float(goal[0]), float(goal[1]))
                self.injection.details["area"] = [round(float(goal[0]), 2),
                                                  round(float(goal[1]), 2)]
            if self._unreachable is not None and math.dist(goal[:2], self._unreachable) <= 3.0:
                from chest_hunter.unreal_bridge.client import NavResult

                self._note("blocked a path query")
                return NavResult("unreachable", (), 0.0)
            return original_nav(start, goal, goal_tolerance=goal_tolerance)

        self._patch(client, "step", step)
        self._patch(client, "nav_query", nav_query)
        agent = getattr(runner.controller, "agent", runner.controller)
        detector = agent.detector
        original_detect = detector.detect

        def detect(frame: np.ndarray, timestamp: float, **kw: Any) -> Any:
            dets = original_detect(frame, timestamp, **kw)
            return self._filter_detections(frame, timestamp, dets)

        self._patch(detector, "detect", detect)
        policy = runner.learned_policy
        if policy is not None:
            original_predict = policy.predict

            def predict(frame: np.ndarray, telemetry: Any) -> Any:
                if self._active("slow_inference") and self.injection is not None \
                        and self.injection.applied < SLOW_INFERENCES:
                    time.sleep(self.plan.intensity / 1000.0)
                    out = original_predict(frame, telemetry)
                    self._note("slowed an inference")
                    return dataclasses.replace(out, latency_ms=out.latency_ms
                                               + self.plan.intensity)
                return original_predict(frame, telemetry)

            self._patch(policy, "predict", predict)

    def _patch(self, obj: Any, name: str, replacement: Any) -> None:
        had = name in vars(obj)
        self._restore.append((obj, name, vars(obj)[name] if had else None))
        setattr(obj, name, replacement)

    def detach(self) -> None:
        """Undo every patch (shared objects - a cached learned policy, a reused connection -
        must not carry the fault into the next episode)."""
        while self._restore:
            obj, name, previous = self._restore.pop()
            if previous is None:
                if name in vars(obj):
                    delattr(obj, name)
            else:
                setattr(obj, name, previous)
        env = self._env() if self.runner is not None else None
        if self.plan.kind == "interaction_failure" and env is not None \
                and self._saved_failure_p is not None:
            env.settings = dataclasses.replace(
                env.settings, interaction_failure_probability=self._saved_failure_p)

    # -- helpers ------------------------------------------------------------------------------

    def _agent(self) -> Any:
        return getattr(self.runner.controller, "agent", self.runner.controller)

    def _now(self) -> float:
        return float(self._agent().now)

    def _active(self, kind: str) -> bool:
        inj = self.injection
        return inj is not None and inj.kind == kind and inj.start <= self._now() <= inj.end

    def _note(self, what: str) -> None:
        if self.injection is not None:
            self.injection.applied += 1
            self.injection.details.setdefault("effects", {})
            self.injection.details["effects"][what] = \
                self.injection.details["effects"].get(what, 0) + 1

    def _progress(self) -> tuple[int, int, int]:
        agent = self._agent()
        resolved = sum(1 for p in agent.world.priors.values() if p.resolved)
        covered = agent.explorer.map.seen_count if getattr(agent, "explorer", None) else 0
        return (agent.metrics.chests_opened, resolved, covered)

    def _counters(self) -> dict[str, float]:
        agent = self._agent()
        m = agent.metrics
        sup = self.runner.supervisor
        sel = self.runner.selector
        return {"recoveries": m.recoveries, "navigation_failures": m.navigation_failures,
                "unreachable": m.unreachable_targets, "targets_lost": m.targets_lost,
                "failed_interactions": m.failed_interactions,
                "localisation_jumps": agent.localisation_jumps,
                "supervisor_events": sup.events.total if sup is not None else 0,
                "resends": self.runner.client.resends,
                "selector_fallbacks": sum(sel.stats.fallbacks.values()) if sel is not None
                else 0,
                "abandoned": m.abandoned_targets,
                **({f"sup:{k}": float(v) for k, v in sup.events.counts.items()}
                   if sup is not None else {}),
                **({f"fallback:{k}": float(v) for k, v in sel.stats.fallbacks.items()}
                   if sel is not None else {})}

    def _tick_before_step(self) -> None:
        if self.runner.controller is None:
            return
        now = self._now()
        plan = self.plan
        if self.finished and self.injection is not None:
            end = self.injection.end
            if self._counters_detect is None and now >= end + DETECTION_GRACE_S:
                self._counters_detect = self._counters()
            if self._progress_window is None and now >= end + RECOVERY_WINDOW_S:
                self._progress_window = self._progress()
        if self.injection is None and now >= plan.start_s and not self.finished and (
                plan.trigger_state is None
                or getattr(getattr(self._agent(), "state", None), "name", None)
                == plan.trigger_state):
            self._start(now)
        inj = self.injection
        if inj is not None and not self.finished and plan.kind == "unexpected_obstacle":
            if self._place_obstacle():
                inj.end = now
                self._stop()
            elif now > inj.end:
                inj.details["skipped"] = "no path segment to block within 60 s"
                self._stop()
        elif inj is not None and not self.finished and plan.kind == "interaction_failure":
            env = self._env()
            failed = env.failed_attempts - self._failed_base if env is not None else 0
            while inj.applied < failed:
                self._note("failed an interaction")
            if failed >= int(plan.intensity) or now > inj.end:
                self._stop()
        elif inj is not None and not self.finished and now > inj.end:
            self._stop()

    def _start(self, now: float) -> None:
        plan = self.plan
        self.injection = Injection(plan.kind, now, now + plan.duration_s,
                                   {"intensity": plan.intensity})
        self._counters_before = self._counters()
        self._progress_before = self._progress()
        env = self._env()
        if plan.kind == "localisation_drift":
            angle = self.rng.uniform(0.0, 2 * math.pi)
            self._offset = np.array([math.cos(angle), math.sin(angle), 0.0]) * plan.intensity
            self.injection.details["offset_m"] = [round(float(v), 3) for v in self._offset]
        elif plan.kind == "interaction_failure" and env is not None:
            self._saved_failure_p = env.settings.interaction_failure_probability
            self._failed_base = env.failed_attempts
            env.settings = dataclasses.replace(env.settings, interaction_failure_probability=1.0)
        elif plan.kind == "bridge_disconnect":
            sock = getattr(self.runner.client, "_sock", None)
            if sock is not None:
                sock.close()                    # the next request finds a dead connection
                self._note("closed the connection")
        if plan.duration_s <= 0:
            self.injection.end = now
        if plan.kind == "unexpected_obstacle":
            # placed once the agent is following a path (see _place_obstacle)
            self.injection.end = now + 60.0

    def _place_obstacle(self) -> bool:
        """Put a 2 m crate on the agent's current path, 3 m ahead: never on the agent, never
        on a chest, never on the path's last 1.5 m (the goal itself stays reachable or not
        according to the map, as a real obstacle would leave it)."""
        env = self._env()
        agent = self._agent()
        follower = getattr(agent, "follower", None)
        if env is None or follower is None or follower.done \
                or getattr(agent.state, "name", "") != "NAVIGATING":
            return False
        ch = env.character
        pts = [(ch.x, ch.y, ch.z), *follower.waypoints[follower.index:]]
        ahead, point = 3.0, None
        total = sum(math.dist(a[:2], b[:2]) for a, b in zip(pts, pts[1:], strict=False))
        if total < ahead + 1.5:
            return False
        for a, b in zip(pts, pts[1:], strict=False):
            seg = math.dist(a[:2], b[:2])
            if seg >= ahead:
                f = ahead / seg if seg else 0.0
                point = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, b[2])
                break
            ahead -= seg
        if point is None or math.dist(point[:2], (ch.x, ch.y)) < 2.6:
            return False
        if any(math.dist(point[:2], (c.x, c.y)) < 2.0 for c in env.chests):
            return False
        from chest_hunter.simulation3d.geometry import box

        cx, cy, z = point
        env.add_obstacle(box(cx - 1.0, cy - 1.0, z, cx + 1.0, cy + 1.0, z + 2.0,
                             "crate", "crate", name="injected obstacle"))
        assert self.injection is not None
        self.injection.details["obstacle_at"] = [round(cx, 2), round(cy, 2), round(z, 2)]
        self._note("placed an obstacle")
        return True

    def _stop(self) -> None:
        env = self._env()
        if self.plan.kind == "interaction_failure" and env is not None \
                and self._saved_failure_p is not None:
            env.settings = dataclasses.replace(
                env.settings, interaction_failure_probability=self._saved_failure_p)
        self.finished = True
        self._progress_after_end = self._progress()

    def _env(self) -> Any:
        server = self.runner.server
        session = getattr(server, "session", None) if server is not None else None
        return getattr(session, "env", None)

    def _filter(self, result: Any) -> Any:
        if self.injection is None:
            return result
        if self.plan.kind == "localisation_drift" and self.injection.start <= self._now():
            result = self._shift(result)
            if not self.injection.applied:
                self._note("shifted odometry")
        if result.observation is None:
            return result
        if self._active("frame_loss"):
            self._note("dropped a frame")
            return dataclasses.replace(result, observation=None)
        if self._active("corrupted_frame"):
            frame = np.array(result.observation.frame)
            h = frame.shape[0]
            band = max(1, int(h * self.plan.intensity))
            top = self.rng.randrange(0, max(1, h - band))
            frame[top:top + band] = 0
            frame.setflags(write=False)
            self._note("corrupted a frame")
            return dataclasses.replace(result, observation=dataclasses.replace(
                result.observation, frame=frame))
        return result

    def _shift(self, result: Any) -> Any:
        off = tuple(float(v) for v in self._offset)

        def odo(o: Any) -> Any:
            p = o.position
            return dataclasses.replace(o, position=(p[0] + off[0], p[1] + off[1], p[2]))

        tel = result.telemetry
        result = dataclasses.replace(result, telemetry=dataclasses.replace(
            tel, odometry=odo(tel.odometry)))
        if result.observation is not None:
            result = dataclasses.replace(result, observation=dataclasses.replace(
                result.observation, odometry=odo(result.observation.odometry)))
        return result

    def _filter_detections(self, frame: np.ndarray, timestamp: float, dets: list[Any]
                           ) -> list[Any]:
        if self._active("missed_detection"):
            if dets:
                self._note("hid detections")
            return []
        if self._active("false_detection"):
            from chest_hunter.vision.chest_detector_3d import Detection3D

            h, w = frame.shape[:2]
            size = max(8, w // 16)
            x = int(w * 0.55) - size // 2
            y = int(h * 0.62) - size // 2
            self._note("added a false detection")
            return [*dets, Detection3D("chest", (x, y, size, int(size * 0.7)), 0.85, timestamp,
                                       "injected")]
        return dets

    # -- assessment ------------------------------------------------------------------------------

    def assess(self, result: Any) -> dict[str, Any]:
        """Score one faulted episode from the agent's and supervisor's own records.

        * **detected** - one of the kind's :data:`DETECTION_SIGNALS` changed between the fault's
          start and ``DETECTION_GRACE_S`` after its end (``None`` for faults the agent cannot
          tell apart from the world: a missed chest, a plausible false chest);
        * **recovered** - the episode did not fail, and within ``RECOVERY_WINDOW_S`` after the
          fault ended the agent opened a chest, resolved a target or saw at least 10 new cells
          (an episode that ended normally inside that window counts as recovered);
        * **terminated_cleanly** - recovered, or the episode ended with an explicit reason.
        """
        inj = self.injection
        if inj is None:
            return {"kind": self.plan.kind, "injected": False,
                    "reason": "the episode ended before the planned start "
                              f"({self.plan.start_s:.0f} s)"}
        if not self.finished:
            self._stop()
        after = self._counters()
        window = self._counters_detect or after
        changed = {k: round(window[k] - self._counters_before.get(k, 0), 3) for k in window
                   if window[k] != self._counters_before.get(k, 0)}
        keys = DETECTION_SIGNALS[inj.kind]
        detected = None if keys is None else any(k in changed for k in keys)
        applied = inj.applied > 0
        normal_end = result.outcome in ("completed", "tick_limit") and \
            result.communication_failure is None
        base = self._progress_after_end
        later = self._progress_window or self._progress()
        progressed = base is not None and (later[0] > base[0] or later[1] > base[1]
                                           or later[2] >= base[2] + 10)
        ended_in_window = self._progress_window is None
        recovered = normal_end and (progressed or (ended_in_window
                                                   and result.outcome == "completed"))
        return {"kind": inj.kind, "injected": True, "applied": applied,
                "start_s": round(inj.start, 2), "end_s": round(inj.end, 2),
                "applications": inj.applied, "details": inj.details, "detected": detected,
                "signals": changed, "progress_after_fault": progressed,
                "recovered": recovered if applied else None,
                "outcome": result.outcome, "end_reason": result.end_reason,
                "terminated_cleanly": recovered or result.end_reason is not None}
