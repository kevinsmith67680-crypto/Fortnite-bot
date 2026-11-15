"""Named agent configurations and experience partitions (Phase 6).

The Phase 6 benchmark, endurance runs and ``simulate-3d`` build the agent from the same
:class:`AgentSetup` description, so a configuration means the same thing everywhere.

Experience partitions: route statistics and spawn estimates are only combined across runs
of the same map and layout (Phase 5) and, from Phase 6 on, the same **environment
version** (layout + environment settings), **agent version** (code version, control policy
and model) and **simulation configuration** (observation size, field of view, decision
interval, planner and compute settings). The partition is stored with every run
(``simulation_runs.parameters.partition``) and with every route record
(``route_memory.partition_key``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Final

from chest_hunter.agent.controller_3d import Agent3DConfig, apply_vision_profile
from chest_hunter.exceptions import ValidationError
from chest_hunter.runtime.checkpoint import config_hash

#: Version of the agent's decision logic. Bump it when a change alters behaviour, so new
#: experience is not mixed with records from an older controller.
AGENT_VERSION: Final = "7.0.0"

PLANNERS: Final = ("none", "adaptive", "greedy", "short_horizon")


@dataclass(frozen=True, slots=True)
class AgentSetup:
    name: str
    policy: str                  # deterministic | learned | hybrid
    planner: str                 # none (Phase 4 rule) | adaptive (Phase 5) | greedy | short_horizon
    supervisor: bool
    adaptive_compute: bool
    smart_recovery: bool
    selector_monitors: bool      # Phase 6 oscillation / failed-hold monitors in the selector
    description: str

    def __post_init__(self) -> None:
        if self.policy not in ("deterministic", "learned", "hybrid"):
            raise ValidationError(f"unknown policy {self.policy!r}")
        if self.planner not in PLANNERS:
            raise ValidationError(f"planner must be one of {', '.join(PLANNERS)}")


PHASE6_SETUPS: Final[dict[str, AgentSetup]] = {s.name: s for s in (
    AgentSetup("phase4_deterministic", "deterministic", "none", False, False, False, False,
               "Phase 4 deterministic visual controller (baseline)"),
    AgentSetup("phase5_hybrid", "hybrid", "none", False, False, False, False,
               "Phase 5 hybrid: learned approach steering when confident"),
    AgentSetup("phase6_hybrid_supervised", "hybrid", "none", True, False, False, True,
               "Phase 5 hybrid + reliability supervisor and selector monitors"),
    AgentSetup("phase6_hybrid_planning", "hybrid", "short_horizon", True, False, True, True,
               "+ short-horizon planning and classified recovery"),
    AgentSetup("phase6_hybrid_optimised", "hybrid", "short_horizon", True, True, True, True,
               "+ adaptive computation (efficient vision profile, unchanged frames skipped)"),
    AgentSetup("phase6_deterministic_supervised", "deterministic", "none", True, False, False,
               False, "ablation: Phase 4 controller + reliability supervisor"),
    AgentSetup("phase6_deterministic_optimised", "deterministic", "short_horizon", True, True,
               True, False, "ablation: Phase 4 controller + supervisor, short-horizon "
               "planning, classified recovery and adaptive computation"),
)}
DEFAULT_PHASE6_SETUPS: Final = ("phase4_deterministic", "phase5_hybrid",
                                "phase6_hybrid_supervised", "phase6_hybrid_planning",
                                "phase6_hybrid_optimised")


def custom_setup(*, policy: str, planner: str, supervisor: bool, adaptive_compute: bool,
                 smart_recovery: bool | None = None) -> AgentSetup:
    smart = planner in ("adaptive", "greedy", "short_horizon") if smart_recovery is None \
        else smart_recovery
    return AgentSetup("custom", policy, planner, supervisor, adaptive_compute, smart,
                      supervisor and policy != "deterministic", "command-line configuration")


def agent_config(settings: Any, setup: AgentSetup) -> Agent3DConfig:
    cfg = Agent3DConfig(decision_ticks=settings.simulation_3d.decision_ticks,
                        smart_recovery=setup.smart_recovery)
    if setup.adaptive_compute:
        perf = settings.performance
        cfg = apply_vision_profile(cfg, perf.vision_profile if perf.adaptive_vision
                                   else "phase5", skip_unchanged=perf.skip_unchanged_frames)
    return cfg


def supervisor_config(settings: Any, setup: AgentSetup) -> Any:
    if not setup.supervisor or not settings.supervisor.enabled:
        return None
    from chest_hunter.runtime.supervisor import SupervisorConfig

    s = settings.supervisor
    return SupervisorConfig(
        mode=s.mode, max_state_seconds=s.max_state_seconds,
        max_recovery_attempts=s.max_recovery_attempts, max_interventions=s.max_interventions,
        loop_detection_window=s.loop_detection_window, loop_repetitions=s.loop_repetitions,
        max_no_progress_seconds=s.max_no_progress_seconds,
        stationary_seconds=s.stationary_seconds, max_goal_plans=s.max_goal_plans,
        max_interaction_failures=s.max_interaction_failures,
        max_invalid_observations=s.max_invalid_observations,
        inference_budget_ms=s.inference_budget_ms)


def policy_options(settings: Any, setup: AgentSetup) -> dict[str, Any]:
    from chest_hunter.learning.policy_selector import SafetyLimits

    lc = settings.learning
    limits = SafetyLimits(timeout_ms=lc.inference_timeout_ms, confidence=lc.hybrid_confidence,
                          learned_observation_rate=lc.learned_observation_rate)
    if setup.selector_monitors:
        limits = replace(limits, oscillation_flips=8, max_failed_holds=5)
    defaults = SafetyLimits()
    if limits == defaults:
        return {}
    return {"limits": limits}


def make_planner(settings: Any, setup: AgentSetup, db: Any, map_id: str,
                 layout_version: str | None, *, partition_key: str | None = None) -> Any:
    if setup.planner == "none":
        return None
    lc = settings.learning
    if setup.planner == "adaptive":
        from chest_hunter.exploration.adaptive_planner import AdaptivePlanner
        from chest_hunter.exploration.frontier_scoring import ScoringWeights

        return AdaptivePlanner.from_database(
            db, map_id, layout_version, prior_strength=lc.spawn_prior_strength,
            weights=ScoringWeights(exploration=lc.exploration_weight, cluster=lc.cluster_weight))
    from chest_hunter.exploration.advanced_planner import AdvancedPlanner, PlanningConfig
    from chest_hunter.exploration.route_memory import RouteMemory
    from chest_hunter.exploration.spawn_estimator import SpawnEstimator

    pc = settings.planning
    config = PlanningConfig(strategy=setup.planner, planning_horizon=pc.planning_horizon,
                            beam_width=pc.beam_width, maximum_candidates=pc.maximum_candidates,
                            planning_budget_ms=pc.planning_budget_ms,
                            exploration_weight=pc.exploration_weight,
                            recovery_weight=pc.recovery_weight, detour_factor=pc.detour_factor)
    estimates = SpawnEstimator(db, prior_strength=lc.spawn_prior_strength).estimate(
        map_id, layout_version)
    memory = RouteMemory(db, map_id, layout_version, partition_key=partition_key,
                         include_legacy=True)
    return AdvancedPlanner(estimates, memory, config)


# -- experience partitions --------------------------------------------------------------------

def environment_version(map_id: str, env_overrides: dict[str, Any] | None) -> str:
    from chest_hunter.agent.runner_3d import _layout_hash

    return config_hash({"map_id": map_id, "layout": _layout_hash(map_id),
                        "environment": dict(sorted((env_overrides or {}).items()))})


def partition(*, map_id: str, env_overrides: dict[str, Any] | None, setup: AgentSetup,
              model_version: str | None, width: int, height: int, fov_deg: float,
              decision_ticks: int) -> dict[str, str]:
    agent = f"{AGENT_VERSION}/{setup.policy}/{model_version or '-'}"
    sim = config_hash({"observation": [width, height], "fov_deg": fov_deg,
                       "decision_ticks": decision_ticks, "planner": setup.planner,
                       "adaptive_compute": setup.adaptive_compute,
                       "smart_recovery": setup.smart_recovery,
                       # the supervisor and the selector monitors change behaviour too: their
                       # episodes must not share a partition with unsupervised ones
                       "supervisor": setup.supervisor,
                       "selector_monitors": setup.selector_monitors})
    env = environment_version(map_id, env_overrides)
    return {"map_id": map_id, "environment_version": env, "agent_version": agent,
            "simulation_config": sim, "key": f"env={env};agent={agent};config={sim}"}


def is_test_map(map_id: str) -> bool:
    return map_id.startswith("proc_test_")


def guard_experience_write(map_id: str, *, allow_test_maps: bool) -> None:
    """Held-out test maps must not leak into persistent training experience."""
    if is_test_map(map_id) and not allow_test_maps:
        raise ValidationError(
            f"refusing to store experience from held-out test map {map_id!r} in a persistent "
            "experience database (pass allow_test_maps / --allow-test-maps to do it on "
            "purpose, e.g. for an online-adaptation study on a copy)")


def setup_dict(setup: AgentSetup) -> dict[str, Any]:
    return asdict(setup)
