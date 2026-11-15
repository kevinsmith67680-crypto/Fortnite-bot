"""Phase 6 chest-search planning: scoring, short-horizon sequences, budgets, ties."""

from __future__ import annotations

import itertools
import math
import unittest
from types import SimpleNamespace
from typing import Any

from chest_hunter.exploration.advanced_planner import (
    AdvancedPlanner,
    PlanningConfig,
    Target,
)
from chest_hunter.localization.world_model_3d import PriorStatus3D, SpawnPrior3D


def prior(loc: int, x: float, y: float, p: float) -> SpawnPrior3D:
    return SpawnPrior3D(loc, f"P{loc}", x, y, 0.0, p)


class FakeWorld:
    def __init__(self, priors: list[SpawnPrior3D]) -> None:
        self.priors = {p.location_id: p for p in priors}

    def unresolved(self) -> list[SpawnPrior3D]:
        return [p for p in self.priors.values() if not p.resolved]


def planner_for(priors: list[SpawnPrior3D], strategy: str, **cfg: Any) -> AdvancedPlanner:
    planner = AdvancedPlanner(config=PlanningConfig(strategy=strategy, **cfg))
    agent = SimpleNamespace(world=FakeWorld(priors), _plan_failures={}, _attempts={},
                            planner=None)
    planner.attach(agent)
    return planner


def choose(planner: AdvancedPlanner, pos: tuple[float, float, float],
           speed: float = 5.0) -> int:
    """What the agent's EXPLORING step does: real path length = straight distance here."""
    pool = planner.agent.world.unresolved()
    scored = [(planner.inspection_score(p, pos, math.dist(pos, p.xyz), speed, pool),
               -p.location_id, p.location_id) for p in pool]
    return max(scored)[2]


#: An isolated spawn point slightly closer than one at the edge of a cluster; all have the
#: same probability, so only the geometry differs.
CLUSTER = [prior(1, 10.0, 0.0, 0.5), prior(2, -12.0, 0.0, 0.5), prior(3, -15.0, 3.0, 0.5),
           prior(4, -15.0, -3.0, 0.5), prior(5, -18.0, 0.0, 0.5)]


class Scoring(unittest.TestCase):
    def test_single_target_score_formula(self) -> None:
        planner = planner_for([], "greedy", exploration_weight=0.2, recovery_weight=0.01,
                              epsilon_s=1.0)
        t = Target(1, (0.0, 0.0, 0.0), p=0.6, q=0.8, info=0.5, risk=2.0)
        expected = (0.6 * 0.8 + 0.2 * 0.5) / (10.0 + 1.0) - 0.01 * 2.0
        self.assertAlmostEqual(planner.single_score(t, 10.0), expected, places=12)

    def test_seconds_include_travel_approach_and_interaction(self) -> None:
        planner = planner_for([], "greedy", interaction_seconds=2.0, approach_overhead_s=4.0)
        t = Target(1, (10.0, 0.0, 0.0), p=0.5, q=0.9, info=0.0, risk=0.1)
        self.assertAlmostEqual(planner._seconds(t, 20.0, 5.0, (0.0, 0.0, 0.0)),
                               20.0 / 5.0 + 4.0 + 2.0 * 0.5)

    def test_higher_probability_and_shorter_trips_score_higher(self) -> None:
        planner = planner_for([], "greedy")
        near = Target(1, (0.0, 0.0, 0.0), 0.6, 0.9, 0.3, 0.1)
        likely = Target(2, (0.0, 0.0, 0.0), 0.9, 0.9, 0.3, 0.1)
        self.assertGreater(planner.single_score(near, 5.0), planner.single_score(near, 25.0))
        self.assertGreater(planner.single_score(likely, 10.0), planner.single_score(near, 10.0))

    def test_unknown_history_falls_back_to_the_configured_probability(self) -> None:
        planner = planner_for(CLUSTER, "greedy")
        t = planner.prior_target(CLUSTER[2])
        self.assertAlmostEqual(t.p, 0.5)
        self.assertAlmostEqual(t.q, 0.9)                    # no route memory


class Sequences(unittest.TestCase):
    def test_greedy_takes_the_closer_isolated_spawn_point(self) -> None:
        self.assertEqual(choose(planner_for(CLUSTER, "greedy"), (0.0, 0.0, 0.0)), 1)

    def test_short_horizon_prefers_the_productive_cluster(self) -> None:
        self.assertEqual(choose(planner_for(CLUSTER, "short_horizon"), (0.0, 0.0, 0.0)), 2)

    def test_sequence_value_rewards_nearby_followers(self) -> None:
        planner = planner_for(CLUSTER, "short_horizon")
        pool = planner._pool(CLUSTER, (0.0, 0.0, 0.0), 5.0)
        a = planner.prior_target(CLUSTER[0])
        b = planner.prior_target(CLUSTER[1])
        va, _ = planner.best_sequence(a, planner._seconds(a, 10.0, 5.0, (0, 0, 0)), pool, 5.0)
        vb, seq = planner.best_sequence(b, planner._seconds(b, 12.0, 5.0, (0, 0, 0)), pool, 5.0)
        self.assertGreater(vb, va)
        self.assertEqual(seq[0], 2)
        self.assertEqual(len(seq), 3)                     # planning_horizon
        self.assertTrue(set(seq[1:]) <= {3, 4, 5})

    def test_visible_chest_choice_considers_what_follows(self) -> None:
        planner = planner_for(CLUSTER[1:], "short_horizon")
        lone = SimpleNamespace(track_id=7, location_id=None, xyz=(9.0, 0.0, 0.0))
        edge = SimpleNamespace(track_id=8, location_id=None, xyz=(-11.0, 0.0, 0.0))
        self.assertIs(planner.choose_track([lone, edge], (0.0, 0.0, 0.0), 5.0), edge)
        greedy = planner_for(CLUSTER[1:], "greedy")
        self.assertIs(greedy.choose_track([lone, edge], (0.0, 0.0, 0.0), 5.0), lone)

    def test_repeated_failures_lower_a_destination(self) -> None:
        twins = [prior(1, 10.0, 0.0, 0.5), prior(2, -10.0, 0.0, 0.5)]
        planner = planner_for(twins, "greedy")
        self.assertEqual(choose(planner, (0.0, 0.0, 0.0)), 1)      # tie -> lower id
        planner.agent._plan_failures[1] = 2                        # failed twice this episode
        self.assertEqual(choose(planner, (0.0, 0.0, 0.0)), 2)
        self.assertGreater(planner.prior_target(twins[0]).risk,
                           planner.prior_target(twins[1]).risk)

    def test_resolved_spawn_points_are_never_candidates(self) -> None:
        priors = [prior(1, 5.0, 0.0, 0.9), prior(2, 30.0, 0.0, 0.2)]
        priors[0].status = PriorStatus3D.EMPTY
        planner = planner_for(priors, "short_horizon")
        self.assertEqual([p.location_id for p in planner.agent.world.unresolved()], [2])
        self.assertEqual(choose(planner, (0.0, 0.0, 0.0)), 2)


class Bounds(unittest.TestCase):
    def test_ties_are_broken_by_spawn_point_id(self) -> None:
        sym = [prior(4, 10.0, 0.0, 0.5), prior(2, -10.0, 0.0, 0.5), prior(3, 0.0, 10.0, 0.5),
               prior(5, 0.0, -10.0, 0.5)]
        for strategy in ("greedy", "short_horizon"):
            results = {choose(planner_for(list(order), strategy), (0.0, 0.0, 0.0))
                       for order in itertools.permutations(sym)}
            self.assertEqual(results, {2}, strategy)

    def test_work_is_bounded_by_candidates_and_beam(self) -> None:
        many = [prior(k, (k % 7) * 6.0, (k // 7) * 6.0, 0.3 + 0.01 * k) for k in range(1, 41)]
        planner = planner_for(many, "short_horizon", maximum_candidates=10, beam_width=4,
                              planning_horizon=3)
        first = planner.prior_target(many[0])
        pool = planner._pool(many, (0.0, 0.0, 0.0), 5.0)
        self.assertEqual(len(pool), 10)
        planner.stats.sequences_evaluated = 0
        planner.best_sequence(first, 5.0, pool, 5.0)
        # level 1: one sequence x <= 10 candidates; level 2: <= 4 sequences x 10 candidates
        self.assertLessEqual(planner.stats.sequences_evaluated, 10 + 4 * 10)

    def test_the_time_budget_stops_the_search(self) -> None:
        planner = planner_for(CLUSTER, "short_horizon", planning_budget_ms=15.0)
        ticks = iter(k * 0.020 for k in range(1000))             # 20 ms per clock reading
        planner._clock = lambda: next(ticks)
        value = choose(planner, (0.0, 0.0, 0.0))
        self.assertIn(value, {1, 2, 3, 4, 5})
        self.assertGreater(planner.stats.budget_overruns, 0)

    def test_latency_is_measured_per_decision(self) -> None:
        planner = planner_for(CLUSTER, "short_horizon")
        choose(planner, (0.0, 0.0, 0.0))
        stats = planner.stats.to_dict()
        self.assertEqual(stats["decisions"], len(CLUSTER))
        self.assertIsNotNone(stats["latency_mean_ms"])

    def test_configuration_is_validated(self) -> None:
        for bad in ({"strategy": "global"}, {"planning_horizon": 0}, {"beam_width": 0},
                    {"maximum_candidates": 0}, {"planning_budget_ms": 0.0},
                    {"detour_factor": 0.5}, {"recovery_weight": -1.0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                PlanningConfig(**bad)


class InEpisode(unittest.TestCase):
    def test_both_strategies_run_a_real_episode(self) -> None:
        from chest_hunter.agent.controller_3d import Agent3DConfig
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.database import DatabaseManager
        from chest_hunter.scenarios import seed_arena_3d
        from tests.phase5_helpers import settings

        for strategy in ("greedy", "short_horizon"):
            db = DatabaseManager(":memory:")
            db.initialize()
            self.addCleanup(db.close)
            seed_arena_3d(db)
            planner = AdvancedPlanner(config=PlanningConfig(strategy=strategy))
            runner = Runner3D.from_settings(db, settings(), mode="hybrid", seed=2,
                                            backend="reference", max_ticks=900, width=320,
                                            height=180, planner=planner,
                                            config=Agent3DConfig(smart_recovery=True))
            res = runner.run()
            self.assertIn(res.outcome, ("completed", "tick_limit"), strategy)
            self.assertGreater(planner.stats.decisions, 0, strategy)
            self.assertEqual(planner.stats.budget_overruns, 0, strategy)


if __name__ == "__main__":
    unittest.main()
