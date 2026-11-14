"""Scoring of candidate destinations for adaptive exploration.

For a spawn point ``i``::

    S_i = (p_i * q_i + lambda * I_i + mu * C_i) / (T_i + eps)

* ``p_i`` - estimated chest probability (Bayesian posterior mean, ``spawn_estimator``);
* ``q_i`` - probability of reaching and opening it (route memory reliability);
* ``I_i`` - expected information gain: entropy of "is there a chest this episode" plus a
  weighted reduction of the posterior variance (``spawn_estimator.information_gain``);
* ``C_i`` - cluster value: nearby unresolved candidates, ``sum_j p_j q_j exp(-d_ij / r)``;
  one trip to a dense cluster pays several times;
* ``T_i`` - expected time: path length / speed x learned travel-time factor, plus the
  interaction and approach overhead.

All terms come from the agent's own estimates and history: no hidden simulator state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    exploration: float = 0.15          # lambda: weight of information gain
    learning: float = 4.0              # weight of posterior-variance reduction inside I_i
    cluster: float = 0.25              # mu
    cluster_radius_m: float = 12.0
    interaction_seconds: float = 2.0
    approach_overhead_s: float = 4.0
    epsilon_s: float = 1.0


@dataclass(frozen=True, slots=True)
class Candidate:
    key: int
    xyz: tuple[float, float, float]
    probability: float
    reliability: float
    entropy_bits: float
    variance_reduction: float
    travel_seconds: float


def information(c: Candidate, w: ScoringWeights) -> float:
    return c.entropy_bits + w.learning * c.variance_reduction * 12.0   # Var of Beta(1,1)=1/12


def cluster_value(c: Candidate, others: Sequence[Candidate], w: ScoringWeights) -> float:
    total = 0.0
    for o in others:
        if o.key == c.key:
            continue
        d = math.dist(c.xyz, o.xyz)
        if d <= 2.5 * w.cluster_radius_m:
            total += o.probability * o.reliability * math.exp(-d / w.cluster_radius_m)
    return total


def score(c: Candidate, others: Sequence[Candidate], w: ScoringWeights) -> float:
    value = c.probability * c.reliability + w.exploration * information(c, w) \
        + w.cluster * cluster_value(c, others, w)
    time = c.travel_seconds + w.interaction_seconds * c.probability + w.approach_overhead_s
    return value / (time + w.epsilon_s)


def baseline_score(probability: float, travel_seconds: float) -> float:
    """The Phase 4 rule (kept as the fallback planner): probability per second of travel."""
    return probability / (travel_seconds + 1.0)
