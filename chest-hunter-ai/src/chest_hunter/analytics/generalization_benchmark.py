"""Familiar vs unseen maps (Phase 5 generalisation test).

* **familiar**: procedural *training* maps (the demonstrations were recorded there), played
  with seeds that were never used for training;
* **unseen**: procedural *test* maps, whose layouts, obstacles, chest locations, lighting,
  chest appearance and start poses were never seen by any trained component.

Each (strategy, map) pair plays its episodes in sequence with online adaptation (the
adaptive strategies update their spawn estimates and route memory after every episode),
so the results also show how throughput develops over repeated experiments on a new map.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from chest_hunter.analytics.policy_comparison import (
    DEFAULT_STRATEGIES,
    run_policy_benchmark,
)
from chest_hunter.analytics.stats import summarise
from chest_hunter.config import Settings
from chest_hunter.exceptions import ValidationError
from chest_hunter.simulation3d.procedural import map_id

#: Seeds for evaluation; the demonstrations use seed 1000, the benchmark seed 42.
EVALUATION_SEED = 5000


def run_generalization_benchmark(settings: Settings, *, familiar: int = 4, unseen: int = 4,
                                 episodes: int = 6, strategies: Sequence[str] = DEFAULT_STRATEGIES,
                                 model_dir: str | Path | None = None,
                                 experience_db: str | Path | None = None, workers: int = 1,
                                 max_ticks: int = 18000, output_dir: str | Path | None = None,
                                 seed: int = EVALUATION_SEED,
                                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if seed == 1000:
        raise ValidationError("seed 1000 is the demonstration (training) seed; use another")
    groups = {"familiar": [map_id("train", i) for i in range(familiar)],
              "unseen": [map_id("test", i) for i in range(unseen)]}
    out = Path(output_dir) if output_dir else None
    results: dict[str, Any] = {}
    for group, maps in groups.items():
        if not maps:
            continue
        results[group] = run_policy_benchmark(
            settings, strategies=strategies, map_ids=maps, seed=seed, episodes=episodes,
            model_dir=model_dir, experience_db=experience_db, workers=workers,
            max_ticks=max_ticks, online=True, output_dir=out / group if out else None,
            progress=progress)
    gap: dict[str, Any] = {}
    if "familiar" in results and "unseen" in results:
        for s in strategies:
            f = results["familiar"]["per_strategy"][s]["throughput_per_min"]
            u = results["unseen"]["per_strategy"][s]["throughput_per_min"]
            if f.get("mean") is not None and u.get("mean") is not None:
                gap[s] = {"familiar_mean": f["mean"], "unseen_mean": u["mean"],
                          "difference": round(u["mean"] - f["mean"], 4),
                          "percent_change": round(100.0 * (u["mean"] - f["mean"]) / f["mean"], 2)
                          if f["mean"] else None}
    summary = {"groups": {g: [str(m) for m in maps] for g, maps in groups.items()},
               "seed": seed, "episodes_per_map": episodes, "gap_unseen_minus_familiar": gap,
               "results": {g: {k: v for k, v in r.items() if k != "rows"}
                           for g, r in results.items()}}
    return summary


def learning_curve(rows: Sequence[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    """Throughput by position in the episode sequence (online adaptation)."""
    by_order: dict[int, list[float]] = {}
    for r in rows:
        if r["strategy"] == strategy:
            by_order.setdefault(int(r["order"]), []).append(float(r["throughput_per_min"]))
    return [{"order": k, **summarise(v)} for k, v in sorted(by_order.items())]
