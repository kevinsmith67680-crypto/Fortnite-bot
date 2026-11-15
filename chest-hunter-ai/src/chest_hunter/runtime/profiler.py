"""Per-component processing-time profiler (Phase 6).

Code paths of the agent and runner are wrapped in named sections::

    with profiler.section("chest_detection"):
        detections = detector.detect(frame, ...)

Each section records its latency (``time.perf_counter``, wall clock) and the CPU time of the
calling thread (``time.thread_time``). The difference matters here: the reference simulator
runs in another thread of the same process, so a section that waits for it (the simulator
step, a navigation query) has a long latency but little CPU time of its own.

Samples are bounded: every component keeps exact count, sum and maximum, plus at most
``max_samples`` latencies for the percentiles (deterministic reservoir sampling once that
many have been seen). ``report()`` ranks components by total time with mean, median,
95th percentile and maximum. The disabled profiler (:data:`NULL_PROFILER`) costs one
attribute lookup per section.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

#: Components measured by the instrumented agent and runner (report order is by total time).
COMPONENTS = (
    "simulator_step",        # frame capture + action: bridge round trip incl. rendering
    "preprocessing",         # colour conversion and segmentation
    "chest_detection",
    "tracking_localisation", # world-model integration: tracking, range, empty votes
    "interaction_cues",
    "route_selection",       # choosing the next target (explore / target selection)
    "navigation_query",      # NavMesh path queries (A* in the research environment)
    "learned_inference",
    "action_generation",     # state-machine tick (includes route selection and queries)
    "supervision",
    "database_persistence",
)


@dataclass(slots=True)
class _Series:
    count: int = 0
    total: float = 0.0
    cpu_total: float = 0.0
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)


class Profiler:
    enabled = True

    def __init__(self, *, max_samples: int = 20_000, seed: int = 0) -> None:
        if max_samples < 10:
            raise ValueError("max_samples must be at least 10")
        self.max_samples = max_samples
        self._series: dict[str, _Series] = {}
        self._rng = random.Random(seed)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        wall0, cpu0 = time.perf_counter(), time.thread_time()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - wall0, time.thread_time() - cpu0)

    def add(self, name: str, seconds: float, cpu_seconds: float = 0.0) -> None:
        s = self._series.get(name)
        if s is None:
            s = self._series[name] = _Series()
        s.count += 1
        s.total += seconds
        s.cpu_total += cpu_seconds
        s.maximum = max(s.maximum, seconds)
        if len(s.samples) < self.max_samples:
            s.samples.append(seconds)
        else:                                   # reservoir sampling (seeded: reproducible)
            j = self._rng.randrange(s.count)
            if j < self.max_samples:
                s.samples[j] = seconds

    def merge(self, other: Profiler) -> None:
        for name, s in other._series.items():
            mine = self._series.setdefault(name, _Series())
            mine.count += s.count
            mine.total += s.total
            mine.cpu_total += s.cpu_total
            mine.maximum = max(mine.maximum, s.maximum)
            room = self.max_samples - len(mine.samples)
            mine.samples.extend(s.samples[:max(0, room)])

    def components(self) -> list[str]:
        return sorted(self._series)

    def report(self) -> list[dict[str, Any]]:
        grand = sum(s.total for s in self._series.values()) or 1.0
        rows: list[dict[str, Any]] = []
        for name, s in self._series.items():
            xs = sorted(s.samples)
            rows.append({
                "component": name, "calls": s.count,
                "mean_ms": round(1000.0 * s.total / s.count, 4) if s.count else None,
                "median_ms": round(1000.0 * _quantile(xs, 0.5), 4) if xs else None,
                "p95_ms": round(1000.0 * _quantile(xs, 0.95), 4) if xs else None,
                "max_ms": round(1000.0 * s.maximum, 4),
                "total_s": round(s.total, 4), "cpu_total_s": round(s.cpu_total, 4),
                "share_of_measured": round(s.total / grand, 4)})
        rows.sort(key=lambda r: (-float(r["total_s"]), str(r["component"])))
        return rows


class _NullProfiler(Profiler):
    enabled = False

    def __init__(self) -> None:
        super().__init__()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        yield

    def add(self, name: str, seconds: float, cpu_seconds: float = 0.0) -> None:
        return None


NULL_PROFILER: Profiler = _NullProfiler()


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def format_report(rows: list[dict[str, Any]]) -> str:
    lines = ["| Component | Calls | Mean ms | Median ms | p95 ms | Max ms | Total s | "
             "CPU s (own thread) | Share |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['component']} | {r['calls']} | {r['mean_ms']:.3f} | "
                     f"{r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['max_ms']:.3f} | "
                     f"{r['total_s']:.2f} | {r['cpu_total_s']:.2f} | "
                     f"{100 * r['share_of_measured']:.1f} % |")
    return "\n".join(lines)
