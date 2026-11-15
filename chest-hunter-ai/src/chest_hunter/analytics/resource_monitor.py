"""Process and system resource monitoring (Phase 6), standard library only.

``ResourceMonitor.sample()`` returns one structured snapshot::

    {"process_cpu_percent": ..., "process_rss_mb": ..., "process_threads": ...,
     "system_cpu_percent": ..., "system_memory_used_mb": ..., "system_memory_total_mb": ...,
     "gpu": {...} | None, "latency_ms": {"perception": ..., "planning": ..., ...}}

* **Process** values describe this Python process only. CPU percent is CPU time used since the
  previous sample divided by the wall time between them (it can exceed 100 % when several
  threads run, e.g. the in-process reference simulator). RSS is the *current* resident set
  (``/proc/self/statm``); where that is unavailable, the peak (``getrusage``) is reported and
  marked as such.
* **System** values describe the whole machine (``/proc/stat``, ``/proc/meminfo``), and are
  ``None`` where those files do not exist.
* **GPU** utilisation is read with ``nvidia-smi`` only when that tool exists; otherwise the
  field is ``None`` with the reason. Nothing is estimated or invented.
* **Latencies** come from a :class:`~chest_hunter.runtime.profiler.Profiler` (mean of the
  samples recorded since the previous snapshot, per component).
* **Unreal frame time** is not available from the Python reference simulator; the field is
  ``None`` unless a caller supplies it (``sample(frame_time_ms=...)``).

Samples are rate-limited by ``interval_seconds`` (wall clock) when taken with
``maybe_sample`` and kept in a bounded history.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

LATENCY_GROUPS: dict[str, tuple[str, ...]] = {
    "perception": ("preprocessing", "chest_detection", "tracking_localisation",
                   "interaction_cues"),
    "planning": ("route_selection",),
    "navigation": ("navigation_query",),
    "policy": ("learned_inference",),
    "action": ("action_generation",),
    "simulator_step": ("simulator_step",),
    "database": ("database_persistence",),
}


def process_rss_mb() -> tuple[float, str]:
    """(megabytes, source): current RSS where possible, else the peak."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            pages = int(statm.read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 2**20, "current"
        except (OSError, ValueError, IndexError):
            pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return peak / divisor, "peak"


def _system_cpu_times() -> tuple[int, int] | None:
    try:
        first = Path("/proc/stat").read_text().splitlines()[0].split()
    except (OSError, IndexError):
        return None
    values = [int(v) for v in first[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def system_memory_mb() -> tuple[float, float] | None:
    """(used, total) megabytes from /proc/meminfo, or None."""
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    total = info.get("MemTotal")
    avail = info.get("MemAvailable")
    if total is None or avail is None:
        return None
    return (total - avail) / 1024.0, total / 1024.0


def gpu_status() -> dict[str, Any]:
    """GPU utilisation via nvidia-smi when present; never estimated."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return {"available": False, "reason": "no GPU tool (nvidia-smi) on this machine"}
    try:
        out = subprocess.run([tool, "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}
    gpus = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"utilisation_percent": float(parts[0]), "memory_used_mb": float(parts[1]),
                         "memory_total_mb": float(parts[2])})
    return {"available": bool(gpus), "gpus": gpus}


class ResourceMonitor:
    def __init__(self, *, interval_seconds: float = 2.0, profiler: Any = None,
                 history: int = 1000, query_gpu: bool = True) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval = interval_seconds
        self.profiler = profiler
        self.samples: deque[dict[str, Any]] = deque(maxlen=history)
        self._gpu = gpu_status() if query_gpu else {"available": False,
                                                    "reason": "GPU query disabled"}
        self._wall = time.perf_counter()
        self._cpu = time.process_time()
        self._sys = _system_cpu_times()
        self._last_sample = -float("inf")
        self._seen: dict[str, tuple[int, float]] = {}
        self.peak_rss_mb = 0.0

    def maybe_sample(self, **extra: Any) -> dict[str, Any] | None:
        if time.perf_counter() - self._last_sample < self.interval:
            return None
        return self.sample(**extra)

    def sample(self, *, frame_time_ms: float | None = None, **extra: Any) -> dict[str, Any]:
        wall, cpu = time.perf_counter(), time.process_time()
        dt = max(1e-9, wall - self._wall)
        process_cpu = 100.0 * (cpu - self._cpu) / dt
        rss, rss_source = process_rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        system_cpu = None
        now_sys = _system_cpu_times()
        if now_sys is not None and self._sys is not None:
            total, idle = now_sys[0] - self._sys[0], now_sys[1] - self._sys[1]
            if total > 0:
                system_cpu = round(100.0 * (total - idle) / total, 2)
        mem = system_memory_mb()
        snapshot: dict[str, Any] = {
            "wall_seconds": round(wall, 3),
            "process_cpu_percent": round(process_cpu, 2),
            "process_rss_mb": round(rss, 2), "process_rss_source": rss_source,
            "process_threads": threading.active_count(),
            "system_cpu_percent": system_cpu,
            "system_memory_used_mb": round(mem[0], 1) if mem else None,
            "system_memory_total_mb": round(mem[1], 1) if mem else None,
            "gpu": self._gpu if self._gpu.get("available") else None,
            "gpu_note": None if self._gpu.get("available") else self._gpu.get("reason"),
            "unreal_frame_time_ms": frame_time_ms,
            "latency_ms": self._latencies(),
        }
        snapshot.update(extra)
        self._wall, self._cpu, self._sys = wall, cpu, now_sys
        self._last_sample = wall
        self.samples.append(snapshot)
        return snapshot

    def _latencies(self) -> dict[str, float | None]:
        """Mean latency per group over the samples recorded since the previous snapshot."""
        if self.profiler is None or not getattr(self.profiler, "enabled", False):
            return {}
        series = getattr(self.profiler, "_series", {})
        out: dict[str, float | None] = {}
        for group, names in LATENCY_GROUPS.items():
            count = total = 0.0
            for name in names:
                s = series.get(name)
                if s is None:
                    continue
                prev_count, prev_total = self._seen.get(name, (0, 0.0))
                count += s.count - prev_count
                total += s.total - prev_total
                self._seen[name] = (s.count, s.total)
            out[group] = round(1000.0 * total / count, 4) if count else None
        return out

    def summary(self) -> dict[str, Any]:
        xs = list(self.samples)
        cpu = [s["process_cpu_percent"] for s in xs]
        rss = [s["process_rss_mb"] for s in xs]
        return {"samples": len(xs), "process_cpu_percent_mean": round(sum(cpu) / len(cpu), 2)
                if cpu else None, "process_rss_mb_first": rss[0] if rss else None,
                "process_rss_mb_last": rss[-1] if rss else None,
                "process_rss_mb_peak": round(self.peak_rss_mb, 2),
                "gpu": self._gpu}


#: Top-level profiler sections of the agent's own thread (the Python decision pipeline).
#: Nested sections are inside these and must not be added again: ``action_generation``
#: contains route_selection, frontier_selection, navigation_query and learned_inference.
PIPELINE_SECTIONS: tuple[str, ...] = (
    "preprocessing", "chest_detection", "tracking_localisation", "interaction_cues",
    "exploration_update", "action_generation", "supervision", "database_persistence")
NESTED_SECTIONS: tuple[str, ...] = ("route_selection", "frontier_selection",
                                    "navigation_query", "learned_inference")


def pipeline_cpu_seconds(components: list[dict[str, Any]]) -> dict[str, float]:
    """CPU seconds of the decision pipeline (agent thread), the bridge client (agent thread,
    inside ``simulator_step``) from a :meth:`Profiler.report`."""
    by = {r["component"]: r for r in components}
    pipeline = sum(float(by[c]["cpu_total_s"]) for c in PIPELINE_SECTIONS if c in by)
    bridge = float(by["simulator_step"]["cpu_total_s"]) if "simulator_step" in by else 0.0
    return {"pipeline": pipeline, "bridge_client": bridge}


def resource_report(results: dict[str, Any], *, map_id: str, seed: int, policy: str,
                    exploration: str, observation: tuple[int, int]) -> dict[str, Any]:
    """Phase 7 resource report from ``profile-agent`` results (one entry per vision profile:
    ``{"components": Profiler.report(), "episodes": [...]}``). The Python decision pipeline
    and the simulator are reported separately; nothing is estimated."""
    gpu = gpu_status()
    out: dict[str, Any] = {
        "environment": "Python reference arena in the same process (NOT Unreal Engine)",
        "machine": {"cpus": os.cpu_count(), "platform": sys.platform,
                    "python": sys.version.split()[0]},
        "map": map_id, "seed": seed, "policy": policy, "exploration": exploration,
        "observation": list(observation), "gpu": gpu,
        "notes": [
            "pipeline_cpu: CPU time of the agent's own thread in the top-level profiled "
            "sections (perception, exploration update, action generation incl. planning, "
            "navigation queries and learned inference, supervision, persistence)",
            "simulator_cpu: process CPU minus the agent thread's sections; it is the Python "
            "reference arena (physics + software renderer) plus runner overhead and the "
            "evaluation-only scoring. It is NOT a measurement of the Unreal graphics workload, "
            "which needs a GPU and was not available",
            "memory: resident set size of the whole process after each episode",
        ],
        "profiles": {},
    }
    for name, res in results.items():
        eps = res["episodes"]
        cpu = pipeline_cpu_seconds(res["components"])
        sim_s = sum(float(e["sim_seconds"]) for e in eps)
        proc = sum(float(e["cpu_seconds"]) for e in eps)
        wall = sum(float(e["wall_seconds"]) for e in eps)
        frames = sum(int(e.get("frames_analysed") or 0) for e in eps)
        rss = [float(e["rss_mb"]) for e in eps if e.get("rss_mb") is not None]
        by = {r["component"]: r for r in res["components"]}
        out["profiles"][name] = {
            "episodes": len(eps), "simulated_seconds": round(sim_s, 1),
            "wall_seconds": round(wall, 1),
            "pipeline_cpu_seconds": round(cpu["pipeline"], 2),
            "pipeline_cpu_per_simulated_second": round(cpu["pipeline"] / sim_s, 4)
            if sim_s else None,
            "pipeline_cpu_ms_per_analysed_frame": round(1000 * cpu["pipeline"] / frames, 3)
            if frames else None,
            "pipeline_share_of_one_core_at_real_time": round(cpu["pipeline"] / sim_s, 4)
            if sim_s else None,
            "bridge_client_cpu_seconds": round(cpu["bridge_client"], 2),
            "process_cpu_seconds": round(proc, 2),
            "simulator_cpu_seconds": round(max(0.0, proc - cpu["pipeline"]
                                               - cpu["bridge_client"]), 2),
            "rss_mb": {"first": rss[0] if rss else None, "last": rss[-1] if rss else None,
                       "max": max(rss) if rss else None},
            "frames_analysed": frames,
            "components": {c: {k: by[c][k] for k in ("calls", "mean_ms", "p95_ms",
                                                      "cpu_total_s")}
                           for c in (*PIPELINE_SECTIONS, *NESTED_SECTIONS, "simulator_step")
                           if c in by},
            "per_episode": eps,
        }
    return out
