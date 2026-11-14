"""Small statistics helpers (no SciPy): Beta distribution and Student-t intervals."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) by bisection on the CDF (accurate to ~1e-10)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def beta_interval(a: float, b: float, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2
    return beta_ppf(tail, a, b), beta_ppf(1.0 - tail, a, b)


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * beta_cdf(x, df / 2.0, 0.5)
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Student-t quantile (bisection), e.g. t_ppf(0.975, 99) = 1.984."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarise(values: Sequence[float], level: float = 0.95) -> dict[str, Any]:
    """Mean, median, standard deviation and a t-based confidence interval for the mean."""
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "ci_low": None,
                "ci_high": None, "min": None, "max": None}
    mean = statistics.fmean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    half = t_ppf(0.5 + level / 2, n - 1) * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "median": round(statistics.median(xs), 4),
            "std": round(std, 4), "ci_low": round(mean - half, 4), "ci_high": round(mean + half, 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4)}
