"""Per-flight signal synthesis: baseline + context + cross-parameter dependency +
temporal dynamics + noise + per-unit sensor bias + faults."""
import numpy as np

from .failure_injection import physical_effect, sensor_effect
from .flight_profiles import DT, lag
from .subsystems import CATALOG


def unit_biases(rng):
    """Per-LRU-serial calibration offset for every parameter (sensor bias)."""
    return {p.name: rng.normal(0, 0.6 * p.noise) for p in CATALOG}


def generate_signals(rng, d, active, bias):
    """Return {param: measured array} on the 10 Hz grid. `d` gains the true values."""
    n = len(d.t)
    measured = {}
    for p in CATALOG:
        val = np.broadcast_to(np.asarray(p.f(d), float), (n,)).copy()
        val, extra_tau = physical_effect(rng, d, active, p.name, val)
        if p.kind == "counter":
            val = np.cumsum(rng.poisson(np.maximum(val, 0) * DT)).astype(float)
        elif p.tau or extra_tau:
            val = lag(val, p.tau + extra_tau)
        d[p.name] = val
        m = val + bias[p.name]
        if p.kind != "counter" and p.noise:
            m = m + rng.normal(0, p.noise, n) + 0.4 * p.noise * lag(rng.normal(0, 1, n), 2.0) * np.sqrt(2 * 2.0 / DT) / 3
        m = sensor_effect(rng, active, p.name, m)
        if p.kind == "discrete":
            m = np.round(m)
        # rare acquisition glitches: isolated outliers and short stuck runs (data-quality problems, not faults)
        k = rng.binomial(n, 2e-4)
        if k:
            idx = rng.integers(0, n, k)
            m[idx] += rng.choice([-1, 1], k) * rng.uniform(0.1, 0.3, k) * (p.hi - p.lo)
        if p.kind == "analog" and rng.random() < 0.02:
            s = rng.integers(0, n - 60)
            m[s:s + int(rng.uniform(10, 60))] = m[s]
        measured[p.name] = np.clip(m, p.lo, p.hi)
    return measured
