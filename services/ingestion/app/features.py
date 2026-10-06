"""Window and flight-level feature extraction (per LRU)."""
import warnings

import numpy as np

WINDOW_S = 60
CONTEXT = {"rpm": "ENGINE_RPM", "alt": "ALTITUDE", "ias": "INDICATED_AIRSPEED", "nz": "ACCEL_Z",
           "oat": "STATIC_TEMPERATURE", "roll": "ROLL"}
SUMMARY_PHASES = ("CLIMB", "CRUISE", "MANEUVER", "TAXI")


def _slope(v):
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return 0.0
    x = np.flatnonzero(ok)
    return float(np.polyfit(x, v[ok], 1)[0])


def _r(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 5)


def lru_features(clean, params, labels, kinds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _lru_features(clean, params, labels, kinds)


def _lru_features(clean, params, labels, kinds):
    """clean: {param: 1 Hz array}. Returns list of (window_id, start, end, phase, features); window -1 = flight summary."""
    n = len(labels)
    rows = []
    ctx = {k: clean[v] for k, v in CONTEXT.items() if v in clean}
    for w, s in enumerate(range(0, n - WINDOW_S // 2, WINDOW_S)):
        e = min(n, s + WINDOW_S)
        vals, cnt = np.unique(labels[s:e], return_counts=True)
        f = {f"ctx_{k}": _r(np.nanmean(v[s:e])) for k, v in ctx.items()}
        for p in params:
            v = clean[p][s:e]
            if kinds.get(p) == "counter":
                f[f"{p}__delta"] = _r(v[-1] - v[0])
                continue
            f.update({f"{p}__mean": _r(np.nanmean(v)), f"{p}__std": _r(np.nanstd(v)), f"{p}__min": _r(np.nanmin(v)),
                      f"{p}__max": _r(np.nanmax(v)), f"{p}__slope": _r(_slope(v))})
        rows.append((w, float(s), float(e), str(vals[np.argmax(cnt)]), f))
    f = {}
    for p in params:
        v = clean[p]
        if kinds.get(p) == "counter":
            f[f"{p}__delta"] = _r(v[-1] - v[0])
            continue
        f[f"{p}__mean"], f[f"{p}__std"], f[f"{p}__p95"] = _r(np.nanmean(v)), _r(np.nanstd(v)), _r(np.nanpercentile(v, 95))
        for ph in SUMMARY_PHASES:
            m = labels == ph
            if m.sum() >= 10:
                f[f"{p}__{ph.lower()}_mean"] = _r(np.nanmean(v[m]))
    for ph in SUMMARY_PHASES:
        m = labels == ph
        for k, v in ctx.items():
            if m.sum() >= 10:
                f[f"ctx_{k}__{ph.lower()}_mean"] = _r(np.nanmean(v[m]))
    f["ctx_duration_s"] = n
    rows.append((-1, 0.0, float(n), "FLIGHT", f))
    return rows
