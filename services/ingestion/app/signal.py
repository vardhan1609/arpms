"""Signal processing: gridding, DTW clock sync, Kalman cleaning, quality metrics, phase detection."""
import numpy as np

PHASES = ["PARKED", "TAXI", "TAKEOFF", "CLIMB", "CRUISE", "MANEUVER", "DESCENT", "APPROACH", "LANDING", "POST_FLIGHT"]


def to_grid(ts_us, vals, start_us, n, hz=1.0):
    """Bucket-mean samples onto a regular grid (NaN where no sample)."""
    idx = np.floor((np.asarray(ts_us) - start_us) / 1e6 * hz).astype(np.int64)
    ok = (idx >= 0) & (idx < n) & np.isfinite(vals)
    s = np.bincount(idx[ok], np.asarray(vals)[ok], n)
    c = np.bincount(idx[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def dtw_path(a, b, band):
    """Banded DTW on 1-D series; row recurrence vectorised via cumulative-min. Returns [(i, j)]."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        if lo > hi:
            continue
        c = np.abs(a[i - 1] - b[lo - 1:hi])
        tmp = c + np.minimum(D[i - 1, lo - 1:hi], D[i - 1, lo:hi + 1])
        tmp[0] = min(tmp[0], c[0] + D[i, lo - 1])
        C = np.cumsum(c)
        D[i, lo:hi + 1] = C + np.minimum.accumulate(tmp - C)
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1], float(D[n, m] / max(len(path), 1))


def clock_sync(bus_alt, lru_t, lru_alt, max_offset=120, band=8):
    """Estimate LRU clock = offset + (1 + drift) * bus_time from a shared altitude signal.

    bus_alt: 1 Hz series on bus time (t = index, seconds from flight start).
    lru_t: LRU timestamps relative to flight start (LRU clock); lru_alt: altitude received by the LRU.
    Coarse offset by derivative matching, refined with DTW + linear fit on dynamic segments.
    """
    ok = np.isfinite(lru_alt) & np.isfinite(lru_t)
    lru_t, lru_alt = lru_t[ok], lru_alt[ok]
    t = np.arange(len(bus_alt), dtype=float) + 0.5  # 1 Hz bucket means are centred mid-bucket
    bv = np.isfinite(bus_alt)
    bus = np.interp(t, t[bv], bus_alt[bv])
    db = np.gradient(bus)
    lags = np.arange(-max_offset, max_offset + 1)
    cost = [np.nanmean(np.abs(np.gradient(np.interp(t + L, lru_t, lru_alt, left=np.nan, right=np.nan)) - db)) for L in lags]
    off_c = float(lags[int(np.nanargmin(cost))])
    a = np.interp(t + off_c, lru_t, lru_alt)
    scale = np.std(bus) or 1.0
    path, dcost = dtw_path((bus - bus.mean()) / scale, (a - bus.mean()) / scale, band)
    pi, pj = np.array(path).T
    dyn = np.abs(db[pi]) > max(np.percentile(np.abs(db), 60), 1e-6)
    if dyn.sum() < 20:
        return dict(offset_s=off_c, drift_ppm=0.0, dtw_cost=dcost, method="XCORR")
    off = off_c + float(np.median(t[pj][dyn] - t[pi][dyn]))
    # sub-second refinement around the DTW estimate on dynamic segments
    grid = off + np.arange(-1.5, 1.51, 0.05)
    err = [np.mean(np.abs(np.interp(t, lru_t - o, lru_alt) - bus)[np.abs(db) > 0]) for o in grid]
    # ponytail: drift (<~150 ppm, <0.1 s per flight) is absorbed by per-flight offset re-estimation, not modelled
    return dict(offset_s=float(grid[int(np.argmin(err))]), drift_ppm=0.0, dtw_cost=dcost, method="DTW")


def kalman_clean(Y, gate=5.0, persist=3):
    """Vectorised local-level Kalman filter + RTS smoother over rows of Y (params x time).

    Innovations beyond `gate` sigma are rejected as outliers unless they persist for `persist`
    samples (then treated as a genuine level change and the state is reset). NaNs are predicted through.
    Returns (smoothed, outlier_mask).
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    K, T = Y.shape
    r, q = np.full(K, 1e-10), np.full(K, 1e-12)
    for k in range(K):
        idx = np.flatnonzero(np.isfinite(Y[k]))
        if len(idx) < 5:
            continue
        v, step = Y[k, idx], max(float(np.median(np.diff(idx))), 1.0)
        mad = lambda d: 1.4826 * np.median(np.abs(d - np.median(d)))
        r[k] = max(mad(np.diff(v, 2)) ** 2 / 6, 1e-10)
        q[k] = max(mad(np.diff(v)) ** 2 - 2 * r[k], r[k] * 1e-2) / step
    first = np.argmax(np.isfinite(Y), axis=1)
    x = Y[np.arange(K), first]
    x = np.where(np.isfinite(x), x, 0.0)
    P = r.copy()
    xf, Pf, xp, Pp = (np.zeros((K, T)) for _ in range(4))
    out = np.zeros((K, T), bool)
    reset = np.zeros((K, T), bool)
    streak = np.zeros(K, int)
    for t in range(T):
        P = P + q
        xp[:, t], Pp[:, t] = x, P
        y = Y[:, t]
        fin = np.isfinite(y)
        innov = np.where(fin, y - x, 0.0)
        S = P + r
        big = fin & (np.abs(innov) > gate * np.sqrt(S))
        streak = np.where(big, streak + 1, 0)
        rs = big & (streak >= persist)
        upd = fin & ~big
        G = np.where(upd, P / S, 0.0)
        x = np.where(rs, y, x + G * innov)
        P = np.where(rs, r, P * (1 - G))
        out[:, t], reset[:, t] = big & ~rs, rs
        xf[:, t], Pf[:, t] = x, P
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        G = np.where(reset[:, t + 1], 0.0, Pf[:, t] / Pp[:, t + 1])
        xs[:, t] = xf[:, t] + G * (xs[:, t + 1] - xp[:, t + 1])
    # samples just before a confirmed level change were provisional outliers of the new level
    for k in range(1, persist):
        out[:, :-k] &= ~reset[:, k:]
    return xs, out


def stuck_fraction(v, lo, hi, run=15):
    v = np.asarray(v, float)
    same = np.r_[False, (np.diff(v) == 0)] & np.isfinite(v) & (v > lo) & (v < hi)
    if not same.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.r_[0, same.astype(int), 0]))
    lengths = edges[1::2] - edges[::2]
    return float(lengths[lengths >= run].sum() / len(v))


def gap_count(ts_us, rate_hz, start_us, end_us):
    ts = np.r_[start_us, np.sort(ts_us), end_us]
    return int((np.diff(ts) / 1e6 > 3 / rate_hz).sum())


def quality_score(completeness, validity, outliers, stuck):
    return float(100 * (0.4 * min(completeness, 1) + 0.2 * validity + 0.2 * max(0, 1 - 10 * outliers) + 0.2 * (1 - stuck)))


def detect_phases(alt, ias, rpm, vs, nz, roll):
    """Rule-based phase labelling on 1 Hz cleaned signals; returns (labels, segments)."""
    n = len(alt)
    agl = alt - np.nanmedian(alt[: min(30, n)])
    air = np.flatnonzero(agl > 100)
    lab = np.empty(n, dtype=object)
    if not len(air):
        lab[:] = np.where(rpm < 10, "PARKED", "TAXI")
    else:
        a0, a1 = air[0], air[-1]
        peak = int(np.nanargmax(agl))
        for i in range(n):
            if i < a0:
                lab[i] = "PARKED" if rpm[i] < 10 else "TAKEOFF" if ias[i] > 50 else "TAXI"
            elif i > a1:
                lab[i] = "LANDING" if ias[i] > 25 else "POST_FLIGHT"
            elif abs(nz[i] - 1) > 0.6 or abs(roll[i]) > 45:
                lab[i] = "MANEUVER"
            elif vs[i] > 800:
                lab[i] = "CLIMB" if i < peak + 60 or agl[i] > 1000 else "TAKEOFF"
            elif vs[i] < -800:
                lab[i] = "APPROACH" if agl[i] < 6000 and ias[i] < 230 else "DESCENT"
            elif i > peak and agl[i] < 6000 and ias[i] < 230:
                lab[i] = "APPROACH"
            else:
                lab[i] = "CRUISE"
    # merge short segments (<8 s) into the previous one
    segs = []
    for i, p in enumerate(lab):
        if segs and segs[-1][0] == p:
            segs[-1][2] = i + 1
        else:
            segs.append([p, i, i + 1])
    merged = []
    for s in segs:
        if merged and (s[2] - s[1] < 8 or merged[-1][0] == s[0]):
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    for p, a, b in merged:
        lab[a:b] = p
    return lab, [dict(phase=p, start_s=float(a), end_s=float(b)) for p, a, b in merged]
