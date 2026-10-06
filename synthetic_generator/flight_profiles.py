"""Synthetic flight profiles -> operating-context series (10 Hz internal grid)."""
import numpy as np
from scipy.signal import lfilter

DT = 0.1
PHASES = ["PARKED", "TAXI", "TAKEOFF", "CLIMB", "CRUISE", "MANEUVER", "DESCENT", "APPROACH", "LANDING", "POST_FLIGHT"]
MISSIONS = {  # name: (maneuver segments, max load factor g, cruise altitude range ft)
    "TRANSIT": (0, 1.5, (25000, 35000)),
    "FUNCTIONAL_CHECK": (0, 2.0, (12000, 20000)),
    "TRAINING": (1, 4.0, (18000, 30000)),
    "HIGH_MANEUVER_TRAINING": (2, 6.5, (15000, 25000)),
}
# per-phase targets: speed kt, throttle 0..1, flap deg
TARGETS = {
    "PARKED": (0, 0.0, 0), "TAXI": (15, 0.18, 10), "TAKEOFF": (160, 1.0, 20), "CLIMB": (360, 0.88, 0),
    "CRUISE": (450, 0.62, 0), "MANEUVER": (480, 0.95, 0), "DESCENT": (320, 0.35, 0),
    "APPROACH": (170, 0.45, 20), "LANDING": (40, 0.12, 30), "POST_FLIGHT": (8, 0.1, 0),
}


class D(dict):
    __getattr__ = dict.__getitem__


def lag(x, tau, dt=DT, x0=None):
    if tau <= 0:
        return np.asarray(x, float)
    a = dt / (tau + dt)
    x = np.asarray(x, float)
    zi = [(1 - a) * (x[0] if x0 is None else x0)]
    return lfilter([a], [1, -(1 - a)], x, zi=zi)[0]


def smooth_noise(rng, n, tau, sd):
    return lag(rng.normal(0, sd, n), tau) * np.sqrt(2 * tau / DT)


def phase_schedule(rng, duration, mission):
    segs, _, _ = MISSIONS[mission]
    fixed = {"PARKED": 40, "TAXI": 45, "TAKEOFF": 20, "LANDING": 20, "POST_FLIGHT": 40}
    rest = duration - sum(fixed.values())
    climb, descent, approach = 0.17 * rest, 0.14 * rest, 0.10 * rest
    man = [rng.uniform(0.12, 0.16) * rest for _ in range(segs)]
    cruise = rest - climb - descent - approach - sum(man)
    order = [("PARKED", 40), ("TAXI", 45), ("TAKEOFF", 20), ("CLIMB", climb)]
    parts = len(man) + 1
    for i in range(parts):
        order.append(("CRUISE", cruise / parts))
        if i < len(man):
            order.append(("MANEUVER", man[i]))
    order += [("DESCENT", descent), ("APPROACH", approach), ("LANDING", 20), ("POST_FLIGHT", 40)]
    return order


def build_context(rng, duration, mission, ambient, aggressive=False):
    """Return (D of context arrays, phase label array, list of (phase, start, end))."""
    n = int(duration / DT)
    t = np.arange(n) * DT
    order = phase_schedule(rng, duration, mission)
    phase = np.empty(n, dtype=object)
    bounds, s = [], 0.0
    for name, dur in order:
        i0, i1 = int(s / DT), min(n, int((s + dur) / DT))
        phase[i0:i1] = name
        bounds.append((name, s, s + dur))
        s += dur
    phase[phase == None] = "POST_FLIGHT"  # noqa: E711
    segs, gmax, alt_rng = MISSIONS[mission]
    if aggressive:
        gmax = 7.5
    cruise_alt = rng.uniform(*alt_rng)

    # altitude: piecewise linear between phase-boundary altitudes, + maneuver oscillation
    knots_t, knots_a = [0.0], [0.0]
    for name, a, b in bounds:
        end_alt = {"PARKED": 0, "TAXI": 0, "TAKEOFF": 300, "CLIMB": cruise_alt, "CRUISE": cruise_alt,
                   "MANEUVER": cruise_alt, "DESCENT": 3000, "APPROACH": 0, "LANDING": 0, "POST_FLIGHT": 0}[name]
        if name == "TAKEOFF":
            knots_t += [a + 0.7 * (b - a)]
            knots_a += [0.0]
        knots_t.append(b)
        knots_a.append(end_alt)
    alt = np.interp(t, knots_t, knots_a)
    man = (phase == "MANEUVER").astype(float)
    man_s = lag(man, 3)
    alt += man_s * 2500 * np.sin(2 * np.pi * t / rng.uniform(40, 70))
    alt += (phase == "CRUISE") * smooth_noise(rng, n, 20, 40)
    alt = np.maximum(alt, 0)

    spd_t = np.array([TARGETS[p][0] for p in phase], float)
    thr_t = np.array([TARGETS[p][1] for p in phase], float)
    flap_t = np.array([TARGETS[p][2] for p in phase], float)
    thr_t = np.clip(thr_t + (phase != "PARKED") * smooth_noise(rng, n, 15, 0.03) + 0.05 * man_s * aggressive, 0, 1)
    spd = np.maximum(lag(spd_t, 9) + (spd_t > 50) * smooth_noise(rng, n, 10, 4), 0)
    thr = lag(thr_t, 1.5)
    eng_on = ((t > 20) & (t < duration - 10)).astype(float)
    rpm = eng_on * lag(np.where(eng_on > 0, 0.25 + 0.75 * thr, 0.0), 1.2)

    load = np.ones(n)
    if segs or aggressive:
        pulses = np.zeros(n)
        idx = np.where(man > 0)[0]
        for c in rng.choice(idx, size=max(1, len(idx) // 120), replace=False) if len(idx) else []:
            w = int(rng.uniform(30, 80))
            pulses[c:c + w] += rng.uniform(0.4, 1.0) * (gmax - 1) * np.hanning(len(pulses[c:c + w]))
        load += pulses
    load += smooth_noise(rng, n, 2, 0.03) * (1 - (phase == "PARKED"))

    amb_g, amb_g_p, hum, wind = ambient
    amb_t = amb_g - 1.98 * alt / 1000
    amb_p = amb_g_p * (1 - 6.8756e-6 * alt) ** 5.2559
    mach = (spd / (661.47 * np.sqrt((amb_t + 273.15) / 288.15)))
    gnd = np.isin(phase, ["PARKED", "TAXI", "LANDING", "POST_FLIGHT"]) | ((phase == "TAKEOFF") & (alt < 1))
    gnd = gnd.astype(float)

    roll = (1 - gnd) * (smooth_noise(rng, n, 25, 8) + man_s * smooth_noise(rng, n, 4, 45))
    roll = np.clip(roll, -85, 85)
    pitch_base = np.select([phase == "CLIMB", phase == "DESCENT", phase == "APPROACH", phase == "TAKEOFF"], [11, -4, -3, 6], 2)
    pitch = lag(pitch_base, 3) + 2.5 * (load - 1) + (1 - gnd) * smooth_noise(rng, n, 5, 0.8)
    hdg_rate = 1091 * np.tan(np.radians(roll)) / np.maximum(spd, 60) * (1 - gnd)
    hdg = (rng.uniform(0, 360) + np.cumsum(hdg_rate) * DT) % 360
    gs_ms = (spd + wind) * 0.5144
    lat0, lon0 = 30.0, 10.0  # synthetic home location, not a real site
    north = np.cumsum(gs_ms * np.cos(np.radians(hdg))) * DT
    east = np.cumsum(gs_ms * np.sin(np.radians(hdg))) * DT
    lat = lat0 + north / 111_000
    lon = lon0 + east / (111_000 * np.cos(np.radians(lat0)))
    stick = np.clip(0.12 * (load - 1) + 0.04 * (pitch - pitch_base) + smooth_noise(rng, n, 1.5, 0.03) * (1 - gnd), -1, 1)
    roll_rate = np.gradient(roll, DT)
    ped = np.clip(smooth_noise(rng, n, 3, 0.05) + (phase == "APPROACH") * wind / 60, -1, 1)
    flap = lag(flap_t, 2)
    spoiler = ((phase == "LANDING") * 40.0)
    elev = 0.22 * 100 * stick + 2 * (load - 1)
    surface_rate = np.gradient(lag(elev, 0.15), DT)
    hyd_dem = np.clip(lag(0.15 + np.abs(surface_rate) / 60 + 0.25 * np.abs(np.gradient(flap, DT)) + 0.25 * man_s, 0.5), 0, 1.5)
    elec = np.clip(0.35 + 0.12 * man_s + 0.08 * gnd * eng_on + smooth_noise(rng, n, 30, 0.03), 0.1, 1)
    cpu = np.clip(0.28 + 0.3 * man_s + 0.1 * np.isin(phase, ["TAKEOFF", "APPROACH"]) + smooth_noise(rng, n, 5, 0.04), 0.05, 1)
    d = D(
        t=t, dt=DT, alt=alt, spd=spd, thr=thr, rpm=rpm, load=load, load_dev=load - 1, man=man_s,
        amb_t=amb_t, amb_p=amb_p, amb_g=np.full(n, amb_g), amb_g_p=np.full(n, amb_g_p), hum=np.full(n, hum),
        wind_x=wind + smooth_noise(rng, n, 5, 2), gnd=gnd, stick=stick, roll_cmd=np.clip(roll_rate / 90, -1, 1),
        ped=ped, elec=elec, hyd_dem=hyd_dem, hyd_dem_s=lag(hyd_dem, 60), eng_on=eng_on, eng_warm=lag(eng_on, 150),
        mach=mach, aoa=np.clip(2.5 + 1.6 * (load - 1) * 400 / np.maximum(spd, 120), -5, 30) * (1 - gnd), roll=roll,
        pitch=pitch, yaw=((hdg + 180) % 360) - 180, hdg=hdg, lat=lat, lon=lon, vs=np.gradient(alt, DT) * 60,
        roll_rate=roll_rate, pitch_rate=np.gradient(pitch, DT), yaw_rate=hdg_rate,
        accel_x=np.gradient(spd, DT) * 0.5144 / 9.81, flap=flap, spoiler=spoiler, surface_rate=surface_rate,
        cpu_dem=cpu, cpu_dem_s=lag(cpu, 30), link_dist=np.hypot(north, east) / 1000,
        fuel0=float(rng.uniform(80, 98)), soc0=float(rng.uniform(85, 100)),
    )
    return d, phase, bounds


def operating_mode(phase):
    return {"PARKED": "GROUND", "TAXI": "GROUND", "POST_FLIGHT": "GROUND", "CRUISE": "AUTOPILOT"}.get(phase, "MANUAL")
