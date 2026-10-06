"""Synthetic fleet, LRU installations, flight schedule and fault plan."""
from datetime import datetime, timedelta


from .failure_injection import MECHANISMS, SENSOR_KINDS, SENSOR_TARGETS, TRAJECTORIES, sensor_mechanism
from .flight_profiles import MISSIONS
from .subsystems import BY_NAME, LRU_TYPES

WINDOW_START = datetime(2026, 3, 2, 6, 0)
MECH = {m.name: m for m in MECHANISMS}

# Demonstration scenarios on the first aircraft (only when the fleet is large enough).
SCENARIOS = {
    1: ("NORMAL_FLIGHT", None),
    2: ("SENSOR_DRIFT", dict(kind="SENSOR_DRIFT", target="BAROMETRIC_ALTITUDE", traj="linear", onset=0.25, s_end=0.75)),
    3: ("SLOW_DEGRADATION", dict(mech="BEARING_DEGRADATION", traj="linear", onset=0.05, s_end=0.95)),
    4: ("HYDRAULIC_DEGRADATION", dict(mech="HYD_PUMP_A_DEGRADATION", traj="exponential", onset=0.2, s_end=0.9)),
    5: ("ELECTRICAL_DEGRADATION", dict(mech="GENERATOR_DEGRADATION", traj="step", onset=0.3, s_end=0.9)),
    6: ("REPEATED_SNAG", dict(mech="COMMUNICATION_DEGRADATION", traj="intermittent", onset=0.05, s_end=0.8, repeat=True)),
    7: ("MULTI_SIGNAL_FAULT", dict(mech="COOLING_DEGRADATION", traj="accelerating", onset=0.15, s_end=0.95)),
    8: ("FALSE_POSITIVE_CONTEXT", None),
}


def aircraft_id(i):
    return f"SYN-AC-{i:03d}"


def make_fleet(rng, n_aircraft, flights_per_aircraft, dur_range):
    aircraft, lrus, flights = [], [], []
    for i in range(1, n_aircraft + 1):
        ac = aircraft_id(i)
        mfg = datetime(2014, 1, 1) + timedelta(days=int(rng.integers(0, 2500)))
        hours0 = float(rng.uniform(300, 2200))
        aircraft.append(dict(
            aircraft_id=ac, aircraft_type="SYN-TRAINER-X", aircraft_variant=str(rng.choice(["X1", "X2"])),
            aircraft_serial_number=f"SYN-SN-{10000 + i}", configuration_id=f"CFG-{rng.integers(1, 4)}",
            software_version=f"OFP-{rng.integers(3, 6)}.{rng.integers(0, 9)}", hardware_configuration=f"HW-{rng.choice(['A', 'B'])}",
            avionics_version=f"AV-{rng.integers(1, 3)}.{rng.integers(0, 5)}", engine_configuration=f"ENG-CFG-{rng.integers(1, 3)}",
            manufacture_date=mfg.date().isoformat(), service_entry_date=(mfg + timedelta(days=200)).date().isoformat(),
            total_flight_hours=round(hours0, 1), total_flight_cycles=int(hours0 * 1.1)))
        for code, (name, subsystem, rt) in LRU_TYPES.items():
            lrus.append(dict(lru_id=f"{ac}-{code}", aircraft_id=ac, lru_type=code, lru_name=name, subsystem=subsystem,
                             remote_terminal_address=rt, part_number=f"SYN-PN-{code}-{rng.integers(100, 999)}",
                             serial_number=f"SYN-{code}-{rng.integers(100000, 999999)}",
                             install_date=(WINDOW_START - timedelta(days=int(rng.integers(30, 900)))).date().isoformat(),
                             hours_since_install=round(float(rng.uniform(20, 900)), 1)))
        t = WINDOW_START + timedelta(hours=float(rng.uniform(0, 24)))
        aggressive = i == 8 and n_aircraft >= 8
        for k in range(flights_per_aircraft):
            mission = "HIGH_MANEUVER_TRAINING" if aggressive else str(rng.choice(list(MISSIONS), p=[0.25, 0.15, 0.4, 0.2]))
            dur = float(rng.uniform(*dur_range))
            flights.append(dict(flight_id=f"{ac}-F{k + 1:04d}", aircraft_id=ac, flight_number=k + 1, start_time=t,
                                duration_s=dur, mission_type=mission, aggressive=aggressive))
            t += timedelta(hours=float(rng.uniform(8, 40)))
    return aircraft, lrus, flights


def _plan(mech, ac, traj, onset_h, life_h, repeat=False, no_repair=False):
    return dict(mechanism=mech, lru_id=f"{ac}-{mech.lru}", traj=traj, onset_h=onset_h, life_h=life_h,
                repeat=repeat, no_repair=no_repair, ended=False)


def plan_faults(rng, aircraft_ids, flights, demo=True):
    """Return {aircraft_id: [fault dict]} with onset/life in flight-hours within the data window."""
    plans = {}
    for idx, ac in enumerate(aircraft_ids, start=1):
        H = sum(f["duration_s"] for f in flights if f["aircraft_id"] == ac) / 3600
        faults = []
        if demo and len(aircraft_ids) >= 8 and idx in SCENARIOS:
            spec = SCENARIOS[idx][1]
            if spec:
                mech = sensor_mechanism(spec["kind"], spec["target"], BY_NAME) if "kind" in spec else MECH[spec["mech"]]
                onset = spec["onset"] * H
                faults.append(_plan(mech, ac, spec["traj"], onset, (H - onset) / spec["s_end"],
                                    repeat=spec.get("repeat", False), no_repair=True))
        else:
            used = set()
            for _ in range(int(rng.integers(2, 5))):
                if rng.random() < 0.3:
                    mech = sensor_mechanism(str(rng.choice(list(SENSOR_KINDS))), str(rng.choice(SENSOR_TARGETS)), BY_NAME)
                else:
                    mech = MECHANISMS[int(rng.integers(len(MECHANISMS)))]
                if mech.lru in used:
                    continue
                used.add(mech.lru)
                traj = "intermittent" if mech.intermittent else str(rng.choice([t for t in TRAJECTORIES if t != "intermittent"]))
                faults.append(_plan(mech, ac, traj, float(rng.uniform(-0.3, 0.6)) * H, float(rng.uniform(0.35, 1.0)) * H))
        plans[ac] = faults
    return plans
