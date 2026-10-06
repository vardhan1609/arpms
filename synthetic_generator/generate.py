"""Deterministic synthetic dataset generator.

    python -m synthetic_generator.generate --seed 42 --scale small --out data

Writes product inputs to <out>/raw and evaluation-only labels to <out>/truth.
Product services must only ever read <out>/raw.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from . import aircraft as fleet
from .bus1553_generator import build_mapping, encode_flight, write_bin, write_hex
from .document_generator import build_documents, write_documents
from .failure_injection import MECHANISMS, all_mechanisms, degradation_state, severity
from .flight_profiles import build_context
from .fta_generator import build_fta
from .maintenance_generator import disposition, fim_ref
from .signal_models import generate_signals, unit_biases
from .snag_generator import ATA, NUISANCE, snag_text
from .subsystems import BY_NAME, CATALOG, LOG_SYNC_PARAMETER, LRU_TYPES

# scale: (aircraft, total flights, flight duration range s)
PRESETS = {"tiny": (3, 9, (420, 480)), "small": (20, 160, (420, 600)),
           "medium": (20, 1000, (420, 720)), "large": (50, 5000, (600, 1800))}
LOG_PARAMS = {}
for _p in CATALOG:
    if _p.src == "LOG":
        LOG_PARAMS.setdefault(_p.lru, []).append(_p.name)
AIRBORNE_EXCLUDE = ["PARKED", "TAXI", "POST_FLIGHT"]
LRU_SECTION = {code: i + 10 for i, code in enumerate(LRU_TYPES)}
NUISANCE_ACTION = dict(NUISANCE)


def to_us(dt):
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def iso(dt):
    return dt.replace(microsecond=0).isoformat()


def simulate_aircraft(job):
    rng = np.random.default_rng([job["seed"], job["idx"]])
    ac, raw, mapping, faults = job["ac"], job["raw"], job["mapping"], job["faults"]
    mechs = list(all_mechanisms(BY_NAME).values())
    lrus = {l["lru_id"]: dict(l) for l in job["lrus"]}
    bias = {lid: unit_biases(rng) for lid in lrus}
    clock = {lid: [rng.uniform(-45, 45), rng.uniform(-150, 150)] for lid in lrus}
    out = {k: [] for k in ("snags", "maint", "flights", "timeline", "faults", "installs", "clocks")}
    counter = {"snag": 0, "mr": 0}
    for k, f in enumerate(faults):
        f.update(fault_id=f"FAULT-{ac}-{k + 1:02d}", onset_flight=None, end_flight=None, end_reason="ONGOING",
                 serial=lrus[f["lru_id"]]["serial_number"])

    def next_id(kind, prefix):
        counter[kind] += 1
        return f"{prefix}-{ac[-3:]}-{counter[kind]:04d}"

    def add_record(when, flight_id, lru_id, sub, title, desc, symptom, obs, val, unit, phase, outcome, m,
                   reported_by="PILOT", replace=True):
        snag_id = next_id("snag", "SNG")
        if m:
            text, action, finding, test = disposition(rng, m, outcome)
        else:
            text = action = NUISANCE_ACTION[title]
            finding, test = "MINOR", "PASS"
        mdate = when + timedelta(hours=float(rng.uniform(1, 6)))
        out["snags"].append(dict(
            snag_id=snag_id, aircraft_id=ac, flight_id=flight_id, snag_date=iso(when), reported_by=reported_by,
            lru_id=lru_id, subsystem=sub, snag_code=f"{ATA[sub]}-{LRU_SECTION.get(lru_id[-3:], 0):02d}",
            snag_title=title, snag_description=desc, reported_symptom=symptom, observed_parameter=obs,
            observed_value=val, observed_unit=unit, flight_phase=phase,
            snag_category=str(rng.choice(["CAT-A", "CAT-B", "CAT-C"], p=[0.1, 0.5, 0.4])), disposition=text,
            action_taken=action, closure_status="CLOSED", closure_date=iso(mdate),
            maintenance_reference=fim_ref(m.code) if m else ""))
        lru = lrus.get(lru_id, {})
        removed = installed = ""
        if replace and outcome in ("FAILURE", "DEGRADED"):
            removed = lru["serial_number"]
            installed = f"SYN-{lru['lru_type']}-{rng.integers(100000, 999999)}"
        out["maint"].append(dict(
            maintenance_record_id=next_id("mr", "MR"), aircraft_id=ac, lru_id=lru_id, part_number=lru.get("part_number", ""),
            serial_number=lru.get("serial_number", ""), flight_id=flight_id, snag_id=snag_id, maintenance_date=iso(mdate),
            maintenance_type="CORRECTIVE", reported_problem=title, maintenance_description=text, action_taken=action,
            finding=finding, test_result=test, removed_serial_number=removed, installed_serial_number=installed,
            technician_id=f"TECH-{rng.integers(1, 40):03d}", maintenance_reference=fim_ref(m.code) if m else "",
            maintenance_status="CLOSED", labor_hours=round(float(rng.uniform(0.5, 6 if removed else 2.5)), 1)))
        return mdate, installed

    # historical (pre-window) snag & maintenance history, no telemetry
    for k in range(int(rng.integers(8, 16))):
        m = mechs[int(rng.integers(len(MECHANISMS)))] if rng.random() < 0.8 else mechs[int(rng.integers(len(mechs)))]
        p = BY_NAME[m.observed]
        when = fleet.WINDOW_START - timedelta(days=float(rng.uniform(10, 800)))
        phase = str(rng.choice(["CLIMB", "CRUISE", "MANEUVER", "APPROACH", "TAKEOFF"]))
        val = round(float(p.lo + (p.hi - p.lo) * rng.uniform(0.2, 0.8)), p.decimals)
        title, desc, symptom = snag_text(rng, m, phase, m.observed, val, p.unit)
        nff = rng.random() < (0.6 if m.intermittent else 0.3)
        add_record(when, f"{ac}-H{k + 1:03d}", f"{ac}-{m.lru}", LRU_TYPES[m.lru][1], title, desc, symptom, m.observed,
                   val, p.unit, phase, "NFF" if nff else "DEGRADED", m, replace=False)

    cum_h = 0.0
    for fl in job["flights"]:
        fid, dur, start = fl["flight_id"], fl["duration_s"], fl["start_time"]
        ambient = (rng.uniform(-5, 38), rng.uniform(995, 1030), rng.uniform(15, 90), rng.uniform(-25, 25))
        d, phase, _ = build_context(rng, dur, fl["mission_type"], ambient, aggressive=fl["aggressive"])
        hrs = cum_h + d.t / 3600
        active = []
        for f in faults:
            f["s"] = None
            if f["end_flight"]:
                continue
            x = (hrs - f["onset_h"]) / f["life_h"]
            s = np.where(x > 0, severity(f["traj"], x), 0.0)
            if s.max() > 0:
                f["s"], f["onset_flight"] = s, f["onset_flight"] or fid
                active.append((f["mechanism"], s))
        meas = generate_signals(rng, d, active, {p.name: bias[f"{ac}-{p.lru}"][p.name] for p in CATALOG})
        start_us = to_us(start)
        rec = encode_flight(rng, mapping, meas, start_us, dur, job["idx"] * 10**12 + fl["flight_number"] * 10**7)
        os.makedirs(f"{raw}/bus1553/{ac}", exist_ok=True)
        ext = "hex" if fl["flight_number"] % 10 == 0 else "bin"
        (write_hex if ext == "hex" else write_bin)(f"{raw}/bus1553/{ac}/{fid}.{ext}", ac, fid, start_us, rec)
        # LRU internal logs: own clock (offset + drift), 1 Hz, gaps / NaNs / duplicates
        os.makedirs(f"{raw}/lru_logs/{ac}", exist_ok=True)
        for lru, names in LOG_PARAMS.items():
            lid = f"{ac}-{lru}"
            clock[lid][0] += rng.normal(0, 1.5)
            off, ppm = clock[lid]
            idx = np.arange(0, len(d.t), 10)
            df = pd.DataFrame({"lru_time": np.round(start_us / 1e6 + off + d.t[idx] * (1 + ppm * 1e-6), 3),
                               "lru_serial": lrus[lid]["serial_number"],
                               LOG_SYNC_PARAMETER: np.round(meas[LOG_SYNC_PARAMETER][idx], 1)})
            for nme in names:
                v = np.round(meas[nme][idx], BY_NAME[nme].decimals)
                v[rng.random(len(v)) < 0.005] = np.nan
                df[nme] = v
            keep = np.ones(len(df), bool)
            for _ in range(int(rng.integers(0, 3))):
                g = int(rng.integers(0, len(df)))
                keep[g:g + int(rng.integers(5, 20))] = False
            df = df[keep]
            df = pd.concat([df, df[rng.random(len(df)) < 0.003]]).sort_values("lru_time", kind="stable")
            df.to_csv(f"{raw}/lru_logs/{ac}/{fid}_{lru}.csv", index=False)
            out["clocks"].append(dict(aircraft_id=ac, flight_id=fid, lru_id=lid, clock_offset_s=off, drift_ppm=ppm))
        end = start + timedelta(seconds=dur)
        out["flights"].append(dict(flight_id=fid, aircraft_id=ac, flight_number=fl["flight_number"], departure_time=iso(start),
                                   arrival_time=iso(end), mission_type=fl["mission_type"],
                                   base=f"SYN-BASE-{1 + job['idx'] % 3}", flight_hours=round(dur / 3600, 3)))
        cum_h += dur / 3600
        for f in faults:
            if f["s"] is None:
                continue
            s, m = f["s"], f["mechanism"]
            out["timeline"].append(dict(
                fault_id=f["fault_id"], aircraft_id=ac, flight_id=fid, lru_id=f["lru_id"], serial_number=f["serial"],
                mechanism_code=m.code, mechanism_name=m.name, sensor_fault=m.sensor, severity_start=float(s[0]),
                severity_end=float(s[-1]), severity_max=float(s.max()), state=degradation_state(float(s[-1])),
                rul_hours=f["onset_h"] + f["life_h"] - cum_h, affected_parameters=";".join(dict.fromkeys(e[0] for e in m.effects))))
        # snags & maintenance driven by fault progression
        for f in faults:
            if f["s"] is None:
                continue
            s_end, m = float(f["s"][-1]), f["mechanism"]
            failed = s_end >= 1.0
            p_snag = 1.0 if failed else 0.5 if f["repeat"] and s_end > 0.2 else float(np.clip((s_end - 0.3) * 1.2, 0, 0.6))
            if rng.random() >= p_snag:
                continue
            outcome = "FAILURE" if failed else "NFF" if f["no_repair"] or s_end < 0.45 or rng.random() < 0.35 else "DEGRADED"
            i = int(rng.choice(np.where(~np.isin(phase, AIRBORNE_EXCLUDE))[0]))
            p = BY_NAME[m.observed]
            val = round(float(meas[m.observed][i]), p.decimals)
            title, desc, symptom = snag_text(rng, m, str(phase[i]), m.observed, val, p.unit)
            mdate, installed = add_record(end, fid, f["lru_id"], LRU_TYPES[m.lru][1], title, desc, symptom, m.observed, val,
                                          p.unit, str(phase[i]), outcome, m)
            if installed:
                lid = f["lru_id"]
                out["installs"].append(dict(lrus[lid], serial_number=installed, install_date=iso(mdate), hours_since_install=0.0))
                lrus[lid]["serial_number"] = installed
                bias[lid] = unit_biases(rng)
                clock[lid] = [rng.uniform(-45, 45), rng.uniform(-150, 150)]
                f.update(end_flight=fid, end_reason="REPLACED_" + outcome)
        if rng.random() < 0.04:
            title = NUISANCE[int(rng.integers(len(NUISANCE)))][0]
            add_record(end, fid, "", "GENERAL", title, title + ".", title.lower(), "", "", "", "POST_FLIGHT", "NUISANCE", None,
                       reported_by="GROUND_CREW")
        if rng.random() < 0.08:
            lid = list(lrus)[int(rng.integers(len(lrus)))]
            out["maint"].append(dict(
                maintenance_record_id=next_id("mr", "MR"), aircraft_id=ac, lru_id=lid, part_number=lrus[lid]["part_number"],
                serial_number=lrus[lid]["serial_number"], flight_id=fid, snag_id="", maintenance_date=iso(end + timedelta(hours=2)),
                maintenance_type="INSPECTION", reported_problem="Scheduled inspection",
                maintenance_description=f"Scheduled inspection IAW IP-{lid[-3:]}. No defects.", action_taken="Inspection",
                finding="NIL", test_result="PASS", removed_serial_number="", installed_serial_number="",
                technician_id=f"TECH-{rng.integers(1, 40):03d}", maintenance_reference=f"IP-{lid[-3:]}",
                maintenance_status="CLOSED", labor_hours=1.0))
    for f in faults:
        m = f["mechanism"]
        out["faults"].append(dict(
            fault_id=f["fault_id"], aircraft_id=ac, lru_id=f["lru_id"], serial_number=f["serial"], mechanism_code=m.code,
            mechanism_name=m.name, subsystem=LRU_TYPES[m.lru][1], sensor_fault=m.sensor, trajectory=f["traj"],
            onset_hours=f["onset_h"], failure_hours=f["onset_h"] + f["life_h"], onset_flight_id=f["onset_flight"],
            end_flight_id=f["end_flight"], end_reason=f["end_reason"], observed_parameter=m.observed, scenario=job["scenario"]))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(seed=42, scale="small", aircraft=None, flights=None, out="data", workers=None, validate=True):
    n_ac, n_fl, dur = PRESETS[scale]
    n_ac = aircraft or n_ac
    per_ac = math.ceil((flights or n_fl) / n_ac)
    raw, truth = os.path.join(out, "raw"), os.path.join(out, "truth")
    for d in (raw, truth):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    rng = np.random.default_rng(seed)
    aircraft_rows, lru_rows, flight_plan = fleet.make_fleet(rng, n_ac, per_ac, dur)
    ids = [a["aircraft_id"] for a in aircraft_rows]
    plans = fleet.plan_faults(rng, ids, flight_plan)
    mapping = build_mapping()
    jobs = [dict(seed=seed, idx=i, ac=ac, raw=raw, mapping=mapping, faults=plans[ac],
                 flights=[f for f in flight_plan if f["aircraft_id"] == ac],
                 lrus=[l for l in lru_rows if l["aircraft_id"] == ac],
                 scenario=fleet.SCENARIOS[i][0] if n_ac >= 8 and i in fleet.SCENARIOS else "RANDOM_FLEET")
            for i, ac in enumerate(ids, start=1)]
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        results = list(ex.map(simulate_aircraft, jobs))
    cat = {k: [r for res in results for r in res[k]] for k in results[0]}

    def csv(rows, rel):
        os.makedirs(os.path.dirname(os.path.join(raw, rel)), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(raw, rel), index=False)

    csv(aircraft_rows, "fleet/aircraft.csv")
    csv(lru_rows + cat["installs"], "fleet/lru_installations.csv")
    csv(cat["flights"], "flights/flight_log.csv")
    csv(cat["snags"], "snags/snag_reports.csv")
    csv(cat["maint"], "maintenance/maintenance_records.csv")
    csv(mapping, "icd/bus1553_mapping.csv")
    csv([dict(parameter_id=p.name, lru_type=p.lru, unit=p.unit, rate_class=p.rate, min_value=p.lo, max_value=p.hi,
              source=p.src, kind=p.kind, decimals=p.decimals) for p in CATALOG], "icd/parameter_dictionary.csv")
    mechs = all_mechanisms(BY_NAME)
    fta = build_fta(mechs)
    for sub in sorted({r["subsystem"] for r in fta}):
        csv([r for r in fta if r["subsystem"] == sub], f"fta/fta_{sub}.csv")
    snags = pd.DataFrame(cat["snags"])
    summary = {sub: f"{len(g)} snags recorded. Most frequent: " + "; ".join(f"{t} ({c})" for t, c in g.snag_title.value_counts().head(5).items())
               for sub, g in snags.groupby("subsystem")}
    write_documents(build_documents(mechs, mapping, fta, summary), os.path.join(raw, "documents"))

    pd.DataFrame(cat["faults"]).to_parquet(f"{truth}/fault_truth.parquet", index=False)
    pd.DataFrame(cat["timeline"]).to_parquet(f"{truth}/degradation_timeline.parquet", index=False)
    pd.DataFrame(cat["clocks"]).to_parquet(f"{truth}/lru_clock_truth.parquet", index=False)
    with open(f"{truth}/scenarios.json", "w") as fh:
        json.dump({j["ac"]: j["scenario"] for j in jobs}, fh, indent=2)

    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(raw) for f in fs)
    log_obs = sum(int(pd.read_csv(f).iloc[:, 3:].notna().sum().sum()) for f in files if "/lru_logs/" in f)
    n_msgs = sum((os.path.getsize(f) - 64) // 100 for f in files if f.endswith(".bin")) + \
        sum(sum(1 for _ in open(f)) - 1 for f in files if f.endswith(".hex"))
    params_per_msg = pd.DataFrame(mapping).groupby("message_id").size().mean()
    manifest = dict(generator="arpms-synthetic", version="1.0", seed=seed, scale=scale, aircraft=n_ac,
                    lrus=len(lru_rows), flights=len(cat["flights"]), parameters=len(CATALOG), bus_messages=int(n_msgs),
                    approx_bus_observations=int(n_msgs * params_per_msg), lru_log_observations=log_obs,
                    snags=len(cat["snags"]), maintenance_records=len(cat["maint"]), faults=len(cat["faults"]),
                    synthetic_notice="All data is synthetic. No real aircraft, ICD or maintenance data.")
    with open(os.path.join(raw, "MANIFEST.json"), "w") as fh:
        json.dump(dict(manifest, files=[dict(path=os.path.relpath(f, raw), bytes=os.path.getsize(f), sha256=sha256(f))
                                        for f in files]), fh, indent=1)
    with open(f"{truth}/generation_config.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    if validate:
        from .validation import validate as run_validation
        manifest["validation"] = run_validation(out)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", choices=PRESETS, default="small")
    ap.add_argument("--aircraft", type=int)
    ap.add_argument("--flights", type=int, help="total flights across the fleet")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    m = generate(a.seed, a.scale, a.aircraft, a.flights, a.out, a.workers, not a.no_validate)
    print(json.dumps({k: v for k, v in m.items() if k != "validation"}, indent=2))
    if "validation" in m:
        print("validation:", m["validation"]["status"])


if __name__ == "__main__":
    main()
