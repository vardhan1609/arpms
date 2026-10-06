"""Phase-2 validation of a generated dataset: decodability, rates, correlations, temporal
dynamics, data-quality imperfections, fault progression, maintenance/snag/FTA consistency.

    python -m synthetic_generator.validation --out data
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import welch  # noqa: E402

from .bus1553_generator import HEADER_SIZE, RECORD  # noqa: E402

PAIRS = [("ENGINE_RPM", "ENGINE_FUEL_FLOW"), ("ENGINE_RPM", "ENGINE_EXHAUST_TEMPERATURE"),
         ("ENGINE_RPM", "HYD_PUMP_RPM"), ("ALTITUDE", "STATIC_PRESSURE")]
SIGNS = {("ALTITUDE", "STATIC_PRESSURE"): -1}


def read_bus(path):
    if path.endswith(".hex"):
        lines = open(path).read().split("\n")[1:]
        return np.frombuffer(bytes.fromhex("".join(lines)), RECORD)
    return np.fromfile(path, RECORD, offset=HEADER_SIZE)


def decode(rec, mapping):
    """{parameter: Series(value, index=timestamp_us)} using valid, de-duplicated messages."""
    rec = rec[rec["message_validity"] == 1]
    out = {}
    for (rt, sa), g in mapping.groupby(["remote_terminal_address", "sub_address"]):
        r = rec[(rec["rt"] == rt) & (rec["sa"] == sa)]
        _, first = np.unique(r["timestamp_us"], return_index=True)
        r = r[first]
        for m in g.itertuples():
            raw = (r["data"][:, m.word_position - 1] >> m.bit_position) & ((1 << m.bit_length) - 1)
            out[m.parameter_id] = pd.Series(raw * m.scaling_factor + m.offset, index=r["timestamp_us"])
    return out


def validate(out):
    raw, vdir = os.path.join(out, "raw"), os.path.join(out, "validation")
    os.makedirs(vdir, exist_ok=True)
    mapping = pd.read_csv(f"{raw}/icd/bus1553_mapping.csv")
    pdict = pd.read_csv(f"{raw}/icd/parameter_dictionary.csv").set_index("parameter_id")
    files = sorted(glob.glob(f"{raw}/bus1553/*/*.bin"))[:6] + sorted(glob.glob(f"{raw}/bus1553/*/*.hex"))[:1]
    checks, stats = {}, {}
    corr = {p: [] for p in PAIRS}
    acf, rate_err, invalid, dup = [], [], [], []
    sig = None
    for f in files:
        rec = read_bus(f)
        invalid.append(float((rec["message_validity"] == 0).mean()))
        dup.append(1 - len(np.unique(rec[["timestamp_us", "rt", "sa"]])) / len(rec))
        vals = decode(rec, mapping)
        sig = sig or vals
        for m in mapping[mapping.rate_class != "EVENT"].drop_duplicates("message_id").itertuples():
            ts = vals[m.parameter_id].index.values
            if len(ts) > 10:
                rate_err.append(abs(np.median(np.diff(ts)) / 1e6 * m.rate_hz - 1))
        for a, b in PAIRS:
            if a in vals and b in vals:
                x = vals[a]
                y = vals[b].reindex(x.index, method="nearest")
                corr[(a, b)].append(float(np.corrcoef(x, y)[0, 1]))
        for p in pdict.index[(pdict.rate_class == "FAST") & (pdict.kind == "analog") & (pdict.source == "1553")]:
            v = vals[p].values
            if v.std() > 0:
                acf.append(float(np.corrcoef(v[:-1], v[1:])[0, 1]))
        lo, hi = pdict.loc[list(vals), "min_value"], pdict.loc[list(vals), "max_value"]
        mins = pd.Series({k: v.min() for k, v in vals.items()})
        maxs = pd.Series({k: v.max() for k, v in vals.items()})
        checks.setdefault("decoded_values_within_ranges", True)
        checks["decoded_values_within_ranges"] &= bool(((mins >= lo - 1e-6) & (maxs <= hi + 1e-6)).all())
    stats["correlations"] = {f"{a}~{b}": float(np.mean(v)) for (a, b), v in corr.items() if v}
    checks["cross_parameter_correlation"] = all(SIGNS.get((a, b), 1) * np.mean(v) > 0.6 for (a, b), v in corr.items() if v)
    stats["median_lag1_autocorrelation_fast"] = float(np.median(acf))
    checks["temporal_dynamics"] = stats["median_lag1_autocorrelation_fast"] > 0.8
    stats["max_relative_rate_error"] = float(np.max(rate_err))
    checks["sampling_rates_match_icd"] = float(np.median(rate_err)) < 0.05
    stats["invalid_message_fraction"], stats["duplicate_fraction"] = float(np.mean(invalid)), float(np.mean(dup))
    checks["imperfections_present"] = 0 < stats["invalid_message_fraction"] < 0.01 and 0 < stats["duplicate_fraction"] < 0.01

    logs = [pd.read_csv(f) for f in sorted(glob.glob(f"{raw}/lru_logs/*/*.csv"))[:40]]
    stats["lru_log_nan_fraction"] = float(np.mean([l.iloc[:, 3:].isna().mean().mean() for l in logs]))
    checks["lru_log_quality_realistic"] = 0 < stats["lru_log_nan_fraction"] < 0.02

    tl = pd.read_parquet(f"{out}/truth/degradation_timeline.parquet")
    faults = pd.read_parquet(f"{out}/truth/fault_truth.parquet")
    mono = [bool(np.all(np.diff(g.severity_end.values) >= -1e-9)) for _, g in tl.groupby("fault_id")]
    checks["failure_progression_monotonic"] = all(mono)
    maint = pd.read_csv(f"{raw}/maintenance/maintenance_records.csv")
    snags = pd.read_csv(f"{raw}/snags/snag_reports.csv")
    replaced = faults[faults.end_reason.str.startswith("REPLACED")]
    checks["replacements_have_maintenance_record"] = all(
        ((maint.flight_id == r.end_flight_id) & (maint.lru_id == r.lru_id) & maint.removed_serial_number.notna()).any()
        for r in replaced.itertuples())
    ms = maint.dropna(subset=["snag_id"]).merge(snags, on="snag_id", suffixes=("", "_snag"))
    checks["maintenance_after_snag"] = bool((pd.to_datetime(ms.maintenance_date) >= pd.to_datetime(ms.snag_date)).all())
    stats["snags_per_lru_max"] = int(snags.groupby("lru_id").size().max())
    rep = faults[faults.scenario == "REPEATED_SNAG"]
    checks["repeated_snag_scenario"] = rep.empty or bool((snags.lru_id.isin(rep.lru_id).sum()) >= 2)

    fta = pd.concat(pd.read_csv(f) for f in glob.glob(f"{raw}/fta/*.csv"))
    parents = dict(zip(fta.child_event, fta.parent_event.fillna("")))
    def reaches_top(e, depth=0):
        return depth < 10 and (str(e).startswith("TE-") or reaches_top(parents.get(e, ""), depth + 1) if e else False)
    checks["fta_covers_all_injected_mechanisms"] = set(faults.mechanism_code) <= set(fta.event_code)
    checks["fta_events_reach_top_event"] = all(reaches_top(e) for e in fta.event_code)
    docs = " ".join(open(f, errors="ignore").read() for f in glob.glob(f"{raw}/documents/*.txt"))
    fims = set(maint.maintenance_reference.dropna()) - {""}
    stats["fim_references"] = len(fims)
    checks["fta_references_resolvable"] = set(fta.reference_document.dropna()) >= {f for f in fims if f.startswith("FIM-")}
    del docs

    # plots
    t0 = min(s.index.min() for s in sig.values())
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for a, p in zip(ax, ["ENGINE_RPM", "ENGINE_FUEL_FLOW", "ENGINE_EXHAUST_TEMPERATURE", "ALTITUDE"]):
        a.plot((sig[p].index - t0) / 1e6, sig[p].values, lw=0.7)
        a.set_ylabel(p, fontsize=7)
    ax[-1].set_xlabel("s")
    fig.savefig(f"{vdir}/signals_flight.png", dpi=90)
    plt.close(fig)
    names = ["ENGINE_RPM", "ENGINE_FUEL_FLOW", "ENGINE_EXHAUST_TEMPERATURE", "ENGINE_OIL_TEMPERATURE",
             "HYD_A_PRESSURE", "GENERATOR_LOAD", "ALTITUDE", "STATIC_PRESSURE", "ENGINE_VIBRATION"]
    base = sig["ENGINE_RPM"]
    df = pd.DataFrame({n: sig[n].reindex(base.index, method="nearest").values for n in names if n in sig})
    fig, a = plt.subplots(figsize=(8, 7))
    im = a.imshow(df.corr(), vmin=-1, vmax=1, cmap="coolwarm")
    a.set_xticks(range(len(df.columns)), df.columns, rotation=90, fontsize=6)
    a.set_yticks(range(len(df.columns)), df.columns, fontsize=6)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{vdir}/correlation_matrix.png", dpi=90)
    plt.close(fig)
    fig, a = plt.subplots(figsize=(9, 4))
    v = sig["ENGINE_VIBRATION"].values
    fr, pw = welch(v - v.mean(), fs=5.0)
    a.semilogy(fr, pw)
    a.set_xlabel("Hz")
    a.set_title("ENGINE_VIBRATION spectrum")
    fig.savefig(f"{vdir}/spectrum.png", dpi=90)
    plt.close(fig)
    fig, a = plt.subplots(figsize=(10, 5))
    for fid, g in tl.groupby("fault_id"):
        a.plot(range(len(g)), g.severity_end.values, marker=".", lw=0.8)
    a.set_xlabel("flight index since onset")
    a.set_ylabel("severity (truth)")
    fig.savefig(f"{vdir}/degradation_trajectories.png", dpi=90)
    plt.close(fig)

    report = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, stats=stats,
                  files_checked=[os.path.relpath(f, raw) for f in files])
    with open(f"{vdir}/validation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    with open(f"{vdir}/validation_report.md", "w") as fh:
        fh.write(f"# Synthetic data validation: {report['status']}\n\n| check | result |\n|---|---|\n")
        fh.writelines(f"| {k} | {'PASS' if v else 'FAIL'} |\n" for k, v in checks.items())
        fh.write("\n```json\n" + json.dumps(stats, indent=2) + "\n```\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    r = validate(ap.parse_args().out)
    print(json.dumps(r, indent=2))
