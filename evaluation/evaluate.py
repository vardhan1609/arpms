"""Evaluate product outputs (DB) against private synthetic ground truth. The only component that reads truth/.

python -m evaluation.evaluate --truth data/truth --out data/evaluation
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from arpms_common import q

RUL_CAP = 3.0


def frame(sql, params=None):
    return pd.DataFrame(q(sql, params))


def evaluate(truth_dir, out_dir):
    faults = pd.read_parquet(f"{truth_dir}/fault_truth.parquet")
    timeline = pd.read_parquet(f"{truth_dir}/degradation_timeline.parquet")
    clocks = pd.read_parquet(f"{truth_dir}/lru_clock_truth.parquet")
    scenarios = json.load(open(f"{truth_dir}/scenarios.json"))
    preds = frame("""SELECT p.flight_id, p.lru_id, p.failure_probability, p.failure_mode, p.anomaly_score, p.rul_hours, p.rul_lower,
                            p.rul_upper, f.aircraft_id, f.departure_time FROM predictions p JOIN flights f USING (flight_id)""")
    if preds.empty:
        raise SystemExit("no predictions in the database - run the pipeline first")
    alerts = frame("SELECT flight_id, lru_id, alert_type, severity FROM alerts")
    lbl = timeline[["flight_id", "lru_id", "mechanism_code", "severity_max", "rul_hours", "sensor_fault"]].rename(columns={"rul_hours": "true_rul"})
    df = preds.merge(lbl, on=["flight_id", "lru_id"], how="left")
    df["faulty"] = df.severity_max.fillna(0) >= 0.3
    df["alerted"] = df.set_index(["flight_id", "lru_id"]).index.isin(alerts.set_index(["flight_id", "lru_id"]).index)
    m = {}
    if df.faulty.any() and (~df.faulty).any():
        m["failure_probability_auc"] = roc_auc_score(df.faulty, df.failure_probability)
        m["anomaly_score_auc"] = roc_auc_score(df.faulty, df.anomaly_score)
    tp, fp = int((df.alerted & df.faulty).sum()), int((df.alerted & ~df.faulty).sum())
    fn = int((~df.alerted & df.faulty).sum())
    m["alert_precision"] = tp / max(tp + fp, 1)
    m["alert_recall_lru_flights"] = tp / max(tp + fn, 1)
    healthy_flights = df[df.severity_max.isna()]
    m["false_alerts_per_100_healthy_lru_flights"] = 100 * healthy_flights.alerted.mean() if len(healthy_flights) else None
    # fault-level detection and lead time (truth RUL at first alert)
    det = []
    for f in faults.itertuples():
        tl = df[(df.lru_id == f.lru_id) & df.mechanism_code.eq(f.mechanism_code)].sort_values("departure_time")
        hit = tl[tl.alerted]
        det.append(dict(fault_id=f.fault_id, scenario=f.scenario, mechanism=f.mechanism_code, detected=len(hit) > 0,
                        lead_time_hours=float(hit.true_rul.iloc[0]) if len(hit) else None,
                        predicted_mode_at_first_alert=hit.failure_mode.iloc[0] if len(hit) else None))
    det = pd.DataFrame(det)
    m["faults"] = len(det)
    m["fault_detection_rate"] = float(det.detected.mean()) if len(det) else None
    m["median_lead_time_hours"] = float(det.lead_time_hours.median()) if det.detected.any() else None
    # failure-mode identification on alerted faulty LRU-flights
    af = df[df.alerted & df.faulty]
    m["mode_accuracy_on_alerted_faulty"] = float((af.failure_mode == af.mechanism_code).mean()) if len(af) else None
    # diagnosis top-1 / top-3
    dg = frame("SELECT flight_id, lru_id, candidates FROM diagnoses").merge(lbl, on=["flight_id", "lru_id"])
    if len(dg):
        ranks = [next((i for i, c in enumerate(r.candidates) if c["event_code"] == r.mechanism_code), 99) for r in dg.itertuples()]
        m["diagnosis_top1"], m["diagnosis_top3"] = float(np.mean([r == 0 for r in ranks])), float(np.mean([r < 3 for r in ranks]))
        m["diagnoses_on_faulty_lru_flights"] = len(dg)
    # RUL against true remaining life (capped like the model target)
    r = df[df.faulty & df.rul_hours.notna()]
    if len(r):
        yt = np.minimum(r.true_rul, RUL_CAP)
        m["rul_mae_hours"] = float(np.mean(np.abs(r.rul_hours - yt)))
        m["rul_interval_coverage"] = float(np.mean((yt >= r.rul_lower) & (yt <= r.rul_upper)))
    cs = frame("SELECT flight_id, lru_id, offset_s FROM clock_sync").merge(clocks, on=["flight_id", "lru_id"])
    if len(cs):
        err = (cs.offset_s - cs.clock_offset_s).abs()
        m["clock_offset_mae_s"], m["clock_offset_p95_s"] = float(err.mean()), float(err.quantile(0.95))
    scen = []
    for ac, name in scenarios.items():
        a = alerts[alerts.flight_id.str.startswith(ac + "-")]
        tf = faults[faults.aircraft_id == ac]
        scen.append(dict(aircraft_id=ac, scenario=name, true_faults=", ".join(tf.mechanism_code) or "-", alerts=len(a),
                         alerted_lrus=", ".join(sorted(a.lru_id.str.replace(ac + "-", "").unique())) or "-",
                         detected=int(det[det.fault_id.isin(tf.fault_id)].detected.sum()) if len(tf) else 0))
    report = dict(metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                  faults=det.to_dict("records"), scenarios=scen)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(report, open(f"{out_dir}/evaluation_report.json", "w"), indent=1, default=str)
    with open(f"{out_dir}/evaluation_report.md", "w") as fh:
        fh.write("# ARPMS evaluation against synthetic ground truth\n\n| metric | value |\n|---|---|\n")
        fh.writelines(f"| {k} | {v} |\n" for k, v in report["metrics"].items())
        fh.write("\n## Scenarios\n\n" + pd.DataFrame(scen).to_markdown(index=False) + "\n\n## Faults\n\n" + det.to_markdown(index=False) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.environ.get("ARPMS_TRUTH_DIR", "data/truth"))
    ap.add_argument("--out", default="data/evaluation")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.truth, a.out)["metrics"], indent=1))
