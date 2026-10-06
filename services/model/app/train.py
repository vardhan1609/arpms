"""Model training & scoring: context-normalised residuals, Isolation Forest, BOCPD, XGBoost (+TreeSHAP), TCN RUL.

Labels come only from maintenance records (confirmed removals) - never from evaluation truth.
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from arpms_common import ARTIFACT_DIR, Jsonb, audit, ex, get_logger, many, q, q1

from . import tcn
from .bocpd import bocpd

warnings.filterwarnings("ignore", category=RuntimeWarning)
log = get_logger()
PHASES = ["PARKED", "TAXI", "TAKEOFF", "CLIMB", "CRUISE", "MANEUVER", "DESCENT", "APPROACH", "LANDING", "POST_FLIGHT"]
GROUND = {"PARKED", "TAXI", "POST_FLIGHT"}
HORIZON = 3  # flights before a confirmed removal labelled with its failure mode
WINDOW = 6  # flights per TCN sequence
RUL_CAP = 3.0  # hours (piecewise-linear RUL target)
HI_COLS = ["anomaly_score", "max_abs_z", "mean_abs_z", "anom_frac", "cp_prob", "p_fail", "flight_hours"]


# ------------------------------------------------------------------ data access
def load_windows(flight_ids=None):
    rows = q("""SELECT f.flight_id, f.lru_id, l.lru_type, f.window_id, f.phase, f.features, fl.aircraft_id,
                       fl.departure_time, fl.flight_hours, fl.cumulative_hours
                FROM features f JOIN lru l USING (lru_id) JOIN flights fl USING (flight_id)
                WHERE f.window_id >= 0 AND (%(f)s::text[] IS NULL OR f.flight_id = ANY(%(f)s))""", dict(f=flight_ids))
    return pd.DataFrame(rows)


def removals():
    """Confirmed removals from maintenance records -> (lru_id, flight_id, event_code)."""
    ref_map = {r["reference_document"]: r["event_code"] for r in q("SELECT reference_document, event_code FROM fta_events "
                                                                   "WHERE reference_document IS NOT NULL")}
    rows = q("""SELECT lru_id, flight_id, maintenance_reference, finding FROM maintenance_records
                WHERE maintenance_type='CORRECTIVE' AND finding LIKE 'CONFIRMED%%' AND coalesce(removed_serial_number,'') <> ''""")
    for r in rows:
        r["event_code"] = ref_map.get(r["maintenance_reference"]) or "BE-" + (r["maintenance_reference"] or "UNK")[4:]
    return pd.DataFrame(rows, columns=["lru_id", "flight_id", "maintenance_reference", "finding", "event_code"])


def flight_order():
    fl = pd.DataFrame(q("SELECT flight_id, aircraft_id, departure_time, flight_hours, cumulative_hours FROM flights "
                        "WHERE ingest_status='INGESTED' ORDER BY aircraft_id, departure_time"))
    fl["idx"] = fl.groupby("aircraft_id").cumcount()
    fl["n"] = fl.groupby("aircraft_id").flight_id.transform("count")
    return fl


# ------------------------------------------------------------------ residual baseline
def context_matrix(df):
    f = df.features
    c = pd.DataFrame({k: f.map(lambda d, k=k: d.get(k)) for k in ("ctx_rpm", "ctx_alt", "ctx_ias", "ctx_nz", "ctx_oat", "ctx_roll")},
                     index=df.index).astype(float).fillna(0)
    c["ctx_roll"] = c.ctx_roll.abs()
    c["rpm2"], c["nz2"], c["ias2"] = c.ctx_rpm ** 2, c.ctx_nz ** 2, c.ctx_ias ** 2
    for p in PHASES:
        c[f"ph_{p}"] = (df.phase == p).astype(float)
    return c.values


def target_matrix(df, targets):
    return pd.DataFrame([{t: d.get(t) for t in targets} for d in df.features], index=df.index).astype(float)


def fit_baseline(win, ref_mask):
    """Per LRU type: Ridge(context) -> expected value of every __mean/__std feature; robust residual sigma."""
    models = {}
    for t, g in win.groupby("lru_type"):
        keys = sorted({k for d in g.features for k in d if not k.startswith("ctx_") and k.endswith(("__mean", "__std"))})
        X, Y = context_matrix(g), target_matrix(g, keys)
        Y = Y.fillna(Y.median())
        ref = ref_mask.loc[g.index].values
        keep = ref.copy()
        for _ in range(2):  # trimmed refit removes residual contamination in the reference set
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[keep], Y.values[keep])
            res = Y.values - m.predict(X)
            sig = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep], 0)), 0)
            sig = np.maximum(sig, 1e-3 * (np.abs(Y.values).mean(0) + 1e-6))
            keep = ref & (np.abs(res / sig).max(1) < 6)
        models[t] = dict(keys=keys, model=m, sigma=sig, median=Y.median().values)
    return models


def residuals(win, base):
    """z-scores per window, per LRU type -> dict type -> (DataFrame z, index)."""
    out = {}
    for t, g in win.groupby("lru_type"):
        b = base.get(t)
        if b is None:
            continue
        Y = target_matrix(g, b["keys"]).fillna(pd.Series(b["median"], index=b["keys"]))
        z = (Y.values - b["model"].predict(context_matrix(g))) / b["sigma"]
        out[t] = pd.DataFrame(np.clip(z, -50, 50), index=g.index, columns=b["keys"])
    return out


def reference_mask(win, fl, rem):
    """Healthy reference windows: first half of each aircraft's flights, not within HORIZON+2 flights of a
    confirmed removal and with no snag on the LRU in the following 3 flights (maintenance data only)."""
    pos = fl.set_index("flight_id")
    w = win.join(pos[["idx", "n"]], on="flight_id")
    bad = set()
    snags = pd.DataFrame(q("SELECT lru_id, flight_id FROM snags WHERE coalesce(lru_id,'') <> '' AND flight_id IN (SELECT flight_id FROM flights)"))
    events = pd.concat([rem[["lru_id", "flight_id"]].assign(span=HORIZON + 2), snags.assign(span=3) if len(snags) else None])
    for e in events.itertuples():
        if e.flight_id not in pos.index:
            continue
        ac, i = pos.loc[e.flight_id, ["aircraft_id", "idx"]]
        for f in fl[(fl.aircraft_id == ac) & (fl.idx <= i) & (fl.idx > i - e.span)].flight_id:
            bad.add((e.lru_id, f))
    return pd.Series([(l, f) not in bad and i < max(n * 0.5, 2) for l, f, i, n in zip(w.lru_id, w.flight_id, w.idx, w.n)],
                     index=win.index)


# ------------------------------------------------------------------ flight-level health indicators
def flight_level(win, zs, iforests):
    rows = []
    for t, z in zs.items():
        g = win.loc[z.index]
        f = iforests[t]
        s = -f["model"].score_samples(z.fillna(0).values)
        ratio = (s - f["median"]) / max(f["thr"] - f["median"], 1e-9)
        g = g.assign(score=ratio)
        air = ~g.phase.isin(GROUND)
        for (fid, lid), gg in g.groupby(["flight_id", "lru_id"]):
            sel = air.loc[gg.index]
            idx = gg.index[sel] if sel.any() else gg.index
            zz = z.loc[idx]
            zm = zz.mean()
            rows.append(dict(flight_id=fid, lru_id=lid, lru_type=t, aircraft_id=gg.aircraft_id.iloc[0],
                             departure_time=gg.departure_time.iloc[0], flight_hours=gg.flight_hours.iloc[0],
                             anomaly_score=float(np.percentile(g.score.loc[idx], 90)),
                             anom_frac=float((g.score.loc[idx] > 1).mean()), max_abs_z=float(zm.abs().max()),
                             mean_abs_z=float(zm.abs().mean()), zmean=zm.to_dict(),
                             windows=[dict(window_id=int(w), score=float(sc), top=zz_top(z.loc[i]))
                                      for i, w, sc in zip(gg.index, gg.window_id, g.score.loc[gg.index])]))
    return pd.DataFrame(rows)


def zz_top(zrow, k=5):
    top = zrow.abs().sort_values(ascending=False).head(k)
    return [dict(feature=f, z=round(float(zrow[f]), 2)) for f in top.index]


def add_changepoints(lvl, rem):
    """BOCPD per LRU over chronologically ordered flight anomaly scores; reset at confirmed removals."""
    lvl = lvl.sort_values(["lru_id", "departure_time"]).copy()
    lvl["cp_prob"] = 0.0
    lvl["run_length"] = 0.0
    removed = set(zip(rem.lru_id, rem.flight_id))
    for lid, g in lvl.groupby("lru_id"):
        seg = np.cumsum([0] + [int((lid, f) in removed) for f in g.flight_id[:-1]])
        for s in np.unique(seg):
            idx = g.index[seg == s]
            x = np.log1p(np.maximum(lvl.loc[idx, "anomaly_score"].values, 0))
            cp, rl = bocpd(x, mu0=float(np.median(x[: max(3, len(x) // 3)])), beta0=0.05)
            lvl.loc[idx, "cp_prob"], lvl.loc[idx, "run_length"] = cp, rl
    return lvl


# ------------------------------------------------------------------ labels
def labels(lvl, fl, rem):
    """Failure-mode label for flights within HORIZON flights up to a confirmed removal; else NORMAL.
    RUL target (hours to removal, capped) for removed units; censored units only where >= RUL_CAP remains."""
    pos = fl.set_index("flight_id")
    lvl = lvl.join(pos[["idx", "cumulative_hours"]], on="flight_id")
    lvl["label"], lvl["rul"] = "NORMAL", np.nan
    for lid, g in lvl.sort_values("idx").groupby("lru_id"):
        r = rem[rem.lru_id == lid]
        start = 0
        for e in r.itertuples():
            if e.flight_id not in pos.index:
                continue
            end_i, end_h = pos.loc[e.flight_id, ["idx", "cumulative_hours"]]
            unit = g[(g.idx >= start) & (g.idx <= end_i)]
            lvl.loc[unit.index[unit.idx > end_i - HORIZON], "label"] = e.event_code
            lvl.loc[unit.index, "rul"] = np.minimum(end_h - unit.cumulative_hours + unit.flight_hours * 0, RUL_CAP)
            start = end_i + 1
        rest = g[g.idx >= start]
        if len(rest):
            remaining = rest.cumulative_hours.max() - rest.cumulative_hours
            ok = rest.index[remaining >= RUL_CAP]
            lvl.loc[ok, "rul"] = RUL_CAP
    return lvl


def class_matrix(lvl):
    Z = pd.DataFrame([{f"{t}:{k}": v for k, v in z.items()} for t, z in zip(lvl.lru_type, lvl.zmean)], index=lvl.index)
    for c in ("anomaly_score", "anom_frac", "max_abs_z", "mean_abs_z", "cp_prob"):
        Z[c] = lvl[c].values
    for t in sorted(lvl.lru_type.unique()):
        Z[f"type:{t}"] = (lvl.lru_type == t).astype(float).values
    return Z


def sequences(lvl, cols=HI_COLS):
    lvl = lvl.sort_values(["lru_id", "departure_time"])
    X = np.zeros((len(lvl), WINDOW, len(cols)), np.float32)
    for lid, g in lvl.groupby("lru_id"):
        v = g[cols].fillna(0).values
        for j, ix in enumerate(g.index):
            seq = v[max(0, j - WINDOW + 1): j + 1]
            X[lvl.index.get_loc(ix), WINDOW - len(seq):] = seq
    return lvl, X


def calibrate(p, pads):
    p = np.array(p, float)
    p[:, 0] -= pads[0]
    p[:, 2] += pads[1]
    return np.clip(p, 0, None)


# ------------------------------------------------------------------ registry
def register(model_type, obj, metrics, training, params, scope="fleet"):
    v = (q1("SELECT max(version) v FROM models WHERE model_type=%s", (model_type,))["v"] or 0) + 1
    mid = f"{model_type}-v{v}"
    os.makedirs(f"{ARTIFACT_DIR}/models", exist_ok=True)
    path = f"{ARTIFACT_DIR}/models/{mid}.joblib"
    joblib.dump(obj, path)
    ex("UPDATE models SET is_active=FALSE WHERE model_type=%s", (model_type,))
    ex("INSERT INTO models VALUES (%s,%s,%s,%s,now(),%s,%s,%s,%s,TRUE)",
       (mid, model_type, v, scope, Jsonb(training), Jsonb(params), Jsonb(metrics), path))
    audit("register_model", "model", mid, {"metrics": metrics})
    return mid


def active(model_type):
    r = q1("SELECT * FROM models WHERE model_type=%s AND is_active", (model_type,))
    return (r["model_id"], joblib.load(r["artifact_path"])) if r else (None, None)


# ------------------------------------------------------------------ training
def train_all(seed=0):
    fl = flight_order()
    win = load_windows()
    if win.empty:
        raise ValueError("no features - ingest flights first")
    rem = removals()
    ref = reference_mask(win, fl, rem)
    training = dict(flights=int(fl.shape[0]), aircraft=int(fl.aircraft_id.nunique()), windows=int(len(win)),
                    reference_windows=int(ref.sum()), label_source="maintenance_records (confirmed removals)",
                    label_horizon_flights=HORIZON, confirmed_removals=int(len(rem)))
    base = fit_baseline(win, ref)
    bid = register("residual_baseline", base, {"lru_types": len(base)}, training, {"model": "Ridge(alpha=1) on context"})
    zs = residuals(win, base)
    iforests = {}
    for t, z in zs.items():
        r = ref.loc[z.index].values
        m = IsolationForest(n_estimators=200, random_state=seed).fit(z.values[r])
        s = -m.score_samples(z.values[r])
        iforests[t] = dict(model=m, median=float(np.median(s)), thr=float(np.percentile(s, 99)))
    iid = register("isolation_forest", iforests, {"lru_types": len(iforests)}, training, {"n_estimators": 200, "threshold": "p99 ref"})
    lvl = add_changepoints(flight_level(win, zs, iforests), rem)
    lvl = labels(lvl, fl, rem).reset_index(drop=True)
    # flight-level anomaly alert threshold: p99 of LRU-flights not labelled as pre-removal (maintenance labels, not truth)
    alert_thr = float(np.percentile(lvl.anomaly_score[lvl.label == "NORMAL"], 99))
    register("bocpd", {"hazard": 1 / 20, "beta0": 0.05, "alert_threshold": alert_thr}, {"anomaly_alert_threshold": alert_thr},
             training, {"hazard": 1 / 20})

    # XGBoost failure-mode classifier, grouped CV by aircraft -> out-of-fold probabilities
    Z = class_matrix(lvl)
    counts = lvl.label.value_counts()
    y_lab = lvl.label.where(lvl.label.map(counts) >= 2, "OTHER_FAULT")
    classes = sorted(y_lab.unique(), key=lambda c: (c != "NORMAL", c))
    y = y_lab.map({c: i for i, c in enumerate(classes)}).values
    groups = lvl.aircraft_id.values
    params = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.5,
                  random_state=seed, tree_method="hist", objective="multi:softprob")
    oof = np.zeros((len(lvl), len(classes)))
    oof[:, 0] = 1.0
    metrics = {}
    if len(classes) > 1:
        for tr, te in GroupKFold(n_splits=min(4, len(set(groups)))).split(Z, y, groups):
            present = np.unique(y[tr])
            remap = {c: i for i, c in enumerate(present)}
            m = xgb.XGBClassifier(**params, num_class=max(len(present), 2))
            m.fit(Z.iloc[tr], np.vectorize(remap.get)(y[tr]))
            p = m.predict_proba(Z.iloc[te])
            oof[te] = 0
            oof[np.ix_(te, present)] = p[:, : len(present)]
        pred = oof.argmax(1)
        metrics = dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                       failure_auc=float(roc_auc_score(y != 0, 1 - oof[:, 0])) if (y != 0).any() else None,
                       classes=len(classes), positives=int((y != 0).sum()), cv="GroupKFold by aircraft")
        final = xgb.XGBClassifier(**params, num_class=max(len(classes), 2)).fit(Z, y)
    else:
        final = None
    lvl["p_fail"] = 1 - oof[:, 0]
    cid = register("xgboost_classifier", dict(model=final, classes=classes, columns=list(Z.columns),
                                              oof={(f, l): oof[i].tolist() for i, (f, l) in enumerate(zip(lvl.flight_id, lvl.lru_id))}),
                   metrics, training, params)

    # TCN RUL on generic health-indicator sequences, grouped CV for honest metrics and OOF predictions
    lvl_s, X = sequences(lvl)
    ok = lvl_s.rul.notna().values
    rul_oof = np.full((len(lvl_s), 3), np.nan)
    rmetrics = {}
    if ok.sum() >= 20:
        g = lvl_s.aircraft_id.values
        for tr, te in GroupKFold(n_splits=min(4, len(set(g[ok])))).split(X[ok], groups=g[ok]):
            tri, tei = np.flatnonzero(ok)[tr], np.flatnonzero(ok)[te]
            rul_oof[tei] = tcn.predict(tcn.fit(X[tri], lvl_s.rul.values[tri], seed=seed), X[tei])
        rest = ~ok
        model = tcn.fit(X[ok], lvl_s.rul.values[ok], seed=seed)
        rul_oof[rest] = tcn.predict(model, X[rest])
        yt = lvl_s.rul.values[ok]
        raw_cov = float(np.mean((yt >= rul_oof[ok, 0]) & (yt <= rul_oof[ok, 2])))
        # split-conformal widening of the 10-90% band using out-of-fold errors
        pads = (max(0.0, float(np.quantile(rul_oof[ok, 0] - yt, 0.9))), max(0.0, float(np.quantile(yt - rul_oof[ok, 2], 0.9))))
        rul_oof = calibrate(rul_oof, pads)
        pr = rul_oof[ok]
        rmetrics = dict(mae_hours=float(np.mean(np.abs(pr[:, 1] - yt))), coverage_80=float(np.mean((yt >= pr[:, 0]) & (yt <= pr[:, 2]))),
                        coverage_80_uncalibrated=raw_cov, conformal_pads=pads, samples=int(ok.sum()), cap_hours=RUL_CAP,
                        cv="GroupKFold by aircraft")
    else:
        model, pads = None, (0.0, 0.0)
    rid = register("tcn_rul", dict(model=model, pads=pads, oof={(f, l): rul_oof[i].tolist() for i, (f, l) in enumerate(zip(lvl_s.flight_id, lvl_s.lru_id))}),
                   rmetrics, training, dict(window=WINDOW, features=HI_COLS, quantiles=list(tcn.QUANTILES), epochs=300))
    log.info(f"trained models classifier={metrics} rul={rmetrics}")
    return dict(residual_baseline=bid, isolation_forest=iid, xgboost_classifier=cid, tcn_rul=rid,
                classifier_metrics=metrics, rul_metrics=rmetrics, training=training)


# ------------------------------------------------------------------ scoring
def shap_top(model, classes, columns, Zrow, cls, k=8):
    contrib = model.get_booster().predict(xgb.DMatrix(Zrow[columns]), pred_contribs=True)
    c = contrib[0, cls, :-1] if contrib.ndim == 3 else contrib[0, :-1]
    order = np.argsort(-np.abs(c))[:k]
    return [dict(feature=columns[i], contribution=round(float(c[i]), 4), value=None if pd.isna(Zrow.iloc[0, i]) else round(float(Zrow.iloc[0, i]), 3))
            for i in order if c[i] != 0]


def score_aircraft(aircraft_id):
    """Score every ingested flight of an aircraft chronologically (BOCPD needs the history)."""
    fl = flight_order()
    fids = fl[fl.aircraft_id == aircraft_id].flight_id.tolist()
    if not fids:
        return {"aircraft_id": aircraft_id, "flights": 0}
    ids = {t: active(t) for t in ("residual_baseline", "isolation_forest", "bocpd", "xgboost_classifier", "tcn_rul")}
    if any(v[1] is None for v in ids.values()):
        raise ValueError("models not trained")
    base, iforests, clf, rul = (ids[t][1] for t in ("residual_baseline", "isolation_forest", "xgboost_classifier", "tcn_rul"))
    versions = {t: v[0] for t, v in ids.items()}
    win = load_windows(fids)
    rem = removals()
    lvl = add_changepoints(flight_level(win, residuals(win, base), iforests), rem).reset_index(drop=True)
    lvl["prev_anomaly"] = lvl.groupby("lru_id").anomaly_score.shift(1).fillna(0)
    thr = ids["bocpd"][1].get("alert_threshold", 1.5)
    Z = class_matrix(lvl).reindex(columns=clf["columns"])
    probs = []
    for i, (f, l) in enumerate(zip(lvl.flight_id, lvl.lru_id)):
        p = clf["oof"].get((f, l))
        if p is None and clf["model"] is not None:
            p = clf["model"].predict_proba(Z.iloc[[i]])[0].tolist()
        probs.append(p or [1.0] + [0.0] * (len(clf["classes"]) - 1))
    probs = np.array(probs)
    lvl["p_fail"] = 1 - probs[:, 0]
    lvl_s, X = sequences(lvl)
    preds = np.array([rul["oof"].get((f, l)) or [np.nan] * 3 for f, l in zip(lvl_s.flight_id, lvl_s.lru_id)], float)
    miss = np.isnan(preds[:, 1])
    if miss.any() and rul["model"] is not None:
        preds[miss] = calibrate(tcn.predict(rul["model"], X[miss]), rul["pads"])
    lvl_s[["rul_lower", "rul_hours", "rul_upper"]] = preds
    lvl = lvl.join(lvl_s[["rul_lower", "rul_hours", "rul_upper"]])
    pred_rows, alert_rows, an_rows, cp_rows = [], [], [], []
    for i, r in lvl.iterrows():
        k = int(np.argmax(probs[i]))
        mode = clf["classes"][k]
        shap = shap_top(clf["model"], clf["classes"], clf["columns"], Z.iloc[[i]], k) if clf["model"] is not None else []
        hi = float(np.clip(1 - 0.5 * r.p_fail - 0.5 * min(max(r.anomaly_score, 0) / 3, 1), 0, 1))
        top_z = sorted(r.zmean.items(), key=lambda kv: -abs(kv[1]))[:8]
        pred_rows.append((r.flight_id, r.lru_id, float(r.p_fail), mode,
                          Jsonb({c: round(float(p), 4) for c, p in zip(clf["classes"], probs[i]) if p >= 0.01}),
                          *(None if pd.isna(v) else float(v) for v in (r.rul_hours, r.rul_lower, r.rul_upper)), hi,
                          Jsonb(dict(shap=shap, top_residuals=[dict(feature=f, z=round(float(z), 2)) for f, z in top_z])),
                          float(r.anomaly_score), float(r.cp_prob), Jsonb(versions)))
        an_rows += [(r.flight_id, r.lru_id, w["window_id"], w["score"], w["score"] > 1, Jsonb(w["top"]), versions["isolation_forest"])
                    for w in r.windows]
        if r.cp_prob > 0.5:
            cp_rows.append((aircraft_id, r.lru_id, "anomaly_score", r.flight_id, float(r.cp_prob), float(r.run_length), "bocpd"))
        evidence = dict(failure_mode=mode, failure_probability=round(float(r.p_fail), 3), anomaly_score=round(float(r.anomaly_score), 2),
                        changepoint_probability=round(float(r.cp_prob), 3), rul_hours=None if pd.isna(r.rul_hours) else round(float(r.rul_hours), 2),
                        shap=shap[:5], top_residuals=[dict(feature=f, z=round(float(z), 2)) for f, z in top_z[:5]], model_versions=versions)
        if r.p_fail >= 0.5:
            alert_rows.append((aircraft_id, r.lru_id, r.flight_id, "CRITICAL" if r.p_fail >= 0.8 else "HIGH", "PREDICTED_FAILURE",
                               f"{r.lru_id}: predicted {mode} (p={r.p_fail:.2f})", Jsonb(evidence)))
        elif r.anomaly_score >= thr and (r.cp_prob >= 0.5 or r.prev_anomaly >= thr):
            alert_rows.append((aircraft_id, r.lru_id, r.flight_id, "MEDIUM", "DEGRADATION_ONSET",
                               f"{r.lru_id}: behaviour change detected (anomaly {r.anomaly_score:.1f}, cp {r.cp_prob:.2f})", Jsonb(evidence)))
        if not pd.isna(r.rul_lower) and r.rul_upper < 1.0 and r.p_fail >= 0.3:
            alert_rows.append((aircraft_id, r.lru_id, r.flight_id, "HIGH", "LOW_RUL",
                               f"{r.lru_id}: RUL {r.rul_hours:.2f} h [{r.rul_lower:.2f}-{r.rul_upper:.2f}]", Jsonb(evidence)))
    for t in ("predictions", "anomaly_scores"):
        ex(f"DELETE FROM {t} WHERE flight_id = ANY(%s)", (fids,))
    ex("DELETE FROM changepoints WHERE aircraft_id=%s", (aircraft_id,))
    ex("DELETE FROM alerts WHERE flight_id = ANY(%s) AND status='OPEN'", (fids,))
    many("INSERT INTO predictions (flight_id, lru_id, failure_probability, failure_mode, mode_probabilities, rul_hours, rul_lower, "
         "rul_upper, health_index, shap, anomaly_score, changepoint_probability, model_versions) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
         pred_rows)
    many("INSERT INTO anomaly_scores VALUES (%s,%s,%s,%s,%s,%s,%s)", an_rows)
    many("INSERT INTO changepoints (aircraft_id, lru_id, feature, flight_id, probability, run_length, model_id) VALUES (%s,%s,%s,%s,%s,%s,%s)", cp_rows)
    many("INSERT INTO alerts (aircraft_id, lru_id, flight_id, severity, alert_type, title, detail) VALUES (%s,%s,%s,%s,%s,%s,%s) "
         "ON CONFLICT (flight_id, lru_id, alert_type) DO NOTHING", alert_rows)
    ex("UPDATE flights SET analysis_status='SCORED' WHERE flight_id = ANY(%s)", (fids,))
    audit("score_aircraft", "aircraft", aircraft_id, {"flights": len(fids), "alerts": len(alert_rows), "models": versions})
    return {"aircraft_id": aircraft_id, "flights": len(fids), "predictions": len(pred_rows), "alerts": len(alert_rows)}
