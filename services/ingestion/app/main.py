"""ingestion-service: raw catalog, 1553 decoding, LRU-log clock sync, cleaning, quality, phases, features."""
import math
import os
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from arpms_common import RAW_DIR, Jsonb, add_health, audit, ex, get_logger, init_schema, many, q, q1

from . import loaders
from .decode import FormatError, decode, read_bus, read_lru_log
from .features import lru_features
from .signal import clock_sync, detect_phases, gap_count, kalman_clean, quality_score, stuck_fraction, to_grid

log = get_logger()
app = FastAPI(title="ARPMS ingestion-service", version="1.0")
add_health(app)
UPLOAD_DIRS = ("bus1553/", "lru_logs/", "snags/", "maintenance/", "documents/", "fta/", "flights/", "fleet/", "icd/")


@app.on_event("startup")
def _startup():
    init_schema()


def nan_list(a, nd=4):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in a]


def process_flight(flight_id):
    t_start = time.time()
    fl = q1("SELECT * FROM flights WHERE flight_id=%s", (flight_id,))
    bus = q1("SELECT * FROM raw_files WHERE flight_id=%s AND file_type='BUS_1553'", (flight_id,))
    if not fl or not bus:
        raise HTTPException(404, f"flight or bus file not found for {flight_id}")
    mapping = pd.DataFrame(q("SELECT * FROM bus_mapping"))
    params = pd.DataFrame(q("SELECT * FROM parameters")).set_index("parameter_id")
    lru_by_type = {r["lru_type"]: r["lru_id"] for r in q("SELECT lru_type, lru_id FROM lru WHERE aircraft_id=%s", (fl["aircraft_id"],))}
    try:
        hdr, rec = read_bus(os.path.join(RAW_DIR, bus["path"]))
    except FormatError as e:
        ex("UPDATE raw_files SET status='ERROR', error=%s WHERE file_id=%s", (str(e), bus["file_id"]))
        raise HTTPException(422, str(e))
    if hdr["flight_id"] != flight_id or hdr["aircraft_id"] != fl["aircraft_id"]:
        raise HTTPException(422, f"bus header {hdr['aircraft_id']}/{hdr['flight_id']} does not match {flight_id}")
    series, errors, msg_stats = decode(rec, mapping)
    start_us = hdr["start_us"]
    dur = (fl["arrival_time"] - fl["departure_time"]).total_seconds()
    n = int(math.ceil(dur))
    end_us = start_us + int(dur * 1e6)
    raw = {p: to_grid(ts, v, start_us, n) for p, (ts, v) in series.items()}
    completeness, validity, gaps, src = {}, {}, {}, {}
    rate = mapping.set_index("parameter_id")
    for p, (ts, _) in series.items():
        m = rate.loc[p]
        st = msg_stats[m.message_id]
        completeness[p] = min(1.0, len(ts) / max(dur * m.rate_hz, 1)) if m.rate_class != "EVENT" else float(len(ts) > 0)
        validity[p] = 1 - (st["invalid"] + st["parity"]) / max(st["total"], 1)
        gaps[p] = gap_count(ts, m.rate_hz, start_us, end_us) if m.rate_class != "EVENT" else 0
        src[p] = bus["file_id"]
    sync_rows = []
    for lf in q("SELECT * FROM raw_files WHERE flight_id=%s AND file_type='LRU_LOG' ORDER BY path", (flight_id,)):
        try:
            df = read_lru_log(os.path.join(RAW_DIR, lf["path"]))
        except FormatError as e:
            ex("UPDATE raw_files SET status='ERROR', error=%s WHERE file_id=%s", (str(e), lf["file_id"]))
            continue
        lru_type = os.path.splitext(lf["path"])[0].rsplit("_", 1)[1]
        shared = next((c for c in df.columns if c in raw), None)
        if shared is None:
            continue
        rel = df.lru_time.values - start_us / 1e6
        s = clock_sync(raw[shared], rel, df[shared].values.astype(float))
        bus_t = (rel - s["offset_s"]) / (1 + s["drift_ppm"] * 1e-6)
        for p in df.columns:
            if p in ("lru_time", "lru_serial", shared) or p not in params.index:
                continue
            v = df[p].values.astype(float)
            raw[p] = to_grid(bus_t * 1e6 + start_us, v, start_us, n)
            completeness[p] = float(np.isfinite(v).sum() / max(dur, 1))
            validity[p], gaps[p], src[p] = 1.0, int((np.diff(bus_t) > 3).sum()), lf["file_id"]
        sync_rows.append((flight_id, lru_by_type.get(lru_type, f"{fl['aircraft_id']}-{lru_type}"), s["offset_s"],
                          s["drift_ppm"], s["dtw_cost"], s["method"]))
        ex("UPDATE raw_files SET status='PROCESSED', processed_at=now() WHERE file_id=%s", (lf["file_id"],))
    names = [p for p in raw if p in params.index]
    analog = [p for p in names if params.loc[p, "kind"] == "analog"]
    clean = {}
    if analog:
        sm, out = kalman_clean(np.vstack([raw[p] for p in analog]))
        for i, p in enumerate(analog):
            clean[p] = np.clip(sm[i], params.loc[p, "min_value"], params.loc[p, "max_value"])
            raw_outliers = out[i].sum() / max(np.isfinite(raw[p]).sum(), 1)
            completeness[p + "#out"] = float(raw_outliers)
    for p in names:
        if p not in clean:
            clean[p] = pd.Series(raw[p]).ffill().bfill().fillna(0).values
    labels, segments = detect_phases(*(clean.get(k, np.zeros(n)) for k in
                                       ("ALTITUDE", "INDICATED_AIRSPEED", "ENGINE_RPM", "VERTICAL_SPEED", "ACCEL_Z", "ROLL")))
    qrows, scores = [], []
    for p in names:
        outl = completeness.pop(p + "#out", 0.0)
        stuck = stuck_fraction(raw[p], params.loc[p, "min_value"], params.loc[p, "max_value"]) if p in analog else 0.0
        sc = quality_score(completeness.get(p, 0), validity.get(p, 1), outl, stuck)
        scores.append(sc)
        issues = [k for k, bad in (("LOW_COMPLETENESS", completeness.get(p, 0) < 0.9), ("INVALID_MESSAGES", validity.get(p, 1) < 0.995),
                                   ("OUTLIERS", outl > 0.005), ("STUCK", stuck > 0.02), ("GAPS", gaps.get(p, 0) > 0)) if bad]
        qrows.append((flight_id, p, completeness.get(p, 0), validity.get(p, 1), outl, stuck, gaps.get(p, 0), sc, Jsonb(issues)))
    ts_sorted = np.sort(rec["timestamp_us"])
    ids_sorted = rec["message_id"][np.argsort(rec["timestamp_us"], kind="stable")]
    frows = []
    for lru_type, grp in params.loc[names].groupby("lru_type"):
        lru_id = lru_by_type.get(lru_type, f"{fl['aircraft_id']}-{lru_type}")
        kinds = grp.kind.to_dict()
        for w, s, e, ph, feats in lru_features(clean, list(grp.index), labels, kinds):
            a, b = np.searchsorted(ts_sorted, [start_us + s * 1e6, start_us + e * 1e6])
            ref = dict(bus_file=bus["path"], bus_file_id=bus["file_id"], start_us=int(start_us + s * 1e6),
                       end_us=int(start_us + e * 1e6), first_message_id=int(ids_sorted[a]) if a < len(ids_sorted) else None,
                       last_message_id=int(ids_sorted[b - 1]) if b > a else None,
                       log_file_ids=sorted({int(src[p]) for p in grp.index if src.get(p) and src[p] != bus["file_id"]}))
            frows.append((flight_id, lru_id, w, s, e, ph, Jsonb(feats), Jsonb(ref)))
    for t in ("telemetry_series", "data_quality", "flight_phases", "clock_sync", "features", "bus_message_errors"):
        ex(f"DELETE FROM {t} WHERE flight_id=%s", (flight_id,))
    t0 = fl["departure_time"]
    many("INSERT INTO telemetry_series VALUES (%s,%s,%s,%s,1.0,%s,%s,%s)",
         [(flight_id, p, lru_by_type.get(params.loc[p, "lru_type"]), t0, nan_list(raw[p]), nan_list(clean[p]), src.get(p))
          for p in names])
    many("INSERT INTO data_quality VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", qrows)
    many("INSERT INTO flight_phases VALUES (%s,%s,%s,%s,%s)",
         [(flight_id, i, s["phase"], s["start_s"], s["end_s"]) for i, s in enumerate(segments)])
    many("INSERT INTO clock_sync VALUES (%s,%s,%s,%s,%s,%s)", sync_rows)
    many("INSERT INTO features VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", frows)
    many("INSERT INTO bus_message_errors (file_id, flight_id, message_id, timestamp_us, rt, sa, error_type, raw_hex) "
         "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
         [(bus["file_id"], flight_id, e["message_id"], e["timestamp_us"], e["rt"], e["sa"], e["error_type"], e["raw_hex"])
          for e in errors])
    qs = float(np.mean(scores)) if scores else 0.0
    ex("UPDATE flights SET quality_score=%s, ingest_status='INGESTED', processed_at=now() WHERE flight_id=%s", (qs, flight_id))
    ex("UPDATE raw_files SET status='PROCESSED', processed_at=now() WHERE file_id=%s", (bus["file_id"],))
    summary = dict(flight_id=flight_id, messages=int(len(rec)), errors=len(errors), parameters=len(names),
                   phases=len(segments), windows=len(frows), lru_logs=len(sync_rows), quality_score=round(qs, 2),
                   seconds=round(time.time() - t_start, 2))
    audit("ingest_flight", "flight", flight_id, summary)
    log.info(f"ingested {summary}")
    return summary


# ---------------------------------------------------------------- write API
@app.post("/api/v1/ingest/catalog")
def api_catalog():
    return loaders.catalog(RAW_DIR)


@app.post("/api/v1/ingest/reference")
def api_reference():
    return loaders.load_reference(RAW_DIR)


@app.post("/api/v1/ingest/flights/{flight_id}")
def api_ingest_flight(flight_id: str):
    return process_flight(flight_id)


@app.post("/api/v1/ingest/upload")
async def api_upload(relative_path: str = Form(...), file: UploadFile = File(...)):
    """Add a raw file. Raw data is immutable: existing paths are never overwritten."""
    rel = os.path.normpath(relative_path).lstrip("/")
    if rel.startswith("..") or not rel.startswith(UPLOAD_DIRS) or not loaders.file_type(rel):
        raise HTTPException(400, f"path must be under one of {UPLOAD_DIRS}")
    full = os.path.join(RAW_DIR, rel)
    if os.path.exists(full):
        raise HTTPException(409, "raw file already exists (raw data is immutable)")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "xb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    audit("upload", "raw_file", rel, {"bytes": os.path.getsize(full)})
    return {"path": rel, **loaders.catalog(RAW_DIR)}


# ---------------------------------------------------------------- read API
@app.get("/api/v1/files")
def api_files(file_type: str | None = None, status: str | None = None, limit: int = 500):
    return q("SELECT * FROM raw_files WHERE (%(t)s::text IS NULL OR file_type=%(t)s) AND (%(s)s::text IS NULL OR status=%(s)s) "
             "ORDER BY file_id LIMIT %(l)s", dict(t=file_type, s=status, l=limit))


@app.get("/api/v1/aircraft")
def api_aircraft():
    return q("""SELECT a.*, count(f.flight_id) AS flights, max(f.departure_time) AS last_flight,
                avg(f.quality_score) AS mean_quality FROM aircraft a LEFT JOIN flights f USING (aircraft_id)
                GROUP BY a.aircraft_id ORDER BY a.aircraft_id""")


@app.get("/api/v1/aircraft/{aircraft_id}")
def api_aircraft_one(aircraft_id: str):
    a = q1("SELECT * FROM aircraft WHERE aircraft_id=%s", (aircraft_id,))
    if not a:
        raise HTTPException(404, "aircraft not found")
    a["lrus"] = q("SELECT * FROM lru WHERE aircraft_id=%s ORDER BY subsystem, lru_type", (aircraft_id,))
    a["flights"] = q("SELECT * FROM flights WHERE aircraft_id=%s ORDER BY departure_time", (aircraft_id,))
    return a


@app.get("/api/v1/lrus/{lru_id}")
def api_lru(lru_id: str):
    l = q1("SELECT * FROM lru WHERE lru_id=%s", (lru_id,))
    if not l:
        raise HTTPException(404, "LRU not found")
    l["installations"] = q("SELECT * FROM lru_installations WHERE lru_id=%s ORDER BY install_date", (lru_id,))
    l["maintenance"] = q("SELECT * FROM maintenance_records WHERE lru_id=%s ORDER BY maintenance_date", (lru_id,))
    l["parameters"] = q("SELECT * FROM parameters WHERE lru_type=%s ORDER BY parameter_id", (l["lru_type"],))
    return l


@app.get("/api/v1/flights")
def api_flights(aircraft_id: str | None = None, limit: int = 1000):
    return q("SELECT * FROM flights WHERE (%(a)s::text IS NULL OR aircraft_id=%(a)s) ORDER BY departure_time DESC LIMIT %(l)s",
             dict(a=aircraft_id, l=limit))


@app.get("/api/v1/flights/{flight_id}")
def api_flight(flight_id: str):
    f = q1("SELECT * FROM flights WHERE flight_id=%s", (flight_id,))
    if not f:
        raise HTTPException(404, "flight not found")
    f["phases"] = q("SELECT phase, start_s, end_s FROM flight_phases WHERE flight_id=%s ORDER BY seq", (flight_id,))
    f["clock_sync"] = q("SELECT * FROM clock_sync WHERE flight_id=%s", (flight_id,))
    f["files"] = q("SELECT file_id, path, file_type, sha256, status FROM raw_files WHERE flight_id=%s", (flight_id,))
    f["bus_error_counts"] = q("SELECT error_type, count(*) AS n FROM bus_message_errors WHERE flight_id=%s GROUP BY 1", (flight_id,))
    return f


@app.get("/api/v1/flights/{flight_id}/telemetry")
def api_telemetry(flight_id: str, parameters: str = "ENGINE_RPM,ALTITUDE", downsample: int = 1):
    rows = q("SELECT parameter_id, lru_id, t0, hz, raw_values, clean_values, source_file_id FROM telemetry_series "
             "WHERE flight_id=%s AND parameter_id = ANY(%s)", (flight_id, parameters.split(",")))
    for r in rows:
        r["raw_values"], r["clean_values"] = r["raw_values"][::downsample], r["clean_values"][::downsample]
        r["hz"] = r["hz"] / downsample
    return rows


@app.get("/api/v1/flights/{flight_id}/quality")
def api_quality(flight_id: str):
    return q("SELECT d.*, p.lru_type, p.unit FROM data_quality d JOIN parameters p USING (parameter_id) "
             "WHERE flight_id=%s ORDER BY score", (flight_id,))


@app.get("/api/v1/flights/{flight_id}/bus-errors")
def api_bus_errors(flight_id: str, limit: int = 200):
    return q("SELECT * FROM bus_message_errors WHERE flight_id=%s ORDER BY timestamp_us LIMIT %s", (flight_id, limit))


@app.get("/api/v1/flights/{flight_id}/features")
def api_features(flight_id: str, lru_id: str | None = None):
    return q("SELECT * FROM features WHERE flight_id=%(f)s AND (%(l)s::text IS NULL OR lru_id=%(l)s) ORDER BY lru_id, window_id",
             dict(f=flight_id, l=lru_id))


@app.get("/api/v1/parameters")
def api_parameters():
    return q("SELECT p.*, m.message_id, m.remote_terminal_address, m.sub_address, m.word_position, m.rate_hz "
             "FROM parameters p LEFT JOIN bus_mapping m USING (parameter_id) ORDER BY lru_type, parameter_id")


@app.get("/api/v1/maintenance")
def api_maintenance(aircraft_id: str | None = None, lru_id: str | None = None, limit: int = 500):
    return q("SELECT * FROM maintenance_records WHERE (%(a)s::text IS NULL OR aircraft_id=%(a)s) AND "
             "(%(l)s::text IS NULL OR lru_id=%(l)s) ORDER BY maintenance_date DESC LIMIT %(n)s", dict(a=aircraft_id, l=lru_id, n=limit))


@app.get("/api/v1/quality/summary")
def api_quality_summary():
    return {"flights": q("SELECT ingest_status, count(*) n, avg(quality_score) q FROM flights GROUP BY 1"),
            "files": q("SELECT file_type, status, count(*) n FROM raw_files GROUP BY 1, 2 ORDER BY 1"),
            "issues": q("SELECT issue, count(*) n FROM data_quality, jsonb_array_elements_text(issues) issue GROUP BY 1"),
            "clock_sync": q("SELECT method, count(*) n, avg(abs(offset_s)) mean_abs_offset FROM clock_sync GROUP BY 1")}
