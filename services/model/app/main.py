"""model-service: training, model registry, anomaly/changepoint/failure/RUL scoring, alerts, explanations."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from arpms_common import add_health, audit, ex, init_schema, q, q1

from . import train

app = FastAPI(title="ARPMS model-service", version="1.0")
add_health(app)


@app.on_event("startup")
def _startup():
    init_schema()


@app.post("/api/v1/models/train")
def api_train(seed: int = 0):
    try:
        return train.train_all(seed)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/v1/models/score/{aircraft_id}")
def api_score(aircraft_id: str):
    try:
        return train.score_aircraft(aircraft_id)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/v1/models")
def api_models():
    return q("SELECT model_id, model_type, version, scope, created_at, training_data, params, metrics, is_active "
             "FROM models ORDER BY model_type, version DESC")


@app.get("/api/v1/models/{model_id}")
def api_model(model_id: str):
    m = q1("SELECT * FROM models WHERE model_id=%s", (model_id,))
    if not m:
        raise HTTPException(404, "model not found")
    return m


@app.get("/api/v1/fleet/health")
def api_fleet_health():
    """Latest prediction per LRU, aggregated per aircraft."""
    return q("""WITH latest AS (
                  SELECT DISTINCT ON (p.lru_id) p.*, f.aircraft_id, f.departure_time FROM predictions p
                  JOIN flights f USING (flight_id) ORDER BY p.lru_id, f.departure_time DESC)
                SELECT aircraft_id, min(health_index) AS min_health, avg(health_index) AS mean_health,
                       max(failure_probability) AS max_failure_probability, min(rul_hours) AS min_rul_hours,
                       (SELECT count(*) FROM alerts a WHERE a.aircraft_id=latest.aircraft_id AND a.status='OPEN') AS open_alerts,
                       max(departure_time) AS last_flight
                FROM latest GROUP BY aircraft_id ORDER BY min_health""")


@app.get("/api/v1/predictions")
def api_predictions(aircraft_id: str | None = None, lru_id: str | None = None, flight_id: str | None = None, limit: int = 2000):
    return q("""SELECT p.*, f.aircraft_id, f.departure_time, f.cumulative_hours, l.lru_type, l.subsystem
                FROM predictions p JOIN flights f USING (flight_id) JOIN lru l USING (lru_id)
                WHERE (%(a)s::text IS NULL OR f.aircraft_id=%(a)s) AND (%(l)s::text IS NULL OR p.lru_id=%(l)s)
                  AND (%(f)s::text IS NULL OR p.flight_id=%(f)s)
                ORDER BY f.departure_time, p.lru_id LIMIT %(n)s""", dict(a=aircraft_id, l=lru_id, f=flight_id, n=limit))


@app.get("/api/v1/anomalies")
def api_anomalies(flight_id: str, lru_id: str | None = None):
    return q("SELECT * FROM anomaly_scores WHERE flight_id=%(f)s AND (%(l)s::text IS NULL OR lru_id=%(l)s) ORDER BY lru_id, window_id",
             dict(f=flight_id, l=lru_id))


@app.get("/api/v1/changepoints")
def api_changepoints(aircraft_id: str | None = None):
    return q("SELECT c.*, f.departure_time FROM changepoints c JOIN flights f USING (flight_id) "
             "WHERE (%(a)s::text IS NULL OR c.aircraft_id=%(a)s) ORDER BY f.departure_time", dict(a=aircraft_id))


@app.get("/api/v1/alerts")
def api_alerts(status: str | None = "OPEN", aircraft_id: str | None = None, limit: int = 500):
    return q("""SELECT a.*, f.departure_time FROM alerts a LEFT JOIN flights f USING (flight_id)
                WHERE (%(s)s::text IS NULL OR a.status=%(s)s) AND (%(a)s::text IS NULL OR a.aircraft_id=%(a)s)
                ORDER BY CASE a.severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END,
                         f.departure_time DESC LIMIT %(n)s""", dict(s=status, a=aircraft_id, n=limit))


class AlertUpdate(BaseModel):
    status: str
    actor: str = "engineer"
    note: str | None = None


@app.patch("/api/v1/alerts/{alert_id}")
def api_alert_update(alert_id: int, body: AlertUpdate):
    if body.status not in ("OPEN", "ACKNOWLEDGED", "CLOSED"):
        raise HTTPException(400, "status must be OPEN, ACKNOWLEDGED or CLOSED")
    if not ex("UPDATE alerts SET status=%s WHERE alert_id=%s", (body.status, alert_id)):
        raise HTTPException(404, "alert not found")
    audit("alert_status", "alert", alert_id, {"status": body.status, "note": body.note}, actor=body.actor)
    return q1("SELECT * FROM alerts WHERE alert_id=%s", (alert_id,))


@app.get("/api/v1/explanations/{flight_id}/{lru_id}")
def api_explain(flight_id: str, lru_id: str):
    p = q1("SELECT * FROM predictions WHERE flight_id=%s AND lru_id=%s", (flight_id, lru_id))
    if not p:
        raise HTTPException(404, "no prediction")
    p["windows"] = q("SELECT window_id, score, is_anomaly, top_features FROM anomaly_scores WHERE flight_id=%s AND lru_id=%s "
                     "ORDER BY window_id", (flight_id, lru_id))
    p["source"] = q("SELECT window_id, start_s, end_s, phase, source_ref FROM features WHERE flight_id=%s AND lru_id=%s AND window_id>=0 "
                    "ORDER BY window_id", (flight_id, lru_id))
    return p
