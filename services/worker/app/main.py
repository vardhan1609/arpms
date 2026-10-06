"""worker: Postgres job queue (SELECT ... FOR UPDATE SKIP LOCKED) that orchestrates the pipeline over the service APIs."""
import json
import os
import threading
import time
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from arpms_common import Jsonb, add_health, audit, ex, get_logger, init_schema, q, q1

URL = {s: os.environ.get(f"{s.upper()}_URL", f"http://{s}-service:8000") for s in ("ingestion", "model", "document", "diagnosis")}
MAX_ATTEMPTS = 3
log = get_logger()
app = FastAPI(title="ARPMS worker", version="1.0")
add_health(app)


def call(service, path, method="POST", timeout=7200):
    req = urllib.request.Request(URL[service] + path, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{service} {path}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}") from e


def run_pipeline(payload):
    """catalog -> reference -> ingest new flights -> knowledge import -> train -> score -> diagnose."""
    out = {"catalog": call("ingestion", "/api/v1/ingest/catalog"), "reference": call("ingestion", "/api/v1/ingest/reference")}
    todo = [f["flight_id"] for f in call("ingestion", "/api/v1/flights?limit=100000", "GET")
            if payload.get("reprocess") or f["ingest_status"] != "INGESTED"]
    # ponytail: flights are ingested sequentially by one worker; run more worker replicas with ingest_flight jobs to parallelise
    failed = []
    for i, fid in enumerate(todo):
        try:
            call("ingestion", f"/api/v1/ingest/flights/{fid}")
        except RuntimeError as e:
            failed.append({"flight_id": fid, "error": str(e)})
        if i % 20 == 0:
            log.info(f"pipeline: ingested {i + 1}/{len(todo)}")
    out["ingested"], out["ingest_failures"] = len(todo) - len(failed), failed
    out["knowledge"] = call("document", "/api/v1/documents/import")
    if payload.get("train", True):
        out["training"] = call("model", "/api/v1/models/train")
    aircraft = [a["aircraft_id"] for a in call("ingestion", "/api/v1/aircraft", "GET")]
    out["scored"] = [call("model", f"/api/v1/models/score/{a}") for a in aircraft]
    out["diagnoses"] = sum(len(call("diagnosis", f"/api/v1/diagnoses/run/{a}")) for a in aircraft)
    return out


HANDLERS = {
    "pipeline": run_pipeline,
    "ingest_flight": lambda p: call("ingestion", f"/api/v1/ingest/flights/{p['flight_id']}"),
    "train": lambda p: call("model", "/api/v1/models/train"),
    "score_aircraft": lambda p: call("model", f"/api/v1/models/score/{p['aircraft_id']}"),
    "diagnose_aircraft": lambda p: call("diagnosis", f"/api/v1/diagnoses/run/{p['aircraft_id']}"),
    "import_knowledge": lambda p: call("document", "/api/v1/documents/import"),
}


def claim():
    return q1("""UPDATE jobs SET status='RUNNING', started_at=now(), attempts=attempts+1 WHERE job_id = (
                   SELECT job_id FROM jobs WHERE status='QUEUED' ORDER BY job_id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *""")


def work_once():
    job = claim()
    if not job:
        return False
    try:
        result = HANDLERS[job["job_type"]](job["payload"] or {})
        ex("UPDATE jobs SET status='DONE', result=%s, error=NULL, finished_at=now() WHERE job_id=%s",
           (Jsonb(result, dumps=lambda o: json.dumps(o, default=str)), job["job_id"]))
        audit("job_done", "job", job["job_id"], {"type": job["job_type"]})
    except Exception as e:  # any failure is recorded on the job; retried up to MAX_ATTEMPTS
        retry = job["attempts"] < MAX_ATTEMPTS
        ex("UPDATE jobs SET status=%s, error=%s, finished_at=now() WHERE job_id=%s",
           ("QUEUED" if retry else "FAILED", str(e)[:2000], job["job_id"]))
        log.error(f"job {job['job_id']} {job['job_type']} failed (attempt {job['attempts']}): {e}")
        if retry:
            time.sleep(5)
    return True


def loop():
    while True:
        try:
            if not work_once():
                time.sleep(2)
        except Exception as e:  # DB restarts etc.: keep the worker alive
            log.error(f"worker loop: {e}")
            time.sleep(5)


def enqueue(job_type, payload=None):
    if job_type not in HANDLERS:
        raise HTTPException(400, f"job_type must be one of {sorted(HANDLERS)}")
    return q1("INSERT INTO jobs (job_type, payload) VALUES (%s,%s) RETURNING *", (job_type, Jsonb(payload or {})))


@app.on_event("startup")
def _startup():
    init_schema()
    ex("UPDATE jobs SET status='QUEUED' WHERE status='RUNNING'")  # jobs interrupted by a restart
    if os.environ.get("ARPMS_AUTORUN") == "1" and not q1("SELECT 1 FROM jobs WHERE job_type='pipeline'"):
        enqueue("pipeline")
    threading.Thread(target=loop, daemon=True).start()


class JobIn(BaseModel):
    job_type: str
    payload: dict = {}


@app.post("/api/v1/jobs")
def api_enqueue(job: JobIn):
    return enqueue(job.job_type, job.payload)


@app.post("/api/v1/pipeline/run")
def api_pipeline(reprocess: bool = False, train: bool = True):
    return enqueue("pipeline", {"reprocess": reprocess, "train": train})


@app.get("/api/v1/jobs")
def api_jobs(limit: int = 100):
    return q("SELECT job_id, job_type, payload, status, attempts, error, created_at, started_at, finished_at FROM jobs "
             "ORDER BY job_id DESC LIMIT %s", (limit,))


@app.get("/api/v1/jobs/{job_id}")
def api_job(job_id: int):
    j = q1("SELECT * FROM jobs WHERE job_id=%s", (job_id,))
    if not j:
        raise HTTPException(404, "job not found")
    return j
