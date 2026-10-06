"""Shared helpers for ARPMS services: DB access, audit trail, logging, health/metrics."""
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://arpms:arpms@localhost:5432/arpms")
RAW_DIR = os.environ.get("ARPMS_RAW_DIR", "data/raw")
ARTIFACT_DIR = os.environ.get("ARPMS_ARTIFACT_DIR", "data/artifacts")
SERVICE = os.environ.get("ARPMS_SERVICE", "arpms")


class JsonFormatter(logging.Formatter):
    def format(self, r):
        return json.dumps({"ts": round(r.created, 3), "level": r.levelname, "service": SERVICE, "msg": r.getMessage()})


def get_logger(name=SERVICE):
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return log


def connect(retries=30):
    for i in range(retries):
        try:
            return psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
        except psycopg.OperationalError:
            if i == retries - 1:
                raise
            time.sleep(2)


_conn = None


@contextmanager
def db():
    """One lazily (re)opened connection per process. ponytail: no pool; move to psycopg_pool if load grows."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = connect()
    try:
        yield _conn
    except psycopg.OperationalError:
        _conn = None
        raise


def q(sql, params=None):
    with db() as c:
        return c.execute(sql, params).fetchall()


def q1(sql, params=None):
    rows = q(sql, params)
    return rows[0] if rows else None


def ex(sql, params=None):
    with db() as c:
        return c.execute(sql, params).rowcount


def many(sql, rows):
    if rows:
        with db() as c, c.cursor() as cur:
            cur.executemany(sql, rows)


def init_schema():
    ex(Path(__file__).with_name("schema.sql").read_text())


def audit(action, entity, entity_id, details=None, actor="system"):
    ex("INSERT INTO audit_log (actor, service, action, entity, entity_id, details) VALUES (%s,%s,%s,%s,%s,%s)",
       (actor, SERVICE, action, entity, str(entity_id), Jsonb(details or {})))


def add_health(app):
    """/health and Prometheus-style /metrics on a FastAPI app."""
    from fastapi import Request
    from fastapi.responses import PlainTextResponse

    stats = {"requests": 0, "errors": 0, "latency_s": 0.0}

    @app.middleware("http")
    async def _count(request: Request, call_next):
        t = time.perf_counter()
        try:
            resp = await call_next(request)
        except Exception:
            stats["errors"] += 1
            raise
        stats["requests"] += 1
        stats["errors"] += resp.status_code >= 500
        stats["latency_s"] += time.perf_counter() - t
        return resp

    @app.get("/health")
    def health():
        ok = True
        try:
            q("SELECT 1")
        except Exception:
            ok = False
        return {"service": SERVICE, "status": "ok" if ok else "degraded", "database": ok}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return "".join(f"arpms_{k}{{service=\"{SERVICE}\"}} {v}\n" for k, v in stats.items())


__all__ = ["Jsonb", "q", "q1", "ex", "many", "db", "audit", "init_schema", "get_logger", "add_health", "RAW_DIR", "ARTIFACT_DIR"]
