# ARPMS — AI-based snag rectification & predictive maintenance (synthetic-data PoC)

Microservices platform that ingests raw aircraft data (MIL-STD-1553-like bus captures, LRU logs, snags, maintenance
records, fault trees, engineering documents), detects degradation, predicts failure mode and remaining useful life (RUL),
diagnoses root causes under fault-tree constraints and produces maintenance recommendations that an engineer must
approve. **All data is synthetic**; the system is decision support only.

## Architecture

```
                     browser
                        |
              frontend (nginx :8080)  -- React dashboard + API gateway (/api/v1/* routed by resource)
     ┌──────────┬───────┴──────┬──────────────┬──────────────┐
 ingestion    model        document       diagnosis        worker
 catalog,     residual     SBERT snag &   FTA-constrained   Postgres job queue
 1553 decode, IForest,     document       Bayesian diag.,   (FOR UPDATE SKIP LOCKED),
 DTW clock    BOCPD,       search, FTA    recommendations,  runs the pipeline over
 sync, Kalman XGBoost+SHAP graph          engineer HITL,    the service APIs
 cleaning,    TCN quantile                PDF reports,
 quality,     RUL, registry,              audit
 phases,      alerts
 features
     └──────────┴──────────────┴──────────────┴──────────────┘
                 PostgreSQL 16 + pgvector (relational + vector store)
```

| Service | Path | Owns |
|---|---|---|
| ingestion-service | `services/ingestion` | `raw_files`, aircraft/LRU/flights/reference data, `telemetry_series`, `clock_sync`, `data_quality`, `flight_phases`, `features`, `bus_message_errors` |
| model-service | `services/model` | `models` registry + artifacts, `anomaly_scores`, `changepoints`, `predictions`, `alerts` |
| document-service | `services/document` | `documents`, `document_chunks` (vector(384)), snag embeddings, `fta_events`, `fta_edges` |
| diagnosis-service | `services/diagnosis` | `diagnoses`, `recommendations`, `audit_log` read API, PDF reports |
| worker | `services/worker` | `jobs` |
| shared | `services/common` | DB helpers, schema, health/metrics middleware, JSON logging, audit |
| generator | `synthetic_generator` | deterministic synthetic dataset (`data/raw` + private `data/truth`) |
| evaluation | `evaluation` | truth-based evaluation report, NASA C-MAPSS RUL benchmark adapter |

Every service exposes `/health` and `/metrics` and OpenAPI docs at `/docs`.

**Truth separation:** product services mount only `./data/raw` (read-only except ingestion, which may add uploaded
files but never overwrites). `./data/truth` is mounted only by the `evaluation` tool container. Model labels come from
maintenance records (confirmed removals), never from truth.

## Quick start (Docker)

```bash
docker compose build
docker compose run --rm generator          # writes ./data/raw and ./data/truth (ARPMS_SCALE=tiny|small|medium, ARPMS_SEED)
docker compose up -d                       # worker auto-runs the full pipeline on first start
open http://localhost:8080                 # Data & pipeline page shows job progress
docker compose run --rm evaluation         # ./data/evaluation/evaluation_report.{md,json}
```

The default `small` scale (20 aircraft, 160 flights, ~17M observations) runs end to end in roughly 10–15 minutes on
2 CPUs. `medium` produces 5M+ observations per subset and takes proportionally longer. Runtime is fully offline:
the sentence-embedding model (`all-MiniLM-L6-v2`) is baked into the image at build time.

Database credentials in `docker-compose.yml` are local-development defaults; set `POSTGRES_USER/PASSWORD/DB` in `.env`
for anything shared.

## Local development (without Docker)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e services/common
docker run -d --name arpms-db -e POSTGRES_USER=arpms -e POSTGRES_PASSWORD=arpms -p 5432:5432 pgvector/pgvector:pg16
python -m synthetic_generator.generate --scale small --out data
# one terminal per service (ports match frontend/vite.config.ts)
(cd services/ingestion && ARPMS_RAW_DIR=../../data/raw uvicorn app.main:app --port 8001)
(cd services/model && ARPMS_RAW_DIR=../../data/raw ARPMS_ARTIFACT_DIR=../../data/artifacts uvicorn app.main:app --port 8002)
(cd services/document && ARPMS_RAW_DIR=../../data/raw ARPMS_SBERT_MODEL=/path/to/all-MiniLM-L6-v2 uvicorn app.main:app --port 8003)
(cd services/diagnosis && uvicorn app.main:app --port 8004)
(cd services/worker && INGESTION_URL=http://localhost:8001 MODEL_URL=http://localhost:8002 DOCUMENT_URL=http://localhost:8003 \
   DIAGNOSIS_URL=http://localhost:8004 uvicorn app.main:app --port 8005)
(cd frontend && npm install && npm run dev)
curl -X POST localhost:8005/api/v1/pipeline/run
```

## Pipeline

1. **Catalog** every raw file with SHA-256 provenance (changed content for a known path is flagged, never overwritten).
2. **Reference data**: fleet, LRU installations (serial-number history), flights, snags, maintenance, parameter dictionary, bus ICD.
3. **Per flight**: decode `.bin`/`.hex` 1553 captures (invalid/parity-failed messages excluded and indexed with raw hex,
   duplicates removed, ICD scaling) → 1 Hz grid → **LRU clock synchronisation** (derivative cross-correlation + banded DTW
   on a shared altitude signal) → **Kalman/RTS cleaning** with outlier gating and level-change persistence →
   **data-quality** scoring (completeness, validity, outliers, stuck, gaps) → **phase detection** → 60 s window and
   per-phase features with source-message traceability.
4. **Models** (`POST /api/v1/models/train`): context-normalised Ridge baselines → residual z-scores; Isolation Forest per
   LRU type on reference (healthy) windows; BOCPD changepoints over per-LRU flight anomaly scores; XGBoost failure-mode
   classifier with TreeSHAP; TCN quantile (10/50/90%) RUL with split-conformal interval calibration. All CV is grouped
   by aircraft. Models are versioned in the `models` registry.
5. **Scoring** per aircraft → predictions, health index, alerts (`PREDICTED_FAILURE`, `DEGRADATION_ONSET`, `LOW_RUL`).
6. **Diagnosis**: candidates restricted to the LRU's FTA subtree, posterior ∝ maintenance-history prior × model mode
   probability × indicating-parameter overlap × snag/document similarity × AND/NOT gate consistency.
7. **Recommendations** are deterministic rules citing FTA/FIM/MM/test-procedure references (sensor faults: verify before
   removal; repeat defects closed NFF: escalate; RUL upper bound < 1 h: before next flight). Each needs an engineer
   ACCEPT / REJECT (note) / OVERRIDE (note + action); every decision is audited. PDF report per diagnosis.

## Evaluation

`python -m evaluation.evaluate` compares DB outputs with private truth: failure-probability/anomaly AUC, alert
precision/recall, false alerts per 100 healthy LRU-flights, fault detection rate and lead time, failure-mode accuracy,
diagnosis top-1/top-3, RUL MAE and interval coverage, clock-offset error, and a per-scenario table.

`python -m evaluation.cmapss --data-dir CMAPSSData --fd FD001` benchmarks the same TCN RUL model on NASA C-MAPSS
(download the dataset separately).

## Tests

```bash
python tests/test_core.py          # or: pytest tests
python services/model/app/bocpd.py # BOCPD self-check
python -m synthetic_generator.validation --out data   # dataset validation
```

See [docs/data_formats.md](docs/data_formats.md) for raw file formats.
