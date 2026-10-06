-- ARPMS relational + vector schema (PostgreSQL 16 + pgvector). Idempotent.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS raw_files (
  file_id SERIAL PRIMARY KEY, path TEXT UNIQUE NOT NULL, file_type TEXT NOT NULL, sha256 TEXT NOT NULL,
  bytes BIGINT, aircraft_id TEXT, flight_id TEXT, status TEXT NOT NULL DEFAULT 'CATALOGED', error TEXT,
  cataloged_at TIMESTAMPTZ DEFAULT now(), processed_at TIMESTAMPTZ);

CREATE TABLE IF NOT EXISTS aircraft (
  aircraft_id TEXT PRIMARY KEY, aircraft_type TEXT, aircraft_variant TEXT, aircraft_serial_number TEXT,
  configuration_id TEXT, software_version TEXT, hardware_configuration TEXT, avionics_version TEXT,
  engine_configuration TEXT, manufacture_date DATE, service_entry_date DATE, total_flight_hours REAL, total_flight_cycles INT);

CREATE TABLE IF NOT EXISTS lru (
  lru_id TEXT PRIMARY KEY, aircraft_id TEXT REFERENCES aircraft, lru_type TEXT, lru_name TEXT, subsystem TEXT,
  remote_terminal_address INT, part_number TEXT, serial_number TEXT, install_date TIMESTAMPTZ);

CREATE TABLE IF NOT EXISTS lru_installations (
  id SERIAL PRIMARY KEY, lru_id TEXT REFERENCES lru, serial_number TEXT, part_number TEXT, install_date TIMESTAMPTZ,
  UNIQUE (lru_id, serial_number));

CREATE TABLE IF NOT EXISTS parameters (
  parameter_id TEXT PRIMARY KEY, lru_type TEXT, unit TEXT, rate_class TEXT, min_value REAL, max_value REAL,
  source TEXT, kind TEXT, decimals INT);

CREATE TABLE IF NOT EXISTS bus_mapping (
  parameter_id TEXT PRIMARY KEY REFERENCES parameters, message_id TEXT, remote_terminal_address INT, sub_address INT,
  word_position INT, bit_position INT, bit_length INT, scaling_factor DOUBLE PRECISION, "offset" DOUBLE PRECISION,
  unit TEXT, rate_class TEXT, rate_hz REAL, lru_type TEXT, mapping_version TEXT DEFAULT 'synthetic-1');

CREATE TABLE IF NOT EXISTS flights (
  flight_id TEXT PRIMARY KEY, aircraft_id TEXT REFERENCES aircraft, flight_number INT, departure_time TIMESTAMPTZ,
  arrival_time TIMESTAMPTZ, mission_type TEXT, base TEXT, flight_hours REAL, cumulative_hours REAL,
  quality_score REAL, ingest_status TEXT DEFAULT 'PENDING', analysis_status TEXT DEFAULT 'PENDING', processed_at TIMESTAMPTZ);
CREATE INDEX IF NOT EXISTS flights_ac ON flights (aircraft_id, departure_time);

CREATE TABLE IF NOT EXISTS flight_phases (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, seq INT, phase TEXT, start_s REAL, end_s REAL,
  PRIMARY KEY (flight_id, seq));

-- invalid / erroneous bus messages only; valid messages stay in the immutable raw file
CREATE TABLE IF NOT EXISTS bus_message_errors (
  id BIGSERIAL PRIMARY KEY, file_id INT REFERENCES raw_files, flight_id TEXT, message_id BIGINT, timestamp_us BIGINT,
  rt INT, sa INT, error_type TEXT, raw_hex TEXT);
CREATE INDEX IF NOT EXISTS bme_flight ON bus_message_errors (flight_id);

-- cleaned, 1 Hz telemetry per flight/parameter (arrays keep row counts small)
CREATE TABLE IF NOT EXISTS telemetry_series (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, parameter_id TEXT REFERENCES parameters, lru_id TEXT,
  t0 TIMESTAMPTZ, hz REAL, raw_values REAL[], clean_values REAL[], source_file_id INT REFERENCES raw_files,
  PRIMARY KEY (flight_id, parameter_id));

CREATE TABLE IF NOT EXISTS clock_sync (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, lru_id TEXT, offset_s REAL, drift_ppm REAL, dtw_cost REAL,
  method TEXT, PRIMARY KEY (flight_id, lru_id));

CREATE TABLE IF NOT EXISTS data_quality (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, parameter_id TEXT, completeness REAL, valid_fraction REAL,
  outlier_fraction REAL, stuck_fraction REAL, gap_count INT, score REAL, issues JSONB, PRIMARY KEY (flight_id, parameter_id));

CREATE TABLE IF NOT EXISTS features (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, lru_id TEXT, window_id INT, start_s REAL, end_s REAL,
  phase TEXT, features JSONB, source_ref JSONB, PRIMARY KEY (flight_id, lru_id, window_id));
CREATE INDEX IF NOT EXISTS features_lru ON features (lru_id);

CREATE TABLE IF NOT EXISTS models (
  model_id TEXT PRIMARY KEY, model_type TEXT, version INT, scope TEXT, created_at TIMESTAMPTZ DEFAULT now(),
  training_data JSONB, params JSONB, metrics JSONB, artifact_path TEXT, is_active BOOLEAN DEFAULT FALSE);

CREATE TABLE IF NOT EXISTS anomaly_scores (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, lru_id TEXT, window_id INT, score REAL, is_anomaly BOOLEAN,
  top_features JSONB, model_id TEXT, PRIMARY KEY (flight_id, lru_id, window_id));

CREATE TABLE IF NOT EXISTS changepoints (
  id BIGSERIAL PRIMARY KEY, aircraft_id TEXT, lru_id TEXT, feature TEXT, flight_id TEXT, probability REAL,
  run_length REAL, model_id TEXT, created_at TIMESTAMPTZ DEFAULT now());

CREATE TABLE IF NOT EXISTS predictions (
  flight_id TEXT REFERENCES flights ON DELETE CASCADE, lru_id TEXT, failure_probability REAL, failure_mode TEXT,
  mode_probabilities JSONB, rul_hours REAL, rul_lower REAL, rul_upper REAL, health_index REAL, shap JSONB,
  anomaly_score REAL, changepoint_probability REAL, model_versions JSONB, created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (flight_id, lru_id));

CREATE TABLE IF NOT EXISTS alerts (
  alert_id BIGSERIAL PRIMARY KEY, aircraft_id TEXT, lru_id TEXT, flight_id TEXT, severity TEXT, alert_type TEXT,
  title TEXT, detail JSONB, status TEXT DEFAULT 'OPEN', created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (flight_id, lru_id, alert_type));

CREATE TABLE IF NOT EXISTS snags (
  snag_id TEXT PRIMARY KEY, aircraft_id TEXT, flight_id TEXT, snag_date TIMESTAMPTZ, reported_by TEXT, lru_id TEXT,
  subsystem TEXT, snag_code TEXT, snag_title TEXT, snag_description TEXT, reported_symptom TEXT, observed_parameter TEXT,
  observed_value TEXT, observed_unit TEXT, flight_phase TEXT, snag_category TEXT, disposition TEXT, action_taken TEXT,
  closure_status TEXT, closure_date TIMESTAMPTZ, maintenance_reference TEXT, embedding vector(384));

CREATE TABLE IF NOT EXISTS maintenance_records (
  maintenance_record_id TEXT PRIMARY KEY, aircraft_id TEXT, lru_id TEXT, part_number TEXT, serial_number TEXT,
  flight_id TEXT, snag_id TEXT, maintenance_date TIMESTAMPTZ, maintenance_type TEXT, reported_problem TEXT,
  maintenance_description TEXT, action_taken TEXT, finding TEXT, test_result TEXT, removed_serial_number TEXT,
  installed_serial_number TEXT, technician_id TEXT, maintenance_reference TEXT, maintenance_status TEXT, labor_hours REAL);

CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT PRIMARY KEY, document_type TEXT, document_name TEXT, version TEXT, revision TEXT, doc_date TEXT,
  subsystem TEXT, lru_type TEXT, file_id INT REFERENCES raw_files);
CREATE TABLE IF NOT EXISTS document_chunks (
  chunk_id BIGSERIAL PRIMARY KEY, document_id TEXT REFERENCES documents ON DELETE CASCADE, section TEXT, page INT,
  text TEXT, embedding vector(384));

CREATE TABLE IF NOT EXISTS fta_events (
  event_code TEXT PRIMARY KEY, fta_document_id TEXT, subsystem TEXT, event_type TEXT, description TEXT,
  gate TEXT, indicating_parameters TEXT[], corrective_action TEXT, reference_document TEXT, page_reference TEXT,
  document_version TEXT);
CREATE TABLE IF NOT EXISTS fta_edges (
  parent_event TEXT, child_event TEXT, gate TEXT, fta_document_id TEXT, PRIMARY KEY (parent_event, child_event));

CREATE TABLE IF NOT EXISTS diagnoses (
  diagnosis_id BIGSERIAL PRIMARY KEY, aircraft_id TEXT, flight_id TEXT, lru_id TEXT, top_event TEXT,
  candidates JSONB, evidence JSONB, model_versions JSONB, created_at TIMESTAMPTZ DEFAULT now(), UNIQUE (flight_id, lru_id));

CREATE TABLE IF NOT EXISTS recommendations (
  recommendation_id BIGSERIAL PRIMARY KEY, diagnosis_id BIGINT REFERENCES diagnoses ON DELETE CASCADE, rank INT,
  event_code TEXT, action TEXT, rationale TEXT, refs JSONB, confidence REAL, status TEXT DEFAULT 'PENDING',
  engineer TEXT, decision_note TEXT, override_action TEXT, decided_at TIMESTAMPTZ);

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(), actor TEXT, service TEXT, action TEXT, entity TEXT,
  entity_id TEXT, details JSONB);

CREATE TABLE IF NOT EXISTS jobs (
  job_id BIGSERIAL PRIMARY KEY, job_type TEXT NOT NULL, payload JSONB DEFAULT '{}', status TEXT DEFAULT 'QUEUED',
  attempts INT DEFAULT 0, error TEXT, result JSONB, created_at TIMESTAMPTZ DEFAULT now(), started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ);
CREATE INDEX IF NOT EXISTS jobs_q ON jobs (status, job_id);
