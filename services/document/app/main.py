"""document-service: FTA import + graph, engineering-document retrieval (RAG), SBERT snag search. Fully offline."""
import os
import re
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pypdf import PdfReader

from arpms_common import RAW_DIR, add_health, audit, ex, init_schema, many, q, q1

MODEL_PATH = os.environ.get("ARPMS_SBERT_MODEL", "models/all-MiniLM-L6-v2")
FTA_REQUIRED = ["fta_document_id", "subsystem", "event_code", "event_type", "event_description", "parent_event",
                "child_event", "logic_gate", "indicating_parameters", "corrective_action", "reference_document"]
app = FastAPI(title="ARPMS document-service", version="1.0")
add_health(app)


@app.on_event("startup")
def _startup():
    init_schema()


@lru_cache(1)
def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_PATH, device="cpu")


def embed(texts):
    return [("[" + ",".join(f"{x:.6f}" for x in v) + "]") for v in
            encoder().encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)]


# ------------------------------------------------------------------ import
def import_fta():
    paths = [r["path"] for r in q("SELECT path FROM raw_files WHERE file_type='FTA' ORDER BY path")]
    if not paths:
        return {"events": 0, "edges": 0}
    df = pd.concat([pd.read_csv(os.path.join(RAW_DIR, p), dtype=str).fillna("") for p in paths], ignore_index=True)
    missing = [c for c in FTA_REQUIRED if c not in df]
    if missing:
        raise HTTPException(422, f"FTA files missing columns {missing}")
    rel = df[df.parent_event != ""]
    gate = rel.drop_duplicates("parent_event").set_index("parent_event").logic_gate.to_dict()
    ev = df.drop_duplicates("event_code")
    many("""INSERT INTO fta_events VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (event_code) DO UPDATE SET
            description=EXCLUDED.description, gate=EXCLUDED.gate, indicating_parameters=EXCLUDED.indicating_parameters,
            corrective_action=EXCLUDED.corrective_action, reference_document=EXCLUDED.reference_document,
            document_version=EXCLUDED.document_version""",
         [(r.event_code, r.fta_document_id, r.subsystem, r.event_type, r.event_description, gate.get(r.event_code) or None,
           [p for p in r.indicating_parameters.split(";") if p], r.corrective_action or None, r.reference_document or None,
           r.get("page_reference") or None, r.get("document_version") or None) for _, r in ev.iterrows()])
    edges = rel.drop_duplicates(["parent_event", "child_event"])
    many("INSERT INTO fta_edges VALUES (%s,%s,%s,%s) ON CONFLICT (parent_event, child_event) DO UPDATE SET gate=EXCLUDED.gate",
         [(r.parent_event, r.child_event, r.logic_gate, r.fta_document_id) for _, r in edges.iterrows()])
    return {"events": len(ev), "edges": len(edges), **validate_fta()}


def pages(path):
    if path.endswith(".pdf"):
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    parts = re.split(r"\[Page (\d+)\]", open(path, encoding="utf-8").read())
    return [(1, parts[0])] + [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def parse_document(path):
    pg = pages(path)
    meta = dict(re.findall(r"^([A-Z_]+): ?(.*)$", "\n".join(t for _, t in pg[:1]), re.M))
    chunks = []
    for page, text in pg:
        for part in re.split(r"\n(?=SECTION \d+)", text):
            body = part.strip()
            if not body or re.match(r"^DOCUMENT_ID:", body):
                continue
            title = body.splitlines()[0] if body.startswith("SECTION") else "Body"
            chunks.append((title, page, re.sub(r"\s+", " ", body)))
    return meta, chunks


def import_documents():
    n_docs = n_chunks = 0
    for f in q("SELECT file_id, path FROM raw_files WHERE file_type='DOCUMENT' ORDER BY path"):
        meta, chunks = parse_document(os.path.join(RAW_DIR, f["path"]))
        doc_id = meta.get("DOCUMENT_ID") or os.path.splitext(os.path.basename(f["path"]))[0]
        ex("""INSERT INTO documents VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (document_id) DO UPDATE SET
              version=EXCLUDED.version, revision=EXCLUDED.revision, file_id=EXCLUDED.file_id""",
           (doc_id, meta.get("DOCUMENT_TYPE"), meta.get("DOCUMENT_NAME"), meta.get("VERSION"), meta.get("REVISION"),
            meta.get("DATE"), meta.get("SUBSYSTEM") or None, (meta.get("LRU_ID") or "").strip() or None, f["file_id"]))
        ex("DELETE FROM document_chunks WHERE document_id=%s", (doc_id,))
        if chunks:
            many("INSERT INTO document_chunks (document_id, section, page, text, embedding) VALUES (%s,%s,%s,%s,%s::vector)",
                 [(doc_id, s, p, t, e) for (s, p, t), e in zip(chunks, embed(f"{doc_id} {s}: {t}" for s, p, t in chunks))])
        n_docs, n_chunks = n_docs + 1, n_chunks + len(chunks)
    return {"documents": n_docs, "chunks": n_chunks}


def embed_snags():
    rows = q("SELECT snag_id, snag_title, snag_description, reported_symptom FROM snags WHERE embedding IS NULL")
    if rows:
        vecs = embed(" ".join(filter(None, (r["snag_title"], r["snag_description"], r["reported_symptom"]))) for r in rows)
        many("UPDATE snags SET embedding=%s::vector WHERE snag_id=%s", [(v, r["snag_id"]) for v, r in zip(vecs, rows)])
    return {"snags_embedded": len(rows)}


@app.post("/api/v1/documents/import")
def api_import():
    out = {**import_fta(), **import_documents(), **embed_snags()}
    audit("import_knowledge", "documents", RAW_DIR, out)
    return out


# ------------------------------------------------------------------ documents
@app.get("/api/v1/documents")
def api_documents(document_type: str | None = None):
    return q("SELECT d.*, count(c.chunk_id) chunks FROM documents d LEFT JOIN document_chunks c USING (document_id) "
             "WHERE (%(t)s::text IS NULL OR document_type=%(t)s) GROUP BY d.document_id ORDER BY document_id", dict(t=document_type))


@app.get("/api/v1/documents/search")
def api_document_search(query: str, k: int = 8, subsystem: str | None = None, document_type: str | None = None):
    """Retrieval for engineer Q&A: top passages with citations (document, version, section, page)."""
    v = embed([query])[0]
    rows = q("""SELECT c.chunk_id, c.document_id, c.section, c.page, c.text, d.document_type, d.document_name, d.version,
                       d.revision, d.subsystem, 1 - (c.embedding <=> %(v)s::vector) AS score
                FROM document_chunks c JOIN documents d USING (document_id)
                WHERE (%(s)s::text IS NULL OR d.subsystem=%(s)s) AND (%(t)s::text IS NULL OR d.document_type=%(t)s)
                ORDER BY c.embedding <=> %(v)s::vector LIMIT %(k)s""", dict(v=v, s=subsystem, t=document_type, k=k))
    for r in rows:
        r["citation"] = f"{r['document_id']} v{r['version']} rev {r['revision']}, {r['section']}, p.{r['page']}"
    return rows


@app.get("/api/v1/documents/{document_id}")
def api_document(document_id: str):
    d = q1("SELECT * FROM documents WHERE document_id=%s", (document_id,))
    if not d:
        raise HTTPException(404, "document not found")
    d["chunks"] = q("SELECT chunk_id, section, page, text FROM document_chunks WHERE document_id=%s ORDER BY chunk_id", (document_id,))
    return d


# ------------------------------------------------------------------ snags
SNAG_COLS = ("snag_id, aircraft_id, flight_id, snag_date, reported_by, lru_id, subsystem, snag_code, snag_title, snag_description, "
             "reported_symptom, observed_parameter, observed_value, observed_unit, flight_phase, disposition, closure_status, maintenance_reference")


@app.get("/api/v1/snags")
def api_snags(aircraft_id: str | None = None, lru_id: str | None = None, limit: int = 500):
    return q(f"SELECT {SNAG_COLS} FROM snags WHERE (%(a)s::text IS NULL OR aircraft_id=%(a)s) AND (%(l)s::text IS NULL OR lru_id=%(l)s) "
             "ORDER BY snag_date DESC LIMIT %(n)s", dict(a=aircraft_id, l=lru_id, n=limit))


@app.get("/api/v1/snags/search")
def api_snag_search(query: str, k: int = 10, subsystem: str | None = None, aircraft_id: str | None = None):
    v = embed([query])[0]
    return q(f"""SELECT {SNAG_COLS}, 1 - (embedding <=> %(v)s::vector) AS similarity FROM snags
                 WHERE embedding IS NOT NULL AND (%(s)s::text IS NULL OR subsystem=%(s)s) AND (%(a)s::text IS NULL OR aircraft_id=%(a)s)
                 ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s""", dict(v=v, s=subsystem, a=aircraft_id, k=k))


@app.get("/api/v1/snags/repeated")
def api_repeated(days: int = 90, min_similarity: float = 0.6):
    """Semantically similar snags on the same LRU position within `days` (repeat-defect candidates)."""
    return q("""SELECT a.lru_id, a.snag_id AS first_snag, b.snag_id AS repeat_snag, a.snag_date AS first_date, b.snag_date AS repeat_date,
                        a.snag_title AS first_title, b.snag_title AS repeat_title, a.disposition AS first_disposition,
                        1 - (a.embedding <=> b.embedding) AS similarity
                 FROM snags a JOIN snags b ON a.lru_id = b.lru_id AND a.snag_date < b.snag_date
                      AND b.snag_date - a.snag_date < make_interval(days => %(d)s)
                 WHERE 1 - (a.embedding <=> b.embedding) >= %(s)s ORDER BY a.lru_id, b.snag_date""", dict(d=days, s=min_similarity))


@app.get("/api/v1/snags/{snag_id}/similar")
def api_snag_similar(snag_id: str, k: int = 10):
    if not q1("SELECT 1 FROM snags WHERE snag_id=%s AND embedding IS NOT NULL", (snag_id,)):
        raise HTTPException(404, "snag not found or not embedded")
    return q(f"""SELECT {SNAG_COLS}, 1 - (embedding <=> (SELECT embedding FROM snags WHERE snag_id=%(id)s)) AS similarity FROM snags
                 WHERE snag_id <> %(id)s AND embedding IS NOT NULL
                 ORDER BY embedding <=> (SELECT embedding FROM snags WHERE snag_id=%(id)s) LIMIT %(k)s""", dict(id=snag_id, k=k))


# ------------------------------------------------------------------ FTA graph
def validate_fta():
    orphans = [r["event_code"] for r in q("""SELECT event_code FROM fta_events e WHERE event_type <> 'TOP'
                                              AND NOT EXISTS (SELECT 1 FROM fta_edges WHERE child_event=e.event_code)""")]
    dangling = [r["c"] for r in q("SELECT DISTINCT child_event c FROM fta_edges WHERE child_event NOT IN (SELECT event_code FROM fta_events) "
                                  "UNION SELECT DISTINCT parent_event FROM fta_edges WHERE parent_event NOT IN (SELECT event_code FROM fta_events)")]
    return {"orphan_events": orphans, "dangling_references": dangling}


@app.get("/api/v1/fta/trees")
def api_fta_trees():
    return q("""SELECT e.event_code AS top_event, e.subsystem, e.description, e.fta_document_id, e.document_version,
                       (SELECT count(*) FROM fta_events b WHERE b.fta_document_id=e.fta_document_id AND b.event_type='BASIC') AS basic_events
                FROM fta_events e WHERE event_type='TOP' ORDER BY subsystem""")


@app.get("/api/v1/fta/tree/{subsystem}")
def api_fta_tree(subsystem: str):
    nodes = q("SELECT * FROM fta_events WHERE subsystem=%s ORDER BY event_type DESC, event_code", (subsystem,))
    if not nodes:
        raise HTTPException(404, "no FTA for subsystem")
    return {"nodes": nodes, "edges": q("SELECT e.* FROM fta_edges e JOIN fta_events p ON p.event_code=e.parent_event WHERE p.subsystem=%s",
                                       (subsystem,))}


@app.get("/api/v1/fta/events/{event_code}")
def api_fta_event(event_code: str):
    e = q1("SELECT * FROM fta_events WHERE event_code=%s", (event_code,))
    if not e:
        raise HTTPException(404, "event not found")
    e["path_to_top"] = q("""WITH RECURSIVE up(code, depth) AS (SELECT %s::text, 0 UNION ALL
                              SELECT g.parent_event, up.depth + 1 FROM fta_edges g JOIN up ON g.child_event = up.code WHERE depth < 20)
                            SELECT up.depth, ev.event_code, ev.event_type, ev.description, ev.gate FROM up JOIN fta_events ev ON ev.event_code = up.code
                            ORDER BY depth""", (event_code,))
    e["children"] = q("SELECT g.child_event, g.gate, ev.event_type, ev.description FROM fta_edges g JOIN fta_events ev ON ev.event_code=g.child_event "
                      "WHERE g.parent_event=%s", (event_code,))
    return e
