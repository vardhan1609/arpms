"""diagnosis-service: FTA-constrained Bayesian diagnosis, deterministic recommendations, engineer decisions, PDF reports, audit."""
import io
import re
from collections import defaultdict
from datetime import timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from arpms_common import Jsonb, add_health, audit, ex, init_schema, many, q, q1

app = FastAPI(title="ARPMS diagnosis-service", version="1.0")
add_health(app)
SNAG_LOOKBACK = timedelta(days=30)


@app.on_event("startup")
def _startup():
    init_schema()


def lru_type_of(code, types):
    return next((t for t in re.split(r"[-_]", code) if t in types), None)


def candidates_for(lru_type):
    """Basic events in the FTA subtree of the LRU's intermediate event (IE-<type>), plus basic events naming the LRU."""
    return q("""WITH RECURSIVE down(code) AS (SELECT %(ie)s::text UNION SELECT g.child_event FROM fta_edges g JOIN down ON g.parent_event=down.code)
                SELECT e.* FROM fta_events e WHERE e.event_type='BASIC' AND (e.event_code IN (SELECT code FROM down)
                      OR e.event_code ~ ('(^|-)' || %(t)s || '(-|$)'))""", dict(ie=f"IE-{lru_type}", t=lru_type))


def gate_factor(code, p_fail_by_type, types):
    """Consistency with the tree logic, best over all paths to a top event:
    AND ancestor -> sibling branches must also be active; NOT ancestor -> this event makes that branch false."""
    edges = q("SELECT parent_event, child_event FROM fta_edges")
    gate = {r["event_code"]: r["gate"] for r in q("SELECT event_code, gate FROM fta_events")}
    parents, children = defaultdict(list), defaultdict(list)
    for e in edges:
        parents[e["child_event"]].append(e["parent_event"])
        children[e["parent_event"]].append(e["child_event"])

    def leaves(c):
        return [c] if not children[c] else [x for k in children[c] for x in leaves(k)]

    def support(c):
        p = max([p_fail_by_type.get(lru_type_of(x, types), 0.0) for x in leaves(c)] or [0.0])
        return 1 - p if gate.get(c) == "NOT" else p

    best = [0.0, []]

    def walk(c, f, notes, depth=0):
        if not parents[c] or depth > 20:
            if f > best[0]:
                best[:] = [f, notes]
            return
        for p in parents[c]:
            f2, n2 = f, list(notes)
            if gate.get(p) == "AND":
                for s in children[p]:
                    if s != c:
                        sp = support(s)
                        f2 *= max(sp, 0.05)
                        n2.append(f"AND gate {p}: sibling {s} support {sp:.2f}")
            elif gate.get(p) == "NOT":
                f2 *= 0.05
                n2.append(f"NOT gate {p}: event negates this branch")
            walk(p, f2, n2, depth + 1)

    walk(code, 1.0, [])
    return best[0], best[1]


def diagnose(flight_id, lru_id):
    pred = q1("""SELECT p.*, f.aircraft_id, f.departure_time, l.lru_type, l.subsystem FROM predictions p
                 JOIN flights f USING (flight_id) JOIN lru l USING (lru_id) WHERE p.flight_id=%s AND p.lru_id=%s""", (flight_id, lru_id))
    if not pred:
        raise HTTPException(404, "no prediction for this flight/LRU - run model scoring first")
    types = {r["lru_type"] for r in q("SELECT DISTINCT lru_type FROM lru")}
    cands = candidates_for(pred["lru_type"])
    if not cands:
        raise HTTPException(422, f"no FTA basic events for {pred['lru_type']}")
    p_fail_by_type = {r["lru_type"]: r["failure_probability"] for r in q(
        "SELECT l.lru_type, p.failure_probability FROM predictions p JOIN lru l USING (lru_id) WHERE p.flight_id=%s", (flight_id,))}
    history = {r["event_code"]: r["n"] for r in q("""SELECT e.event_code, count(*) n FROM maintenance_records m
                 JOIN fta_events e ON e.reference_document=m.maintenance_reference WHERE m.finding LIKE 'CONFIRMED%%' GROUP BY 1""")}
    resid = {}
    for r in (pred["shap"] or {}).get("top_residuals", []):
        resid[r["feature"].split("__")[0]] = resid.get(r["feature"].split("__")[0], 0) + abs(r["z"])
    total_z = sum(resid.values()) or 1.0
    modes = pred["mode_probabilities"] or {}
    snags = q("""SELECT snag_id, snag_title, disposition FROM snags WHERE lru_id=%s AND snag_date <= %s AND snag_date > %s""",
              (lru_id, pred["departure_time"], pred["departure_time"] - SNAG_LOOKBACK))
    out = []
    for c in cands:
        prior = (history.get(c["event_code"], 0) + 1) * (0.5 if re.search(r"-S\d+$", c["event_code"]) else 1.0)
        l_model = 0.05 + modes.get(c["event_code"], 0.0)
        overlap = sum(z for p, z in resid.items() if p in (c["indicating_parameters"] or [])) / total_z
        l_params = 0.2 + overlap
        sim = q1("""SELECT max(1 - (s.embedding <=> ch.embedding)) AS s FROM snags s, document_chunks ch
                    WHERE s.lru_id=%s AND s.snag_date <= %s AND s.snag_date > %s AND s.embedding IS NOT NULL
                      AND ch.text LIKE '%%' || %s || '%%'""",
                 (lru_id, pred["departure_time"], pred["departure_time"] - SNAG_LOOKBACK, c["reference_document"] or "~none~"))["s"]
        l_snag = 0.5 + (sim or 0.0)
        g, notes = gate_factor(c["event_code"], p_fail_by_type, types)
        out.append(dict(event_code=c["event_code"], description=c["description"], corrective_action=c["corrective_action"],
                        reference_document=c["reference_document"], score=prior * l_model * l_params * l_snag * g,
                        evidence=dict(prior_weight=prior, model_mode_probability=round(modes.get(c["event_code"], 0.0), 4),
                                      indicating_parameter_overlap=round(overlap, 3), snag_document_similarity=None if sim is None else round(sim, 3),
                                      gate_factor=round(g, 3), gate_notes=notes)))
    z = sum(c["score"] for c in out) or 1.0
    p_fault = float(max(pred["failure_probability"], min(1.0, (pred["anomaly_score"] or 0) / 3)))
    for c in out:
        c["posterior_given_fault"] = round(c["score"] / z, 4)
        c["probability"] = round(p_fault * c.pop("score") / z, 4)
    out.sort(key=lambda c: -c["probability"])
    evidence = dict(failure_probability=pred["failure_probability"], failure_mode=pred["failure_mode"], anomaly_score=pred["anomaly_score"],
                    changepoint_probability=pred["changepoint_probability"], rul_hours=pred["rul_hours"],
                    rul_interval=[pred["rul_lower"], pred["rul_upper"]], top_residuals=(pred["shap"] or {}).get("top_residuals", [])[:5],
                    shap=(pred["shap"] or {}).get("shap", [])[:5], recent_snags=snags, p_fault_present=round(p_fault, 4))
    row = q1("""INSERT INTO diagnoses (aircraft_id, flight_id, lru_id, top_event, candidates, evidence, model_versions)
                VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (flight_id, lru_id) DO UPDATE SET candidates=EXCLUDED.candidates,
                evidence=EXCLUDED.evidence, model_versions=EXCLUDED.model_versions, created_at=now() RETURNING diagnosis_id""",
             (pred["aircraft_id"], flight_id, lru_id, f"TE-{pred['subsystem']}", Jsonb(out[:10]), Jsonb(evidence, dumps=_dumps),
              Jsonb(pred["model_versions"])))
    did = row["diagnosis_id"]
    ex("DELETE FROM recommendations WHERE diagnosis_id=%s AND status='PENDING'", (did,))
    many("INSERT INTO recommendations (diagnosis_id, rank, event_code, action, rationale, refs, confidence) VALUES (%s,%s,%s,%s,%s,%s,%s)",
         [(did, i + 1, *r) for i, r in enumerate(recommend(pred, out, snags))])
    audit("diagnose", "diagnosis", did, {"flight_id": flight_id, "lru_id": lru_id, "top": out[0]["event_code"]})
    return get_diagnosis(did)


def _dumps(o):
    import json
    return json.dumps(o, default=str)


def recommend(pred, cands, snags):
    """Deterministic, rule-based recommendations; every action cites the FTA / FIM / manual references it comes from."""
    docs = {r["document_type"]: r["document_id"] for r in q("SELECT document_type, document_id FROM documents WHERE lru_type=%s", (pred["lru_type"],))}
    recs = []
    nff = [s for s in snags if "NO FAULT" in (s["disposition"] or "").upper()]
    if len(snags) >= 2 and nff:
        recs.append((cands[0]["event_code"], "Repeat defect: do not close as NO FAULT FOUND again - perform full fault isolation and consider unit removal",
                     f"{len(snags)} snags on {pred['lru_id']} within {SNAG_LOOKBACK.days} days, {len(nff)} closed NFF",
                     Jsonb([s["snag_id"] for s in snags] + [cands[0]["reference_document"]]), cands[0]["posterior_given_fault"]))
    for c in cands[:3]:
        if c["probability"] < 0.05 and recs:
            break
        sensor = bool(re.search(r"-S\d+$", c["event_code"]))
        action = c["corrective_action"] or "Perform fault isolation"
        if sensor:
            action = f"Verify sensor before LRU removal: {action}"
        elif pred["rul_upper"] is not None and pred["rul_upper"] < 1.0:
            action = f"Schedule before next flight: {action}"
        ev = c["evidence"]
        rationale = (f"{c['description']}: P={c['probability']:.2f} (model mode p={ev['model_mode_probability']:.2f}, "
                     f"indicating-parameter overlap {ev['indicating_parameter_overlap']:.2f}, gate factor {ev['gate_factor']:.2f})")
        refs = [r for r in (c["reference_document"], docs.get("FAULT_ISOLATION"), docs.get("MAINTENANCE_MANUAL"), docs.get("TEST_PROCEDURE")) if r]
        recs.append((c["event_code"], action, rationale, Jsonb(refs), c["probability"]))
    if not recs or cands[0]["probability"] < 0.2:
        recs.append((None, f"Continue monitoring {pred['lru_id']}; re-assess after next flight", "Low fault probability", Jsonb([]),
                     1 - (cands[0]["probability"] if cands else 0)))
    return recs


def get_diagnosis(did):
    d = q1("SELECT d.*, f.departure_time FROM diagnoses d JOIN flights f USING (flight_id) WHERE diagnosis_id=%s", (did,))
    if not d:
        raise HTTPException(404, "diagnosis not found")
    d["recommendations"] = q("SELECT * FROM recommendations WHERE diagnosis_id=%s ORDER BY rank", (did,))
    return d


# ------------------------------------------------------------------ API
@app.post("/api/v1/diagnoses/run/{aircraft_id}")
def api_run(aircraft_id: str, min_probability: float = 0.3):
    """Diagnose every LRU-flight with an alert or failure probability >= threshold."""
    targets = q("""SELECT DISTINCT p.flight_id, p.lru_id FROM predictions p JOIN flights f USING (flight_id)
                   LEFT JOIN alerts a ON a.flight_id=p.flight_id AND a.lru_id=p.lru_id
                   WHERE f.aircraft_id=%s AND (p.failure_probability >= %s OR a.alert_id IS NOT NULL)""", (aircraft_id, min_probability))
    return [{"flight_id": t["flight_id"], "lru_id": t["lru_id"], "diagnosis_id": diagnose(t["flight_id"], t["lru_id"])["diagnosis_id"]}
            for t in targets]


@app.post("/api/v1/diagnoses/{flight_id}/{lru_id}")
def api_diagnose(flight_id: str, lru_id: str):
    return diagnose(flight_id, lru_id)


@app.get("/api/v1/diagnoses")
def api_diagnoses(aircraft_id: str | None = None, limit: int = 200):
    return q("""SELECT d.diagnosis_id, d.aircraft_id, d.flight_id, d.lru_id, d.top_event, d.created_at, d.candidates->0 AS top_candidate,
                       (SELECT count(*) FROM recommendations r WHERE r.diagnosis_id=d.diagnosis_id AND r.status='PENDING') AS pending
                FROM diagnoses d WHERE (%(a)s::text IS NULL OR aircraft_id=%(a)s) ORDER BY created_at DESC LIMIT %(n)s""",
             dict(a=aircraft_id, n=limit))


@app.get("/api/v1/diagnoses/{diagnosis_id}")
def api_diagnosis(diagnosis_id: int):
    return get_diagnosis(diagnosis_id)


class Decision(BaseModel):
    decision: str
    engineer: str
    note: str | None = None
    override_action: str | None = None


@app.post("/api/v1/recommendations/{recommendation_id}/decision")
def api_decide(recommendation_id: int, body: Decision):
    """Human-in-the-loop: nothing is actioned without an engineer decision; reject/override need a reason."""
    status = {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED", "OVERRIDE": "OVERRIDDEN"}.get(body.decision.upper())
    if not status or not body.engineer.strip():
        raise HTTPException(400, "decision must be ACCEPT, REJECT or OVERRIDE and engineer is required")
    if status != "ACCEPTED" and not (body.note or "").strip():
        raise HTTPException(400, "a note is required to reject or override")
    if status == "OVERRIDDEN" and not (body.override_action or "").strip():
        raise HTTPException(400, "override_action is required")
    r = q1("""UPDATE recommendations SET status=%s, engineer=%s, decision_note=%s, override_action=%s, decided_at=now()
              WHERE recommendation_id=%s AND status='PENDING' RETURNING *""",
           (status, body.engineer, body.note, body.override_action, recommendation_id))
    if not r:
        raise HTTPException(409, "recommendation not found or already decided")
    audit(f"recommendation_{status.lower()}", "recommendation", recommendation_id,
          {"note": body.note, "override_action": body.override_action}, actor=body.engineer)
    return r


@app.get("/api/v1/audit")
def api_audit(entity: str | None = None, entity_id: str | None = None, limit: int = 300):
    return q("SELECT * FROM audit_log WHERE (%(e)s::text IS NULL OR entity=%(e)s) AND (%(i)s::text IS NULL OR entity_id=%(i)s) "
             "ORDER BY id DESC LIMIT %(n)s", dict(e=entity, i=entity_id, n=limit))


@app.get("/api/v1/reports/{diagnosis_id}.pdf")
def api_report(diagnosis_id: int):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    d = get_diagnosis(diagnosis_id)
    ev = d["evidence"]
    st = getSampleStyleSheet()
    P = lambda t, s="BodyText": Paragraph(str(t).replace("&", "&amp;").replace("<", "&lt;"), st[s])
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8)])
    fmt = lambda v: "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    story = [P(f"Maintenance diagnosis report #{diagnosis_id}", "Title"),
             P("SYNTHETIC DATA - decision support only. Engineer approval is required before any maintenance action."),
             Spacer(1, 8),
             Table([["Aircraft", d["aircraft_id"], "LRU", d["lru_id"]], ["Flight", d["flight_id"], "Date", str(d["departure_time"])[:19]],
                    ["Failure probability", fmt(ev["failure_probability"]), "Predicted mode", ev["failure_mode"]],
                    ["RUL (h)", fmt(ev["rul_hours"]), "RUL 80% interval", " - ".join(fmt(v) for v in ev["rul_interval"])],
                    ["Anomaly score", fmt(ev["anomaly_score"]), "Changepoint prob.", fmt(ev["changepoint_probability"])]], style=grid),
             Spacer(1, 8), P("Fault candidates (FTA-constrained)", "Heading2"),
             Table([["Event", "Description", "P", "Evidence"]] +
                   [[c["event_code"], P(c["description"]), f"{c['probability']:.2f}",
                     P(f"mode p={c['evidence']['model_mode_probability']}, params={c['evidence']['indicating_parameter_overlap']}, "
                       f"snag sim={c['evidence']['snag_document_similarity']}, gate={c['evidence']['gate_factor']}")]
                    for c in d["candidates"][:6]], colWidths=[90, 170, 35, 200], style=grid),
             Spacer(1, 8), P("Recommendations", "Heading2"),
             Table([["#", "Action", "Refs", "Status / engineer"]] +
                   [[r["rank"], P(r["override_action"] or r["action"]), P(", ".join(r["refs"] or [])),
                     P(f"{r['status']} {r['engineer'] or ''} {r['decision_note'] or ''}")] for r in d["recommendations"]],
                   colWidths=[20, 230, 120, 125], style=grid),
             Spacer(1, 8), P("Top contributing signals", "Heading2"),
             P("; ".join(f"{t['feature']} z={t['z']}" for t in ev.get("top_residuals", []))),
             P("SHAP: " + "; ".join(f"{s['feature']} {s['contribution']:+.3f}" for s in ev.get("shap", []))),
             P(f"Model versions: {d['model_versions']}")]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=f"ARPMS diagnosis {diagnosis_id}").build(story)
    buf.seek(0)
    audit("report_generated", "diagnosis", diagnosis_id)
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename=diagnosis-{diagnosis_id}.pdf"})
