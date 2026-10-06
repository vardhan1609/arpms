"""Synthetic engineering documents (ICD, LRU spec, MM, FIM, procedures, notes, FTA) as TXT + PDF."""
import os

from .maintenance_generator import fim_ref, mm_ref
from .subsystems import CATALOG, LRU_TYPES

BANNER = "SYNTHETIC DOCUMENT - generated for software development. Not a real aircraft specification."


def _doc(doc_id, dtype, name, subsystem, lru, sections, version="1.0", revision="A", date="2025-06-01"):
    head = [f"DOCUMENT_ID: {doc_id}", f"DOCUMENT_TYPE: {dtype}", f"DOCUMENT_NAME: {name}", f"VERSION: {version}",
            f"REVISION: {revision}", f"DATE: {date}", f"SUBSYSTEM: {subsystem}", f"LRU_ID: {lru}", BANNER, ""]
    body, page = [], 1
    for i, (title, text) in enumerate(sections, 1):
        if i > 1 and i % 3 == 1:
            page += 1
        body += [f"[Page {page}]", f"SECTION {i}: {title}", text, ""]
    return doc_id, "\n".join(head + body)


def build_documents(mechs, mapping, fta_rows, snag_summary):
    docs = []
    for lru, (name, sub, rt) in LRU_TYPES.items():
        params = [p for p in CATALOG if p.lru == lru]
        lm = [m for m in mechs.values() if m.lru == lru]
        table = "\n".join(f"{p.name} | {p.unit} | {p.lo}..{p.hi} | {p.rate} | {p.src}" for p in params)
        docs.append(_doc(f"SPEC-{lru}", "LRU_SPECIFICATION", f"{name} specification", sub, lru, [
            ("Scope", f"Synthetic specification of the {name} ({lru}), remote terminal {rt}."),
            ("Parameters", "PARAMETER | UNIT | SYNTHETIC RANGE | RATE | SOURCE\n" + table),
            ("Operating context", f"{name} outputs depend on engine state, flight phase and ambient conditions."),
        ]))
        docs.append(_doc(mm_ref(lru), "MAINTENANCE_MANUAL", f"{name} maintenance manual", sub, lru,
                         [("General", f"Removal / installation and servicing of the {name}.")] +
                         [(f"Task {fim_ref(m.code)} - {m.description}", f"Corrective action: {m.action}. Indicating parameters: "
                           + ", ".join(dict.fromkeys(e[0] for e in m.effects)) + ".") for m in lm]))
        docs.append(_doc(f"FID-{lru}", "FAULT_ISOLATION", f"{name} fault isolation", sub, lru,
                         [(f"{fim_ref(m.code)} {m.description}", f"Symptoms: {'; '.join(m.symptoms)}. Isolate by checking "
                           + ", ".join(dict.fromkeys(e[0] for e in m.effects)) + f" against reference trends. Then: {m.action}.")
                          for m in lm]))
        docs.append(_doc(f"TP-{lru}", "TEST_PROCEDURE", f"{name} test procedure", sub, lru, [
            ("Ground test", f"Run engine at idle and 80% RPM, record {', '.join(p.name for p in params[:4])}."),
            ("Acceptance", "Values shall remain within the synthetic reference band of the LRU specification."),
        ]))
        docs.append(_doc(f"IP-{lru}", "INSPECTION_PROCEDURE", f"{name} inspection procedure", sub, lru, [
            ("Visual inspection", f"Inspect {name} connectors, mounts and harness for damage, corrosion and chafing."),
            ("Periodic", "Every 100 synthetic flight hours or on snag."),
        ]))
    for sub in sorted({s for _, s, _ in LRU_TYPES.values()}):
        docs.append(_doc(f"EN-{sub}", "ENGINEERING_NOTE", f"{sub} engineering note", sub, "", [
            ("Trend monitoring", f"Persistent shifts of {sub.lower()} parameters across flights indicate degradation onset."),
            ("Context", "High-g manoeuvres legitimately raise vibration, temperatures and hydraulic demand."),
        ]))
        docs.append(_doc(f"OP-{sub}", "OPERATING_PROCEDURE", f"{sub} operating procedure", sub, "", [
            ("Normal operation", f"Monitor {sub.lower()} cautions; record snag with flight phase and indication values."),
        ]))
        rows = [r for r in fta_rows if r["subsystem"] == sub]
        docs.append(_doc(f"FTA-{sub}", "FTA", f"{sub} fault tree analysis", sub, "", [
            ("Tree", "\n".join(f"{r['parent_event'] or 'ROOT'} -[{r['logic_gate'] or '-'}]-> {r['event_code']}: {r['event_description']}"
                               for r in rows)),
        ]))
        if sub in snag_summary:
            docs.append(_doc(f"SR-{sub}", "HISTORICAL_SNAG_REPORT", f"{sub} historical snag summary", sub, "",
                             [("Summary", snag_summary[sub])]))
    docs.append(_doc("ICD-1553-SYN", "ICD", "Synthetic 1553 bus ICD", "ALL", "", [
        ("Scope", "Synthetic message / word / bit mapping. Replace with the real ICD through configuration."),
        ("Mapping", "\n".join(f"{m['parameter_id']} | {m['message_id']} | RT{m['remote_terminal_address']} SA{m['sub_address']} "
                              f"W{m['word_position']} B{m['bit_position']}+{m['bit_length']} | x{m['scaling_factor']:.6g} + {m['offset']}"
                              for m in mapping)),
    ]))
    return docs


def write_documents(docs, out_dir, pdf_types=("LRU_SPECIFICATION", "MAINTENANCE_MANUAL", "FAULT_ISOLATION", "FTA", "ICD")):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    os.makedirs(out_dir, exist_ok=True)
    files = []
    for doc_id, text in docs:
        dtype = text.split("\n")[1].split(": ")[1]
        if dtype in pdf_types:
            path = os.path.join(out_dir, f"{doc_id}.pdf")
            c = canvas.Canvas(path, pagesize=A4, invariant=1)
            y = 800
            for line in text.split("\n"):
                if line.startswith("[Page ") and y < 800:
                    c.showPage()
                    y = 800
                for chunk in [line[i:i + 105] for i in range(0, max(len(line), 1), 105)]:
                    if y < 40:
                        c.showPage()
                        y = 800
                    c.setFont("Helvetica", 8)
                    c.drawString(30, y, chunk)
                    y -= 11
            c.save()
        else:
            path = os.path.join(out_dir, f"{doc_id}.txt")
            with open(path, "w") as fh:
                fh.write(text)
        files.append(path)
    return files
