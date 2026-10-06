"""Synthetic FTA source documents (one per subsystem) as edge rows."""
from .failure_injection import SENSOR_KINDS
from .maintenance_generator import fim_ref
from .subsystems import CATALOG, LRU_TYPES


def build_fta(mechs):
    rows = []
    subsystems = sorted({s for _, s, _ in LRU_TYPES.values()})
    for sub in subsystems:
        doc = f"FTA-{sub}"
        top = f"TE-{sub}"
        meta = dict(fta_document_id=doc, fta_name=f"{sub.title()} function FTA", subsystem=sub, document_version="1.0",
                    revision="A", effective_date="2025-06-01", document_title=f"SYNTHETIC Fault Tree Analysis - {sub}",
                    source_file=f"documents/{doc}.pdf")

        def edge(parent, gate, code, etype, desc, inter="", basic="", params="", action="", ref="", page=1):
            rows.append(dict(meta, section=f"{3 + len(rows) % 5}", page_reference=str(page), top_event=top,
                             intermediate_event=inter, basic_event=basic, event_code=code, event_type=etype,
                             event_description=desc, parent_event=parent, child_event=code, logic_gate=gate,
                             indicating_parameters=params, corrective_action=action,
                             engineering_note="Synthetic tree for software development only",
                             reference_document=ref))

        edge("", "", top, "TOP", f"Loss or degradation of {sub.lower().replace('_', ' ')} function")
        for lru, (name, s, _) in LRU_TYPES.items():
            if s != sub:
                continue
            inter = f"IE-{lru}"
            edge(top, "OR", inter, "INTERMEDIATE", f"{name} malfunction", inter=inter, page=2)
            lru_params = ";".join(p.name for p in CATALOG if p.lru == lru)
            for m in mechs.values():
                if m.lru != lru:
                    continue
                params = lru_params if m.sensor else ";".join(dict.fromkeys(e[0] for e in m.effects))
                edge(inter, "OR", m.code, "BASIC", m.description, inter=inter, basic=m.code, params=params,
                     action=m.action if not m.sensor else f"{SENSOR_KINDS[m.name][2].format(p=name + ' affected')}",
                     ref=fim_ref(m.code), page=3)
        if sub == "HYDRAULIC":  # AND gate: both pumps degraded -> total hydraulic power loss
            edge(top, "OR", "IE-HYD-DUAL", "INTERMEDIATE", "Total hydraulic power loss (both pumps)", inter="IE-HYD-DUAL", page=4)
            for c in ("BE-HYD-HPA-01", "BE-HYD-HPB-01"):
                rows.append(dict(rows[-1], parent_event="IE-HYD-DUAL", child_event=c, event_code=c, event_type="BASIC",
                                 logic_gate="AND", event_description=mechs[c].description, basic_event=c,
                                 indicating_parameters=";".join(dict.fromkeys(e[0] for e in mechs[c].effects)),
                                 corrective_action=mechs[c].action, reference_document=fim_ref(c)))
        if sub == "ELECTRICAL":  # NOT gate: bus instability with healthy battery
            edge(top, "OR", "IE-ELE-BUS", "INTERMEDIATE", "DC bus instability with battery healthy", inter="IE-ELE-BUS", page=4)
            rows.append(dict(rows[-1], parent_event="IE-ELE-BUS", child_event="BE-ELE-PDU-01", event_code="BE-ELE-PDU-01",
                             event_type="BASIC", logic_gate="AND", event_description=mechs["BE-ELE-PDU-01"].description,
                             basic_event="BE-ELE-PDU-01", reference_document=fim_ref("BE-ELE-PDU-01")))
            edge("IE-ELE-BUS", "AND", "NG-ELE-BAT", "INTERMEDIATE", "Battery NOT degraded", inter="NG-ELE-BUS", page=4)
            rows.append(dict(rows[-1], parent_event="NG-ELE-BAT", child_event="BE-ELE-BAT-01", event_code="BE-ELE-BAT-01",
                             event_type="BASIC", logic_gate="NOT", event_description=mechs["BE-ELE-BAT-01"].description,
                             basic_event="BE-ELE-BAT-01", reference_document=fim_ref("BE-ELE-BAT-01")))
    return rows
