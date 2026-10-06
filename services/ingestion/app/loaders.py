"""Raw-file catalog and structured reference-data loading (fleet, ICD, flights, snags, maintenance)."""
import hashlib
import os

import pandas as pd

from arpms_common import audit, many, q

FILE_TYPES = [("bus1553/", "BUS_1553"), ("lru_logs/", "LRU_LOG"), ("fleet/aircraft", "AIRCRAFT"),
              ("fleet/lru_installations", "LRU_INSTALLATIONS"), ("flights/", "FLIGHT_LOG"), ("snags/", "SNAGS"),
              ("maintenance/", "MAINTENANCE"), ("icd/bus1553_mapping", "BUS_MAPPING"),
              ("icd/parameter_dictionary", "PARAMETER_DICTIONARY"), ("fta/", "FTA"), ("documents/", "DOCUMENT")]
REQUIRED = {
    "AIRCRAFT": ["aircraft_id"], "LRU_INSTALLATIONS": ["lru_id", "aircraft_id", "lru_type", "serial_number"],
    "FLIGHT_LOG": ["flight_id", "aircraft_id", "departure_time", "arrival_time"],
    "SNAGS": ["snag_id", "aircraft_id", "snag_description"], "MAINTENANCE": ["maintenance_record_id", "aircraft_id", "lru_id"],
    "BUS_MAPPING": ["parameter_id", "remote_terminal_address", "sub_address", "word_position", "bit_position", "bit_length",
                    "scaling_factor", "offset"],
    "PARAMETER_DICTIONARY": ["parameter_id", "lru_type", "unit", "min_value", "max_value"],
}


def file_type(rel):
    return next((t for prefix, t in FILE_TYPES if rel.startswith(prefix)), None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flight_of(rel, ftype):
    stem = os.path.splitext(os.path.basename(rel))[0]
    if ftype == "BUS_1553":
        return stem
    if ftype == "LRU_LOG":
        return stem.rsplit("_", 1)[0]
    return None


def catalog(raw_dir):
    """Register every raw file (immutable; never modified). Changed content for a known path is flagged."""
    known = {r["path"]: r["sha256"] for r in q("SELECT path, sha256 FROM raw_files")}
    new, changed = [], []
    for dp, _, files in os.walk(raw_dir):
        for name in sorted(files):
            full = os.path.join(dp, name)
            rel = os.path.relpath(full, raw_dir)
            ftype = file_type(rel)
            if not ftype:
                continue
            digest = sha256(full)
            if rel in known:
                if known[rel] != digest:
                    changed.append(rel)
                continue
            fl = flight_of(rel, ftype)
            new.append((rel, ftype, digest, os.path.getsize(full), fl.rsplit("-F", 1)[0] if fl else None, fl))
    many("INSERT INTO raw_files (path, file_type, sha256, bytes, aircraft_id, flight_id) VALUES (%s,%s,%s,%s,%s,%s) "
         "ON CONFLICT (path) DO NOTHING", new)
    if changed:
        many("UPDATE raw_files SET status='CONTENT_CHANGED', error='raw file modified after cataloging' WHERE path=%s",
             [(c,) for c in changed])
    audit("catalog", "raw_files", raw_dir, {"new": len(new), "changed": len(changed)})
    return {"new_files": len(new), "changed_files": changed}


def read_csv(raw_dir, ftype):
    rows = q("SELECT path FROM raw_files WHERE file_type=%s ORDER BY path", (ftype,))
    if not rows:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(os.path.join(raw_dir, r["path"]), keep_default_na=True) for r in rows], ignore_index=True)
    missing = [c for c in REQUIRED.get(ftype, []) if c not in df]
    if missing:
        raise ValueError(f"{ftype}: missing required columns {missing}")
    return df


def upsert(table, df, pk, cols=None):
    if df.empty:
        return 0
    cols = cols or list(df.columns)
    keys = pk.split(",")
    df = df[cols].astype(object).where(pd.notna(df[cols]), None)
    upd = ",".join(f'"{c}"=EXCLUDED."{c}"' for c in cols if c not in keys) or None
    sql = (f'INSERT INTO {table} ({",".join(chr(34) + c + chr(34) for c in cols)}) VALUES ({",".join(["%s"] * len(cols))}) '
           f'ON CONFLICT ({pk}) ' + (f"DO UPDATE SET {upd}" if upd else "DO NOTHING"))
    many(sql, df.values.tolist())
    return len(df)


def load_reference(raw_dir):
    counts = {}
    params = read_csv(raw_dir, "PARAMETER_DICTIONARY")
    counts["parameters"] = upsert("parameters", params, "parameter_id")
    counts["bus_mapping"] = upsert("bus_mapping", read_csv(raw_dir, "BUS_MAPPING"), "parameter_id")
    counts["aircraft"] = upsert("aircraft", read_csv(raw_dir, "AIRCRAFT"), "aircraft_id")
    inst = read_csv(raw_dir, "LRU_INSTALLATIONS").sort_values("install_date", kind="stable")
    current = inst.groupby("lru_id").tail(1)
    counts["lru"] = upsert("lru", current, "lru_id", ["lru_id", "aircraft_id", "lru_type", "lru_name", "subsystem",
                                                       "remote_terminal_address", "part_number", "serial_number", "install_date"])
    upsert("lru_installations", inst, "lru_id,serial_number", ["lru_id", "serial_number", "part_number", "install_date"])
    fl = read_csv(raw_dir, "FLIGHT_LOG").sort_values(["aircraft_id", "departure_time"])
    fl["cumulative_hours"] = fl.groupby("aircraft_id").flight_hours.cumsum()
    counts["flights"] = upsert("flights", fl, "flight_id", ["flight_id", "aircraft_id", "flight_number", "departure_time",
                                                            "arrival_time", "mission_type", "base", "flight_hours", "cumulative_hours"])
    sn = read_csv(raw_dir, "SNAGS")
    if not sn.empty:
        sn = sn.astype({"observed_value": str}).replace({"nan": None})
        counts["snags"] = upsert("snags", sn, "snag_id")
    counts["maintenance_records"] = upsert("maintenance_records", read_csv(raw_dir, "MAINTENANCE"), "maintenance_record_id")
    audit("load_reference", "dataset", raw_dir, counts)
    return counts
