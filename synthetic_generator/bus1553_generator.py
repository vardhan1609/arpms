"""Synthetic MIL-STD-1553-like bus capture: mapping (synthetic ICD) + binary/hex files."""
import struct

import numpy as np

from .flight_profiles import DT
from .subsystems import CATALOG, LRU_TYPES, RATES

MAGIC, VERSION = b"SYN1553\x00", 1
HEADER = struct.Struct("<8sH16s24sIq")  # magic, version, aircraft_id, flight_id, record_count, start_us
HEADER_SIZE = 64
RECORD = np.dtype([
    ("message_id", "<u8"), ("timestamp_us", "<i8"), ("bus_id", "u1"), ("rt", "u1"), ("sa", "u1"),
    ("direction", "u1"), ("command_word", "<u2"), ("status_word", "<u2"), ("word_count", "u1"),
    ("message_validity", "u1"), ("error_status", "<u2"), ("message_sequence", "<u4"), ("parity", "<u4"),
    ("data", "<u2", (32,)),
])
PACKED = {"CHANNEL_STATUS": 0, "ANTENNA_STATUS": 2}  # 2-bit discretes sharing one word


def build_mapping():
    """Synthetic parameter -> (RT, SA, word, bit) mapping. Replaceable by a real ICD via config."""
    rows, sa_next = [], {}
    for lru, (_, _, rt) in LRU_TYPES.items():
        for rate in RATES:
            params = [p for p in CATALOG if p.lru == lru and p.rate == rate and p.src == "1553"]
            word = 0
            for i in range(0, len(params), 32):
                sa = sa_next[rt] = sa_next.get(rt, 0) + 1
                word = 0
                for p in params[i:i + 32]:
                    if p.name in PACKED:
                        bitpos, bitlen = PACKED[p.name], 2
                        if bitpos == 0:
                            word += 1
                    else:
                        word += 1
                        bitpos, bitlen = 0, 16
                    span = (2 ** bitlen - 1)
                    scale = 1.0 if p.kind != "analog" else (p.hi - p.lo) / span
                    offset = 0.0 if p.kind != "analog" else p.lo
                    rows.append(dict(parameter_id=p.name, message_id=f"MSG-RT{rt:02d}-SA{sa:02d}", remote_terminal_address=rt,
                                     sub_address=sa, word_position=word, bit_position=bitpos, bit_length=bitlen,
                                     scaling_factor=scale, offset=offset, unit=p.unit, rate_class=rate, rate_hz=RATES[rate],
                                     lru_type=lru))
    return rows


def odd_parity(words):
    bits = np.unpackbits(words.view(np.uint8).reshape(*words.shape, 2), axis=-1).sum(-1)
    return (bits % 2 == 0).astype(np.uint32)  # parity bit set when popcount is even -> odd total


def encode_flight(rng, mapping, measured, start_us, duration, msg_id_base, imperfections=True):
    by_msg = {}
    for m in mapping:
        by_msg.setdefault(m["message_id"], []).append(m)
    n_grid = len(next(iter(measured.values())))
    chunks = []
    for mid, rows in by_msg.items():
        rt, sa, rate = rows[0]["remote_terminal_address"], rows[0]["sub_address"], rows[0]["rate_hz"]
        times = np.arange(rng.uniform(0, 1 / rate), duration, 1 / rate)
        if rows[0]["rate_class"] == "EVENT":
            vals = np.stack([measured[r["parameter_id"]] for r in rows])
            change = np.where(np.any(np.diff(vals, axis=1) != 0, axis=0))[0] * DT + DT
            times = np.unique(np.r_[times, change])
        times = times + (rng.normal(0, 0.002, len(times)) if imperfections else 0)
        times = times[(times >= 0) & (times < duration)]
        idx = np.clip(np.round(times / DT).astype(int), 0, n_grid - 1)
        rec = np.zeros(len(times), RECORD)
        rec["timestamp_us"] = start_us + np.round(times * 1e6).astype(np.int64)
        rec["rt"], rec["sa"], rec["direction"] = rt, sa, 1
        wc = max(r["word_position"] for r in rows)
        rec["word_count"] = wc
        rec["command_word"] = (rt << 11) | (1 << 10) | (sa << 5) | (wc & 31)
        rec["status_word"] = rt << 11
        rec["message_validity"] = 1
        for r in rows:
            raw = np.round((measured[r["parameter_id"]][idx] - r["offset"]) / r["scaling_factor"])
            raw = np.clip(raw, 0, 2 ** r["bit_length"] - 1).astype(np.uint16)
            rec["data"][:, r["word_position"] - 1] |= (raw << r["bit_position"]).astype(np.uint16)
        rec["bus_id"] = rng.choice([0, 1], len(rec), p=[0.97, 0.03])  # mostly bus A
        chunks.append(rec)
    rec = np.concatenate(chunks)
    if imperfections:
        keep = np.ones(len(rec), bool)
        for rt in rng.choice(sorted({m["remote_terminal_address"] for m in mapping}), rng.integers(0, 3), replace=False):
            g0 = start_us + rng.uniform(0, duration - 10) * 1e6  # communication gap
            keep &= ~((rec["rt"] == rt) & (rec["timestamp_us"] >= g0) & (rec["timestamp_us"] < g0 + rng.uniform(2, 8) * 1e6))
        rec = rec[keep]
        dup = rng.random(len(rec)) < 0.002
        rec = np.concatenate([rec, rec[dup]])
    rec = rec[np.argsort(rec["timestamp_us"], kind="stable")]
    rec["parity"] = (odd_parity(rec["data"]) << np.arange(32, dtype=np.uint32)).sum(axis=1).astype(np.uint32)
    if imperfections:
        bad = np.where(rng.random(len(rec)) < 0.0005)[0]  # bit errors (parity left stale)
        for i in bad:
            w = rng.integers(0, max(1, rec["word_count"][i]))
            rec["data"][i, w] ^= np.uint16(1 << rng.integers(0, 16))
        inv = rng.random(len(rec)) < 0.001  # invalid messages
        rec["message_validity"][inv] = 0
        rec["error_status"][inv] = rng.choice([0x01, 0x02, 0x04], inv.sum())
        rec["data"][inv] = 0
    rec["message_sequence"] = np.arange(len(rec), dtype=np.uint32)
    rec["message_id"] = np.uint64(msg_id_base) + rec["message_sequence"].astype(np.uint64)
    return rec


def write_bin(path, aircraft_id, flight_id, start_us, rec):
    with open(path, "wb") as fh:
        fh.write(HEADER.pack(MAGIC, VERSION, aircraft_id.encode(), flight_id.encode(), len(rec), start_us).ljust(HEADER_SIZE, b"\0"))
        fh.write(rec.tobytes())


def write_hex(path, aircraft_id, flight_id, start_us, rec):
    with open(path, "w") as fh:
        fh.write(f"# SYN1553 HEX v{VERSION} aircraft={aircraft_id} flight={flight_id} start_us={start_us} records={len(rec)}\n")
        for r in rec:
            fh.write(r.tobytes().hex() + "\n")
