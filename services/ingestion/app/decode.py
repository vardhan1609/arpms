"""Readers for raw source files. See docs/data_formats.md for the 1553 capture format."""
import numpy as np
import pandas as pd

MAGIC = b"SYN1553\x00"
HEADER_SIZE = 64
RECORD = np.dtype([
    ("message_id", "<u8"), ("timestamp_us", "<i8"), ("bus_id", "u1"), ("rt", "u1"), ("sa", "u1"),
    ("direction", "u1"), ("command_word", "<u2"), ("status_word", "<u2"), ("word_count", "u1"),
    ("message_validity", "u1"), ("error_status", "<u2"), ("message_sequence", "<u4"), ("parity", "<u4"),
    ("data", "<u2", (32,)),
])


class FormatError(ValueError):
    pass


def read_bus(path):
    """Return (header dict, structured record array) for a .bin or .hex capture."""
    if path.endswith(".hex"):
        with open(path) as fh:
            head = fh.readline()
            if not head.startswith("# SYN1553 HEX"):
                raise FormatError(f"{path}: not a SYN1553 hex capture")
            meta = dict(kv.split("=", 1) for kv in head.split()[4:])
            body = bytes.fromhex(fh.read().replace("\n", ""))
        if len(body) % RECORD.itemsize:
            raise FormatError(f"{path}: truncated record")
        rec = np.frombuffer(body, RECORD).copy()
        return dict(aircraft_id=meta["aircraft"], flight_id=meta["flight"], start_us=int(meta["start_us"]),
                    record_count=int(meta["records"])), rec
    with open(path, "rb") as fh:
        hdr = fh.read(HEADER_SIZE)
    if hdr[:8] != MAGIC:
        raise FormatError(f"{path}: bad magic")
    version = int.from_bytes(hdr[8:10], "little")
    if version != 1:
        raise FormatError(f"{path}: unsupported version {version}")
    count = int.from_bytes(hdr[50:54], "little")
    rec = np.fromfile(path, RECORD, offset=HEADER_SIZE)
    if len(rec) != count:
        raise FormatError(f"{path}: header says {count} records, file has {len(rec)}")
    return dict(aircraft_id=hdr[10:26].rstrip(b"\0").decode(), flight_id=hdr[26:50].rstrip(b"\0").decode(),
                start_us=int.from_bytes(hdr[54:62], "little", signed=True), record_count=count), rec


def parity_ok(rec):
    """Odd parity per data word, packed as a 32-bit mask (bit i = parity bit of word i)."""
    bits = np.unpackbits(rec["data"].view(np.uint8).reshape(len(rec), 32, 2), axis=-1).sum(-1)
    expected = (bits % 2 == 0).astype(np.uint64)
    stored = (rec["parity"][:, None].astype(np.uint64) >> np.arange(32, dtype=np.uint64)) & 1
    return (expected == stored).all(axis=1)


def first_all(good, rt, sa):
    return good["timestamp_us"][(good["rt"] == rt) & (good["sa"] == sa)]


def decode(rec, mapping):
    """Engineering-unit series per parameter + error rows + per-message stats.

    Invalid and parity-failing messages are excluded from values and reported as errors;
    exact duplicates (same RT/SA/timestamp) are dropped.
    """
    invalid = rec["message_validity"] == 0
    bad_parity = ~invalid & ~parity_ok(rec)
    errors = [dict(message_id=int(r["message_id"]), timestamp_us=int(r["timestamp_us"]), rt=int(r["rt"]), sa=int(r["sa"]),
                   error_type="INVALID" if inv else "PARITY", raw_hex=r.tobytes().hex())
              for r, inv in zip(rec[invalid | bad_parity], invalid[invalid | bad_parity])]
    good = rec[~invalid & ~bad_parity]
    series, stats = {}, {}
    for (rt, sa), g in mapping.groupby(["remote_terminal_address", "sub_address"]):
        sel = (rec["rt"] == rt) & (rec["sa"] == sa)
        r = good[(good["rt"] == rt) & (good["sa"] == sa)]
        ts, first = np.unique(r["timestamp_us"], return_index=True)
        r = r[first]
        stats[g.message_id.iloc[0]] = dict(total=int(sel.sum()), invalid=int((sel & invalid).sum()),
                                           parity=int((sel & bad_parity).sum()), duplicates=int(len(first_all(good, rt, sa)) - len(first)))
        for m in g.itertuples():
            raw = (r["data"][:, m.word_position - 1].astype(np.int64) >> m.bit_position) & ((1 << m.bit_length) - 1)
            series[m.parameter_id] = (ts, raw * m.scaling_factor + m.offset)
    return series, errors, stats


def read_lru_log(path):
    df = pd.read_csv(path)
    if "lru_time" not in df or df.shape[1] < 4:
        raise FormatError(f"{path}: missing lru_time / parameter columns")
    return df.drop_duplicates().sort_values("lru_time").reset_index(drop=True)
