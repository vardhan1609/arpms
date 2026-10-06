"""Runnable checks for the non-trivial logic: `python tests/test_core.py` (or pytest).
DB-backed checks run only when the database is reachable and populated."""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "services", "ingestion")]
from app import decode, signal  # noqa: E402
from synthetic_generator import bus1553_generator as gen  # noqa: E402


def _records(n=50):
    rec = np.zeros(n, decode.RECORD)
    rng = np.random.default_rng(1)
    rec["message_id"] = np.arange(n)
    rec["timestamp_us"] = np.arange(n) * 20_000
    rec["rt"], rec["sa"], rec["word_count"], rec["message_validity"] = 3, 1, 4, 1
    rec["data"][:, :4] = rng.integers(0, 2 ** 16, (n, 4))
    rec["parity"] = (gen.odd_parity(rec["data"]) << np.arange(32, dtype=np.uint32)).sum(axis=1).astype(np.uint32)
    return rec


def test_bus_roundtrip_and_parity():
    rec = _records()
    with tempfile.TemporaryDirectory() as d:
        for ext, writer in (("bin", gen.write_bin), ("hex", gen.write_hex)):
            p = os.path.join(d, f"x.{ext}")
            writer(p, "SYN-AC-001", "SYN-AC-001-F0001", 123, rec)
            hdr, back = decode.read_bus(p)
            assert hdr["flight_id"] == "SYN-AC-001-F0001" and hdr["start_us"] == 123 and hdr["record_count"] == len(rec)
            assert (back == rec).all()
        assert decode.parity_ok(rec).all()
        rec["data"][7, 2] ^= 1 << 5  # single bit error
        ok = decode.parity_ok(rec)
        assert not ok[7] and ok.sum() == len(rec) - 1
        bad = os.path.join(d, "bad.bin")
        open(bad, "wb").write(b"NOTMAGIC" + bytes(100))
        try:
            decode.read_bus(bad)
            raise AssertionError("bad magic accepted")
        except decode.FormatError:
            pass


def _altitude(t):
    return np.interp(t, [0, 300, 900, 1500, 2100, 2700, 3000], [0, 0, 3000, 3200, 1800, 2500, 0]) + 40 * np.sin(t / 37)


def test_clock_sync_recovers_offset():
    t = np.arange(3000.0)
    offset, drift = -41.5, 80e-6
    bus_alt = _altitude(t + 0.5)          # 1 Hz bucket means sit mid-bucket
    lru_true = np.arange(0, 3000, 1.0)
    lru_t = offset + (1 + drift) * lru_true
    est = signal.clock_sync(bus_alt, lru_t, _altitude(lru_true))
    assert abs(est["offset_s"] - offset) < 1.0, est


def test_kalman_rejects_spike_keeps_step():
    rng = np.random.default_rng(0)
    y = np.r_[np.full(200, 10.0), np.full(200, 20.0)] + rng.normal(0, 0.1, 400)
    y[100] = 60.0
    y[150:155] = np.nan
    s, out = signal.kalman_clean(y[None, :])
    assert out[0, 100] and abs(s[0, 100] - 10) < 0.5
    assert np.isfinite(s).all() and abs(s[0, 350] - 20) < 0.5 and out[0].sum() < 5


def _db_ready():
    try:
        from arpms_common import q1
        return bool(q1("SELECT 1 FROM predictions LIMIT 1"))
    except Exception:
        return False


def test_diagnosis_and_hitl():
    if not _db_ready():
        print("skip: database not populated")
        return
    for m in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[m]
    sys.path.insert(0, os.path.join(ROOT, "services", "diagnosis"))
    from fastapi import HTTPException
    from arpms_common import q1
    from app import main as dg
    p = q1("SELECT flight_id, lru_id FROM predictions ORDER BY failure_probability DESC LIMIT 1")
    d = dg.get_diagnosis(dg.diagnose(p["flight_id"], p["lru_id"])["diagnosis_id"])
    pf = d["evidence"]["p_fault_present"]
    assert 0.5 < sum(c["posterior_given_fault"] for c in d["candidates"]) <= 1.001
    assert all(c["probability"] <= pf + 1e-6 for c in d["candidates"])
    assert d["recommendations"] and all(r["status"] == "PENDING" for r in d["recommendations"])
    rid = d["recommendations"][0]["recommendation_id"]
    for bad in (dict(decision="REJECT", engineer="e1"), dict(decision="OVERRIDE", engineer="e1", note="n")):
        try:
            dg.api_decide(rid, dg.Decision(**bad))
            raise AssertionError(f"accepted {bad}")
        except HTTPException as e:
            assert e.status_code == 400
    assert dg.api_decide(rid, dg.Decision(decision="ACCEPT", engineer="e1"))["status"] == "ACCEPTED"
    try:
        dg.api_decide(rid, dg.Decision(decision="ACCEPT", engineer="e2"))
        raise AssertionError("decided twice")
    except HTTPException as e:
        assert e.status_code == 409


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
