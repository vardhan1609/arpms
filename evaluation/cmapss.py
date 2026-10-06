"""NASA C-MAPSS adapter: benchmark the model-service TCN quantile RUL model on public turbofan data.

python -m evaluation.cmapss --data-dir /path/to/CMAPSSData --fd FD001
Expects train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt (26 whitespace-separated columns: unit, cycle, 3 settings, 21 sensors).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "model"))
from app import tcn  # noqa: E402

COLS = ["unit", "cycle", "s1", "s2", "s3"] + [f"x{i}" for i in range(1, 22)]
CAP, WINDOW = 125, 30


def load(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def windows(df, feats, last_only=False):
    X, keys = [], []
    for u, g in df.groupby("unit"):
        v = g[feats].values
        ends = [len(v)] if last_only else range(1, len(v) + 1)
        for e in ends:
            w = v[max(0, e - WINDOW): e]
            X.append(np.vstack([np.repeat(w[:1], WINDOW - len(w), 0), w]))
            keys.append((u, e))
    return np.array(X, np.float32), keys


def run(data_dir, fd, epochs=150):
    tr, te = load(f"{data_dir}/train_{fd}.txt"), load(f"{data_dir}/test_{fd}.txt")
    rul_true = np.loadtxt(f"{data_dir}/RUL_{fd}.txt", ndmin=1)
    feats = [c for c in COLS[2:] if tr[c].std() > 1e-4]
    tr["rul"] = np.minimum(tr.groupby("unit").cycle.transform("max") - tr.cycle, CAP)
    X, keys = windows(tr, feats)
    y = np.array([tr[(tr.unit == u)].rul.values[e - 1] for u, e in keys], np.float32)
    model = tcn.fit(X, y, epochs=epochs)
    Xt, _ = windows(te, feats, last_only=True)
    p = tcn.predict(model, Xt)
    yt = np.minimum(rul_true, CAP)
    d = p[:, 1] - yt
    return dict(fd=fd, units=len(yt), rmse=float(np.sqrt(np.mean(d ** 2))),
                nasa_score=float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))),
                coverage_80=float(np.mean((yt >= p[:, 0]) & (yt <= p[:, 2]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    print(json.dumps(run(a.data_dir, a.fd, a.epochs), indent=1))
