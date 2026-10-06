"""Bayesian online changepoint detection (Adams & MacKay 2007), Normal-Gamma conjugate model."""
import numpy as np
from scipy.special import gammaln


def bocpd(x, hazard=1 / 20, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """Return (cp_prob, map_run_length) per observation; cp_prob = P(run length <= 2)."""
    x = np.asarray(x, float)
    T = len(x)
    R = np.zeros(T + 1)
    R[0] = 1.0
    mu, kappa, alpha, beta = (np.array([v]) for v in (mu0, kappa0, alpha0, beta0))
    cp, rl = np.zeros(T), np.zeros(T)
    for t, xt in enumerate(x):
        # Student-t predictive
        scale = beta * (kappa + 1) / (alpha * kappa)
        df = 2 * alpha
        logp = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * scale)
                - (df + 1) / 2 * np.log1p((xt - mu) ** 2 / (df * scale)))
        pred = np.exp(logp)
        growth = R[: t + 1] * pred * (1 - hazard)
        change = (R[: t + 1] * pred * hazard).sum()
        R[1: t + 2] = growth
        R[0] = change
        R[: t + 2] /= R[: t + 2].sum() or 1.0
        cp[t] = R[: min(3, t + 2)].sum() if t >= 3 else 0.0
        rl[t] = float(np.argmax(R[: t + 2]))
        mu_n = (kappa * mu + xt) / (kappa + 1)
        beta_n = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu, kappa = np.r_[mu0, mu_n], np.r_[kappa0, kappa + 1]
        alpha, beta = np.r_[alpha0, alpha + 0.5], np.r_[beta0, beta_n]
    return cp, rl


if __name__ == "__main__":  # self-check: a level shift is detected
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 0.3, 30), rng.normal(2, 0.3, 10)]
    cp, _ = bocpd(x)
    assert cp[31:34].max() > 0.5 and cp[5:29].max() < 0.5, cp
    print("bocpd ok")
