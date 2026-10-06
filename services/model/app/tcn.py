"""Temporal convolutional network for RUL with quantile (pinball) outputs."""
import torch
from torch import nn

QUANTILES = (0.1, 0.5, 0.9)


class TCN(nn.Module):
    def __init__(self, n_in, ch=32, k=3, dilations=(1, 2, 4)):
        super().__init__()
        layers, c = [], n_in
        for d in dilations:
            layers += [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(c, ch, k, dilation=d), nn.ReLU()]
            c = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(ch, len(QUANTILES))

    def forward(self, x):  # x: (B, W, F)
        h = self.net(x.transpose(1, 2))[:, :, -1]
        out = self.head(h)
        return torch.cumsum(torch.cat([out[:, :1], nn.functional.softplus(out[:, 1:])], 1), 1)  # monotone quantiles


def pinball(pred, y):
    q = torch.tensor(QUANTILES)
    e = y[:, None] - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def fit(X, y, epochs=300, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    m = TCN(X.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = pinball(m(xt), yt)
        loss.backward()
        opt.step()
    return {"state": m.state_dict(), "mu": mu, "sd": sd, "n_in": X.shape[-1]}


def predict(model, X):
    m = TCN(model["n_in"])
    m.load_state_dict(model["state"])
    m.eval()
    with torch.no_grad():
        return m(torch.tensor((X - model["mu"]) / model["sd"], dtype=torch.float32)).numpy()
