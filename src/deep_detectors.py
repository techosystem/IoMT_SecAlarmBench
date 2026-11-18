"""
Deep anomaly detectors for IoMT-SecAlarmBench.

Each detector plugs into eval_harness's Detector interface and operates on the
[N, C, T] window tensor from build_dataset(return_windows=True). All are trained on
physiology-normal training windows only and score every window by reconstruction or
prediction error, so their scores feed the same evaluation as Isolation Forest.

Detectors: LSTM-AE, USAD, GDN, TranAD and OmniAnomaly. They are implemented here
rather than taken from TSB-AD or DeepOD, which assume a continuous series with their
own internal windowing and metrics; this benchmark is pre-windowed and evaluated per
window. A GPU is used when available.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eval_harness import Detector


def _device(dev):
    return torch.device(dev) if dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NormMixin:
    """Per-channel standardization fit on the training (normal) windows."""
    def _fit_norm(self, X):                       # X: [N, C, T]
        self.mu_ = X.mean(axis=(0, 2), keepdims=True)
        self.sd_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
    def _apply_norm(self, X):
        return (X - self.mu_) / self.sd_


# --------------------------------------------------------------------------- #
# Reconstruction base (fit on normal -> score = per-window reconstruction MSE)
# --------------------------------------------------------------------------- #

class WindowReconstructionDetector(Detector, _NormMixin):
    name = "recon"

    def __init__(self, epochs=30, lr=1e-3, batch=256, device=None, seed=20260723, verbose=False):
        self.epochs = epochs; self.lr = lr; self.batch = batch
        self.device = _device(device); self.seed = seed; self.verbose = verbose
        self.model = None

    def _build(self, C, T):                       # returns nn.Module mapping [B,C,T]->[B,C,T-ish]
        raise NotImplementedError

    def _target(self, x):                         # reconstruction target (may be pooled)
        return x

    def fit(self, X):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        N, C, T = X.shape
        self._fit_norm(X)
        Xn = torch.tensor(self._apply_norm(X), dtype=torch.float32)
        self.model = self._build(C, T).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(N)
            tot = 0.0
            for i in range(0, N, self.batch):
                xb = Xn[perm[i:i + self.batch]].to(self.device)
                opt.zero_grad()
                rec = self.model(xb)
                loss = loss_fn(rec, self._target(xb))
                loss.backward(); opt.step()
                tot += loss.item() * xb.shape[0]
            if self.verbose:
                print(f"  [{self.name}] epoch {ep+1}/{self.epochs} loss {tot/N:.5f}")
        return self

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        Xn = torch.tensor(self._apply_norm(X), dtype=torch.float32)
        out = []
        for i in range(0, len(Xn), self.batch):
            xb = Xn[i:i + self.batch].to(self.device)
            rec = self.model(xb)
            err = ((rec - self._target(xb)) ** 2).mean(dim=tuple(range(1, rec.ndim)))
            out.append(err.cpu().numpy())
        return np.concatenate(out)                # higher = more anomalous


# --------------------------------------------------------------------------- #
# LSTM autoencoder (pooling front-end for long waveform windows)
# --------------------------------------------------------------------------- #

class _LSTMAE(nn.Module):
    def __init__(self, C, T, pool=10, hidden=64, latent=32):
        super().__init__()
        self.pool = nn.AvgPool1d(pool)
        self.Tp = T // pool
        self.enc = nn.LSTM(C, hidden, batch_first=True)
        self.to_lat = nn.Linear(hidden, latent)
        self.dec = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)

    def forward(self, x):                          # x: [B, C, T]
        xp = self.pool(x).transpose(1, 2)          # [B, Tp, C]
        _, (h, _) = self.enc(xp)
        z = self.to_lat(h[-1])                      # [B, latent]
        zr = z.unsqueeze(1).repeat(1, self.Tp, 1)  # [B, Tp, latent]
        d, _ = self.dec(zr)
        return self.out(d).transpose(1, 2)          # [B, C, Tp]


class LSTMAEDetector(WindowReconstructionDetector):
    name = "LSTM-AE"
    def __init__(self, pool=10, hidden=64, latent=32, **kw):
        super().__init__(**kw); self.pool = pool; self.hidden = hidden; self.latent = latent
    def _build(self, C, T):
        self._pooler = nn.AvgPool1d(self.pool)
        return _LSTMAE(C, T, self.pool, self.hidden, self.latent)
    def _target(self, x):                           # target is the pooled window
        return self._pooler(x)


# --------------------------------------------------------------------------- #
# USAD (adversarially trained dense autoencoder on flattened windows)
# --------------------------------------------------------------------------- #

class _MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers[:-1])      # drop last ReLU
    def forward(self, x): return self.net(x)


class USADDetector(Detector, _NormMixin):
    """USAD (Audibert et al., 2020): shared encoder + two decoders trained adversarially.
    Anomaly score = alpha*||W-AE1(W)|| + beta*||W-AE2(AE1(W))||."""
    name = "USAD"

    def __init__(self, latent=40, hidden=256, epochs=30, lr=1e-4, batch=256,
                 alpha=0.5, beta=0.5, clip=1.0, device=None, seed=20260723, verbose=False):
        self.latent = latent; self.hidden = hidden; self.epochs = epochs; self.lr = lr
        self.batch = batch; self.alpha = alpha; self.beta = beta; self.clip = clip
        self.device = _device(device); self.seed = seed; self.verbose = verbose

    def _flatten(self, X):
        return X.reshape(X.shape[0], -1)            # [N, C*T]

    def fit(self, X):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self._fit_norm(X)
        W = torch.tensor(self._flatten(self._apply_norm(X)), dtype=torch.float32)
        N, D = W.shape
        self.E = _MLP([D, self.hidden, self.latent]).to(self.device)
        self.D1 = _MLP([self.latent, self.hidden, D]).to(self.device)
        self.D2 = _MLP([self.latent, self.hidden, D]).to(self.device)
        opt1 = torch.optim.Adam(list(self.E.parameters()) + list(self.D1.parameters()), lr=self.lr)
        opt2 = torch.optim.Adam(list(self.E.parameters()) + list(self.D2.parameters()), lr=self.lr)
        p1 = list(self.E.parameters()) + list(self.D1.parameters())
        p2 = list(self.E.parameters()) + list(self.D2.parameters())
        for ep in range(1, self.epochs + 1):
            perm = torch.randperm(N)
            for i in range(0, N, self.batch):
                w = W[perm[i:i + self.batch]].to(self.device)
                z = self.E(w); w1 = self.D1(z); w2d1 = self.D2(self.E(w1))
                l1 = (1 / ep) * ((w - w1) ** 2).mean() + (1 - 1 / ep) * ((w - w2d1) ** 2).mean()
                opt1.zero_grad(); l1.backward()
                torch.nn.utils.clip_grad_norm_(p1, self.clip); opt1.step()
                z = self.E(w); w2 = self.D2(z); w2d1 = self.D2(self.E(self.D1(z)))
                l2 = (1 / ep) * ((w - w2) ** 2).mean() - (1 - 1 / ep) * ((w - w2d1) ** 2).mean()
                opt2.zero_grad(); l2.backward()
                torch.nn.utils.clip_grad_norm_(p2, self.clip); opt2.step()
            if self.verbose:
                print(f"  [USAD] epoch {ep}/{self.epochs} l1 {l1.item():.4f} l2 {l2.item():.4f}")
        return self

    @torch.no_grad()
    def score(self, X):
        for m in (self.E, self.D1, self.D2): m.eval()
        X = np.asarray(X, dtype=np.float32)
        W = torch.tensor(self._flatten(self._apply_norm(X)), dtype=torch.float32)
        out = []
        for i in range(0, len(W), self.batch):
            w = W[i:i + self.batch].to(self.device)
            w1 = self.D1(self.E(w)); w2d1 = self.D2(self.E(w1))
            s = self.alpha * ((w - w1) ** 2).mean(1) + self.beta * ((w - w2d1) ** 2).mean(1)
            out.append(s.cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
# GDN — Graph Deviation Network (Deng & Hooi 2021): learned channel graph +
# graph-attention forecasting. Coupling-aware by construction — each channel is
# forecast from its graph neighbours, so an attack that breaks ECG-PPG coupling
# raises the forecast deviation. Adapted to per-window scoring: within each window
# a sliding sub-window forecasts the next (pooled) step; the per-window score is the
# per-sensor-normalized max deviation, averaged over positions.
# Note: with the 2-channel core cluster the learned top-k graph is degenerate
# (each channel attends to the other); the value here is the cross-channel graph
# attention, which is exactly the coupling mechanism under test.
# --------------------------------------------------------------------------- #

class _GDNNet(nn.Module):
    def __init__(self, C, w, emb_dim=32, hidden=64, topk=None):
        super().__init__()
        self.C = C
        self.topk = topk if topk else max(1, C - 1)
        self.emb = nn.Embedding(C, emb_dim)
        self.lin = nn.Linear(w, hidden)
        self.att = nn.Linear(2 * (hidden + emb_dim), 1)
        self.out = nn.Sequential(nn.Linear(hidden + emb_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):                              # x: [B, C, w]
        B = x.shape[0]
        idx = torch.arange(self.C, device=x.device)
        v = self.emb(idx)                              # [C, emb]
        h = torch.relu(self.lin(x))                    # [B, C, hidden]
        vB = v.unsqueeze(0).expand(B, -1, -1)          # [B, C, emb]
        g = torch.cat([h, vB], dim=-1)                 # [B, C, hidden+emb]
        vn = F.normalize(v, dim=1)
        sim = vn @ vn.t()                              # [C, C] cosine sim of embeddings
        sim = sim - torch.eye(self.C, device=x.device) * 1e9
        nbr = sim.topk(self.topk, dim=1).indices       # [C, topk]
        zs = []
        for i in range(self.C):
            js = nbr[i]                                # [topk]
            gi = g[:, i:i + 1, :].expand(-1, self.topk, -1)
            gj = g[:, js, :]
            e = self.att(torch.cat([gi, gj], dim=-1)).squeeze(-1)   # [B, topk]
            a = torch.softmax(torch.relu(e), dim=1).unsqueeze(-1)
            zs.append((a * h[:, js, :]).sum(1))        # [B, hidden]
        z = torch.stack(zs, dim=1)                     # [B, C, hidden]
        pred = self.out(torch.cat([z, vB], dim=-1)).squeeze(-1)     # [B, C]
        return pred


class GDNDetector(Detector, _NormMixin):
    name = "GDN"

    def __init__(self, pool=10, w=20, emb_dim=32, hidden=64, topk=None, epochs=30,
                 lr=1e-3, batch=512, train_pos=16, clip=1.0, device=None,
                 seed=20260723, verbose=False):
        self.pool = pool; self.w = w; self.emb_dim = emb_dim; self.hidden = hidden
        self.topk = topk; self.epochs = epochs; self.lr = lr; self.batch = batch
        self.train_pos = train_pos; self.clip = clip
        self.device = _device(device); self.seed = seed; self.verbose = verbose

    def _pool(self, X):                                # [N,C,T] -> [N,C,Tp]
        t = torch.tensor(np.asarray(X, dtype=np.float32))
        return F.avg_pool1d(t, self.pool)

    def _subwins(self, Xp):                            # [b,C,Tp] -> inputs[b*num,C,w], targets[b*num,C], num
        b, C, Tp = Xp.shape
        u = Xp.unfold(2, self.w + 1, 1)                # [b, C, num, w+1]
        num = u.shape[2]
        u = u.permute(0, 2, 1, 3).reshape(b * num, C, self.w + 1)
        return u[:, :, :self.w], u[:, :, -1], num

    def fit(self, X):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32); C = X.shape[1]
        self._fit_norm(X)
        Xp = self._pool(self._apply_norm(X))           # [N,C,Tp]
        N, _, Tp = Xp.shape
        num_all = Tp - self.w
        # sample train_pos forecast positions per window
        rng = np.random.default_rng(self.seed)
        starts = rng.integers(0, num_all, size=(N, self.train_pos))
        inp = torch.empty(N * self.train_pos, C, self.w)
        tgt = torch.empty(N * self.train_pos, C)
        k = 0
        for n in range(N):
            for s in starts[n]:
                inp[k] = Xp[n, :, s:s + self.w]; tgt[k] = Xp[n, :, s + self.w]; k += 1
        self.model = _GDNNet(C, self.w, self.emb_dim, self.hidden, self.topk).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        M = inp.shape[0]
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(M); tot = 0.0
            for i in range(0, M, self.batch):
                b = perm[i:i + self.batch]
                xb = inp[b].to(self.device); yb = tgt[b].to(self.device)
                opt.zero_grad()
                loss = ((self.model(xb) - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                opt.step(); tot += loss.item() * xb.shape[0]
            if self.verbose:
                print(f"  [GDN] epoch {ep+1}/{self.epochs} loss {tot/M:.5f}")
        # per-sensor error normalization stats on a train sample
        with torch.no_grad():
            self.model.eval()
            err = ((self.model(inp[:5000].to(self.device)) - tgt[:5000].to(self.device)) ** 2).cpu().numpy()
        self.err_med = np.median(err, axis=0)          # [C]
        self.err_iqr = (np.percentile(err, 75, axis=0) - np.percentile(err, 25, axis=0)) + 1e-6
        return self

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        X = np.asarray(X, dtype=np.float32); N = X.shape[0]
        med = torch.tensor(self.err_med, device=self.device)
        iqr = torch.tensor(self.err_iqr, device=self.device)
        out = np.empty(N, dtype=np.float32)
        CH = 512
        for i in range(0, N, CH):
            Xp = self._pool(self._apply_norm(X[i:i + CH])).to(self.device)   # [b,C,Tp]
            b = Xp.shape[0]
            inp, tgt, num = self._subwins(Xp)
            err = ((self.model(inp) - tgt) ** 2)                              # [b*num, C]
            err = (err - med) / iqr                                          # per-sensor normalize
            err = err.reshape(b, num, -1)
            wscore = err.amax(dim=2).mean(dim=1)                             # max over sensors, mean over positions
            out[i:i + b] = wscore.cpu().numpy()
        return out


# --------------------------------------------------------------------------- #
# TranAD (Tuli et al. 2022): transformer encoder + two decoders with two-phase
# focus-score self-conditioning. Phase 1 reconstructs the (pooled) window; the
# per-feature phase-1 error becomes a focus score conditioning phase 2. Score =
# mean of the two reconstruction errors. Window is pooled so attention is tractable.
# --------------------------------------------------------------------------- #

class _TranAD(nn.Module):
    def __init__(self, C, Tp, d_model=64, nhead=4, layers=1, dim_ff=128, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, Tp, d_model) * 0.02)
        self.enc_inp = nn.Linear(2 * C, d_model)
        self.tgt_inp = nn.Linear(C, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        d1 = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        d2 = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.dec1 = nn.TransformerDecoder(d1, layers)
        self.dec2 = nn.TransformerDecoder(d2, layers)
        self.out1 = nn.Linear(d_model, C)
        self.out2 = nn.Linear(d_model, C)

    def _enc(self, W, focus):
        x = self.enc_inp(torch.cat([W, focus], -1)) + self.pos
        return self.encoder(x)

    def forward(self, W):                              # W: [B, Tp, C]
        tgt = self.tgt_inp(W) + self.pos
        o1 = self.out1(self.dec1(tgt, self._enc(W, torch.zeros_like(W))))
        focus = (o1 - W) ** 2
        o2 = self.out2(self.dec2(tgt, self._enc(W, focus)))
        return o1, o2


class TranADDetector(Detector, _NormMixin):
    name = "TranAD"

    def __init__(self, pool=10, d_model=64, nhead=4, layers=1, epochs=30, lr=1e-3,
                 batch=256, clip=1.0, device=None, seed=20260723, verbose=False):
        self.pool = pool; self.d_model = d_model; self.nhead = nhead; self.layers = layers
        self.epochs = epochs; self.lr = lr; self.batch = batch; self.clip = clip
        self.device = _device(device); self.seed = seed; self.verbose = verbose

    def _prep(self, X):                                # [N,C,T] -> [N,Tp,C] normalized+pooled
        t = torch.tensor(self._apply_norm(np.asarray(X, dtype=np.float32)))
        return F.avg_pool1d(t, self.pool).transpose(1, 2)

    def fit(self, X):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32); C = X.shape[1]
        self._fit_norm(X)
        Wd = self._prep(X)                              # [N, Tp, C]
        N, Tp, _ = Wd.shape
        self.model = _TranAD(C, Tp, self.d_model, self.nhead, self.layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for ep in range(1, self.epochs + 1):
            perm = torch.randperm(N); tot = 0.0
            for i in range(0, N, self.batch):
                w = Wd[perm[i:i + self.batch]].to(self.device)
                o1, o2 = self.model(w)
                loss = (1 / ep) * ((o1 - w) ** 2).mean() + (1 - 1 / ep) * ((o2 - w) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                opt.step(); tot += loss.item() * w.shape[0]
            if self.verbose:
                print(f"  [TranAD] epoch {ep}/{self.epochs} loss {tot/N:.5f}")
        return self

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        Wd = self._prep(X)
        out = []
        for i in range(0, len(Wd), self.batch):
            w = Wd[i:i + self.batch].to(self.device)
            o1, o2 = self.model(w)
            s = 0.5 * ((o1 - w) ** 2).mean(dim=(1, 2)) + 0.5 * ((o2 - w) ** 2).mean(dim=(1, 2))
            out.append(s.cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
# OmniAnomaly (Su et al. 2019): stochastic recurrent VAE. A GRU encodes the
# (pooled) window; a per-timestep Gaussian latent z is sampled (reparameterization);
# a GRU decoder emits a Gaussian observation model (mean AND learned variance).
# Trained by ELBO (Gaussian reconstruction log-likelihood + KL). Anomaly score =
# reconstruction NLL ("reconstruction probability") — higher = more anomalous. This
# is the faithful probabilistic core; the original paper's planar-flow + linear-Gaussian
# state-space refinements are omitted. logvars are clamped and grads clipped to
# prevent the variance-collapse instability VAEs are prone to.
# --------------------------------------------------------------------------- #

_LOG2PI = float(np.log(2 * np.pi))


class _OmniVAE(nn.Module):
    def __init__(self, C, hidden=64, latent=16):
        super().__init__()
        self.enc = nn.GRU(C, hidden, batch_first=True)
        self.emu = nn.Linear(hidden, latent)
        self.elv = nn.Linear(hidden, latent)
        self.dec = nn.GRU(latent, hidden, batch_first=True)
        self.dmu = nn.Linear(hidden, C)
        self.dlv = nn.Linear(hidden, C)

    def forward(self, x, sample=True):                 # x: [B, Tp, C]
        h, _ = self.enc(x)
        zmu = self.emu(h); zlv = self.elv(h).clamp(-5, 5)
        z = zmu + torch.randn_like(zmu) * torch.exp(0.5 * zlv) if sample else zmu
        d, _ = self.dec(z)
        xmu = self.dmu(d); xlv = self.dlv(d).clamp(-5, 3)
        return xmu, xlv, zmu, zlv


class OmniAnomalyDetector(Detector, _NormMixin):
    name = "OmniAnomaly"

    def __init__(self, pool=10, hidden=64, latent=16, epochs=30, lr=1e-3, batch=256,
                 beta=1.0, clip=1.0, device=None, seed=20260723, verbose=False):
        self.pool = pool; self.hidden = hidden; self.latent = latent; self.epochs = epochs
        self.lr = lr; self.batch = batch; self.beta = beta; self.clip = clip
        self.device = _device(device); self.seed = seed; self.verbose = verbose

    def _prep(self, X):
        t = torch.tensor(self._apply_norm(np.asarray(X, dtype=np.float32)))
        return F.avg_pool1d(t, self.pool).transpose(1, 2)     # [N, Tp, C]

    def _nll(self, w, xmu, xlv):                        # per-window mean Gaussian NLL
        return 0.5 * (xlv + (w - xmu) ** 2 * torch.exp(-xlv) + _LOG2PI)

    def fit(self, X):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32); C = X.shape[1]
        self._fit_norm(X)
        Wd = self._prep(X); N = Wd.shape[0]
        self.model = _OmniVAE(C, self.hidden, self.latent).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for ep in range(1, self.epochs + 1):
            perm = torch.randperm(N); tot = 0.0
            for i in range(0, N, self.batch):
                w = Wd[perm[i:i + self.batch]].to(self.device)
                xmu, xlv, zmu, zlv = self.model(w, sample=True)
                recon = self._nll(w, xmu, xlv).sum(dim=(1, 2)).mean()
                kl = (-0.5 * (1 + zlv - zmu ** 2 - torch.exp(zlv))).sum(dim=(1, 2)).mean()
                loss = recon + self.beta * kl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                opt.step(); tot += loss.item() * w.shape[0]
            if self.verbose:
                print(f"  [Omni] epoch {ep}/{self.epochs} elbo {tot/N:.4f}")
        return self

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        Wd = self._prep(X); out = []
        for i in range(0, len(Wd), self.batch):
            w = Wd[i:i + self.batch].to(self.device)
            xmu, xlv, _, _ = self.model(w, sample=False)   # posterior mean for a stable score
            s = self._nll(w, xmu, xlv).mean(dim=(1, 2))     # reconstruction NLL
            out.append(s.cpu().numpy())
        return np.concatenate(out)


if __name__ == "__main__":
    # smoke test on the harness demo (CPU, few epochs)
    import numpy as np
    from injection_generator import GeneratorConfig
    from eval_harness import build_dataset, patient_wise_split, run_rq1
    FS = 125.0; rng = np.random.default_rng(0); recs = []
    for i in range(6):
        for atype, tv in [("Asystole", False), ("Tachycardia", True), ("VentricularTachycardia", False)]:
            T = int(FS * 300); t = np.arange(T) / FS; ph = 2 * np.pi * (70 + 3 * i) / 60.0 * t
            sig = np.vstack([np.sin(ph) + 0.02 * rng.standard_normal(T),
                             0.9 * np.sin(ph - 0.6) + 0.02 * rng.standard_normal(T)])
            sig = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
            recs.append(dict(record_id=f"{atype[:3]}{i}", signal=sig, channels=["ECG", "PPG"],
                             alarm_type=atype, true_alarm=tv, alarm_onset_sample=int(T * 0.85), fs=FS))
    df, feat_cols, windows = build_dataset(recs, GeneratorConfig(), return_windows=True, verbose=True)
    df = patient_wise_split(df)
    for det in (LSTMAEDetector(epochs=3, verbose=True), USADDetector(epochs=3, verbose=True),
                GDNDetector(epochs=3, verbose=True)):
        print(f"\n########## {det.name} ##########")
        run_rq1(df, windows, det)
