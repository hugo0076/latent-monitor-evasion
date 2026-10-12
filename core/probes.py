"""Concept monitors: logistic-regression, MLP and attention-pooled probes.

Each monitor is a one-vs-rest classifier for one concept. `logreg` mean-pools and reads one
linear direction. `mlp` mean-pools and applies a hidden layer. `attention` pools over token
positions with learned weights.

All kinds share one calibration: `z = (raw - threshold) / scale`. `threshold` is the midpoint
of the two class means and `scale` is the pooled standard deviation of the training scores.

The MLP and attention heads train for a fixed number of steps. Their reported holdout AUROC
is the final iterate. No checkpoint is selected on the holdout.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from core.activations import pad_blocks

KINDS = ("logreg", "mlp", "attention")

NEEDS_TOKENS: dict[str, bool] = {"logreg": False, "mlp": False, "attention": True}
# logreg reads raw activations, so w is in the space of Δh. The heads standardise.
STANDARDISED: dict[str, bool] = {"logreg": False, "mlp": True, "attention": True}


def seed_for(*parts: Any) -> int:
    """A stable seed from naming parts. Uses blake2, because builtin `hash()` changes
    between processes."""
    key = "/".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def rng_for(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(seed_for(*parts))


def allocate(n_total: int, n_sources: int) -> list[int]:
    """Split `n_total` draws as evenly as possible across `n_sources`."""
    base, remainder = divmod(n_total, n_sources)
    return [base + (1 if i < remainder else 0) for i in range(n_sources)]


@dataclass(frozen=True)
class Arm:
    """One side of a fit: pooled vectors, and token blocks when the head needs them."""

    pooled: np.ndarray
    blocks: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.pooled)


@dataclass(frozen=True)
class ConceptData:
    """The four arms one concept's monitor is fitted and scored on."""

    pos: Arm
    neg: Arm
    pos_holdout: Arm
    neg_holdout: Arm


@dataclass
class MonitorSet:
    """K stacked per-concept monitors of one kind, indexed by position in `names`."""

    kind: str
    names: list[str]
    layer: int
    params: dict[str, np.ndarray]
    mu: np.ndarray
    sd: np.ndarray
    threshold: np.ndarray
    scale: np.ndarray
    train_auroc: np.ndarray
    holdout_auroc: np.ndarray  # final iterate, the reported value
    holdout_auroc_best: np.ndarray  # best value seen in fitting, diagnostic only
    n_train_pos: np.ndarray
    n_train_neg: np.ndarray

    @property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def raw(self, H: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Uncalibrated head output for every concept. [N, K].

        An attention head cannot score a pooled vector: with one position its softmax is
        uniform, which is a different function. Pass the per-token block.
        """
        if H.ndim == 2:
            if NEEDS_TOKENS[self.kind]:
                raise ValueError(
                    f"{self.kind!r} pools over token positions and cannot be scored from a "
                    f"pooled [N, D] vector; pass the per-token block [N, T, D]"
                )
            H = H[:, None, :]
        if mask is None:
            mask = np.ones(H.shape[:2], dtype=np.float32)
        with torch.no_grad():
            P = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in self.params.items()
            }
            P["mu"] = torch.as_tensor(self.mu)
            P["sd"] = torch.as_tensor(self.sd)
            out = _forward(
                self.kind,
                P,
                torch.as_tensor(np.asarray(H), dtype=torch.float32),
                torch.as_tensor(np.asarray(mask), dtype=torch.float32),
            )
        return out.numpy()

    def logits(self, H: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Calibrated logits `(raw - threshold) / scale`. Higher = more concept. [N, K]."""
        return (self.raw(H, mask) - self.threshold) / self.scale

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            kind=self.kind,
            names=np.array(self.names, dtype=object),
            layer=self.layer,
            mu=self.mu,
            sd=self.sd,
            threshold=self.threshold,
            scale=self.scale,
            train_auroc=self.train_auroc,
            holdout_auroc=self.holdout_auroc,
            holdout_auroc_best=self.holdout_auroc_best,
            n_train_pos=self.n_train_pos,
            n_train_neg=self.n_train_neg,
            **{f"p_{k}": v for k, v in self.params.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "MonitorSet":
        z = np.load(path, allow_pickle=True)
        return cls(
            kind=str(z["kind"]),
            names=[str(x) for x in z["names"]],
            layer=int(z["layer"]),
            params={k[2:]: z[k] for k in z.files if k.startswith("p_")},
            mu=z["mu"],
            sd=z["sd"],
            threshold=z["threshold"],
            scale=z["scale"],
            train_auroc=z["train_auroc"],
            holdout_auroc=z["holdout_auroc"],
            holdout_auroc_best=z["holdout_auroc_best"],
            n_train_pos=z["n_train_pos"],
            n_train_neg=z["n_train_neg"],
        )


def _forward(
    kind: str, P: dict[str, torch.Tensor], h: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Raw head output for all K concepts. h [B,T,D], mask [B,T] -> [B,K]."""
    x = (h.float() - P["mu"]) / P["sd"] if STANDARDISED[kind] else h.float()
    m = mask.float()
    if kind in ("logreg", "mlp"):
        xp = (x * m.unsqueeze(-1)).sum(1) / m.sum(1).clamp_min(1.0).unsqueeze(-1)
        if kind == "logreg":
            return xp @ P["w"].T - P["b"]
        hid = F.gelu(torch.einsum("bd,khd->bkh", xp, P["W1"]) + P["b1"])
        return torch.einsum("bkh,kh->bk", hid, P["w2"]) + P["b2"]
    if kind == "attention":
        d = x.shape[-1]
        s = torch.einsum("btd,kd->bkt", x, P["q"]) / (d**0.5)
        s = s.masked_fill(m.unsqueeze(1) < 0.5, float("-inf"))
        pooled = torch.einsum("bkt,btd->bkd", torch.softmax(s, dim=-1), x)
        return torch.einsum("bkd,kd->bk", pooled, P["v"]) + P["b"]
    raise ValueError(f"unknown monitor kind {kind!r}")


def _calibrate(z_pos: np.ndarray, z_neg: np.ndarray) -> tuple[float, float]:
    return (
        float(0.5 * (z_pos.mean() + z_neg.mean())),
        float(max(np.concatenate([z_pos, z_neg]).std(), 1e-6)),
    )


def _auroc(z_pos: np.ndarray, z_neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(z_pos)), np.zeros(len(z_neg))])
    return float(roc_auc_score(y, np.concatenate([z_pos, z_neg])))


def _fit_logreg(data: ConceptData, *, C: float, max_iter: int, seed: int) -> dict:
    """Fit the linear probe on every available negative.

    That is 49 x 150 = 7350 negatives for a training concept and 59 x 150 = 8850 for a
    held-out concept. With no subsample, the fit is deterministic given the split.
    `class_weight="balanced"` balances the 150 positives against the negatives. The
    calibrated threshold replaces the intercept.
    """
    X = np.concatenate([data.pos.pooled, data.neg.pooled])
    y = np.concatenate([np.ones(len(data.pos)), np.zeros(len(data.neg))]).astype(int)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
        class_weight="balanced",
    ).fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    w = (w / np.linalg.norm(w)).astype(np.float32)

    z_pos, z_neg = data.pos.pooled @ w, data.neg.pooled @ w
    threshold, scale = _calibrate(z_pos, z_neg)
    return {
        "params": {"w": w, "b": np.float32(0.0)},
        "train_auroc": _auroc(z_pos, z_neg),
        "holdout_auroc": _auroc(
            data.pos_holdout.pooled @ w, data.neg_holdout.pooled @ w
        ),
        "holdout_auroc_best": _auroc(
            data.pos_holdout.pooled @ w, data.neg_holdout.pooled @ w
        ),
        "threshold": threshold,
        "scale": scale,
    }


def _fit_head(
    kind: str,
    data: ConceptData,
    *,
    mu: torch.Tensor,
    sd: torch.Tensor,
    hidden: int,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    batch_size: int = 512,
) -> dict:
    """Fit one MLP or attention head with minibatch BCE for a fixed number of steps.

    Each minibatch holds equal numbers of positives and negatives with equal weights, which
    matches `class_weight="balanced"` for the logistic probe. Minibatches keep memory bounded:
    the padded negative set of one attention head is about 30 GB.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    keep_tokens = NEEDS_TOKENS[kind]
    D = data.pos.pooled.shape[-1]

    # bfloat16, not float16, because gemma-3-27b activations overflow float16.
    def park(arm: Arm, width: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        if keep_tokens:
            H, m = pad_blocks(arm.blocks, width)
            X = torch.as_tensor(H).to(device, torch.bfloat16)
        else:
            X = torch.as_tensor(arm.pooled[:, None, :]).to(device, torch.float32)
            m = np.ones((len(arm), 1), np.float32)
        return X, torch.as_tensor(m).to(device, torch.bfloat16)

    width = None
    if keep_tokens:
        width = max(
            len(b)
            for arm in (data.pos, data.neg, data.pos_holdout, data.neg_holdout)
            for b in arm.blocks
        )
    parked = {
        name: park(getattr(data, name), width)
        for name in ("pos", "neg", "pos_holdout", "neg_holdout")
    }

    def take(
        name: str, idx: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice a minibatch off the parked arm and standardise it."""
        X, M = parked[name]
        sel = torch.as_tensor(idx, device=X.device)
        return (X[sel].float() - mu) / sd, M[sel].float()

    n_pos, n_neg = len(data.pos), len(data.neg)
    if kind == "mlp":
        W1 = (
            (torch.randn(hidden, D, generator=gen) / D**0.5).to(device).requires_grad_()
        )
        b1 = torch.zeros(hidden, device=device, requires_grad=True)
        w2 = (
            (torch.randn(hidden, generator=gen) / hidden**0.5)
            .to(device)
            .requires_grad_()
        )
        b2 = torch.zeros((), device=device, requires_grad=True)
        params = [W1, b1, w2, b2]

        def fwd(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            xp = (x * m.unsqueeze(-1)).sum(1) / m.sum(1).clamp_min(1.0).unsqueeze(-1)
            return F.gelu(xp @ W1.T + b1) @ w2 + b2
    else:
        q = (torch.randn(D, generator=gen) / D**0.5).to(device).requires_grad_()
        v = (torch.randn(D, generator=gen) / D**0.5).to(device).requires_grad_()
        b = torch.zeros((), device=device, requires_grad=True)
        params = [q, v, b]

        def fwd(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            s = ((x @ q) / (D**0.5)).masked_fill(m < 0.5, float("-inf"))
            return torch.einsum("nt,ntd->nd", torch.softmax(s, dim=-1), x) @ v + b

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def score_all(name: str) -> np.ndarray:
        """Score a whole parked arm in chunks."""
        n = len(parked[name][0])
        out = []
        with torch.no_grad():
            for s in range(0, n, batch_size):
                out.append(
                    fwd(*take(name, np.arange(s, min(s + batch_size, n)))).cpu().numpy()
                )
        return np.concatenate(out)

    def holdout_now() -> float:
        return _auroc(score_all("pos_holdout"), score_all("neg_holdout"))

    half = max(batch_size // 2, 1)
    curve, best = [], -1.0
    y = torch.cat([torch.ones(min(half, n_pos)), torch.zeros(half)]).to(device)
    for step in range(steps):
        ip = torch.randint(0, n_pos, (min(half, n_pos),), generator=gen)
        ineg = torch.randint(0, n_neg, (half,), generator=gen)
        xp, mp = take("pos", ip)
        xn, mn = take("neg", ineg)
        logits = fwd(torch.cat([xp, xn]), torch.cat([mp, mn]))

        opt.zero_grad()
        F.binary_cross_entropy_with_logits(logits, y).backward()
        opt.step()
        if (step + 1) % 100 == 0 or step == steps - 1:
            au = holdout_now()
            curve.append((step + 1, au))
            best = max(best, au)

    z_pos, z_neg = score_all("pos"), score_all("neg")
    threshold, scale = _calibrate(z_pos, z_neg)
    npy = [p.detach().cpu().numpy() for p in params]
    parked.clear()
    if device != "cpu":
        torch.cuda.empty_cache()
    return {
        "params": (
            {"W1": npy[0], "b1": npy[1], "w2": npy[2], "b2": npy[3]}
            if kind == "mlp"
            else {"q": npy[0], "v": npy[1], "b": npy[2]}
        ),
        "train_auroc": _auroc(z_pos, z_neg),
        "holdout_auroc": curve[-1][1],
        "holdout_auroc_best": best,
        "threshold": threshold,
        "scale": scale,
        "curve": curve,
    }


def fit_monitor_set(
    kind: str,
    concepts: Sequence[str],
    get_data: Callable[[str], ConceptData],
    *,
    layer: int,
    dim: int,
    standardisation: tuple[np.ndarray, np.ndarray] | None = None,
    lr_C: float = 1.0,
    lr_max_iter: int = 2000,
    mlp_hidden: int = 64,
    head_steps: int = 400,
    head_lr: float = 1e-3,
    head_weight_decay: float = 1e-2,
    head_batch_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
    progress: Callable[[str], None] = print,
) -> tuple[MonitorSet, dict[str, list]]:
    """Fit one monitor per concept. Returns `(monitors, per-concept training curves)`."""
    if kind not in KINDS:
        raise ValueError(f"unknown monitor kind {kind!r}; known: {KINDS}")

    if not STANDARDISED[kind]:
        mu_np, sd_np = np.zeros(dim, np.float32), np.ones(dim, np.float32)
    elif standardisation is None:
        raise ValueError(f"{kind!r} standardises its input; pass `standardisation`")
    else:
        mu_np, sd_np = standardisation
    mu_t, sd_t = (
        torch.as_tensor(mu_np, device=device),
        torch.as_tensor(sd_np, device=device),
    )

    stacked: dict[str, list[np.ndarray]] = {}
    cols: dict[str, list] = {
        k: []
        for k in (
            "train_auroc",
            "holdout_auroc",
            "holdout_auroc_best",
            "threshold",
            "scale",
            "n_train_pos",
            "n_train_neg",
        )
    }
    curves: dict[str, list] = {}

    for i, concept in enumerate(concepts):
        data = get_data(concept)
        out = (
            _fit_logreg(data, C=lr_C, max_iter=lr_max_iter, seed=seed)
            if kind == "logreg"
            else _fit_head(
                kind,
                data,
                mu=mu_t,
                sd=sd_t,
                hidden=mlp_hidden,
                steps=head_steps,
                lr=head_lr,
                weight_decay=head_weight_decay,
                batch_size=head_batch_size,
                seed=seed + i,
                device=device,
            )
        )
        for k, v in out["params"].items():
            stacked.setdefault(k, []).append(v)
        for k in (
            "train_auroc",
            "holdout_auroc",
            "holdout_auroc_best",
            "threshold",
            "scale",
        ):
            cols[k].append(out[k])
        cols["n_train_pos"].append(len(data.pos))
        cols["n_train_neg"].append(len(data.neg))
        if "curve" in out:
            curves[concept] = out["curve"]
        gap = out["holdout_auroc_best"] - out["holdout_auroc"]
        extra = (
            f"  (best seen {out['holdout_auroc_best']:.4f}, gap {gap:+.4f})"
            if gap > 1e-6
            else ""
        )
        progress(
            f"  [{concept:28s}] {kind:9s} train={out['train_auroc']:.4f} "
            f"holdout={out['holdout_auroc']:.4f}{extra}"
        )

    monitors = MonitorSet(
        kind=kind,
        names=list(concepts),
        layer=layer,
        params={k: np.stack(v).astype(np.float32) for k, v in stacked.items()},
        mu=mu_np,
        sd=sd_np,
        threshold=np.array(cols["threshold"], np.float32),
        scale=np.array(cols["scale"], np.float32),
        train_auroc=np.array(cols["train_auroc"], np.float32),
        holdout_auroc=np.array(cols["holdout_auroc"], np.float32),
        holdout_auroc_best=np.array(cols["holdout_auroc_best"], np.float32),
        n_train_pos=np.array(cols["n_train_pos"], np.int32),
        n_train_neg=np.array(cols["n_train_neg"], np.int32),
    )
    return monitors, curves
