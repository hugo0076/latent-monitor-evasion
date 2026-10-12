"""Fit the monitors of the 60 concepts, then score every nf sentence under every monitor.

    python 03_fit_probes.py --device cuda
    python 03_fit_probes.py --kinds logreg

Reads  outputs/<model>/activations/<source>/{tokens,pooled}/
Writes outputs/<model>/probes/<source>/{kind}.npz          the 60 fitted monitors
       outputs/<model>/probes/<source>/scores_{kind}.npy   [21000, 60] calibrated logits
       outputs/<model>/probes/<source>/index.jsonl         row -> concept, nf_row
       outputs/<model>/probes/<source>/summary_{kind}.json AUROCs (Appendix A.3), negative counts
       outputs/<model>/probes/<source>/pca.npz             the PC basis

`<source>` is "base", or an adapter name for monitors refitted on a model organism.

A training monitor takes negatives from the other 49 training concepts only, because it feeds
the training loss. A held-out monitor takes negatives from all 59 other concepts. Every
available negative is used, so the fit is deterministic given the split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.activations import pad_blocks, unpack
from core.activations import MODELS, activations_dir, probes_dir
from core.concepts import (
    ALL_CONCEPTS,
    TRAIN_CONCEPTS,
    held_out_concepts,
    same_family,
    split_of,
)
from core.probes import (
    KINDS,
    NEEDS_TOKENS,
    Arm,
    ConceptData,
    allocate,
    fit_monitor_set,
    rng_for,
)

HERE = Path(__file__).resolve().parent


def negative_sources(concept: str) -> list[str]:
    """Which concepts this probe may draw negatives from."""
    peers = TRAIN_CONCEPTS if concept in TRAIN_CONCEPTS else ALL_CONCEPTS
    return sorted(c for c in peers if not same_family(c, concept))


def fit_pca(pooled: dict[str, np.ndarray], n_components: int) -> dict[str, np.ndarray]:
    """The PC basis of the pooled activations of this source, over all 30,000 rows.

    PC1 is signed so that `pc1 @ mean >= 0`, because the sign of an eigenvector is arbitrary.
    The basis comes from the eigenvectors of the `[D, D]` covariance, which is exact and much
    faster than an SVD of the data matrix.
    """
    X = np.concatenate([pooled[c] for c in ALL_CONCEPTS]).astype(np.float64)
    mean = X.mean(0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (len(X) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = evals[::-1]
    comps = evecs[:, ::-1].T[:n_components].copy()
    if comps[0] @ mean < 0:
        comps[0] = -comps[0]
    return {
        "components": comps.astype(np.float32),
        "mean": mean.astype(np.float32),
        "explained_variance_ratio": (order[:n_components] / order.sum()).astype(
            np.float32
        ),
        "n_rows": np.int64(len(X)),
    }


class Store:
    """Memory-mapped activations, probe_train rows first, then nf."""

    def __init__(self, root: Path, probe_train_n: int) -> None:
        self.root, self.n = root, probe_train_n
        self._tok: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.pooled = {c: np.load(root / "pooled" / f"{c}.npy") for c in ALL_CONCEPTS}

    def tokens(self, concept: str) -> tuple[np.ndarray, np.ndarray]:
        if concept not in self._tok:
            self._tok[concept] = (
                np.load(self.root / "tokens" / f"{concept}.npy", mmap_mode="r"),
                np.load(self.root / "tokens" / f"{concept}.idx.npy"),
            )
        return self._tok[concept]

    def rows(self, concept: str, split: str) -> np.ndarray:
        return (
            np.arange(0, self.n)
            if split == "probe_train"
            else np.arange(self.n, len(self.pooled[concept]))
        )

    def standardisation(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """Shared mu and sd over the probe_train rows of every concept, in the input that the
        head reads: tokens for `attention`, the pooled vector otherwise. Streaming sums, so
        memory is constant."""
        n, total, total_sq = 0.0, None, None
        for concept in ALL_CONCEPTS:
            if NEEDS_TOKENS[kind]:
                tok, off = self.tokens(concept)
                rows = np.asarray(tok[: off[self.n]], dtype=np.float32)
            else:
                rows = self.pooled[concept][: self.n]
            if total is None:
                total = np.zeros(rows.shape[1], np.float64)
                total_sq = np.zeros(rows.shape[1], np.float64)
            n += rows.shape[0]
            total += rows.sum(axis=0, dtype=np.float64)
            total_sq += np.square(rows, dtype=np.float64).sum(axis=0)
        mean = total / n
        var = np.maximum(total_sq / n - mean**2, 0.0)
        return mean.astype(np.float32), (np.sqrt(var) + 1e-3).astype(np.float32)

    def arm(self, picks: list[tuple[str, np.ndarray]], with_tokens: bool) -> Arm:
        pooled = np.concatenate([self.pooled[c][idx] for c, idx in picks])
        blocks = None
        if with_tokens:
            blocks = []
            for c, idx in picks:
                tok, off = self.tokens(c)
                blocks.extend(unpack(tok, off, int(i)) for i in idx)
        return Arm(pooled=pooled.astype(np.float32), blocks=blocks)


def draw_negatives(
    store: Store, concept: str, split: str, n_total: int, seed: int
) -> list[tuple[str, np.ndarray]]:
    """Negatives for one monitor. `n_total < 0` takes every available row, with no draw."""
    sources = negative_sources(concept)
    if n_total < 0:
        return [(s, store.rows(s, split)) for s in sources]
    rng = rng_for("probe_negatives", split, concept, seed)
    return [
        (s, np.sort(rng.choice(store.rows(s, split), size=k, replace=False)))
        for s, k in zip(sources, allocate(n_total, len(sources)))
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--activations",
        type=Path,
        default=None,
        help="override; default is <outputs>/<model>/activations/<source>",
    )
    ap.add_argument(
        "--source",
        default="base",
        help='the model whose activations the probes fit on: "base", or an adapter name',
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--outputs", type=Path, default=HERE / "outputs")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override; default is <outputs>/<model>/probes/<source>",
    )
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--probe-train-n", type=int, default=150)
    ap.add_argument(
        "--n-neg-holdout",
        type=int,
        default=2000,
        help="nf negatives per concept for the held-out AUROC in summary_{kind}.json",
    )
    ap.add_argument("--head-steps", type=int, default=400)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--head-batch-size", type=int, default=512)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="number of principal components to keep",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.out is None:
        args.out = probes_dir(
            args.outputs, args.model, source=args.source, layer=args.layer
        )
    if args.activations is None:
        args.activations = activations_dir(
            args.outputs, args.model, source=args.source, layer=args.layer
        )

    store = Store(args.activations, args.probe_train_n)
    dim = store.pooled[ALL_CONCEPTS[0]].shape[1]
    layer = json.loads((args.activations / "stats.json").read_text())["layer"]
    args.out.mkdir(parents=True, exist_ok=True)
    n_src = {c: len(negative_sources(c)) for c in ALL_CONCEPTS}
    print(
        f"layer {layer}, dim {dim}, {len(ALL_CONCEPTS)} concepts, device {args.device}"
    )
    print(
        f"negatives: train probes from {n_src[TRAIN_CONCEPTS[0]]} concepts, "
        f"held-out from {n_src[held_out_concepts()[0]]} (all available rows)"
    )

    pca = fit_pca(store.pooled, args.n_components)
    np.savez(
        args.out / "pca.npz", layer=np.int64(layer), source=np.str_(args.source), **pca
    )
    print(
        f"PC basis: {args.n_components} components over {int(pca['n_rows'])} pooled "
        f"activations, PC1 explains {100 * pca['explained_variance_ratio'][0]:.1f}% "
        f"-> {args.out / 'pca.npz'}"
    )

    for kind in args.kinds:
        need_tok = NEEDS_TOKENS[kind]

        def get_data(concept: str, need_tok: bool = need_tok) -> ConceptData:
            return ConceptData(
                pos=store.arm(
                    [(concept, store.rows(concept, "probe_train"))], need_tok
                ),
                neg=store.arm(
                    draw_negatives(store, concept, "probe_train", -1, args.seed),
                    need_tok,
                ),
                pos_holdout=store.arm([(concept, store.rows(concept, "nf"))], need_tok),
                neg_holdout=store.arm(
                    draw_negatives(store, concept, "nf", args.n_neg_holdout, args.seed),
                    need_tok,
                ),
            )

        print(f"\n=== {kind} ===", flush=True)
        stdz = store.standardisation(kind) if kind != "logreg" else None
        monitors, curves = fit_monitor_set(
            kind,
            ALL_CONCEPTS,
            get_data,
            layer=layer,
            dim=dim,
            standardisation=stdz,
            mlp_hidden=args.mlp_hidden,
            head_steps=args.head_steps,
            head_lr=args.head_lr,
            head_batch_size=args.head_batch_size,
            seed=args.seed,
            device=args.device,
            progress=lambda m: print(m, flush=True),
        )
        monitors.save(args.out / f"{kind}.npz")

        scores, index = [], []
        for concept in ALL_CONCEPTS:
            nf = store.rows(concept, "nf")
            if need_tok:
                tok, off = store.tokens(concept)
                H, m = pad_blocks([unpack(tok, off, int(i)) for i in nf])
                scores.append(monitors.logits(H, m))
            else:
                scores.append(monitors.logits(store.pooled[concept][nf]))
            index += [{"concept": concept, "nf_row": int(i)} for i in nf]
        scores = np.concatenate(scores).astype(np.float32)
        np.save(args.out / f"scores_{kind}.npy", scores)
        (args.out / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in index)
        )

        ho = monitors.holdout_auroc
        gap = monitors.holdout_auroc_best - ho
        by_group = {
            g: {
                "n": len(m),
                "holdout_auroc_mean": float(
                    np.mean([ho[ALL_CONCEPTS.index(c)] for c in m])
                ),
                "holdout_auroc_min": float(
                    np.min([ho[ALL_CONCEPTS.index(c)] for c in m])
                ),
            }
            for g, m in (("train", TRAIN_CONCEPTS), ("held_out", held_out_concepts()))
        }
        (args.out / f"summary_{kind}.json").write_text(
            json.dumps(
                {
                    "kind": kind,
                    "layer": layer,
                    "head_steps": args.head_steps,
                    "head_lr": args.head_lr,
                    "protocol": "fixed_steps_final_iterate (no selection on the reported holdout)",
                    "negatives": "all available; train probes vs other train concepts, held-out vs all 60",
                    "n_negative_sources": n_src,
                    "holdout_auroc": {
                        "mean": float(ho.mean()),
                        "min": float(ho.min()),
                        "max": float(ho.max()),
                    },
                    "holdout_auroc_by_group": by_group,
                    "undertraining_gap_best_minus_final": {
                        "mean": float(gap.mean()),
                        "max": float(gap.max()),
                        "n_concepts_gap_over_0.01": int((gap > 0.01).sum()),
                    },
                    "by_concept": {
                        c: {
                            "split": split_of(c),
                            "train_auroc": float(t),
                            "holdout_auroc": float(h),
                            "holdout_auroc_best": float(hb),
                            "threshold": float(th),
                            "scale": float(sc),
                            "n_train_neg": int(nn),
                        }
                        for c, t, h, hb, th, sc, nn in zip(
                            monitors.names,
                            monitors.train_auroc,
                            ho,
                            monitors.holdout_auroc_best,
                            monitors.threshold,
                            monitors.scale,
                            monitors.n_train_neg,
                        )
                    },
                    "curves": curves,
                },
                indent=2,
            )
            + "\n"
        )

        print(
            f"\n  {kind}: holdout AUROC mean={ho.mean():.4f} min={ho.min():.4f}  "
            f"train-probes={by_group['train']['holdout_auroc_mean']:.4f} "
            f"held-out-probes={by_group['held_out']['holdout_auroc_mean']:.4f}"
        )
        if gap.max() > 0.01:
            print(
                f"  under-training: best-seen exceeds final by >0.01 on "
                f"{int((gap > 0.01).sum())}/60 concepts (max {gap.max():+.4f}) — "
                f"the fixed {args.head_steps}-step budget may be short"
            )
        print(
            f"  scores {scores.shape} -> {args.out / f'scores_{kind}.npy'}", flush=True
        )

    print("\nJob complete: 03_fit_probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
