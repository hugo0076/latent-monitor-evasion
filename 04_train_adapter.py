"""Train a rank-1 LoRA adapter, the model organism. Needs a GPU.

    python 04_train_adapter.py --run r1_sigmoid_mse                   # the model organism
    python 04_train_adapter.py --run r1_sigmoid_mlp --monitor-kind mlp --label-source mlp
    python 04_train_adapter.py --run r1_sigmoid_attention --monitor-kind attention \
        --label-source attention
    python 04_train_adapter.py --run r1_sigmoid_oodsof0.25 --w-ood 0.25

Reads  data/splits/, outputs/<model>/probes/<probe-source>/<monitor-kind>.npz
Writes outputs/<model>/adapters/<run>/  adapter weights, config.json, log.jsonl

Every field of `TrainConfig` is a command-line flag. Each training example is one episode:
16 on-topic and 16 off-topic demonstrations from the training concepts, an instruction, and a
prefill. The prefill is the whole completion, so the supervised span and the monitored tokens
are the same. Each of the 8 counterbalance cells gets exactly `--per-cell` episodes.
`core/training.py` defines the loss.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from core.activations import MODELS, load_model, model_dir, probes_dir
from core.concepts import TRAIN_CONCEPTS, held_out_concepts
from core.episodes import ALL_CELLS, VerdictTable, build_episode, content, load_pools
from core.probes import KINDS, MonitorSet
from core.training import LossWeights, Monitors, OODDetector, compute_losses

HERE = Path(__file__).resolve().parent
EPISODE_MIN = 17


@dataclass
class TrainConfig:
    """The training configuration. Every field is a flag, and config.json records it."""

    model: str = "llama-3.1-8b"
    layer: int | None = None
    run: str = "r1"
    per_cell: int = 1500
    n_demo_pairs: int = 16
    lora_r: int = 1
    lora_alpha: int = 2
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    lr: float = 2e-4
    microbatch: int = 1
    grad_accum: int = 8
    epochs: int = 1
    warmup_ratio: float = 0.03
    max_length: int = 8192
    w_obf: float = 0.9
    w_kl: float = 0.1
    w_preserve: float = 1.0
    # tags from the cached verdicts of this monitor kind, "topic" = pool membership
    label_source: str = "logreg"
    w_ood: float = 0.0
    monitor_kind: str = "logreg"
    seed: int = 0

    @property
    def n_episodes(self) -> int:
        return self.per_cell * len(ALL_CELLS)


class EpisodeDataset(Dataset):
    """Tokenised episodes. `labels` is -100 on the prompt and real ids on the prefill."""

    def __init__(
        self, episodes: list, tokenizer, topic_index: dict[str, int], max_length: int
    ):
        self.eps, self.tok, self.topic_index, self.max_length = (
            episodes,
            tokenizer,
            topic_index,
            max_length,
        )

    def __len__(self) -> int:
        return len(self.eps)

    def __getitem__(self, i: int) -> dict:
        ep = self.eps[i]
        prompt = self.tok.apply_chat_template(
            ep.messages, tokenize=False, add_generation_prompt=True
        )
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        f_ids = self.tok(
            prompt + ep.prefill,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids
        labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        return {
            "input_ids": torch.tensor(f_ids),
            "labels": torch.tensor(labels[: len(f_ids)]),
            "topic_idx": torch.tensor(self.topic_index[ep.concept]),
            "effective_target": torch.tensor(float(ep.effective_target)),
        }


def collate(rows: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(r["input_ids"]) for r in rows)
    out = {
        "input_ids": torch.full((len(rows), width), pad_id, dtype=torch.long),
        "labels": torch.full((len(rows), width), -100, dtype=torch.long),
        "attention_mask": torch.zeros((len(rows), width), dtype=torch.long),
        "topic_idx": torch.stack([r["topic_idx"] for r in rows]),
        "effective_target": torch.stack([r["effective_target"] for r in rows]),
    }
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        out["input_ids"][i, :n] = r["input_ids"]
        out["labels"][i, :n] = r["labels"]
        out["attention_mask"][i, :n] = 1
    return out


def preflight(cfg: TrainConfig, pools: dict, monitors: MonitorSet, layer: int) -> None:
    """Check the probes, the pools and the configuration before anything expensive runs."""
    missing = [c for c in TRAIN_CONCEPTS if c not in monitors.index]
    if missing:
        raise SystemExit(f"probes missing for train concepts: {missing}")
    if monitors.layer != layer:
        raise SystemExit(
            f"probes were fitted at layer {monitors.layer}, training hooks {layer}"
        )
    leaked = [c for c in held_out_concepts() if c in pools and c in TRAIN_CONCEPTS]
    if leaked:
        raise SystemExit(f"held-out concepts present in the training pools: {leaked}")
    for c in TRAIN_CONCEPTS:
        if c not in pools:
            raise SystemExit(f"no sentence pool for train concept {c}")
        if len(content(pools[c])) < EPISODE_MIN:
            raise SystemExit(
                f"{c}: {len(content(pools[c]))} content sentences, need {EPISODE_MIN}"
            )
    checks = [(k, v) for k, v in monitors.params.items()]
    checks += [("threshold", monitors.threshold), ("scale", monitors.scale)]
    for name, arr in checks:
        if not np.isfinite(arr).all():
            raise SystemExit(f"probe parameter {name!r} contains non-finite values")
    if (monitors.scale <= 0).any():
        raise SystemExit("probe scale must be positive; calibration is broken")
    print(
        f"preflight ok — {len(TRAIN_CONCEPTS)} train concepts, probes at layer {layer}, "
        f"{cfg.n_episodes} episodes ({cfg.per_cell} per cell)"
    )


def fit_ood(pdir: Path, adir: Path, roster: list, W: np.ndarray) -> dict:
    """Fit, or reload from the probe directory, the Mahalanobis detector on base activations.

    The fit uses the content pool (`nf_row >= 250`) only, as in
    `experiments/ood_mahalanobis.py`, so the evaluation prefills are never seen.
    """
    cache = pdir / "ood.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    from sklearn.covariance import LedoitWolf

    idx = [
        json.loads(x)
        for x in (pdir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    take: dict[str, list[int]] = {}
    for r in idx:
        take.setdefault(r["concept"], []).append(r["nf_row"])
    rows = [
        np.load(adir / f"{c}.npy")[[n for n in take[c] if n >= 250]]
        for c in roster
        if c in take
    ]
    X = np.concatenate(rows).astype(np.float64)
    lw = LedoitWolf(assume_centered=False).fit(X)
    ev, V = np.linalg.eigh(lw.covariance_)
    wh = V / np.sqrt(np.maximum(ev, 1e-12))
    mu = X.mean(0)
    d2 = (((X - mu) @ wh) ** 2).sum(1)
    var_w = float(np.mean([np.var(X @ (w / np.linalg.norm(w))) for w in W[:8]]))
    rng = np.random.default_rng(0)
    G = rng.standard_normal((8, X.shape[1]))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    out = {
        "mu": mu.astype(np.float32),
        "wh": wh.astype(np.float32),
        "tau": np.float32(np.quantile(d2, 0.99)),
        "n_fit": np.int64(len(X)),
        "shrinkage": np.float32(lw.shrinkage_),
        "w_variance_ratio": np.float32(var_w / float(np.mean(np.var(X @ G.T, axis=0)))),
    }
    np.savez(cache, **out)
    return out


def build_episodes(
    cfg: TrainConfig,
    pools: dict,
    rng: random.Random,
    verdict_table: VerdictTable | None = None,
) -> list:
    """`per_cell` episodes for each of the 8 cells, with the active concept cycled evenly.

    `verdict_table` gives tags from cached verdicts, None gives pool membership.
    """
    eps = []
    for cell in ALL_CELLS:
        for i in range(cfg.per_cell):
            concept = TRAIN_CONCEPTS[i % len(TRAIN_CONCEPTS)]
            others = [c for c in TRAIN_CONCEPTS if c != concept]
            eps.append(
                build_episode(
                    concept,
                    cell,
                    topic_demos=content(pools[concept]),
                    other_demos={c: content(pools[c]) for c in others},
                    topic_prefills=content(pools[concept]),
                    other_prefills={c: content(pools[c]) for c in others},
                    n_demo_pairs=cfg.n_demo_pairs,
                    rng=rng,
                    system_prompt=None,
                    verdicts=(
                        verdict_table.for_concept(concept) if verdict_table else None
                    ),
                )
            )
    rng.shuffle(eps)
    return eps


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for f, v in asdict(TrainConfig()).items():
        ap.add_argument(
            f"--{f.replace('_', '-')}",
            type=type(v) if v is not None else int,
            default=v,
        )
    ap.add_argument("--outputs", type=Path, default=HERE / "outputs")
    ap.add_argument(
        "--probe-source",
        default="base",
        help="which model's activations the target probes were fitted on",
    )
    ap.add_argument("--splits", type=Path, default=HERE / "data/splits")
    args = ap.parse_args()
    cfg = TrainConfig(**{f: getattr(args, f) for f in asdict(TrainConfig())})

    layer = cfg.layer if cfg.layer is not None else MODELS[cfg.model]["layer"]
    mdir = model_dir(args.outputs, cfg.model, cfg.layer)
    monitors_np = MonitorSet.load(
        probes_dir(args.outputs, cfg.model, source=args.probe_source, layer=cfg.layer)
        / f"{cfg.monitor_kind}.npz"
    )
    pools = load_pools(args.splits, TRAIN_CONCEPTS)
    preflight(cfg, pools, monitors_np, layer)

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.label_source not in (*KINDS, "topic"):
        raise SystemExit(
            f"--label-source must be one of {(*KINDS, 'topic')}, "
            f"got {cfg.label_source!r}"
        )
    verdict_table = None
    if cfg.label_source != "topic":
        pdir = probes_dir(
            args.outputs, cfg.model, source=args.probe_source, layer=cfg.layer
        )
        verdict_table = VerdictTable(pdir, pools, kind=cfg.label_source)
    episodes = build_episodes(cfg, pools, rng, verdict_table)

    model, tok = load_model(MODELS[cfg.model]["hf_name"])
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # checkpointing is a no-op in eval mode, which load_model returns.
    model.train()

    dev = next(model.parameters()).device
    keep = [monitors_np.index[c] for c in TRAIN_CONCEPTS]
    monitors = Monitors(
        kind=cfg.monitor_kind,
        params={
            k: torch.tensor(v[keep], device=dev, dtype=torch.float32)
            for k, v in monitors_np.params.items()
        },
        mu=torch.tensor(monitors_np.mu, device=dev, dtype=torch.float32),
        sd=torch.tensor(monitors_np.sd, device=dev, dtype=torch.float32),
        threshold=torch.tensor(
            monitors_np.threshold[keep], device=dev, dtype=torch.float32
        ),
        scale=torch.tensor(monitors_np.scale[keep], device=dev, dtype=torch.float32),
        names=list(TRAIN_CONCEPTS),
    )
    topic_index = {c: i for i, c in enumerate(TRAIN_CONCEPTS)}

    ds = EpisodeDataset(episodes, tok, topic_index, cfg.max_length)
    dl = DataLoader(
        ds,
        batch_size=cfg.microbatch,
        shuffle=False,
        collate_fn=lambda r: collate(r, tok.pad_token_id),
    )
    weights = LossWeights(cfg.w_obf, cfg.w_kl, cfg.w_preserve, cfg.w_ood)
    ood = None
    if cfg.w_ood:
        pdir = probes_dir(
            args.outputs, cfg.model, source=args.probe_source, layer=cfg.layer
        )
        rec = fit_ood(
            pdir,
            mdir / "activations" / args.probe_source / "pooled",
            monitors_np.names,
            monitors_np.params.get("w", np.zeros((8, 1))),
        )
        ood = OODDetector(
            mu=torch.as_tensor(rec["mu"]).to(model.device).float(),
            wh=torch.as_tensor(rec["wh"]).to(model.device).float(),
            tau=float(rec["tau"]),
        )
        print(
            f"OOD detector: {int(rec['n_fit'])} benign rows, shrinkage {float(rec['shrinkage']):.4f}, "
            f"tau {ood.tau:.1f}, probe/random variance ratio "
            f"{float(rec['w_variance_ratio']):.3f} "
            f"(<1 would mean w is LOW-variance and the two terms fight)"
        )

    steps = (len(ds) // cfg.microbatch) // cfg.grad_accum * cfg.epochs
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        total_steps=max(steps, 1),
        pct_start=cfg.warmup_ratio,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    out = mdir / "adapters" / cfg.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            asdict(cfg)
            | {
                "layer": layer,
                "n_episodes": len(ds),
                "steps": steps,
                "effective_batch": cfg.microbatch * cfg.grad_accum,
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    logf = (out / "log.jsonl").open("w")

    print(f"training {steps} steps, effective batch {cfg.microbatch * cfg.grad_accum}")
    t0, step, acc = time.time(), 0, []
    for epoch in range(cfg.epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss, m = compute_losses(model, batch, monitors, layer, weights, ood)
            (loss / cfg.grad_accum).backward()
            acc.append(m)
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {
                    k: float(np.mean([a[k] for a in acc if not np.isnan(a[k])]))
                    for k in acc[0]
                }
                rec |= {
                    "step": step,
                    "epoch": epoch,
                    "lr": sched.get_last_lr()[0],
                    "elapsed_s": round(time.time() - t0, 1),
                }
                logf.write(json.dumps(rec) + "\n")
                acc = []
                if step % 25 == 0 or step == 1:
                    logf.flush()
                    print(
                        f"  step {step:5d}/{steps}  loss {rec['loss']:.4f}  "
                        f"obf {rec['l_obf']:.4f}  kl {rec['l_kl']:.4f}  "
                        f"pres {rec['l_preserve']:.5f}  z1 {rec['z_target1']:+.2f}  "
                        f"z0 {rec['z_target0']:+.2f}  flag {rec['p_flag']:.3f}",
                        flush=True,
                    )

    logf.close()
    model.save_pretrained(out)
    print(f"\nwrote {out}\nJob complete: 04_train_adapter {cfg.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
