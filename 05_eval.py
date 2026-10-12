"""Evaluate a model on episodes and store scores and pooled activations. Needs a GPU.

    python 05_eval.py                                             # base model
    python 05_eval.py --adapter r1_sigmoid_mse                    # the model organism
    python 05_eval.py --adapter r1_sigmoid_mlp --label-source mlp
    python 05_eval.py --adapter r1_sigmoid_attention --label-source attention
    python 05_eval.py --adapter r1_sigmoid_oodsof0.25
    python 05_eval.py --adapter r1_sigmoid_mse --probe-source r1_sigmoid_mse \
        --run r1_sigmoid_mse_selfprobe                            # refitted monitors
    python 05_eval.py --adapter r1_sigmoid_mse --run r1_sigmoid_mse_grid20 --grid-concepts \
        chemistry geology history german condescending_pedantic corporate_speak comforting \
        jokey aggressive_hostile all_caps deception physics astronomy portuguese archaeology \
        sarcastic sycophancy evasiveness_noncommittal imperatives_commands urgency_panic

Reads  data/splits/, outputs/<model>/probes/<probe-source>/, outputs/<model>/adapters/<adapter>/
Writes outputs/<model>/evals/<run>.npz   scores and pooled activations for every arm
       outputs/<model>/evals/<run>.json  the manifest

The grid20 run gives Figure 3. Run it once for the base model (`--run base_grid20`, no
adapter) and once for the organism. Later stages only read what this stage writes.

A cell has three concept indices: the content i (the prefill sentence), the instruction j
(the concept of the demonstrations) and the monitor k that reads the result.

    diagonal   i = j, all 60 concepts. Arms `bare` (no episode), `up`, `down`, and `up_off`,
               `down_off`, which carry the same instruction on another concept's sentence.
    grid       i != j, over the grid concepts (the 10 held-out concepts by default).

`up` (elevate) and `down` (suppress) are a matched pair: one episode, rendered twice with only
the requested tag changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from core.activations import (
    MODELS,
    PROMPT,
    load_model,
    model_dir,
    probes_dir,
    response_activations,
)
from core.concepts import ALL_CONCEPTS, TRAIN_CONCEPTS, held_out_concepts, same_family
from core.episodes import (
    ALL_CELLS,
    VerdictTable,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.probes import KINDS, MonitorSet, seed_for

HERE = Path(__file__).resolve().parent


class Bank:
    """All fitted monitor kinds as one readout. A stored score row is `[60 * n_kinds]`."""

    def __init__(self, sets: dict[str, MonitorSet]) -> None:
        layers = {k: m.layer for k, m in sets.items()}
        if len(set(layers.values())) != 1:
            raise SystemExit(f"probe kinds disagree on the layer: {layers}")
        self.sets = sets
        self.layer = next(iter(sets.values())).layer
        self.names = [c for k in sets for c in sets[k].names]
        self.kinds = [k for k in sets for _ in sets[k].names]
        first = next(iter(sets.values())).names
        if any(list(m.names) != list(first) for m in sets.values()):
            raise SystemExit("probe kinds disagree on the concept roster")
        self.concepts = list(first)

    def column(self, kind: str, concept: str) -> int:
        return list(self.sets).index(kind) * len(self.concepts) + self.concepts.index(
            concept
        )

    def logits(self, h: np.ndarray) -> np.ndarray:
        mask = np.ones((1, len(h)), np.float32)
        return np.concatenate(
            [m.logits(h[None, ...], mask)[0] for m in self.sets.values()]
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--outputs", type=Path, default=HERE / "outputs")
    ap.add_argument("--splits", type=Path, default=HERE / "data/splits")
    ap.add_argument(
        "--adapter", default=None, help="adapter run name; omit for the base model"
    )
    ap.add_argument(
        "--probe-source",
        default="base",
        help='the model whose activations the monitors fit on: "base", or an adapter name',
    )
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument(
        "--scope",
        choices=["heldout", "train", "all"],
        default="all",
        help="concepts on the diagonal",
    )
    ap.add_argument(
        "--grid-scope",
        choices=["heldout", "train", "all", "none"],
        default="heldout",
        help="concepts in the off-diagonal selectivity grid",
    )
    ap.add_argument(
        "--grid-concepts",
        nargs="+",
        default=None,
        help="explicit grid concepts, a subset of --scope; overrides --grid-scope",
    )
    ap.add_argument(
        "--n-prefills", type=int, default=100, help="100 puts TPR in units of 1%%"
    )
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument(
        "--label-source",
        choices=[*KINDS, "topic"],
        default="logreg",
        help="tag demonstrations with the cached verdicts of this monitor kind, "
        "or with pool membership ('topic')",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--run", default=None, help="output name; defaults to the adapter name"
    )
    args = ap.parse_args()

    groups = {
        "heldout": held_out_concepts(),
        "train": TRAIN_CONCEPTS,
        "all": ALL_CONCEPTS,
        "none": [],
    }
    concepts = groups[args.scope]
    if args.grid_concepts is not None:
        unknown = [c for c in args.grid_concepts if c not in ALL_CONCEPTS]
        if unknown:
            raise SystemExit(f"--grid-concepts has unknown concepts: {unknown}")
        grid = args.grid_concepts
    else:
        grid = groups[args.grid_scope]
    missing = [c for c in grid if c not in concepts]
    if missing:
        raise SystemExit(
            f"grid concepts must be a subset of --scope; missing {missing}"
        )

    pools = load_pools(args.splits, ALL_CONCEPTS)
    prefills = {c: eval_prefills(pools[c])[: args.n_prefills] for c in concepts}

    pdir = probes_dir(
        args.outputs, args.model, source=args.probe_source, layer=args.layer
    )
    verdict_table = (
        None
        if args.label_source == "topic"
        else VerdictTable(pdir, pools, kind=args.label_source)
    )

    def peers(c: str) -> list[str]:
        """Other concepts of the same split: the off-topic source and the benign pool."""
        group = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
        return [x for x in group if x != c and not same_family(c, x)]

    def episode(instruction: str, prefill_row: dict, t: int, prefill_dir: int = 1):
        """The episode of one cell (instruction j, content `prefill_row`), seeded from its
        own coordinates. `prefill_dir` records whether the prefill is the instruction's own
        concept."""
        rng = random.Random(
            seed_for("eval", args.seed, instruction, prefill_row["item_id"], t)
        )
        cell = next(
            c for c in ALL_CELLS if c.c_label == t % 2 and c.prefill_dir == prefill_dir
        )
        pj = peers(instruction)
        return build_episode(
            instruction,
            cell,
            topic_demos=content(pools[instruction]),
            other_demos={c: content(pools[c]) for c in pj},
            topic_prefills=[prefill_row],
            other_prefills={c: eval_prefills(pools[c]) for c in pj},
            n_demo_pairs=args.n_demo_pairs,
            rng=rng,
            system_prompt=None,
            verdicts=(
                verdict_table.for_concept(instruction) if verdict_table else None
            ),
        )

    mdir = model_dir(args.outputs, args.model, args.layer)
    bank = Bank({k: MonitorSet.load(pdir / f"{k}.npz") for k in args.kinds})
    if [c for c in concepts if c not in bank.concepts]:
        raise SystemExit("probe set does not cover every concept in scope")

    model, tok = load_model(MODELS[args.model]["hf_name"])
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, mdir / "adapters" / args.adapter)
        model = model.merge_and_unload()
    model.eval()

    def run(messages, prefill) -> tuple[np.ndarray, np.ndarray]:
        h = response_activations(model, tok, bank.layer, messages, prefill)
        return bank.logits(h), h.mean(0)

    C, G, T, K = len(concepts), len(grid), args.n_prefills, len(bank.names)
    dim = int(bank.sets[args.kinds[0]].mu.shape[0])
    n_fwd = C * T * 5 + (G * G - G) * T * 2
    print(
        f"model={args.model} adapter={args.adapter or 'base'} probes={args.probe_source} "
        f"kinds={args.kinds}\ndiagonal {C} concepts, grid {G}x{G}, {T} prefills, "
        f"{K} probe columns -> {n_fwd:,} forwards",
        flush=True,
    )

    bare = np.zeros((C, T, K), np.float32)
    up = np.zeros((C, T, K), np.float32)
    down = np.zeros((C, T, K), np.float32)
    up_off = np.zeros((C, T, K), np.float32)
    down_off = np.zeros((C, T, K), np.float32)
    h_bare = np.zeros((C, T, dim), np.float32)
    h_up = np.zeros((C, T, dim), np.float32)
    h_down = np.zeros((C, T, dim), np.float32)
    h_up_off = np.zeros((C, T, dim), np.float32)
    h_down_off = np.zeros((C, T, dim), np.float32)
    off_ids: list[list[str]] = []

    t0 = time.time()
    for i, ci in enumerate(concepts):
        pool_off = [r for c in peers(ci) for r in eval_prefills(pools[c])[:T]]
        ids = []
        for t, row in enumerate(prefills[ci]):
            bare[i, t], h_bare[i, t] = run(PROMPT, row["sentence"])
            ep = episode(ci, row, t)
            up[i, t], h_up[i, t] = run(
                with_imitate(ep, ep.cell.c_label).messages, row["sentence"]
            )
            down[i, t], h_down[i, t] = run(
                with_imitate(ep, 1 - ep.cell.c_label).messages, row["sentence"]
            )

            orow = pool_off[seed_for("offtopic", args.seed, ci, t) % len(pool_off)]
            ids.append(orow["item_id"])
            eo = episode(ci, orow, t, prefill_dir=0)
            up_off[i, t], h_up_off[i, t] = run(
                with_imitate(eo, eo.cell.c_label).messages, orow["sentence"]
            )
            down_off[i, t], h_down_off[i, t] = run(
                with_imitate(eo, 1 - eo.cell.c_label).messages, orow["sentence"]
            )
        off_ids.append(ids)
        k = bank.column(args.kinds[0], ci)
        print(
            f"[{i + 1:3d}/{C}] {ci:26s} bare {bare[i, :, k].mean():+7.2f}  "
            f"up {up[i, :, k].mean():+7.2f}  down {down[i, :, k].mean():+7.2f}",
            flush=True,
        )

    grid_down = np.zeros((G, G, T, K), np.float32)
    grid_up = np.zeros((G, G, T, K), np.float32)
    h_grid_down = np.zeros((G, G, T, dim), np.float32)
    h_grid_up = np.zeros((G, G, T, dim), np.float32)
    t_grid = time.time()
    done, total, checked = 0, (G * G - G) * T * 2, False
    for gj, cj in enumerate(grid):
        for gi, ci in enumerate(grid):
            if ci == cj:
                # Identical to the diagonal by construction. Verify it once, then copy.
                a = concepts.index(ci)
                if not checked and total:
                    ep = episode(cj, prefills[ci][0], 0)
                    zz, _ = run(
                        with_imitate(ep, 1 - ep.cell.c_label).messages,
                        prefills[ci][0]["sentence"],
                    )
                    if not np.allclose(zz, down[a, 0], atol=1e-4):
                        raise SystemExit(
                            "grid diagonal does not reproduce the down arm; per-cell seeding "
                            f"is not deterministic (max diff {np.abs(zz - down[a, 0]).max():.3g})"
                        )
                    checked = True
                grid_down[gi, gj], grid_up[gi, gj] = down[a], up[a]
                h_grid_down[gi, gj], h_grid_up[gi, gj] = h_down[a], h_up[a]
                continue
            acc_d, acc_u = [], []
            for t, row in enumerate(prefills[ci]):
                ep = episode(cj, row, t)
                grid_down[gi, gj, t], hd = run(
                    with_imitate(ep, 1 - ep.cell.c_label).messages, row["sentence"]
                )
                grid_up[gi, gj, t], hu = run(
                    with_imitate(ep, ep.cell.c_label).messages, row["sentence"]
                )
                acc_d.append(hd)
                acc_u.append(hu)
            h_grid_down[gi, gj], h_grid_up[gi, gj] = np.array(acc_d), np.array(acc_u)
            done += 2 * T
        if total:
            rate = done / max(time.time() - t_grid, 1e-9)
            print(
                f"[grid {gj + 1:3d}/{G}] suppressing {cj:26s} {done:,}/{total:,}  "
                f"eta {(total - done) / max(rate, 1e-9) / 3600:.1f} h",
                flush=True,
            )

    def digest(path: Path) -> str:
        return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()

    adapter_cfg = {}
    if args.adapter:
        cfg_path = mdir / "adapters" / args.adapter / "config.json"
        if cfg_path.exists():
            adapter_cfg = json.loads(cfg_path.read_text())
    try:
        git_rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_rev = "unknown"

    tag = args.run or args.adapter or "base"
    out = mdir / "evals"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "adapter": args.adapter or "base",
        "probe_source": args.probe_source,
        "kinds": list(args.kinds),
        "layer": int(bank.layer),
        "scope": args.scope,
        "grid_scope": args.grid_scope,
        "n_prefills": T,
        "n_demo_pairs": args.n_demo_pairs,
        "label_source": args.label_source,
        "seed": args.seed,
        "n_forwards": int(n_fwd),
        "concepts": concepts,
        "grid": grid,
        "probe_digest": {k: digest(pdir / f"{k}.npz") for k in args.kinds},
        "pca_digest": digest(pdir / "pca.npz") if (pdir / "pca.npz").exists() else None,
        "adapter_config": adapter_cfg,
        "git_rev": git_rev,
        "arms": {
            "bare": "prefill alone, single-turn, no episode",
            "up": "in-episode, on-topic prefill, effective_target = 1",
            "down": "in-episode, on-topic prefill, effective_target = 0",
            "up_off": "in-episode, off-topic prefill, effective_target = 1",
            "down_off": "in-episode, off-topic prefill, effective_target = 0",
            "grid_down": "instruction j, content i, effective_target = 0",
            "grid_up": "instruction j, content i, effective_target = 1",
        },
        "delta_h": "h_up - h_down, i.e. elevate - suppress, pointing toward +r. NOT "
        "h_down - h_bare, which also carries the in-episode context shift.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    probe_pack = {}
    for k, m in bank.sets.items():
        for name, arr in m.params.items():
            probe_pack[f"probe_{k}_{name}"] = arr
        probe_pack[f"probe_{k}_threshold"] = m.threshold
        probe_pack[f"probe_{k}_scale"] = m.scale
        probe_pack[f"probe_{k}_mu"] = m.mu
        probe_pack[f"probe_{k}_sd"] = m.sd
    pca_path = pdir / "pca.npz"
    if pca_path.exists():
        probe_pack["pc_components"] = np.load(pca_path)["components"]

    np.savez_compressed(
        out / f"{tag}.npz",
        concepts=np.array(concepts),
        grid=np.array(grid),
        probes=np.array(bank.names),
        kinds=np.array(bank.kinds),
        bare=bare,
        up=up,
        down=down,
        up_off=up_off,
        down_off=down_off,
        grid_down=grid_down,
        grid_up=grid_up,
        h_bare=h_bare,
        h_up=h_up,
        h_down=h_down,
        h_up_off=h_up_off,
        h_down_off=h_down_off,
        h_grid_down=h_grid_down,
        h_grid_up=h_grid_up,
        prefill_ids=np.array([[r["item_id"] for r in prefills[c]] for c in concepts]),
        offtopic_ids=np.array(off_ids),
        manifest=np.str_(json.dumps(manifest)),
        **probe_pack,
    )
    (out / f"{tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")

    k0 = args.kinds[0]
    diag = np.array(
        [
            down[a, :, bank.column(k0, c)].mean()
            - bare[a, :, bank.column(k0, c)].mean()
            for a, c in enumerate(concepts)
        ]
    )
    print(
        f"\nelapsed {manifest['elapsed_s'] / 60:.1f} min   diagonal delta-z ({k0}) "
        f"{diag.mean():+.3f}"
    )
    print(
        f"wrote {out / f'{tag}.npz'}  ({(out / f'{tag}.npz').stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
