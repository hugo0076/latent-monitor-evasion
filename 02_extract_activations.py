"""Extract per-token residual-stream activations at the monitored layer. Needs a GPU.

    python 02_extract_activations.py

Reads  data/splits/{probe_train,nf}/{concept}.jsonl
Writes outputs/<model>/activations/<source>/tokens/{concept}.npy      [total_tokens, D] float32
       outputs/<model>/activations/<source>/tokens/{concept}.idx.npy  [501] int64 offsets
       outputs/<model>/activations/<source>/pooled/{concept}.npy      [500, D] float32
       outputs/<model>/activations/<source>/stats.json

`<source>` is "base", or the adapter name when --adapter is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.activations import (
    MODELS,
    activations_dir,
    extract,
    load_model,
    model_dir,
    pack,
    pool,
)
from core.concepts import ALL_CONCEPTS

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument(
        "--layer", type=int, default=None, help="default: the model's monitor layer"
    )
    ap.add_argument("--splits", type=Path, default=HERE / "data/splits")
    ap.add_argument("--outputs", type=Path, default=HERE / "outputs")
    ap.add_argument(
        "--adapter",
        default=None,
        help="extract from this adapter's model; source name defaults to it",
    )
    ap.add_argument(
        "--max-response-tokens",
        type=int,
        default=224,
        help="covers the longest observed prefill (208)",
    )
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    layer = args.layer if args.layer is not None else MODELS[args.model]["layer"]
    out = activations_dir(
        args.outputs, args.model, source=args.adapter or "base", layer=layer
    )
    for sub in ("tokens", "pooled"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(
        f"model={args.model} layer={layer} source={args.adapter or 'base'} "
        f"concepts={len(ALL_CONCEPTS)} -> {out}",
        flush=True,
    )
    model, tokenizer = load_model(MODELS[args.model]["hf_name"])
    if args.adapter:
        from peft import PeftModel

        adapter_path = (
            model_dir(args.outputs, args.model, layer) / "adapters" / args.adapter
        )
        if not adapter_path.exists():
            raise SystemExit(f"no adapter at {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        model.eval()

    truncated: dict[str, int] = {}
    lengths: list[int] = []
    peak = 0.0
    for concept in ALL_CONCEPTS:
        rows = []
        for split in ("probe_train", "nf"):
            path = args.splits / split / f"{concept}.jsonl"
            rows += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

        blocks, n_trunc = extract(
            model,
            tokenizer,
            layer,
            [r["sentence"] for r in rows],
            max_response_tokens=args.max_response_tokens,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        tokens, offsets = pack(blocks)
        np.save(out / "tokens" / f"{concept}.npy", tokens)
        np.save(out / "tokens" / f"{concept}.idx.npy", offsets)
        np.save(out / "pooled" / f"{concept}.npy", pool(tokens, offsets))

        if not np.isfinite(tokens).all():
            raise SystemExit(
                f"{concept}: {int((~np.isfinite(tokens)).sum())} non-finite activations "
                f"(dtype {tokens.dtype}). The storage dtype cannot represent this model's range."
            )
        peak = max(peak, float(np.abs(tokens).max()))

        truncated[concept] = n_trunc
        n_tok = np.diff(offsets)
        lengths.extend(n_tok.tolist())
        print(
            f"[{concept:28s}] {len(blocks):4d} sentences  resp tokens "
            f"mean={n_tok.mean():6.1f} max={n_tok.max():4d}  truncated={n_trunc}  "
            f"{tokens.nbytes / 1e9:.2f} GB",
            flush=True,
        )

    total_truncated = sum(truncated.values())
    stats = {
        "model": args.model,
        "hf_name": MODELS[args.model]["hf_name"],
        "layer": layer,
        "max_response_tokens": args.max_response_tokens,
        "n_concepts": len(ALL_CONCEPTS),
        "row_order": "probe_train_then_nf",
        "storage": "ragged: tokens[idx[i]:idx[i+1]] is sentence i",
        "total_token_vectors": int(sum(lengths)),
        "response_tokens": {
            "mean": float(np.mean(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "p99": float(np.percentile(lengths, 99)),
        },
        "dtype": "float32",
        "max_abs_activation": peak,
        "n_truncated_total": int(total_truncated),
        "n_truncated_per_concept": {c: n for c, n in truncated.items() if n},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"\nmax |activation| = {peak:.6g}  (float16 would overflow above 65504)")
    print(
        f"response tokens: mean {stats['response_tokens']['mean']:.1f}, "
        f"max {stats['response_tokens']['max']}, p99 {stats['response_tokens']['p99']:.0f}"
    )
    if total_truncated:
        print(
            f"WARNING: {total_truncated} sentences exceeded the {args.max_response_tokens}-token "
            "cap and were truncated. Raise --max-response-tokens and re-extract."
        )
    else:
        print(f"no sentence hit the {args.max_response_tokens}-token cap")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
