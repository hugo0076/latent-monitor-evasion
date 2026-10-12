"""Figure 1 (panels a-d). CPU only, reads stored eval artifacts.

    python figures/fig_teaser.py

Reads  outputs/llama-3.1-8b/evals/{base,r1_sigmoid_mse}.npz, outputs/llama-3.1-8b/probes/base/
       experiments/outputs/label_corruption/, experiments/outputs/nicl_scaling/
Writes figures/outputs/teaser.{pdf,png}

    a  one episode, as a schematic with illustrative scores
    b  TPR @ 1% FPR of each arm, on training and held-out concepts
    c  paired Cohen's d_z against the fraction of in-context verdicts flipped
    d  per-episode cos(Δh, w) against the number of demonstrations

Error bars in b and bands in c are 95% bootstrap intervals over concepts. The amplified arm in b
is h_bare - 8 Δh: the base edit, times 8, on the single-turn activations of the same response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from core.activations import model_dir, probes_dir  # noqa: E402
from core.concepts import TRAIN_CONCEPTS, held_out_concepts  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = ROOT / "experiments/outputs"
OUT = ROOT / "outputs"
MODEL = "llama-3.1-8b"
LORA = "r1_sigmoid_mse"
N_BOOT = 2000
C_BASE, C_AMP, C_FT = "#9aa5b1", "#5b8ff9", "#e8684a"


def boot_ci(x, rng, n=N_BOOT):
    d = np.asarray(x)[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


EMOJI = HERE / "assets/emoji"
CODEPOINT = {"person": "1f9d1", "robot": "1f916", "lens": "1f50d"}
_cache: dict = {}


def emoji(name: str, px: int = 512):
    """Twemoji image `name` as an RGBA array, from the files in assets/emoji.

    The SVG is rasterised at `px` when cairosvg is installed. Otherwise the 72x72 PNG is used.
    The graphics are CC-BY 4.0 (see assets/emoji/LICENSE.md), so a figure that uses them must
    carry the attribution.
    """
    if name in _cache:
        return _cache[name]
    cp = CODEPOINT[name]
    try:
        import io

        import cairosvg
        from PIL import Image

        buf = cairosvg.svg2png(
            url=str(EMOJI / f"{cp}.svg"), output_width=px, output_height=px
        )
        img = (
            np.asarray(Image.open(io.BytesIO(buf)).convert("RGBA"), dtype=float) / 255.0
        )
    except Exception:
        img = plt.imread(EMOJI / f"{cp}.png")
    _cache[name] = img
    return img


def data_per_point(ax) -> float:
    """Data units per typographic point on the x axis of `ax`.

    Boxes are in data units and text and emoji are in points. This converts between the two.
    """
    bb = ax.get_window_extent()
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    return xr / (bb.width / ax.get_figure().dpi * 72.0)


def glyph(ax, name: str, x: float, y: float, size_data: float = 0.5) -> None:
    """Place an emoji `size_data` data units across, whatever the panel size."""
    img = emoji(name)
    zoom = (size_data / data_per_point(ax)) / img.shape[1]
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom),
        (x, y),
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=6,
    )
    ax.add_artist(ab)


DESIGN_ASPECT = 4.224 / 5.76


LENS = 0.52
HH = 0.46
GAP = 0.24
CMAP_SCORE = plt.get_cmap(style.SCORE_CMAP)
OUTCOME = {"1": "#c23a2b", "0": "#2f8f4e"}

# The per-token scores are illustrative, not measured: physics tokens score high.
DOCTOR = [
    ("The", 0.05),
    (" doctor", 0.10),
    (" asked", 0.05),
    (" about", 0.04),
    (" my", 0.04),
    (" sleep", 0.08),
    (" and", 0.04),
    (" I", 0.05),
]
DIPOLE = [
    ("An", 0.20),
    (" electric", 0.85),
    (" dip", 0.95),
    ("ole", 0.97),
    (" p", 0.72),
    (" in", 0.25),
    (" a", 0.15),
    (" uniform", 0.62),
    (" \u2026", None),
]
ENDO = [
    ("End", 0.15),
    ("oc", 0.10),
    ("yt", 0.08),
    ("osis", 0.12),
    (" via", 0.06),
    (" cl", 0.10),
    ("ath", 0.12),
    ("rin", 0.10),
    ("-co", 0.09),
    ("ated", 0.08),
    (" \u2026", None),
]
MASS = [
    "Mass",
    "-energy",
    " equivalence",
    " E",
    " =",
    " mc",
    "²",
    " follows",
    " from",
    " special",
    " rel",
    "ativity",
]
HOT = [0.90, 0.93, 0.85, 0.70, 0.55, 0.95, 0.95, 0.30, 0.22, 0.80, 0.90, 0.92]
COLD = [0.12, 0.10, 0.15, 0.10, 0.08, 0.18, 0.18, 0.06, 0.05, 0.12, 0.14, 0.12]
FINAL_HOT = list(zip(MASS, HOT))
FINAL_COLD = list(zip(MASS, COLD))


def measurer(ax, fs: float):
    fig = ax.get_figure()
    r = fig.canvas.get_renderer()
    dpp = data_per_point(ax)

    def w_of(t: str) -> float:
        if not t:
            return 0.0
        tt = ax.text(0, 0, t, fontsize=fs)
        w = tt.get_window_extent(renderer=r).width / fig.dpi * 72.0 * dpp
        tt.remove()
        return w

    return w_of


def token_cells(ax, tokens, max_w: float, fs: float):
    """Return the sentence to draw and one (start, end, score) cell per token.

    Cell extents come from measured prefix widths, so each cell spans its token's glyphs and
    the gaps in the score strip are the token boundaries. The sentence is cut at a token
    boundary, with an ellipsis, when it is wider than `max_w`.
    """
    w_of = measurer(ax, fs)
    shown, cells, prefix = "", [], ""
    for tok, sv in tokens:
        if w_of(prefix + tok) > max_w:
            shown = prefix + " …"
            break
        cells.append((w_of(prefix), w_of(prefix + tok), sv))
        prefix += tok
        shown = prefix
    return shown, cells


def gradient_bar(ax, x, y, w, h):
    n = 60
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (x + w * i / n, y),
                w / n + 0.005,
                h,
                fc=CMAP_SCORE(i / (n - 1)),
                ec="none",
                zorder=3,
            )
        )


USER_X0 = 1.28
ROBOT_X = 8.41


def panel_schematic(ax) -> None:
    """Panel a: one episode drawn as a chat.

    Human turns are on the left. The model is on the right, with the monitor lens and the verdict
    chip. The score strips follow Llama-3.1 token boundaries. Scenarios A and B are the two
    alternative final turns (imitate 1 or imitate 0) on the same forced sentence.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect(1 / DESIGN_ASPECT, adjustable="box")
    ax.axis("off")
    aspect = DESIGN_ASPECT

    def lens_on_head(rx, ry, gsz):
        """Draw the monitor lens over the robot's head."""
        glyph(ax, "lens", rx + gsz * 0.40, ry + gsz * 0.34, size_data=LENS * 1.05)

    def verdict(xc, yc, txt, col):
        sw = 0.56
        sh = sw * aspect
        ax.add_patch(
            FancyBboxPatch(
                (xc - sw / 2, yc - sh / 2),
                sw,
                sh,
                boxstyle="round,pad=0.0,rounding_size=0.10",
                fc=col,
                ec="none",
                zorder=3,
                mutation_aspect=aspect,
            )
        )
        ax.text(
            xc,
            yc,
            txt,
            va="center_baseline",
            ha="center",
            fontsize=10.5,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    def user_bubble(y, text, *, edge="#c8cfd6", lw=0.9):
        x0, w = USER_X0, 6.05
        ax.add_patch(
            FancyBboxPatch(
                (x0, y),
                w,
                HH,
                boxstyle="round,pad=0.04,rounding_size=0.13",
                fc="#ffffff",
                ec=edge,
                lw=lw,
                zorder=2,
                mutation_aspect=aspect,
            )
        )
        ax.text(
            x0 + 0.20,
            y + HH / 2,
            text,
            va="center",
            ha="left",
            fontsize=9.0,
            color=style.INK,
            zorder=3,
        )
        glyph(ax, "person", x0 - 0.52, y + HH / 2, size_data=0.58)

    def robot_bubble(
        y, tokens, tag, tagcol, *, parts=None, edge="#c8cfd6", lw=0.9, fc="#f2f5f8"
    ):
        w = 6.05
        rx = ROBOT_X
        x0 = rx - 0.60 - w
        ax.add_patch(
            FancyBboxPatch(
                (x0, y),
                w,
                HH,
                boxstyle="round,pad=0.04,rounding_size=0.13",
                fc=fc,
                ec=edge,
                lw=lw,
                zorder=2,
                mutation_aspect=aspect,
            )
        )
        if parts:
            w_of = measurer(ax, 9.0)
            cx = x0 + 0.20
            for txt, weight in parts:
                ax.text(
                    cx,
                    y + HH / 2,
                    txt,
                    va="center",
                    ha="left",
                    fontsize=9.0,
                    color=style.INK,
                    fontweight=weight,
                    zorder=3,
                )
                cx += w_of(txt)
        else:
            shown, cells = token_cells(ax, tokens, w - 0.40, 9.0)
            ax.text(
                x0 + 0.20,
                y + HH / 2 + 0.07,
                shown,
                va="center",
                ha="left",
                fontsize=9.0,
                color=style.INK,
                zorder=3,
            )
            for cs, ce, sv in cells:
                if sv is None:
                    continue
                ax.add_patch(
                    Rectangle(
                        (x0 + 0.20 + cs, y + 0.062),
                        ce - cs - 0.028,
                        0.085,
                        fc=CMAP_SCORE(sv),
                        ec="none",
                        zorder=2.5,
                    )
                )
        gsz = 0.66
        glyph(ax, "robot", rx, y + HH / 2, size_data=gsz)
        lens_on_head(rx, y + HH / 2, gsz)
        verdict(rx + 0.86, y + HH / 2, tag, tagcol)

    ax.add_patch(
        FancyBboxPatch(
            (0.35, 8.88),
            9.2,
            1.08,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            fc="#eef2f6",
            ec="#c8cfd6",
            lw=0.8,
            ls=(0, (3, 2)),
            zorder=2,
            mutation_aspect=aspect,
        )
    )
    glyph(ax, "lens", 0.84, 9.43, size_data=0.54)
    ax.text(
        1.26,
        9.66,
        "Latent monitor watching for:  physics",
        fontsize=9.1,
        va="center",
        color=style.INK,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        1.26,
        9.20,
        "reads activations",
        fontsize=7.7,
        va="center",
        color=style.MUTED,
        style="italic",
        zorder=3,
    )
    kx, ky, kw, kh = 6.15, 9.13, 1.6, 0.14
    gradient_bar(ax, kx, ky, kw, kh)
    ax.text(
        kx - 0.10,
        ky + kh / 2,
        "benign",
        fontsize=7.7,
        color=style.MUTED,
        ha="right",
        va="center",
        zorder=3,
    )
    ax.text(
        kx + kw + 0.10,
        ky + kh / 2,
        "flagged",
        fontsize=7.7,
        color=style.MUTED,
        ha="left",
        va="center",
        zorder=3,
    )

    y = 8.88 - 0.26 - HH
    BOTTOM = 0.30
    ORZONE = 0.55
    POLICY = 0.36
    gapzone = (y + HH - BOTTOM) - (5 * (2 * HH + GAP) + 2 * GAP) - ORZONE - POLICY

    def corner_tag(iy, lab):
        tw = 1.30
        ax.add_patch(
            FancyBboxPatch(
                (USER_X0 + 0.17, iy + HH - 0.065),
                tw,
                0.26,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                fc="#52606d",
                ec="none",
                zorder=4,
                mutation_aspect=aspect,
            )
        )
        ax.text(
            USER_X0 + 0.17 + tw / 2,
            iy + HH + 0.065,
            lab,
            fontsize=6.8,
            color="white",
            ha="center",
            va="center",
            zorder=5,
        )

    def turn(instr_parts, tokens, tag, scenario=None):
        nonlocal y
        if scenario:
            user_bubble(y, None, edge="#52606d", lw=1.4)
            corner_tag(y, f"Scenario {scenario}")
        else:
            user_bubble(y, None)
        w_of = measurer(ax, 9.0)
        cx = USER_X0 + 0.20
        for txt, weight in instr_parts:
            ax.text(
                cx,
                y + HH / 2,
                txt,
                va="center",
                ha="left",
                fontsize=9.0,
                color=style.INK,
                fontweight=weight,
                zorder=3,
            )
            cx += w_of(txt)
        y -= HH + GAP
        if scenario:
            robot_bubble(
                y, tokens, tag, OUTCOME[tag], edge="#e8684a", lw=1.7, fc="#fdf3f0"
            )
        else:
            robot_bubble(y, tokens, tag, OUTCOME[tag])
        y -= HH + GAP

    for toks, tag in [(DOCTOR, "0"), (DIPOLE, "1"), (ENDO, "0")]:
        turn([("Say something.", "normal")], toks, tag)

    zone_top = y + HH + GAP
    mid = zone_top - gapzone / 2
    ax.text(
        5.0,
        mid + 0.30,
        "\u22ee",
        fontsize=11,
        ha="center",
        va="center",
        color=style.MUTED,
    )
    ax.text(
        5.0,
        mid,
        "n turns, each with a monitor score",
        fontsize=9.0,
        ha="center",
        va="center",
        color=style.MUTED,
        style="italic",
    )
    ax.text(
        5.0,
        mid - 0.30,
        "\u22ee",
        fontsize=11,
        ha="center",
        va="center",
        color=style.MUTED,
    )
    y -= gapzone - GAP

    for k, (digit, final) in enumerate((("1", FINAL_HOT), ("0", FINAL_COLD))):
        turn(
            [
                ("Say something. Imitate label ", "normal"),
                (digit, "bold"),
                (".", "normal"),
            ],
            final,
            digit,
            scenario="A" if k == 0 else "B",
        )
        if k == 0:
            seam_top = y + HH + GAP
            ax.text(
                5.0,
                seam_top - ORZONE / 2,
                "or",
                fontsize=9.0,
                ha="center",
                va="center",
                color=style.MUTED,
            )
            y -= ORZONE - GAP
    ax.text(
        5.0,
        y + HH + GAP - 0.38,
        "policy forced: the same physics-positive sentence",
        fontsize=8.2,
        color="#e8684a",
        style="italic",
        ha="center",
        va="top",
    )


C_NONE = "#3e4c59"
HELD_HATCH = "///"
ARMS = [
    ("none", C_NONE, "no attack"),
    ("base", C_BASE, "base model"),
    ("amp", C_AMP, "amplified base\n" + r"($\times$8)"),
    ("ft", C_FT, "model organism"),
]
ALPHA = 8.0


def evasion_split() -> dict:
    """Per-concept TPR @ 1% FPR for each arm, split into trained and held-out concepts.

    The threshold is the 99th percentile of the monitor's scores on the other concepts of the
    same split. The result is cached in figures/outputs/.evasion_split.npz and recomputed when
    either eval .npz is newer than the cache.
    """
    mdir = model_dir(OUT, MODEL)
    srcs = [mdir / "evals" / "base.npz", mdir / "evals" / f"{LORA}.npz"]
    cache = HERE / "outputs" / ".evasion_split.npz"
    if cache.exists() and cache.stat().st_mtime > max(f.stat().st_mtime for f in srcs):
        z = np.load(cache)
        return {
            sp: {k: z[f"{sp}|{k}"] for k, _, _ in ARMS}
            for sp in ("trained", "held out")
        }
    out = _evasion_split_uncached()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{sp}|{k}": v for sp, d in out.items() for k, v in d.items()})
    return out


def _evasion_split_uncached() -> dict:
    mdir, pdir = model_dir(OUT, MODEL), probes_dir(OUT, MODEL, source="base")
    zb = np.load(mdir / "evals" / "base.npz", allow_pickle=True)
    zf = np.load(mdir / "evals" / f"{LORA}.npz", allow_pickle=True)
    cs = [str(x) for x in zb["concepts"]]
    roster = [str(x) for x in zb["probes"]][: len(zb["probe_logreg_scale"])]
    W = zb["probe_logreg_w"].astype(np.float64)
    THR = zb["probe_logreg_threshold"].astype(np.float64)
    SCA = zb["probe_logreg_scale"].astype(np.float64)
    S = np.load(pdir / "scores_logreg.npy")
    idxr = [
        json.loads(x)
        for x in (pdir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    conc = np.array([r["concept"] for r in idxr])
    ho, tr = held_out_concepts(), list(TRAIN_CONCEPTS)
    out = {sp: {k: [] for k, _, _ in ARMS} for sp in ("trained", "held out")}
    for sp, names in (("trained", tr), ("held out", ho)):
        for c in names:
            if c not in cs:
                continue
            i, k = cs.index(c), roster.index(c)
            grp = tr if c in tr else ho
            ben = S[np.isin(conc, [x for x in grp if x != c and x in cs]), k]
            thr = float(np.quantile(ben, 0.99))
            hb = np.asarray(zb["h_bare"])[i].astype(np.float64)
            dh = (np.asarray(zb["h_up"])[i] - np.asarray(zb["h_down"])[i]).astype(
                np.float64
            )
            for kk, v in (
                ("none", hb),
                ("base", np.asarray(zb["h_down"])[i].astype(np.float64)),
                ("amp", hb - ALPHA * dh),
                ("ft", np.asarray(zf["h_down"])[i].astype(np.float64)),
            ):
                out[sp][kk].append(float((((v @ W[k] - THR[k]) / SCA[k]) > thr).mean()))
    return {sp: {k: np.array(v) for k, v in d.items()} for sp, d in out.items()}


def panel_evasion(ax, rng) -> None:
    D = evasion_split()
    x = np.arange(len(ARMS))
    for j, sp in enumerate(("trained", "held out")):
        for i, (k, col, _) in enumerate(ARMS):
            v = D[sp][k]
            m = v.mean()
            lo, hi = boot_ci(v, rng)
            ax.bar(
                i + (j - 0.5) * 0.34,
                m,
                0.30,
                color=col,
                hatch=HELD_HATCH if j else None,
                edgecolor="white",
                lw=0.6,
                yerr=[[m - lo], [hi - m]],
                capsize=2.5,
                error_kw={"lw": 0.8, "ecolor": "#1f2933"},
            )
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, _, n in ARMS], fontsize=10.5)
    ax.set_ylabel("TPR @ 1% FPR", fontsize=11.7)
    ax.set_ylim(0, 1.14)
    ax.tick_params(axis="y", labelsize=7.0)
    h = [
        plt.Rectangle((0, 0), 1, 1, fc="#7b8794", ec="white", lw=0.6),
        plt.Rectangle((0, 0), 1, 1, fc="#7b8794", ec="white", lw=0.6, hatch=HELD_HATCH),
    ]
    ax.legend(
        h,
        ["trained concepts", "held-out concepts"],
        fontsize=9.9,
        frameon=False,
        ncol=2,
        loc="upper right",
        handlelength=1.1,
    )
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)


def panel_corruption(ax, rng) -> None:
    for run, col, lab in (
        (LORA, C_FT, "model organism"),
        ("base", C_BASE, "base model"),
    ):
        f = ART / "label_corruption" / f"{MODEL}_{run}_fine.json"
        if not f.exists():
            f = ART / "label_corruption" / f"{MODEL}_{run}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        P = [x["p"] for x in d["levels"]]
        per = [[v["cohens_dz"] for v in x["by_concept"].values()] for x in d["levels"]]
        m = [np.mean(v) for v in per]
        lo = [boot_ci(v, rng)[0] for v in per]
        hi = [boot_ci(v, rng)[1] for v in per]
        ax.plot(P, m, "o-", color=col, lw=1.8, ms=4, label=lab)
        ax.fill_between(P, lo, hi, color=col, alpha=0.20, lw=0)
    ax.axhline(0, color="#52606d", lw=0.7, ls=":")
    ax.axvline(0.5, color="#52606d", lw=0.7, ls=":")
    ax.set_xlabel("Fraction of in-context verdicts flipped", fontsize=11.1)
    ax.set_ylabel(r"Edit size (paired Cohen's $d$)", fontsize=11.4)
    ax.tick_params(labelsize=6.6)
    ax.legend(fontsize=10.2, frameon=False, loc="upper right")
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)


def panel_nicl(ax, rng) -> None:
    d = json.loads((ART / "nicl_scaling" / f"{MODEL}.json").read_text())
    n = d["demo_pairs"]
    # x counts single demonstrations, two per pair
    x = [2 * k for k in n]
    for run, col, lab in (
        (LORA, C_FT, "model organism"),
        ("base", C_BASE, "base model"),
    ):
        if run not in d["by_run"]:
            continue
        R = d["by_run"][run]
        ax.plot(
            x,
            [R[str(k)]["pe_cos_dh_w"] for k in n],
            "o-",
            color=col,
            lw=1.9,
            ms=4,
            label=lab,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x], fontsize=6.8)
    ax.set_xlabel("# of examples in context", fontsize=11.4)
    ax.set_ylabel(r"Edit alignment $\cos(\Delta h,\ w)$", fontsize=11.4)
    ax.set_ylim(0, None)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.legend(fontsize=9.9, frameon=False, loc="lower right")
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)


PANELS = {
    "a": panel_schematic,
    "b": panel_evasion,
    "c": panel_corruption,
    "d": panel_nicl,
}
FIGSIZE = (11, 6.4)
GRID = dict(
    width_ratios=[5.97, 3.45, 3.45],
    height_ratios=[1.05, 1.0],
    hspace=0.38,
    wspace=0.26,
    left=0.030,
    right=0.958,
    top=0.945,
    bottom=0.085,
)
SLOT = {"a": (slice(None), 0), "b": (0, slice(1, 3)), "c": (1, 1), "d": (1, 2)}


def build(rng):
    """Build the figure."""
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 3, **GRID)
    axes = {k: fig.add_subplot(gs[v]) for k, v in SLOT.items()}
    axes["a"].set_xlim(0, 10)
    axes["a"].set_ylim(0, 10)
    axes["a"].set_aspect(1 / DESIGN_ASPECT, adjustable="box")
    b = axes["a"].get_position()
    axes["a"].set_position([b.x0, b.y0 - 0.03, b.width, b.height])
    fig.canvas.draw()
    for k, ax in axes.items():
        panel_schematic(ax) if k == "a" else PANELS[k](ax, rng)
    fig.canvas.draw()
    for k, ax in axes.items():
        e = ax.get_window_extent()
        x0 = e.x0 / (fig.get_figwidth() * fig.dpi)
        y1 = e.y1 / (fig.get_figheight() * fig.dpi)
        if k == "a":
            y1 += 0.03
        fig.text(
            x0 - 0.020,
            min(y1 + 0.040, 0.995),
            k,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    style.apply()

    fig = build(rng)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"teaser.{ext}", dpi=200)
    print(f"wrote {args.out}/teaser.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
