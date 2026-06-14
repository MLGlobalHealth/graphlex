"""Publication figures for the graphlex paper, built from EXISTING result files.

Three figures, all reproducible, no LLM / no inference re-runs (only reads .ans
files + manifests; the only "compute" is replaying sweep.py's deterministic splits
to recompute the classical/logreg baseline + balanced accuracy, exactly as
eval/balanced_rescore.py does).

  1. fig_regret_heatmap.png / fig_regret_heatmap_regret.png
       Cross-domain flexibility. methods x 30 datasets (grouped by science domain).
       Variant A: cell = balanced accuracy. Variant B: cell = regret
       (best non-LLM baseline - method), diverging cmap centred at 0.
  2. fig_label_efficiency.png
       The crossover. 3 panels (family / proteins / imdb): accuracy vs #labels/class
       for logreg (1..12) overlaid with graphlex+LLM (Opus / Qwen-14B / Qwen-32B) at
       k = 1,3,5.
  3. fig_zero_label.png
       2 panels (family / mutag): logreg curve (chance at 0 labels) vs graphlex+LLM
       (Opus) at 0 and 3 shots -- the zero-label point where logreg = chance.

Run headless (Agg). Uses the fmsn venv:
    /home/scratch/fmsn-dev/.venv/bin/python eval/make_figures.py

The expensive sweep recomputation is cached to figures/_sweep_cache.json; pass
--refresh to force recomputation.
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = "/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex"
EVAL = os.path.join(REPO, "eval")
FIGDIR = os.path.join(EVAL, "figures")
SWEEP = "/home/scratch/bench_out/sweep"
LABELCURVE = "/home/scratch/bench_out/labelcurve"
ZEROLABEL = "/home/scratch/bench_out/zerolabel"
CACHE = os.path.join(FIGDIR, "_sweep_cache.json")

os.makedirs(FIGDIR, exist_ok=True)

# Domain ordering for the heatmap columns (the 8 sciences).
DOMAIN_ORDER = ["chemistry", "biology", "neuroscience", "social",
                "vision", "synthetic", "citation", "archaeology"]

# Shared answer-line parser + balanced-accuracy metric (see eval/_common.py).
sys.path.insert(0, EVAL)
from _common import parse_ans, bal_acc  # noqa: E402


# ===========================================================================
# 1. SWEEP -- balanced accuracy table (reuses balanced_rescore.py logic)
# ===========================================================================
def compute_sweep_table(refresh=False):
    """Return list of dicts: domain, dataset, chance, classic, majority,
    qwen14, qwen32, opus (balanced accuracy, mean over seeds).

    Reuses eval/balanced_rescore.py's functions verbatim (import). Cached to
    figures/_sweep_cache.json because it replays TUDataset splits + logreg.
    """
    if not refresh and os.path.exists(CACHE):
        with open(CACHE) as fh:
            return json.load(fh)

    sys.path.insert(0, REPO)
    sys.path.insert(0, EVAL)
    import balanced_rescore as br  # noqa: E402  (does the heavy lifting on import)

    rows = []
    for ds, meta in sorted(br.man["meta"].items(),
                           key=lambda kv: (kv[1]["domain"], kv[0])):
        ncls = meta["classes"]
        cl, mj = br.classical_majority_balacc(ds)
        rows.append({
            "domain": meta["domain"],
            "dataset": ds,
            "chance": 1.0 / ncls,
            "classic": float(cl),
            "majority": float(mj),
            "qwen14": br.llm_balacc(ds, "qwen"),
            "qwen32": br.llm_balacc(ds, "qwen32"),
            "opus": br.llm_balacc(ds, "opus"),
        })
    with open(CACHE, "w") as fh:
        json.dump(rows, fh, indent=2)
    return rows


def _order_rows(rows):
    """Order datasets by DOMAIN_ORDER then dataset name; return (rows, domain
    boundary indices, list of (domain, start, end))."""
    dom_rank = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    rows = sorted(rows, key=lambda r: (dom_rank.get(r["domain"], 99),
                                       r["dataset"].lower()))
    groups = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["domain"] != rows[start]["domain"]:
            groups.append((rows[start]["domain"], start, i))
            start = i
    return rows, groups


METHODS = [("classic", "Classical\n(logreg)"),
           ("qwen14", "Qwen-14B"),
           ("qwen32", "Qwen-32B"),
           ("opus", "Opus")]


def fig_heatmap(rows, mode="acc"):
    """mode='acc' -> balanced accuracy; mode='regret' -> best-non-LLM - method."""
    rows, groups = _order_rows(rows)
    n = len(rows)
    method_keys = [k for k, _ in METHODS]
    M = np.full((len(method_keys), n), np.nan)
    for j, r in enumerate(rows):
        baseline = max(r["classic"], r["majority"])  # best non-LLM baseline
        for i, mk in enumerate(method_keys):
            v = r[mk]
            if v is None:
                continue
            if mode == "regret":
                # regret = baseline - method  -> positive = method behind.
                # Flip sign so positive (warm) = LLM AHEAD, intuitive reading.
                M[i, j] = v - baseline
            else:
                M[i, j] = v

    fig_w = max(13, 0.42 * n + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))

    if mode == "regret":
        lim = np.nanmax(np.abs(M))
        lim = max(lim, 0.05)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = plt.get_cmap("RdBu").copy()
        cbar_label = ("balanced-accuracy advantage\nover best non-LLM baseline\n"
                      "(red = LLM ahead, blue = behind)")
        title = ("Cross-domain flexibility: graphlex+LLM advantage over the best "
                 "non-LLM baseline (balanced accuracy)\n"
                 "5 shots/class, 3 seeds, 30 datasets across 8 sciences")
        annot_fmt = "{:+.2f}"
    else:
        norm = None
        cmap = plt.get_cmap("viridis").copy()
        cbar_label = "balanced accuracy (macro-averaged per-class recall)"
        title = ("Cross-domain balanced accuracy, 30 datasets across 8 sciences "
                 "(5 shots/class, 3 seeds)")
        annot_fmt = "{:.2f}"
    cmap.set_bad(color="0.8")  # grey for missing (NaN) cells

    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm,
                   vmin=(0.0 if mode == "acc" else None),
                   vmax=(1.0 if mode == "acc" else None))

    # Hatch the missing cells so "unscored" reads differently from a real value.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch="////", edgecolor="0.55",
                                           linewidth=0.0))
            else:
                # in-cell value
                txt = annot_fmt.format(M[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                        color="white" if mode == "acc" and M[i, j] < 0.55
                        else "black")

    ax.set_yticks(range(len(method_keys)))
    ax.set_yticklabels([lbl for _, lbl in METHODS], fontsize=8)
    ax.set_xticks(range(n))
    ax.set_xticklabels([r["dataset"] for r in rows], rotation=90, fontsize=6)
    ax.set_ylim(len(method_keys) - 0.5, -0.5)

    # Domain separators + labels along the top. Stagger labels vertically so
    # narrow (1-dataset) groups like citation / archaeology don't collide.
    prev_e = None
    level = 0
    for dom, s, e in groups:
        if s != 0:
            ax.axvline(s - 0.5, color="white", linewidth=2.2)
            ax.axvline(s - 0.5, color="black", linewidth=0.6)
        narrow = (e - s) <= 2
        # alternate the y-offset only among consecutive narrow groups
        if narrow and prev_e is not None and (s - prev_e) <= 2:
            level = 1 - level
        else:
            level = 0
        y = -0.78 - (0.85 if (narrow and level == 1) else 0.0)
        ax.text((s + e - 1) / 2.0, y, dom, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", rotation=0)
        prev_e = e

    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=10, pad=26)

    # Legend for missing cells.
    ax.legend(handles=[Patch(facecolor="0.8", hatch="////", edgecolor="0.55",
                             label="not scored (LLM off-format)")],
              loc="upper left", bbox_to_anchor=(0.0, -0.32), fontsize=7,
              frameon=False)

    fig.tight_layout()
    suffix = "_regret" if mode == "regret" else ""
    out = os.path.join(FIGDIR, f"fig_regret_heatmap{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, M, rows


# ===========================================================================
# 2. LABEL-EFFICIENCY CROSSOVER (reuses score_labelcurve.py logic)
# ===========================================================================
def labelcurve_data():
    man = json.load(open(os.path.join(LABELCURVE, "manifest.json")))

    def llm_scores(tname, model):
        out = {}
        files = man["tasks"][tname]["files"]
        addir = os.path.join(LABELCURVE, tname, "ans", model)
        if not os.path.isdir(addir):
            return out
        for fn, meta in files.items():
            ans = os.path.join(addir, os.path.basename(fn).replace(".txt", ".ans"))
            if not os.path.exists(ans):
                continue
            pred = parse_ans(ans)
            if not pred:
                continue
            tt = {i: str(l).upper() for i, l in meta["truth"]}
            acc = sum(1 for i, l in tt.items() if pred.get(i) == l) / len(tt)
            out.setdefault(meta["k"], []).append(acc)
        return out

    data = {}
    for t, td in man["tasks"].items():
        data[t] = {
            "chance": td["chance"],
            "logreg": {int(k): v for k, v in td["logreg"].items()},
            "opus": llm_scores(t, "opus"),
            "qwen14": llm_scores(t, "qwen"),
            "qwen32": llm_scores(t, "qwen32"),
        }
    return man["k_logreg"], data


def _ms(d):
    """k->list -> sorted (ks, means, stds)."""
    ks = sorted(d)
    return (np.array(ks),
            np.array([np.mean(d[k]) for k in ks]),
            np.array([np.std(d[k]) for k in ks]))


def fig_label_efficiency(k_logreg, data):
    titles = {"family": "family (synthetic, network-science prior)",
              "proteins": "PROTEINS (biology, weak prior)",
              "imdb": "IMDB (social, weak prior)"}
    order = [t for t in ["family", "proteins", "imdb"] if t in data]
    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.2),
                             sharey=False)
    if len(order) == 1:
        axes = [axes]

    style = {
        "opus":   dict(color="#1f77b4", marker="o", label="graphlex+LLM (Opus)"),
        "qwen32": dict(color="#9467bd", marker="^", label="graphlex+LLM (Qwen-32B)"),
        "qwen14": dict(color="#2ca02c", marker="s", label="graphlex+LLM (Qwen-14B)"),
    }

    for ax, t in zip(axes, order):
        d = data[t]
        ch = d["chance"]
        # logreg curve (full k grid)
        lk, lm, ls = _ms(d["logreg"])
        ax.plot(lk, lm, color="#d62728", marker="D", lw=2.0, ms=5,
                label="logreg (trained on same labels)", zorder=3)
        ax.fill_between(lk, lm - ls, lm + ls, color="#d62728", alpha=0.12)
        # LLM points (k = 1,3,5 where available)
        for mk in ["opus", "qwen32", "qwen14"]:
            if not d[mk]:
                continue
            kk, mm, ss = _ms(d[mk])
            ax.errorbar(kk, mm, yerr=ss, lw=1.6, ms=6, capsize=2.5,
                        zorder=4, **style[mk])
        ax.axhline(ch, color="0.4", ls=":", lw=1.2)
        ax.text(k_logreg[-1], ch, " chance", va="center", ha="left",
                fontsize=7.5, color="0.4")
        ax.set_xscale("log")
        ax.set_xticks(k_logreg)
        ax.set_xticklabels([str(k) for k in k_logreg])
        ax.minorticks_off()
        ax.set_xlabel("labeled examples per class (k)")
        ax.set_title(titles.get(t, t), fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_ylim(min(0.3, ch - 0.05), 1.02)
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Label-efficiency crossover: at low k, graphlex+LLM (frontier "
                 "model) leads on prior-rich tasks;\nlogreg catches up as labels "
                 "accumulate. Mean ± std over seeds.",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    out = os.path.join(FIGDIR, "fig_label_efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, data


# ===========================================================================
# 3. ZERO-LABEL (reuses zero_label manifest + same answer parser)
# ===========================================================================
def zerolabel_data():
    man = json.load(open(os.path.join(ZEROLABEL, "manifest.json")))

    def opus_scores(tname):
        out = {}
        files = man["tasks"][tname]["files"]
        addir = os.path.join(ZEROLABEL, tname, "ans", "opus")
        for fn, meta in files.items():
            ans = os.path.join(addir, os.path.basename(fn).replace(".txt", ".ans"))
            if not os.path.exists(ans):
                continue
            pred = parse_ans(ans)
            if not pred:
                continue
            tt = {i: str(l).upper() for i, l in meta["truth"]}
            acc = sum(1 for i, l in tt.items() if pred.get(i) == l) / len(tt)
            out.setdefault(meta["shot"], []).append(acc)
        return out

    data = {}
    for t, td in man["tasks"].items():
        data[t] = {
            "chance": td["chance"],
            "logreg": {int(k): v for k, v in td["logreg"].items()},
            "opus": opus_scores(t),  # shot -> [acc]
        }
    return man["llm_shots"], man["logreg_k"], data


def fig_zero_label(llm_shots, logreg_k, data):
    titles = {"family": "family (network-science prior)",
              "mutag": "MUTAG (chemistry prior)"}
    order = [t for t in ["family", "mutag"] if t in data]
    fig, axes = plt.subplots(1, len(order), figsize=(5.0 * len(order), 4.3))
    if len(order) == 1:
        axes = [axes]

    for ax, t in zip(axes, order):
        d = data[t]
        ch = d["chance"]
        # logreg curve: chance at 0 labels, then trained values.
        lk = sorted(d["logreg"])
        lx = [0] + lk
        ly = [ch] + [np.mean(d["logreg"][k]) for k in lk]
        lerr = [0] + [np.std(d["logreg"][k]) for k in lk]
        ax.errorbar(lx, ly, yerr=lerr, color="#d62728", marker="D", lw=2.0,
                    ms=6, capsize=3, label="logreg (chance at 0 labels)", zorder=3)
        # graphlex+LLM (Opus) at shots 0 and 3.
        ok = sorted(d["opus"])
        ox = list(ok)
        oy = [np.mean(d["opus"][s]) for s in ok]
        oerr = [np.std(d["opus"][s]) for s in ok]
        ax.errorbar(ox, oy, yerr=oerr, color="#1f77b4", marker="o", lw=2.0,
                    ms=8, capsize=3, label="graphlex+LLM (Opus)", zorder=4)

        ax.axhline(ch, color="0.4", ls=":", lw=1.2)
        ax.text(max(logreg_k), ch, " chance", va="center", ha="left",
                fontsize=8, color="0.4")

        # Annotate the zero-label point.
        if 0 in d["opus"]:
            z = np.mean(d["opus"][0])
            ax.annotate(
                f"0 labels: {z:.2f}\n(logreg = chance {ch:.2f})",
                xy=(0, z), xytext=(0.9, max(z - 0.22, ch + 0.04)),
                fontsize=8.5, color="#1f77b4", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3))
        ax.set_xlabel("labeled examples per class")
        ax.set_xticks([0] + logreg_k)
        ax.set_title(titles.get(t, t), fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_ylim(min(0.25, ch - 0.1), 1.02)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Zero-label capability: graphlex+LLM produces a useful answer with "
                 "0 labels, where logreg can only guess (chance).\n"
                 "On a strong-prior domain (family) 0-label LLM ≈ logreg trained "
                 "on 10 labels/class.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIGDIR, "fig_zero_label.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, data


# ===========================================================================
# main
# ===========================================================================
def _verify(path):
    ok = os.path.exists(path) and os.path.getsize(path) > 5000
    sz = os.path.getsize(path) if os.path.exists(path) else 0
    return ok, sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="force sweep recomputation (ignore cache)")
    args = ap.parse_args()

    print("Computing sweep balanced-accuracy table "
          "(reusing balanced_rescore.py)...")
    rows = compute_sweep_table(refresh=args.refresh)

    out_acc, Macc, ordered = fig_heatmap(rows, mode="acc")
    out_reg, Mreg, _ = fig_heatmap(rows, mode="regret")

    k_logreg, lc = labelcurve_data()
    out_lc, _ = fig_label_efficiency(k_logreg, lc)

    llm_shots, logreg_k, zl = zerolabel_data()
    out_zl, _ = fig_zero_label(llm_shots, logreg_k, zl)

    # ---- summaries -------------------------------------------------------
    print("\n" + "=" * 78)
    print("FIGURES WRITTEN (verify + key numbers)")
    print("=" * 78)

    # Fig 1 key numbers
    method_keys = [k for k, _ in METHODS]
    opus_idx = method_keys.index("opus")
    opus_reg = Mreg[opus_idx, :]
    opus_reg_valid = opus_reg[~np.isnan(opus_reg)]
    n_ahead = int((opus_reg_valid > 0).sum())
    n_within = int((opus_reg_valid >= -0.05).sum())
    n_subst_worse = int((opus_reg_valid < -0.10).sum())
    ndom = len(set(r["domain"] for r in ordered))
    qwen14_missing = int(np.isnan(Macc[method_keys.index("qwen14"), :]).sum())
    qwen32_missing = int(np.isnan(Macc[method_keys.index("qwen32"), :]).sum())

    ok, sz = _verify(out_acc)
    print(f"\n[1a] {out_acc}\n     exists={ok} size={sz/1024:.0f}KB")
    print(f"     Balanced accuracy, {Macc.shape[1]} datasets x {ndom} domains, "
          f"4 methods. Opus mean balanced acc = "
          f"{np.nanmean(Macc[opus_idx]):.3f}; classical = "
          f"{np.nanmean(Macc[0]):.3f}. Missing cells (hatched): "
          f"Qwen-14B {qwen14_missing}, Qwen-32B {qwen32_missing}.")
    ok, sz = _verify(out_reg)
    print(f"\n[1b] {out_reg}\n     exists={ok} size={sz/1024:.0f}KB")
    print(f"     Regret (Opus - best non-LLM baseline): mean "
          f"{opus_reg_valid.mean():+.3f} over n={len(opus_reg_valid)}; "
          f"Opus ahead on {n_ahead}/{len(opus_reg_valid)}, within 0.05 on "
          f"{n_within}/{len(opus_reg_valid)}, substantially worse (>0.10) on "
          f"{n_subst_worse}/{len(opus_reg_valid)}.")

    # Fig 2 key numbers
    ok, sz = _verify(out_lc)
    fam = lc.get("family", {})
    fam_o1 = np.mean(fam["opus"][1]) if fam.get("opus", {}).get(1) else float("nan")
    fam_l1 = np.mean(fam["logreg"][1])
    print(f"\n[2]  {out_lc}\n     exists={ok} size={sz/1024:.0f}KB")
    print(f"     Crossover, 3 panels (family/proteins/imdb), logreg k=1..12 vs "
          f"graphlex+LLM at k=1,3,5. Headline: family k=1 Opus "
          f"{fam_o1:.3f} vs logreg {fam_l1:.3f} "
          f"(+{fam_o1 - fam_l1:.3f}); logreg overtakes as k grows.")

    # Fig 3 key numbers
    ok, sz = _verify(out_zl)
    zfam = zl.get("family", {})
    zfam0 = np.mean(zfam["opus"][0]) if zfam.get("opus", {}).get(0) else float("nan")
    zfam_l10 = np.mean(zfam["logreg"][10]) if zfam.get("logreg", {}).get(10) else float("nan")
    zmut = zl.get("mutag", {})
    zmut0 = np.mean(zmut["opus"][0]) if zmut.get("opus", {}).get(0) else float("nan")
    print(f"\n[3]  {out_zl}\n     exists={ok} size={sz/1024:.0f}KB")
    print(f"     Zero-label, 2 panels (family/mutag). family 0-label Opus "
          f"{zfam0:.3f} ≈ logreg@10 labels {zfam_l10:.3f} (logreg@0 = chance "
          f"{zfam['chance']:.3f}); mutag 0-label Opus {zmut0:.3f} > chance "
          f"{zmut['chance']:.3f}.")
    print()


if __name__ == "__main__":
    main()
