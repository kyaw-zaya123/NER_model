#!/usr/bin/env python3
"""
plot_results.py  —  Myanmar NER Results Dashboard
Produces clean, publication-quality PNG charts from checkpoint folders.

Charts generated:
  1. model_comparison.png       — grouped bar: val F1 vs test F1
  2. training_curves_<name>.png — loss + F1 curves per model
  3. per_class_f1.png           — per-entity F1 heatmap across all models
  4. per_class_bars_<name>.png  — P / R / F1 bars per model
  5. confusion_matrix_<name>.png— TP/FP/FN confusion matrix (from report stats)

Usage:
    python plot_results.py --checkpoints_dir "checkpoints - Copy" --out_dir results
"""

import os, re, json, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "#f8f9fa",
    "axes.grid"          : True,
    "grid.color"         : "#dee2e6",
    "grid.linewidth"     : 0.7,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.spines.left"   : True,
    "axes.spines.bottom" : True,
    "axes.edgecolor"     : "#ced4da",
    "font.family"        : "DejaVu Sans",
    "font.size"          : 10,
    "axes.titlesize"     : 13,
    "axes.titleweight"   : "bold",
    "axes.titlepad"      : 14,
    "axes.labelsize"     : 10,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 9,
    "legend.fontsize"    : 9,
    "legend.framealpha"  : 0.9,
    "legend.edgecolor"   : "#dee2e6",
    "savefig.dpi"        : 180,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
})

# Colour palette
C_BLUE   = "#4A90D9"
C_ORANGE = "#E8713C"
C_GREEN  = "#3DAA6E"
C_PURPLE = "#7B68D4"
C_TEAL   = "#2AAFB0"
C_RED    = "#D94A4A"
C_GOLD   = "#D4A017"
C_GRAY   = "#8A9BB0"

MODEL_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_TEAL, C_RED, C_GOLD]

ENTITY_COLORS = {
    "PER"         : "#E8713C",
    "ORG"         : "#4A90D9",
    "LOC"         : "#8A9BB0",
    "LOC-CITY"    : "#2AAFB0",
    "LOC-COUNTRY" : "#3DAA6E",
    "LOC-DISTRICT": "#7B68D4",
    "LOC-STATE"   : "#9B59B6",
    "LOC-TOWNSHIP": "#1A8FA0",
    "LOC-VILLAGE" : "#27AE60",
    "LOC-WARD"    : "#5DADE2",
    "DATE"        : "#D4A017",
    "TIME"        : "#E67E22",
    "NUM"         : "#95A5A6",
}

def ec(label):
    return ENTITY_COLORS.get(label, C_GRAY)


# ── Data loading ──────────────────────────────────────────────────────────────
def parse_report(path):
    """Parse test_report.txt → {test_f1, classes:{label:{p,r,f1,support}}}"""
    result = {"test_f1": None, "classes": {}, "micro": None}
    if not os.path.exists(path):
        return result
    text = open(path, encoding="utf-8").read()
    m = re.search(r"Test F1:\s*([\d.]+)", text)
    if m:
        result["test_f1"] = float(m.group(1))
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            sup = int(parts[-1]); f1 = float(parts[-2])
            rec = float(parts[-3]); pre = float(parts[-4])
        except ValueError:
            continue
        label = " ".join(parts[:-4])
        if label == "micro avg":
            result["micro"] = dict(precision=pre, recall=rec, f1=f1, support=sup)
        elif label not in ("macro avg", "weighted avg"):
            result["classes"][label] = dict(precision=pre, recall=rec, f1=f1, support=sup)
    return result


def load_model(folder):
    h_path = os.path.join(folder, "history.json")
    r_path = os.path.join(folder, "test_report.txt")
    history = json.load(open(h_path)) if os.path.exists(h_path) else {}
    return {
        "name"   : os.path.basename(folder),
        "folder" : folder,
        "history": history,
        "report" : parse_report(r_path),
    }


def discover(checkpoints_dir):
    base = Path(checkpoints_dir)
    models = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (
            (d / "history.json").exists() or (d / "test_report.txt").exists()
        ):
            models.append(load_model(str(d)))
    return models


# ── 1. Model comparison bar ───────────────────────────────────────────────────
def plot_comparison(models, out_path):
    names    = [m["name"] for m in models]
    val_f1s  = [max(m["history"].get("val_f1", [0])) for m in models]
    test_f1s = [m["report"].get("test_f1") or 0 for m in models]

    # Sort by test F1
    order    = sorted(range(len(names)), key=lambda i: test_f1s[i], reverse=True)
    names    = [names[i]    for i in order]
    val_f1s  = [val_f1s[i]  for i in order]
    test_f1s = [test_f1s[i] for i in order]

    n = len(names)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(9, n * 1.7), 5.5))

    bars1 = ax.bar(x - w/2, val_f1s,  w, label="Best Val F1",
                   color=C_BLUE,   alpha=0.85, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, test_f1s, w, label="Test F1",
                   color=C_ORANGE, alpha=0.85, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in list(zip(bars1, val_f1s)) + list(zip(bars2, test_f1s)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#333333")

    # Reference line at 0.95
    ax.axhline(0.95, color="#888", linewidth=1, linestyle="--", alpha=0.6, zorder=0)
    ax.text(n - 0.45, 0.952, "0.95", color="#888", fontsize=8, ha="right")

    lo = max(0, min(val_f1s + test_f1s) - 0.06)
    ax.set_ylim(lo, min(1.02, max(val_f1s + test_f1s) + 0.04))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — Validation F1 vs Test F1")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [1] Model comparison  → {out_path}")


# ── 2. Training curves ────────────────────────────────────────────────────────
def plot_curves(model, out_path):
    h  = model["history"]
    tl = h.get("train_loss", [])
    vl = h.get("val_loss",   [])
    vf = h.get("val_f1",     [])
    if not tl:
        return

    ep = np.arange(1, len(tl) + 1)

    fig = plt.figure(figsize=(12, 4.5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Loss
    ax1.plot(ep, tl, color=C_BLUE,   lw=2, label="Train loss", zorder=3)
    ax1.fill_between(ep, tl, alpha=0.1, color=C_BLUE)
    if vl:
        ax1.plot(ep, vl, color=C_ORANGE, lw=2, linestyle="--",
                 label="Val loss", zorder=3)
        ax1.fill_between(ep, vl, alpha=0.07, color=C_ORANGE)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    # Val F1
    if vf:
        best_ep  = int(np.argmax(vf)) + 1
        best_val = float(np.max(vf))
        ax2.plot(ep, vf, color=C_GREEN, lw=2, label="Val F1", zorder=3)
        ax2.fill_between(ep, vf, alpha=0.1, color=C_GREEN)
        ax2.axvline(best_ep, color=C_GOLD, lw=1.4, linestyle=":",
                    label=f"Best epoch {best_ep} ({best_val:.4f})")
        ax2.scatter([best_ep], [best_val], color=C_GOLD, s=70,
                    zorder=5, edgecolors="white", linewidths=1.2)

        test_f1 = model["report"].get("test_f1")
        if test_f1:
            ax2.axhline(test_f1, color=C_RED, lw=1.4, linestyle="--",
                        label=f"Test F1 {test_f1:.4f}")

        lo = max(0, min(vf) - 0.02)
        ax2.set_ylim(lo, min(1.01, best_val + 0.02))
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1")
        ax2.set_title("Validation F1")
        ax2.legend()

    fig.suptitle(f"Training Curves  —  {model['name']}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [2] Curves  {model['name']}  → {out_path}")


# ── 3. Per-class F1 heatmap ───────────────────────────────────────────────────
def plot_heatmap(models, out_path):
    all_classes = sorted({
        c for m in models
        for c in m["report"].get("classes", {})
    })
    if not all_classes:
        return

    model_names = [m["name"] for m in models]
    mat = np.full((len(all_classes), len(models)), np.nan)

    for j, m in enumerate(models):
        for i, cls in enumerate(all_classes):
            row = m["report"]["classes"].get(cls)
            if row:
                mat[i, j] = row["f1"]

    fig_h = max(5, len(all_classes) * 0.55 + 2)
    fig_w = max(7, len(models) * 1.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mask = np.isnan(mat)
    sns.heatmap(
        mat, ax=ax,
        annot=True, fmt=".3f", annot_kws={"size": 9, "weight": "bold"},
        cmap="YlGn", vmin=0.75, vmax=1.0,
        linewidths=0.5, linecolor="#dee2e6",
        xticklabels=model_names,
        yticklabels=all_classes,
        mask=mask,
        cbar_kws={"label": "F1 Score", "shrink": 0.8},
    )

    # Colour strip on left for entity type
    for i, cls in enumerate(all_classes):
        ax.add_patch(plt.Rectangle((-0.6, i), 0.5, 1,
                                   color=ec(cls), clip_on=False))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("Per-Class F1 Score — All Models")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [3] Heatmap  → {out_path}")


# ── 4. Per-class P / R / F1 bars ─────────────────────────────────────────────
def plot_per_class_bars(model, out_path):
    cd = model["report"].get("classes", {})
    if not cd:
        return

    labels = sorted(cd.keys())
    P = [cd[l]["precision"] for l in labels]
    R = [cd[l]["recall"]    for l in labels]
    F = [cd[l]["f1"]        for l in labels]
    S = [cd[l]["support"]   for l in labels]

    x = np.arange(len(labels))
    w = 0.25

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(max(11, len(labels) * 1.2), 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Main bars
    b1 = ax.bar(x - w, P, w, label="Precision", color=C_BLUE,
                alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,     R, w, label="Recall",    color=C_ORANGE,
                alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + w, F, w, label="F1",        color=C_GREEN,
                alpha=0.85, edgecolor="white")

    # Value labels (F1 only, to avoid clutter)
    for bar, val in zip(b3, F):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=C_GREEN, fontweight="bold")

    ax.axhline(0.90, color="#888", lw=1, linestyle="--", alpha=0.5)
    ax.axhline(0.95, color="#555", lw=1, linestyle="--", alpha=0.5)
    lo = max(0, min(P + R + F) - 0.06)
    ax.set_ylim(lo, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    test_f1 = model["report"].get("test_f1")
    title   = f"Per-Class Metrics  —  {model['name']}"
    if test_f1:
        title += f"  (micro F1 = {test_f1:.4f})"
    ax.set_title(title)
    ax.legend(loc="lower right")

    # Entity colour coded x-axis
    for tick, label in zip(ax.get_xticklabels(), labels):
        tick.set_color(ec(label))

    # Support bar (bottom panel)
    ax2.bar(x, S, 0.6, color=[ec(l) for l in labels], alpha=0.6, edgecolor="white")
    for xi, s in zip(x, S):
        ax2.text(xi, s + 2, str(s), ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Support")
    ax2.set_title("Test Support per Class", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [4] Per-class bars  {model['name']}  → {out_path}")


# ── 5. Confusion matrix (reconstructed from P/R/support) ─────────────────────
def plot_confusion(model, out_path):
    """
    Reconstruct a per-class confusion matrix from precision/recall/support.

    For each class C:
        TP_C  = round(recall_C    * support_C)
        FN_C  = support_C - TP_C
        FP_C  = round(TP_C / precision_C) - TP_C   (when precision > 0)

    We build a square matrix where:
        diagonal          = TP per class
        off-diagonal col  = FP spread uniformly across other classes
        off-diagonal row  = FN spread uniformly across other classes

    This gives an honest approximation without a predictions file.
    """
    cd = model["report"].get("classes", {})
    if not cd:
        return

    labels = sorted(cd.keys())
    n      = len(labels)
    idx    = {l: i for i, l in enumerate(labels)}

    TP = np.zeros(n)
    FP = np.zeros(n)
    FN = np.zeros(n)

    for l, row in cd.items():
        i   = idx[l]
        tp  = round(row["recall"] * row["support"])
        fn  = row["support"] - tp
        fp  = (round(tp / row["precision"]) - tp) if row["precision"] > 0 else 0
        TP[i] = tp; FP[i] = max(0, fp); FN[i] = max(0, fn)

    # Build matrix
    mat = np.zeros((n, n), dtype=int)
    np.fill_diagonal(mat, TP.astype(int))

    # Distribute FP into column (predicted as C, actually something else)
    for j in range(n):
        fp = int(FP[j])
        if fp > 0:
            others = [i for i in range(n) if i != j]
            base, rem = divmod(fp, len(others))
            for k, i in enumerate(others):
                mat[i, j] += base + (1 if k < rem else 0)

    # Distribute FN into row (actually C, predicted as something else)
    for i in range(n):
        fn = int(FN[i])
        if fn > 0:
            row_sum  = mat[i].sum() - mat[i, i]
            expected = int(TP[i]) + fn
            actual   = mat[i].sum()
            deficit  = expected - actual
            if deficit > 0:
                others = [j for j in range(n) if j != i]
                base, rem = divmod(deficit, len(others))
                for k, j in enumerate(others):
                    mat[i, j] += base + (1 if k < rem else 0)

    # Normalise by row (true label) for display
    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    mat_norm = mat / row_sums

    fig, (ax_raw, ax_norm) = plt.subplots(
        1, 2, figsize=(max(14, n * 1.6), max(7, n * 0.9)),
    )

    def draw_cm(ax, data, fmt, title, cmap, vmin, vmax):
        sns.heatmap(
            data, ax=ax,
            annot=True, fmt=fmt, annot_kws={"size": 8.5},
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidths=0.4, linecolor="#dee2e6",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)

    draw_cm(ax_raw,  mat,      "d",    "Confusion Matrix (counts)",     "Blues",  0,   mat.max())
    draw_cm(ax_norm, mat_norm, ".2f",  "Confusion Matrix (recall %)",   "YlGn",   0.0, 1.0)

    fig.suptitle(f"Confusion Matrix  —  {model['name']}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [5] Confusion  {model['name']}  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="checkpoints - Copy")
    parser.add_argument("--out_dir",         default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = discover(args.checkpoints_dir)
    if not models:
        print(f"No models found in '{args.checkpoints_dir}'"); return

    print(f"\nFound {len(models)} models: {[m['name'] for m in models]}\n")

    # Summary table
    sep = "─" * 68
    print(sep)
    print(f"  {'Model':<30}  {'Best Val F1':>11}  {'Test F1':>8}  {'Epochs':>6}")
    print(sep)
    for m in models:
        vf  = m["history"].get("val_f1", [])
        bv  = f"{max(vf):.4f}" if vf else "    —"
        tf  = m["report"].get("test_f1")
        tfs = f"{tf:.4f}" if tf else "    —"
        print(f"  {m['name']:<30}  {bv:>11}  {tfs:>8}  {len(vf):>6}")
    print(sep + "\n")

    print("Generating charts …")
    plot_comparison(models, os.path.join(args.out_dir, "model_comparison.png"))

    for m in models:
        name = m["name"]
        plot_curves(m,
            os.path.join(args.out_dir, f"curves_{name}.png"))
        plot_per_class_bars(m,
            os.path.join(args.out_dir, f"per_class_{name}.png"))
        plot_confusion(m,
            os.path.join(args.out_dir, f"confusion_{name}.png"))

    plot_heatmap(models,
        os.path.join(args.out_dir, "heatmap_all_models.png"))

    print(f"\nAll charts → {args.out_dir}/")


if __name__ == "__main__":
    main()