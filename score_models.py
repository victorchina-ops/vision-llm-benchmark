"""
Score model predictions against ground truth and generate comparison plots.

Scoring weights (100 points total per image):
  total_people         50%  — closeness score: max(0, 1 - |pred-truth| / max(truth,1))
  males                10%
  females              10%
  children             10%
  people_with_backpack 10%
  bicycle_present      10%  — exact boolean match

Outputs:
  scores.csv           — granular: one row per (run, variant, image, metric)
  scores_summary.csv   — overall + per-metric score per (run, variant)
  scores_overall.png   — bar chart: overall weighted score, all models
  scores_by_metric.png — one subplot per metric, all models side by side
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

RUNS = {
    "exp1":          Path("runs/2026-03-31_10-52-52"),
    "exp2":          Path("runs/2026-03-31_14-46-36"),
    "claude-haiku":  Path("runs/claude_haiku"),
    "claude-sonnet": Path("runs/claude_sonnet"),
    "claude-opus":   Path("runs/claude_opus"),
}
GT_PATH = Path("ground_truth.csv")
OUT_DIR = Path(".")

WEIGHTS = {
    "total_people":         0.50,
    "males":                0.10,
    "females":              0.10,
    "children":             0.10,
    "people_with_backpack": 0.10,
    "bicycle_present":      0.10,
}

# colour per run — exp1 blue, exp2 orange, claude-haiku purple
RUN_COLORS = {
    "exp1":          "#5c85d6",
    "exp2":          "#e07b39",
    "claude-haiku":  "#9c27b0",
    "claude-sonnet": "#e91e8c",
    "claude-opus":   "#c0392b",
}

METRIC_DISPLAY = [
    ("total_score",    "Overall\n(weighted)", True),
    ("people_score",   "People count\n(50%)",  False),
    ("males_score",    "Males\n(10%)",          False),
    ("females_score",  "Females\n(10%)",        False),
    ("children_score", "Children\n(10%)",       False),
    ("backpack_score", "Backpack\n(10%)",        False),
    ("bicycle_score",  "Bicycle\n(10%)",         False),
]

# ── Scoring helpers ───────────────────────────────────────────────────────────

def count_score(pred, truth) -> float:
    truth = int(truth)
    pred  = int(pred)
    if truth == 0:
        return 1.0 if pred == 0 else max(0.0, 1.0 - abs(pred))
    return max(0.0, 1.0 - abs(pred - truth) / truth)


def bool_score(pred, truth) -> float:
    def to_bool(v):
        if isinstance(v, bool): return v
        return str(v).strip().lower() in ("true", "1", "yes")
    return 1.0 if to_bool(pred) == to_bool(truth) else 0.0


# ── Load ground truth ─────────────────────────────────────────────────────────

gt = pd.read_csv(GT_PATH)
gt["bicycle_present"] = gt["bicycle_present"].astype(str).str.strip().str.lower() \
                            .map({"true": True, "false": False, "1": True, "0": False})

# ── Score all runs ────────────────────────────────────────────────────────────

all_rows     = []
summary_rows = []

for run_label, run_dir in RUNS.items():
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    df = df[~df["parse_error"]]

    for (family, quant, model_tag), grp in df.groupby(["family", "quant", "model_tag"]):
        variant_label = f"{family}/{quant}"
        per_image = []

        for _, pred_row in grp.iterrows():
            img    = pred_row["image"]
            gt_row = gt[gt["image"] == img]
            if gt_row.empty:
                continue
            gt_row = gt_row.iloc[0]

            ms = {
                "total_people":         count_score(pred_row["total_people"],         gt_row["total_people"]),
                "males":                count_score(pred_row["males"],                gt_row["males"]),
                "females":              count_score(pred_row["females"],              gt_row["females"]),
                "children":             count_score(pred_row["children"],             gt_row["children"]),
                "people_with_backpack": count_score(pred_row["people_with_backpack"], gt_row["people_with_backpack"]),
                "bicycle_present":      bool_score( pred_row["bicycle_present"],      gt_row["bicycle_present"]),
            }
            weighted = sum(ms[m] * WEIGHTS[m] for m in WEIGHTS) * 100

            for metric, score in ms.items():
                all_rows.append({
                    "run": run_label, "family": family, "quant": quant,
                    "model_tag": model_tag, "variant": variant_label,
                    "image": img, "metric": metric,
                    "score_0_1": round(score, 4),
                    "weight": WEIGHTS[metric],
                    "weighted_pts": round(score * WEIGHTS[metric] * 100, 2),
                })
            all_rows.append({
                "run": run_label, "family": family, "quant": quant,
                "model_tag": model_tag, "variant": variant_label,
                "image": img, "metric": "TOTAL",
                "score_0_1": round(weighted / 100, 4),
                "weight": 1.0, "weighted_pts": round(weighted, 2),
            })
            per_image.append(ms | {"_weighted": weighted})

        if not per_image:
            continue
        tmp = pd.DataFrame(per_image)
        summary_rows.append({
            "run":            run_label,
            "family":         family,
            "quant":          quant,
            "model_tag":      model_tag,
            "variant":        variant_label,
            "total_score":    round(tmp["_weighted"].mean(), 2),
            "people_score":   round(tmp["total_people"].mean() * 100, 2),
            "males_score":    round(tmp["males"].mean() * 100, 2),
            "females_score":  round(tmp["females"].mean() * 100, 2),
            "children_score": round(tmp["children"].mean() * 100, 2),
            "backpack_score": round(tmp["people_with_backpack"].mean() * 100, 2),
            "bicycle_score":  round(tmp["bicycle_present"].mean() * 100, 2),
        })

# ── Save CSVs ─────────────────────────────────────────────────────────────────

pd.DataFrame(all_rows).to_csv(OUT_DIR / "scores.csv", index=False)
print(f"Saved: scores.csv")

summary_df = pd.DataFrame(summary_rows).sort_values("total_score", ascending=False)
summary_df.to_csv(OUT_DIR / "scores_summary.csv", index=False)
print(f"Saved: scores_summary.csv")

print("\n=== Overall Scores ===")
print(summary_df[["run", "variant", "total_score", "people_score",
                   "males_score", "females_score", "children_score",
                   "backpack_score", "bicycle_score"]].to_string(index=False))

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _bar_colors(df):
    return [RUN_COLORS.get(r, "#888") for r in df["run"]]


def _ytick_labels(df):
    return [f"[{r}]  {v}" for r, v in zip(df["run"], df["variant"])]


def _legend_patches():
    return [mpatches.Patch(color=c, label=lbl) for lbl, c in RUN_COLORS.items()
            if lbl in summary_df["run"].values]


# ── Plot 1: Overall score ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, max(6, len(summary_df) * 0.45)))
fig.suptitle("Overall Weighted Score vs Ground Truth\n(people 50% · gender 20% · children 10% · backpack 10% · bicycle 10%)",
             fontsize=12, fontweight="bold")

x      = np.arange(len(summary_df))
colors = _bar_colors(summary_df)
bars   = ax.barh(x, summary_df["total_score"], color=colors, edgecolor="white", height=0.7)
ax.set_xlim(0, 110)
ax.set_yticks(x)
ax.set_yticklabels(_ytick_labels(summary_df), fontsize=8)
ax.set_xlabel("Score (0–100)", fontsize=9)
ax.axvline(100, color="#ccc", linewidth=0.8, linestyle="--")
for bar, score in zip(bars, summary_df["total_score"]):
    ax.text(min(score + 0.8, 103), bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}", va="center", fontsize=8, fontweight="bold")

ax.legend(handles=_legend_patches(), fontsize=9, loc="lower right")
plt.tight_layout()
fig.savefig(OUT_DIR / "scores_overall.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: scores_overall.png")

# ── Plot 2: Per-metric breakdown ──────────────────────────────────────────────

metric_cols   = [c for c, _, overall in METRIC_DISPLAY if not overall]
metric_titles = {c: t for c, t, overall in METRIC_DISPLAY if not overall}

n_metrics = len(metric_cols)
fig, axes = plt.subplots(1, n_metrics,
                          figsize=(3.8 * n_metrics, max(6, len(summary_df) * 0.45)),
                          sharey=True)
fig.suptitle("Score per Metric vs Ground Truth — All Models", fontsize=13, fontweight="bold")

for ax, col in zip(axes, metric_cols):
    colors = _bar_colors(summary_df)
    bars   = ax.barh(np.arange(len(summary_df)), summary_df[col],
                     color=colors, edgecolor="white", height=0.7)
    ax.set_xlim(0, 115)
    ax.set_title(metric_titles[col], fontsize=9, fontweight="bold")
    ax.axvline(100, color="#ccc", linewidth=0.8, linestyle="--")
    for bar, score in zip(bars, summary_df[col]):
        ax.text(min(score + 0.8, 104), bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}", va="center", fontsize=7)

axes[0].set_yticks(np.arange(len(summary_df)))
axes[0].set_yticklabels(_ytick_labels(summary_df), fontsize=7)

fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
fig.savefig(OUT_DIR / "scores_by_metric.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved: scores_by_metric.png")
