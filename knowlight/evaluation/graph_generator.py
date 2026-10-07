import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Directory with generated reports
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def _find_latest_report_json(base_dir: Path, subfolder_name: str, prefix: str, ds_key: str) -> Optional[Path]:
    """Finds the latest JSON report file (supporting subfolder and _2, _3 numbering)."""
    search_dirs = [base_dir / subfolder_name, base_dir]
    base_stem = f"{prefix}_{ds_key}"
    pattern = re.compile(rf"^{re.escape(base_stem)}(?:_(\d+))?\.json$")

    candidates = []
    for s_dir in search_dirs:
        if s_dir.exists():
            for f in s_dir.glob(f"{base_stem}*.json"):
                m = pattern.match(f.name)
                if m:
                    idx = int(m.group(1)) if m.group(1) else 1
                    candidates.append((idx, f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _get_next_figure_filepaths(target_dir: Path, base_stem: str) -> Tuple[Path, Path]:
    """
    Returns next available PDF and PNG file paths using sequential numbering:
    1st: {base_stem}.pdf / .png
    2nd: {base_stem}_2.pdf / .png
    3rd: {base_stem}_3.pdf / .png
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    first_pdf = target_dir / f"{base_stem}.pdf"
    first_png = target_dir / f"{base_stem}.png"

    if not first_pdf.exists() and not first_png.exists():
        return first_pdf, first_png

    pattern = re.compile(rf"^{re.escape(base_stem)}_(\d+)\.(?:pdf|png)$")
    existing_nums = [1]
    for f in target_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                existing_nums.append(int(m.group(1)))

    next_num = max(existing_nums) + 1
    return target_dir / f"{base_stem}_{next_num}.pdf", target_dir / f"{base_stem}_{next_num}.png"


# 5 Canonical datasets with curated labels, colors, and markers
DATASETS_CONFIG = {
    "amazon_toys_and_games": {"label": "Amazon Toys & Games", "color": "#1f77b4", "marker": "o"},
    "movielens":             {"label": "MovieLens-1M",        "color": "#2ca02c", "marker": "s"},
    "amazon_cd_2014":        {"label": "Amazon CD & Vinyl",   "color": "#ff7f0e", "marker": "^"},
    "amazon_beauty":         {"label": "Amazon Beauty",       "color": "#9467bd", "marker": "D"},
    "yelp":                  {"label": "Yelp",                "color": "#d62728", "marker": "v"},
}

DATASET_DISPLAY_NAMES = {
    "movielens": "MovieLens-1M",
    "amazon_cd_2014": "Amazon CD",
    "amazon_toys_and_games": "Amazon Toys",
    "amazon_beauty": "Amazon Beauty",
    "yelp": "Yelp"
}

# ==============================================================================
# 1. SPARSITY / DEGREE-STRATIFIED GRAPH GENERATION
# ==============================================================================

BUCKETS = ["bucket_1", "bucket_2", "bucket_3"]
BUCKET_LABELS = [
    "Bucket 1\n(Sparse / Near-Cold)",
    "Bucket 2\n(Medium)",
    "Bucket 3\n(Dense / Warm)"
]

SPARSITY_PANELS = [
    {"metric": "NDCG@5",  "title": "(a) Relative Gain on Top-5 Ranking (Δ% NDCG@5)"},
    {"metric": "NDCG@20", "title": "(b) Relative Gain on Full Ranking (Δ% NDCG@20)"}
]


def generate_sparsity_graph(reports_dir: Optional[Path] = None) -> bool:
    """
    Generates side-by-side line plots showing relative improvement (Δ%)
    across user interaction terciles (sparsity buckets) for NDCG@5 and NDCG@20.
    """
    reports_dir = reports_dir or REPORTS_DIR
    datasets_data = {}
    for ds_key in DATASETS_CONFIG.keys():
        json_path = _find_latest_report_json(reports_dir, "sparsity_analysis", "sparsity_analysis", ds_key)
        if json_path and json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                datasets_data[ds_key] = json.load(f)
            logging.info(f"[Sparsity Graph] Loaded latest report for '{ds_key}': {json_path.name}")
        else:
            print(f"[Notice] Sparsity report not found for '{ds_key}'. Skipping.")

    if not datasets_data:
        print(f"[Warning] No sparsity reports found in {reports_dir}. Skipping sparsity graph generation.")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=300)

    for ax, panel in zip(axes, SPARSITY_PANELS):
        metric = panel["metric"]

        # Reference baseline line at 0%
        ax.axhline(0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8, label="Baseline (0%)")

        for ds_key, cfg in DATASETS_CONFIG.items():
            if ds_key not in datasets_data:
                continue

            data = datasets_data[ds_key]
            deltas = data["delta_improvements_percent"]
            y_values = [deltas[b][metric] for b in BUCKETS]

            ax.plot(
                BUCKET_LABELS,
                y_values,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=2.2,
                markersize=7.5
            )

        ax.set_title(panel["title"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("User Interaction Terciles", fontsize=10.5, labelpad=8)
        ax.set_ylabel(f"Relative Improvement over Baseline (Δ% {metric})", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=9.5)

    # Unified shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=len(handles),
            frameon=True,
            facecolor="white",
            edgecolor="#dddddd",
            fontsize=9.5
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    target_dir = reports_dir / "sparsity_analysis" if (reports_dir / "sparsity_analysis").exists() else reports_dir
    output_pdf, output_png = _get_next_figure_filepaths(target_dir, "sparsity_graph")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Sparsity Graph] Successfully generated plots:")
    print(f"- PDF: {output_pdf}")
    print(f"- PNG: {output_png}")
    return True


# ==============================================================================
# 2. BEYOND-ACCURACY (COVERAGE & DIVERSITY) GRAPH GENERATION
# ==============================================================================

COVERAGE_DIVERSITY_PANELS = [
    {
        "metric_prefix": "Coverage",
        "title": "(a) Catalog Coverage (%)",
        "ylabel": "Catalog Coverage (%)",
        "format": "{:.1f}%"
    },
    {
        "metric_prefix": "Personalization",
        "title": "(b) Personalization (Inter-User Diversity)",
        "ylabel": "Personalization (1 - Cosine Sim)",
        "format": "{:.3f}"
    },
    {
        "metric_prefix": "IntraListDiversity",
        "title": "(c) Intra-List Diversity (ILD)",
        "ylabel": "Mean Pairwise Cosine Distance",
        "format": "{:.3f}"
    },
    {
        "metric_prefix": "Novelty",
        "title": "(d) Novelty (Self-Information in bits)",
        "ylabel": "Average Novelty (bits)",
        "format": "{:.2f}"
    }
]


def generate_coverage_diversity_graph(k: int = 10, reports_dir: Optional[Path] = None) -> bool:
    """
    Generates a 2x2 publication-quality comparison figure:
    Baseline LightGCN vs Pipeline 3 (Intent-Augmented) across all available datasets.
    """
    reports_dir = reports_dir or REPORTS_DIR
    dataset_keys = list(DATASETS_CONFIG.keys())

    reports = {}
    for ds_key in dataset_keys:
        json_path = _find_latest_report_json(reports_dir, "coverage_analysis", "coverage_diversity", ds_key)
        if json_path and json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                reports[ds_key] = json.load(f)
            logging.info(f"[Coverage Graph] Loaded latest report for '{ds_key}': {json_path.name}")

    available_datasets = [ds for ds in dataset_keys if ds in reports]

    if not available_datasets:
        print(f"[Notice] No coverage & diversity reports found in {reports_dir}. Skipping diversity graph.")
        return False

    print(f"\n[Coverage & Diversity Graph] Found reports for {len(available_datasets)} datasets: {available_datasets}")

    color_baseline = "#5B6E85"      # Muted slate blue/gray
    color_pipeline = "#2E7D32"      # Rich emerald green
    color_delta_pos = "#1B5E20"     # Dark green badge
    color_delta_neg = "#B71C1C"     # Dark red badge

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=300)
    axes_flat = axes.flatten()

    x = np.arange(len(available_datasets))
    bar_width = 0.35
    x_labels = [DATASET_DISPLAY_NAMES.get(ds, ds) for ds in available_datasets]

    for ax_idx, panel in enumerate(COVERAGE_DIVERSITY_PANELS):
        ax = axes_flat[ax_idx]
        metric_name = f"{panel['metric_prefix']}@{k}"

        baseline_vals = []
        pipeline_vals = []
        delta_vals = []

        for ds in available_datasets:
            data = reports[ds]
            b_val = data.get("baseline_metrics", {}).get(metric_name, 0.0)
            p_val = data.get("pipeline_metrics", {}).get(metric_name, 0.0)
            d_val = data.get("delta_percent", {}).get(metric_name, 0.0)

            baseline_vals.append(b_val)
            pipeline_vals.append(p_val)
            delta_vals.append(d_val)

        bars_base = ax.bar(
            x - bar_width / 2,
            baseline_vals,
            width=bar_width,
            label="Baseline LightGCN",
            color=color_baseline,
            edgecolor="#333333",
            linewidth=0.8,
            alpha=0.9
        )
        bars_pipe = ax.bar(
            x + bar_width / 2,
            pipeline_vals,
            width=bar_width,
            label="Pipeline 3 (Intent-Augmented)",
            color=color_pipeline,
            edgecolor="#1E4620",
            linewidth=0.8,
            alpha=0.95
        )

        max_val = max(max(baseline_vals or [0.0]), max(pipeline_vals or [0.0]))
        min_val = min(min(baseline_vals or [0.0]), min(pipeline_vals or [0.0]))
        val_range = max_val - min_val if max_val > min_val else (max_val if max_val > 0 else 1.0)
        ax.set_ylim(0, max_val + val_range * 0.28)

        fmt = panel["format"]
        for b_bar, p_bar, d_val, b_val, p_val in zip(
            bars_base, bars_pipe, delta_vals, baseline_vals, pipeline_vals
        ):
            ax.text(
                b_bar.get_x() + b_bar.get_width() / 2,
                b_bar.get_height() + val_range * 0.015,
                fmt.format(b_val),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#444444",
                fontweight="medium"
            )

            ax.text(
                p_bar.get_x() + p_bar.get_width() / 2,
                p_bar.get_height() + val_range * 0.015,
                fmt.format(p_val),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#1B5E20",
                fontweight="bold"
            )

            delta_color = color_delta_pos if d_val >= 0 else color_delta_neg
            badge_text = f"{d_val:+.1f}%"
            ax.text(
                p_bar.get_x() + p_bar.get_width() / 2,
                p_bar.get_height() + val_range * 0.085,
                badge_text,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=delta_color,
                    edgecolor="none",
                    alpha=0.9
                )
            )

        ax.set_title(panel["title"], fontsize=12.5, fontweight="bold", pad=12)
        ax.set_ylabel(panel["ylabel"], fontsize=10.5, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight="medium")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=9.5)
        ax.set_axisbelow(True)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    target_dir = reports_dir / "coverage_analysis" if (reports_dir / "coverage_analysis").exists() else reports_dir
    out_pdf, out_png = _get_next_figure_filepaths(target_dir, f"coverage_diversity_graph_k{k}")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Coverage & Diversity Graph] Successfully generated plots (K={k}):")
    print(f"- PDF: {out_pdf}")
    print(f"- PNG: {out_png}")
    return True


# ==============================================================================
# 3. PIPELINE COMPARISON GRAPH GENERATION (OVERALL NDCG@20)
# ==============================================================================

# Verified benchmark results across the 3 pipelines and Baseline LightGCN
# Pipeline 1: Score Distillation
# Pipeline 2: Contrastive Embedding Distillation
# Pipeline 3: Intent-Guided Graph Augmentation (evaluated with optimal tau)

VERIFIED_PIPELINE_RESULTS = {
    "movielens": {
        "baseline": 0.0574,
        "pipeline_1": 0.0574,
        "pipeline_2": 0.0588,
        "pipeline_3": 0.0573,
    },
    "amazon_cd_2014": {
        "baseline": 0.0420,
        "pipeline_1": 0.0420,
        "pipeline_2": 0.0442,
        "pipeline_3": 0.0433,
    },
    "amazon_toys_and_games": {
        "baseline": 0.0161,
        "pipeline_1": 0.0161,
        "pipeline_2": 0.0167,
        "pipeline_3": 0.0166,
    },
    "amazon_beauty": {
        "baseline": 0.0153,
        "pipeline_1": 0.0153,
        "pipeline_2": 0.0151,
        "pipeline_3": 0.0153,
    },
    "yelp": {
        "baseline": 0.0268,
        "pipeline_1": 0.0273,
        "pipeline_2": 0.0284,
        "pipeline_3": 0.0281,
    },
}

PIPELINE_COMPARISON_DATASETS = [
    "movielens",
    "amazon_cd_2014",
    "amazon_toys_and_games",
    "amazon_beauty",
    "yelp",
]


def generate_pipeline_comparison_graph(
    reports_dir: Optional[Path] = None,
    show_deltas: bool = True
) -> bool:
    """
    Generates a grouped bar chart comparing the three KnowLight distillation pipelines
    against the baseline (LightGCN) on NDCG@20 across all 5 benchmark datasets.

    - X-axis: 5 datasets (MovieLens-1M, Amazon CD, Amazon Toys, Amazon Beauty, Yelp).
    - Y-axis: NDCG@20.
    - 3 grouped columns per dataset:
        1. Pipeline 1: Score Distillation
        2. Pipeline 2: Contrastive Distillation
        3. Pipeline 3: Graph Augmentation (Best tau)
    - Baseline: A horizontal dashed gray line across each dataset's 3-column cluster.
    """
    reports_dir = reports_dir or REPORTS_DIR

    # Load dynamic values for Baseline and Pipeline 3 from latest sparsity analysis if available
    comparison_data = {}
    for ds_key in PIPELINE_COMPARISON_DATASETS:
        defaults = dict(VERIFIED_PIPELINE_RESULTS[ds_key])
        json_path = _find_latest_report_json(reports_dir, "sparsity_analysis", "sparsity_analysis", ds_key)
        if json_path and json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                b_ndcg = rep.get("baseline_metrics", {}).get("overall", {}).get("NDCG@20")
                p3_ndcg = rep.get("pipeline_metrics", {}).get("overall", {}).get("NDCG@20")
                if b_ndcg is not None and p3_ndcg is not None:
                    defaults["baseline"] = round(b_ndcg, 4)
                    defaults["pipeline_3"] = round(p3_ndcg, 4)
                    logging.info(
                        f"[Pipeline Comparison] Dynamically verified '{ds_key}' from {json_path.name}: "
                        f"Baseline={defaults['baseline']:.4f}, Pipeline 3 (best tau)={defaults['pipeline_3']:.4f}"
                    )
            except Exception as e:
                logging.warning(f"Could not parse {json_path}: {e}. Falling back to verified defaults.")
        comparison_data[ds_key] = defaults

    # Plot configuration
    x_labels = [DATASET_DISPLAY_NAMES.get(k, k) for k in PIPELINE_COMPARISON_DATASETS]
    x = np.arange(len(PIPELINE_COMPARISON_DATASETS))
    width = 0.24

    baseline_vals = [comparison_data[k]["baseline"] for k in PIPELINE_COMPARISON_DATASETS]
    p1_vals = [comparison_data[k]["pipeline_1"] for k in PIPELINE_COMPARISON_DATASETS]
    p2_vals = [comparison_data[k]["pipeline_2"] for k in PIPELINE_COMPARISON_DATASETS]
    p3_vals = [comparison_data[k]["pipeline_3"] for k in PIPELINE_COMPARISON_DATASETS]

    c_p1 = "#2980B9"  # Strong Blue (Pipeline 1: Score Distillation)
    c_p2 = "#E67E22"  # Warm Amber / Orange (Pipeline 2: Contrastive Distillation)
    c_p3 = "#27AE60"  # Emerald Green (Pipeline 3: Graph Augmentation)
    c_base = "#333333" # Charcoal Gray for baseline line

    fig, ax = plt.subplots(figsize=(13.8, 6.4), dpi=300)

    rects1 = ax.bar(
        x - width, p1_vals, width,
        label="Pipeline 1: Score Distillation",
        color=c_p1, alpha=0.92, edgecolor="#1B4F72", linewidth=0.6
    )
    rects2 = ax.bar(
        x, p2_vals, width,
        label="Pipeline 2: Contrastive Distillation",
        color=c_p2, alpha=0.92, edgecolor="#7E5109", linewidth=0.6
    )
    rects3 = ax.bar(
        x + width, p3_vals, width,
        label=r"Pipeline 3: Graph Augmentation (Best $\tau$)",
        color=c_p3, alpha=0.92, edgecolor="#145A32", linewidth=0.6
    )

    # Draw horizontal baseline reference line across each 3-column cluster
    for i in range(len(PIPELINE_COMPARISON_DATASETS)):
        b_val = baseline_vals[i]
        x_min = x[i] - 1.62 * width
        x_max = x[i] + 1.62 * width
        ax.hlines(
            b_val, x_min, x_max,
            colors=c_base, linestyles="--", linewidth=2.2, zorder=5,
            label="Baseline (LightGCN)" if i == 0 else ""
        )

    # Add numeric labels and delta badges above bars
    def autolabel_bars(rects, base_vals, text_color):
        for i, rect in enumerate(rects):
            h = rect.get_height()
            b = base_vals[i]
            diff_pct = ((h - b) / b) * 100 if b > 0 else 0.0

            val_str = f"{h:.4f}"
            if not show_deltas:
                full_text = val_str
            else:
                if abs(diff_pct) < 0.1:
                    delta_str = "0.0%"
                elif diff_pct > 0:
                    delta_str = f"+{diff_pct:.1f}%"
                else:
                    delta_str = f"{diff_pct:.1f}%"
                full_text = f"{val_str}\n({delta_str})"

            is_positive = (diff_pct > 0.1)
            ax.annotate(
                full_text,
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold" if is_positive else "normal",
                color=text_color,
                linespacing=0.95
            )

    autolabel_bars(rects1, baseline_vals, "#1A5276")
    autolabel_bars(rects2, baseline_vals, "#935116")
    autolabel_bars(rects3, baseline_vals, "#196F3D")

    ax.set_ylabel("NDCG@20", fontsize=12.5, fontweight="bold", labelpad=10)
    ax.set_title("Overall Performance Comparison Across Pipelines (NDCG@20)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11.5, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=10.5)
    ax.set_ylim(0, 0.073)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=11,
        loc="upper right"
    )

    plt.tight_layout()

    out_pdf, out_png = _get_next_figure_filepaths(reports_dir, "pipeline_comparison_ndcg20")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Pipeline Comparison Graph] Successfully generated plots:")
    print(f"- PDF: {out_pdf}")
    print(f"- PNG: {out_png}")
    return True


# ==============================================================================
# 4. UNIFIED CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KnowLight Unified Thesis Graph Generator (Sparsity, Diversity & Pipelines)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "sparsity", "diversity", "pipelines"],
        default="all",
        help="Type of graph to generate: 'all' (default, generates all 3), 'sparsity', 'diversity', or 'pipelines'"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K cutoff for coverage & diversity graph (default: 10)"
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="data/reports",
        help="Path to directory containing reports JSON files (default: data/reports)"
    )

    args = parser.parse_args()
    rep_dir = PROJECT_ROOT / args.reports_dir

    if args.type in ["all", "sparsity"]:
        generate_sparsity_graph(reports_dir=rep_dir)

    if args.type in ["all", "diversity"]:
        generate_coverage_diversity_graph(k=args.k, reports_dir=rep_dir)

    if args.type in ["all", "pipelines"]:
        generate_pipeline_comparison_graph(reports_dir=rep_dir)


if __name__ == "__main__":
    main()

