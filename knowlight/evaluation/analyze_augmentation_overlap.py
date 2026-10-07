#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph Augmentation Candidate vs Test Set Overlap Analyzer
=============================================================================
This module performs a comprehensive empirical evaluation of the semantic pseudo-edges
injected during Graph Augmentation against the ground-truth items in the evaluation test set.

It investigates two critical dimensions across any Top-K cutoff:
1. Exact Item ID Overlap (Data Leakage & Exact Hit Rate):
   Verifies if and for how many users the exact test item ID was added as a pseudo-edge.
2. Category / Genre Semantic Overlap (Semantic Alignment):
   Verifies whether the injected candidate items share the same semantic category / genre
   as the user's future test item.

The analysis is performed for both:
- tau = 0.0 (Unthresholded candidate selection, e.g., augmented_graphs_1)
- tau > 0.0 (Optimal statistical thresholding, e.g., augmented_graphs_3)
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np
import scipy.sparse as sp

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowlight.datasets import KnowLightDataLoader

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_DATASETS = [
    "movielens",
    "amazon_cd_2014",
    "amazon_toys_and_games",
    "amazon_beauty",
    "yelp"
]

DEFAULT_KS = [1, 2, 5, 8, 10]
DEFAULT_TEACHER = "Llama-3-8B-Instruct"

DATASET_DISPLAY_NAMES = {
    "movielens": "MovieLens-1M",
    "amazon_cd_2014": "Amazon CD & Vinyl",
    "amazon_toys_and_games": "Amazon Toys & Games",
    "amazon_beauty": "Amazon Beauty",
    "yelp": "Yelp"
}


# ==============================================================================
# 1. CATEGORY & GENRE PARSER
# ==============================================================================

def extract_categories_from_text(text: str) -> Set[str]:
    """
    Extracts set of normalized genres / categories from metadata text strings
    supporting MovieLens, Amazon, and Yelp conventions.
    """
    categories: Set[str] = set()
    if not text:
        return categories

    # 1. MovieLens format: "Genres: Animation|Children's|Comedy"
    if "Genres:" in text:
        after = text.split("Genres:")[1].strip()
        for g in after.split("|"):
            g_clean = g.strip().lower()
            if g_clean:
                categories.add(g_clean)

    # 2. Amazon / Yelp format: "Categories: Cat1 > Cat2 ... |" or "Categories: Cat1, Cat2 ... |"
    if "Categories:" in text:
        part = text.split("Categories:")[1].split("|")[0].strip()
        delim = ">" if ">" in part else ("," if "," in part else None)
        if delim:
            for c in part.split(delim):
                c_clean = c.strip().lower()
                # Exclude root/generic category names
                if c_clean and c_clean not in {"cds & vinyl", "toys & games", "beauty", "all beauty", "movies & tv"}:
                    categories.add(c_clean)
        elif part.strip():
            categories.add(part.strip().lower())

    return categories


def load_item_categories_map(dataset_name: str, num_items: int) -> Dict[int, Set[str]]:
    """
    Loads item-to-categories mapping for all catalog items (0-indexed).
    """
    handled_json = PROJECT_ROOT / "data" / "raw" / dataset_name / "handled" / "item_text.json"
    raw_meta_file = PROJECT_ROOT / "data" / "raw" / dataset_name / "item_metadata.json"

    item_categories: Dict[int, Set[str]] = {}

    # 1. Try handled item_text.json (1-indexed keys for Amazon / Yelp)
    if handled_json.exists():
        try:
            with open(handled_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                try:
                    idx = int(k) - 1
                    if 0 <= idx < num_items:
                        item_categories[idx] = extract_categories_from_text(str(v))
                except ValueError:
                    continue
            if item_categories:
                return item_categories
        except Exception as e:
            logging.debug(f"Failed loading handled/item_text.json ({e})")

    # 2. Try raw item_metadata.json (1-indexed or raw id keys)
    if raw_meta_file.exists():
        try:
            with open(raw_meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for i in range(num_items):
                str_i = str(i + 1)
                if str_i in meta:
                    item_categories[i] = extract_categories_from_text(str(meta[str_i]))
                elif str(i) in meta:
                    item_categories[i] = extract_categories_from_text(str(meta[str(i)]))
            return item_categories
        except Exception as e:
            logging.debug(f"Failed loading item_metadata.json ({e})")

    return item_categories


# ==============================================================================
# 2. CACHE DIRECTORY RESOLVER
# ==============================================================================

def find_augmented_dirs(cache_base: Path) -> List[Tuple[int, Path]]:
    """Lists and sorts all augmented_graphs* directories by index ascending."""
    pattern = re.compile(r"^augmented_graphs(?:_?(\d+))?$")
    dirs = []
    if cache_base.exists():
        for d in cache_base.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    idx = int(m.group(1)) if m.group(1) else 1
                    dirs.append((idx, d))
    dirs.sort(key=lambda x: (x[0], x[1].name))
    return dirs


def resolve_threshold_directories(
    cache_base: Path,
    user_tau_zero_dir: Optional[str] = None,
    user_tau_opt_dir: Optional[str] = None
) -> Tuple[Optional[Tuple[str, Path, float]], Optional[Tuple[str, Path, float]]]:
    """
    Identifies the target directories for:
    1. tau = 0.0 (Unthresholded candidate selection)
    2. tau > 0.0 (Optimal statistical threshold)
    Returns: ((label, Path, tau_value), (label, Path, tau_value))
    """
    all_dirs = find_augmented_dirs(cache_base)
    if not all_dirs:
        return None, None

    tau_zero_info = None
    tau_opt_info = None

    # Helper to inspect files in a directory and extract threshold value
    def inspect_dir_tau(d: Path) -> Tuple[bool, float]:
        npz_files = list(d.glob("*.npz"))
        for f in npz_files:
            m = re.search(r"th([\d\.]+)", f.stem)
            if m:
                val = float(m.group(1))
                return True, val
        # If no explicit th tag in filename, assume 0.0
        return bool(npz_files), 0.0

    # Helper to resolve a directory path flexibly from relative root, cache_base, or pure folder name
    def resolve_dir_path(user_input: str) -> Optional[Path]:
        raw_p = Path(user_input)
        # 1. Absolute path
        if raw_p.is_absolute() and raw_p.exists():
            return raw_p
        # 2. Relative to current working dir or PROJECT_ROOT
        if raw_p.exists():
            return raw_p.resolve()
        if (PROJECT_ROOT / raw_p).exists():
            return (PROJECT_ROOT / raw_p).resolve()
        # 3. Path inside cache_base
        if (cache_base / raw_p).exists():
            return (cache_base / raw_p).resolve()
        # 4. Folder name inside cache_base
        if (cache_base / raw_p.name).exists():
            return (cache_base / raw_p.name).resolve()
        return None

    # Explicit user overrides
    if user_tau_zero_dir:
        p0 = resolve_dir_path(user_tau_zero_dir)
        if p0 is not None and p0.exists():
            _, t0 = inspect_dir_tau(p0)
            tau_zero_info = (p0.name, p0, t0)
            logging.info(f"Using user-specified tau=0 directory: '{p0.name}' ({p0})")
        else:
            logging.warning(f"Could not locate --tau_zero_dir: '{user_tau_zero_dir}'. Falling back to automatic detection.")

    if user_tau_opt_dir:
        p1 = resolve_dir_path(user_tau_opt_dir)
        if p1 is not None and p1.exists():
            _, t1 = inspect_dir_tau(p1)
            tau_opt_info = (p1.name, p1, t1)
            logging.info(f"Using user-specified tau>0 directory: '{p1.name}' ({p1})")
        else:
            logging.warning(f"Could not locate --tau_opt_dir: '{user_tau_opt_dir}'. Falling back to automatic detection.")

    # Automatic discovery (only if not explicitly overridden by user)
    for idx, d in all_dirs:
        has_files, tau_val = inspect_dir_tau(d)
        if not has_files:
            continue
        if tau_val == 0.0 and user_tau_zero_dir is None and tau_zero_info is None:
            tau_zero_info = (d.name, d, 0.0)
        elif tau_val > 0.0 and user_tau_opt_dir is None:
            # Keep the latest directory with tau > 0.0
            tau_opt_info = (d.name, d, tau_val)

    # Fallback for tau_zero: use first directory if tau_zero_info is still None
    if tau_zero_info is None and all_dirs:
        first_d = all_dirs[0][1]
        _, t = inspect_dir_tau(first_d)
        tau_zero_info = (first_d.name, first_d, t)

    return tau_zero_info, tau_opt_info


# ==============================================================================
# 3. CORE EVALUATION ENGINE
# ==============================================================================

def evaluate_graph_overlap_for_k(
    graph_path: Path,
    k_val: int,
    tau_val: float,
    train_dict: Dict[int, Set[int]],
    test_pos_items: Dict[int, List[int]],
    item_categories: Dict[int, Set[str]],
    num_test_users: int
) -> Dict[str, Any]:
    """
    Evaluates both ID overlap and Category/Genre overlap for a given augmented graph file.
    """
    augmented_r = sp.load_npz(graph_path).tocsr()

    hits_id_count = 0
    hits_genre_count = 0
    users_with_pseudo_edges = 0
    total_pseudo_edges_count = 0
    users_with_evaluable_genre = 0

    for u, t_items in test_pos_items.items():
        if not t_items:
            continue

        test_item = t_items[0]
        test_genres = item_categories.get(test_item, set())

        train_items = train_dict.get(u, set())
        user_row = augmented_r.getrow(u)
        nonzero_cols = set(user_row.indices)

        pseudo_items = nonzero_cols - train_items
        if pseudo_items:
            users_with_pseudo_edges += 1
            total_pseudo_edges_count += len(pseudo_items)

        # 1. Exact ID Overlap
        if test_item in pseudo_items:
            hits_id_count += 1

        # 2. Category / Genre Overlap
        if test_genres and pseudo_items:
            users_with_evaluable_genre += 1
            pseudo_genres = set()
            for p in pseudo_items:
                pseudo_genres.update(item_categories.get(p, set()))
            if pseudo_genres.intersection(test_genres):
                hits_genre_count += 1

    pct_id_all = (hits_id_count / num_test_users) * 100 if num_test_users > 0 else 0.0
    pct_id_edges = (hits_id_count / users_with_pseudo_edges) * 100 if users_with_pseudo_edges > 0 else 0.0

    pct_genre_eval = (hits_genre_count / users_with_evaluable_genre) * 100 if users_with_evaluable_genre > 0 else 0.0
    pct_genre_edges = (hits_genre_count / users_with_pseudo_edges) * 100 if users_with_pseudo_edges > 0 else 0.0
    pct_genre_all = (hits_genre_count / num_test_users) * 100 if num_test_users > 0 else 0.0

    avg_edges_per_user = total_pseudo_edges_count / max(num_test_users, 1)

    return {
        "k": k_val,
        "tau": tau_val,
        "filename": graph_path.name,
        "num_test_users": num_test_users,
        "users_with_pseudo_edges": users_with_pseudo_edges,
        "avg_pseudo_edges_per_user": avg_edges_per_user,
        # Exact ID Hit metrics
        "id_hits_count": hits_id_count,
        "id_hits_pct_test_users": pct_id_all,
        "id_hits_pct_users_with_edges": pct_id_edges,
        # Category / Genre Match metrics
        "genre_hits_count": hits_genre_count,
        "genre_evaluable_users": users_with_evaluable_genre,
        "genre_hits_pct_evaluable": pct_genre_eval,
        "genre_hits_pct_users_with_edges": pct_genre_edges,
        "genre_hits_pct_test_users": pct_genre_all
    }


def analyze_dataset(
    dataset_name: str,
    teacher_name: str = DEFAULT_TEACHER,
    ks: Optional[List[int]] = None,
    tau_zero_dir_override: Optional[str] = None,
    tau_opt_dir_override: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Runs complete overlap evaluation for both tau=0.0 and tau>0.0 on a given dataset."""
    ks = ks or DEFAULT_KS
    display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.upper())

    print("\n" + "=" * 95)
    print(f" GRAPH AUGMENTATION CANDIDATES VS TEST SET OVERLAP ANALYSIS: {display_name.upper()}")
    print("=" * 95)

    # 1. Load Data
    try:
        dl = KnowLightDataLoader(dataset_name=dataset_name)
        dl.load_data()
    except Exception as e:
        logging.error(f"Failed to load dataset '{dataset_name}': {e}")
        return None

    train_dict = dl.train_dict
    test_pos_items = dl.test_pos_items
    num_users = dl.num_users
    num_items = dl.num_items
    num_test_users = len([u for u in test_pos_items.keys() if len(test_pos_items[u]) > 0])

    print(f"Total users: {num_users} | Catalog items: {num_items} | Test set users: {num_test_users}")

    # 2. Load Item Categories
    item_categories = load_item_categories_map(dataset_name, num_items)
    has_categories = len(item_categories) > 0
    print(f"Semantic metadata (Genres / Categories): {'Available for ' + str(len(item_categories)) + ' items' if has_categories else 'Not available'}")

    # 3. Locate Target Cache Directories
    teacher_tag = Path(teacher_name).name.replace("/", "_")
    cache_base = PROJECT_ROOT / "data" / "cache" / teacher_tag / dataset_name

    tau_zero_info, tau_opt_info = resolve_threshold_directories(
        cache_base,
        user_tau_zero_dir=tau_zero_dir_override,
        user_tau_opt_dir=tau_opt_dir_override
    )

    targets = []
    if tau_zero_info is not None:
        targets.append(("tau_zero", tau_zero_info[0], tau_zero_info[1], tau_zero_info[2]))
    if tau_opt_info is not None:
        targets.append(("tau_opt", tau_opt_info[0], tau_opt_info[1], tau_opt_info[2]))

    if not targets:
        print(f"[Notice] No augmented graph directories found in: {cache_base}")
        return None

    dataset_results = {
        "dataset": dataset_name,
        "display_name": display_name,
        "num_users": num_users,
        "num_items": num_items,
        "num_test_users": num_test_users,
        "has_categories": has_categories,
        "evaluations": {}
    }

    # 4. Run Evaluation for each regime (tau=0.0 and tau>0.0)
    for tag, dir_name, dir_path, tau_val in targets:
        cond_label = f"Unthresholded (tau=0.0)" if tau_val == 0.0 else f"Optimal Threshold (tau={tau_val})"
        print(f"\n--- Regime: {cond_label} | Directory: {dir_name} ---")

        npz_files = list(dir_path.glob("*.npz"))
        regime_results = {}

        for k in ks:
            matching = [f for f in npz_files if f.name.startswith(f"aug_r_k{k}_")]
            if not matching:
                continue

            file_to_load = matching[0]
            res = evaluate_graph_overlap_for_k(
                graph_path=file_to_load,
                k_val=k,
                tau_val=tau_val,
                train_dict=train_dict,
                test_pos_items=test_pos_items,
                item_categories=item_categories,
                num_test_users=num_test_users
            )
            regime_results[str(k)] = res

        dataset_results["evaluations"][tag] = {
            "label": cond_label,
            "dir_name": dir_name,
            "tau": tau_val,
            "results_per_k": regime_results
        }

    # 5. Print Formatted Console Table
    _print_dataset_summary_table(dataset_results, ks)

    return dataset_results


def _print_dataset_summary_table(data: Dict[str, Any], ks: List[int]):
    """Pretty prints summary comparison table for console."""
    has_cats = data.get("has_categories", False)
    evals = data.get("evaluations", {})

    print(f"\nCOMPLETE OVERLAP RESULTS ({data['display_name']}):")
    header = f"{'Top-K':<6} | {'Regime / Threshold':<28} | {'Users w/ Edges':<15} | {'ID Hits (%)':<15} | {'Genre/Cat Match (%)':<22}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for k in ks:
        str_k = str(k)
        for tag, tag_data in evals.items():
            res_k = tag_data["results_per_k"].get(str_k)
            if not res_k:
                continue

            regime_desc = f"{tag_data['dir_name']} (tau={tag_data['tau']})"
            users_e = f"{res_k['users_with_pseudo_edges']} ({res_k['users_with_pseudo_edges']/data['num_test_users']*100:.1f}%)"
            id_hit = f"{res_k['id_hits_count']} ({res_k['id_hits_pct_test_users']:.3f}%)"

            if has_cats and res_k['genre_evaluable_users'] > 0:
                genre_hit = f"{res_k['genre_hits_count']} ({res_k['genre_hits_pct_evaluable']:.2f}%)"
            else:
                genre_hit = "N/A"

            print(f"k={k:<4} | {regime_desc:<28} | {users_e:<15} | {id_hit:<15} | {genre_hit:<22}")
        print("-" * len(header))
    print("=" * len(header) + "\n")


# ==============================================================================
# 4. REPORT GENERATOR (JSON & MARKDOWN)
# ==============================================================================

def _write_dataset_markdown_report(f, ds_key: str, data: Dict[str, Any], ks: List[int]):
    """Writes detailed markdown table for a single dataset."""
    f.write(f"# KnowLight: Graph Augmentation vs Test Set Overlap Analysis - {data['display_name']}\n\n")
    f.write(f"This report presents the empirical overlap analysis for **{data['display_name']}** between Top-$K$ semantic candidates ")
    f.write("injected via **Graph Augmentation (Pipeline 3)** and the ground-truth target item in the **Test Set**, ")
    f.write("comparing both the unthresholded regime ($\\tau=0.0$) and the optimal statistical threshold regime ($\\tau > 0.0$).\n\n")

    f.write(f"- **Total users:** {data['num_users']} | **Catalog items:** {data['num_items']} | **Test set users:** {data['num_test_users']}\n\n")

    f.write("## Detailed Breakdown across Top-K Cutoffs\n\n")
    f.write("| $K$ | Regime | Cache Directory | Users w/ Edges | Exact ID Hits | % Test Users | Category/Genre Matches | % Coherent Users |\n")
    f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    evals = data.get("evaluations", {})
    for k in ks:
        str_k = str(k)
        for tag, tag_data in evals.items():
            res_k = tag_data["results_per_k"].get(str_k)
            if not res_k:
                continue
            r_label = f"$\\tau={tag_data['tau']}$"
            c_dir = f"`{tag_data['dir_name']}`"
            u_edges = f"{res_k['users_with_pseudo_edges']} ({res_k['users_with_pseudo_edges']/data['num_test_users']*100:.1f}%)"
            id_hits = f"{res_k['id_hits_count']}"
            id_pct = f"{res_k['id_hits_pct_test_users']:.3f}%"

            if res_k.get("genre_evaluable_users", 0) > 0:
                cat_hits = f"{res_k['genre_hits_count']}"
                cat_pct = f"{res_k['genre_hits_pct_evaluable']:.2f}%"
            else:
                cat_hits = "N/A"
                cat_pct = "N/A"

            f.write(f"| {k} | {r_label} | {c_dir} | {u_edges} | {id_hits} | {id_pct} | {cat_hits} | **{cat_pct}** |\n")

    f.write("\n")


def export_reports(all_results: Dict[str, Any], output_dir: Path, ks: List[int]):
    """Saves both JSON raw data and formatted Markdown report including dataset name in filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export per-dataset individual reports with dataset name in filename
    for ds_key, data in all_results.items():
        ds_json_path = output_dir / f"augmentation_overlap_summary_{ds_key}.json"
        with open(ds_json_path, "w", encoding="utf-8") as f:
            json.dump({ds_key: data}, f, indent=2)
        print(f"[Report] JSON data for '{ds_key}' saved to: {ds_json_path}")

        ds_md_path = output_dir / f"augmentation_overlap_report_{ds_key}.md"
        with open(ds_md_path, "w", encoding="utf-8") as f:
            _write_dataset_markdown_report(f, ds_key, data, ks)
        print(f"[Report] Markdown document for '{ds_key}' saved to: {ds_md_path}")

    # 2. Combined report if multiple datasets are analyzed
    if len(all_results) > 1:
        combined_json_path = output_dir / "augmentation_overlap_summary_all.json"
        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"[Report] Combined JSON data saved to: {combined_json_path}")

        combined_md_path = output_dir / "augmentation_overlap_report_all.md"
        with open(combined_md_path, "w", encoding="utf-8") as f:
            f.write("# KnowLight: Comprehensive Graph Augmentation vs Test Set Overlap Analysis\n\n")
            f.write("This report presents the empirical overlap analysis between Top-$K$ semantic candidates injected ")
            f.write("via **Graph Augmentation (Pipeline 3)** and the ground-truth target item in the **Test Set**, ")
            f.write("comparing both the unthresholded regime ($\\tau=0.0$) and the optimal statistical threshold regime ($\\tau > 0.0$).\n\n")

            f.write("## 1. Summary Table at $K=2$ (Optimal Thesis Configuration)\n\n")
            f.write("| Dataset | Catalog ($|I|$) | Test Users | Overlap ID ($\\tau=0$) | Overlap ID ($\\tau>0$) | Category/Genre Coherence ($\\tau>0$) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

            for ds_key, data in all_results.items():
                num_items = data["num_items"]
                num_test = data["num_test_users"]
                evals = data.get("evaluations", {})

                # tau=0 results at k=2
                t0_res = evals.get("tau_zero", {}).get("results_per_k", {}).get("2")
                t0_str = f"{t0_res['id_hits_count']} ({t0_res['id_hits_pct_test_users']:.3f}%)" if t0_res else "N/A"

                # tau>0 results at k=2
                t1_res = evals.get("tau_opt", {}).get("results_per_k", {}).get("2")
                t1_str = f"{t1_res['id_hits_count']} ({t1_res['id_hits_pct_test_users']:.3f}%)" if t1_res else "N/A"

                # category overlap at k=2
                if t1_res and t1_res.get("genre_evaluable_users", 0) > 0:
                    cat_str = f"{t1_res['genre_hits_count']} ({t1_res['genre_hits_pct_evaluable']:.2f}%)"
                elif t0_res and t0_res.get("genre_evaluable_users", 0) > 0:
                    cat_str = f"{t0_res['genre_hits_count']} ({t0_res['genre_hits_pct_evaluable']:.2f}%)"
                else:
                    cat_str = "N/A"

                f.write(f"| **{data['display_name']}** | {num_items} | {num_test} | {t0_str} | {t1_str} | **{cat_str}** |\n")

            f.write("\n---\n\n")

            f.write("## 2. Detailed Breakdown per Dataset and Top-K Cutoff\n\n")
            for ds_key, data in all_results.items():
                f.write(f"### {data['display_name']}\n\n")
                f.write(f"- **Total users:** {data['num_users']} | **Catalog items:** {data['num_items']} | **Test set users:** {data['num_test_users']}\n\n")

                f.write("| $K$ | Regime | Cache Directory | Users w/ Edges | Exact ID Hits | % Test Users | Category/Genre Matches | % Coherent Users |\n")
                f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

                evals = data.get("evaluations", {})
                for k in ks:
                    str_k = str(k)
                    for tag, tag_data in evals.items():
                        res_k = tag_data["results_per_k"].get(str_k)
                        if not res_k:
                            continue
                        r_label = f"$\\tau={tag_data['tau']}$"
                        c_dir = f"`{tag_data['dir_name']}`"
                        u_edges = f"{res_k['users_with_pseudo_edges']} ({res_k['users_with_pseudo_edges']/data['num_test_users']*100:.1f}%)"
                        id_hits = f"{res_k['id_hits_count']}"
                        id_pct = f"{res_k['id_hits_pct_test_users']:.3f}%"

                        if res_k.get("genre_evaluable_users", 0) > 0:
                            cat_hits = f"{res_k['genre_hits_count']}"
                            cat_pct = f"{res_k['genre_hits_pct_evaluable']:.2f}%"
                        else:
                            cat_hits = "N/A"
                            cat_pct = "N/A"

                        f.write(f"| {k} | {r_label} | {c_dir} | {u_edges} | {id_hits} | {id_pct} | {cat_hits} | **{cat_pct}** |\n")

                f.write("\n")

        print(f"[Report] Combined Markdown report saved to: {combined_md_path}")


# ==============================================================================
# 5. CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KnowLight: Unified Graph Augmentation vs Test Set Overlap Analysis (tau=0 and tau>0 across Top-K cutoffs)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to analyze ('all', or specific comma-separated names: movielens, amazon_cd_2014, amazon_toys_and_games, amazon_beauty, yelp)"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=DEFAULT_TEACHER,
        help="Teacher LLM model name (default: Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=DEFAULT_KS,
        help="List of Top-K cutoffs to evaluate (default: 1 2 5 8 10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reports/augmentation_overlap",
        help="Output directory to save JSON and Markdown reports (default: data/reports/augmentation_overlap)"
    )
    parser.add_argument(
        "--tau_zero_dir",
        type=str,
        default=None,
        help="Directory name override for tau=0.0 (default: auto-detect)"
    )
    parser.add_argument(
        "--tau_opt_dir",
        type=str,
        default=None,
        help="Directory name override for tau>0.0 (default: auto-detect)"
    )

    args = parser.parse_args()

    # Determine datasets list
    if args.dataset == "all":
        datasets_to_run = DEFAULT_DATASETS
    else:
        datasets_to_run = [d.strip() for d in args.dataset.split(",")]

    print("\n" + "=" * 95)
    print(" STARTING UNIFIED GRAPH AUGMENTATION VS TEST SET OVERLAP ANALYSIS")
    print(f" Teacher: {args.teacher} | Top-K values: {args.ks}")
    print(f" Target datasets: {datasets_to_run}")
    print("=" * 95)

    all_summary = {}

    for ds in datasets_to_run:
        res = analyze_dataset(
            dataset_name=ds,
            teacher_name=args.teacher,
            ks=args.ks,
            tau_zero_dir_override=args.tau_zero_dir,
            tau_opt_dir_override=args.tau_opt_dir
        )
        if res:
            all_summary[ds] = res

    # Export reports
    out_dir = PROJECT_ROOT / args.output_dir
    export_reports(all_summary, out_dir, args.ks)



if __name__ == "__main__":
    main()
