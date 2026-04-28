#!/usr/bin/env python3
"""
Visualization pipeline for SIMD Finance Benchmarks paper.
Generates figures and LaTeX tables from Google Benchmark JSON outputs.
"""

import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FIGURES = PROJECT_ROOT / "paper" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "paper" / "tables"

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# Pattern class mapping (as defined in paper Section 1.4)
PATTERN_CLASSES = {
    "vwap": "Reduction-heavy",
    "ewma": "Dependency chain (IIR)",
    "book_imbalance": "Masked/Conditional",
    "kyle_lambda": "Division-heavy",
    "atr": "Sliding Window + IIR",
    "rsi": "Dual IIR + Masked",
    "sharpe_ratio": "Reduction + sqrt",
    "sortino_ratio": "Reduction + Masked + sqrt",
    "max_drawdown": "Sequential Max + Cumulative",
    "calmar_ratio": "Sequential Max + Division",
    "correlation": "Multiple Reductions + sqrt",
    "beta": "Multiple Reductions + Division",
    "alpha": "Multiple Reductions + Linear",
    "var": "Selection / Sorting",
    "convolution": "Sliding Window + FIR",
    "cvar": "Selection + Tail Reduction",
}

# Kernel display names for plots
KERNEL_NAMES = {
    "vwap": "VWAP",
    "ewma": "EWMA",
    "book_imbalance": "Book Imb.",
    "kyle_lambda": "Kyle λ",
    "atr": "ATR",
    "rsi": "RSI",
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "max_drawdown": "Max DD",
    "calmar_ratio": "Calmar",
    "correlation": "Corr.",
    "beta": "Beta",
    "alpha": "Alpha",
    "var": "VaR",
    "convolution": "Conv.",
    "cvar": "CVaR",
}

# ─────────────────────────────────────────────────────────────
# Data Loading & Parsing
# ─────────────────────────────────────────────────────────────
def parse_benchmark_json(filepath: Path) -> pd.DataFrame:
    """Parse Google Benchmark JSON output into DataFrame."""
    with open(filepath) as f:
        data = json.load(f)

    rows = []
    for bench in data["benchmarks"]:
        if bench.get("run_type") == "aggregate":
            continue
        if bench.get("run_type") != "iteration":
            continue

        name = bench["name"].strip()
        # Pattern: BM_KERNEL_Impl/Size
        match = re.match(r"BM_([A-Z_a-z0-9]+)_(.+)/(\d+)", name)
        if not match:
            continue

        kernel, impl, size = match.groups()
        kernel = kernel.lower()
        impl = impl.lower()

        impl_mapping = {
            "scalar": "scalar",
            "simd": "std_simd",
            "std_simd": "std_simd",
            # Intrinsics variants
            "avx": "avx2",
            "avx2": "avx2",
            "avx512": "avx512",  # ✅ Added AVX-512 support
            "sse": "sse4",
            "sse4": "sse4",
            "neon": "neon",
        }
        impl = impl_mapping.get(impl, impl)

        # Remove _std suffix if present
        if kernel.endswith("_std"):
            kernel = kernel[:-4]

        rows.append({
            "kernel": kernel,
            "impl": impl,
            "size": int(size),
            "time_ns": bench["real_time"],
            "cpu_ns": bench["cpu_time"],
            "bytes_per_sec": bench.get("bytes_per_second", 0),
            "iterations": bench["iterations"],
        })

    return pd.DataFrame(rows)

def load_all_results() -> pd.DataFrame:
    """Load and merge all benchmark JSON files from results/."""
    if not RESULTS_DIR.exists():
        raise RuntimeError(f"Results directory not found: {RESULTS_DIR}")

    dfs = []
    json_files = list(RESULTS_DIR.glob("*_results.json"))
    if not json_files:
        raise RuntimeError(f"No *_results.json files found in {RESULTS_DIR}")

    print(f"   Found {len(json_files)} JSON files")
    for json_file in json_files:
        try:
            df = parse_benchmark_json(json_file)
            if not df.empty:
                dfs.append(df)
                print(f"   ✓ Parsed: {json_file.name} ({len(df)} entries)")
        except Exception as e:
            print(f"   ⚠️  Error parsing {json_file.name}: {e}")

    if not dfs:
        raise RuntimeError("No valid benchmark data found in any JSON file")

    return pd.concat(dfs, ignore_index=True)

def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Compute speedup vs scalar for each kernel/size."""
    speedups = []

    print(f"   [debug] Total rows in DataFrame: {len(df)}")
    print(f"   [debug] Unique kernels: {df['kernel'].unique()}")
    print(f"   [debug] Unique impls: {df['impl'].unique()}")

    for kernel in df["kernel"].unique():
        kernel_df = df[df["kernel"] == kernel]
        if "size" not in kernel_df.columns:
            continue

        for size in kernel_df["size"].unique():
            subset = kernel_df[kernel_df["size"] == size]
            impls_in_subset = subset["impl"].unique()
            print(f"   [debug] {kernel}/{size}: impls = {impls_in_subset}")

            scalar_rows = subset[subset["impl"] == "scalar"]
            if len(scalar_rows) == 0:
                continue

            scalar_time = scalar_rows["time_ns"].values[0]

            for _, row in subset.iterrows():
                if row["impl"] == "scalar" or row["time_ns"] <= 0:
                    continue
                speedup = scalar_time / row["time_ns"]
                speedups.append({
                    "kernel": kernel,
                    "size": int(size),
                    "impl": row["impl"],
                    "speedup": speedup,
                    "time_ns": row["time_ns"],
                    "pattern": PATTERN_CLASSES.get(kernel, "Unknown"),
                })

    if not speedups:
        return pd.DataFrame(columns=["kernel", "size", "impl", "speedup", "time_ns", "pattern"])

    return pd.DataFrame(speedups)

# ─────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────
def plot_speedup_by_pattern(df: pd.DataFrame):
    """Figure 1: Speedup vs Scalar, grouped by pattern class."""
    if df.empty or "size" not in df.columns:
        return

    target_size = 1 << 20
    available_sizes = df["size"].unique()
    if target_size not in available_sizes and len(available_sizes) > 0:
        target_size = min(available_sizes, key=lambda x: abs(x - (1 << 20)))

    agg = df[(df["size"] == target_size)].groupby(["kernel", "pattern", "impl"])["speedup"].median().reset_index()
    if agg.empty:
        return

    agg["kernel_display"] = agg["kernel"].map(KERNEL_NAMES)
    pivot = agg.pivot_table(index=["kernel_display", "pattern"], columns="impl", values="speedup")

    color_map = {
        "std_simd": "#4E79A7",  # Blue
        "avx2": "#F28E2B",      # Orange
        "avx512": "#59A14F",    # Green
        "sse4": "#E15759",      # Red
        "neon": "#76B7B2",      # Teal
    }
    available_impls = [c for c in pivot.columns if c in color_map]
    if not available_impls:
        return

    plt.figure(figsize=(14, max(8, len(available_sizes) * 0.4 + len(pivot) * 0.35)))
    ax = pivot[available_impls].plot(
        kind="barh",
        color={k: color_map[k] for k in available_impls},
        edgecolor="black",
        width=0.8,
        alpha=0.9
    )

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.5, label="Scalar baseline")
    ax.set_xlabel("Speedup vs Scalar (higher is better)")
    ax.set_ylabel("Kernel")
    # ax.set_title(...)  # ❌ Убрано по запросу
    ax.legend(title="Implementation", loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f×", padding=2, fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "speedup_by_pattern.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_FIGURES / "speedup_by_pattern.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Saved: {OUTPUT_FIGURES / 'speedup_by_pattern.pdf'}")


def plot_api_gap(df: pd.DataFrame):
    """Figure 2: API Gap (std::simd vs Intrinsics) by pattern class."""
    if not any("avx" in str(i) for i in df["impl"].unique()):
        return

    plt.figure(figsize=(12, 7))
    gap_data = []
    target_size = 1 << 20

    for kernel in df["kernel"].unique():
        subset = df[(df["kernel"] == kernel) & (df["size"] == target_size)]
        simd_time = subset[subset["impl"] == "std_simd"]["time_ns"].values
        avx_times = subset[subset["impl"].str.contains("avx", case=False, na=False)]["time_ns"].values

        if len(simd_time) > 0 and len(avx_times) > 0:
            gap = (simd_time[0] - avx_times.min()) / avx_times.min() * 100
            gap_data.append({
                "kernel": KERNEL_NAMES.get(kernel, kernel),
                "pattern": PATTERN_CLASSES.get(kernel, "Unknown"),
                "gap_percent": max(0, gap),
            })

    if not gap_data:  # ✅ Исправлено (было: if not gap_)
        return

    gap_df = pd.DataFrame(gap_data).sort_values("gap_percent", ascending=False)
    ax = sns.barplot(data=gap_df, x="gap_percent", y="kernel", hue="pattern", palette="Set2", dodge=False)

    ax.axvline(x=10.0, color="red", linestyle=":", linewidth=1, label="10% threshold")
    ax.set_xlabel("API Gap: (std::simd − Intrinsics) / Intrinsics [%]")
    ax.set_ylabel("Kernel")
    # ax.set_title(...)  # ❌ Убрано
    ax.legend(title="Pattern Class", loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "api_gap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_FIGURES / "api_gap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Saved: {OUTPUT_FIGURES / 'api_gap.pdf'}")


def plot_scaling(df: pd.DataFrame):
    """Figure 3: Speedup vs Problem Size for representative kernels."""
    plt.figure(figsize=(10, 6))
    representatives = {
        "alpha": "Multiple Reductions",
        "ewma": "Dependency Chain",
        "book_imbalance": "Masked/Conditional",
    }
    colors = {"std_simd": "#4E79A7"}

    for kernel, label in representatives.items():
        subset = df[(df["kernel"] == kernel) & (df["impl"].isin(["std_simd"]))]
        if subset.empty:
            continue
        plt.plot(subset["size"], subset["speedup"], marker="o",
                 label=f"{KERNEL_NAMES[kernel]} ({label})", color=colors["std_simd"], linestyle="-")

    plt.xscale("log", base=2)
    plt.xlabel("Problem Size (elements, log scale)")
    plt.ylabel("Speedup vs Scalar")
    # plt.title(...)  # ❌ Убрано
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.3, which="both")
    plt.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "scaling.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_FIGURES / "scaling.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Saved: {OUTPUT_FIGURES / 'scaling.pdf'}")


def plot_pattern_heatmap(df: pd.DataFrame):
    """Figure 4: Heatmap of median speedup by Pattern × Implementation."""
    plt.figure(figsize=(10, 6))
    target_size = 1 << 20
    available_sizes = df["size"].unique()
    if target_size not in available_sizes and len(available_sizes) > 0:
        target_size = min(available_sizes, key=lambda x: abs(x - (1 << 20)))

    agg = df[(df["size"] == target_size)].groupby(["pattern", "impl"])["speedup"].median().reset_index()
    if agg.empty:
        return

    pivot = agg.pivot(index="pattern", columns="impl", values="speedup")
    impl_order = ["std_simd", "avx2", "avx512", "sse4", "neon"]
    pivot = pivot[[c for c in impl_order if c in pivot.columns]]

    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Median Speedup vs Scalar"})
    ax.set_xlabel("Implementation")
    ax.set_ylabel("Pattern Class")
    # ax.set_title(...)  # ❌ Убрано

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "pattern_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_FIGURES / "pattern_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Saved: {OUTPUT_FIGURES / 'pattern_heatmap.pdf'}")


# ─────────────────────────────────────────────────────────────
# LaTeX Table Generation
# ─────────────────────────────────────────────────────────────
def generate_latex_tables(df: pd.DataFrame):
    """Generate publication-ready LaTeX tables for Section 4."""
    target_size = 1 << 20
    available_sizes = df["size"].unique()
    if target_size not in available_sizes and len(available_sizes) > 0:
        target_size = min(available_sizes, key=lambda x: abs(x - (1 << 20)))

    table1 = df[(df["size"] == target_size)].copy()
    table1["pattern"] = table1["kernel"].map(PATTERN_CLASSES)
    table1["kernel_display"] = table1["kernel"].map(KERNEL_NAMES)

    pivot = table1.pivot_table(
        index=["kernel_display", "pattern"],
        columns="impl",
        values="speedup",
        aggfunc="median"
    ).round(2)

    impl_order = ["std_simd", "avx2", "avx512", "sse4", "neon"]
    available_cols = [c for c in impl_order if c in pivot.columns]

    header_cols = " & ".join([r"\textbf{" + impl.replace("_", "::") + r"}" for impl in available_cols])

    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{SIMD Speedup vs Scalar Baseline (~1M elements, median)}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{@{}l l " + "c" * len(available_cols) + r"@{}}",
        r"\toprule",
        r"\textbf{Kernel} & \textbf{Pattern Class} & " + header_cols + r" \\",
        r"\midrule"
    ]

    for (kernel, pattern), row in pivot.iterrows():
        values = []
        for impl in available_cols:
            val = row.get(impl)
            if pd.notna(val) and isinstance(val, (int, float)):
                values.append(f"{val:.2f}×")
            else:
                values.append("--")
        tex_lines.append(f"{kernel} & {pattern} & " + " & ".join(values) + r" \\")

    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open(OUTPUT_TABLES / "main_results.tex", "w") as f:
        f.write("\n".join(tex_lines))
    print(f"✓ Saved: {OUTPUT_TABLES / 'main_results.tex'}")


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def main():
    print("🔍 Loading benchmark results...")
    df_raw = load_all_results()
    print(f"   Loaded {len(df_raw)} benchmark entries")

    if not df_raw.empty and "size" in df_raw.columns:
        print(f"   Sizes: {sorted(df_raw['size'].unique())}")
        print(f"   Impls: {df_raw['impl'].unique()}")

    print("📈 Computing speedups...")
    df_speedup = compute_speedup(df_raw)
    if df_speedup.empty:
        print("⚠️  Speedup DataFrame empty. Skipping visualization.")
        return

    print("🎨 Generating figures...")
    plot_speedup_by_pattern(df_speedup)
    plot_api_gap(df_speedup)
    plot_scaling(df_speedup)
    plot_pattern_heatmap(df_speedup)

    print("📄 Generating LaTeX tables...")
    generate_latex_tables(df_speedup)
    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.titlesize": 12,
    })
    main()