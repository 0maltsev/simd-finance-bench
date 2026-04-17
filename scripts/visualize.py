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
# Determine project root (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ✅ Path to JSON results (as per your structure)
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
        # ✅ Skip aggregate rows (*_mean, *_median, *_stddev, *_cv)
        if bench.get("run_type") == "aggregate":
            continue

        # Skip if not an iteration run
        if bench.get("run_type") != "iteration":
            continue

        # Clean name (remove trailing spaces from JSON)
        name = bench["name"].strip()

        # ✅ FIX: Parse BM_KERNEL_Impl/Size where Impl can contain underscores
        # Pattern: BM_ + kernel + _ + impl (anything until /) + / + size
        match = re.match(r"BM_([A-Z_a-z0-9]+)_(.+)/(\d+)", name)
        if not match:
            continue

        kernel, impl, size = match.groups()
        kernel = kernel.lower()
        impl = impl.lower()

        # ✅ Normalize implementation names
        impl_mapping = {
            "scalar": "scalar",
            "simd": "std_simd",
            "std_simd": "std_simd",
            # Intrinsics variants
            "avx": "avx2",
            "avx2": "avx2",
            "sse": "sse4",
            "sse4": "sse4",
            "neon": "neon",
        }
        impl = impl_mapping.get(impl, impl)

        # ✅ Normalize kernel names (remove _std suffix if present)
        # e.g., "correlation_std" → "correlation"
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
            print(f"   [debug] Skipping {kernel}: no 'size' column")
            continue

        for size in kernel_df["size"].unique():
            subset = kernel_df[kernel_df["size"] == size]

            # Debug: check what we have for this kernel/size
            impls_in_subset = subset["impl"].unique()
            print(f"   [debug] {kernel}/{size}: impls = {impls_in_subset}")

            # Find scalar time
            scalar_rows = subset[subset["impl"] == "scalar"]
            if len(scalar_rows) == 0:
                print(f"   [debug]   ⚠️  No scalar found for {kernel}/{size}")
                continue

            scalar_time = scalar_rows["time_ns"].values[0]
            print(f"   [debug]   scalar_time = {scalar_time:.2f} ns")

            # Compute speedup for all non-scalar implementations
            for _, row in subset.iterrows():
                if row["impl"] == "scalar":
                    continue
                if row["time_ns"] <= 0:
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
                print(f"   [debug]   {row['impl']}: speedup = {speedup:.2f}×")

    if not speedups:
        print("   ⚠️  Warning: No speedup data computed. Check input data format.")
        return pd.DataFrame(columns=["kernel", "size", "impl", "speedup", "time_ns", "pattern"])

    result_df = pd.DataFrame(speedups)
    print(f"   [debug] Speedup DataFrame: {len(result_df)} entries")
    print(f"   [debug] Speedup columns: {list(result_df.columns)}")
    return result_df


# ─────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────
def plot_speedup_by_pattern(df: pd.DataFrame):
    """Figure 1: Speedup vs Scalar, grouped by pattern class."""
    if df.empty or "size" not in df.columns:
        print("   ⚠️  Skipping plot_speedup_by_pattern: empty DataFrame or missing 'size' column")
        return

    plt.figure(figsize=(14, 8))

    # Aggregate: median speedup at 1M elements (representative size)
    target_size = 1 << 20
    available_sizes = df["size"].unique()
    if target_size not in available_sizes and len(available_sizes) > 0:
        # Use closest available size
        target_size = min(available_sizes, key=lambda x: abs(x - (1 << 20)))
        print(f"   [debug] Using alternative size {target_size} instead of 1M")

    agg = df[(df["size"] == target_size)].groupby(["kernel", "pattern", "impl"])["speedup"].median().reset_index()

    if agg.empty:
        print("   ⚠️  No data for plot_speedup_by_pattern at size", target_size)
        return

    agg["kernel_display"] = agg["kernel"].map(KERNEL_NAMES)

    # Pivot for grouped bar chart
    pivot = agg.pivot_table(index=["kernel_display", "pattern"], columns="impl", values="speedup")

    # Plot
    color_map = {"std_simd": "#4E79A7"}
    available_impls = [c for c in color_map.keys() if c in pivot.columns]

    if not available_impls:
        print("   ⚠️  No valid implementations found for plotting")
        return

    ax = pivot[available_impls].plot(kind="barh", figsize=(14, 10),
                                     color={k: color_map[k] for k in available_impls},
                                     edgecolor="black", width=0.8)

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.5, label="Scalar baseline")
    ax.set_xlabel("Speedup vs Scalar (higher is better)")
    ax.set_ylabel("Kernel")
    ax.set_title("Figure 1: SIMD Speedup by Pattern Class (~1M elements, median)")
    ax.legend(title="Implementation", loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "speedup_by_pattern.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_FIGURES / "speedup_by_pattern.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Saved: {OUTPUT_FIGURES / 'speedup_by_pattern.pdf'}")


def plot_api_gap(df: pd.DataFrame):
    """Figure 2: API Gap (std::simd vs Intrinsics) by pattern class."""
    # Skip if no AVX implementations
    if not any("avx" in str(i) for i in df["impl"].unique()):
        print("   ⚠️  Skipping API gap plot: no AVX implementations found")
        return

    plt.figure(figsize=(12, 7))

    # Compute gap: (intrinsics_time - simd_time) / intrinsics_time * 100
    gap_data = []
    for kernel in df["kernel"].unique():
        for size in [1 << 20]:  # Representative size
            subset = df[(df["kernel"] == kernel) & (df["size"] == size)]

            simd_time = subset[subset["impl"] == "std_simd"]["time_ns"].values
            avx_times = subset[subset["impl"].str.contains("avx", case=False, na=False)]["time_ns"].values

            if len(simd_time) > 0 and len(avx_times) > 0:
                gap = (simd_time[0] - avx_times.min()) / avx_times.min() * 100
                pattern = PATTERN_CLASSES.get(kernel, "Unknown")
                gap_data.append({
                    "kernel": KERNEL_NAMES.get(kernel, kernel),
                    "pattern": pattern,
                    "gap_percent": max(0, gap),  # Clamp to >= 0
                })

    if not gap_data:
        print("   ⚠️  No gap data to plot")
        return

    gap_df = pd.DataFrame(gap_data).sort_values("gap_percent", ascending=False)

    # Horizontal bar chart
    ax = sns.barplot(data=gap_df, x="gap_percent", y="kernel",
                     hue="pattern", palette="Set2", dodge=False)

    ax.axvline(x=10.0, color="red", linestyle=":", linewidth=1, label="10% threshold")
    ax.set_xlabel("API Gap: (std::simd − Intrinsics) / Intrinsics [%]")
    ax.set_ylabel("Kernel")
    ax.set_title("Figure 2: API Overhead of std::simd vs Intrinsics (1M elements)")
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

    # Select 3 representative kernels from different pattern classes
    representatives = {
        "alpha": "Multiple Reductions",  # Best case
        "ewma": "Dependency Chain",      # Worst case
        "book_imbalance": "Masked/Conditional", # Mid case
    }

    colors = {"std_simd": "#4E79A7"}

    for kernel, label in representatives.items():
        subset = df[(df["kernel"] == kernel) & (df["impl"].isin(["std_simd"]))]
        if subset.empty:
            continue
        plt.plot(subset["size"], subset["speedup"],
                 marker="o", label=f"{KERNEL_NAMES[kernel]} ({label})",
                 color=colors["std_simd"], linestyle="-")

    plt.xscale("log", base=2)
    plt.xlabel("Problem Size (elements, log scale)")
    plt.ylabel("Speedup vs Scalar")
    plt.title("Figure 3: Speedup Scaling with Problem Size")
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

    # Aggregate: median speedup at 1M elements
    target_size = 1 << 20
    available_sizes = df["size"].unique()
    if target_size not in available_sizes and len(available_sizes) > 0:
        target_size = min(available_sizes, key=lambda x: abs(x - (1 << 20)))

    agg = df[(df["size"] == target_size)].groupby(["pattern", "impl"])["speedup"].median().reset_index()

    if agg.empty:
        print("   ⚠️  No data for heatmap")
        return

    pivot = agg.pivot(index="pattern", columns="impl", values="speedup")

    # Reorder columns
    impl_order = ["std_simd"]
    pivot = pivot[[c for c in impl_order if c in pivot.columns]]

    # Plot heatmap
    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                     cbar_kws={"label": "Median Speedup vs Scalar"})

    ax.set_xlabel("Implementation")
    ax.set_ylabel("Pattern Class")
    ax.set_title("Figure 4: Median Speedup by Pattern Class × Implementation (~1M elements)")

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

    # Pivot: rows=kernels, columns=impl, values=speedup
    pivot = table1.pivot_table(
        index=["kernel_display", "pattern"],
        columns="impl",
        values="speedup",
        aggfunc="median"
    ).round(2)

    # Define column order for consistent output
    impl_order = ["std_simd", "avx2", "sse4", "neon"]
    available_cols = [c for c in impl_order if c in pivot.columns]

    # Format LaTeX header dynamically
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

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(OUTPUT_TABLES / "main_results.tex", "w") as f:
        f.write("\n".join(tex_lines))
    print(f"✓ Saved: {OUTPUT_TABLES / 'main_results.tex'}")


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def main():
    print("🔍 Loading benchmark results...")
    df_raw = load_all_results()
    print(f"   Loaded {len(df_raw)} benchmark entries from {df_raw['kernel'].nunique() if not df_raw.empty else 0} kernels")
    print(f"   Columns: {list(df_raw.columns)}")

    if not df_raw.empty and "size" in df_raw.columns:
        print(f"   Sizes found: {sorted(df_raw['size'].unique())}")
        print(f"   Implementations: {df_raw['impl'].unique()}")

    print("📈 Computing speedups...")
    df_speedup = compute_speedup(df_raw)

    if df_speedup.empty:
        print("⚠️  Speedup DataFrame is empty. Skipping visualization.")
        return

    print("🎨 Generating figures...")
    plot_speedup_by_pattern(df_speedup)
    plot_api_gap(df_speedup)
    plot_scaling(df_speedup)
    plot_pattern_heatmap(df_speedup)

    print("📄 Generating LaTeX tables...")
    generate_latex_tables(df_speedup)

    print("\n✅ Visualization pipeline complete!")
    print(f"   Figures: {OUTPUT_FIGURES}")
    print(f"   Tables:  {OUTPUT_TABLES}")


if __name__ == "__main__":
    # Set publication style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    })

    main()