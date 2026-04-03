from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.family"] = "DejaVu Serif"


def save_figure(fig, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(history: pd.DataFrame, metric_col: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss", color="#005f73", linewidth=2)
    ax.plot(history["epoch"], history[metric_col], label=ylabel, color="#bb3e03", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend(frameon=False)
    save_figure(fig, output_path)


def plot_parity(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    # 关闭网格
    ax.grid(False)

    # 散点（更小 + 紫色调）
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        s=20,                 # 点更小（原来是38）
        alpha=0.9,
        color="#936AB1",      # 紫色
        edgecolor="none",
        ax=ax
    )

    # 对角线（紫色调）
    lower = min(df[x].min(), df[y].min())
    upper = max(df[x].max(), df[y].max())
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="#68259e",      # 浅一点的紫色
        linewidth=1.8
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    save_figure(fig, output_path)


def plot_dataset_overview(split_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    sns.scatterplot(
        data=split_df,
        x="n_atoms",
        y="mmgbsa_mean",
        hue="split",
        palette={"train": "#0a9396", "val": "#ee9b00", "test": "#ca6702"},
        s=60,
        ax=axes[0],
    )
    axes[0].set_xlabel("Atoms per complex")
    axes[0].set_ylabel("Mean MM/GBSA (kcal/mol)")
    axes[0].set_title("Demo30 sample coverage")

    sns.histplot(
        data=split_df,
        x="mmgbsa_mean",
        hue="split",
        bins=12,
        multiple="stack",
        palette={"train": "#0a9396", "val": "#ee9b00", "test": "#ca6702"},
        ax=axes[1],
    )
    axes[1].set_xlabel("Mean MM/GBSA (kcal/mol)")
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("Energetic label distribution")
    save_figure(fig, output_path)


def plot_mmgbsa_error(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["abs_error"] = (plot_df["pred_mmgbsa"] - plot_df["true_mmgbsa"]).abs()
    plot_df = plot_df.sort_values("abs_error", ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=plot_df, x="sample_id", y="abs_error", color="#94d2bd", edgecolor="black", ax=ax)
    ax.set_xlabel("Complex")
    ax.set_ylabel(r"|Prediction - target| (kcal/mol)")
    ax.set_title("Per-complex MM/GBSA absolute error on frame 0")
    ax.tick_params(axis="x", labelrotation=90)
    save_figure(fig, output_path)


def plot_mmgbsa_regime_comparison(df: pd.DataFrame, output_path: Path, label_col: str = "setting") -> None:
    melted = df.melt(id_vars=[label_col], value_vars=["mae", "rmse", "pearson_r", "spearman_r"], var_name="metric")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    sns.barplot(data=melted, x="metric", y="value", hue=label_col, palette="Set2", ax=ax)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("MM/GBSA training regime comparison")
    save_figure(fig, output_path)


def plot_structure_correlation_histogram(
    df: pd.DataFrame,
    method: str,
    output_path: Path,
    bins: int = 35,
) -> None:
    corr_values = []

    for _, group in df.groupby("structure_id"):
        # 至少要有两个点才能算相关性
        if len(group) < 2:
            continue

        corr = group["true_rmsf"].corr(group["pred_rmsf"], method=method)
        if pd.notna(corr):
            corr_values.append(corr)

    corr_values = np.array(corr_values, dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.grid(False)

    ax.hist(
        corr_values,
        bins=bins,
        density=True,          # 概率密度；如果你想严格“每个bin概率”，见后面备注
        color="#7b7fc6",
        edgecolor="#3d3a66",
        linewidth=1.1,
    )

    mean_corr = float(np.mean(corr_values)) if len(corr_values) > 0 else float("nan")

    method_name = "Pearson" if method.lower() == "pearson" else "Spearman"
    ax.set_xlabel(f"{method_name} correlation")
    ax.set_ylabel("Probability")
    ax.set_title(f"Structure correlations mean {mean_corr:.2f}")

    save_figure(fig, output_path)