#!/usr/bin/env python3
"""
Enhanced LOB-GAN Metrics Script

Generates:
  1) marginals_conditional_Q.png   — Q distributions conditioned on Δp ≥ kφ (if enabled)
  2) marginals_all_levels.png      — Unconditional Q distributions per level
  3) avg_lob_shape.png             — Average book shape across levels
  4) correlation_matrices.png      — Real and fake correlation matrices (red/blue, values in cells)
  5) Console-only Frobenius norm of correlation difference (no heatmap)

Queue normalization: Q_tilde = sign(Q) * sqrt(|Q| / C), where C = E[|Q|] per level
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
from numpy.linalg import norm

from train_model import MARKET_DEPTH

BASE_FOLDER = Path(__file__).resolve().parent
DAY = "20191002"
REAL_CSV = BASE_FOLDER / f"out/data/{DAY}/FLEX_L2_SNAPSHOT.csv"
FAKE_CSV = BASE_FOLDER / f"generated_lob.csv"

# =====================
# CONFIGURATION
# =====================
class Config:
    """Centralized configuration"""
    # Paths
    REAL_CSV: str = REAL_CSV
    FAKE_CSV: Optional[str] = FAKE_CSV  # Set to path when generated data is ready
    OUTPUT_DIR: str = "./out/metrics"
    
    # Plot settings
    GRID_COLS: int = 2
    DRAW_VLINE_ZERO: bool = True
    PLOT_CONDITIONAL_Q: bool = False      # Conditional Q distributions (like paper)
    PLOT_ALL_LEVELS: bool = True          # All level marginals (unconditional)
    PLOT_LOB_SHAPE: bool = True           # Average LOB shape
    PLOT_CORRELATION: bool = True         # Correlation matrices + Frobenius
    
    # Conditional plot settings
    CONDITIONAL_KS: List[int] = [1, 2, 3]  # Tick thresholds: |Δp| ≥ k
    CONDITION_ON_MID: bool = True          # True: condition on Δmid, False: best bid/ask
    
    # Style
    FIGURE_DPI: int = 300
    REAL_COLOR: str = "#4A90E2"
    FAKE_COLOR: str = "#F5A623"
    REAL_ALPHA: float = 0.6
    FAKE_ALPHA: float = 0.6


# =====================
# DATA LOADING
# =====================
class LOBData:
    """Container for LOB snapshot data"""
    def __init__(self, prices_bid: np.ndarray, qty_bid: np.ndarray,
                 prices_ask: np.ndarray, qty_ask: np.ndarray):
        self.prices_bid = prices_bid  # (n, K)
        self.qty_bid = qty_bid
        self.prices_ask = prices_ask
        self.qty_ask = qty_ask
        self.K = prices_bid.shape[1]
        self.n_samples = prices_bid.shape[0]
        
        # Derived quantities
        self.best_bid = prices_bid[:, 0]
        self.best_ask = prices_ask[:, 0]
        self.mid = (self.best_bid + self.best_ask) / 2.0
        self.spread = self.best_ask - self.best_bid
        
        # Level labels
        self.labels_bid = [f"Bid {i}" for i in range(self.K, 0, -1)]
        self.labels_ask = [f"Ask {i}" for i in range(1, self.K + 1)]
        self.labels_all = self.labels_bid + self.labels_ask

        # Truncate to configured market depth
        self.prices_bid = self.prices_bid[:, :MARKET_DEPTH]
        self.qty_bid = self.qty_bid[:, :MARKET_DEPTH]
        self.prices_ask = self.prices_ask[:, :MARKET_DEPTH]
        self.qty_ask = self.qty_ask[:, :MARKET_DEPTH]
        self.labels_ask = self.labels_ask[:MARKET_DEPTH]
        self.labels_bid = self.labels_bid[:MARKET_DEPTH]
        self.labels_all = self.labels_bid + self.labels_ask
        self.K = self.prices_bid.shape[1]


def parse_csv_header(path: str) -> List[str]:
    """Read CSV header"""
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip().split(",")


def extract_level_columns(header: List[str]) -> Tuple[List[str], List[str], List[str], List[str], int]:
    """Extract bid/ask price/quantity column names and infer K"""
    bid_px = sorted([c for c in header if c.startswith("bidPx_")],
                    key=lambda s: int(s.split("_")[1]))
    bid_q  = sorted([c for c in header if c.startswith("bidQty_")],
                    key=lambda s: int(s.split("_")[1]))
    ask_px = sorted([c for c in header if c.startswith("askPx_")],
                    key=lambda s: int(s.split("_")[1]))
    ask_q  = sorted([c for c in header if c.startswith("askQty_")],
                    key=lambda s: int(s.split("_")[1]))
    
    K = min(len(bid_px), len(bid_q), len(ask_px), len(ask_q))
    return bid_px[:K], bid_q[:K], ask_px[:K], ask_q[:K], K


def load_lob_csv(path: str) -> LOBData:
    """Load L2 snapshot CSV into LOBData structure"""
    print(f"Loading: {path}")
    
    header = parse_csv_header(path)
    name_to_idx = {c: i for i, c in enumerate(header)}
    bid_px_cols, bid_q_cols, ask_px_cols, ask_q_cols, K = extract_level_columns(header)
    
    print(f"  Detected K = {K} levels")
    
    usecols = ([name_to_idx[c] for c in bid_px_cols] +
               [name_to_idx[c] for c in bid_q_cols] +
               [name_to_idx[c] for c in ask_px_cols] +
               [name_to_idx[c] for c in ask_q_cols])
    
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols)
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols)
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols)
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols)
    print(f"  Loaded {data.shape[0]} samples")
    
    prices_bid = data[:, 0:K]
    qty_bid    = data[:, K:2*K]
    prices_ask = data[:, 2*K:3*K]
    qty_ask    = data[:, 3*K:4*K]
    
    return LOBData(prices_bid, qty_bid, prices_ask, qty_ask)


# =====================
# NORMALIZATION
# =====================
class QueueNormalizer:
    """Handles queue normalization with constants fitted on real data"""
    def __init__(self, real_qty_bid: np.ndarray, real_qty_ask: np.ndarray):
        eps = 1e-12
        self.C_bid = np.mean(np.abs(real_qty_bid), axis=0) + eps
        self.C_ask = np.mean(np.abs(real_qty_ask), axis=0) + eps
        
    def normalize(self, qty_bid: np.ndarray, qty_ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sqrt normalization: sign(Q) * sqrt(|Q| / C)"""
        q_bid_tilde = np.sign(qty_bid) * np.sqrt(np.abs(qty_bid) / self.C_bid)
        q_ask_tilde = np.sign(qty_ask) * np.sqrt(np.abs(qty_ask) / self.C_ask)
        return q_bid_tilde, q_ask_tilde


# =====================
# ANALYSIS UTILITIES
# =====================
def infer_tick_size(prices: np.ndarray) -> float:
    """Infer tick size from price grid"""
    unique_prices = np.unique(prices[~np.isnan(prices)])
    diffs = np.diff(unique_prices)
    positive_diffs = diffs[diffs > 0]
    return np.min(positive_diffs) if len(positive_diffs) > 0 else 0.01


def compute_kde(data: np.ndarray, x_min: float, x_max: float, n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kernel density estimate"""
    if len(data) < 2:
        return np.array([]), np.array([])
    
    xs = np.linspace(x_min, x_max, n_points)
    try:
        kde = gaussian_kde(data)
        ys = kde(xs)
        integral = np.trapz(ys, xs)
        if integral > 1e-12:
            ys = ys / integral
        return xs, ys
    except Exception:
        return np.array([]), np.array([])


# =====================
# PLOTTING FUNCTIONS
# =====================
def plot_conditional_marginals(
    real_q_bid: np.ndarray,
    real_q_ask: np.ndarray,
    real_data: LOBData,
    tick_size: float,
    output_path: str,
    fake_q_bid: Optional[np.ndarray] = None,
    fake_q_ask: Optional[np.ndarray] = None,
    fake_data: Optional[LOBData] = None,
    ks: List[int] = [1, 2],
    levels: List[int] = [0, 1],
    grid_cols: int = 2,
):
    """
    Plot conditional marginals Q_t conditioned on |Δp(t)| ≥ k*φ
    Similar to the paper figure.
    """
    print("\nGenerating conditional marginal plots (like paper figure)...")
    
    # Compute price changes
    if Config.CONDITION_ON_MID:
        real_price = real_data.mid
    else:
        real_price = real_data.best_bid
    
    real_delta_price = np.diff(real_price)
    real_delta_ticks = np.rint(real_delta_price / tick_size).astype(int)
    
    # Align queues: use Q_t where delta is measured from t-1 to t
    real_q_bid_aligned = real_q_bid[1:, :]
    real_q_ask_aligned = real_q_ask[1:, :]
    
    has_fake = (fake_q_bid is not None and fake_q_ask is not None and fake_data is not None)
    if has_fake:
        if Config.CONDITION_ON_MID:
            fake_price = fake_data.mid
        else:
            fake_price = fake_data.best_bid
        
        fake_delta_price = np.diff(fake_price)
        fake_delta_ticks = np.rint(fake_delta_price / tick_size).astype(int)
        fake_q_bid_aligned = fake_q_bid[1:, :]
        fake_q_ask_aligned = fake_q_ask[1:, :]
    
    panels = []
    for k in ks:
        for level_idx in levels:
            mask_real = np.abs(real_delta_ticks) >= k
            n_real = np.sum(mask_real)
            
            # Ask side
            real_ask_data = real_q_ask_aligned[mask_real, level_idx]
            if has_fake:
                mask_fake = np.abs(fake_delta_ticks) >= k
                fake_ask_data = fake_q_ask_aligned[mask_fake, level_idx]
                n_fake = np.sum(mask_fake)
            else:
                fake_ask_data = None
                n_fake = 0
            
            panels.append(("Ask", level_idx + 1, k, real_ask_data, fake_ask_data, n_real, n_fake))
            
            # Bid side
            real_bid_data = real_q_bid_aligned[mask_real, level_idx]
            if has_fake:
                fake_bid_data = fake_q_bid_aligned[mask_fake, level_idx]
            else:
                fake_bid_data = None
            
            panels.append(("Bid", level_idx + 1, k, real_bid_data, fake_bid_data, n_real, n_fake))
    
    n_panels = len(panels)
    n_rows = int(np.ceil(n_panels / grid_cols))
    fig, axes = plt.subplots(n_rows, grid_cols, figsize=(6 * grid_cols, 3 * n_rows))
    axes = axes.ravel() if n_panels > 1 else [axes]
    
    # Common x-limits
    all_data = [p[3] for p in panels if len(p[3]) > 0]
    if has_fake:
        all_data.extend([p[4] for p in panels if p[4] is not None and len(p[4]) > 0])
    
    if all_data:
        x_min = min(np.nanmin(d) for d in all_data)
        x_max = max(np.nanmax(d) for d in all_data)
        span = x_max - x_min
        x_min -= 0.15 * span
        x_max += 0.15 * span
    else:
        x_min, x_max = -5, 5
    
    for idx, (side, level, k, real_data_panel, fake_data_panel, n_real, n_fake) in enumerate(panels):
        ax = axes[idx]
        
        if len(real_data_panel) < 2:
            ax.text(0.5, 0.5, f"Insufficient samples\n(n={len(real_data_panel)})",
                    ha="center", va="center", fontsize=10)
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"$Q_{{p+{k}\\phi}}(t)$, {side} {level}, W1: N/A", fontsize=12)
            continue
        
        xs_r, ys_r = compute_kde(real_data_panel, x_min, x_max)
        if len(xs_r) > 0:
            ax.fill_between(xs_r, ys_r, alpha=Config.REAL_ALPHA,
                            color=Config.REAL_COLOR, label="Real",
                            linewidth=1.5, edgecolor=Config.REAL_COLOR)
        
        w1_str = ""
        if fake_data_panel is not None and len(fake_data_panel) >= 2:
            xs_f, ys_f = compute_kde(fake_data_panel, x_min, x_max)
            if len(xs_f) > 0:
                ax.fill_between(xs_f, ys_f, alpha=Config.FAKE_ALPHA,
                                color=Config.FAKE_COLOR, label="Fake",
                                linewidth=1.5, edgecolor=Config.FAKE_COLOR)
                w1 = wasserstein_distance(real_data_panel.ravel(), fake_data_panel.ravel())
                w1_str = f", W1: {w1:.3f}"
        
        if Config.DRAW_VLINE_ZERO:
            ax.axvline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.8)
        
        ax.set_title(f"$Q_{{p+{k}\\phi}}(t)$, {side} {level}{w1_str}", fontsize=12, pad=10)
        ax.set_xlabel("Normalized Queue Size", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(x_min, x_max)
        
        legend_label = f"$|\\Delta p_t| \\geq {k}\\phi$"
        ax.legend(title=legend_label, fontsize=9, title_fontsize=9,
                  framealpha=0.95, loc="upper right")
        ax.grid(alpha=0.2, linewidth=0.5)
    
    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_level_marginals(
    real_q_all: np.ndarray,
    labels: List[str],
    output_path: str,
    fake_q_all: Optional[np.ndarray] = None,
    grid_cols: int = 2,
):
    """
    Plot unconditional marginals for Bid 1,2,3 and Ask 1,2,3 only.

    - Uses the first 3 bid columns in real_q_all as Bid 1–3
      (columns 0,1,2 → bidQty_0,1,2).
    - Uses the first 3 ask columns as Ask 1–3
      (columns K,K+1,K+2 → askQty_0,1,2).
    - Bids are shown as NEGATIVE (real and fake),
      asks are positive.
    """
    print("\nGenerating all-level marginal plots (Bid 1–3 and Ask 1–3)...")

    # Number of levels per side, based on data layout
    K = real_q_all.shape[1] // 2
    levels_to_plot = min(3, K)  # in case K < 3

    # Column indices: best 3 bid levels and best 3 ask levels
    bid_indices = list(range(0, levels_to_plot))                # 0,1,2 → Bid 1,2,3
    ask_indices = list(range(K, K + levels_to_plot))            # K,K+1,K+2 → Ask 1,2,3
    selected_indices = bid_indices + ask_indices

    # Custom labels so they are exactly Bid 1–3, Ask 1–3
    selected_labels = (
        [f"Bid {i+1}" for i in range(levels_to_plot)] +
        [f"Ask {i+1}" for i in range(levels_to_plot)]
    )

    # Slice and sign-adjust: bids negative, asks positive
    real_selected = real_q_all[:, selected_indices].copy()
    n_bids = len(bid_indices)
    real_selected[:, :n_bids] *= -1  # make bids negative

    if fake_q_all is not None:
        fake_selected = fake_q_all[:, selected_indices].copy()
        fake_selected[:, :n_bids] *= -1  # make fake bids negative too
    else:
        fake_selected = None

    n_panels = len(selected_indices)  # should be 6 if K >= 3
    n_rows = int(np.ceil(n_panels / grid_cols))
    fig, axes = plt.subplots(n_rows, grid_cols, figsize=(12, 2.5 * n_rows))
    axes = axes.ravel()

    # Global x-limits across all selected levels
    if fake_selected is not None:
        all_data = np.concatenate([real_selected, fake_selected], axis=0)
    else:
        all_data = real_selected

    x_min, x_max = np.nanmin(all_data), np.nanmax(all_data)
    span = x_max - x_min if x_max > x_min else 1.0
    x_min -= 0.1 * span
    x_max += 0.1 * span

    # Plot each marginal
    for panel_idx in range(n_panels):
        ax = axes[panel_idx]

        real_series = real_selected[:, panel_idx]
        xs_r, ys_r = compute_kde(real_series, x_min, x_max)
        if len(xs_r) > 0:
            ax.fill_between(xs_r, ys_r, alpha=Config.REAL_ALPHA,
                            color=Config.REAL_COLOR, label="Real")

        title_suffix = ""
        if fake_selected is not None:
            fake_series = fake_selected[:, panel_idx]
            xs_f, ys_f = compute_kde(fake_series, x_min, x_max)
            if len(xs_f) > 0:
                ax.fill_between(xs_f, ys_f, alpha=Config.FAKE_ALPHA,
                                color=Config.FAKE_COLOR, label="Fake")
                # W1 based on signed series (matching what we plot)
                w1 = wasserstein_distance(real_series.ravel(), fake_series.ravel())
                title_suffix = f" (W1={w1:.3f})"

        if Config.DRAW_VLINE_ZERO:
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(f"{selected_labels[panel_idx]}{title_suffix}", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, linewidth=0.5)

    # Turn off any unused subplots
    for i in range(n_panels, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")





def plot_average_lob_shape(
    real_q_all: np.ndarray,
    labels: List[str],
    output_path: str,
    fake_q_all: Optional[np.ndarray] = None,
):
    """
    Plot average LOB shape with bids shown as negative quantities
    (visual convention: depth extends left for bids, right for asks)
    """
    print("\nGenerating average LOB shape plot...")
    
    K = len(labels) // 2
    
    # Real: bids negative, asks positive
    real_shaped = np.hstack([-real_q_all[:, :K], real_q_all[:, K:]])
    mean_real = np.nanmean(real_shaped, axis=0)
    
    x = np.arange(len(labels))
    width = 0.35 if fake_q_all is None else 0.4
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if fake_q_all is None:
        ax.bar(x, mean_real, width, color=Config.REAL_COLOR,
               alpha=0.7, label="Real", edgecolor="black", linewidth=0.5)
    else:
        # Fake: also shaped with bids negative, asks positive
        fake_shaped = np.hstack([-fake_q_all[:, :K], fake_q_all[:, K:]])
        mean_fake = np.nanmean(fake_shaped, axis=0)
        
        ax.bar(x - width/2, mean_real, width, color=Config.REAL_COLOR,
               alpha=0.7, label="Real", edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, mean_fake, width, color=Config.FAKE_COLOR,
               alpha=0.7, label="Fake", edgecolor="black", linewidth=0.5)
    
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(K - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Mid")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Normalized Queue Size", fontsize=11, fontweight="bold")
    ax.set_title("Average LOB Shape (Bids Negative, Asks Positive)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =====================
# CORRELATION ANALYSIS (matrices + Frobenius only)
# =====================
def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix for normalized queue data.
    Input shape: (n_samples, n_features)
    Output shape: (n_features, n_features)
    """
    C = np.corrcoef(data, rowvar=False)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def plot_correlation_matrices(
    real_q_all: np.ndarray,
    labels: List[str],
    output_path: str,
    fake_q_all: Optional[np.ndarray] = None,
):
    """
    Plot correlation matrices for real and fake data side by side,
    using ONLY 6 features in this order:

        Ask 1, Bid 1, Ask 2, Bid 2, Ask 3, Bid 3

    So the resulting correlation matrices are 6x6.
    Red = positive, blue = negative, values shown in each cell.
    """
    print("\nGenerating correlation matrix plots (Ask1,Bid1,Ask2,Bid2,Ask3,Bid3)...")

    # Number of levels per side, based on the data layout
    total_features = real_q_all.shape[1]
    K = total_features // 2          # bids: 0..K-1, asks: K..2K-1
    levels_to_use = min(3, K)        # in case K < 3

    # Build indices: [Ask1, Bid1, Ask2, Bid2, Ask3, Bid3]
    selected_indices = []
    for l in range(levels_to_use):
        ask_idx = K + l   # Ask (l+1)
        bid_idx = l       # Bid (l+1)
        selected_indices.append(ask_idx)
        selected_indices.append(bid_idx)

    # Slice the data to just those 6 features (or fewer if K<3)
    real_sel = real_q_all[:, selected_indices]
    fake_sel = fake_q_all[:, selected_indices] if fake_q_all is not None else None

    # Compute 6x6 correlation matrices
    real_corr = compute_correlation_matrix(real_sel)
    if fake_sel is not None:
        fake_corr = compute_correlation_matrix(fake_sel)

    # Construct labels in the desired order
    selected_labels = []
    for l in range(levels_to_use):
        selected_labels.append(f"Ask {l+1}")
        selected_labels.append(f"Bid {l+1}")
    n = len(selected_labels)  # should be 6 if levels_to_use=3

    # Make figure
    if fake_sel is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        axes = [axes]

    vmin, vmax = -1, 1

    # ---- REAL MATRIX ----
    ax_real = axes[0]
    im_real = ax_real.imshow(
        real_corr,
        cmap="RdBu",      # red positive, blue negative
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax_real.set_title("Real Correlation (A1,B1,A2,B2,A3,B3)", fontsize=14)
    ax_real.set_xticks(range(n))
    ax_real.set_yticks(range(n))
    ax_real.set_xticklabels(selected_labels, rotation=45, ha="right")
    ax_real.set_yticklabels(selected_labels)

    # numbers in cells
    for i in range(n):
        for j in range(n):
            val = real_corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax_real.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    cbar_real = plt.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04)
    cbar_real.set_label("Correlation", fontsize=10, fontweight="bold")

    # ---- FAKE MATRIX ----
    if fake_sel is not None:
        ax_fake = axes[1]
        im_fake = ax_fake.imshow(
            fake_corr,
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax_fake.set_title("Fake Correlation (A1,B1,A2,B2,A3,B3)", fontsize=14)
        ax_fake.set_xticks(range(n))
        ax_fake.set_yticks(range(n))
        ax_fake.set_xticklabels(selected_labels, rotation=45, ha="right")
        ax_fake.set_yticklabels(selected_labels)

        for i in range(n):
            for j in range(n):
                val = fake_corr[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax_fake.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

        cbar_fake = plt.colorbar(im_fake, ax=ax_fake, fraction=0.046, pad=0.04)
        cbar_fake.set_label("Correlation", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")




def compute_frobenius_correlation(
    real_q_all: np.ndarray,
    fake_q_all: np.ndarray
) -> float:
    """
    Compute Frobenius norm between real and fake correlation matrices.
    No plots. Just prints the number.
    """
    real_corr = compute_correlation_matrix(real_q_all)
    fake_corr = compute_correlation_matrix(fake_q_all) 

    diff = real_corr - fake_corr
    frob = norm(diff, "fro") 
    frob_real = norm(real_corr, "fro")
    frob_centered = frob / frob_real if frob_real > 1e-12 else frob 


    print("\n======= CORRELATION FROBENIUS METRIC =======")
    print(f"|| C_real - C_fake ||_F  =  {frob:.6f}") 
    print(f"||C_real-C_fake||_F / ||C_real||_F = {frob_centered:.6f}")
    print("============================================\n")

    return frob


# =====================
# MAIN EXECUTION
# =====================
def main():
    """Main execution pipeline"""
    print("=" * 60)
    print("LOB-GAN Metrics & Diagnostics")
    print("=" * 60)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    files = [
        "out/data/20191001/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191002/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191003/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191004/FLEX_L2_SNAPSHOT.csv",
    ]
    
    real_q_all = None
    for real_csv in files:
        real_data = load_lob_csv(real_csv)
        
        # Make queues positive (we handle sign visually for bids later)
        real_data.qty_bid = np.abs(real_data.qty_bid)
        real_data.qty_ask = np.abs(real_data.qty_ask)

        normalizer = QueueNormalizer(real_data.qty_bid, real_data.qty_ask)
        real_q_bid_norm, real_q_ask_norm = normalizer.normalize(real_data.qty_bid, real_data.qty_ask)

        real_q = np.hstack([real_q_bid_norm, real_q_ask_norm])
        real_q_all = np.vstack([real_q_all, real_q]) if real_q_all is not None else real_q
    
    all_prices = np.concatenate([real_data.best_bid, real_data.best_ask])
    tick_size = infer_tick_size(all_prices)
    print(f"\nInferred tick size: {tick_size:.4f}")
    
    fake_data = None
    fake_q_bid_norm = None
    fake_q_ask_norm = None
    fake_q_all = None
    
    if Config.FAKE_CSV and os.path.exists(Config.FAKE_CSV):
        print("\nFake data detected - loading for comparison...")
        fake_data = load_lob_csv(Config.FAKE_CSV)

        # Ensure fake queues are positive like real (fixes sign issues in avg LOB shape)
        fake_data.qty_bid = np.abs(fake_data.qty_bid)
        fake_data.qty_ask = np.abs(fake_data.qty_ask)

        fake_q_bid_norm, fake_q_ask_norm = normalizer.normalize(fake_data.qty_bid, fake_data.qty_ask)
        fake_q_all = np.hstack([fake_q_bid_norm, fake_q_ask_norm])
    else:
        print("\nNo fake data - generating plots for real data only")
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # 1. Conditional marginals (if enabled)
    if Config.PLOT_CONDITIONAL_Q and fake_data is not None:
        plot_conditional_marginals(
            real_q_bid=real_q_bid_norm,
            real_q_ask=real_q_ask_norm,
            real_data=real_data,
            tick_size=tick_size,
            output_path=os.path.join(Config.OUTPUT_DIR, "marginals_conditional_Q.png"),
            fake_q_bid=fake_q_bid_norm,
            fake_q_ask=fake_q_ask_norm,
            fake_data=fake_data,
            ks=Config.CONDITIONAL_KS,
            levels=[0, 1],
            grid_cols=Config.GRID_COLS,
        )
    
    # 2. All level marginals
    if Config.PLOT_ALL_LEVELS:
        plot_all_level_marginals(
            real_q_all=real_q_all,
            labels=real_data.labels_all,
            output_path=os.path.join(Config.OUTPUT_DIR, "marginals_all_levels.png"),
            fake_q_all=fake_q_all,
            grid_cols=Config.GRID_COLS,
        )
    
    # 3. Average LOB shape
    if Config.PLOT_LOB_SHAPE:
        plot_average_lob_shape(
            real_q_all=real_q_all,
            labels=real_data.labels_all,
            output_path=os.path.join(Config.OUTPUT_DIR, "avg_lob_shape.png"),
            fake_q_all=fake_q_all,
        )
    
    # 4. Correlation matrices + Frobenius number (no difference heatmap)
    if Config.PLOT_CORRELATION and fake_q_all is not None:
        plot_correlation_matrices(
            real_q_all=real_q_all,
            labels=real_data.labels_all,
            output_path=os.path.join(Config.OUTPUT_DIR, "correlation_matrices.png"),
            fake_q_all=fake_q_all,
        )
        compute_frobenius_correlation(real_q_all, fake_q_all)
    elif Config.PLOT_CORRELATION:
        print("\nSkipping correlation outputs (no fake data available).")
    
    print("\n" + "=" * 60)
    print("✓ All visualizations generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
