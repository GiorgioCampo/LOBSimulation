#!/usr/bin/env python3
"""
Enhanced LOB-GAN Metrics Script (unconditional only)

Generates:
  1) marginals_all_levels.png      — Unconditional Q distributions per level (Bid 1–3, Ask 1–3)
  2) avg_lob_shape.png             — Average book shape across levels
  3) correlation_matrices.png      — Real and fake correlation matrices (red/blue, values in cells)
  4) midprice_direction_matrix.png — P(Δp_mid>0 | Δp_mid≠ 0, bins of Q^a_1, Q^b_1) for real vs fake
  5) Console-only Frobenius norm of correlation difference (no heatmap)

Queue normalization: Q_tilde = sign(Q) * sqrt(|Q| / C), where C = E[|Q|] per level
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance, ks_2samp
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
    OUTPUT_DIR: str = "./figs"
    
    # Plot settings
    GRID_COLS: int = 2
    DRAW_VLINE_ZERO: bool = True
    PLOT_ALL_LEVELS: bool = True          # All level marginals (unconditional)
    PLOT_LOB_SHAPE: bool = True           # Average LOB shape
    PLOT_CORRELATION: bool = True         # Correlation matrices + Frobenius
    
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
# PLOTTING FUNCTIONS (UNCONDITIONAL)
# =====================
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

def run_ks_tests(
    real_q_all: np.ndarray,
    fake_q_all: np.ndarray,
    n_levels: int = 3,
) -> None:
    """
    Run two-sample Kolmogorov–Smirnov tests for the first n_levels
    of Bid and Ask queues.

    Uses the *normalized* queues stored in real_q_all and fake_q_all:
        columns 0..K-1  = bid levels 1..K
        columns K..2K-1 = ask levels 1..K
    Prints D statistic and p-value for each level.
    """
    # Number of levels per side
    total_features = real_q_all.shape[1]
    K = total_features // 2
    n_levels = min(n_levels, K)

    bid_indices = list(range(n_levels))            # 0,1,2 → Bid 1,2,3
    ask_indices = list(range(K, K + n_levels))     # K,K+1,K+2 → Ask 1,2,3
    indices = bid_indices + ask_indices

    level_labels = (
        [f"Bid {i+1}" for i in range(n_levels)] +
        [f"Ask {i+1}" for i in range(n_levels)]
    )

    print("\n======= KS TESTS (Real vs Fake, normalized queue sizes) =======")
    for label, idx in zip(level_labels, indices):
        real_vals = real_q_all[:, idx]
        fake_vals = fake_q_all[:, idx]

        # Drop NaN / inf just in case
        mask_real = np.isfinite(real_vals)
        mask_fake = np.isfinite(fake_vals)
        real_clean = real_vals[mask_real]
        fake_clean = fake_vals[mask_fake]

        if len(real_clean) == 0 or len(fake_clean) == 0:
            print(f"{label:6s}: skipped (no finite data)")
            continue

        D, p = ks_2samp(real_clean, fake_clean)
        print(f"{label:6s}: D = {D:.4f},  p = {p:.3e}")
    print("===============================================================\n")

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
# MIDPRICE DIRECTION MATRIX (REAL & FAKE)
# =====================
def compute_midprice_direction_matrix(
    mid_prices: np.ndarray,
    best_bid_qty: np.ndarray,
    best_ask_qty: np.ndarray,
    n_quantiles: int = 10,
) -> np.ndarray:
    """
    Compute matrix M[k, l] =
        P(Δp_mid > 0 | Δp_mid ≠ 0,
          Q^a_1(t) in [q^a_k, q^a_{k+1}],
          Q^b_1(t) in [q^b_l, q^b_{l+1}] )

    Inputs are 1D arrays over time:
        mid_prices : shape (T,)
        best_bid_qty : shape (T,)
        best_ask_qty : shape (T,)
    """
    mid_prices = np.asarray(mid_prices).ravel()
    best_bid_qty = np.asarray(best_bid_qty).ravel()
    best_ask_qty = np.asarray(best_ask_qty).ravel()

    if len(mid_prices) < 2:
        raise ValueError("Need at least 2 time steps to compute price changes.")

    # Δp_mid(t -> t+1)
    delta_mid = mid_prices[1:] - mid_prices[:-1]

    # Use queues at time t (start of transition)
    bid_t = np.abs(best_bid_qty[:-1])
    ask_t = np.abs(best_ask_qty[:-1])

    # Condition on Δp_mid ≠ 0
    mask_nz = delta_mid != 0.0
    delta_mid = delta_mid[mask_nz]
    bid_t = bid_t[mask_nz]
    ask_t = ask_t[mask_nz]

    if len(delta_mid) == 0:
        raise ValueError("No non-zero price changes found in data.")

    # Quantile edges for bid & ask
    q_edges_bid = np.quantile(bid_t, np.linspace(0.0, 1.0, n_quantiles + 1))
    q_edges_ask = np.quantile(ask_t, np.linspace(0.0, 1.0, n_quantiles + 1))

    # Matrix M[ask_quantile, bid_quantile]
    M = np.full((n_quantiles, n_quantiles), np.nan, dtype=float)

    # Loop over quantile bins
    for k in range(n_quantiles):      # ask bin
        ask_low = q_edges_ask[k]
        ask_high = q_edges_ask[k + 1]

        ask_mask = (ask_t >= ask_low) & (ask_t <= ask_high)

        for l in range(n_quantiles):  # bid bin
            bid_low = q_edges_bid[l]
            bid_high = q_edges_bid[l + 1]

            bid_mask = (bid_t >= bid_low) & (bid_t <= bid_high)

            idx = ask_mask & bid_mask
            n_bin = np.sum(idx)
            if n_bin == 0:
                continue  # leave as NaN if no samples

            # Among these, probability Δp_mid > 0 (we already conditioned on Δp_mid ≠ 0)
            p_up = np.mean(delta_mid[idx] > 0)
            M[k, l] = p_up

    return M


def plot_midprice_direction_matrices(
    M_real: Optional[np.ndarray],
    output_path: str,
    M_fake: Optional[np.ndarray] = None,
):
    """
    Plot midprice-direction matrices.

    If M_fake is provided, shows REAL (left) and FAKE (right) side by side.
    Otherwise, just plots the single matrix (real or fake).

    M[k,l] = P(Δp_mid>0 | Δp_mid≠ 0, Q^a_1 in ask-bin k, Q^b_1 in bid-bin l).
    """
    if M_real is None and M_fake is None:
        print("No midprice-direction matrices to plot.")
        return

    # Decide what we have
    have_real = M_real is not None
    have_fake = M_fake is not None

    if have_real and have_fake:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        mats = [M_real, M_fake]
        titles = [
            r"Real: $P(\Delta p_{\mathrm{mid}}>0 \mid \Delta p_{\mathrm{mid}}\neq 0)$",
            r"Fake: $P(\Delta p_{\mathrm{mid}}>0 \mid \Delta p_{\mathrm{mid}}\neq 0)$",
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        axes = [ax]
        if have_real:
            mats = [M_real]
            titles = [r"Real: $P(\Delta p_{\mathrm{mid}}>0 \mid \Delta p_{\mathrm{mid}}\neq 0)$"]
        else:
            mats = [M_fake]
            titles = [r"Fake: $P(\Delta p_{\mathrm{mid}}>0 \mid \Delta p_{\mathrm{mid}}\neq 0)$"]

    # Make NaNs show up as grey
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgrey")

    vmin, vmax = 0.0, 1.0
    ims = []

    def _plot_single(ax, M, title):
        M_masked = np.ma.masked_invalid(M)
        n_q_ask, n_q_bid = M.shape

        im = ax.imshow(
            M_masked,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )

        # Put values in cells
        for i in range(n_q_ask):
            for j in range(n_q_bid):
                val = M[i, j]
                if np.isnan(val):
                    continue
                txt = f"{val:.2f}"
                # choose text color based on background
                color = "white" if (val < 0.3 or val > 0.7) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

        ax.set_xlabel("# Bid quantile", fontsize=9)
        ax.set_ylabel("# Ask quantile", fontsize=9)
        ax.set_xticks(np.arange(n_q_bid))
        ax.set_yticks(np.arange(n_q_ask))
        ax.set_title(title, fontsize=10)

        return im

    for ax, M, title in zip(axes, mats, titles):
        ims.append(_plot_single(ax, M, title))

    # Shared colorbar on the right
    mappable = ims[0]
    cbar = fig.colorbar(
        mappable,
        ax=axes,
        location="right",
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(r"$P(\Delta p_{\mathrm{mid}}>0 \mid \Delta p_{\mathrm{mid}}\neq 0)$",
                   fontsize=9)

    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")



# =====================
# MAIN EXECUTION
# =====================
def main():
    """Main execution pipeline"""
    print("=" * 60)
    print("LOB-GAN Metrics & Diagnostics (Unconditional)")
    print("=" * 60)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    files = [
        "out/data/20191001/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191002/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191003/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191004/FLEX_L2_SNAPSHOT.csv",
    ]
    
    # Storage for ALL real data
    all_real_bid_qty = []
    all_real_ask_qty = []
    all_mid_real = []
    all_best_bid_q_real = []
    all_best_ask_q_real = []
    
    # Keep labels from first file (they're all the same)
    labels_all = None

    for real_csv in files:
        real_data = load_lob_csv(real_csv)
        
        # Save labels from first file
        if labels_all is None:
            labels_all = real_data.labels_all

        # collect data for midprice-direction matrix
        all_mid_real.append(real_data.mid)
        
        # Make queues positive (we handle sign visually for bids later)
        real_data.qty_bid = np.abs(real_data.qty_bid)
        real_data.qty_ask = np.abs(real_data.qty_ask)
        
        all_best_bid_q_real.append(real_data.qty_bid[:, 0])
        all_best_ask_q_real.append(real_data.qty_ask[:, 0])
        
        # Save for aggregation
        all_real_bid_qty.append(real_data.qty_bid)
        all_real_ask_qty.append(real_data.qty_ask)

    # Concatenate all real data
    all_real_bid_qty = np.vstack(all_real_bid_qty)
    all_real_ask_qty = np.vstack(all_real_ask_qty)
    all_mid_real = np.concatenate(all_mid_real)
    all_best_bid_q_real = np.concatenate(all_best_bid_q_real)
    all_best_ask_q_real = np.concatenate(all_best_ask_q_real)
    
    # NOW create normalizer from ALL real data
    normalizer = QueueNormalizer(all_real_bid_qty, all_real_ask_qty)
    
    # Normalize all real data
    real_q_bid_norm, real_q_ask_norm = normalizer.normalize(all_real_bid_qty, all_real_ask_qty)
    real_q_all = np.hstack([real_q_bid_norm, real_q_ask_norm])

    M_real = None
    try:
        M_real = compute_midprice_direction_matrix(
            mid_prices=all_mid_real,
            best_bid_qty=all_best_bid_q_real,
            best_ask_qty=all_best_ask_q_real,
            n_quantiles=10,
        )
    except ValueError as e:
        print(f"Skipping REAL midprice-direction matrix: {e}")
    
    fake_data = None
    fake_q_bid_norm = None
    fake_q_ask_norm = None
    fake_q_all = None
    M_fake = None
    
    if Config.FAKE_CSV and os.path.exists(Config.FAKE_CSV):
        print("\nFake data detected - loading for comparison...")
        fake_data = load_lob_csv(Config.FAKE_CSV)

        # Ensure fake queues are positive like real (fixes sign issues in avg LOB shape)
        fake_data.qty_bid = np.abs(fake_data.qty_bid)
        fake_data.qty_ask = np.abs(fake_data.qty_ask)

        fake_q_bid_norm, fake_q_ask_norm = normalizer.normalize(fake_data.qty_bid, fake_data.qty_ask)
        fake_q_all = np.hstack([fake_q_bid_norm, fake_q_ask_norm])

        # Midprice-direction matrix for FAKE
        try:
            M_fake = compute_midprice_direction_matrix(
                mid_prices=fake_data.mid,
                best_bid_qty=np.abs(fake_data.qty_bid[:, 0]),
                best_ask_qty=np.abs(fake_data.qty_ask[:, 0]),
                n_quantiles=10,
            )
        except ValueError as e:
            print(f"Skipping FAKE midprice-direction matrix: {e}")
    else:
        print("\nNo fake data - generating plots for real data only")
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # 1. All level marginals (unconditional)
    if Config.PLOT_ALL_LEVELS:
        plot_all_level_marginals(
            real_q_all=real_q_all,
            labels=labels_all,
            output_path=os.path.join(Config.OUTPUT_DIR, "marginals_all_levels.png"),
            fake_q_all=fake_q_all,
            grid_cols=Config.GRID_COLS,
        )
    
    # 2. Average LOB shape
    if Config.PLOT_LOB_SHAPE:
        plot_average_lob_shape(
            real_q_all=real_q_all,
            labels=labels_all,
            output_path=os.path.join(Config.OUTPUT_DIR, "avg_lob_shape.png"),
            fake_q_all=fake_q_all,
        )
    
    # 3. Correlation matrices + Frobenius number (no difference heatmap)
    if Config.PLOT_CORRELATION and fake_q_all is not None:
        plot_correlation_matrices(
            real_q_all=real_q_all,
            labels=labels_all,
            output_path=os.path.join(Config.OUTPUT_DIR, "correlation_matrices.png"),
            fake_q_all=fake_q_all,
        )
        compute_frobenius_correlation(real_q_all, fake_q_all) 
        # 3b. KS tests for first 3 bid/ask levels
        run_ks_tests(real_q_all, fake_q_all, n_levels=3)

    elif Config.PLOT_CORRELATION:
        print("\nSkipping correlation outputs (no fake data available).")

    # 4. Midprice-direction matrix (real vs fake)
    plot_midprice_direction_matrices(
        M_real=M_real,
        M_fake=M_fake,
        output_path=os.path.join(Config.OUTPUT_DIR, "midprice_direction_matrix.png"),
    )
    
    print("\n" + "=" * 60)
    print("✓ All visualizations generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    main()