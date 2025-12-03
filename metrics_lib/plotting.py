import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from typing import List, Optional

from .config import Config
from .utils import compute_kde
from .data_loader import LOBData
from .metrics import compute_correlation_matrix

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
    Plot conditional marginals.
    FIXED: Handles potential length mismatch between Q data and P data
           by slicing to the minimum common length before calculating diff and alignment.
    """
    print("\nGenerating conditional marginal plots (Directional correction applied)...")
    
    # 1. Calculate Price Changes
    real_price = real_data.mid
    
    # --- FIX START ---
    # Ensure Q and P arrays share the same length before differencing and alignment
    N_p = len(real_price)
    N_q_ask = len(real_q_ask)
    N_q_bid = len(real_q_bid)
    
    # The final aligned arrays must have length N_final - 1
    N_final = min(N_p, N_q_ask, N_q_bid)
    N_aligned = N_final - 1 # This is the length of the mask and the aligned Q arrays

    if N_aligned <= 0:
        print("Error: Input data arrays are too short or empty for differencing.")
        return

    # 1. Slice Price array to N_final and calculate diff (resulting length N_aligned)
    real_delta_price = np.diff(real_price[:N_final])
    real_delta_ticks = np.rint(real_delta_price / tick_size).astype(int)
    
    # 2. Slice Q arrays to N_final and align Q(t) with Delta P(t) (resulting length N_aligned)
    real_q_bid_aligned = real_q_bid[1:N_final, :] 
    real_q_ask_aligned = real_q_ask[1:N_final, :]
    
    # Check alignment integrity
    if len(real_delta_ticks) != len(real_q_ask_aligned):
         print(f"FATAL ALIGNMENT ERROR: Delta Ticks ({len(real_delta_ticks)}) != Q Aligned ({len(real_q_ask_aligned)})")
         return
    # --- FIX END ---
    
    has_fake = (fake_q_bid is not None and fake_q_ask is not None and fake_data is not None)
    if has_fake:
        fake_price = fake_data.mid

        # --- FAKE DATA FIX START ---
        N_p_fake = len(fake_price)
        N_q_ask_fake = len(fake_q_ask)
        N_q_bid_fake = len(fake_q_bid)
        
        N_final_fake = min(N_p_fake, N_q_ask_fake, N_q_bid_fake)
        N_aligned_fake = N_final_fake - 1

        if N_aligned_fake <= 0:
            print("Warning: Fake data arrays are too short or empty.")
            has_fake = False # Disable fake plotting if data is insufficient

        if has_fake:
            fake_delta_price = np.diff(fake_price[:N_final_fake])
            fake_delta_ticks = np.rint(fake_delta_price / tick_size).astype(int)
            
            fake_q_bid_aligned = fake_q_bid[1:N_final_fake, :]
            fake_q_ask_aligned = fake_q_ask[1:N_final_fake, :]
        # --- FAKE DATA FIX END ---
        
    panels = []
    
    for k in ks:
        for level_idx in levels:
            # ==========================================
            # CRITICAL FIX: DIRECTIONAL CONDITIONING (Logic is already correct)
            # ==========================================
            
            # --- ASK SIDE: Condition on Price INCREASE (>= k) ---
            # Mask is guaranteed to be the same length as real_q_ask_aligned now
            mask_real_ask = real_delta_ticks >= k
            # The rest of the logic remains the same...
            
            # ... [Real Ask Data extraction] ...
            n_real_ask = np.sum(mask_real_ask)
            real_ask_data = real_q_ask_aligned[mask_real_ask, level_idx]
            
            if has_fake:
                mask_fake_ask = fake_delta_ticks >= k
                # ... [Fake Ask Data extraction] ...
                fake_ask_data = fake_q_ask_aligned[mask_fake_ask, level_idx]
            else:
                fake_ask_data = None
            
            panels.append({
                "side": "Ask", 
                "level": level_idx + 1, 
                "k": k, 
                "real": real_ask_data, 
                "fake": fake_ask_data,
                "cond_label": f"\\Delta p \\geq {k}\\phi" 
            })
            
            # --- BID SIDE: Condition on Price DECREASE (<= -k) ---
            mask_real_bid = real_delta_ticks <= -k
            # ... [Real Bid Data extraction] ...
            real_bid_data = real_q_bid_aligned[mask_real_bid, level_idx]
            
            if has_fake:
                mask_fake_bid = fake_delta_ticks <= -k
                # ... [Fake Bid Data extraction] ...
                fake_bid_data = fake_q_bid_aligned[mask_fake_bid, level_idx]
            else:
                fake_bid_data = None
            
            panels.append({
                "side": "Bid", 
                "level": level_idx + 1, 
                "k": k, 
                "real": real_bid_data, 
                "fake": fake_bid_data,
                "cond_label": f"\\Delta p \\leq -{k}\\phi" 
            })

    # Plotting Logic (Standardized)
    n_panels = len(panels)
    n_rows = int(np.ceil(n_panels / grid_cols))
    fig, axes = plt.subplots(n_rows, grid_cols, figsize=(6 * grid_cols, 3 * n_rows))
    axes = axes.ravel() if n_panels > 1 else [axes]
    
    # Calculate common X limits to make plots comparable
    all_series = []
    for p in panels:
        if len(p['real']) > 0: all_series.append(p['real'])
        if p['fake'] is not None and len(p['fake']) > 0: all_series.append(p['fake'])
    
    if all_series:
        x_min = min(np.nanmin(d) for d in all_series)
        x_max = max(np.nanmax(d) for d in all_series)
        # Add padding
        span = x_max - x_min
        x_min -= 0.1 * span
        x_max += 0.1 * span
    else:
        x_min, x_max = -5, 5

    for idx, p in enumerate(panels):
        ax = axes[idx]
        real_data_panel = p['real']
        fake_data_panel = p['fake']
        
        # Safety check for empty data (common with large K)
        if len(real_data_panel) < 5:
            ax.text(0.5, 0.5, "Insufficient events", ha="center", va="center")
            ax.set_title(f"{p['side']} {p['level']} | {p['cond_label']}")
            continue

        # Plot Real
        xs_r, ys_r = compute_kde(real_data_panel, x_min, x_max)
        if len(xs_r) > 0:
            ax.fill_between(xs_r, ys_r, alpha=Config.REAL_ALPHA,
                            color=Config.REAL_COLOR, label="Real",
                            linewidth=1.5)

        # Plot Fake & Calc W1
        w1_str = ""
        if fake_data_panel is not None and len(fake_data_panel) > 5:
            xs_f, ys_f = compute_kde(fake_data_panel, x_min, x_max)
            if len(xs_f) > 0:
                ax.fill_between(xs_f, ys_f, alpha=Config.FAKE_ALPHA,
                                color=Config.FAKE_COLOR, label="Fake",
                                linewidth=1.5)
                
                w1 = wasserstein_distance(real_data_panel.ravel(), fake_data_panel.ravel())
                w1_str = f", W1: {w1:.3f}"

        if Config.DRAW_VLINE_ZERO:
            ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)

        # Title Formatting
        # Matches logic: Side Level, Condition, W1 score
        title_text = f"{p['side']} {p['level']}{w1_str}"
        ax.set_title(title_text, fontsize=12)
        
        # Legend with Condition
        ax.legend(title=f"Cond: ${p['cond_label']}$", loc="upper right", fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.grid(alpha=0.2)

    # Clean up empty axes
    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


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
