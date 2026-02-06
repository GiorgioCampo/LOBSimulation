import sys
import os
from pathlib import Path
from typing import List

# Add project root to sys.path to allow importing databento_utils
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from databento_utils import load_dbn_to_numpy
from metrics_lib.config import Config
from metrics_lib.data_loader import LOBData
from metrics_lib.normalization import ZScoreNormalizer, QueueNormalizer
from metrics_lib.plotting import (
    plot_all_level_marginals,
    plot_average_lob_shape,
    plot_correlation_matrices,
    plot_midprice_direction_matrices
)
from metrics_lib.metrics import compute_midprice_direction_matrix

def analyze_lob_files(file_paths: List[str], market_depth: int = Config.MARKET_DEPTH, output_dir: str = "out/data_analysis"):
    """
    Aggregate data from multiple DBN files and perform a comprehensive analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_arrs = []
    all_ts = []
    
    print(f"Loading and aggregating {len(file_paths)} files...")
    for path in file_paths:
        try:
            # Load data with centralized interval
            arr, ts = load_dbn_to_numpy(
                path, 
                interval_ms=Config.INTERVAL_MS, 
                market_depth=market_depth,
                start_minute=Config.START_MINUTE
            )
            if len(arr) > 0:
                all_arrs.append(arr)
                all_ts.append(ts)
            else:
                print(f"  Skipping {Path(path).name}: No data found.")
        except Exception as e:
            print(f"  Error loading {Path(path).name}: {e}")
            
    if not all_arrs:
        print("No valid data found in any of the files.")
        return

    # Aggregate data
    combined_arr = np.concatenate(all_arrs, axis=0)
    combined_ts = np.concatenate(all_ts, axis=0)
    T = len(combined_arr)
    print(f"Aggregation complete. Total samples: {T}")

    # Extract levels
    # Order: [askPx_0, askQty_0, bidPx_0, bidQty_0, ...]
    ask_px = combined_arr[:, 0::4]
    ask_qty = combined_arr[:, 1::4]
    bid_px = combined_arr[:, 2::4]
    bid_qty = combined_arr[:, 3::4] # Bids are already negative from databento_utils
    
    mid_price = (ask_px[:, 0] + bid_px[:, 0]) / 2.0
    spread = ask_px[:, 0] - bid_px[:, 0]
    
    # --- METRICS COMPATIBLE OBJECTS ---
    # --- JUMP DETECTION & REMOVAL ---
    # Heuristic: Jump > 25 ticks (tunable)
    tick_size_abs = np.abs(np.diff(mid_price))
    tick_size_nonzero = tick_size_abs[tick_size_abs > 1e-6]
    tick_size = np.min(tick_size_nonzero) if len(tick_size_nonzero) > 0 else 0.125
    print(f"  Detected Tick Size for Analysis: {tick_size}")
    
    JUMP_THRESHOLD_TICKS = 25
    jump_threshold = JUMP_THRESHOLD_TICKS * tick_size
    
    price_diff = np.diff(mid_price, prepend=mid_price[0])
    jump_indices = np.where(np.abs(price_diff) > jump_threshold)[0]
    
    print(f"\n[JUMP DETECTION] Threshold: {jump_threshold:.4f} ({JUMP_THRESHOLD_TICKS} ticks)")
    
    # "Stitching" - Reconstruct price path ignoring large jumps
    clean_diffs = price_diff.copy()
    
    # Mask for filtering data for metrics (exclude regions around jumps)
    metrics_mask = np.ones(T, dtype=bool)
    WINDOW = 10 # Exclude +/- 100 samples around jump
    
    if len(jump_indices) > 0:
        print(f"Found {len(jump_indices)} jumps - Stitching path & Inspecting Volumes...")
        for idx in jump_indices:
            # Timestamp conversion
            ts_ns = combined_ts[idx]
            ts_dt = pd.to_datetime(ts_ns, unit='ns')
            jump_size = price_diff[idx]
            
            # Inspect Volume at Jump (using basic normalization to check relative magnitude)
            # Just checking raw bid/ask qty at level 0
            vol_bid = np.abs(bid_qty[idx, 0])
            vol_ask = ask_qty[idx, 0]
            print(f"  - Removed Jump at {ts_dt} | Index: {idx} | Jump: {jump_size:.4f} ({jump_size/tick_size:.1f} ticks)")
            print(f"    -> Volume at Jump: Bid={vol_bid:.2f}, Ask={vol_ask:.2f}")
            
            # Remove jump by setting diff to 0 (effectively holding previous price)
            clean_diffs[idx] = 0.0
            
            # Mark window as invalid for metrics
            start_mask = max(0, idx - WINDOW)
            end_mask = min(T, idx + WINDOW)
            metrics_mask[start_mask:end_mask] = False

    # Reconstruct mid_price from cleaned diffs
    stitched_mid_price = np.cumsum(clean_diffs) + mid_price[0]

    # --- PRICE CHANGE STATISTICS ---
    clean_diffs_ticks = clean_diffs / tick_size
    avg_abs_change = np.mean(np.abs(clean_diffs_ticks))
    nonzero_diffs = clean_diffs_ticks[clean_diffs_ticks != 0]
    avg_abs_nonzero_change = np.mean(np.abs(nonzero_diffs)) if len(nonzero_diffs) > 0 else 0.0
    max_change = np.max(clean_diffs_ticks)
    min_change = np.min(clean_diffs_ticks)
    pct_zeros = np.mean(clean_diffs_ticks == 0) * 100

    print(f"\n[PRICE CHANGE STATS (Cleaned)]")
    print(f"  Avg Absolute Change: {avg_abs_change:.4f} ticks")
    print(f"  Avg Abs Change (Non-Zero): {avg_abs_nonzero_change:.4f} ticks")
    print(f"  Max Up Move: {max_change:.2f} ticks")
    print(f"  Max Down Move: {min_change:.2f} ticks")
    print(f"  Zero Changes: {pct_zeros:.2f}%")
    print("-" * 30)

    # --- METRICS COMPATIBLE OBJECTS (CLEANED) ---
    labels_all = ([f"Bid {i+1}" for i in range(market_depth)] + 
                  [f"Ask {i+1}" for i in range(market_depth)])
    
    # Normalization
    normalizer = QueueNormalizer(bid_qty, ask_qty, scale_factor_bid=1, scale_factor_ask=1)
    
    # 1. Normalize ALL data (for time series or inspection)
    norm_bid, norm_ask = normalizer.normalize(bid_qty, ask_qty)
    norm_all = np.hstack([norm_bid, norm_ask])
    
    # 2. Filter data for aggregate metrics (pdfs, shapes, correlations)
    # We use the mask generated during jump detection
    norm_bid_clean = norm_bid[metrics_mask]
    norm_ask_clean = norm_ask[metrics_mask]
    norm_all_clean = np.hstack([norm_bid_clean, norm_ask_clean])
    
    print(f"\n[METRICS FILTERING] Removed {T - np.sum(metrics_mask)} samples around jumps.")
    print(f"Remaining samples for analysis: {len(norm_all_clean)}")

    # 1. Plot Mid-Price (Stitched vs Original)
    plot_step = max(1, T // 10000) # Revert to scalable step
    plt_mid_orig = mid_price[::plot_step]
    plt_mid_stitched = stitched_mid_price[::plot_step]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Original (faded)
    ax1.plot(plt_mid_orig, color='gray', alpha=0.4, label='Original Mid Price', linewidth=0.5)
    
    # Plot Stitched (prominent)
    ax1.plot(plt_mid_stitched, color='blue', alpha=0.9, label='Stitched Mid Price', linewidth=0.8)
    
    ax1.set_xlabel(f'Time Step (sampling interval: {Config.INTERVAL_MS}ms)')
    ax1.set_ylabel('Price', color='blue')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.title(f"Aggregated LOB Dynamics (Original vs Stitched) - {len(jump_indices)} Jumps Adjusted")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_dynamics.png")
    plt.close()
    
    # 1.5. Spread Over Time
    spread_ticks = spread / tick_size
    plt_spread = spread_ticks[::plot_step]
    
    plt.figure(figsize=(12, 4))
    plt.plot(plt_spread, color='red', alpha=0.7, linewidth=0.5, label='Spread')
    plt.title(f"Bid-Ask Spread Over Time")
    plt.xlabel(f'Time Step (sampling interval: {Config.INTERVAL_MS}ms)')
    plt.ylabel("Spread (Ticks)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spread_over_time.png")
    plt.close()
    
    # 2. Marginals All Levels (CLEANED DATA)
    plot_all_level_marginals(
        real_q_all=norm_all_clean,
        labels=labels_all,
        output_path=f"{output_dir}/marginals_all_levels.png",
        fake_q_all=None
    )

    # 3. Average LOB Shape (CLEANED DATA)
    plot_average_lob_shape(
        real_q_all=norm_all_clean,
        labels=labels_all,
        output_path=f"{output_dir}/avg_lob_shape.png",
        fake_q_all=None
    )

    # 4. Correlation Matrices (CLEANED DATA)
    plot_correlation_matrices(
        real_q_all=norm_all_clean,
        labels=labels_all,
        output_path=f"{output_dir}/correlation_matrices.png",
        fake_q_all=None
    )

    # 5. Midprice Direction Matrix
    tick_size_abs = np.abs(np.diff(mid_price))
    tick_size_nonzero = tick_size_abs[tick_size_abs > 1e-6]
    tick_size = np.min(tick_size_nonzero) if len(tick_size_nonzero) > 0 else 0.125
    print(f"  Detected Tick Size for Analysis: {tick_size}")

    try:
        M_real = compute_midprice_direction_matrix(
            mid_prices=mid_price,
            best_bid_qty=np.abs(bid_qty[:, 0]),
            best_ask_qty=np.abs(ask_qty[:, 0]),
            n_quantiles=10
        )
        plot_midprice_direction_matrices(
            M_real=M_real,
            output_path=f"{output_dir}/midprice_direction_matrix.png",
            M_fake=None
        )
    except Exception as e:
        print(f"  Could not compute midprice direction matrix: {e}")

    # 6. Volatility (Log returns)
    log_returns = np.diff(np.log(mid_price))
    plt.figure(figsize=(10, 5))
    plt.hist(log_returns, bins=100, color='purple', alpha=0.7, log=True, density=True)
    plt.title(f"Aggregated Log Returns Distribution ({Config.INTERVAL_MS}ms)")
    plt.xlabel("Log Return")
    plt.ylabel("Density (Log Scale)")
    plt.savefig(f"{output_dir}/aggregated_volatility_hist.png")
    plt.close()

    print(f"Aggregated analysis complete. {T} samples processed.")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    DATA_DIR = Path("D:/Amsterdam/Thesis/order_book_databento/dbn_out")
    dbn_files = sorted(DATA_DIR.glob("*.dbn.zst"))
    
    if not dbn_files:
        print("No DBNS files found.")
    else:
        # Filter out clearly empty files (size based)
        file_paths = [str(f) for f in dbn_files if f.stat().st_size > 1000]
        if file_paths:
            analyze_lob_files(file_paths, market_depth=Config.MARKET_DEPTH)
        else:
            print("No non-empty DBNS files found.")
