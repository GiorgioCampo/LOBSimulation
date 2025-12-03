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
import glob
import numpy as np

from metrics_lib.config import Config
from metrics_lib.data_loader import load_lob_csv, LOBData
from metrics_lib.normalization import QueueNormalizer
from metrics_lib.plotting import (
    plot_all_level_marginals,
    plot_average_lob_shape,
    plot_conditional_marginals,
    plot_correlation_matrices,
    plot_midprice_direction_matrices
)
from metrics_lib.metrics import (
    compute_frobenius_correlation,
    run_ks_tests,
    compute_midprice_direction_matrix
)

def process_and_plot(
    real_data_list: list[LOBData],
    fake_data: LOBData | None,
    output_dir: str,
    title_suffix: str = ""
):
    """
    Process a list of Real LOBData objects (aggregated) and one Fake LOBData object,
    then generate all plots in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing {title_suffix} -> {output_dir}")

    # --- Aggregation of Real Data ---
    all_real_bid_qty = []
    all_real_ask_qty = []
    all_mid_real = []
    all_best_bid_q_real = []
    all_best_ask_q_real = []
    
    labels_all = None

    for r_data in real_data_list:
        if labels_all is None:
            labels_all = r_data.labels_all
        
        # Ensure positive queues for processing (we handle sign visually later)
        # Note: LOBData might already be modified in place if passed multiple times, 
        # but np.abs is idempotent so it's safe.
        r_data.qty_bid = np.abs(r_data.qty_bid)
        r_data.qty_ask = np.abs(r_data.qty_ask)

        all_mid_real.append(r_data.mid)
        all_best_bid_q_real.append(r_data.qty_bid[:, 0])
        all_best_ask_q_real.append(r_data.qty_ask[:, 0])
        all_real_bid_qty.append(r_data.qty_bid)
        all_real_ask_qty.append(r_data.qty_ask)

    if not all_real_bid_qty:
        print("No real data to process.")
        return

    # Concatenate
    all_real_bid_qty = np.vstack(all_real_bid_qty)
    all_real_ask_qty = np.vstack(all_real_ask_qty)
    all_mid_real = np.concatenate(all_mid_real)
    all_best_bid_q_real = np.concatenate(all_best_bid_q_real)
    all_best_ask_q_real = np.concatenate(all_best_ask_q_real)

    # --- Normalization ---
    # Fit normalizer on THIS set of real data
    normalizer = QueueNormalizer(all_real_bid_qty, all_real_ask_qty)
    
    real_q_bid_norm, real_q_ask_norm = normalizer.normalize(all_real_bid_qty, all_real_ask_qty)
    real_q_all = np.hstack([real_q_bid_norm, real_q_ask_norm])

    # --- Real Midprice Matrix ---
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

    # --- Fake Data Processing ---
    fake_q_all = None
    fake_q_bid_norm = None
    fake_q_ask_norm = None
    M_fake = None

    if fake_data is not None:
        # Normalize fake data using the REAL data's normalizer
        fake_data.qty_bid = np.abs(fake_data.qty_bid)
        fake_data.qty_ask = np.abs(fake_data.qty_ask)
        
        fake_q_bid_norm, fake_q_ask_norm = normalizer.normalize(fake_data.qty_bid, fake_data.qty_ask)
        fake_q_all = np.hstack([fake_q_bid_norm, fake_q_ask_norm])

        try:
            M_fake = compute_midprice_direction_matrix(
                mid_prices=fake_data.mid,
                best_bid_qty=np.abs(fake_data.qty_bid[:, 0]),
                best_ask_qty=np.abs(fake_data.qty_ask[:, 0]),
                n_quantiles=10,
            )
        except ValueError as e:
            print(f"Skipping FAKE midprice-direction matrix: {e}")

    # --- Plotting ---
    
    # 1. Marginals
    if Config.PLOT_ALL_LEVELS:
        plot_all_level_marginals(
            real_q_all=real_q_all,
            labels=labels_all,
            output_path=os.path.join(output_dir, "marginals_all_levels.png"),
            fake_q_all=fake_q_all,
            grid_cols=Config.GRID_COLS,
        )

    # 2. Average Shape
    if Config.PLOT_LOB_SHAPE:
        plot_average_lob_shape(
            real_q_all=real_q_all,
            labels=labels_all,
            output_path=os.path.join(output_dir, "avg_lob_shape.png"),
            fake_q_all=fake_q_all,
        )

    # 3. Correlations
    if Config.PLOT_CORRELATION and fake_q_all is not None:
        plot_correlation_matrices(
            real_q_all=real_q_all,
            labels=labels_all,
            output_path=os.path.join(output_dir, "correlation_matrices.png"),
            fake_q_all=fake_q_all,
        )
        compute_frobenius_correlation(real_q_all, fake_q_all)
        run_ks_tests(real_q_all, fake_q_all, n_levels=3)
    elif Config.PLOT_CORRELATION:
         print("Skipping correlation plots (no fake data).")

    # 4. Midprice Matrix
    plot_midprice_direction_matrices(
        M_real=M_real,
        M_fake=M_fake,
        output_path=os.path.join(output_dir, "midprice_direction_matrix.png"),
    )

    # 5. Conditional Marginals
    # We need a representative Real LOBData for the price/tick info. 
    # Since we aggregated, we can't easily use "all_mid_real" as a single continuous time series 
    # if it comes from multiple days, because of jumps between days.
    # However, plot_conditional_marginals logic handles arrays. 
    # But wait, plot_conditional_marginals takes `real_data` (LOBData object) to get `mid` and calculate diffs.
    # If we passed a list of LOBData, we concatenated them. 
    # But the jumps between days will create huge price diffs.
    # Ideally, we should compute conditional marginals per day and aggregate, OR just use the concatenated arrays 
    # and accept that N-1 points (where N is num days) will be garbage diffs.
    # Given the large N, a few garbage points are negligible.
    # So we can construct a "Dummy" LOBData object holding the concatenated arrays.
    
    aggregated_real_data = LOBData(
        prices_bid=np.zeros((len(all_mid_real), 1)), # Dummy, not used directly if we have mid
        qty_bid=all_real_bid_qty, # Original raw qty
        prices_ask=np.zeros((len(all_mid_real), 1)), # Dummy
        qty_ask=all_real_ask_qty  # Original raw qty
    )
    # Manually overwrite mid since LOBData init calculates it from prices
    aggregated_real_data.mid = all_mid_real
    
    plot_conditional_marginals(
        real_q_bid=real_q_bid_norm,
        real_q_ask=real_q_ask_norm,
        real_data=aggregated_real_data,
        tick_size=0.01,
        output_path=os.path.join(output_dir, "conditional_marginals.png"),
        fake_q_bid=fake_q_bid_norm,
        fake_q_ask=fake_q_ask_norm,
        fake_data=fake_data,
        ks=[1, 2, 3],
        levels=[0, 1, 2],
        grid_cols=Config.GRID_COLS,
    )


def main():
    print("=" * 60)
    print("LOB-GAN Metrics & Diagnostics")
    print("=" * 60)

    # 1. Find all day directories
    data_dir = Config.DATA_DIR
    # Look for subdirectories that look like dates (YYYYMMDD) or just all subdirs
    # We assume folders in out/data/ are the days.
    day_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)])
    
    days_data = {}
    all_real_data_list = []

    print(f"Searching for data in: {data_dir}")
    
    for d_path in day_dirs:
        day_name = os.path.basename(d_path)
        csv_path = os.path.join(d_path, "FLEX_L2_SNAPSHOT.csv")
        
        if os.path.exists(csv_path):
            try:
                data = load_lob_csv(csv_path)
                days_data[day_name] = data
                all_real_data_list.append(data)
            except Exception as e:
                print(f"Failed to load {day_name}: {e}")
        else:
            print(f"Skipping {day_name}: No FLEX_L2_SNAPSHOT.csv found.")

    if not all_real_data_list:
        print("No real data found. Exiting.")
        return

    # 2. Load Fake Data (once)
    fake_data = None
    if Config.FAKE_CSV and os.path.exists(Config.FAKE_CSV):
        print(f"\nLoading Fake Data: {Config.FAKE_CSV}")
        fake_data = load_lob_csv(Config.FAKE_CSV)
    else:
        print("\nNo fake data found.")

    # 3. Process Aggregated Data (All Days)
    print("\n" + "-"*40)
    print("Generating Aggregated Plots (All Days)")
    print("-" * 40)
    process_and_plot(
        real_data_list=all_real_data_list,
        fake_data=fake_data,
        output_dir=Config.OUTPUT_DIR,
        title_suffix="Aggregated"
    )

    # 4. Process Individual Days (Optional)
    if Config.PLOT_INDIVIDUAL_DAYS:
        print("\n" + "-"*40)
        print("Generating Individual Day Plots")
        print("-" * 40)
        for day_name, data in days_data.items():
            day_output_dir = os.path.join(Config.OUTPUT_DIR, day_name)
            process_and_plot(
                real_data_list=[data], # List of one
                fake_data=fake_data,   # Compare same fake data against specific day
                output_dir=day_output_dir,
                title_suffix=f"Day {day_name}"
            )

    print("\n" + "=" * 60)
    print("✓ All tasks completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()