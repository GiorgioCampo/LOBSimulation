
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from databento_utils import load_dbn_to_numpy
from metrics_lib.config import Config
from metrics_lib.normalization import QueueNormalizer

def optimize_scale_factors(file_paths, market_depth=10, target_quantile=0.99, target_val=3.0):
    """
    Optimizes scale factors for QueueNormalizer such that the 
    specified quantile of the normalized data maps to target_val.
    """
    print(f"Loading data from {len(file_paths)} files...")
    
    all_bid_qty = []
    all_ask_qty = []
    
    for fp in file_paths:
        try:
            arr, _ = load_dbn_to_numpy(str(fp), market_depth=market_depth)
            if len(arr) == 0: continue
            
            # Extract quantities
            # Format: askPx, askQty, bidPx, bidQty
            # bidQty is at index 3, 7, 11...
            # askQty is at index 1, 5, 9...
            
            bid_indices = [4*k + 3 for k in range(market_depth)]
            ask_indices = [4*k + 1 for k in range(market_depth)]
            
            # Note: load_dbn_to_numpy negates bid quantities! 
            # We need absolute values.
            
            bids = np.abs(arr[:, bid_indices])
            asks = np.abs(arr[:, ask_indices])
            
            all_bid_qty.append(bids)
            all_ask_qty.append(asks)
            
        except Exception as e:
            print(f"Error loading {fp}: {e}")

    if not all_bid_qty:
        print("No data loaded.")
        return

    combined_bids = np.vstack(all_bid_qty)
    combined_asks = np.vstack(all_ask_qty)
    
    print(f"Data shape: {combined_bids.shape}")
    
    # Per-Level Optimization
    abs_bids = np.abs(combined_bids)
    abs_asks = np.abs(combined_asks)
    
    mean_abs_bids = np.mean(abs_bids, axis=0) # (market_depth,)
    mean_abs_asks = np.mean(abs_asks, axis=0)
    
    # Calculate 99th percentile of raw quantities (abs)
    q99_bids = np.percentile(abs_bids, target_quantile * 100, axis=0)
    q99_asks = np.percentile(abs_asks, target_quantile * 100, axis=0)
    
    # scales = sqrt(q99) / (target_val * mean)
    scales_bid = np.sqrt(q99_bids) / (target_val * mean_abs_bids)
    scales_ask = np.sqrt(q99_asks) / (target_val * mean_abs_asks)
    
    print("\n--- Optimized Scales Per Level ---")
    print(f"Bid Levels 0-{market_depth-1}:")
    print(scales_bid)
    print(f"Ask Levels 0-{market_depth-1}:")
    print(scales_ask)
    
    # Validation
    print("\n--- Validation with Per-Level Scales ---")
    normalizer = QueueNormalizer(combined_bids, combined_asks, 
                                 scale_factor_bid=scales_bid, 
                                 scale_factor_ask=scales_ask)
    
    # Note: normalize expects signed quantities if inputs are signed.
    # But combined_bids is already abs(). Bids usually negative.
    # Let's restore sign for validation to be rigorous, or just pass abs if normalize handles it.
    # normalize: sign(Q) * sqrt(|Q|) ...
    # If we pass abs, sign is +1. That's fine for magnitude check.
    
    norm_bids, norm_asks = normalizer.normalize(combined_bids, combined_asks)
    
    # Check 99th percentile of normalized data (absolute)
    actual_q99_bid = np.percentile(np.abs(norm_bids), target_quantile * 100, axis=0)
    actual_q99_ask = np.percentile(np.abs(norm_asks), target_quantile * 100, axis=0)
    
    print(f"Resulting Per-Level 99th Percentiles (Target: {target_val}):")
    print(f"Bid Levels: {actual_q99_bid}")
    print(f"Ask Levels: {actual_q99_ask}")
    
    print("\nCOPY THESE ARRAYS FOR USE in your code:")
    print("-" * 40)
    # Use array2string for clean output
    print(f"SCALE_FACTOR_BID = np.array({np.array2string(scales_bid, separator=', ')})")
    print(f"SCALE_FACTOR_ASK = np.array({np.array2string(scales_ask, separator=', ')})")
    print("-" * 40)

    # Plot per level distributions singularly
    for i in range(market_depth):
        plt.figure(figsize=(10, 5))
        plt.hist(np.abs(norm_bids[:, i]), bins=100, alpha=0.5, label='Bids', density=True, range=(0, 5))
        plt.hist(np.abs(norm_asks[:, i]), bins=100, alpha=0.5, label='Asks', density=True, range=(0, 5))
        plt.axvline(target_val, color='r', linestyle='--', label=f'Target 99% ({target_val})')
        plt.title(f"Distribution of Normalized Quantities (Level {i})")
        plt.legend()
        plt.savefig(f"normalization_dist_level_{i}.png")
        print(f"Saved distribution plot to normalization_dist_level_{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--files", type=int, default=15, help="Number of files to use")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dbn.zst"))[:args.files]
    
    if not files:
        print(f"No files found in {data_dir}")
    else:
        optimize_scale_factors(files)
