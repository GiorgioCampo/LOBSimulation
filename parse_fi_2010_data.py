#!/usr/bin/env python3
import os
import numpy as np

DATA_DIR = "./BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/"
TICK_THRESHOLD = 50

def remove_price_jumps(arr, tick_threshold=25, verbose=False):
    """
    Remove jumps between days by detecting outliers in price changes (> tick_threshold)
    and offsetting all PRICE columns from that point onwards.
    
    Uses the 5% quantile method to estimate tick size, then detects jumps in ticks.
    
    Args:
        arr: numpy array of shape (T, 40) with columns in order:
             askPx_0, askQty_0, bidPx_0, bidQty_0, askPx_1, askQty_1, ...
        tick_threshold: threshold in ticks to consider a jump as an outlier
        verbose: whether to print debug information
    
    Returns:
        arr: modified array with price jumps removed
    """
    price_indices = np.arange(0, 40, 2)  # [0, 2, 4, 6, ..., 38]
    
    # We'll use the best bid (bidPx_0 at index 2) to estimate tick size
    best_bid_idx = 2
    best_bid = arr[:, best_bid_idx]
    
    # Estimate tick size using 5% quantile method (same as training)
    bid_diff = np.diff(best_bid)
    bid_diffs_nonzero = bid_diff[bid_diff != 0.0]
    abs_diffs = np.abs(bid_diffs_nonzero)
    k = max(1, int(0.05 * len(abs_diffs)))  # number of smallest diffs to consider
    delta = np.median(np.sort(abs_diffs)[:k])
    
    if verbose:
        print(f"  Estimated tick size (delta): {delta:.6f}")
    
    # Convert best bid price differences to ticks
    bid_diff_ticks = bid_diff / delta
    
    # Find outliers where abs(diff in ticks) > tick_threshold
    outlier_mask = np.abs(bid_diff_ticks) > tick_threshold
    outlier_indices = np.where(outlier_mask)[0] + 1  # +1 because diff reduces length by 1
    
    if verbose:
        print(f"  Found {len(outlier_indices)} price jumps > {tick_threshold} ticks")
        if len(outlier_indices) > 0:
            print(f"  Jump locations (timesteps): {outlier_indices[:10]}...")  # Show first 10
            print(f"  Jump magnitudes (ticks): {bid_diff_ticks[outlier_mask][:10]}...")
            print(f"  Jump magnitudes (price): {bid_diff[outlier_mask][:10]}...")
    
    # For each outlier, calculate the offset and apply it to all subsequent prices
    # We need to process from the end to the beginning to avoid cumulative effects
    for idx in reversed(outlier_indices):
        # The jump occurs between idx-1 and idx
        # Calculate the offset as the difference at idx-1
        offset = bid_diff[idx - 1]
        
        if verbose and len(outlier_indices) <= 10:
            print(f"  Applying offset {offset:.4f} (price) = {offset/delta:.2f} ticks from timestep {idx} onwards")
        
        # Apply offset to all price columns from idx onwards
        arr[idx:, price_indices] -= offset
    
    return arr

def process_file(in_path, out_path, remove_jumps=True, tick_threshold=25):
    # Load and transpose: rows = timesteps, columns = features
    arr = np.loadtxt(in_path, dtype=float)
    # arr = arr.T # No more needed, updated file and transposed before.
    
    if arr.shape[1] < 40:
        raise ValueError(f"{in_path}: expected at least 40 columns, got {arr.shape[1]}")
    
    arr = arr[:, :40]  # keep first 40 columns only

    # The data is already in the correct order:
    # askPx_0, askQty_0, bidPx_0, bidQty_0, askPx_1, askQty_1, ...

    # Just negate all ask quantity columns: indices 1,5,9,...,37
    # (Should be opposite since they are already negative?)
    ask_qty_indices = np.arange(1, 40, 4)      # [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]
    arr[:, ask_qty_indices] = -arr[:, ask_qty_indices]

    # Remove price jumps between days
    if remove_jumps:
        print(f"Removing price jumps for {os.path.basename(in_path)}...")
        arr = remove_price_jumps(arr, tick_threshold=tick_threshold, verbose=True)

    # Header matches the actual column order
    col_names = []
    for level in range(10):
        col_names.extend([f"askPx_{level}", f"askQty_{level}", f"bidPx_{level}", f"bidQty_{level}"])

    # Save
    np.savetxt(
        out_path,
        arr,
        delimiter=",",
        header=",".join(col_names),
        comments="",
        fmt="%.10g"
    )

    return arr

def validate(arr, verbose=False, max_reports=5):
    # arr shape: (T, 40)
    bad_ask_bid = []      # ask <= bid (same level)
    bad_ask_order = []    # ask levels not increasing
    bad_bid_order = []    # bid levels not decreasing

    for t in range(arr.shape[0]):
        ask = arr[t,  0:10]
        askq = arr[t, 10:20]
        bid = arr[t, 20:30]
        bidq = arr[t, 30:40]

        # ask should be strictly > bid for each level (or at least >= depending on convention)
        mask_ab = ~(ask > bid)   # True where constraint violated
        if mask_ab.any():
            bad_ask_bid.append((t, np.where(mask_ab)[0].tolist()))

        # ask levels should be non-decreasing (ask0 <= ask1 <= ...)
        if not np.all(np.diff(ask) >= 0):
            bad_ask_order.append(t)

        # bid levels should be non-increasing (bid0 >= bid1 >= ...)
        if not np.all(np.diff(bid) <= 0):
            bad_bid_order.append(t)

    if verbose:
        print("Validation summary:")
        print(" rows checked:", arr.shape[0])
        print(" ask<=bid violations:", len(bad_ask_bid))
        print(" ask-level ordering violations:", len(bad_ask_order))
        print(" bid-level ordering violations:", len(bad_bid_order))
        if bad_ask_bid:
            print("Examples of ask<=bid violations (row, levels):")
            for x in bad_ask_bid[:max_reports]:
                print(" ", x)
    return {
        "rows": arr.shape[0],
        "ask<=bid_count": len(bad_ask_bid),
        "ask_order_violations": len(bad_ask_order),
        "bid_order_violations": len(bad_bid_order),
        "examples": bad_ask_bid[:max_reports],
    }

if __name__ == "__main__":
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and f.endswith(".txt")]
    for fn in files:
        in_path = os.path.join(DATA_DIR, fn)
        out_fn = fn
        if out_fn.endswith(".txt"):
            out_fn = out_fn.replace(".txt", "_processed.csv")
        else:
            out_fn = out_fn + "_processed.csv"
        out_path = os.path.join(DATA_DIR, out_fn)

        arr = process_file(in_path, out_path, remove_jumps=True, tick_threshold=TICK_THRESHOLD)
        res = validate(arr, verbose=True)
        print(f"Processed {fn} -> {out_fn} ; validation: {res}")
