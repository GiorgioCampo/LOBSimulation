import numpy as np
import databento as db
from pathlib import Path
from typing import Tuple, List
from metrics_lib.normalization import ZScoreNormalizer, QueueNormalizer

from metrics_lib.config import Config

def load_dbn_to_numpy(
    path: str, 
    interval_ms: int = 100, 
    market_depth: int = Config.MARKET_DEPTH,
    start_hour: int = Config.START_HOUR,
    start_minute: int = Config.START_MINUTE,
    end_hour: int = Config.END_HOUR
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast loading and sampling of .dbn.zst files to NumPy with time filtering.
    
    Args:
        path: Path to .dbn.zst file
        interval_ms: Sampling interval in milliseconds
        market_depth: Number of levels to extract (max 10)
        start_hour: Start hour (inclusive, UTC)
        start_minute: Start minute (inclusive, UTC)
        end_hour: End hour (exclusive, UTC)
        
    Returns:
        arr: NumPy array of shape (T, 4 * market_depth) 
             Order: [askPx_0, askQty_0, bidPx_0, bidQty_0, ...]
        timestamps: NumPy array of shape (T,) nanoseconds since epoch
    """
    store = db.DBNStore.from_file(path)
    
    # We use to_ndarray as it's much faster than iterating records in Python
    # and bypasses most of Pandas' overhead.
    records = store.to_ndarray()
    
    if len(records) == 0:
        return np.array([]), np.array([])

    ts = records['ts_event']
    
    # Time Filtering (NumPy optimized, avoids Pandas overhead)
    # ts is uint64 nanoseconds (UTC)
    # Convert to time of day components
    ts_seconds = ts // 1_000_000_000
    ts_total_minutes = (ts_seconds // 60) % 1440 # Minutes since midnight UTC
    
    # Convert start/end settings to total minutes
    start_time_mins = start_hour * 60 + start_minute
    end_time_mins = end_hour * 60
    
    if start_hour is not None and end_hour is not None: # Check for None on hours is sufficient
        if start_time_mins < end_time_mins:
            time_mask = (ts_total_minutes >= start_time_mins) & (ts_total_minutes < end_time_mins)
        else:
            # Handle overnight case (e.g. 22:30 to 02:00)
            time_mask = (ts_total_minutes >= start_time_mins) | (ts_total_minutes < end_time_mins)
            
        records = records[time_mask]
        ts = records['ts_event']
        
        if len(records) == 0:
            return np.array([]), np.array([])
    
    # Define interval in nanoseconds
    interval_ns = interval_ms * 1_000_000
    
    start_ts = ts[0]
    end_ts = ts[-1]
    
    # Create target timestamps for sampling
    target_ts = np.arange(start_ts, end_ts, interval_ns)
    
    # Find the indices of the last record before or at each target_ts
    # searchsorted 'right' - 1 gives us the index such that ts[idx] <= target_ts
    indices = np.searchsorted(ts, target_ts, side='right') - 1
    
    # Mask out indices < 0 (if any target_ts < first ts, which shouldn't happen here)
    valid_mask = indices >= 0
    indices = indices[valid_mask]
    target_ts = target_ts[valid_mask]
    
    # Sample records
    sampled_records = records[indices]
    
    # Prepare the 40-column array
    # askPx_k, askQty_k, bidPx_k, bidQty_k
    T = len(sampled_records)
    arr = np.zeros((T, 4 * market_depth), dtype=np.float64)
    
    # Databento fixed-precision multiplier is 1e-9
    PRICE_SCALE = 1e-9

    for k in range(market_depth):
        # Databento MBP-10 fields: ask_px_00, ask_sz_00, bid_px_00, bid_sz_00
        # Columns mapped to model expected order:
        # askPx_k (index 4*k), askQty_k (4*k+1), bidPx_k (4*k+2), bidQty_k (4*k+3)
        
        bid_px_field = f'bid_px_{k:02d}'
        ask_px_field = f'ask_px_{k:02d}'
        bid_sz_field = f'bid_sz_{k:02d}'
        ask_sz_field = f'ask_sz_{k:02d}'
        
        arr[:, 4*k]   = sampled_records[ask_px_field].astype(np.float64) * PRICE_SCALE
        arr[:, 4*k+1] = sampled_records[ask_sz_field].astype(np.float32)
        arr[:, 4*k+2] = sampled_records[bid_px_field].astype(np.float64) * PRICE_SCALE
        # Negate bid quantities as expected by the GAN model
        arr[:, 4*k+3] = -sampled_records[bid_sz_field].astype(np.float32)
        
    return arr, target_ts

    return arr, target_ts

def load_dbn_structured(
    path: str, 
    interval_ms: int = 100, 
    start_hour: int = Config.START_HOUR,
    start_minute: int = Config.START_MINUTE,
    end_hour: int = Config.END_HOUR
) -> np.ndarray:
    """
    Loads and samples DBN data but returns the FULL structured array (preserving action, side, etc.)
    """
    store = db.DBNStore.from_file(path)
    records = store.to_ndarray()
    
    if len(records) == 0:
        return np.array([])

    ts = records['ts_event']
    
    # Time Filtering
    ts_seconds = ts // 1_000_000_000
    ts_total_minutes = (ts_seconds // 60) % 1440
    
    start_time_mins = start_hour * 60 + start_minute
    end_time_mins = end_hour * 60
    
    if start_hour is not None and end_hour is not None:
        if start_time_mins < end_time_mins:
            time_mask = (ts_total_minutes >= start_time_mins) & (ts_total_minutes < end_time_mins)
        else:
            time_mask = (ts_total_minutes >= start_time_mins) | (ts_total_minutes < end_time_mins)
            
        records = records[time_mask]
        ts = records['ts_event']
        
        if len(records) == 0:
            return np.array([])
            
    # Sampling (if interval > 0)
    if interval_ms > 0:
        interval_ns = interval_ms * 1_000_000
        start_ts = ts[0]
        end_ts = ts[-1]
        target_ts = np.arange(start_ts, end_ts, interval_ns)
        indices = np.searchsorted(ts, target_ts, side='right') - 1
        valid_mask = indices >= 0
        indices = indices[valid_mask]
        
        sampled_records = records[indices]
    else:
        # If interval 0 or None, return all (filtered) records
        sampled_records = records

    return sampled_records

if __name__ == "__main__":
    # Quick test
    import time
    test_path = "d:/Amsterdam/Thesis/Order Book Data/dbn_out/ES_L2_2025-01-13.dbn.zst"
    
    if Path(test_path).exists():
        start = time.time()
        arr, ts = load_dbn_to_numpy(test_path)
        print(f"Loaded {len(arr)} samples in {time.time() - start:.2f}s")
        print(f"Array shape: {arr.shape}")
        print(f"First sample columns (Level 0):")
        print(f"  AskPx: {arr[0,0]}, AskQty: {arr[0,1]}, BidPx: {arr[0,2]}, BidQty: {arr[0,3]}")
    else:
        print(f"Test file {test_path} not found.")
