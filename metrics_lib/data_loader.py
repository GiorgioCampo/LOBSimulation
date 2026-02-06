import numpy as np
from typing import List, Tuple
from metrics_lib.config import Config

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

        # Truncate to configured market depth (or available depth)
        target_k = min(Config.MARKET_DEPTH, self.K)
        self.prices_bid = self.prices_bid[:, :target_k]
        self.qty_bid = self.qty_bid[:, :target_k]
        self.prices_ask = self.prices_ask[:, :target_k]
        self.qty_ask = self.qty_ask[:, :target_k]
        self.labels_ask = self.labels_ask[:target_k]
        self.labels_bid = self.labels_bid[:target_k]
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
    
    # FIXED: Removed repeated np.loadtxt calls
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols)
    print(f"  Loaded {data.shape[0]} samples")
    
    prices_bid = data[:, 0:K]
    qty_bid    = data[:, K:2*K]
    prices_ask = data[:, 2*K:3*K]
    qty_ask    = data[:, 3*K:4*K]
    
    return LOBData(prices_bid, qty_bid, prices_ask, qty_ask)


def load_lob_dbn(path: str, interval_ms: int = 100, market_depth: int = Config.MARKET_DEPTH) -> LOBData:
    """Load L2 snapshot from raw Databento .dbn.zst file into LOBData structure"""
    print(f"Loading raw DBN: {path}")
    
    # Local import to avoid circular dependency or path issues
    from databento_utils import load_dbn_to_numpy
    
    arr, ts = load_dbn_to_numpy(path, interval_ms=interval_ms, market_depth=market_depth)
    
    if len(arr) == 0:
        raise ValueError(f"No samples loaded from {path}")
        
    print(f"  Loaded {len(arr)} samples")
    
    # Extract levels from the interleaved array
    # Order: [askPx_0, askQty_0, bidPx_0, bidQty_0, ...]
    ask_px = arr[:, 0::4]
    ask_qty = arr[:, 1::4]
    bid_px = arr[:, 2::4]
    # bid quantities are already negative in load_dbn_to_numpy
    bid_qty = arr[:, 3::4] 
    
    return LOBData(bid_px, bid_qty, ask_px, ask_qty)
