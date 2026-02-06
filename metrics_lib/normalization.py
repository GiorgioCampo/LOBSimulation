import numpy as np
from typing import Tuple

class ZScoreNormalizer:
    """
    Handles Z-score normalization for LOB volumes.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, real_qty_bid: np.ndarray, real_qty_ask: np.ndarray):
        """Fit std on real bid/ask quantities, using 0 as mean to preserve signs."""
        data = np.concatenate([real_qty_bid, real_qty_ask], axis=1)
        self.mean = np.zeros(data.shape[1])
        # Calculate RMS-like scale to capture dispersion around 0
        self.std = np.sqrt(np.mean(data**2, axis=0))
        self.std[self.std == 0] = 1.0

    def normalize(self, qty_bid: np.ndarray, qty_ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None:
            raise ValueError("Normalizer not fitted.")
        K = qty_bid.shape[1]
        # Total fitted levels per side
        total_K = len(self.std) // 2
        
        # Use only the first K levels for both bid and ask
        std_bid = self.std[:K]
        std_ask = self.std[total_K : total_K + K]
        
        norm_bid = qty_bid / std_bid
        norm_ask = qty_ask / std_ask
        return norm_bid, norm_ask

    def denormalize(self, norm_bid: np.ndarray, norm_ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None:
            raise ValueError("Normalizer not fitted.")
        K = norm_bid.shape[1]
        # Total fitted levels per side
        total_K = len(self.std) // 2
        
        # Use only the first K levels for both bid and ask
        std_bid = self.std[:K]
        std_ask = self.std[total_K : total_K + K]
        
        raw_bid = norm_bid * std_bid
        raw_ask = norm_ask * std_ask
        return raw_bid, raw_ask

class QueueNormalizer:
    """Handles queue normalization with constants fitted on real data"""
    def __init__(self, real_qty_bid: np.ndarray, real_qty_ask: np.ndarray, 
                scale_factor_bid: float | np.ndarray = 1, scale_factor_ask: float | np.ndarray = 1):
        eps = 1e-12
        # If scale factors are arrays, they should have shape (market_depth,)
        # Handle array scale factors that might be larger than current depth
        depth = real_qty_bid.shape[1]
        
        if isinstance(scale_factor_bid, np.ndarray) and scale_factor_bid.shape[0] > depth:
            scale_factor_bid = scale_factor_bid[:depth]
        if isinstance(scale_factor_ask, np.ndarray) and scale_factor_ask.shape[0] > depth:
            scale_factor_ask = scale_factor_ask[:depth]

        self.C_bid = np.mean(np.abs(real_qty_bid), axis=0) * scale_factor_bid + eps
        self.C_ask = np.mean(np.abs(real_qty_ask), axis=0) * scale_factor_ask + eps
        
    def normalize(self, qty_bid: np.ndarray, qty_ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sqrt normalization: sign(Q) * sqrt(|Q| / C)"""
        # Ensure C matches the input depth
        depth = qty_bid.shape[-1]
        C_bid = self.C_bid[:depth]
        C_ask = self.C_ask[:depth]

        q_bid_tilde = np.sign(qty_bid) * np.sqrt(np.abs(qty_bid)) / C_bid
        q_ask_tilde = np.sign(qty_ask) * np.sqrt(np.abs(qty_ask)) / C_ask
        return q_bid_tilde, q_ask_tilde

    def denormalize(self, q_bid_tilde: np.ndarray, q_ask_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Invert sqrt normalization: sign(q_tilde) * (q_tilde^2 * C)"""
        # Ensure C matches the input depth
        depth = q_bid_tilde.shape[-1]
        C_bid = self.C_bid[:depth]
        C_ask = self.C_ask[:depth]
        
        qty_bid = np.sign(q_bid_tilde) * (np.square(q_bid_tilde * C_bid))
        qty_ask = np.sign(q_ask_tilde) * (np.square(q_ask_tilde * C_ask))
        return qty_bid, qty_ask
