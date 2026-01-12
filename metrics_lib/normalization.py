import numpy as np
from typing import Tuple

class QueueNormalizer:
    """Handles queue normalization with constants fitted on real data"""
    def __init__(self, real_qty_bid: np.ndarray, real_qty_ask: np.ndarray):
        eps = 1e-12
        self.C_bid = np.std(np.abs(real_qty_bid), axis=0) * 10 + eps
        self.C_ask = np.std(np.abs(real_qty_ask), axis=0) * 5 + eps

        # self.C_bid = 0.1
        # self.C_ask = 0.1
        
    def normalize(self, qty_bid: np.ndarray, qty_ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sqrt normalization: sign(Q) * sqrt(|Q| / C)"""
        q_bid_tilde = np.sign(qty_bid) * np.sqrt(np.abs(qty_bid)) / self.C_bid
        q_ask_tilde = np.sign(qty_ask) * np.sqrt(np.abs(qty_ask)) / self.C_ask
        return q_bid_tilde, q_ask_tilde

    def denormalize(self, q_bid_tilde: np.ndarray, q_ask_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Invert sqrt normalization: sign(q_tilde) * (q_tilde^2 * C)"""
        qty_bid = np.sign(q_bid_tilde) * (np.square(q_bid_tilde * self.C_bid))
        qty_ask = np.sign(q_ask_tilde) * (np.square(q_ask_tilde * self.C_ask))
        return qty_bid, qty_ask
