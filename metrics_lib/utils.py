import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple

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
