import numpy as np
from scipy.stats import ks_2samp
from numpy.linalg import norm
from typing import List

def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix for normalized queue data.
    Input shape: (n_samples, n_features)
    Output shape: (n_features, n_features)
    """
    C = np.corrcoef(data, rowvar=False)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

def run_ks_tests(
    real_q_all: np.ndarray,
    fake_q_all: np.ndarray,
    n_levels: int = 3,
) -> None:
    """
    Run two-sample Kolmogorov–Smirnov tests for the first n_levels
    of Bid and Ask queues.

    Uses the *normalized* queues stored in real_q_all and fake_q_all:
        columns 0..K-1  = bid levels 1..K
        columns K..2K-1 = ask levels 1..K
    Prints D statistic and p-value for each level.
    """
    # Number of levels per side
    total_features = real_q_all.shape[1]
    K = total_features // 2
    n_levels = min(n_levels, K)

    bid_indices = list(range(n_levels))            # 0,1,2 → Bid 1,2,3
    ask_indices = list(range(K, K + n_levels))     # K,K+1,K+2 → Ask 1,2,3
    indices = bid_indices + ask_indices

    level_labels = (
        [f"Bid {i+1}" for i in range(n_levels)] +
        [f"Ask {i+1}" for i in range(n_levels)]
    )

    print("\n======= KS TESTS (Real vs Fake, normalized queue sizes) =======")
    for label, idx in zip(level_labels, indices):
        real_vals = real_q_all[:, idx]
        fake_vals = fake_q_all[:, idx]

        # Drop NaN / inf just in case
        mask_real = np.isfinite(real_vals)
        mask_fake = np.isfinite(fake_vals)
        real_clean = real_vals[mask_real]
        fake_clean = fake_vals[mask_fake]

        if len(real_clean) == 0 or len(fake_clean) == 0:
            print(f"{label:6s}: skipped (no finite data)")
            continue

        D, p = ks_2samp(real_clean, fake_clean)
        print(f"{label:6s}: D = {D:.4f},  p = {p:.3e}")
    print("===============================================================\n")


def mean_var_deviation(
    real_X: np.ndarray,
    fake_X: np.ndarray,
    eps: float = 1e-8
):
    """
    Computes mean and variance relative deviations between real and fake data.

    Parameters
    ----------
    real_X : np.ndarray
        Shape (N, d), real centered LOB states
    fake_X : np.ndarray
        Shape (N, d), fake centered LOB states
    eps : float
        Numerical stability constant

    Returns
    -------
    mean_dev : float
        Average relative deviation of means
    var_dev : float
        Average relative deviation of variances
    """

    # First-order moments
    mu_real = real_X.mean(axis=0)
    mu_fake = fake_X.mean(axis=0)

    # Second-order moments
    var_real = real_X.var(axis=0)
    var_fake = fake_X.var(axis=0)

    # Relative deviations
    mean_dev = np.mean(
        np.abs(mu_fake - mu_real) / (np.abs(mu_real) + eps)
    )

    var_dev = np.mean(
        np.abs(var_fake - var_real) / (var_real + eps)
    )

    return mean_dev, var_dev

def compute_frobenius_correlation(
    real_q_all: np.ndarray,
    fake_q_all: np.ndarray
) -> float:
    """
    Compute Frobenius norm between real and fake correlation matrices.
    No plots. Just prints the number.
    """
    real_corr = compute_correlation_matrix(real_q_all)
    fake_corr = compute_correlation_matrix(fake_q_all) 

    diff = real_corr - fake_corr
    frob = norm(diff, "fro") 
    frob_real = norm(real_corr, "fro")
    frob_centered = frob / frob_real if frob_real > 1e-12 else frob 

    # print("\n======= CORRELATION FROBENIUS METRIC =======")
    # print(f"|| C_real - C_fake ||_F  =  {frob:.6f}") 
    # print(f"||C_real-C_fake||_F / ||C_real||_F = {frob_centered:.6f}")
    # print("============================================\n")

    return frob_centered

def compute_midprice_direction_matrix(
    mid_prices: np.ndarray,
    best_bid_qty: np.ndarray,
    best_ask_qty: np.ndarray,
    n_quantiles: int = 10,
) -> np.ndarray:
    """
    Compute matrix M[k, l] =
        P(Δp_mid > 0 | Δp_mid ≠ 0,
          Q^a_1(t) in [q^a_k, q^a_{k+1}],
          Q^b_1(t) in [q^b_l, q^b_{l+1}] )

    Inputs are 1D arrays over time:
        mid_prices : shape (T,)
        best_bid_qty : shape (T,)
        best_ask_qty : shape (T,)
    """
    mid_prices = np.asarray(mid_prices).ravel()
    best_bid_qty = np.asarray(best_bid_qty).ravel()
    best_ask_qty = np.asarray(best_ask_qty).ravel()

    if len(mid_prices) < 2:
        raise ValueError("Need at least 2 time steps to compute price changes.")

    # Δp_mid(t -> t+1)
    delta_mid = mid_prices[1:] - mid_prices[:-1]

    # Use queues at time t (start of transition)
    bid_t = np.abs(best_bid_qty[:-1])
    ask_t = np.abs(best_ask_qty[:-1])

    # Condition on Δp_mid ≠ 0
    mask_nz = delta_mid != 0.0
    delta_mid = delta_mid[mask_nz]
    bid_t = bid_t[mask_nz]
    ask_t = ask_t[mask_nz]

    if len(delta_mid) == 0:
        raise ValueError("No non-zero price changes found in data.")

    # Quantile edges for bid & ask
    q_edges_bid = np.quantile(bid_t, np.linspace(0.0, 1.0, n_quantiles + 1))
    q_edges_ask = np.quantile(ask_t, np.linspace(0.0, 1.0, n_quantiles + 1))

    # Matrix M[ask_quantile, bid_quantile]
    M = np.full((n_quantiles, n_quantiles), np.nan, dtype=float)

    # Loop over quantile bins
    for k in range(n_quantiles):      # ask bin
        ask_low = q_edges_ask[k]
        ask_high = q_edges_ask[k + 1]

        ask_mask = (ask_t >= ask_low) & (ask_t <= ask_high)

        for l in range(n_quantiles):  # bid bin
            bid_low = q_edges_bid[l]
            bid_high = q_edges_bid[l + 1]

            bid_mask = (bid_t >= bid_low) & (bid_t <= bid_high)

            idx = ask_mask & bid_mask
            n_bin = np.sum(idx)
            if n_bin == 0:
                continue  # leave as NaN if no samples

            # Among these, probability Δp_mid > 0 (we already conditioned on Δp_mid ≠ 0)
            p_up = np.mean(delta_mid[idx] > 0)
            M[k, l] = p_up

    return M

def compute_price_frobenius(m_real, m_fake):
    diff = m_real - m_fake
    frob = norm(diff, "fro") 
    frob_real = norm(m_real, "fro")
    frob_centered = frob / frob_real if frob_real > 1e-12 else frob 

    # print("\n======= CORRELATION FROBENIUS METRIC =======")
    # print(f"|| C_real - C_fake ||_F  =  {frob:.6f}") 
    # print(f"||C_real-C_fake||_F / ||C_real||_F = {frob_centered:.6f}")
    # print("============================================\n")

    return frob_centered