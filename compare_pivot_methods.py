import torch
import numpy as np

def pivot_counting(X, center_idx):
    """Previous methodology: Counting negative values (bids)"""
    return (X < 0).sum(dim=1) - center_idx

def pivot_transition_first(X, center_idx):
    """Current methodology in gan_model.py: First transition from bid to ask"""
    is_bid = X < 0
    is_ask = X > 0
    transition = is_bid[:, :-1] & is_ask[:, 1:]
    pivot = transition.float().argmax(dim=1)
    has_transition = transition.any(dim=1)
    # The transition index 'pivot' corresponds to the last bid.
    # If pivot = center_idx - 1, the shift is 0.
    shift = torch.where(has_transition, pivot - (center_idx - 1), torch.zeros_like(pivot))
    return shift

def pivot_transition_closest(X, center_idx):
    """Logic from utils.py: Transition closest to center"""
    B, T = X.shape
    is_bid = X < 0
    is_ask = X > 0
    transition = is_bid[:, :-1] & is_ask[:, 1:]
    
    device = X.device
    center = center_idx - 1 # transition index if shift=0
    idxs = torch.arange(T - 1, device=device).unsqueeze(0)
    
    dist = torch.abs(idxs - center)
    dist = torch.where(transition, dist, torch.full_like(dist, T))
    
    pivot = dist.argmin(dim=1)
    has_transition = transition.any(dim=1)
    shift = torch.where(has_transition, pivot - center, torch.zeros_like(pivot))
    return shift

def run_comparison():
    MARKET_DEPTH = 4
    MAX_PRICE_CHANGE = 4
    # T = 2 * (4 + 4) = 16
    T = 16
    CENTER = 8 # market_depth + max_price_change
    
    print(f"Parameters: MARKET_DEPTH={MARKET_DEPTH}, MAX_PRICE_CHANGE={MAX_PRICE_CHANGE}")
    print(f"Parameters: T={T}, CENTER={CENTER}\n")
    
    # Base pattern (balanced, 4 bids, 4 asks)
    base_pattern = [-8, -5, -4, -2, 2, 3, 4, 7]
    
    # Test cases: (name, true_shift)
    test_shifts = [
        ("Base Case", 0),
        ("Price Up (+1 tick)", 1),
        ("Price Up (+3 ticks)", 3),
        ("Price Down (-1 tick)", -1),
        ("Price Down (-2 ticks)", -2),
    ]
    
    for name, true_shift in test_shifts:
        X = torch.zeros(1, T)
        # In _create_imbalanced_states:
        # idx_q1 = center - 1 + shift
        # idx_a1 = center + shift
        
        # Place base pattern such that q1 is at center-1+shift
        # Bids are reverse recorded in patterns usually, but here we just need a transition.
        # Let's place q4, q3, q2, q1 at indices:
        # center-4+shift, center-3+shift, center-2+shift, center-1+shift
        # Asks a1, a2, a3, a4 at indices:
        # center+shift, center+1+shift, center+2+shift, center+3+shift
        
        start = (CENTER + true_shift) - 4
        for i, v in enumerate(base_pattern):
            idx = start + i
            if 0 <= idx < T:
                X[0, idx] = v
        
        data_str = "[" + ", ".join([f"{x:3.0f}" for x in X[0]]) + "]"
        c_shift = int(pivot_counting(X, CENTER).item())
        cl_shift = int(pivot_transition_closest(X, CENTER).item())
        
        print(f"Case: {name}")
        print(f"  Data:  {data_str}")
        print(f"  Target Shift: {true_shift:2d} | ClosestTrans Detection: {cl_shift:2d}")
        print("-" * 80)

if __name__ == "__main__":
    run_comparison()
