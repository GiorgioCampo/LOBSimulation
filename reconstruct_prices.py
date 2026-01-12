"""
Price Reconstruction and Comparison Script

This script demonstrates how to:
1. Load the trained imbalanced GAN model
2. Generate sequences autoregressively
3. Decode prices from the imbalanced order book states
4. Compare reconstructed prices to real prices
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from train_model_imbalance import LobGanDatasetImbalance, DATA_DIR, MARKET_DEPTH
from model.gan_model import Generator
from utils import _get_next_state, _random_side_swap

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 12
HIDDEN_G = 64
MODEL_PATH = Path("out/models/generator_imbalanced.pth")  # or generator_imbalanced.pth
GENERATION_HORIZON = 200
NUM_PATHS = 1000


class PriceDecoder:
    """
    Decodes prices from imbalanced order book states.
    
    The key idea:
    - Track the balance between bid and ask columns
    - When balance shifts, it indicates a price change
    - Accumulate price changes to reconstruct the price path
    """
    
    def __init__(self, market_depth: int, initial_price: float, tick_size: float):
        """
        Args:
            market_depth: Number of levels per side
            initial_price: Starting price (e.g., initial mid-price)
            tick_size: Size of one price tick
        """
        self.market_depth = market_depth
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.cumulative_ticks = 0
        
    def detect_price_change(self, state: np.ndarray) -> int:
        """
        Detect price change from an imbalanced state.
        
        The state has 4*market_depth columns:
        - First part: bid quantities (negative)
        - Second part: ask quantities (positive)
        
        We detect imbalance by looking at the sign pattern.
        
        Args:
            state: Order book state (2*market_depth,)
        
        Returns:
            price_change_ticks: Estimated price change in ticks
        """
        # Count positive and negative values
        positive_count = np.sum(state > 0)
        negative_count = np.sum(state < 0)
        
        # Calculate imbalance by checking distance from centered snapshot
        # Expected: market_depth positive, market_depth negative
        # If more negative than expected -> price increased
        # If more positive than expected -> price decreased
        imbalance = negative_count - 2 * MARKET_DEPTH
        
        # Convert to tick change
        # This is a heuristic - each column shift represents ~1 tick
        # price_change_ticks = imbalance // 2
        
        return imbalance
    
    def decode_sequence(self, states: np.ndarray) -> np.ndarray:
        """
        Decode a sequence of states into prices.
        
        Args:
            states: Sequence of states (T, 2*market_depth)
        
        Returns:
            prices: Reconstructed prices (T,)
        """
        prices = np.zeros(len(states))
        self.cumulative_ticks = 0
        
        for t, state in enumerate(states):
            # Detect price change from current state
            tick_change = self.detect_price_change(state)
            self.cumulative_ticks += tick_change
            
            # Reconstruct price
            prices[t] = self.initial_price + self.cumulative_ticks * self.tick_size
        
        return prices
    
    def reset(self):
        """Reset the decoder state."""
        self.cumulative_ticks = 0

def load_real_data(csv_path: str, n_samples: int = 1000):
    """
    Load real data from CSV for comparison.
    
    Args:
        csv_path: Path to CSV file
        n_samples: Number of samples to load
    
    Returns:
        real_prices: Real mid-prices
        real_states: Real imbalanced states
        tick_size: Estimated tick size
    """
    print(f"Loading real data from: {csv_path}")
    
    # Create dataset to get processed states
    dataset = LobGanDatasetImbalance(
        file_paths=[csv_path],
        market_depth=MARKET_DEPTH,
        max_price_change=3
    )
    
    # Load raw CSV to get prices
    df = pd.read_csv(csv_path)
    
    # Extract prices
    best_bid = df['bidPx_0'].values
    best_ask = df['askPx_0'].values
    mid_prices = (best_bid + best_ask) / 2.0

    # Sample n_samples paths from mid_prices
    price_paths = []
    indexes = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(mid_prices) - GENERATION_HORIZON - 1)
        indexes.append(idx)
        price_paths.append(mid_prices[idx: idx + GENERATION_HORIZON])
    
    # Estimate tick size
    bid_diff = np.diff(best_bid)
    bid_diffs_nonzero = bid_diff[bid_diff != 0.0]
    abs_diffs = np.abs(bid_diffs_nonzero)
    k = max(1, int(0.05 * len(abs_diffs)))  # number of smallest diffs to consider
    tick_size = np.median(np.sort(abs_diffs)[:k])
    print(f"  Estimated tick size: {tick_size:.4f}")
    
    # Get states (denormalized)
    # states = dataset.S[:n_samples].numpy()
    # states = states * dataset.std + dataset.mean
    
    print(f"  Loaded {len(mid_prices)} samples")
    print(f"  Estimated tick size: {tick_size:.4f}")
    print(f"  Price range: [{mid_prices.min():.2f}, {mid_prices.max():.2f}]")
    
    return price_paths, indexes, dataset.S, tick_size, dataset


def generate_fake_data(generator, dataset, z_dim, device, initial_states = [], horizon=100, num_paths=10):
    """
    Generate fake sequences using the trained generator.
    
    Args:
        generator: Trained generator model
        dataset: Dataset (for initial states and normalization)
        z_dim: Noise dimension
        device: Device to run on
        horizon: Number of steps to generate
        num_paths: Number of paths to generate
    
    Returns:
        generated_states: Generated states (num_paths, horizon, 2*market_depth)
    """
    print(f"\nGenerating {num_paths} paths of length {horizon}...")
    
    generator.eval()
    
    if initial_states is None or len(initial_states) == 0:
        # Sample initial states
        total_samples = len(dataset.S)
        if total_samples > num_paths:
            indices = np.linspace(0, total_samples - 1, num_paths, dtype=int)
        else:
            indices = np.arange(total_samples)
    else:
        indices = initial_states
        
    current_s = dataset.S[indices].to(device)  # (num_paths, s_dim)
    # Generate sequences
    generated_states = []

    print(current_s)
    
    with torch.no_grad():
        for t in range(horizon):
            z = torch.randn(current_s.size(0), z_dim, device=device)
            x_next = generator(z, current_s)

            generated_states.append(x_next.cpu().numpy())
            current_s = _get_next_state(x_next)
            current_s = _random_side_swap(current_s, p=0)  # Random sign flip for diversity
    
    # Stack into array (num_paths, horizon, features)
    generated_states = np.stack(generated_states, axis=1)
    
    # Denormalize using the square root normalizer
    print("  De-normalizing generated states...")
    # generated_states has shape (num_paths, horizon, 4 * market_depth)
    # We split it into bid and ask parts to denormalize
    # market_depth is dataset.market_depth
    # Our imbalanced state has 4*market_depth columns. 
    # x_s[i, :n_bid_cols] = bid_qty[i, :n_bid_cols][::-1]
    # x_s[i, n_bid_cols:n_bid_cols + n_ask_cols] = ask_qty[i, :n_ask_cols]
    # This means the columns are NOT fixed per side. n_bid_cols varies.

    # ALTERNATIVELY, the user might be fine with just de-normalizing using the best-level C?
    # Or maybe I should modify denormalize to handle the mixed state?
    
    # Actually, in the imbalanced state, the levels are shifted.
    # But for square root norm, C is often very similar across levels if they are already Z-scored.
    # But let's be precise.
    
    # In PriceDecoder.decode_sequence we call it sample by sample.
    # Let's de-normalize sample by sample.


    denorm_states = np.zeros_like(generated_states)
    for p in range(num_paths):
        for t in range(horizon):
            state = generated_states[p, t]
            # Detect imbalance to find split point
            negative_count = np.sum(state < 0)
            pivot = negative_count # since negative are bids and they are first
            
            # This is tricky because C is per level (0 to 9 in training)
            # but here we have market_depth + max_price_change.
            # Let's assume for now that we use the FIRST market_depth C values.
            
            # Let's just apply the transformation element-wise if we can map it.
            # Actually, the simplest way is to implement a generic denormalize 
            # that just uses the sign and square, and maybe a mean C if we want to be simple,
            # OR we pass the full C arrays and map them.
            
            # BUT WAIT! If the input was Z-scored, C is likely near 0.8 for all levels.
            
            # Let's see how metrics1.py does it. It fits on ALL real data and normalizes.
            
            # Let's just de-normalize manually here since the mapping is complex.
            for j in range(len(state)):
                val = state[j]
                if j < pivot: # Bid
                    # How to map j to actual level? 
                    # Bids are in reverse: bid_k, ..., bid_0
                    # So pivot-1 is bid_0.
                    level = (pivot - 1) - j
                    if level < len(dataset.normalizer.C_bid):
                        C = dataset.normalizer.C_bid[level]
                    else:
                        C = dataset.normalizer.C_bid[-1]
                    denorm_states[p, t, j] = np.sign(val) * (val * C) ** 2
                else: # Ask
                    level = j - pivot
                    if level < len(dataset.normalizer.C_ask):
                        C = dataset.normalizer.C_ask[level]
                    else:
                        C = dataset.normalizer.C_ask[-1]
                    denorm_states[p, t, j] = np.sign(val) * (val * C)**2 
    
    generated_states = denorm_states
    print(f"  Generated shape: {generated_states.shape}")
    
    return generated_states


def plot_price_comparison(real_prices_list, decoded_prices_list, tick_size, save_path=None):
    """
    Plot real vs decoded prices.
    
    Args:
        real_prices: Real prices (T,)
        decoded_prices_list: List of decoded price sequences [(T,), ...]
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(14, 6))

    real_prices_list = real_prices_list[:50] // tick_size # Plot first 5 paths for clarity
    decoded_prices_list = decoded_prices_list[:50] // tick_size
    
    # Plot real prices
    for i, real_prices in enumerate(real_prices_list):  # drop last timestamp
        plt.plot(real_prices - real_prices[0], ':', color='deepskyblue', alpha=0.75)
    
    # Plot decoded prices
    for i, decoded_prices in enumerate(decoded_prices_list):
        plt.plot(decoded_prices - decoded_prices[0], ':', color="orange", alpha=0.75)

    # Set background color
    # plt.gca().set_facecolor('#e0e0e')
    plt.plot([], [], ':', color='deepskyblue', label='Real')  # Dummy for legend
    plt.plot([], [], ':', color='orange', label='Generated')  # Dummy for legend
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Mid-Price', fontsize=12)
    plt.title('Real vs Generated Prices', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def compute_price_statistics(real_prices, decoded_prices_list, tick_size):
    """
    Compute statistics comparing real and decoded prices.
    
    Args:
        real_prices: Real prices (T,)
        decoded_prices_list: List of decoded price sequences
    """
    print("\n" + "="*80)
    print("PRICE STATISTICS")
    print("="*80)
    
    # Real price statistics
    real_returns = np.diff(real_prices) // tick_size
    real_prices = np.array(real_prices)
    print(f"\nReal Prices:")
    print(f"  Mean: {real_prices.mean():.4f}")
    print(f"  Std: {real_prices.std():.4f}")
    print(f"  Min: {real_prices.min():.4f}")
    print(f"  Max: {real_prices.max():.4f}")
    print(f"  Return Mean: {real_returns.mean():.6f}")
    print(f"  Return Std: {real_returns.std():.6f}")
    
    # Generated price statistics
    print(f"\nGenerated Prices (averaged over {len(decoded_prices_list)} paths):")
    
    all_decoded = np.array(decoded_prices_list)
    decoded_mean = all_decoded.mean(axis=0)
    decoded_std = all_decoded.std(axis=0)
    
    decoded_returns = np.diff(all_decoded, axis=1)  // tick_size
    
    print(f"  Mean: {decoded_mean.mean():.4f}")
    print(f"  Std: {decoded_std.mean():.4f}")
    print(f"  Min: {all_decoded.min():.4f}")
    print(f"  Max: {all_decoded.max():.4f}")
    print(f"  Return Mean: {decoded_returns.mean():.6f}")
    print(f"  Return Std: {decoded_returns.std():.6f}")

import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_distribution(real_seqs, fake_seqs, tick_size, save_path=None):
    """
    Smooth CDFs + empirical deciles for real and fake separately
    (replicates Figure 7b behaviour).
    """

    plt.style.use("ggplot")

    # --- Convert to tick-return distributions ---
    rp = np.asarray(real_seqs)
    if rp.ndim == 1:
        real_returns = np.array([rp[-1] - rp[0]], dtype=float) // tick_size
    else:
        real_returns = np.array([seq[-1] - seq[0] for seq in rp], dtype=float) // tick_size

    fake_returns = np.array([seq[-1] - seq[0] for seq in fake_seqs], dtype=float) // tick_size

    # --- Empirical deciles for EACH distribution ---
    #   10%, 20%, ... 90%
    dec_percentiles = np.arange(0, 105, 5)

    real_deciles = np.percentile(real_returns, dec_percentiles)
    fake_deciles = np.percentile(fake_returns, dec_percentiles)

    # Their ECDF values (empirical, not smoothed)
    real_ecdf_dec = dec_percentiles / 100.0
    fake_ecdf_dec = dec_percentiles / 100.0

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # Decile markers – each distribution uses ITS OWN deciles
    ax.plot(real_deciles, real_ecdf_dec, label="Real",
               color="deepskyblue", marker="x", linewidth=1.8)

    ax.plot(fake_deciles, fake_ecdf_dec, label="Generated",
               color="orange", marker="x", linewidth=1.8)

    # Styling
    ax.set_xlabel("ticks [δ]", color="black")
    ax.set_ylabel("cdf", color="black")
    ax.set_title("Smoothed CDF with Empirical Deciles (Real vs Generated)", color="black")
    ax.legend()

    ax.grid(True, alpha=0.25, linestyle="--")
    for s in ax.spines.values():
        s.set_color("black")
    ax.tick_params(colors="black")

    leg = ax.legend(frameon=True)
    # leg.get_frame().set_facecolor("#fff")
    # leg.get_frame().set_edgecolor("black")
    for t in leg.get_texts():
        t.set_color("black")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    plt.show()


def save_reconstruction_results(generated_states, tick_size, initial_prices, market_depth, horizon, num_paths):
    """
    Save reconstructed LOB data into two CSV files:
    1. Mean over time (averaging all paths at each time step)
    2. Concatenated paths (appending all paths with price adjustments for continuity)
    """
    print(f"\nProcessing and saving reconstructed data for {num_paths} paths...")
    
    all_paths_data = []
    
    for i in range(num_paths):
        path_data = []
        # In PriceDecoder.decode_sequence, initial_price is the price BEFORE the first state's change
        current_price = initial_prices[i]
        
        for t in range(horizon):
            state = generated_states[i, t]
            
            # Detect tick change (same logic as PriceDecoder.detect_price_change)
            # state has 4*market_depth columns
            negative_count = np.sum(state < 0)
            tick_change = negative_count - 2 * market_depth
            
            # Update mid price (cumulative)
            current_price += tick_change * tick_size
            
            # Pivot point for quantities
            # The split between bid and ask in the X state shifts with tick_change
            pivot = 2 * market_depth + tick_change
            
            # Extract quantities and prices for all levels
            record = {
                'path_id': i,
                'step': t,
                'mid_price': current_price
            }
            
            for j in range(market_depth):
                # Bid level j (j=0 is best bid)
                # Bids are stored in reverse order in X (bid_k, ..., bid_0)
                bid_idx = pivot - 1 - j
                if 0 <= bid_idx < state.shape[0]:
                   record[f'bidQty_{j}'] = abs(state[bid_idx])
                else:
                   record[f'bidQty_{j}'] = 0.0
                
                # Assume 1 tick spread for simplicity (mid +/- 0.5 ticks)
                record[f'bidPx_{j}'] = current_price - (j + 0.5) * tick_size
                
                # Ask level j (j=0 is best ask)
                # Asks are stored in normal order in X (ask_0, ..., ask_k)
                ask_idx = pivot + j
                if 0 <= ask_idx < state.shape[0]:
                    record[f'askQty_{j}'] = abs(state[ask_idx])
                else:
                    record[f'askQty_{j}'] = 0.0
                    
                record[f'askPx_{j}'] = current_price + (j + 0.5) * tick_size
                
            path_data.append(record)
        all_paths_data.append(path_data)
        
    # Create out directory if it doesn't exist
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Mean over time (x-axis)
    # Combine all paths and group by step
    flat_data = [item for path in all_paths_data for item in path]
    df_all = pd.DataFrame(flat_data)
    
    print("  Calculating mean over all paths...")
    df_mean = df_all.groupby('step').mean().reset_index()
    # Remove path_id from mean calculation
    if 'path_id' in df_mean.columns:
        df_mean.drop(columns=['path_id'], inplace=True)
        
    mean_csv_path = out_dir / 'reconstructed_mean.csv'
    df_mean.to_csv(mean_csv_path, index=False)
    print(f"  Saved mean reconstructed data to {mean_csv_path}")
    
    # 2. Concatenated paths with adjustment ("at the same level")
    print("  Concatenating paths with continuity adjustments...")
    adjusted_flat_data = []
    last_price = 0
    
    for i in range(num_paths):
        path = all_paths_data[i]
        if i == 0:
            last_price = path[-1]['mid_price']
            adjusted_flat_data.extend(path)
        else:
            # Adjust current path to start where previous path ended
            # previous_path_end = last_price
            # current_path_start = path[0]['mid_price']
            price_offset = last_price - path[0]['mid_price']
            
            for record in path:
                adj_record = record.copy()
                adj_record['mid_price'] += price_offset
                for j in range(market_depth):
                    adj_record[f'bidPx_{j}'] += price_offset
                    adj_record[f'askPx_{j}'] += price_offset
                adjusted_flat_data.append(adj_record)
            
            last_price = adjusted_flat_data[-1]['mid_price']
            
    df_concat = pd.DataFrame(adjusted_flat_data)
    concat_csv_path = out_dir / 'reconstructed_concatenated.csv'
    df_concat.to_csv(concat_csv_path, index=False)
    print(f"  Saved concatenated reconstructed data to {concat_csv_path}")




def main():
    print("="*80)
    print("PRICE RECONSTRUCTION AND COMPARISON")
    print("="*80)
    
    # Find CSV files
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {DATA_DIR}")
        return
    
    # Use the last file (same as training)
    csv_path = str(csv_files[0])
    print(f"\nUsing CSV: {csv_files[0].name}")
    
    # Load real data
    real_prices, indexes, real_states, tick_size, dataset = load_real_data(csv_path, n_samples=NUM_PATHS)
    print("Normalization C")
    print(dataset.normalizer.C_bid, dataset.normalizer.C_ask)
    # Check if model exists
    # Otherwise: test run with real data only
    if not MODEL_PATH.exists():
        print(f"\nWARNING: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python train_model_imbalance.py")
        print("\nFor now, we'll just demonstrate price decoding on real data...")
        
        # Decode real prices
        decoder = PriceDecoder(
            market_depth=MARKET_DEPTH,
            initial_price=real_prices[0],
            tick_size=tick_size
        )
        
        decoded_real = decoder.decode_sequence(real_states)
        
        # Plot comparison
        plot_price_comparison(
            real_prices,
            [decoded_real],
            tick_size,
            save_path="out/price_reconstruction_real_only.png"
        )
        
        return
    
    # Load trained generator
    print(f"\nLoading trained generator from: {MODEL_PATH}")
    
    x_dim = 4 * MARKET_DEPTH
    s_dim = 2 * MARKET_DEPTH
    
    generator = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN_G, out_dim=x_dim)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.to(DEVICE)
    generator.eval()
    
    print("  Generator loaded successfully!")
    
    # Generate fake data
    generated_states = generate_fake_data(
        generator, dataset, Z_DIM, DEVICE,
        initial_states=indexes,
        horizon=GENERATION_HORIZON,
        num_paths=NUM_PATHS
    )
    
    # decoders = []
    # # Decode prices for each generated path
    # decoder = PriceDecoder(
    #     market_depth=MARKET_DEPTH,
    #     initial_price=real_prices[0],
    #     tick_size=tick_size
    # )
    
    decoded_prices_list = []
    
    for i in range(NUM_PATHS):
        decoder = PriceDecoder(
            market_depth=MARKET_DEPTH,
            initial_price=real_prices[i][0],
            tick_size=tick_size
        )
        # decoder.reset()
        decoded_prices = decoder.decode_sequence(generated_states[i])
        decoded_prices_list.append(decoded_prices)
    
    # Save quantities and prices per level to CSVs
    initial_prices = [real_prices[i][0] for i in range(NUM_PATHS)]
    save_reconstruction_results(
        generated_states, tick_size, initial_prices, 
        MARKET_DEPTH, GENERATION_HORIZON, NUM_PATHS
    )
    
    # Compute statistics
    compute_price_statistics(real_prices, decoded_prices_list, tick_size)

    plot_cumulative_distribution(real_prices, decoded_prices_list, tick_size,
        save_path="out/price_reconstruction_cdf.png"
    )
    
    # Plot comparison
    plot_price_comparison(
        real_prices,
        decoded_prices_list,  # Plot first 5 paths for clarity
        tick_size,
        save_path="out/price_reconstruction_comparison.png"
    )
    


    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
