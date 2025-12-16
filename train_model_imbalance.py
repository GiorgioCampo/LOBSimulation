# train_model_imbalance.py
"""
Imbalanced LOB GAN Training Script

This script implements a GAN for LOB simulation using the imbalanced order book approach
from Cont et al. "Limit Order Book Simulation with Generative Adversarial Networks".

Key concepts:
- S-states are computed based on price changes (no price change, increase, or decrease)
- Only volumes are used as features (prices are hidden)
- Columns are dynamically balanced based on price tick changes
- Ask quantities are negative, bid quantities are positive
"""

import torch
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model.gan_model import train_gan, Generator, Discriminator
from plots import plot_epochs_evolution, plot_time_series

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"
PLOTS_DIR = BASE_DIR / "out/plots/"
DATA_DIR = BASE_DIR / "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/"

# Hyperparameters
Z_DIM = 3              # Noise dimension
HIDDEN_D = 64
HIDDEN_G = 64
BATCH = 2048
EPOCHS = 2500
CRITIC_STEPS_INITIAL = 15
CRITIC_STEPS_FINAL = 1
GAMMA = 0.95            # Decay rate for critic steps
MARKET_DEPTH = 3         # Number of levels to use (max 10 based on CSV)
SHUFFLE_DATA = True
LAMBDA_GP = 10
LR_D = 1e-4
LR_G = 1e-5

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
#                  IMBALANCED LOB DATASET
# ===============================================================
class LobGanDatasetImbalance(Dataset):
    """
    Imbalanced LOB Dataset based on Cont et al.
    
    This dataset:
    1. Loads CSV files with 40 columns (10 ask prices, 10 ask quantities, 10 bid prices, 10 bid quantities)
    2. Computes price changes between consecutive snapshots
    3. Creates imbalanced order book states where columns shift based on price changes
    4. Uses only volumes as features (prices are used only to detect changes)
    
    The key idea:
    - By default, first MARKET_DEPTH columns are positive (bids), next MARKET_DEPTH are negative (asks)
    - When price increases by N ticks, N columns shift from bid to ask side
    - When price decreases by N ticks, N columns shift from ask to bid side
    """
    
    def __init__(
        self,
        file_paths: List[str],
        market_depth: int = 10,
        max_price_change: int = 3,  # Maximum price change in ticks to handle
    ):
        """
        Args:
            file_paths: List of CSV file paths
            market_depth: Number of price levels to use per side
            max_price_change: Maximum price change in ticks (usually <= 3)
        """
        self.market_depth = market_depth
        self.max_price_change = max_price_change
        
        # We need at least market_depth + max_price_change columns per side
        self.total_levels = 10  # CSV has 10 levels per side
        assert market_depth + max_price_change <= self.total_levels, \
            f"market_depth ({market_depth}) + max_price_change ({max_price_change}) must be <= {self.total_levels}"
        
        # Load and process data
        self._load_and_process(file_paths)
        
        print(f"Imbalanced LOB Dataset built: S={self.S.shape}, X={self.X.shape}")
        print(f"  Market depth: {self.market_depth}")
        print(f"  Max price change handled: {self.max_price_change} ticks")
    
    def _load_and_process(self, file_paths: List[str]):
        """Load CSV files and create imbalanced order book states."""
        all_X_states = []
        all_S_states = []
        all_price_changes = []
        
        for file_path in file_paths:
            print(f"Loading: {file_path}")
            df = pd.read_csv(file_path)
            
            # Extract columns
            ask_px_cols = [f'askPx_{i}' for i in range(self.total_levels)]
            ask_qty_cols = [f'askQty_{i}' for i in range(self.total_levels)]
            bid_px_cols = [f'bidPx_{i}' for i in range(self.total_levels)]
            bid_qty_cols = [f'bidQty_{i}' for i in range(self.total_levels)]
            
            # Extract data
            ask_px = df[ask_px_cols].values  # (n_samples, 10)
            ask_qty = df[ask_qty_cols].values
            bid_px = df[bid_px_cols].values
            bid_qty = df[bid_qty_cols].values
            
            print(f"  Loaded {len(df)} samples")
            
            # Compute price changes (in ticks)
            # We'll use the best bid/ask to determine price changes
            best_bid = bid_px[:, 0]
            best_ask = ask_px[:, 0]
            mid_price = (best_bid + best_ask) / 2.0
            
            # Compute tick size (assume constant, use median spread as proxy)
            spread = best_ask - best_bid

            # Extract only qty_0 for both bid and ask
            bid_qty_0 = bid_qty[:, 0]
            ask_qty_0 = ask_qty[:, 0]
            
            # Plot all time series using the generalized function
            plot_time_series(mid_price, "Mid Price", ylabel="Price", 
                           filename="mid_price.png", output_dir=str(PLOTS_DIR))
            
            plot_time_series(spread, "Spread", ylabel="Spread", 
                           filename="spread.png", output_dir=str(PLOTS_DIR))
            
            plot_time_series(best_ask, "Best Ask", ylabel="Ask", 
                           filename="best_ask.png", output_dir=str(PLOTS_DIR))
            
            plot_time_series(best_bid, "Best Bid", ylabel="Bid", 
                           filename="best_bid.png", output_dir=str(PLOTS_DIR))
            
            # Compute and print bid diff statistics
            bid_diff = np.diff(best_bid)
            plot_time_series(bid_diff, "Bid Diff", ylabel="Diff", 
                           filename="bid_diff.png", output_dir=str(PLOTS_DIR),
                           print_stats=True, stats_label="Bid diff")
            
            plot_time_series(bid_qty_0, "Bid Quantity (Level 0)", ylabel="Bid Quantity", 
                           filename="bid_qty_0.png", output_dir=str(PLOTS_DIR),
                           print_stats=True, stats_label="Bid qty_0")
            
            plot_time_series(ask_qty_0, "Ask Quantity (Level 0)", ylabel="Ask Quantity", 
                           filename="ask_qty_0.png", output_dir=str(PLOTS_DIR),
                           print_stats=True, stats_label="Ask qty_0")
                           
            # Compute price changes in ticks
            mid_price_diff = np.diff(mid_price)

            # Estimate tick size
            bid_diff = np.diff(best_bid)
            bid_diffs_nonzero = bid_diff[bid_diff != 0.0]
            abs_diffs = np.abs(bid_diffs_nonzero)
            k = max(1, int(0.05 * len(abs_diffs)))  # number of smallest diffs to consider
            delta = np.median(np.sort(abs_diffs)[:k])

            # Round to nearest tick
            price_change_ticks = np.round(mid_price_diff / delta).astype(int)
            print(f"  Price change ticks (first 20): {price_change_ticks[:20]}")
            print(f"  Estimated tick size: {delta}")
            print(f"  Price change ticks stats: min={price_change_ticks.min()}, max={price_change_ticks.max()}")
            # Fraction of ticks inside allowed range
            frac_in_range = np.mean(
                (price_change_ticks >= -self.max_price_change) & 
                (price_change_ticks <= self.max_price_change)
            )*100.0
            print(f"  Price change ticks in range: {frac_in_range:.2f}%")

            # Clip to max_price_change
            price_change_ticks = np.clip(price_change_ticks, -self.max_price_change, self.max_price_change)
            
            # Prepend 0 for first sample (no previous price)
            price_change_ticks = np.concatenate([[0], price_change_ticks])
            
            # Create imbalanced states
            # For each sample, we create a state vector with balanced columns based on previous price change
            X_states, S_states = self._create_imbalanced_states(bid_qty, ask_qty, price_change_ticks)

            all_X_states.append(X_states)
            all_S_states.append(S_states)
            all_price_changes.append(price_change_ticks)
        
        # Concatenate all files
        combined_X_states = np.concatenate(all_X_states, axis=0)
        combined_S_states = np.concatenate(all_S_states, axis=0)
        
        # Standardize globally
        # self.mean = combined_states.mean(axis=0)
        # self.std = combined_states.std(axis=0) + 1e-8
        # combined_states = (combined_states - self.mean) / self.std
        
        # Create (X_next, S_curr) pairs
        # S states should be the centered states at time t
        # X states should be the (possibly) imbalanced states at time t+1
        self.X = torch.tensor(combined_X_states[1:], dtype=torch.float32)
        self.S = torch.tensor(combined_S_states[:-1], dtype=torch.float32)
        
        # Store for later use
        self.qty_indices = list(range(2 * self.market_depth))
        self.bid_qty_indices = list(range(self.market_depth))
        self.ask_qty_indices = list(range(self.market_depth, 2 * self.market_depth))
        self.labels = [f"Bid {i}" for i in range(self.market_depth, 0, -1)] + \
                      [f"Ask {i}" for i in range(1, self.market_depth + 1)]
    
    def _create_imbalanced_states(
        self, 
        bid_qty: np.ndarray, 
        ask_qty: np.ndarray, 
        price_change_ticks: np.ndarray
    ) -> np.ndarray:
        """
        Create imbalanced order book states based on price changes.
        
        The key idea:
        - We maintain a "centered" view of the order book with 2*market_depth columns
        - Bid quantities are positive, ask quantities are negative
        - When price increases, the center shifts up (more ask columns)
        - When price decreases, the center shifts down (more bid columns)
        
        Args:
            bid_qty: Bid quantities (n_samples, 10)
            ask_qty: Ask quantities (n_samples, 10)
            price_change_ticks: Price changes in ticks (n_samples,)
        
        Returns:
            states: Imbalanced states (n_samples, 2*market_depth)
        """
        n_samples = len(bid_qty)
        x_s = np.zeros((n_samples, 4 * self.market_depth), dtype=np.float32)
        s_s = np.zeros((n_samples, 2 * self.market_depth), dtype=np.float32)

        for i in range(n_samples):
            # Price change only compared to previous state
            price_change = price_change_ticks[i]
            
            # Determine the split point between bid and ask
            # Default: market_depth bids, market_depth asks
            # Price increase: more bids, fewer asks
            # Price decrease: fewer bids, more asks
            n_bid_cols = 2 * self.market_depth + price_change
            n_ask_cols = 2 * self.market_depth - price_change
            
            # Check n_bid_cols and n_ask_cols sum to 2*market_depth
            assert n_bid_cols + n_ask_cols == 4 * self.market_depth, \
                f"n_bid_cols ({n_bid_cols}) + n_ask_cols ({n_ask_cols}) != {2 * self.market_depth}"                                                           
            
            # Fill bid columns (negative) in reverse order (bid_3, bid_2, bid_1, bid_0)
            x_s[i, :n_bid_cols] = bid_qty[i, :n_bid_cols][::-1]
            # Fill ask columns (positive)
            x_s[i, n_bid_cols:n_bid_cols + n_ask_cols] = ask_qty[i, :n_ask_cols]

            # Fill S states (centered)
            s_s[i, :self.market_depth] = bid_qty[i, :self.market_depth][::-1]
            s_s[i, self.market_depth:2 * self.market_depth] = ask_qty[i, :self.market_depth]
        
        return x_s, s_s
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """Returns (X_next, S_curr) pair."""
        return self.X[idx], self.S[idx]


# ===============================================================
#                      TRAINING ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    
    # Find all CSV files in the data directory
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        print("Please check the data directory path.")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Use first file for initial testing (you can add more later)
    files = [str(csv_files[0])] #atttenzione spread diverso per multipli giorni?
    
    # Create dataset
    dataset = LobGanDatasetImbalance(
        file_paths=files,
        market_depth=MARKET_DEPTH,
        max_price_change=3
    )

    # Create Train / Validation split
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Show first 5 samples of dataset
    for i in range(0, 20):
        x_sample, s_sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  X: {x_sample.numpy()}")
        print(f"  S: {s_sample.numpy()}")

    # MLflow experiment tracking
    mlflow.set_experiment("gan_imbalanced_experiments")
    
    with mlflow.start_run():
        
        model_name = "imbalanced"
        
        # Log hyperparameters
        mlflow.log_param("BATCH", BATCH)
        mlflow.log_param("Z_DIM", Z_DIM)
        mlflow.log_param("HIDDEN_G", HIDDEN_G)
        mlflow.log_param("HIDDEN_D", HIDDEN_D)
        mlflow.log_param("EPOCHS", EPOCHS)
        mlflow.log_param("LR_D", LR_D)
        mlflow.log_param("LR_G", LR_G)
        mlflow.log_param("GAMMA", GAMMA)
        mlflow.log_param("LAMBDA_GP", LAMBDA_GP)
        mlflow.log_param("CRITIC_STEPS_INITIAL", CRITIC_STEPS_INITIAL)
        mlflow.log_param("CRITIC_STEPS_FINAL", CRITIC_STEPS_FINAL)
        mlflow.log_param("MARKET_DEPTH", MARKET_DEPTH)
        mlflow.log_param("model_name", model_name)
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=SHUFFLE_DATA)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)
        
        # Get dimensions
        x_dim = dataset.X.shape[1]
        s_dim = dataset.S.shape[1]
        
        print(f"\nModel dimensions:")
        print(f"  X dimension: {x_dim}")
        print(f"  S dimension: {s_dim}")
        print(f"  Z dimension: {Z_DIM}")
        
        # Create models
        G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN_G, out_dim=x_dim)
        D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN_D)
        
        # Train
        generator, discriminator, d_loss, g_loss, w_dist, frob_level_history, frob_diff_history, price_frob_history, mean_dev_history, var_dev_history = train_gan(
            generator=G, discriminator=D, 
            train_dataloader=train_loader, val_dataloader=val_loader,
            wgan=True, num_epochs=EPOCHS,
            z_dim=Z_DIM, device=device,
            critic_steps_initial=CRITIC_STEPS_INITIAL,
            critic_steps_final=CRITIC_STEPS_FINAL,
            gamma=GAMMA, lambda_gp=LAMBDA_GP,
            lr_d=LR_D, lr_g=LR_G,
            save_model=True, model_name=model_name
        )
        
        # Log metrics
        for epoch, (d, g, w) in enumerate(zip(d_loss, g_loss, w_dist)):
            mlflow.log_metric("d_loss", d, step=epoch)
            mlflow.log_metric("g_loss", g, step=epoch)
            mlflow.log_metric("wasserstein_distance", w, step=epoch)
        
        # Log trained models
        mlflow.pytorch.log_model(generator, "generator")
        mlflow.pytorch.log_model(discriminator, "discriminator")
    
    # Save trained models
    torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}.pth")
    torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}.pth")
    
    # Plot training curves
    plot_epochs_evolution([d_loss], ["Discriminator loss"])
    plot_epochs_evolution([g_loss], ["Generator loss"])
    plot_epochs_evolution([w_dist], ["Wasserstein distance"])
    plot_epochs_evolution([frob_level_history, frob_diff_history], ["Q Frobenius level", "Q Frobenius difference"])
    plot_epochs_evolution([price_frob_history], ["Price Frobenius"])
    plot_epochs_evolution([mean_dev_history, var_dev_history], ["Mean deviation", "Variance deviation"])

    print("\nTraining complete!")
