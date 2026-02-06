# train_model_imbalance.py
"""
Imbalanced LOB GAN Training Script - Databento Edition

This script implements a GAN for LOB simulation using the imbalanced order book approach.
Optimized for Databento (DBNS) data with 100ms interval sampling and Z-score normalization.
"""

import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model.gan_model import train_gan, Generator, Discriminator
from plots import plot_epochs_evolution, plot_time_series, plot_distribution
from databento_utils import load_dbn_to_numpy, ZScoreNormalizer, QueueNormalizer
from metrics_lib.config import Config

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"
PLOTS_DIR = BASE_DIR / "out/plots/"
DATA_DIR = Path("D:\Amsterdam\Thesis\order_book_databento\dbn_out")

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
Z_DIM = 64              # Noise dimension
HIDDEN_D = 128
HIDDEN_G = 128
BATCH = 512
EPOCHS = 500
CRITIC_STEPS_INITIAL = 5
CRITIC_STEPS_FINAL = 5
GAMMA = 0.999            # Decay rate for critic steps
SHUFFLE_DATA = True
LAMBDA_GP = 10
LR_D = 1e-4
LR_G = 1e-4

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 1 - TRAIN_SPLIT



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LobGanDatasetImbalance(Dataset):
    """
    Imbalanced LOB Dataset based on Cont et al., optimized for Databento.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        market_depth: int = 3,
        interval_ms: int = 100,
        max_price_change: int = 3,
    ):
        self.market_depth = market_depth
        self.interval_ms = interval_ms
        self.max_price_change = max_price_change
        
        # We need at least market_depth + max_price_change columns per side
        self.total_levels = 10  # CSV has 10 levels per side
        assert market_depth + max_price_change <= self.total_levels, \
            f"market_depth ({market_depth}) + max_price_change ({max_price_change}) must be <= {self.total_levels}"
        
        # Load and process data
        self._load_and_process(file_paths)
        
        print(f"Imbalanced LOB Dataset built: S={self.S.shape}, X={self.X.shape}")
        print(f"  Market depth: {self.market_depth}, Interval: {self.interval_ms}ms")

    def _load_and_process(self, file_paths: List[str]):
        all_data = []
        all_price_changes = []
        all_mid_prices = []
        
        for file_path in file_paths:
            print(f"Loading: {file_path}")
            # Load all available levels to allow for shifting
            arr, ts = load_dbn_to_numpy(file_path, interval_ms=self.interval_ms, market_depth=self.total_levels)
            
            if len(arr) == 0:
                continue

            # Prices are at indices 0, 4, 8, ... (ask) and 2, 6, 10, ... (bid)
            best_ask = arr[:, 0]
            best_bid = arr[:, 2]
            mid_price = (best_ask + best_bid) / 2.0
            
            # Estimate tick size (delta) from mid price changes
            mid_diff_abs = np.abs(np.diff(mid_price))
            mid_diffs_nonzero = mid_diff_abs[mid_diff_abs > 1e-6] # Avoid floating point noise
            if len(mid_diffs_nonzero) > 0:
                delta = np.min(mid_diffs_nonzero)
            else:
                delta = 0.25 # Default fallback
                
            print(f"  Estimated tick size (from mid-moves): {delta}")
            self.tick_size = delta
            
            # Compute price changes in ticks
            mid_price_diff = np.diff(mid_price)
            price_change_ticks = np.round(mid_price_diff / delta).astype(int)
            
            # Clip to max_price_change
            price_change_ticks = np.clip(price_change_ticks, -self.max_price_change, self.max_price_change)
            
            # Prepend 0 for first sample (no previous price)
            price_change_ticks = np.concatenate([[0], price_change_ticks])
            
            all_data.append(arr)
            all_price_changes.append(price_change_ticks)
            all_mid_prices.append(mid_price)

        combined_arr = np.concatenate(all_data, axis=0)
        combined_price_changes = np.concatenate(all_price_changes, axis=0)
        self.mid_prices = np.concatenate(all_mid_prices, axis=0)

        # Extract quantities for normalization
        bid_qty_indices = [4*k+3 for k in range(self.total_levels)]
        ask_qty_indices = [4*k+1 for k in range(self.total_levels)]
        
        bid_qtys = combined_arr[:, bid_qty_indices]
        ask_qtys = combined_arr[:, ask_qty_indices]
        
        if Config.NORMALIZATION_METHOD == "zscore":
            print(f"  Applying Z-Score Normalization...")
            self.normalizer = ZScoreNormalizer()
            self.normalizer.fit(bid_qtys, ask_qtys)
        elif Config.NORMALIZATION_METHOD == "queue":
            print(f"  Applying Queue Normalization...")
            self.normalizer = QueueNormalizer(bid_qtys, ask_qtys, 
                                              scale_factor_bid=Config.SCALE_FACTOR_BID, 
                                              scale_factor_ask=Config.SCALE_FACTOR_ASK)
        else:
            raise ValueError(f"Unknown NORMALIZATION_METHOD: {Config.NORMALIZATION_METHOD}")
            
        norm_bid, norm_ask = self.normalizer.normalize(bid_qtys, ask_qtys)
        
        # Put normalized quantities back into a copy of the array
        norm_arr = combined_arr.copy()
        for i, idx in enumerate(bid_qty_indices):
            norm_arr[:, idx] = norm_bid[:, i]
        for i, idx in enumerate(ask_qty_indices):
            norm_arr[:, idx] = norm_ask[:, i]

        # Create imbalanced states
        X_states, S_states = self._create_imbalanced_states(norm_arr, combined_price_changes)

        self.X = torch.tensor(X_states[1:], dtype=torch.float32)
        self.S = torch.tensor(S_states[:-1], dtype=torch.float32)

    def _create_imbalanced_states(self, arr: np.ndarray, price_change_ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(arr)
        # X dimension = 2 * (market_depth + max_price_change) to allow shift
        # But for discriminator simplicity, let's keep X size constant and just shift the window
        # Cont et al. use a larger window for X than S.
        
        # Let's use 2*MARKET_DEPTH for S and 2*(MARKET_DEPTH + MAX_PRICE_CHANGE) for X
        x_dim = 2 * (self.market_depth + self.max_price_change)
        x_s = np.zeros((n_samples, x_dim), dtype=np.float32)
        s_s = np.zeros((n_samples, 2 * self.market_depth), dtype=np.float32)

        for i in range(n_samples):
            shift = price_change_ticks[i]
            
            # Centered view (S_state) at time t
            # Standard order: [q_d^b, ..., q_1^b, q_1^a, ..., q_d^a]
            for k in range(self.market_depth):
                # Bids: far to near (Level D to Level 1)
                s_s[i, self.market_depth - 1 - k] = arr[i, 4*k+3]
                # Asks: near to far (Level 1 to Level D)
                s_s[i, self.market_depth + k] = arr[i, 4*k+1]

            # Imbalanced view (X_state) at time t
            # Base center: self.market_depth + self.max_price_change
            # If shift = +1 (price up), we see more bids and fewer asks in the window
            center = self.market_depth + self.max_price_change
            start_bid = center - self.market_depth - shift
            start_ask = center - shift
            
            # Fill bids (reversed)
            for k in range(self.total_levels):
                idx = center - 1 - k + shift
                if 0 <= idx < x_dim:
                    x_s[i, idx] = arr[i, 4*k+3]
            
            # Fill asks
            for k in range(self.total_levels):
                idx = center + k + shift
                if 0 <= idx < x_dim:
                    x_s[i, idx] = arr[i, 4*k+1]
            
        return x_s, s_s

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load the best previous model")
    args = parser.parse_args()

    dbn_files = sorted(DATA_DIR.glob("*.dbn.zst"))
    if not dbn_files:
        print(f"No .dbn.zst files found in {DATA_DIR}")
        exit(1)

    # Use a subset of files for training
    files = [str(f) for f in dbn_files if f.stat().st_size > 1000]
    
    dataset = LobGanDatasetImbalance(
        file_paths=files[:15], # Use first 3 days
        market_depth=Config.MARKET_DEPTH,
        interval_ms=Config.INTERVAL_MS,
        max_price_change=Config.MAX_PRICE_CHANGE
    )

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    mlflow.set_experiment("gan_databento_adam_optimized")
    
    with mlflow.start_run():
        mlflow.log_params({
            "BATCH": BATCH, "Z_DIM": Z_DIM, "EPOCHS": EPOCHS,
            "LR_D": LR_D, "LR_G": LR_G, "MARKET_DEPTH": Config.MARKET_DEPTH,
            "INTERVAL_MS": Config.INTERVAL_MS, "MAX_PRICE_CHANGE": Config.MAX_PRICE_CHANGE,
            "NORMALIZATION": "Queue-Optimization"
        })
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=SHUFFLE_DATA)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)
        
        x_dim = dataset.X.shape[1]
        s_dim = dataset.S.shape[1]
        
        G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN_G, out_dim=x_dim).to(device)
        D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN_D).to(device)

        if args.load:
            path_g = MODELS_DIR / "generator_databento_gan_best.pth"
            path_d = MODELS_DIR / "discriminator_databento_gan_best.pth"
            if path_g.exists() and path_d.exists():
                print(f"Loading best previous model from {MODELS_DIR}...")
                G.load_state_dict(torch.load(path_g, map_location=device))
                D.load_state_dict(torch.load(path_d, map_location=device))
            else:
                print("No 'best_frob' model found. Starting from scratch.")
        
        results = train_gan(
            generator=G, discriminator=D, 
            train_dataloader=train_loader, val_dataloader=val_loader,
            wgan=True, num_epochs=EPOCHS,
            z_dim=Z_DIM, device=device,
            critic_steps_initial=CRITIC_STEPS_INITIAL,
            critic_steps_final=CRITIC_STEPS_FINAL,
            gamma=GAMMA, lambda_gp=LAMBDA_GP,
            lr_d=LR_D, lr_g=LR_G,
            save_model=True, model_name="databento_gan",
            market_depth=Config.MARKET_DEPTH,
            max_price_change=Config.MAX_PRICE_CHANGE
        )
        
        # Unpack results
        (g_model, d_model, d_loss, g_loss, w_dist, 
         frob_level, frob_diff, price_frob, mean_dev, var_dev) = results
        
        for epoch, (d, g, w, fl, fd, pf, md, vd) in enumerate(zip(
            d_loss, g_loss, w_dist, frob_level, frob_diff, price_frob, mean_dev, var_dev
        )):
            mlflow.log_metric("d_loss", d, step=epoch)
            mlflow.log_metric("g_loss", g, step=epoch)
            mlflow.log_metric("wasserstein_distance", w, step=epoch)
            mlflow.log_metric("frobenius_correlation", fl, step=epoch)
            mlflow.log_metric("frobenius_correlation_diff", fd, step=epoch)
            mlflow.log_metric("price_frobenius", pf, step=epoch)
            mlflow.log_metric("mean_deviation", md, step=epoch)
            mlflow.log_metric("variance_deviation", vd, step=epoch)

    print("\nTraining complete!")
