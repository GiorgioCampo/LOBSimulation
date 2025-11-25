# dataset_lob_gan.py
import torch
import mlflow
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model.gan_model import train_gan, Generator, Discriminator
from plots import plot_epochs_evolution

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"

# Use OPTUNA for hyperparameter tuning: https://optuna.org/
Z_DIM = 6               # Noise dimension. 25% - 50% of total dimensions
HIDDEN_D = 64
HIDDEN_G = 32
BATCH = 8
EPOCHS = 300
CRITIC_STEPS_INITIAL = 5
CRITIC_STEPS_FINAL = 2
GAMMA = 0.97            # Decay rate for critic steps
MARKET_DEPTH = 3
SHUFFLE_DATA = True
LAMBDA_GP = 10         # Increase to penalize discriminator more
LR_D = 2e-5
LR_G = 1e-5
USE_DIFFS = False
INCLUDE_DIFFS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
#                      DATASET (NEW)
# ===============================================================
class LOBGANDataset(Dataset):
    """
    Limit Order Book Dataset with proper preprocessing:
    1. Loads multiple CSV files (one per day)
    2. Optionally applies differentiation PER FILE (to avoid cross-day artifacts)
    3. Concatenates all files
    4. Standardizes the concatenated data globally
    5. Filters L2 columns and prepares (X_next, S_curr) pairs
    """

    def __init__(
        self,
        file_paths: List[str],
        market_depth: int = 5,
        differentiate: bool = False,
        include_diffs_in_state: bool = False,
    ):
        """
        Args:
            file_paths: List of CSV file paths (one per day)
            market_depth: Number of price levels to include (0 to market_depth-1)
            differentiate: If True, replace data with first-order differences per file
            include_diffs_in_state: If True, augment state S with differences [snapshot, diff]
        """
        self.market_depth = market_depth
        self.differentiate = differentiate
        self.include_diffs_in_state = include_diffs_in_state

        # Load and preprocess data
        lob_df = self._load_and_preprocess(file_paths)

        # Convert to torch tensors
        self._build_tensors(lob_df)

        print(f"Dataset built: S={self.S.shape}, X={self.X.shape}")

    def _load_and_preprocess(self, file_paths: List[str]) -> pd.DataFrame:
        """Load files, optionally differentiate per file, concatenate, and standardize."""
        processed_dfs = []

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            
            # Handle time column
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            
            # Keep only last record per timestamp
            df = df.sort_index()
            df = df.groupby(df.index).last()

            # Separate numeric and non-numeric columns
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

            # Apply differentiation PER FILE if requested
            if self.differentiate:
                numeric_df = df[numeric_cols].diff().fillna(0)
            else:
                numeric_df = df[numeric_cols].copy()

            # Reattach non-numeric columns
            for col in non_numeric_cols:
                numeric_df[col] = df[col]

            processed_dfs.append(numeric_df)

        # Concatenate all files
        combined_df = pd.concat(processed_dfs, ignore_index=False)

        # Standardize globally AFTER concatenation
        numeric_cols = combined_df.select_dtypes(include='number').columns.tolist()
        combined_df[numeric_cols] = (
            combined_df[numeric_cols] - combined_df[numeric_cols].mean()
        ) / (combined_df[numeric_cols].std() + 1e-8)

        return combined_df

    def _build_tensors(self, lob_df: pd.DataFrame):
        """Extract L2 columns and build training tensors."""
        # Select L2 snapshot columns in order
        valid_cols = []
        for col in lob_df.columns:
            if any(col.startswith(prefix) for prefix in ["bidPx_", "bidQty_", "askPx_", "askQty_"]):
                level = int(col.split('_')[-1])
                if level < self.market_depth:
                    valid_cols.append(col)

        print(f"Selected columns: {valid_cols}")
        lob_df = lob_df[valid_cols]

        # Convert to numpy
        mat = lob_df.values.astype(np.float32)

        # Build state representation
        if self.include_diffs_in_state:
            # Compute differences for state augmentation
            diff = np.diff(mat, axis=0)
            diff = np.concatenate([np.zeros_like(diff[:1]), diff], axis=0)
            S = np.concatenate([mat, diff], axis=1)
        else:
            S = mat

        # Create (X_next, S_curr) pairs
        X = mat[1:]  # Next snapshot
        S = S[:-1]   # Current state

        self.S = torch.tensor(S, dtype=torch.float32)
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Returns (X_next, S_curr) pair."""
        return self.X[idx], self.S[idx]

# ===============================================================
#                      TRAINING ENTRY POINT
# ===============================================================
if __name__ == "__main__":

    files = [
        "out/data/20191001/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191002/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191003/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191004/FLEX_L2_SNAPSHOT.csv"
    ]

    # Option 1: Raw snapshots, standardized
    dataset = LOBGANDataset(
        file_paths=files,
        market_depth=MARKET_DEPTH,
        differentiate=USE_DIFFS,
        include_diffs_in_state=INCLUDE_DIFFS
    )

    # # Option 2: First-order differences (per day), standardized
    # dataset = LOBGANDataset(
    #     file_paths=files,
    #     market_depth=MARKET_DEPTH,
    #     differentiate=True,
    #     include_diffs_in_state=False
    # )

    # # Option 3: Raw snapshots with differences augmented in state
    # dataset = LOBGANDataset(
    #     file_paths=files,
    #     market_depth=MARKET_DEPTH,
    #     differentiate=False,
    #     include_diffs_in_state=True
    # )
    
    # mlflow.set_experiment("gan_experiments")

    with mlflow.start_run():

        if USE_DIFFS: model_name = "diffs" 
        elif INCLUDE_DIFFS: model_name = "augmented" 
        else: model_name = "raw"

        # log hyperparameters
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
        mlflow.log_param("USE_DIFFS", USE_DIFFS)
        mlflow.log_param("INCLUDE_DIFFS", INCLUDE_DIFFS)
        mlflow.log_param("model_name", model_name)

        # your original pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = DataLoader(dataset, batch_size=BATCH, shuffle=SHUFFLE_DATA)

        x_dim = dataset.X.shape[1]
        s_dim = dataset.S.shape[1]

        G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN_G, out_dim=x_dim)
        D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN_D)

        generator, discriminator, d_loss, g_loss, w_dist = train_gan(
            generator=G, discriminator=D, dataloader=loader,
            wgan=True, num_epochs=EPOCHS,
            z_dim=Z_DIM, device=device,
            critic_steps_initial=CRITIC_STEPS_INITIAL,
            critic_steps_final=CRITIC_STEPS_FINAL,
            gamma=GAMMA, lambda_gp=LAMBDA_GP,
            lr_d=LR_D, lr_g=LR_G,
            save_model=True, model_name=model_name
        )

        # log full per-epoch curves
        for epoch, (d, g, w) in enumerate(zip(d_loss, g_loss, w_dist)):
            mlflow.log_metric("d_loss", d, step=epoch)
            mlflow.log_metric("g_loss", g, step=epoch)
            mlflow.log_metric("wasserstein_distance", w, step=epoch)

        # log trained models
        mlflow.pytorch.log_model(generator, "generator")
        mlflow.pytorch.log_model(discriminator, "discriminator")

    # Save trained models
    torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}.pth")
    torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}.pth")

    # Plot d_loss, g_loss, w_dist and save them
    plot_epochs_evolution(d_loss, "Discriminator loss")
    plot_epochs_evolution(g_loss, "Generator loss")
    plot_epochs_evolution(w_dist, "Wasserstein distance")
    
    
