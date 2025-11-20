# dataset_lob_gan.py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

from model.gan_model import train_gan, Generator, Discriminator
from plots import plot_real_vs_generated

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"

# Use OPTUNA for hyperparameter tuning: https://optuna.org/
Z_DIM = 15             # Noise dimension. 25% - 50% of total dimensions
HIDDEN = 64
BATCH = 4
EPOCHS = 500
CRITIC_STEPS = 4
MARKET_DEPTH = 3
SHUFFLE_DATA = True
LAMBDA_GP = 10        # Increase to penalize discriminator more
LR_D = 6e-5
LR_G = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
#                      DATASET (NEW)
# ===============================================================
class LOBGanDataset(Dataset):
    """
    - Accepts raw L2_SNAPSHOT df (possibly containing time column)
    - Keeps only last record per timestamp
    - Extracts numeric L2 columns
    - Normalizes inside the dataset
    - Adds first-order differences (Δ snapshot)
    - Returns (X_next, S_curr)
    """

    def __init__(self, lob_df: pd.DataFrame, market_depth: int = 5):
        super().__init__()

        self.market_depth = market_depth

        # ------------------------------
        # Index handling
        # ------------------------------
        if not isinstance(lob_df.index, pd.DatetimeIndex):
            if "time" in lob_df.columns:
                lob_df["time"] = pd.to_datetime(lob_df["time"])
                lob_df = lob_df.set_index("time")
            else:
                raise ValueError("Input must have DatetimeIndex or a 'time' column")

        lob_df = lob_df.sort_index()

        # keep last record per timestamp
        lob_df = lob_df.groupby(lob_df.index).last()

        # --- Select L2 snapshot columns in the SAME order as the input DataFrame ---
        valid_cols = []
        for c in lob_df.columns:
            if any(c.startswith(prefix) for prefix in ["bidPx_", "bidQty_", "askPx_", "askQty_"]) and c[-1].isdigit() and int(c[-1]) <= market_depth:
                valid_cols.append(c)

        lob_df = lob_df[valid_cols]

        # Convert to numpy for speed
        mat = lob_df.values.astype(np.float32)

        # ------------------------------
        # Normalize inside dataset
        # ------------------------------
        mu = mat.mean(axis=0, keepdims=True)
        sigma = mat.std(axis=0, keepdims=True) + 1e-8
        mat_norm = (mat - mu) / sigma

        # store mean/std for later denorm if needed
        self.mean = torch.tensor(mu.flatten(), dtype=torch.float32)
        self.std = torch.tensor(sigma.flatten(), dtype=torch.float32)

        # ------------------------------
        # First-order differences
        # ------------------------------
        diff = np.diff(mat_norm, axis=0)
        diff = np.concatenate([np.zeros_like(diff[:1]), diff], axis=0)

        # final augmented state S_t = [snapshot, diff]
        S = np.concatenate([mat_norm, diff], axis=1)

        # next state X_{t+1}
        X = mat_norm[1:]
        S = S[:-1]

        self.S = torch.tensor(S, dtype=torch.float32)
        self.X = torch.tensor(X, dtype=torch.float32)

        print(f"Dataset built: S={self.S.shape}, X={self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]


# ===============================================================
#                      TRAINING ENTRY POINT
# ===============================================================
import sys 
import os

if __name__ == "__main__":
    df_1 = pd.read_csv("out/data/20191001/FLEX_L2_SNAPSHOT.csv")
    df_2 = pd.read_csv("out/data/20191002/FLEX_L2_SNAPSHOT.csv")
    df_3 = pd.read_csv("out/data/20191003/FLEX_L2_SNAPSHOT.csv")
    df_4 = pd.read_csv("out/data/20191004/FLEX_L2_SNAPSHOT.csv")

    lob_df = pd.concat([df_1], ignore_index=True)

    if "time" in lob_df.columns:
        lob_df["time"] = pd.to_datetime(lob_df["time"])

    dataset = LOBGanDataset(lob_df, market_depth=MARKET_DEPTH)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=SHUFFLE_DATA)

    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN, out_dim=x_dim).to(device)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN).to(device)

    gen_path = Path(str(MODELS_DIR / "generator_") + str(MARKET_DEPTH) + "layers.pth")
    disc_path = Path(str(MODELS_DIR / "discriminator_") + str(MARKET_DEPTH) + "layers.pth")

    if "--load" in sys.argv:
        print("Loading pretrained models...")

        if not gen_path.exists() or not disc_path.exists():
            raise FileNotFoundError("Model files not found.")

        G.load_state_dict(torch.load(gen_path, map_location=device))
        D.load_state_dict(torch.load(disc_path, map_location=device))
        G.eval()
        D.eval()
        print("Pre trained model loaded.")

    generator, discriminator = train_gan(generator=G, discriminator=D, dataloader=loader, wgan=True, num_epochs=EPOCHS, z_dim=Z_DIM, 
                                            device=device, critic_steps=CRITIC_STEPS, lambda_gp=LAMBDA_GP, lr_d=LR_D, lr_g=LR_G, 
                                            market_depth=MARKET_DEPTH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    print("Models trained and saved.")
"""
    # Number of timestamps to simulate and compare to true ones
    time_index = 1000
    generated = []
    
    with torch.no_grad():
        current_s = dataset.S[-time_index].unsqueeze(0).to(device)
        first_s = current_s.values
        for _ in range(time_index):
            z = torch.randn(1, Z_DIM, device=device)
            x_next = G(z, curr_s)
            generated.append(x_next.cpu().numpy().flatten())
            curr_s = torch.cat([x_next, torch.zeros_like(x_next)], dim=1)

    # --- Keep only numeric L2 columns (exclude 'time') ---
    # TODO: THIS MUST BE REVISED / numeric_df or lob_df?
    numeric_cols = [c for c in lob_df.columns if not ('time' in c.lower() or 'date' in c.lower() or 'time_elapsed' in c.lower())]
    print(f"Numeric columns: {numeric_cols}")
    print(f"Total columns: {lob_df.columns}")
    print(f"Generated columns: {generated[0].shape}")

    df_gen = pd.DataFrame(generated, columns=numeric_cols)

    # --- Build timestamps spaced by 10 seconds ---
    if isinstance(numeric_df.index, pd.DatetimeIndex):
        last_ts = numeric_df.index[-time_index]
    elif 'time' in numeric_df.columns:
        print("Using last 'time' column for timestamp generation.")
        last_ts = pd.to_datetime(numeric_df['time'].iloc[-time_index])
    else:
        last_ts = pd.Timestamp.now()

    timestamps = numeric_df["time"].iloc[-time_index:].tolist()
    
    # Determine correct baseline state
    baseline_abs_state = raw_numeric_df.iloc[-time_index].values

    # De-standardize differences
    df_gen = df_gen * std + mean
    numeric_df[numeric_cols] = numeric_df[numeric_cols] * std + mean

    # ------------------------------
    # Inverse Log-transform
    # ------------------------------
    df_gen = np.expm1(df_gen)
    numeric_df[numeric_cols] = np.expm1(numeric_df[numeric_cols])

    # --- DE-DIFFERENTIATE USING LOCAL BASELINE ---
    # df_gen = df_gen.cumsum()
    # df_gen = df_gen + baseline_abs_state   # <-- Use correct local first state
    df_gen.insert(0, "time", timestamps) # insert as first column

    # numeric_df[numeric_cols] = numeric_df[numeric_cols].cumsum()
    # numeric_df[numeric_cols] = numeric_df[numeric_cols] + baseline_abs_state  # <-- same baseline


    df_gen.to_csv("generated_lob.csv", index=False)
    print("Saved generated_lob.csv")

    plot_real_vs_generated(numeric_df, df_gen, time_index)

    
"""




    
