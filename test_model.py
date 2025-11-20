import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from train_model import MODELS_DIR, LOBGanDataset, Z_DIM, HIDDEN, MARKET_DEPTH
from model.gan_model import Generator, Discriminator
from plots import plot_real_vs_generated_conf  # updated plot function
from pathlib import Path

# ------------------- CONFIG -------------------
DATA_FILE = "out/data/20191002/FLEX_L2_SNAPSHOT.csv"
N_OUTPUT_ROWS = 300       # timesteps per path
N_PATHS = 100             # number of generated paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------

if __name__ == "__main__":
    # ------------------- LOAD DATA -------------------
    lob_df = pd.read_csv(DATA_FILE)

    # Drop non-numeric columns
    non_numeric_cols = [c for c in lob_df.columns if lob_df[c].dtype == 'object' 
                        or 'time' in c.lower() or 'date' in c.lower()]
    numeric_df = lob_df.drop(columns=non_numeric_cols, errors="ignore")

    # Save mean/std for de-normalization
    mean = numeric_df.mean()
    std = numeric_df.std()
    numeric_df = (numeric_df - mean) / std

    # Reattach time column if exists
    if 'time' in lob_df.columns:
        numeric_df['time'] = pd.to_datetime(lob_df['time'])
    lob_df = numeric_df

    # ------------------- DATASET -------------------
    dataset = LOBGanDataset(lob_df)
    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    # ------------------- LOAD MODELS -------------------
    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN, out_dim=x_dim).to(DEVICE)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN).to(DEVICE)

    G.load_state_dict(torch.load(Path(str(MODELS_DIR / "generator_") + str(MARKET_DEPTH) + "layers.pth"), map_location=DEVICE))
    D.load_state_dict(torch.load(Path(str(MODELS_DIR / "discriminator_") + str(MARKET_DEPTH) + "layers.pth"), map_location=DEVICE))
    G.eval()
    D.eval()

    # ------------------- PREPARE TIMESTAMPS -------------------
    if isinstance(lob_df.index, pd.DatetimeIndex):
        timestamps = lob_df.index[-N_OUTPUT_ROWS:].tolist()
    elif 'time' in lob_df.columns:
        timestamps = pd.to_datetime(lob_df['time'].iloc[-N_OUTPUT_ROWS:]).tolist()
    else:
        timestamps = [pd.Timestamp.now()] * N_OUTPUT_ROWS

    # ------------------- NUMERIC COLUMNS -------------------
    numeric_cols = [c for c in lob_df.columns if not ('time' in c.lower() or 'date' in c.lower() 
                                                      or 'time_elapsed' in c.lower())]

    # ------------------- GENERATE N PATHS -------------------
    df_gen_list = []

    with torch.no_grad():
        for path_idx in tqdm(range(N_PATHS), desc="Generating paths..."):
            current_s = dataset.S[-N_OUTPUT_ROWS].unsqueeze(0).to(DEVICE)

            generated = []
            for _ in range(N_OUTPUT_ROWS):
                z = torch.randn(1, Z_DIM, device=DEVICE)
                x_next = G(z, current_s)
                # de-normalize immediately
                x_next_denorm = x_next.cpu() * torch.tensor(std.values, dtype=torch.float32) \
                                + torch.tensor(mean.values, dtype=torch.float32)
                generated.append(x_next_denorm.numpy().flatten())
                current_s = x_next

            df_gen = pd.DataFrame(generated, columns=numeric_cols)
            if path_idx == 0:
                # Save first path as csv
                df_gen.to_csv("generated_lob.csv", index=False)
            df_gen["time"] = timestamps
            df_gen_list.append(df_gen)

    lob_df[numeric_cols] = lob_df[numeric_cols] * std + mean

    # ------------------- PLOT -------------------
    plot_real_vs_generated_conf(lob_df, df_gen_list, time_index=N_OUTPUT_ROWS, save=True)
    print(f"Generated {N_PATHS} paths, each with {N_OUTPUT_ROWS} timesteps → plot saved.")
