import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from train_model import MODELS_DIR, LOBGANDataset, Z_DIM, HIDDEN_D, HIDDEN_G, MARKET_DEPTH, USE_DIFFS, INCLUDE_DIFFS
from model.gan_model import Generator, Discriminator
from plots import plot_real_vs_generated_conf  # updated plot function
from pathlib import Path

# ------------------- CONFIG -------------------
DATA_FILE = "out/data/20191002/FLEX_L2_SNAPSHOT.csv"
N_OUTPUT_ROWS = 1000       # timesteps per path
N_PATHS = 50              # number of generated paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------

if __name__ == "__main__":
    # ------------------- LOAD DATA -------------------
    lob_df = pd.read_csv(DATA_FILE)
    
    # Preprocess to match LOBGANDataset
    if 'time' in lob_df.columns:
        lob_df['time'] = pd.to_datetime(lob_df['time'])
        lob_df = lob_df.set_index('time')
    lob_df = lob_df.sort_index()
    lob_df = lob_df.groupby(lob_df.index).last()

    # ------------------- NUMERIC COLUMNS -------------------
    numeric_cols = [c for c in lob_df.columns 
            if any(c.startswith(p) for p in ["bidPx_", "bidQty_", "askPx_", "askQty_"])
            and int(c.split('_')[-1]) < MARKET_DEPTH]
    
    mean = torch.tensor(lob_df[numeric_cols].mean())
    std = torch.tensor(lob_df[numeric_cols].std())

    # ------------------- DATASET -------------------
    dataset = LOBGANDataset([DATA_FILE], market_depth=MARKET_DEPTH, differentiate=USE_DIFFS, include_diffs_in_state=INCLUDE_DIFFS)
    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    # ------------------- LOAD MODELS -------------------
    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN_G, out_dim=x_dim).to(DEVICE)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN_D).to(DEVICE)

    if USE_DIFFS:
        model_name = "diffs"
    elif INCLUDE_DIFFS:
        model_name = "augmented"
    else:
        model_name = "raw"
    
    G.load_state_dict(torch.load(MODELS_DIR / f"generator_{model_name}_best_qty.pth"))
    D.load_state_dict(torch.load(MODELS_DIR / f"discriminator_{model_name}_best_qty.pth"))
    G.eval()
    D.eval()

    # ------------------- PREPARE TIMESTAMPS & RANDOM START -------------------
    full_timestamps = lob_df.index
    
    # Relaxed context logic: start anywhere, cap generation if needed
    max_idx = len(dataset.X) - 1
    min_idx = 0 
    
    start_idx = 100 #np.random.randint(min_idx, max_idx)
    
    # Calculate how many steps we can generate
    n_steps = min(N_OUTPUT_ROWS, len(dataset.X) - 1 - start_idx)
    
    if n_steps <= 0:
        # Fallback if we picked the very last index
        start_idx = max_idx - 10
        n_steps = min(N_OUTPUT_ROWS, len(dataset.X) - 1 - start_idx)

    print(f"Start Index: {start_idx}, Generating {n_steps} steps (Requested: {N_OUTPUT_ROWS})")

    # prev_x is the starting state for generation
    # dataset.X[start_idx] corresponds to mat[start_idx+1]
    prev_x = dataset.X[start_idx].unsqueeze(0).to(DEVICE)
    
    # Timestamps for the generated sequence
    # dataset.X[start_idx] is at full_timestamps[start_idx+1]
    # Generated starts at full_timestamps[start_idx+2]
    timestamps = full_timestamps[start_idx + 2 : start_idx + 2 + n_steps].tolist()

    # ------------------- GENERATE N PATHS -------------------
    df_gen_list = []
    csv_dfs = []
    all_generated_arrays = []  # Store raw arrays for averaging

    with torch.no_grad():
        for path_idx in tqdm(range(N_PATHS), desc="Generating paths..."):
            # initial snapshot (normalized)
            curr_x = prev_x.clone()

            # initial state S0 = [snapshot, diff=0]
            zero_diff = torch.zeros_like(curr_x)
            current_s = torch.cat([curr_x, zero_diff], dim=1)

            generated = []

            for _ in range(n_steps):
                z = torch.randn(1, Z_DIM, device=DEVICE)
                x_next = G(z, current_s)

                # -------- FIRST-ORDER DIFF UPDATE ----------
                diff_next = x_next - curr_x
                current_s = torch.cat([x_next, diff_next], dim=1)
                curr_x = x_next

                # denorm
                x_denorm = torch.round(x_next.cpu() * std + mean, decimals=2)
                
                generated.append(x_denorm.numpy().flatten())
            
            gen_array = np.array(generated)
            all_generated_arrays.append(gen_array)
            
            df_gen = pd.DataFrame(gen_array, columns=numeric_cols)
            
            # Store for CSV (without time column)
            csv_dfs.append(df_gen.copy())
            
            # Store for Plot (with time column)
            df_gen["time"] = timestamps
            df_gen_list.append(df_gen)

    # Save ALL paths to CSV for metrics calculation
    full_gen_df = pd.concat(csv_dfs, ignore_index=True)
    full_gen_df.to_csv("generated_lob.csv", index=False)
    print(f"Saved {len(full_gen_df)} rows (from {N_PATHS} paths) to generated_lob.csv")

    # ------------------- PREPARE REAL DATA FOR PLOT -------------------
    # "Print still the hole real" -> Show entire history up to end of generation
    # dataset.S is mat[:-1].
    # We want to end at start_idx + 1 + n_steps (corresponding to end of generation)
    # But actually, let's just show everything up to that point.
    
    slice_end = start_idx + 1 + n_steps
    
    # Ensure we don't go out of bounds of S
    slice_end = min(slice_end, len(dataset.S))
    
    real_slice = dataset.S[:slice_end, : MARKET_DEPTH * 4]
    real_df = pd.DataFrame(real_slice, columns=numeric_cols)
    
    # Timestamps for real_df
    real_df["time"] = full_timestamps[:slice_end].tolist()
    real_df = real_df.set_index("time")
    real_df = real_df[numeric_cols] * std.cpu().numpy() + mean.cpu().numpy()

    # ------------------- PLOT -------------------
    print(f"Saved averaged LOB (outliers removed) to generated_lob_averaged.csv")
    for column in numeric_cols:
        plot_real_vs_generated_conf(
            real_df, df_gen_list, 
            time_index=n_steps, 
            save=True, 
            column=column
        )
    print(f"Generated {N_PATHS} paths, each with {n_steps} timesteps → plot saved.")
