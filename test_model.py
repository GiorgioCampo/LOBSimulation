import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
<<<<<<< HEAD
from train_model import MODELS_DIR, LOBGanDataset, Z_DIM, HIDDEN, MARKET_DEPTH
from model.gan_model import Generator, Discriminator
from plots import plot_real_vs_generated_conf  # updated plot function
from pathlib import Path

# ------------------- CONFIG -------------------
DATA_FILE = "out/data/20191002/FLEX_L2_SNAPSHOT.csv"
N_OUTPUT_ROWS = 300       # timesteps per path
N_PATHS = 100             # number of generated paths
=======
from train_model import MODELS_DIR, LOBGANDataset, Z_DIM, HIDDEN, MARKET_DEPTH
from model.gan_model import Generator, Discriminator
from plots import plot_real_vs_generated_conf

# ------------------- CONFIG -------------------
DATA_FILE = "out/data/20191002/FLEX_L2_SNAPSHOT.csv"
N_OUTPUT_ROWS = 300
N_PATHS = 100
>>>>>>> refs/remotes/origin/master
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DIFFS = False
INCLUDE_DIFFS = False
# ---------------------------------------------

if __name__ == "__main__":
    # ------------------- COMPUTE STATS BEFORE DATASET -------------------
    # Load raw data to extract mean/std for denormalization
    lob_df_raw = pd.read_csv(DATA_FILE)
    
    # Get numeric columns
    numeric_cols = lob_df_raw.select_dtypes(include='number').columns.tolist()
    
    # Compute stats based on whether we're using diffs
    if USE_DIFFS:
        # Compute mean/std of differences
        lob_diffs = lob_df_raw[numeric_cols].diff().fillna(0)
        mean = lob_diffs.mean()
        std = lob_diffs.std()
    else:
        # Compute mean/std of raw levels
        mean = lob_df_raw[numeric_cols].mean()
        std = lob_df_raw[numeric_cols].std()
    
    # Extract timestamps
    if 'time' in lob_df_raw.columns:
        timestamps = pd.to_datetime(lob_df_raw['time']).unique().tolist()
    else:
        timestamps = [pd.Timestamp.now()] * len(lob_df_raw)
    
    # ------------------- DATASET -------------------
    # Let the dataset handle all preprocessing
    dataset = LOBGANDataset(
        file_paths=[DATA_FILE],
        market_depth=MARKET_DEPTH,
        differentiate=USE_DIFFS,
        include_diffs_in_state=INCLUDE_DIFFS
    )
    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]
    
    # Get L2 column names from dataset
    l2_cols = [c for c in lob_df_raw.columns 
               if any(c.startswith(p) for p in ["bidPx_", "bidQty_", "askPx_", "askQty_"])
               and int(c.split('_')[-1]) < MARKET_DEPTH]
    
    # ------------------- LOAD MODELS -------------------
    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN, out_dim=x_dim).to(DEVICE)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN).to(DEVICE)
<<<<<<< HEAD

    G.load_state_dict(torch.load(Path(str(MODELS_DIR / "generator_") + str(MARKET_DEPTH) + "layers.pth"), map_location=DEVICE))
    D.load_state_dict(torch.load(Path(str(MODELS_DIR / "discriminator_") + str(MARKET_DEPTH) + "layers.pth"), map_location=DEVICE))
=======
    
    if USE_DIFFS:
        model_name = "diffs"
    elif INCLUDE_DIFFS:
        model_name = "augmented"
    else:
        model_name = "raw"
    
    G.load_state_dict(torch.load(MODELS_DIR / f"generator_{model_name}.pth", map_location=DEVICE))
    D.load_state_dict(torch.load(MODELS_DIR / f"discriminator_{model_name}.pth", map_location=DEVICE))
>>>>>>> refs/remotes/origin/master
    G.eval()
    D.eval()
    
    # ------------------- GENERATE N PATHS -------------------
    df_gen_list = []
    
    # Prepare for denormalization (only L2 columns)
    mean_l2 = mean[l2_cols].values
    std_l2 = std[l2_cols].values
    
    # Get initial state from raw data (NOT differentiated)
    initial_state_raw = lob_df_raw[l2_cols].iloc[0].values

    initial_state_raw_gen = lob_df_raw.groupby("time").last()[l2_cols].iloc[-N_OUTPUT_ROWS]
    
    with torch.no_grad():
        for path_idx in tqdm(range(N_PATHS), desc="Generating paths"):
            current_s = dataset.S[-N_OUTPUT_ROWS].unsqueeze(0).to(DEVICE)
            
            generated = []
            current_level = initial_state_raw_gen.copy()  # Start from actual price levels
            
            for _ in range(N_OUTPUT_ROWS):
                z = torch.randn(1, Z_DIM, device=DEVICE)
                x_next = G(z, current_s)
                
                # Denormalize
                x_next_denorm = (x_next.cpu().numpy().flatten() * std_l2) + mean_l2
                
                if USE_DIFFS:
                    # Model outputs diffs -> integrate to get levels
                    current_level = current_level + x_next_denorm
                    generated.append(current_level.copy())
                else:
                    # Model outputs levels directly
                    generated.append(x_next_denorm)
                    current_level = x_next_denorm
                
                current_s = x_next
            
            df_gen = pd.DataFrame(generated, columns=l2_cols)
            df_gen["time"] = timestamps[-N_OUTPUT_ROWS:]
            df_gen_list.append(df_gen)
            
            if path_idx == 0:
                df_gen.to_csv("generated_lob.csv", index=False)
    
    # ------------------- PLOT -------------------
    plot_columns = [
        "bidPx_0", "bidPx_1", "bidPx_2", #"bidPx_3", "bidPx_4",
        "askPx_0", "askPx_1", "askPx_2", #"askPx_3", "askPx_4"
    ]

    true_df = lob_df_raw.groupby("time").last()[plot_columns][1:]
    true_df["time"] = timestamps[1:len(true_df)+1]  # S is offset by 1

    for column in plot_columns:
        plot_real_vs_generated_conf(
            true_df, df_gen_list, 
            time_index=N_OUTPUT_ROWS, 
            save=True, 
            column=column
        )
    
    print(f"Generated {N_PATHS} paths, each with {N_OUTPUT_ROWS} timesteps → plots saved.")