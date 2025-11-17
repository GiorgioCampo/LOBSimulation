# dataset_lob_gan.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model.gan_model import train_gan, Generator, Discriminator
from plots import plot_real_vs_generated
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"

# Use OPTUNA for hyperparameter tuning: https://optuna.org/
Z_DIM = 6               # Noise dimension. 25% - 50% of total dimensions
HIDDEN = 256
BATCH = 32
EPOCHS = 100
CRITIC_STEPS = 13
MARKET_DEPTH = 5
SHUFFLE_DATA = True
LAMBDA_GP = 8         # Increase to penalize discriminator more
LR_D = 2e-4
LR_G = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LOBGanDataset(Dataset):
    """
    Prepares (X_{t+Δt}, S_t) pairs for the GAN model from an L2_SNAPSHOT DataFrame.
    - Keeps only the last record per timestamp
    - Uses only L2 snapshot columns (bidPx/askPx/bidQty/askQty)
    - Returns consecutive pairs of snapshots for sequence learning
    """

    def __init__(self, lob_df: pd.DataFrame, market_depth: int = 5):
        super().__init__()

        # --- Ensure timestamps are datetime and sorted ---
        if not isinstance(lob_df.index, pd.DatetimeIndex):
            if "time" in lob_df.columns:
                lob_df["time"] = pd.to_datetime(lob_df["time"])
                lob_df = lob_df.set_index("time")
            else:
                raise ValueError("L2_SNAPSHOT DataFrame must have a DatetimeIndex or 'time' column.")

        lob_df = lob_df.sort_index()

        # --- Keep only the last record for duplicate timestamps ---
        lob_df = lob_df.groupby(lob_df.index).last()

        # --- Select L2 snapshot columns in the SAME order as the input DataFrame ---
        valid_cols = []
        for c in lob_df.columns:
            if any(c.startswith(prefix) for prefix in ["bidPx_", "bidQty_", "askPx_", "askQty_"]):
                valid_cols.append(c)

        lob_df = lob_df[valid_cols]

        # --- Build (S_t, X_{t+Δt}) pairs ---
        data = lob_df.values
        self.S = torch.tensor(data[:-1], dtype=torch.float32) # S_t = X_{t}
        self.X = torch.tensor(data[1:], dtype=torch.float32)  # X_t = X_{t+Δt}

        print(f"X: {self.X.shape}, S: {self.S.shape}")

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]



if __name__ == "__main__":
    # List of CSV files
    files = [
        "out/data/20191001/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191002/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191003/FLEX_L2_SNAPSHOT.csv",
        "out/data/20191004/FLEX_L2_SNAPSHOT.csv"
    ]

    dfs = []

    for file in files:
        df = pd.read_csv(file)

        # Select only numeric columns for standardization
        numeric_cols = df.select_dtypes(include='number').columns
        numeric_df = df[numeric_cols]

        # Standardize numeric columns
        numeric_df = (numeric_df - numeric_df.mean()) / numeric_df.std()

        # Reattach non-numeric columns (like time or date)
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        for c in non_numeric_cols:
            numeric_df[c] = df[c]

        dfs.append(numeric_df)

    # Concatenate all days
    lob_df = pd.concat(dfs, ignore_index=True)

    # Optional: remove initial rows to limit low liquidity effects
    # lob_df = lob_df.iloc[500:]

    # Final check: convert 'time' column to datetime if exists
    if 'time' in lob_df.columns:
        lob_df['time'] = pd.to_datetime(lob_df['time'])
    dataset = LOBGanDataset(lob_df, market_depth=MARKET_DEPTH)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=SHUFFLE_DATA)

    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    for X, S in loader:
        print("Next-state X:", X.shape, "Condition S:", S.shape)
        break

    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN, out_dim=x_dim)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN)

    generator, discriminator = train_gan(generator=G, discriminator=D, dataloader=loader, wgan=True, num_epochs=EPOCHS,
              z_dim=Z_DIM, device=device, critic_steps=CRITIC_STEPS, lambda_gp=LAMBDA_GP, lr_d=LR_D, lr_g=LR_G)

    # Save trained models
    torch.save(generator.state_dict(), MODELS_DIR / "generator.pt")
    torch.save(discriminator.state_dict(), MODELS_DIR / "discriminator.pt")

""""
    # Number of timestamps to simulate and compare to true ones
    time_index = 1000
    generated = []
    
    with torch.no_grad():
        current_s = dataset.S[-time_index].unsqueeze(0).to(device)
        first_s = current_s.values
        for _ in range(time_index):
            z = torch.randn(1, Z_DIM, device=device)
            x_next = G(z, current_s)
            generated.append(x_next.cpu().numpy().flatten())
            current_s = x_next  # feed next state as new condition

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

    # --- DE-DIFFERENTIATE USING LOCAL BASELINE ---
    # df_gen = df_gen.cumsum()
    # df_gen = df_gen + baseline_abs_state   # <-- Use correct local first state
    df_gen.insert(0, "time", timestamps) # insert as first column

    # numeric_df[numeric_cols] = numeric_df[numeric_cols].cumsum()
    # numeric_df[numeric_cols] = numeric_df[numeric_cols] + baseline_abs_state  # <-- same baseline


    # --- Save ---
    df_gen.to_csv("generated_lob.csv", index=False)
    print(f"Generated {len(df_gen)} rows → saved to generated_lob.csv")

    plot_real_vs_generated(numeric_df, df_gen, time_index)

    
"""




    
