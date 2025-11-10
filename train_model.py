# dataset_lob_gan.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model.gan_model import train_gan, Generator, Discriminator
from plots import plot_real_vs_generated
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"

Z_DIM = 8               # Noise dimension. 25% - 50% of total dimensions
HIDDEN = 256
BATCH = 16
EPOCHS = 100
CRITIC_STEPS = 5
MARKET_DEPTH = 5
SHUFFLE_DATA = True
LAMBDA_GP = 15          # Increase to penalize discriminator more

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

        # --- Select L2 snapshot columns ---
        cols = []
        for side in ["bidPx", "askPx", "bidQty", "askQty"]:
            for i in range(market_depth):
                c = f"{side}_{i}"
                if c in lob_df.columns:
                    cols.append(c)
        lob_df = lob_df[cols].astype(float).dropna()

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
    df_1 = pd.read_csv("out/data/20191001/FLEX_L2_SNAPSHOT.csv")
    df_2 = pd.read_csv("out/data/20191002/FLEX_L2_SNAPSHOT.csv")
    df_3 = pd.read_csv("out/data/20191003/FLEX_L2_SNAPSHOT.csv")
    df_4 = pd.read_csv("out/data/20191004/FLEX_L2_SNAPSHOT.csv")


    # Concatenate all days
    lob_df = pd.concat([df_1]) # df_2, df_3, df_4

    # Drop non-numeric columns for normalization
    non_numeric_cols = [c for c in lob_df.columns if lob_df[c].dtype == 'object' or 'time' in c.lower() or 'date' in c.lower()]
    numeric_df = lob_df.drop(columns=non_numeric_cols, errors="ignore")

    # Normalize
    mean = numeric_df.mean()
    std = numeric_df.std()
    numeric_df = (numeric_df - mean) / std

    # Reattach time column
    if 'time' in lob_df.columns:
        numeric_df['time'] = pd.to_datetime(lob_df['time'])

    lob_df = numeric_df

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
              z_dim=Z_DIM, device=device, critic_steps=CRITIC_STEPS, lambda_gp=LAMBDA_GP)

    # Save trained models
    torch.save(generator.state_dict(), MODELS_DIR / "generator.pt")
    torch.save(discriminator.state_dict(), MODELS_DIR / "discriminator.pt")

    generated = []
    with torch.no_grad():
        current_s = dataset.S[-1].unsqueeze(0).to(device)
        for _ in range(1500):
            z = torch.randn(1, Z_DIM, device=device)
            x_next = G(z, current_s)
            generated.append(x_next.cpu().numpy().flatten())
            current_s = x_next  # feed next state as new condition

    # --- Keep only numeric L2 columns (exclude 'time') ---
    # TODO: THIS MUST BE REVISED
    numeric_cols = [c for c in lob_df.columns if not ('time' in c.lower() or 'date' in c.lower() or 'time_elapsed' in c.lower())]
    print(f"Numeric columns: {numeric_cols}")
    print(f"Total columns: {lob_df.columns}")
    print(f"Generated columns: {generated[0].shape}")

    df_gen = pd.DataFrame(generated, columns=numeric_cols)

    # --- Build timestamps spaced by 10 seconds ---
    if isinstance(lob_df.index, pd.DatetimeIndex):
        last_ts = lob_df.index[-1]
    elif 'time' in lob_df.columns:
        print("Using last 'time' column for timestamp generation.")
        last_ts = pd.to_datetime(lob_df['time'].iloc[-1])
    else:
        last_ts = pd.Timestamp.now()

    timestamps = [last_ts + pd.Timedelta(seconds=10 * i) for i in range(1, 1500 + 1)]
    
    # Output in format hh:mm:ss:ms
    timestamps = [ts.strftime("%H:%M:%S.%f")[:-3] for ts in timestamps]
    df_gen.insert(0, "time", timestamps)  # insert as first column

    # --- Save ---
    df_gen.to_csv("generated_lob.csv", index=False)
    print(f"Generated {len(df_gen)} rows → saved to generated_lob.csv")

    plot_real_vs_generated(lob_df, df_gen)

    





    
