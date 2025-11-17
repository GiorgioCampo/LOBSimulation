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

Z_DIM = 8               # Noise dimension. 25% - 50% of total dimensions
HIDDEN = 256
BATCH = 16
EPOCHS = 100
CRITIC_STEPS = 15
MARKET_DEPTH = 5
SHUFFLE_DATA = True
LAMBDA_GP = 8          # Increase to penalize discriminator more

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

        # ------------------------------
        # Select numeric columns (Px/Qty)
        # ------------------------------
        cols = []
        for side in ["bidPx", "askPx", "bidQty", "askQty"]:
            for i in range(market_depth):
                c = f"{side}_{i}"
                if c in lob_df.columns:
                    cols.append(c)

        df = lob_df[cols].astype(float).dropna()

        # Convert to numpy for speed
        mat = df.values.astype(np.float32)

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
if __name__ == "__main__":
    df_1 = pd.read_csv("out/data/20191001/FLEX_L2_SNAPSHOT.csv")
    # df_2 = pd.read_csv("out/data/20191002/FLEX_L2_SNAPSHOT.csv")
    # df_3 = pd.read_csv("out/data/20191003/FLEX_L2_SNAPSHOT.csv")
    # df_4 = pd.read_csv("out/data/20191004/FLEX_L2_SNAPSHOT.csv")

    # Concatenate all days
    lob_df = pd.concat([df_1]) 

    dataset = LOBGanDataset(lob_df, market_depth=MARKET_DEPTH)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=SHUFFLE_DATA)

    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN, out_dim=x_dim)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN)

    generator, discriminator = train_gan(
        generator=G,
        discriminator=D,
        dataloader=loader,
        wgan=True,
        num_epochs=EPOCHS,
        z_dim=Z_DIM,
        device=device,
        critic_steps=CRITIC_STEPS,
        lambda_gp=LAMBDA_GP,
    )

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), MODELS_DIR / "generator.pt")
    torch.save(discriminator.state_dict(), MODELS_DIR / "discriminator.pt")

    # ===============================================================
    # Generate sequence
    # ===============================================================
    generated = []
    with torch.no_grad():
        curr_s = dataset.S[-1].unsqueeze(0).to(device)
        for _ in range(1500):
            z = torch.randn(1, Z_DIM, device=device)
            x_next = G(z, curr_s)
            generated.append(x_next.cpu().numpy().flatten())
            curr_s = torch.cat([x_next, torch.zeros_like(x_next)], dim=1)

    df_gen = pd.DataFrame(generated)

    # timestamps
    if isinstance(lob_df.index, pd.DatetimeIndex):
        last_ts = lob_df.index[-1]
    else:
        last_ts = pd.Timestamp.now()

    timestamps = [last_ts + pd.Timedelta(seconds=10 * i) for i in range(1, 1501)]
    timestamps = [ts.strftime("%H:%M:%S.%f")[:-3] for ts in timestamps]
    df_gen.insert(0, "time", timestamps)

    df_gen.to_csv("generated_lob.csv", index=False)
    print("Saved generated_lob.csv")

    plot_real_vs_generated(lob_df, df_gen)
