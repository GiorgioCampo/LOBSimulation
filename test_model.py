import pandas as pd
import torch
from plots import plot_real_vs_generated
from train_model import MODELS_DIR, LOBGanDataset, Z_DIM, HIDDEN
from model.gan_model import Generator, Discriminator

N_OUTPUT_ROWS = 750

if __name__ == '__main__':
    lob_df = pd.read_csv("out/data/20191001/FLEX_L2_SNAPSHOT.csv")
    dataset = LOBGanDataset(lob_df)

    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    G = Generator(z_dim=Z_DIM, s_dim=s_dim, hidden_dim=HIDDEN, out_dim=x_dim)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=HIDDEN)
    G.to(device)
    D.to(device)
    G.load_state_dict(torch.load(MODELS_DIR / "generator.pth"))
    D.load_state_dict(torch.load(MODELS_DIR / "discriminator.pth"))
    G.eval()
    D.eval()

    generated = []
    with torch.no_grad():
        last_s = dataset.S[-1].unsqueeze(0).to(device)
        last_snapshot = dataset.X[-1].unsqueeze(0).to(device)

        for _ in range(N_OUTPUT_ROWS):
            z = torch.randn(1, Z_DIM, device=device)
            x_next = G(z, last_s)

            # ----------- FIX: DENORMALIZE -----------
            x_denorm = x_next * dataset.std.to(device) + dataset.mean.to(device)
            generated.append(x_denorm.cpu().numpy().flatten())

            # compute Δ for next S
            diff_next = x_next - last_snapshot

            last_s = torch.cat([x_next, diff_next], dim=1)
            last_snapshot = x_next

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

    timestamps = [last_ts + pd.Timedelta(seconds=10 * i) for i in range(1, N_OUTPUT_ROWS + 1)]
    
    # Output in format hh:mm:ss:ms
    timestamps = [ts.strftime("%H:%M:%S.%f")[:-3] for ts in timestamps]
    df_gen.insert(0, "time", timestamps)  # insert as first column

    # --- Save ---
    df_gen.to_csv("generated_lob.csv", index=False)
    print(f"Generated {len(df_gen)} rows → saved to generated_lob.csv")

    plot_real_vs_generated(lob_df, df_gen)