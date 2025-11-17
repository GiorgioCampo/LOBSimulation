import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_edf
)

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from train_model import LOBGanDataset
from model.gan_model import Generator, Discriminator, train_gan


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------
# WGAN VALIDATION OBJECTIVE (same as your version)
# ----------------------------------------------------
def evaluate_wgan_metric(G, D, val_loader, z_dim, device, alpha=0.6):
    G.eval()
    D.eval()

    gen_adv_scores = []
    wasserstein_scores = []

    with torch.no_grad():
        for real_x, s in val_loader:
            real_x = real_x.to(device)
            s      = s.to(device)

            b = real_x.size(0)
            z = torch.randn(b, z_dim, device=device)

            fake_x = G(z, s)

            d_real = D(real_x, s).mean().item()
            d_fake = D(fake_x, s).mean().item()

            # generator quality: lower is better
            gen_adv = -d_fake
            
            # wasserstein distance: positive and decreasing as model improves
            w_dist = d_real - d_fake

            gen_adv_scores.append(gen_adv)
            wasserstein_scores.append(w_dist)

    g_mean = sum(gen_adv_scores) / len(gen_adv_scores)
    w_mean = sum(wasserstein_scores) / len(wasserstein_scores)

    # combine them
    # alpha=0.6 biases toward generator quality
    # (1-alpha)=0.4 uses wasserstein structure
    # both should decrease as quality improves
    return alpha * g_mean + (1 - alpha) * w_mean


# ----------------------------------------------------
# OPTUNA OBJECTIVE
# ----------------------------------------------------
def objective(trial):

    z_dim      = trial.suggest_int("z_dim", 4, 16)
    # hidden     = trial.suggest_int("hidden_dim", 32, 256)
    lr_g       = trial.suggest_float("lr_g", 1e-6, 1e-4, log=True)
    lr_d       = trial.suggest_float("lr_d", 1e-4, 1e-3, log=True)
    critic_k   = trial.suggest_int("critic_steps", 5, 25)
    lambda_gp  = trial.suggest_float("lambda_gp", 1.0, 12)

    # ---- Load dataset (same as your code) ----
    df_1 = pd.read_csv("out/data/20191001/FLEX_L2_SNAPSHOT.csv")
    df_2 = pd.DataFrame([]) #pd.read_csv("out/data/20191002/FLEX_L2_SNAPSHOT.csv")
    df_3 = pd.read_csv("out/data/20191003/FLEX_L2_SNAPSHOT.csv")
    df_4 = pd.DataFrame([]) #pd.read_csv("out/data/20191004/FLEX_L2_SNAPSHOT.csv")

    lob_df = pd.concat([df_1, df_2, df_3, df_4])

    non_numeric_cols = [
        c for c in lob_df.columns
        if lob_df[c].dtype == 'object' or
           'time' in c.lower() or
           'date' in c.lower()
    ]
    numeric_df = lob_df.drop(columns=non_numeric_cols, errors="ignore")

    mean = numeric_df.mean()
    std  = numeric_df.std()
    numeric_df = (numeric_df - mean) / std

    if 'time' in lob_df.columns:
        numeric_df['time'] = pd.to_datetime(lob_df['time'])

    dataset = LOBGanDataset(numeric_df, market_depth=5)

    train_size = int(len(dataset) * 0.8)
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]

    G = Generator(z_dim=z_dim, s_dim=s_dim, hidden_dim=256, out_dim=x_dim).to(device)
    D = Discriminator(x_dim=x_dim, s_dim=s_dim, hidden_dim=256).to(device)

    train_gan(
        generator=G,
        discriminator=D,
        dataloader=train_loader,
        z_dim=z_dim,
        num_epochs=8,
        device=device,
        wgan=True,
        critic_steps=critic_k,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_gp=lambda_gp
    )

    metric = evaluate_wgan_metric(G, D, val_loader, z_dim, device)

    trial.report(metric, step=0)
    return metric



# ----------------------------------------------------
# RUN + SAVE STUDY + PLOTS
# ----------------------------------------------------
def run_optuna():

    # --- Persistent SQLite backend -------
    storage = "sqlite:///optuna_lobgan.db"
    study_name = "lobgan_opt"

    # Create or reload study
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print("Loaded existing Optuna study.")
    except:
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    direction="minimize")
        print("Created new Optuna study.")

    # ----- Run optimization -----
    study.optimize(objective, n_trials=100)

    print("\nBest Trial:")
    print(study.best_trial.params)

    # ---- Save CSV for external analysis ----
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)

    # ---- Save plots ----
    os.makedirs("./out/optuna", exist_ok=True)

    plot_optimization_history(study).write_image("./out/optuna/optimization_history.pdf")
    plot_param_importances(study).write_image("./out/optuna/param_importances.pdf")
    plot_parallel_coordinate(study).write_image("./out/optuna/parallel_coordinate.pdf")
    # plot_pareto_front(study).write_image("./figs/optuna/pareto_front.pdf")
    plot_edf(study).write_image("./out/optuna/edf.pdf")

    print("Saved Optuna plots under ./out/optuna/")


if __name__ == "__main__":
    run_optuna()
