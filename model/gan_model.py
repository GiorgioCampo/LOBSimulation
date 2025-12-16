import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm as sn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from utils import _get_next_state
import metrics_lib.metrics as m

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "out/models/"
DEBUG = True

MARKET_DEPTH = 3
# Random seed
# torch.manual_seed(156)


# -----------------------
# Simple dataset stub
# -----------------------
class SimpleOrderBookDataset(Dataset):
    """Placeholder dataset. Replace with real (X_{t+Δt}, S_t) pairs."""
    def __init__(self, n_samples=10000, x_dim=32, s_dim=16):
        super().__init__()
        self.X = torch.randn(n_samples, x_dim)    # target next-state samples (real order-book states)
        self.S = torch.randn(n_samples, s_dim)    # conditioning state
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]

# -----------------------
# Generator
# -----------------------
class Generator(nn.Module):
    """
    Generator implements:
      - Two separate branches that process Z_t (noise) and S_t (condition) separately.
      - Each branch has two fully-connected layers (h1_i and h2_i).
      - The outputs of the branches (h2_1 and h2_2) are concatenated.
      - Two final fully-connected layers map the concatenated vector to the generated X_{t+Δt}.
    Activations:
      - ReLU for hidden layers
      - Linear for output (since queue sizes can be positive or negative)
    """
    def __init__(self, z_dim=32, s_dim=16, hidden_dim=64, out_dim=32):
        super().__init__()
        # Z branch (h1_z, h2_z)
        self.z_fc1 = nn.Linear(z_dim, hidden_dim)
        self.z_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # S branch (h1_s, h2_s)
        self.s_fc1 = nn.Linear(s_dim, hidden_dim)
        self.s_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # After concatenation -> two final layers (h_final1, h_final2)
        self.final_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_fc2 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.ReLU()
        # output is linear (no activation), per your description

    def forward(self, z, s):
        # z: (batch, z_dim), s: (batch, s_dim)
        z = self.activation(self.z_fc1(z))
        h2_z = self.activation(self.z_fc2(z))      # h2,1

        s = self.activation(self.s_fc1(s))
        h2_s = self.activation(self.s_fc2(s))      # h2,2

        # Concatenate h2_z and h2_s
        h_cat = torch.cat([h2_z, h2_s], dim=1)

        h = self.activation(self.final_fc1(h_cat))
        out = self.final_fc2(h)   # linear output: can be positive or negative
        # out = torch.nn.functional.softplus(out)
        return out

# -----------------------
# Discriminator / Critic
# -----------------------
class Discriminator(nn.Module):
    """
    Discriminator receives (X_{t+Δt}, S_t) and outputs a scalar score:
      - For vanilla GAN: output fed into BCEWithLogitsLoss (so we keep a final linear)
      - For WGAN: output is a real-valued 'critic' score (no sigmoid).
    Architecture:
      - We'll mirror the '6 fully-connected layers' theme by implementing two branch layers
        for X and S separately, then concatenating and using two more layers, and a final output.
      - This keeps the architecture symmetric to the generator's design philosophy.
    """
    def __init__(self, x_dim=32, s_dim=16, hidden_dim=64):
        super().__init__()

        # X branch (process candidate next-state)
        self.x_fc1 = nn.Linear(x_dim, hidden_dim)
        self.x_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # S branch (process conditioning state)
        self.s_fc1 = nn.Linear(s_dim, hidden_dim)
        self.s_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # After concatenation -> two more layers + output
        self.final_fc1 = nn.Linear(hidden_dim * 2,hidden_dim)
        self.final_fc2 = nn.Linear(hidden_dim, 1)  # single scalar output

        self.activation = nn.ReLU()

    def forward(self, x, s):
        x = self.activation(self.x_fc1(x))
        x = self.activation(self.x_fc2(x))

        s = self.activation(self.s_fc1(s))
        s = self.activation(self.s_fc2(s))

        h = torch.cat([x, s], dim=1)
        h = self.activation(self.final_fc1(h))
        out = self.final_fc2(h)  # scalar

        return out.squeeze(1)  # shape (batch,)   
        

# -----------------------
# Training skeleton
# -----------------------
def train_gan(
    *,
    generator,
    discriminator,
    train_dataloader,
    val_dataloader,
    z_dim=32,
    num_epochs=50,
    device='cpu',
    wgan=False,
    critic_steps_initial=5,
    critic_steps_final=1,
    gamma=0.1,
    lr_g=1e-4,
    lr_d=1e-4,
    lambda_gp=10,
    save_model=False,
    save_every=10,
    model_name=""
):
    """
    Training skeleton for vanilla GAN or WGAN.
    - If wgan=True: use WGAN critic updates (no sigmoid), RMSprop or Adam can be used.
    - Gradient clipping is applied to discriminator parameters (as requested).
    """

    generator.to(device)
    discriminator.to(device)

    # Choice of optimizers:
    # - commonly: Adam for G, Adam or RMSprop for D (WGAN recommends RMSprop originally)
    # if wgan:
    #     # WGAN commonly used RMSprop (original paper), but Adam also sometimes used
    #     opt_d = optim.RMSprop(discriminator.parameters(), lr=lr_d)
    #     opt_g = optim.RMSprop(generator.parameters(), lr=lr_g)
    # else:
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))

    # Loss for vanilla GAN
    #bce_loss = nn.BCEWithLogitsLoss()

    best_w_dist = None
    g_loss_history = []
    d_loss_history = []
    w_dist_history = []
    frob_level_history = []
    frob_diff_history = []
    price_frob_history = []
    mean_dev_history = []
    var_dev_history = []
    


    # -----------------------
    # Helper to compute Frobenius metric
    # -----------------------
    def compute_metrics(generator, dataloader, device, z_dim):
        """
        Computes evaluation metrics for the generator:
        - Frobenius norm on level correlations
        - Frobenius norm on difference correlations
        - Mean deviation
        - Variance deviation

        Assumes dataloader is temporally ordered.
        """

        generator.eval()

        real_states = []
        fake_states = []

        mid_changes_real = []
        mid_changes_fake = []

        best_bids_real = []
        best_bids_fake = []

        best_asks_real = []
        best_asks_fake = []

        with torch.no_grad():
            for X, S in dataloader:
                X = X.to(device)        # (B, T)
                S = S.to(device)        # already centered
                B = X.size(0)

                z = torch.randn(B, z_dim, device=device)

                # Generate fake next state
                X_fake = generator(z, S)   # (B, T)

                # =========================
                # Mid-price changes (per sample)
                # =========================
                # Count negatives per row
                neg_real = (X < 0).sum(dim=1)
                neg_fake = (X_fake < 0).sum(dim=1)

                # Mid-price change in ticks
                # print((neg_real - 2 * MARKET_DEPTH))
                mid_changes_real.extend((neg_real - 2 * MARKET_DEPTH).cpu().numpy())
                mid_changes_fake.extend((neg_fake - 2 * MARKET_DEPTH).cpu().numpy())

                # =========================
                # Center states
                # =========================
                X_real_c = _get_next_state(X)       # (B, 2D)
                X_fake_c = _get_next_state(X_fake)  # (B, 2D)

                # =========================
                # Best bid / ask (per sample)
                # =========================
                best_bids_real.extend(X_real_c[:, MARKET_DEPTH - 1].cpu().numpy())
                best_bids_fake.extend(X_fake_c[:, MARKET_DEPTH - 1].cpu().numpy())

                # Best ask = min (closest to zero)
                best_asks_real.extend(X_real_c[:, MARKET_DEPTH].cpu().numpy())
                best_asks_fake.extend(X_fake_c[:, MARKET_DEPTH].cpu().numpy())

        # =========================
        # Store centered states
        # =========================
        real_states.append(X_real_c.cpu())
        fake_states.append(X_fake_c.cpu())


        # Concatenate in temporal order
        real_X = torch.cat(real_states, dim=0).numpy()
        fake_X = torch.cat(fake_states, dim=0).numpy()

        # =========================
        # Temporal alignment
        # =========================
        # X_t and X_{t+Δt}
        real_X_t  = real_X[:-1]
        real_X_t1 = real_X[1:]

        fake_X_t  = fake_X[:-1]
        fake_X_t1 = fake_X[1:]

        # =========================
        # Metrics
        # =========================

        metrics = {}

        # 1. Level correlation Frobenius
        metrics["frob_corr_level"] = m.compute_frobenius_correlation(
            real_X_t1, fake_X_t1
        )

        # 2. Difference correlation Frobenius
        metrics["frob_corr_diff"] = m.compute_frobenius_correlation(
            real_X_t1 - real_X_t,
            fake_X_t1 - fake_X_t
        )

        # 3. Mean / Variance deviation
        mean_dev, var_dev = m.mean_var_deviation(
            real_X_t1, fake_X_t1
        )
        metrics["mean_dev"] = mean_dev
        metrics["var_dev"] = var_dev

        # 4. Conditional Price Changes Matrix Frobenius
        m_real = m.compute_midprice_direction_matrix(mid_changes_real, best_bids_real, best_asks_real)
        m_fake = m.compute_midprice_direction_matrix(mid_changes_fake, best_bids_fake, best_asks_fake)

        metrics["frob_price"] = m.compute_price_frobenius(m_real, m_fake)

        generator.train()
        return metrics


    # Initialize best metric
    best_frobenius = float('inf')

    for epoch in range(num_epochs):

        critic_steps = int(max(critic_steps_final, critic_steps_initial * gamma ** (epoch)))

        d_losses, g_losses, gps = [], [], []
        real_scores, fake_scores = [], []
        gnorms_D, gnorms_G = [], []
        per_sample_norms_list = []
        
        for i, (real_x, s) in enumerate(train_dataloader):
            batch_size = real_x.size(0)
            real_x = real_x.to(device)
            s = s.to(device)

            # ====== Update Discriminator ======
            for _ in range(critic_steps):
                z = torch.randn(batch_size, z_dim, device=device)
                fake_x = generator(z, s).detach()

                # Shouldnt we use the abs value? Otherwise positive and negative values will cancel each other out
                d_real = discriminator(real_x, s)
                d_fake = discriminator(fake_x, s)

                # Gradient penalty
                u = torch.rand(batch_size, 1, device=device).expand_as(real_x)
                x_hat = (u * real_x + (1 - u) * fake_x).requires_grad_(True)
                d_hat = discriminator(x_hat, s)
                
                grad = torch.autograd.grad(
                    d_hat, x_hat,
                    grad_outputs=torch.ones_like(d_hat),
                    create_graph=True, retain_graph=True
                )[0]
                gp = ((grad.view(batch_size, -1).norm(2, dim=1) - 1)**2).mean()

                wasserstein = - (d_real.mean() - d_fake.mean())
                d_loss = wasserstein + lambda_gp * gp

                opt_d.zero_grad()
                d_loss.backward()
                
                # CALCULATE BEFORE CLIPPING/STEPPING
                gnorm_d = sum(p.grad.norm()**2 for p in discriminator.parameters() if p.grad is not None)**0.5
                
                # Gradient clipping (suggested by ChatGPT)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
                opt_d.step()
                
                # Store metrics
                real_scores.append(d_real.mean().item())
                fake_scores.append(d_fake.mean().item())
                gps.append(gp.item())
                d_losses.append(d_loss.item())
                gnorms_D.append(gnorm_d.item())

                # store per-sample norms for logging
                per_sample_norms_list.append(grad.view(batch_size, -1).norm(2, dim=1).detach().cpu())

            # ====== Update Generator ======
            z = torch.randn(batch_size, z_dim, device=device)
            gen_x = generator(z, s)
            g_loss = -discriminator(gen_x, s).mean()
            
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10)

            opt_g.zero_grad()
            g_loss.backward()
            
            # CALCULATE BEFORE STEPPING
            gnorm_g = sum(p.grad.norm()**2 for p in generator.parameters() if p.grad is not None)**0.5
            
            opt_g.step()

         
            g_losses.append(g_loss.item())
            gnorms_G.append(gnorm_g.item())

        # =======================
        # Epoch summary metrics
        # =======================
        E_real = np.mean(real_scores)
        E_fake = np.mean(fake_scores)
        w_dist = E_real - E_fake

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"D_loss: {np.mean(d_losses):.4f} | G_loss: {np.mean(g_losses):.4f} | "
            f"E_real: {E_real:.4f} | E_fake: {E_fake:.4f} | W_dist: {w_dist:.4f} | "
            f"GP: {np.mean(gps):.4f} | "
            f"||grad_D||: {np.mean(gnorms_D):.4f} | ||grad_G||: {np.mean(gnorms_G):.4f}"
        )

        if DEBUG:
            # ============================================================
            # 1. GP diagnostics — recompute norms from the stored gp batch
            # ============================================================
            # Concatenate all per-sample norms
            all_per_sample_norms = torch.cat(per_sample_norms_list)

            print(
                f"[GP] mean={all_per_sample_norms.mean():.4f}  "
                f"std={all_per_sample_norms.std():.4f}  "
                f"min={all_per_sample_norms.min():.4f}  max={all_per_sample_norms.max():.4f}  "
                f"pct_in_[0.8,1.2]={((all_per_sample_norms>=0.8)&(all_per_sample_norms<=1.2)).float().mean():.2f}"
            )

            # ======================================================
            # 2. Critic output stats — re-evaluate on fresh batches
            # ======================================================
            print(
                f"[D] real: mean={np.mean(real_scores):.2f}  std={np.std(real_scores):.2f}  "
                f"min={np.min(real_scores):.2f}  max={np.max(real_scores):.2f}"
            )
            print(
                f"[D] fake: mean={np.mean(fake_scores):.2f}  std={np.std(fake_scores):.2f}  "
                f"min={np.min(fake_scores):.2f}  max={np.max(fake_scores):.2f}"
            )

            # ======================================
            # 3. Generator output magnitude checks
            # ======================================
            with torch.no_grad():
                z_dbg = torch.randn(batch_size, z_dim, device=device)
                gen_dbg = generator(z_dbg, s)

            print(
                f"[GEN] abs_max={gen_dbg.abs().max().item():.3e}  "
                f"mean={gen_dbg.mean().item():.3e}  std={gen_dbg.std().item():.3e}"
            )

        print("Critic Steps used: ", critic_steps)

        # Append global metrics for later plot
        d_loss_history.append(np.mean(d_losses))
        g_loss_history.append(np.mean(g_losses))
        w_dist_history.append(w_dist)

        # Calculate Frobenius Correlation
        metrics = compute_metrics(generator, val_dataloader, device, z_dim)
        current_frobenius = metrics["frob_corr_level"]
        current_frobenius_diff = metrics["frob_corr_diff"]
        current_price_frob = metrics["frob_price"]
        current_mean_dev = metrics["mean_dev"]
        current_var_dev = metrics["var_dev"]

        frob_level_history.append(current_frobenius)
        frob_diff_history.append(current_frobenius_diff)
        price_frob_history.append(current_price_frob)
        mean_dev_history.append(current_mean_dev)
        var_dev_history.append(current_var_dev)

        print("\n======= METRICS =======")
        print(f"Frobenius Correlation: {current_frobenius:.6f}")
        print(f"Frobenius Correlation Diff: {current_frobenius_diff:.6f}")
        print(f"Price Frobenius: {current_price_frob:.6f}")
        print(f"Mean Deviation: {current_mean_dev:.6f}")
        print(f"Variance Deviation: {current_var_dev:.6f}")
        print("========================\n")

        # Save Best Model based on Frobenius Correlation
        if current_price_frob < best_frobenius:
            print(f"New best model found (Frobenius: {current_price_frob:.6f}). Saving...")
            best_frobenius = current_price_frob
            torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}_best.pth")
            torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}_best.pth")

        # Periodic Save
        if save_model and epoch % save_every == 0:
            print(f"Saving models...")
            torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}.pth")
            torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}.pth")

    return (generator, discriminator, d_loss_history, g_loss_history, w_dist_history,
           frob_level_history, frob_diff_history, price_frob_history, mean_dev_history, var_dev_history)

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # hyperparameters (tune as needed)
    Z_DIM = 32
    S_DIM = 16
    X_DIM = 32
    HIDDEN = 128
    BATCH = 64
    EPOCHS = 10

    dataset = SimpleOrderBookDataset(n_samples=2000, x_dim=X_DIM, s_dim=S_DIM)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    # choose WGAN or vanilla GAN
    use_wgan = True  # set True to train WGAN-style critic

    G = Generator(z_dim=Z_DIM, s_dim=S_DIM, hidden_dim=HIDDEN, out_dim=X_DIM)
    D = Discriminator(x_dim=X_DIM, s_dim=S_DIM, hidden_dim=HIDDEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_gan(
        generator=G,
        discriminator=D,
        dataloader=loader,
        z_dim=Z_DIM,
        num_epochs=EPOCHS,
        device=device,
        wgan=True,          # still True (we're in Wasserstein mode)
        critic_steps=5,
        lr_g=1e-4,
        lr_d=1e-4
    )
