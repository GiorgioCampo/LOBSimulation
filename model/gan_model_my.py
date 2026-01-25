import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm as sn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "out/models/"
DEBUG = False

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
# Helper: Autoregressive Generation
# -----------------------
def generate_autoregressive_data(generator, dataset, device, z_dim, horizon=100, num_paths=100):
    generator.eval()
    
    # 1. Sample Initial States
    if hasattr(dataset, 'X') and hasattr(dataset, 'S'):
        # Deterministic sampling
        total_samples = len(dataset.S)
        if total_samples > num_paths:
            indices = np.linspace(0, total_samples - 1, num_paths, dtype=int)
        else:
            indices = np.arange(total_samples)
        current_s = dataset.S[indices].to(device)
    else:
        # Fallback
        # We need to access the dataloader to get a batch if dataset doesn't have X/S
        # This helper might need dataloader passed in, or we assume dataset has X/S for now
        # as the main use case supports it. 
        # If strictly needed, we can pass a starting batch.
        raise NotImplementedError("Dataset must have .S attribute for this helper.")

    differentiate = getattr(dataset, 'differentiate', False)
    include_diffs = getattr(dataset, 'include_diffs_in_state', False)

    # 2. Generate
    generated_data = []
    with torch.no_grad():
        # Use tqdm only if it's a large generation (optional, maybe skip for speed in metric)
        # or pass a flag. For now, let's keep it simple/silent or minimal.
        iterator = range(horizon)
        
        for t in iterator:
            z = torch.randn(current_s.size(0), z_dim, device=device)
            x_next = generator(z, current_s)
            generated_data.append(x_next.cpu())
            
            if include_diffs:
                x_dim_val = x_next.size(1)
                x_prev = current_s[:, :x_dim_val]
                diff_next = x_next - x_prev
                current_s = torch.cat([x_next, diff_next], dim=1)
            else:
                current_s = x_next

    # 3. Flatten
    fake_X = torch.stack(generated_data, dim=1) 
    fake_X = fake_X.reshape(-1, fake_X.size(2))
    return fake_X.numpy()


# -----------------------
# Training skeleton
# -----------------------
def train_gan(
    *,
    generator,
    discriminator,
    dataloader,
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
    

    # -----------------------
    # Helper to compute Frobenius metric (Autoregressive)
    # -----------------------
    def compute_metric(generator, dataloader, device, z_dim, horizon=100, validation_batch_size=500):
        # Try importing metrics1
        try:
            import metrics1
        except ImportError:
            import sys
            sys.path.append(str(BASE_DIR))
            import metrics1

        # Generate Data
        fake_X_np = generate_autoregressive_data(
            generator, dataloader.dataset, device, z_dim, horizon, validation_batch_size
        )
        
        # Real Data
        if hasattr(dataloader.dataset, 'X'):
            real_X_np = dataloader.dataset.X.numpy()
        else:
             # Fallback
            real_list = []
            for real_x, s in dataloader:
                real_list.append(real_x)
            real_X_np = torch.cat(real_list, dim=0).numpy()

        dataset = dataloader.dataset

        # 5. Compute metric
        # Note: real_X_np is typically (N_samples, X_Dim)
        # fake_X_np is (Num_paths * Horizon, X_Dim)
        
        # Retrieve indices if available
        price_indices = getattr(dataset, 'price_indices', None)
        qty_indices = getattr(dataset, 'qty_indices', None)
        bid_qty_indices = getattr(dataset, 'bid_qty_indices', None)
        ask_qty_indices = getattr(dataset, 'ask_qty_indices', None)
        
        # Compute Price Metric (No normalization)
        if price_indices is not None and len(price_indices) > 0:
            frob_price = metrics1.compute_frobenius_correlation(
                real_X_np[:, price_indices], 
                fake_X_np[:, price_indices]
            )
        else:
            frob_price = float('inf')

        # Compute Qty Metric (With Queue Normalization)
        if (bid_qty_indices is not None and len(bid_qty_indices) > 0 and 
            ask_qty_indices is not None and len(ask_qty_indices) > 0):
            
            # Initialize Normalizer with ALL real data (or the subset we have)
            # We use the full real_X_np for this to get robust constants
            real_bid_q = real_X_np[:, bid_qty_indices]
            real_ask_q = real_X_np[:, ask_qty_indices]
            
            normalizer = metrics1.QueueNormalizer(real_bid_q, real_ask_q)
            
            # Normalize Real Quantities
            real_bid_norm, real_ask_norm = normalizer.normalize(real_bid_q, real_ask_q)
            real_qty_norm = np.concatenate([real_bid_norm, real_ask_norm], axis=1)
            
            # Normalize Fake Quantities
            fake_bid_q = fake_X_np[:, bid_qty_indices]
            fake_ask_q = fake_X_np[:, ask_qty_indices]
            
            fake_bid_norm, fake_ask_norm = normalizer.normalize(fake_bid_q, fake_ask_q)
            fake_qty_norm = np.concatenate([fake_bid_norm, fake_ask_norm], axis=1)
            
            # Compute Metric on Normalized Quantities
            frob_qty = metrics1.compute_frobenius_correlation(
                real_qty_norm, 
                fake_qty_norm
            )
        elif qty_indices is not None and len(qty_indices) > 0:
             # Fallback if specific bid/ask indices missing but general qty exists
            frob_qty = metrics1.compute_frobenius_correlation(
                real_X_np[:, qty_indices], 
                fake_X_np[:, qty_indices]
            )
        else:
            # Fallback
            full_frob = metrics1.compute_frobenius_correlation(real_X_np, fake_X_np)
            frob_price = full_frob
            frob_qty = full_frob
        
        generator.train()
        return frob_price, frob_qty

    # -----------------------
    # Helper: Save Training Plots
    # -----------------------
    def save_training_plots(generator, dataloader, device, z_dim, epoch, output_dir):
        try:
            import metrics1
            import os
        except ImportError:
            return

        # Create epoch directory
        epoch_dir = output_dir / f"epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 1. Generate Fake Data (Autoregressive)
        # Use a reasonable number of paths for plotting (e.g. 500 or 1000)
        fake_X_np = generate_autoregressive_data(
            generator, dataloader.dataset, device, z_dim, horizon=100, num_paths=500
        )
        
        # 2. Get Real Data
        dataset = dataloader.dataset
        if hasattr(dataset, 'X'):
            real_X_np = dataset.X.numpy()
        else:
            return # Cannot plot without real data
        
            
        # 3. Extract Components
        bid_px_idx = getattr(dataset, 'bid_px_indices', [])
        ask_px_idx = getattr(dataset, 'ask_px_indices', [])
        bid_qty_idx = getattr(dataset, 'bid_qty_indices', [])
        ask_qty_idx = getattr(dataset, 'ask_qty_indices', [])
        labels = getattr(dataset, 'labels', [])
        
        if not (bid_px_idx and ask_px_idx and bid_qty_idx and ask_qty_idx):
            print("Skipping plots: Missing indices in dataset.")
            return

        # Real Components
        real_px_bid = real_X_np[:, bid_px_idx]
        real_px_ask = real_X_np[:, ask_px_idx]
        real_qty_bid = real_X_np[:, bid_qty_idx]
        real_qty_ask = real_X_np[:, ask_qty_idx]
        
        # Fake Components
        fake_px_bid = fake_X_np[:, bid_px_idx]
        fake_px_ask = fake_X_np[:, ask_px_idx]
        fake_qty_bid = fake_X_np[:, bid_qty_idx]
        fake_qty_ask = fake_X_np[:, ask_qty_idx]
        
        # 4. Normalize Quantities
        # Initialize normalizer with ALL real data
        normalizer = metrics1.QueueNormalizer(real_qty_bid, real_qty_ask)
        
        real_q_bid_norm, real_q_ask_norm = normalizer.normalize(real_qty_bid, real_qty_ask)
        fake_q_bid_norm, fake_q_ask_norm = normalizer.normalize(fake_qty_bid, fake_qty_ask)
        
        # Combined Normalized Quantities for some plots
        real_q_all = np.hstack([real_q_bid_norm, real_q_ask_norm])
        fake_q_all = np.hstack([fake_q_bid_norm, fake_q_ask_norm])
        
        # 5. Create LOBData objects (for conditional plots)
        # Note: LOBData expects raw quantities, it might handle normalization internally or we pass normalized?
        # Checking metrics1: LOBData stores raw. plot_conditional_marginals takes LOBData AND normalized queues.
        real_data = metrics1.LOBData(real_px_bid, real_qty_bid, real_px_ask, real_qty_ask)
        fake_data = metrics1.LOBData(fake_px_bid, fake_qty_bid, fake_px_ask, fake_qty_ask)
        
        # 6. Compute Midprice Direction Matrices
        try:
            M_real = metrics1.compute_midprice_direction_matrix(
                real_data.mid, np.abs(real_data.qty_bid[:, 0]), np.abs(real_data.qty_ask[:, 0])
            )
            M_fake = metrics1.compute_midprice_direction_matrix(
                fake_data.mid, np.abs(fake_data.qty_bid[:, 0]), np.abs(fake_data.qty_ask[:, 0])
            )
        except Exception as e:
            print(f"Plotting error (Midprice Matrix): {e}")
            M_real, M_fake = None, None

        # 7. Generate Plots
        print(f"Generating training plots for Epoch {epoch}...")
        
        # 1. Conditional Marginals
        try:
            metrics1.plot_conditional_marginals(
                real_q_bid=real_q_bid_norm,
                real_q_ask=real_q_ask_norm,
                real_data=real_data,
                output_path=str(epoch_dir / "conditional_marginals.png"),
                fake_q_bid=fake_q_bid_norm,
                fake_q_ask=fake_q_ask_norm,
                fake_data=fake_data,
                ks=[1, 2],
                levels=[0, 1]
            )
        except Exception as e: print(f"Plot error (Conditional): {e}")

        # 2. Correlation Matrices
        try:
            metrics1.plot_correlation_matrices(
                real_q_all=real_q_all,
                labels=labels,
                output_path=str(epoch_dir / "correlation_matrices.png"),
                fake_q_all=fake_q_all
            )
        except Exception as e: print(f"Plot error (Correlation): {e}")

        # 3. Midprice Direction Matrices
        try:
            metrics1.plot_midprice_direction_matrices(
                M_real=M_real,
                output_path=str(epoch_dir / "midprice_direction_matrix.png"),
                M_fake=M_fake
            )
        except Exception as e: print(f"Plot error (Midprice Plot): {e}")

        # 4. Average LOB Shape
        try:
            metrics1.plot_average_lob_shape(
                real_q_all=real_q_all,
                labels=labels,
                output_path=str(epoch_dir / "avg_lob_shape.png"),
                fake_q_all=fake_q_all
            )
        except Exception as e: print(f"Plot error (LOB Shape): {e}")

        # 5. All Level Marginals
        try:
            metrics1.plot_all_level_marginals(
                real_q_all=real_q_all,
                labels=labels,
                output_path=str(epoch_dir / "marginals_all_levels.png"),
                fake_q_all=fake_q_all
            )
        except Exception as e: print(f"Plot error (Marginals): {e}")


    # Initialize best metrics
    best_frobenius_price = float('inf')
    best_frobenius_qty = float('inf')
    
    # Create frames directory
    frames_dir = BASE_DIR / "out/training_frames"
    import os
    os.makedirs(frames_dir, exist_ok=True)

    for epoch in range(num_epochs):

        critic_steps = int(max(critic_steps_final, critic_steps_initial * gamma ** (epoch)))

        d_losses, g_losses, gps = [], [], []
        real_scores, fake_scores = [], []
        gnorms_D, gnorms_G = [], []
        per_sample_norms_list = []
        
        for i, (real_x, s) in enumerate(dataloader):
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

        # Calculate Frobenius Correlations
        frob_price, frob_qty = compute_metric(generator, dataloader, device, z_dim)
        print(f"Frobenius Correlation - Price: {frob_price:.6f} | Qty: {frob_qty:.6f}")

        # Save Best Model based on Price Correlation
        if frob_price < best_frobenius_price:
            print(f"New best Price model found (Frobenius: {frob_price:.6f}). Saving...")
            best_frobenius_price = frob_price
            torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}_best_price.pth")
            torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}_best_price.pth")

        # Save Best Model based on Qty Correlation
        if frob_qty < best_frobenius_qty:
            print(f"New best Qty model found (Frobenius: {frob_qty:.6f}). Saving...")
            best_frobenius_qty = frob_qty
            torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}_best_qty.pth")
            torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}_best_qty.pth")

        # Periodic Save
        if save_model and epoch % save_every == 0:
            print(f"Saving models...")
            torch.save(generator.state_dict(), MODELS_DIR / f"generator_{model_name}.pth")
            torch.save(discriminator.state_dict(), MODELS_DIR / f"discriminator_{model_name}.pth")
            
            # Save Training Plots (Frames)
            save_training_plots(generator, dataloader, device, z_dim, epoch, frames_dir)

    return generator, discriminator, d_loss_history, g_loss_history, w_dist_history

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
