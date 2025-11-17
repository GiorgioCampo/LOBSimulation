import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent[1]
MODELS_DIR = BASE_DIR / "out/models/"

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
        self.final_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
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
    dataloader,
    z_dim=32,
    num_epochs=50,
    device='cpu',
    wgan=False,
    critic_steps=5,
    lr_g=1e-4,
    lr_d=1e-4,
    lambda_gp=10
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
    if wgan:
        # WGAN commonly used RMSprop (original paper), but Adam also sometimes used
        opt_d = optim.RMSprop(discriminator.parameters(), lr=lr_d)
        opt_g = optim.RMSprop(generator.parameters(), lr=lr_g)
    else:
        opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

    # Loss for vanilla GAN
    bce_loss = nn.BCEWithLogitsLoss()

    best_w_dist = None
    for epoch in range(num_epochs):
        real_scores_epoch = []
        fake_scores_epoch = []
        for i, (real_x, s) in enumerate(dataloader):
            batch_size = real_x.size(0)
            real_x = real_x.to(device)
            s = s.to(device)

            # ====== (1) Update Critic ======
            for _ in range(critic_steps):
                z = torch.randn(batch_size, z_dim, device=device)
                fake_x = generator(z, s).detach()

                d_real = discriminator(real_x, s)
                d_fake = discriminator(fake_x, s)

                real_scores_epoch.append(d_real.detach().cpu().numpy().mean())
                fake_scores_epoch.append(d_fake.detach().cpu().numpy().mean())

                # Wasserstein component
                wasserstein_loss = -(d_real.mean() - d_fake.mean())

                # Gradient penalty
                u = torch.rand(batch_size, 1, device=device).expand_as(real_x)
                x_hat = (u * real_x + (1 - u) * fake_x).requires_grad_(True)
                d_hat = discriminator(x_hat, s)
                grad = torch.autograd.grad(
                    d_hat, x_hat,
                    grad_outputs=torch.ones_like(d_hat),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                grad = grad.view(batch_size, -1)
                gp = ((grad.norm(2, dim=1) - 1)**2).mean()

                d_loss = wasserstein_loss + lambda_gp * gp

                opt_d.zero_grad()
                d_loss.backward()

                # Gradient clipping (suggested by ChatGPT)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5)

                opt_d.step()

            # ====== (2) Update Generator ======
            z = torch.randn(batch_size, z_dim, device=device)
            gen_x = generator(z, s)

            if wgan:
                g_loss = -discriminator(gen_x, s).mean()
            else:
                g_logits = discriminator(gen_x, s)
                g_loss = bce_loss(g_logits, torch.ones(batch_size, device=device))

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        # ====== TO REVISE - IS IT STILL VALID THE GRADIENT CALCULATION? ======
        with torch.no_grad():
            # Recompute for logging
            z = torch.randn(batch_size, z_dim, device=device)
            fake_x = generator(z, s)
            E_real = discriminator(real_x, s).mean().item()
            E_fake = discriminator(fake_x, s).mean().item()

            # gradient norms (after last step of the epoch)
            def grad_norm(params):
                total = 0.0
                for p in params:
                    if p.grad is not None:
                        total += (p.grad.data.norm()**2).item()
                return total**0.5

            gnorm_D = grad_norm(discriminator.parameters())
            gnorm_G = grad_norm(generator.parameters())

        real_score_mean = float(np.mean(real_scores_epoch))
        fake_score_mean = float(np.mean(fake_scores_epoch)) 
        wasserstein_est = real_score_mean - fake_score_mean

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | "
            f"E_real: {real_score_mean:.4f} | E_fake: {fake_score_mean:.4f} |  Gap/W_dist: {wasserstein_est:.4f} "
            f"GP: {gp.item():.4f} | "
            f"||grad_D||: {gnorm_D:.4f} | ||grad_G||: {gnorm_G:.4f}"
        )

        # If W_dist is less then last, save model
        if (best_w_dist is None) or (wasserstein_est < best_w_dist):
            print(f"New best W_dist: {wasserstein_est:.3f}. Saving models...")
            best_w_dist = wasserstein_est
            torch.save(generator.state_dict(), MODELS_DIR / "generator.pth")
            torch.save(discriminator.state_dict(), MODELS_DIR / "discriminator.pth")

    return generator, discriminator

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
        lr_d=1e-4,
    )
