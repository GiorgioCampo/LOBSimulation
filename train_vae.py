import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from train_model_imbalance import LobGanDatasetImbalance, MARKET_DEPTH, DATA_DIR
from model.vae_model import LOBVAE, vae_loss
import numpy as np

# Hyperparameters
HISTORY_WINDOW = 5
LATENT_DIM = 32
HIDDEN_DIM = 128
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_SAVE_PATH = "out/models/vae_history.pth"

def train_vae():
    # dataset.s will have shape (N, H * 40)
    dataset = LobGanDatasetImbalance(DATA_DIR, market_depth=MARKET_DEPTH, history_window=HISTORY_WINDOW)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_dim = dataset.s.shape[1]
    model = LOBVAE(input_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training VAE on {len(dataset)} samples...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for _, s in dataloader:
            s = s.to(DEVICE)
            optimizer.zero_grad()
            
            recon_s, mu, logvar = model(s)
            loss = vae_loss(recon_s, s, mu, logvar)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataset.s)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
            
    torch.save(model.state_dict(), VAE_SAVE_PATH)
    print(f"VAE saved to {VAE_SAVE_PATH}")

if __name__ == "__main__":
    import os
    if not os.path.exists("out/models"):
        os.makedirs("out/models")
    train_vae()
