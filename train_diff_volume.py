import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import numpy as np

from model.diff_volume import DiffVolume
from model.diffusion_utils import DiffusionUtils
from train_model_imbalance import LobGanDatasetImbalance # Use imbalanced databento dataset
from metrics_lib.config import Config

# Configuration
MARKET_DEPTH = Config.MARKET_DEPTH
MAX_PRICE_CHANGE = Config.MAX_PRICE_CHANGE
INTERVAL_MS = Config.INTERVAL_MS
HIDDEN_DIM = 64
N_LAYERS = 32
N_HEADS = 4
BATCH_SIZE = 512
EPOCHS = 200
LR = 1e-4
DIFF_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("D:/Amsterdam/Thesis/order_book_databento/dbn_out")

def train():
    # 1. Load Data
    dbn_files = sorted(DATA_DIR.glob("*.dbn.zst"))
    if not dbn_files:
        print(f"No .dbn.zst files found in {DATA_DIR}")
        return

    # Use first few files for training
    files = [str(f) for f in dbn_files if f.stat().st_size > 1000]
    
    dataset = LobGanDatasetImbalance(
        file_paths=files[:5], 
        market_depth=MARKET_DEPTH,
        interval_ms=INTERVAL_MS,
        max_price_change=MAX_PRICE_CHANGE
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model and Utils
    x_dim = dataset.X.shape[1]
    c_dim = dataset.S.shape[1]
    
    model = DiffVolume(
        input_dim=x_dim,
        cond_context_dim=c_dim,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS
    ).to(DEVICE)
    
    diffusion_utils = DiffusionUtils(n_steps=DIFF_STEPS, device=DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Training Loop
    mlflow.set_experiment("diff_volume_databento")
    
    with mlflow.start_run():
        mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
        mlflow.log_param("EPOCHS", EPOCHS)
        mlflow.log_param("LR", LR)
        mlflow.log_param("DIFF_STEPS", DIFF_STEPS)
        mlflow.log_param("HIDDEN_DIM", HIDDEN_DIM)
        mlflow.log_param("N_LAYERS", N_LAYERS)
        mlflow.log_param("MARKET_DEPTH", MARKET_DEPTH)
        mlflow.log_param("MAX_PRICE_CHANGE", MAX_PRICE_CHANGE)
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_losses = []
            
            for x_0, c in loader:
                x_0 = x_0.to(DEVICE)
                c = c.to(DEVICE)
                batch_size = x_0.size(0)
                
                # Sample time steps
                t = torch.randint(0, DIFF_STEPS, (batch_size,), device=DEVICE).long()
                
                # Add noise
                noise = torch.randn_like(x_0)
                x_t = diffusion_utils.q_sample(x_0, t, noise)
                
                # Predict score (or noise-scaled version)
                # Denoising Score Matching objective:
                # Loss = || s_theta(x_t, t, c) - target_score ||^2
                score_pred = model(x_t, t, c)
                score_target = diffusion_utils.get_score_target(noise, t)
                
                loss = F.mse_loss(score_pred, score_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), MODELS_DIR / f"diff_volume_epoch_{epoch+1}.pth")
        
        # Save final model
        torch.save(model.state_dict(), MODELS_DIR / "diff_volume_final.pth")
        mlflow.pytorch.log_model(model, "diff_volume_model")

if __name__ == "__main__":
    import torch.nn.functional as F
    train()
