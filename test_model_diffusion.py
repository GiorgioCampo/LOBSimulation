import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model.diff_volume import DiffVolume
from model.diffusion_utils import DiffusionUtils
from train_model_imbalance import LobGanDatasetImbalance
from metrics_lib.config import Config

# Configuration
MARKET_DEPTH = Config.MARKET_DEPTH
MAX_PRICE_CHANGE = Config.MAX_PRICE_CHANGE
INTERVAL_MS = Config.INTERVAL_MS
HIDDEN_DIM = 64
N_LAYERS = 32
N_HEADS = 4
DIFF_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "out/models/"
DATA_DIR = Path("D:/Amsterdam/Thesis/order_book_databento/dbn_out")
OUT_DIR = BASE_DIR / "out/generated_diffusion/"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_pivot_shift(data: torch.Tensor, center_idx: int):
    """
    Finds the index where the data transitions from bid (<0) to ask (>0).
    data: (batch, length)
    """
    B, T = data.shape
    is_bid = data < 0
    is_ask = data > 0
    transition = is_bid[:, :-1] & is_ask[:, 1:]
    
    # Distance from center_idx - 1 (the theoretical center)
    center = center_idx - 1
    idxs = torch.arange(T - 1, device=data.device).unsqueeze(0)
    dist = torch.abs(idxs - center)
    
    # Mask non-transitions
    dist = torch.where(transition, dist, torch.full_like(dist, float(T)))
    
    pivot = dist.argmin(dim=1)
    has_transition = transition.any(dim=1)
    
    # shift = pivot - center
    shift = torch.where(has_transition, pivot - center, torch.zeros_like(pivot))
    return shift, pivot

def generate_paths(n_paths=5, n_steps=200):
    # 1. Load Data for initialization and normalization
    dbn_files = sorted(DATA_DIR.glob("*.dbn.zst"))
    files = [str(f) for f in dbn_files if f.stat().st_size > 1000]
    
    dataset = LobGanDatasetImbalance(
        file_paths=files[:1], 
        market_depth=MARKET_DEPTH,
        interval_ms=INTERVAL_MS,
        max_price_change=MAX_PRICE_CHANGE
    )
    
    x_dim = dataset.X.shape[1]
    s_dim = dataset.S.shape[1]
    tick_size = dataset.tick_size
    
    # 2. Load Model
    model = DiffVolume(
        input_dim=x_dim,
        cond_context_dim=s_dim,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS
    ).to(DEVICE)
    
    model_path = MODELS_DIR / "diff_volume_final.pth"
    if not model_path.exists():
        # Try to find checkpoint
        checkpoints = list(MODELS_DIR.glob("diff_volume_epoch_*.pth"))
        if checkpoints:
            model_path = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1]
        else:
            print("No trained model found. Please train first.")
            return

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    diffusion_utils = DiffusionUtils(n_steps=DIFF_STEPS, device=DEVICE)
    
    # 3. Generation Loop
    X_CENTER = MARKET_DEPTH + MAX_PRICE_CHANGE
    all_paths = []
    
    for p in range(n_paths):
        print(f"Generating path {p+1}/{n_paths}")
        
        # Random start from dataset
        start_idx = np.random.randint(0, len(dataset) - n_steps)
        current_s = dataset.S[start_idx:start_idx+1].to(DEVICE)
        
        price_history = [0.0] # Relative to start
        volume_history = []
        
        current_price = 0.0
        
        for _ in tqdm(range(n_steps)):
            # Sample X_{t+1} using diffusion
            # p_sample_loop expects (batch_size, input_dim)
            x_gen = diffusion_utils.p_sample_loop(model, (1, x_dim), current_s)
            
            # Find shift in generated X
            shift, pivot = find_pivot_shift(x_gen, X_CENTER)
            shift_val = shift.item()
            
            # Update price
            current_price += shift_val * tick_size
            price_history.append(current_price)
            
            # Extract new S_{t+1} from X_{t+1}
            # Standard order: [q_d^b, ..., q_1^b, q_1^a, ..., q_d^a]
            # q_1^b is at pivot, q_1^a at pivot + 1
            pivot_idx = pivot.item()
            offsets = np.arange(-(MARKET_DEPTH - 1), MARKET_DEPTH + 1)
            indices = pivot_idx + offsets
            indices = np.clip(indices, 0, x_dim - 1)
            
            new_s = x_gen[0, indices].unsqueeze(0)
            current_s = new_s
            
            # Store some volume info (e.g. best bid/ask)
            volume_history.append(x_gen[0, [pivot_idx, pivot_idx+1]].cpu().numpy())
            
        all_paths.append({
            'prices': price_history,
            'volumes': np.array(volume_history)
        })

    # 4. Plot Results
    plt.figure(figsize=(12, 6))
    for i, path in enumerate(all_paths):
        plt.plot(path['prices'], label=f'Path {i+1}')
    plt.title("Generated Price Paths (Diffusion)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price Change (Ticks)")
    plt.grid(True)
    plt.legend()
    plt.savefig(OUT_DIR / "generated_prices.png")
    plt.close()

    # Save paths to CSV
    # Each path as a column for prices
    df_prices = pd.DataFrame({f'Path_{i}': p['prices'] for i, p in enumerate(all_paths)})
    df_prices.to_csv(OUT_DIR / "generated_prices.csv", index=False)
    
    print(f"Generation complete. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    generate_paths(n_paths=1, n_steps=5)
