from pathlib import Path
from typing import Optional

# Define base paths relative to the project root or this file
# Assuming this file is in metrics_lib/, so parent is LOBSimulation/
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Use the same data directory as train_model_imbalance.py
DATA_DIR = BASE_FOLDER / "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/"

# For real data, we'll use the processed CSV files from the training data directory
# The metrics will look for CSV files in this directory
FAKE_CSV = BASE_FOLDER / "generated_lob.csv"
FAKE_CSV = BASE_FOLDER / "out/reconstructed_concatenated.csv"

class Config:
    """Centralized configuration"""
    # Paths
    DATA_DIR: str = str(DATA_DIR)
    FAKE_CSV: Optional[str] = str(FAKE_CSV)  # Set to path when generated data is ready
    OUTPUT_DIR: str = "./out/metrics"
    
    # Plot settings
    GRID_COLS: int = 2
    DRAW_VLINE_ZERO: bool = True
    PLOT_ALL_LEVELS: bool = True          # All level marginals (unconditional)
    PLOT_LOB_SHAPE: bool = True           # Average LOB shape
    PLOT_CORRELATION: bool = True         # Correlation matrices + Frobenius
    PLOT_INDIVIDUAL_DAYS: bool = True     # Plot metrics for each day individually
    
    # Style
    FIGURE_DPI: int = 300
    REAL_COLOR: str = "#4A90E2"
    FAKE_COLOR: str = "#F5A623"
    REAL_ALPHA: float = 0.6
    FAKE_ALPHA: float = 0.6
