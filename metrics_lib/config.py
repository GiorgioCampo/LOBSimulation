from pathlib import Path
from typing import Optional
import numpy as np

# Define base paths relative to the project root or this file
# Assuming this file is in metrics_lib/, so parent is LOBSimulation/
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Point to the raw Databento data directory
DATA_DIR = Path(r"D:\Amsterdam\Thesis\order_book_databento\dbn_out")

# For real data, we'll use the processed CSV files from the training data directory
# The metrics will look for CSV files in this directory
# FAKE_CSV = BASE_FOLDER / "generated_lob.csv"
FAKE_CSV = BASE_FOLDER / "out/metrics_data/fake/generated.csv"

class Config:
    """Centralized configuration"""
    # Paths
    DATA_DIR: str = str(DATA_DIR)
    FAKE_CSV: Optional[str] = str(FAKE_CSV)  # Set to path when generated data is ready
    OUTPUT_DIR: str = "./out/metrics"
    
    # Core settings
    INTERVAL_MS: int = 1000
    MARKET_DEPTH: int = 6       # Default depth (number of levels per side)
    MAX_PRICE_CHANGE: int = 3    # Default max tick movement per step
    PLOT_LEVELS: int = 3         # Number of levels to show in diagnostic plots
    
    # Time Filtering (UTC hours)
    START_HOUR: int = 15
    START_MINUTE: int = 30
    END_HOUR: int = 20
    
    # Plot settings
    GRID_COLS: int = 2
    DRAW_VLINE_ZERO: bool = True
    PLOT_ALL_LEVELS: bool = True          # All level marginals (unconditional)
    PLOT_LOB_SHAPE: bool = True           # Average LOB shape
    PLOT_CORRELATION: bool = True         # Correlation matrices + Frobenius
    PLOT_INDIVIDUAL_DAYS: bool = False     # Plot metrics for each day individually
    
    # Style
    FIGURE_DPI: int = 300
    REAL_COLOR: str = "#4A90E2"
    FAKE_COLOR: str = "#F5A623"
    REAL_ALPHA: float = 0.6
    FAKE_ALPHA: float = 0.6

    # Normalization
    # Options: "zscore", "queue"
    NORMALIZATION_METHOD: str = "zscore"
    SCALE_FACTOR_BID = np.array([0.11813817, 0.07603832, 0.06854401, 0.06335132, 0.06005475, 0.05812741,
    0.05673298, 0.05377442, 0.05361673, 0.04632622])
    SCALE_FACTOR_ASK = np.array([0.12008103, 0.08061185, 0.07212465, 0.06790647, 0.06689713, 0.06533857,
    0.06199458, 0.05988983, 0.0613501 , 0.05291737])
