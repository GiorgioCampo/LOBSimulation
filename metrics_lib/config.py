from pathlib import Path
from typing import Optional

# Define base paths relative to the project root or this file
# Assuming this file is in metrics_lib/, so parent is LOBSimulation/
BASE_FOLDER = Path(__file__).resolve().parent.parent
DAY = "20191002"
REAL_CSV = BASE_FOLDER / f"out/data/{DAY}/FLEX_L2_SNAPSHOT.csv"
FAKE_CSV = BASE_FOLDER / f"generated_lob.csv"

class Config:
    """Centralized configuration"""
    # Paths
    REAL_CSV: str = str(REAL_CSV)
    FAKE_CSV: Optional[str] = str(FAKE_CSV)  # Set to path when generated data is ready
    OUTPUT_DIR: str = "./out/metrics"
    DATA_DIR: str = str(BASE_FOLDER / "out/data")
    
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
