# Imbalanced LOB GAN Implementation

## Overview

This implementation creates a new GAN type that uses CSV data with order book imbalance based on price changes, following the approach from Cont et al. "Limit Order Book Simulation with Generative Adversarial Networks".

## Key Files Created

### 1. `train_model_imbalance.py`
Main training script containing:
- **`LobGanDatasetImbalance` class**: Dataset that implements the imbalanced order book approach
- **Training loop**: Uses existing GAN architecture from `model/gan_model.py`
- **MLflow integration**: Tracks experiments and hyperparameters

### 2. `test_imbalanced_dataset.py`
Test script to verify the dataset implementation works correctly.

## How It Works

### Imbalanced Order Book Concept

The key innovation is how the dataset represents the order book state:

1. **Price Change Detection**:
   - Computes mid-price changes between consecutive snapshots
   - Converts changes to "ticks" (discrete price levels)
   - Tracks cumulative price changes

2. **Dynamic Column Balancing**:
   - **Default state**: First `MARKET_DEPTH` columns are positive (bids), next `MARKET_DEPTH` columns are negative (asks)
   - **Price increase by N ticks**: Shifts N columns from bid side to ask side
   - **Price decrease by N ticks**: Shifts N columns from ask side to bid side

3. **Volume-Only Features**:
   - Only quantities are used as features (not prices)
   - Bid quantities are positive
   - Ask quantities are negative
   - Prices are used internally only to detect changes

### Example

With `MARKET_DEPTH=10`:

```
Initial state (no price change):
Columns 0-9:   Positive (bid quantities)
Columns 10-19: Negative (ask quantities)

After +1 tick price increase:
Columns 0-8:   Positive (9 bid quantities)
Columns 9-19:  Negative (11 ask quantities)

After -1 tick price decrease:
Columns 0-10:  Positive (11 bid quantities)
Columns 11-19: Negative (9 ask quantities)
```

### Dataset Structure

**Input CSV**: 40 columns
- `askPx_0` to `askPx_9`: Ask prices (10 levels)
- `askQty_0` to `askQty_9`: Ask quantities (10 levels)
- `bidPx_0` to `bidPx_9`: Bid prices (10 levels)
- `bidQty_0` to `bidQty_9`: Bid quantities (10 levels)

**Output**: `(X_next, S_curr)` pairs
- `X_next`: Next LOB state (2 * MARKET_DEPTH columns)
- `S_curr`: Current LOB state (2 * MARKET_DEPTH columns)
- All values are standardized (zero mean, unit variance)

## Usage

### Testing the Dataset

```bash
python test_imbalanced_dataset.py
```

This will:
- Load a CSV file
- Create the dataset with different market depths
- Verify shapes and indices
- Display statistics

### Training the Model

```bash
python train_model_imbalance.py
```

This will:
- Load CSV files from `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/`
- Create the imbalanced dataset
- Train the GAN
- Save models to `out/models/`
- Track experiments in MLflow

## Hyperparameters

Key hyperparameters in `train_model_imbalance.py`:

```python
Z_DIM = 12               # Noise dimension
HIDDEN_D = 64            # Discriminator hidden size
HIDDEN_G = 64            # Generator hidden size
BATCH = 128              # Batch size
EPOCHS = 2500            # Training epochs
MARKET_DEPTH = 10        # Number of levels per side
LR_D = 4e-5              # Discriminator learning rate
LR_G = 2e-5              # Generator learning rate
LAMBDA_GP = 10           # Gradient penalty weight
```

## Next Steps

### Price Decoding During Test Phase

To decode prices during generation:

1. **Track cumulative imbalance**: Monitor the shift in bid/ask column balance
2. **Compute price ticks**: Each shift represents a price tick change
3. **Reconstruct prices**: Apply cumulative ticks to initial price level

This will be implemented in a separate test script (e.g., `test_model_imbalance.py`).

## Implementation Details

### `_create_imbalanced_states` Method

This is the core method that creates the imbalanced states:

```python
def _create_imbalanced_states(bid_qty, ask_qty, price_change_ticks):
    # Track cumulative price change
    cumulative_price_change = 0
    
    for each sample:
        cumulative_price_change += price_change_ticks[i]
        
        # Determine split between bid and ask
        n_bid_cols = MARKET_DEPTH - cumulative_price_change
        n_ask_cols = MARKET_DEPTH + cumulative_price_change
        
        # Fill bid columns (positive)
        # Fill ask columns (negative)
```

### Standardization

- Data is standardized **globally** across all files
- Mean and std are computed on the combined dataset
- This ensures consistent scaling across training samples

## Advantages of This Approach

1. **Price-invariant features**: Model learns volume patterns independent of absolute price levels
2. **Natural price dynamics**: Price changes emerge from order book imbalance
3. **Efficient representation**: Uses only 2*MARKET_DEPTH features instead of 4*MARKET_DEPTH (prices + quantities)
4. **Realistic simulation**: Follows the actual mechanism of price formation in limit order books
