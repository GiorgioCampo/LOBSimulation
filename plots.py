import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_real_vs_generated(lob_df, df_gen, time_index, save=True):
    real = lob_df.copy()
    if not isinstance(real.index, pd.DatetimeIndex):
        if "time" in real.columns:
            real["time"] = pd.to_datetime(real["time"])
            real = real.set_index("time")

    gen = df_gen.copy()
    if not isinstance(gen.index, pd.DatetimeIndex):
        gen["time"] = pd.to_datetime(gen["time"])
        gen = gen.set_index("time")

    combined = pd.concat([real, gen])

    os.makedirs("out/plots", exist_ok=True)
    # save_ts_plot(
    #     combined[["bidPx_0"]],
    #     title="Real vs Generated L0 Bid Price",
    #     ylabel="Price",
    #     fname="real_plus_generated_bidPx0.png",
    #     out_dir="out/plots"
    # )

    # Remove last timestamp (biased time at 19, probably consolidated final price)
    real = real.iloc[:-1]
    gen = gen.iloc[:-1]

    plt.figure()
    plt.plot(gen.index, gen["bidPx_0"], label="Generated", linestyle="--")
    plt.plot(real.index, real["bidPx_0"], label="Real")
    plt.axvline(real.index[-time_index], color="red", linestyle=":", label="Transition")
    plt.legend()
    plt.title("Real vs Generated continuation (bidPx_0)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    if save:
        plt.savefig("out/plots/real_vs_generated_bidPx0.pdf")
    else:
        plt.show()
    plt.close()

def plot_real_vs_generated_conf(lob_df, df_gen_list, time_index, 
                                ci_low=0.05, ci_high=0.95, column="bidPx_0",
                                save=True):
    """
    lob_df       : real LOB dataframe
    df_gen_list  : list of generated dataframes (length N paths)
    time_index   : when the continuation begins
    """

    # --- Prepare REAL ---
    real = lob_df.copy()
    if not isinstance(real.index, pd.DatetimeIndex):
        if "time" in real.columns:
            real["time"] = pd.to_datetime(real["time"])
            real = real.set_index("time")

    # Drop last timestamp (your note)
    real = real.iloc[:-1]

    # --- Prepare GENERATED paths ---
    dfs = []
    for gen_df in df_gen_list:
        gen = gen_df.copy()
        if not isinstance(gen.index, pd.DatetimeIndex):
            gen["time"] = pd.to_datetime(gen["time"])
            gen = gen.set_index("time")
        dfs.append(gen.iloc[:-1][column])  # keep only price column

    # Stack into matrix: shape = (N_paths, T)
    gen_matrix = np.vstack([g.values for g in dfs])

    # Compute statistics
    median = np.median(gen_matrix, axis=0)
    low_band = np.quantile(gen_matrix, ci_low, axis=0)
    high_band = np.quantile(gen_matrix, ci_high, axis=0)

    # All generated paths share the same time index
    t_gen = dfs[0].index

    # --- Plot ---
    os.makedirs("out/plots", exist_ok=True)

    plt.figure(figsize=(11, 5))

    # Confidence band
    plt.fill_between(t_gen, low_band, high_band, 
                     alpha=0.25, color="blue", label=f"{ci_low*100:.0f}-{ci_high*100:.0f}% band")

    # Median line
    plt.plot(t_gen, median, color="blue", linewidth=1.5, label="Generated median")

    # Plot one random path as well
    # path_idx = np.random.randint(len(df_gen_list))
    # plt.plot(df_gen_list[path_idx].index, df_gen_list[path_idx][column], color="blue", linewidth=0.8)

    # Real L0 line
    plt.plot(real.index, real[column], color="black", linewidth=1.2, label="Real")

    # Transition marker
    plt.axvline(real.index[time_index], color="red", linestyle=":", label="Transition")

    plt.title("Real vs Generated (Confidence Bands)")
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"out/plots/real_vs_generated_conf_bands_{column}.pdf")
    else:
        plt.show()
    plt.close()


def plot_time_series(data, title, xlabel="Time", ylabel="Value", 
                     filename=None, output_dir="out/plots", 
                     print_stats=False, stats_label=None):
    """
    Plot a time series and optionally print statistics.
    
    Args:
        data: 1D array-like data to plot
        title: Plot title
        xlabel: X-axis label (default: "Time")
        ylabel: Y-axis label (default: "Value")
        filename: Output filename (if None, derived from title)
        output_dir: Directory to save the plot (default: "out/plots")
        print_stats: Whether to print statistics (default: False)
        stats_label: Label for statistics output (if None, uses title)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print statistics if requested
    if print_stats:
        label = stats_label if stats_label else title
        print(f"  {label} stats: mean={np.mean(data):.6f}, median={np.median(data):.6f}, "
              f"std={np.std(data):.6f}, min={np.min(data):.6f}, max={np.max(data):.6f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    # Save plot
    if filename is None:
        # Derive filename from title
        filename = title.lower().replace(" ", "_") + ".png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()


def plot_epochs_evolution(metric, metric_name):
    plt.figure(figsize=(11, 5))
    if "distance" in metric_name:
        plt.yscale("log")
    plt.plot(metric)
    plt.title("Epochs evolution")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric_name}")
    plt.tight_layout()
    plt.savefig(f"out/plots/{metric_name}_evolution.pdf")
    plt.close()
