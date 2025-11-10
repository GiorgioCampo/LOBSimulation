import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_real_vs_generated(lob_df, df_gen, save=True):
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

    plt.figure()
    plt.plot(real.index, real["bidPx_0"], label="Real")
    plt.plot(gen.index, gen["bidPx_0"], label="Generated", linestyle="--")
    plt.axvline(real.index[-1], color="red", linestyle=":", label="Transition")
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