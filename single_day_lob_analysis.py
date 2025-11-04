"""
Single-day LOB analysis from structured VAR paths.

Path layout:
  ROOT / VENUE / YYYYMMDD / SYMBOL.var
Example:
  ./Data/var/NASDAQ/20191001/MSFT.var

What this does:
- Loads one .var (via your utils.extractDataFramesFromVar)
- Filters to regular trading hours (default 10:00–15:00)
- Computes: midprice, spread, per-level volumes L0..L4, top-N depth + imbalance
- Plots: time series, histograms (L0)
- Plots: a order-book depth chart at one snapshot (stairs + fills)

Usage:
  python single_day_lob_analysis.py --root ./Data/var --venue NASDAQ \
    --date 20191001 --symbol FLEX --start 10:00 --end 15:00 \
    --out out/FLEX_20191001 --topN 5 --zip
"""

import sys, os, types, argparse, zipfile, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Minimal fallback so utils.py can import SortedDict ----------
sortedcontainers = types.ModuleType("sortedcontainers")
class SortedDict(dict):
    def items(self):
        for k in sorted(super().keys()):
            yield (k, super().__getitem__(k))
sortedcontainers.SortedDict = SortedDict
sys.modules["sortedcontainers"] = sortedcontainers
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Single-day LOB analysis from VAR files")
    p.add_argument("--root",   required=True, help="Root folder (e.g., ./Data/var)")
    p.add_argument("--venue",  required=True, help="Venue/exchange folder (e.g., NASDAQ)")
    p.add_argument("--date",   required=True, help="Trading date YYYYMMDD (e.g., 20191001)")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g., MSFT)")
    p.add_argument("--start",  default="10:00", help="Session start HH:MM (default: 10:00)")
    p.add_argument("--end",    default="15:00", help="Session end HH:MM (default: 15:00)")
    p.add_argument("--topN",   type=int, default=5, help="Depth levels to use (default: 5 → L0..L4)")
    p.add_argument("--out",    default=None, help="Output directory (default: out/<symbol>_<date>)")
    p.add_argument("--snapshot", default=None,
                   help="Snapshot time HH:MM:SS for depth chart (default: median timestamp of session)")
    p.add_argument("--no-zip", dest="zip_outputs", action="store_false", help="Disable zipping outputs")
    p.set_defaults(zip_outputs=True)
    return p.parse_args()

def build_var_path(root, venue, yyyymmdd, symbol):
    return os.path.join(root, venue, yyyymmdd, f"{symbol}.var")

def filter_between(df, start_t, end_t):
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df
    out = df.between_time(start_t, end_t).copy()
    if not out.empty and "time_elapsed" in out.columns:
        first_t = float(out.iloc[0]["time_elapsed"])
        out.loc[:, "time_elapsed"] = out["time_elapsed"].astype(float) - first_t
    return out

def save_ts_plot(series_or_df, title, ylabel, fname, out_dir):
    plt.figure()
    if isinstance(series_or_df, pd.Series):
        series_or_df.plot()
    else:
        series_or_df.plot()
    plt.title(title); plt.xlabel("Time"); plt.ylabel(ylabel)
    path = os.path.join(out_dir, fname)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

def save_hist(series, bins, title, xlabel, fname, out_dir):
    plt.figure()
    series.dropna().plot(kind="hist", bins=bins)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    path = os.path.join(out_dir, fname)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

# ---------- Pretty order-book depth chart (stairs + filled areas) ----------
def _depth_arrays_from_row(row, topN):
    """
    From an L2 row, build price & cumulative qty arrays for bid and ask sides.
    Returns (bid_px, bid_cum, ask_px, ask_cum). Rows already numeric.
    """
    bid_pairs, ask_pairs = [], []
    for i in range(topN):
        bpx = row.get(f"bidPx_{i}"); bq = row.get(f"bidQty_{i}")
        apx = row.get(f"askPx_{i}"); aq = row.get(f"askQty_{i}")
        if pd.notna(bpx) and pd.notna(bq): bid_pairs.append((float(bpx), float(bq)))
        if pd.notna(apx) and pd.notna(aq): ask_pairs.append((float(apx), float(aq)))

    # Sort bids descending by price, asks ascending by price
    bid_pairs.sort(key=lambda x: -x[0])
    ask_pairs.sort(key=lambda x:  x[0])

    # Cumulative quantities
    bid_px, bid_cum = [], []
    cum = 0.0
    for px, q in bid_pairs:
        cum += q; bid_px.append(px); bid_cum.append(cum)

    ask_px, ask_cum = [], []
    cum = 0.0
    for px, q in ask_pairs:
        cum += q; ask_px.append(px); ask_cum.append(cum)

    return bid_px, bid_cum, ask_px, ask_cum

def save_order_book_depth_chart(lob_df, snapshot_ts, topN, out_path):
    """
    Draw a classic depth chart at a single snapshot time:
      - step lines for bids/asks
      - filled areas under each side
      - vertical lines for mid, best bid, best ask
    """
    row = lob_df.loc[snapshot_ts]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    bid_px, bid_cum, ask_px, ask_cum = _depth_arrays_from_row(row, topN)
    if not bid_px or not ask_px:
        raise RuntimeError("Not enough L2 levels to build a depth chart at the chosen time.")

    mid = (row["bidPx_0"] + row["askPx_0"]) / 2.0

    def stairs_xy(px_list, cum_list):
        x, y = [], []
        for i, (px, cy) in enumerate(zip(px_list, cum_list)):
            if i == 0:
                x.extend([px, px]); y.extend([0.0, cy])
            else:
                x.extend([px, px]); y.extend([y[-1], cy])
        return np.array(x), np.array(y)

    bx, by = stairs_xy(bid_px, bid_cum)
    ax, ay = stairs_xy(ask_px, ask_cum)

    plt.figure(figsize=(9, 6))
    plt.step(bx, by, where="post", linewidth=1.6, label="Bid depth")
    plt.step(ax, ay, where="post", linewidth=1.6, label="Ask depth")
    plt.fill_between(bx, 0, by, step="post", alpha=0.25)
    plt.fill_between(ax, 0, ay, step="post", alpha=0.25)
    plt.axvline(float(row["bidPx_0"]), linestyle="--", linewidth=1.2, label="Best Bid")
    plt.axvline(float(row["askPx_0"]), linestyle="--", linewidth=1.2, label="Best Ask")
    plt.axvline(float(mid),            linestyle=":",  linewidth=1.2, label="Mid")
    plt.title(f"Order Book Depth @ {snapshot_ts.time()}  (Top {topN})")
    plt.xlabel("Price"); plt.ylabel("Cumulative size")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return out_path

# ---------- Depth charts with respect to time ----------
def save_relative_level_heatmap(lob_df, topN, out_path):
    """
    Heatmap over time for per-level quantities aligned by *relative level index*.
    Columns: [-topN..-1] for bids (L0 closest to mid is -1), [1..topN] for asks.
    Value: raw quantity at that level (not cumulative).
    """
    times = lob_df.index.to_pydatetime()
    nT = len(times)
    cols = list(range(-topN, 0)) + list(range(1, topN + 1))
    mat = np.full((nT, len(cols)), np.nan, dtype=float)

    for t_i, (_, row) in enumerate(lob_df.iterrows()):
        for i in range(topN):
            bq = row.get(f"bidQty_{i}")
            aq = row.get(f"askQty_{i}")
            if pd.notna(bq): mat[t_i, topN - (i+1)] = float(bq)        # map L0->-1 at last of bid block
            if pd.notna(aq): mat[t_i, topN + i]     = float(aq)        # map L0->+1 at first of ask block

    plt.figure(figsize=(11, 5))
    # imshow, time on x, level on y → transpose to (levels x time)
    img = plt.imshow(mat.T, aspect="auto", interpolation="nearest",
                     origin="lower")
    plt.colorbar(img, fraction=0.046, pad=0.04, label="Qty")
    plt.yticks(ticks=np.arange(len(cols)), labels=[str(c) for c in cols])
    # X ticks: sparse times
    xticks = np.linspace(0, nT-1, num=min(8, nT), dtype=int)
    xticklabels = [lob_df.index[i].strftime("%H:%M") for i in xticks]
    plt.xticks(xticks, xticklabels, rotation=0)
    plt.title(f"Depth by Relative Level over Time (Top {topN})")
    plt.xlabel("Time"); plt.ylabel("Level index (−=bid, +=ask)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return out_path

def save_cumulative_depth_heatmap(lob_df, side, topN, out_path):
    """
    Heatmap over time for CUMULATIVE depth by level on a given side ('bid' or 'ask').
    Rows (y): level index 0..topN-1; Columns (x): time; Values: cumulative qty up to that level.
    """
    times = lob_df.index.to_pydatetime()
    nT = len(times)
    mat = np.full((topN, nT), np.nan, dtype=float)

    qty_prefix = "bidQty_" if side == "bid" else "askQty_"
    for t_i, (_, row) in enumerate(lob_df.iterrows()):
        cum = 0.0
        for i in range(topN):
            q = row.get(f"{qty_prefix}{i}")
            if pd.notna(q):
                cum += float(q)
                mat[i, t_i] = cum

    plt.figure(figsize=(11, 5))
    img = plt.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")
    plt.colorbar(img, fraction=0.046, pad=0.04, label="Cumulative qty")
    yticks = np.arange(topN)
    plt.yticks(yticks, [f"L{i}" for i in yticks])
    xticks = np.linspace(0, nT-1, num=min(8, nT), dtype=int)
    xticklabels = [lob_df.index[i].strftime("%H:%M") for i in xticks]
    plt.xticks(xticks, xticklabels)
    plt.title(f"{side.capitalize()} Cumulative Depth by Level over Time (Top {topN})")
    plt.xlabel("Time"); plt.ylabel("Level")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return out_path

def zip_outputs(out_dir, zip_name=None):
    if zip_name is None:
        zip_name = os.path.basename(os.path.normpath(out_dir)) + ".zip"
    zip_path = os.path.join(os.path.dirname(out_dir), zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for png in glob.glob(os.path.join(out_dir, "*.png")):
            zf.write(png, arcname=os.path.basename(png))
    return zip_path

def main():
    args = parse_args()
    var_path = build_var_path(args.root, args.venue, args.date, args.symbol)
    if not os.path.isfile(var_path):
        raise FileNotFoundError(f"Could not find VAR file at: {var_path}")

    # import your loader
    sys.path.append(os.path.dirname(var_path))
    sys.path.append(os.getcwd())
    from utils import extractDataFramesFromVar

    df_map = extractDataFramesFromVar(var_path)
    # Visualize how the data looks like after extraction
    # DF DESCRIPTION
    # 1) AGG_TRADE
    #    contains all trades by timestamp, with buy / sell (if market ?) or limit order details
    # 2) BBO
    #    contains best bid and best ask by timestamp for the selected day
    # 3) BBO_DELTA
    #    contains best bid and best ask deltas by timestamp for the selected day
    # 4) MICRO_PX
    #    no idea of what this is
    # 5) OFI
    #    order fill? or similar
    # 6) ORDER_SUMMARY
    #    contains details of added / deleted orders
    # 7) L2_SNAPSHOT
    #    level 5 (lol) order book details. Apparently there are some changes in the positions 
    #    (either queues sizes or prices) even if the timestamp is exactly! the same. Also bid / ask levels seem equally spaced
    #    between timestamps, which is suspicious. To check:
    #    plt.figure(); plt.plot(df_map["L2_SNAPSHOT"]["askPx_0"].astype(float), label="0"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_1"].astype(float), label="1"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_2"].astype(float), label="2"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_3"].astype(float), label="3"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_4"].astype(float), label="4"); plt.legend(); plt.show()

    # In general, check for artifacts (weird values ex. price falling back to opening price, etc.) and general cleaness of the dataset
    # Check extractDataFramesFromVar to see how it manipulates it - also there were some timestamp matches in the var files
 
    #print(df_map)

    # Export all dataframes for later reuse
    os.makedirs(f"out/data/{args.date}", exist_ok=True)
    for k, v in df_map.items():
        v.to_csv(f"out/data/{args.date}/FLEX_{k}.csv", index=False)

    filtered = {k: filter_between(v, args.start, args.end) for k, v in df_map.items()}

    lob = filtered.get("L2_SNAPSHOT", pd.DataFrame())
    if lob.empty:
        raise RuntimeError("No L2_SNAPSHOT data found in the selected time window.")

    # Ensure numeric for price/qty columns we care about
    for c in lob.columns:
        if ("Px_" in c) or ("Qty_" in c) or (c in ["bidPx_0","askPx_0","bidQty_0","askQty_0"]):
            with pd.option_context('mode.chained_assignment', None):
                lob[c] = pd.to_numeric(lob[c], errors="coerce")

    # Metrics
    metrics = {}
    if {"bidPx_0","askPx_0"}.issubset(lob.columns):
        metrics["midprice"] = (lob["bidPx_0"] + lob["askPx_0"]) / 2.0
        metrics["spread"]   = (lob["askPx_0"] - lob["bidPx_0"]).abs()

    def side_levels(prefix, N):
        cols = [f"{prefix}Qty_{i}" for i in range(N) if f"{prefix}Qty_{i}" in lob.columns]
        if not cols: return pd.DataFrame(index=lob.index)
        d = lob[cols].copy()
        d.columns = [f"{prefix}L{i}" for i in range(len(cols))]
        return d

    bid_levels = side_levels("bid", args.topN)
    ask_levels = side_levels("ask", args.topN)

    if not bid_levels.empty and not ask_levels.empty:
        metrics["bid_depth_topN"] = bid_levels.sum(axis=1)
        metrics["ask_depth_topN"] = ask_levels.sum(axis=1)
        denom = (metrics["bid_depth_topN"] + metrics["ask_depth_topN"]).replace(0, np.nan)
        metrics["imbalance_topN"] = (metrics["bid_depth_topN"] - metrics["ask_depth_topN"]) / denom

    metrics_df = pd.DataFrame(metrics)

    # Output dir
    out_dir = args.out or os.path.join("out", f"{args.symbol}_{args.date}")
    os.makedirs(out_dir, exist_ok=True)

    # Standard plots
    outputs = []
    if "midprice" in metrics:
        outputs.append(save_ts_plot(pd.Series(metrics["midprice"], index=lob.index),
                                    "Midprice over time", "Price", "midprice.png", out_dir))
    if "spread" in metrics:
        outputs.append(save_ts_plot(pd.Series(metrics["spread"], index=lob.index),
                                    "Bid-ask spread over time", "Spread", "spread.png", out_dir))
    if not bid_levels.empty:
        outputs.append(save_ts_plot(bid_levels, f"Bid L0..L{args.topN-1} volumes",
                                    "Quantity", "bid_levels.png", out_dir))
    if not ask_levels.empty:
        outputs.append(save_ts_plot(ask_levels, f"Ask L0..L{args.topN-1} volumes",
                                    "Quantity", "ask_levels.png", out_dir))
    if {"bid_depth_topN","ask_depth_topN","imbalance_topN"}.issubset(metrics.keys()):
        depth_df = pd.DataFrame({
            "Bid depth (topN)": metrics["bid_depth_topN"],
            "Ask depth (topN)": metrics["ask_depth_topN"]
        }, index=lob.index)
        outputs.append(save_ts_plot(depth_df, "Top-N depth (bid vs ask)",
                                    "Quantity", "topN_depth.png", out_dir))
        outputs.append(save_ts_plot(pd.Series(metrics["imbalance_topN"], index=lob.index),
                                    "Top-N order book imbalance", "Imbalance",
                                    "imbalance.png", out_dir))
    if "bidQty_0" in lob.columns:
        outputs.append(save_hist(pd.to_numeric(lob["bidQty_0"], errors="coerce"),
                                 bins=40, title="Histogram of L0 bid quantities",
                                 xlabel="Qty", fname="hist_bidL0_qty.png", out_dir=out_dir))
    if "askQty_0" in lob.columns:
        outputs.append(save_hist(pd.to_numeric(lob["askQty_0"], errors="coerce"),
                                 bins=40, title="Histogram of L0 ask quantities",
                                 xlabel="Qty", fname="hist_askL0_qty.png", out_dir=out_dir))

    # Depth chart (one polished snapshot)
    if len(lob.index) > 0:
        if args.snapshot:
            target = pd.to_datetime(args.snapshot).time()
            diffs = np.array([abs((pd.Timestamp.combine(pd.Timestamp.today(), t) -
                                   pd.Timestamp.combine(pd.Timestamp.today(), target)).total_seconds())
                              for t in lob.index.time])
            ts = lob.index[int(diffs.argmin())]
        else:
            ts = lob.index[int(len(lob.index) // 2)]  # median snapshot
        depth_path = os.path.join(out_dir, "order_book_depth.png")
        outputs.append(save_order_book_depth_chart(lob, ts, args.topN, depth_path))

    # NEW: time-evolving depth charts
    outputs.append(save_relative_level_heatmap(lob, args.topN,
                                               os.path.join(out_dir, "depth_relative_levels_heatmap.png")))
    outputs.append(save_cumulative_depth_heatmap(lob, "bid", args.topN,
                                                 os.path.join(out_dir, "depth_cumulative_bid_heatmap.png")))
    outputs.append(save_cumulative_depth_heatmap(lob, "ask", args.topN,
                                                 os.path.join(out_dir, "depth_cumulative_ask_heatmap.png")))

    # Zip if requested
    if args.zip_outputs:
        zip_path = zip_outputs(out_dir)
        print("Zipped outputs:", zip_path)

    print("Saved plots:")
    for p in outputs:
        print("-", p)

if __name__ == "__main__":
    main()
