import pandas as pd
from datetime import datetime
import torch

MARKET_DEPTH = 3


def extractDataFramesFromVar(filePath):
    """
    Extract dataframes for each type from a VAR file.
    """

    df_map = {}

    with open(filePath) as f:
        rows = []
        headers = []
        is_end_of_headers = False

        for line in f.readlines():
            if is_end_of_headers:
                rows.append([x.strip() for x in line.split(',')])
            elif "END_OF_HEADERS" in line:
                is_end_of_headers = True
            else:
                headers.append([x.strip() for x in line.split(',')])

    # Build DataFrames for each header
    for header in headers:
        df_type = header[0]
        df_rows = []

        for row in rows:
            if df_type in row:
                df_rows.append(row[1:])

        if df_rows:
            df_map[df_type] = pd.DataFrame(df_rows, columns=header[1:])
            df_map[df_type] = df_map[df_type].infer_objects()

            # Robust datetime parsing
            if 'time' in df_map[df_type].columns:
                if 'date' in df_map[df_type].columns:
                    datestr = df_map[df_type]['date'].astype(str).str.strip()
                    timestr = df_map[df_type]['time'].astype(str).str.strip()

                    dt = pd.to_datetime(datestr + " " + timestr, errors="coerce")

                    # Fallback 1: standard datetime format
                    mask = dt.isna()
                    if mask.any():
                        dt2 = pd.to_datetime(
                            datestr[mask] + " " + timestr[mask],
                            format="%Y-%m-%d %H:%M:%S",
                            errors="coerce"
                        )
                        dt[mask] = dt2

                    # Fallback 2: pad missing fractional seconds
                    mask = dt.isna()
                    if mask.any():
                        padded = timestr[mask].where(
                            timestr[mask].str.contains(r"\."), 
                            timestr[mask] + ".000"
                        )
                        dt3 = pd.to_datetime(datestr[mask] + " " + padded, errors="coerce")
                        dt[mask] = dt3
                else:
                    # Only a 'time' column present
                    timestr = df_map[df_type]['time'].astype(str).str.strip()
                    dt = pd.to_datetime(timestr, errors="coerce")

                    # Fallbacks if needed
                    mask = dt.isna()
                    if mask.any():
                        dt2 = pd.to_datetime(timestr[mask], format="%H:%M:%S", errors="coerce")
                        dt[mask] = dt2

                    mask = dt.isna()
                    if mask.any():
                        padded = timestr[mask].where(
                            timestr[mask].str.contains(r"\."), 
                            timestr[mask] + ".000"
                        )
                        dt3 = pd.to_datetime(padded, errors="coerce")
                        dt[mask] = dt3

                # Add datetime columns
                df_map[df_type]['datetime'] = dt
                df_map[df_type]['date'] = dt.dt.date
                df_map[df_type]['time'] = dt.dt.time
                df_map[df_type]['time_elapsed'] = (
                    (dt - dt.dt.normalize()) / pd.Timedelta('1 second')
                ).astype(float)
                df_map[df_type].set_index('datetime', inplace=True)

    return df_map



def _random_side_swap(current_s, p=0.5):
    """
    current_s: (N, 2D) CUDA tensor
      [ bids (neg, far→near) | asks (pos, near→far) ]

    returns: (N, 2D) CUDA tensor with same canonical structure
    """

    device = current_s.device
    N, _ = current_s.shape
    D = MARKET_DEPTH

    # row-wise Bernoulli
    flip = (torch.rand(N, device=device) < p).unsqueeze(1)  # (N,1)

    bids = current_s[:, :D]      # neg, far → near
    asks = current_s[:, D:]      # pos, near → far

    # side swap + sign flip + depth reversal
    swapped = torch.cat(
        (
            -asks.flip(dims=[1]),  # asks → bids (far → near)
            -bids.flip(dims=[1])   # bids → asks (near → far)
            # -bids,
            # -asks
        ),
        dim=1
    )

    out = torch.where(flip, swapped, current_s)

    # # Debug print
    # print("Before:")
    # print(current_s)
    # print("After:")
    # print(out)

    return out

def _get_next_state(x_state: torch.Tensor):
    """
    x_state: (N, T) CUDA tensor
    returns: (N, 2*MARKET_DEPTH) CUDA tensor
    """

    device = x_state.device
    N, T = x_state.shape

    # print(x_state)

    # 1. detect sign changes
    sign = torch.sign(x_state)
    sign_change = sign[:, :-1] * sign[:, 1:] < 0     # (N, T-1) bool

    # 2. choose pivot = sign change closest to center
    center = T // 2
    idxs = torch.arange(T - 1, device=device).unsqueeze(0)  # (1, T-1)

    # distance from center, invalid positions masked with +inf
    dist = torch.abs(idxs - center)
    dist = torch.where(sign_change, dist, torch.full_like(dist, T))

    pivot = dist.argmin(dim=1) + 1    # (N,)

    # if no sign change → fallback to center
    no_change = sign_change.sum(dim=1) == 0
    pivot = torch.where(no_change, torch.full_like(pivot, center), pivot)

    # 3. build window indices
    offsets = torch.arange(-MARKET_DEPTH, MARKET_DEPTH, device=device)  # (2D,)
    window_idx = pivot.unsqueeze(1) + offsets.unsqueeze(0)              # (N, 2D)

    # clamp to valid range
    window_idx = window_idx.clamp(0, T - 1)

    # 4. gather
    batch_idx = torch.arange(N, device=device).unsqueeze(1)
    current_s = x_state[batch_idx, window_idx]   # (N, 2*MARKET_DEPTH)

    # print("  Previous state:", x_state[0:3, :])
    # print("  Current state:", current_s[0:3, :])

    return current_s