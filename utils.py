import pandas as pd


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
