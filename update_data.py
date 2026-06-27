import yfinance as yf
import pandas as pd
import numpy as np
import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load config
with open(os.path.join(BASE_DIR, "config.yaml")) as f:
    CFG = yaml.safe_load(f)


def _rename_cols(df, cols):
    """Safely rename columns by position after flattening MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
    existing = list(df.columns)
    if len(existing) >= len(cols):
        df.columns = cols[:len(existing)] + list(df.columns[len(cols):])
    return df


def _download_macro_ticker(ticker, colname, csv_path):
    """Download and merge a single macro ticker."""
    print(f"  {ticker}...", end=" ")
    try:
        new = yf.download(ticker, period="60d", interval="1d", progress=False)
        if isinstance(new.columns, pd.MultiIndex):
            new.columns = [c[0] for c in new.columns]
        new = new[["Close"]].copy()
        new.columns = [colname]
        new.sort_index(inplace=True)
        print(f"{new.index[-1].strftime('%Y-%m-%d')} {new[colname].iloc[-1]:.2f}")

        try:
            csv_old = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        except FileNotFoundError:
            csv_old = pd.DataFrame()
        all_data = pd.concat([csv_old, new])
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        all_data.sort_index(inplace=True)
        all_data.to_csv(csv_path)
        return all_data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def _download_fred_series(series_id, colname, csv_path):
    """Download FRED series data via direct CSV URL (no API key needed)."""
    print(f"  FRED:{series_id}...", end=" ")
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        new = pd.read_csv(url)
        # FRED CSVs have first column as date (various names: DATE, observation_date, etc.)
        date_col = new.columns[0]
        new[date_col] = pd.to_datetime(new[date_col])
        new.set_index(date_col, inplace=True)
        new.columns = [colname]
        new = new[pd.to_numeric(new[colname], errors="coerce").notna()]
        new[colname] = new[colname].astype(float)
        new = new[new[colname] > 0]
        new.sort_index(inplace=True)
        print(f"{new.index[-1].strftime('%Y-%m-%d')} {new[colname].iloc[-1]:.2f}")

        try:
            csv_old = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        except FileNotFoundError:
            csv_old = pd.DataFrame()
        all_data = pd.concat([csv_old, new])
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        all_data.sort_index(inplace=True)
        all_data.to_csv(csv_path)
        return all_data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def run():
    # ========== GC=F DAILY (XAUUSD) ==========
    print("Downloading GC=F daily...")
    df = yf.download("GC=F", period="60d", interval="1d", progress=False)
    df = _rename_cols(df, ["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    print(f"  Last: {df.index[-1].strftime('%Y-%m-%d')} ${df['Close'].iloc[-1]:.2f}")

    daily_csv = os.path.join(BASE_DIR, "xauusd_daily.csv")
    try:
        old = pd.read_csv(daily_csv, parse_dates=["Date"]).set_index("Date")
        if old.empty:
            combined = df
        else:
            combined = pd.concat([old, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
    except FileNotFoundError:
        combined = df

    # ========== GC=F 4H (incremental) ==========
    print("\nDownloading GC=F 4H...")
    try:
        df_4h = yf.download("GC=F", period="60d", interval="4h", progress=False)
        if isinstance(df_4h.columns, pd.MultiIndex):
            df_4h.columns = [c[0] for c in df_4h.columns]
        df_4h.reset_index(inplace=True)
        if "Datetime" in df_4h.columns:
            df_4h.rename(columns={"Datetime": "Date"}, inplace=True)
        df_4h = df_4h[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df_4h["Date"] = pd.to_datetime(df_4h["Date"]).dt.tz_localize(None)
        df_4h.set_index("Date", inplace=True)
        df_4h.sort_index(inplace=True)

        csv_4h = os.path.join(BASE_DIR, "xauusd_4h.csv")
        try:
            old_4h = pd.read_csv(csv_4h, parse_dates=["Date"])
            old_4h["Date"] = pd.to_datetime(old_4h["Date"]).dt.tz_localize(None)
            old_4h.set_index("Date", inplace=True)
            combined_4h = pd.concat([old_4h, df_4h])
            combined_4h = combined_4h[~combined_4h.index.duplicated(keep="last")]
            combined_4h.sort_index(inplace=True)
        except FileNotFoundError:
            combined_4h = df_4h
        combined_4h.to_csv(csv_4h)
        print(f"  4H: {len(combined_4h)} candles, last: {combined_4h.index[-1]}")
    except Exception as e:
        print(f"  4H error: {e}")

    # ========== MACRO TICKERS (from config) ==========
    print("\nMacro tickers:")
    macro_cols = []
    for ticker, (colname, csv_file) in CFG["data"]["macro_tickers"].items():
        csv_path = os.path.join(BASE_DIR, csv_file)
        result = _download_macro_ticker(ticker, colname, csv_path)
        if result is not None:
            combined[colname] = result[colname]
            macro_cols.append(colname)

    # ========== FRED SERIES ==========
    print("\nFRED series:")
    fred_series = {
        "T5YIE": ("Breakeven_5Y", "breakeven_5y.csv"),
        "T10YIE": ("Breakeven_10Y", "breakeven_10y.csv"),
        "GPRD": ("GPR_Index", "gpr_daily.csv"),
    }
    for series_id, (colname, csv_file) in fred_series.items():
        csv_path = os.path.join(BASE_DIR, csv_file)
        result = _download_fred_series(series_id, colname, csv_path)
        if result is not None:
            combined[colname] = result[colname]
            macro_cols.append(colname)

    # ========== COT DATA ==========
    print("\nCOT data:")
    try:
        from cot_data import download_cot, load_cot_features
        download_cot()
        cot_features = load_cot_features()
        if cot_features is not None:
            for col in cot_features.columns:
                combined[col] = cot_features[col]
                macro_cols.append(col)
            print(f"  COT features merged: {list(cot_features.columns)}")
    except Exception as e:
        print(f"  COT error: {e}")

    # Forward fill all macro columns
    all_macro = ["VIX_Close", "SPY_Close", "US10Y_Close", "OIL_Close", "DXY_Close",
                 "TIP_Close", "Silver_Close", "BTC_Close", "USDJPY_Close", "EURUSD_Close",
                 "Copper_Close", "Breakeven_5Y", "Breakeven_10Y", "GPR_Index",
                 "Spec_Net_Long_Pct", "Spec_Positioning_Change", "Commercial_Net_Short_Pct"]
    for col in all_macro:
        if col in combined.columns:
            combined[col] = combined[col].ffill()

    combined.to_csv(daily_csv)
    present = [c for c in all_macro if c in combined.columns]
    coverage = {c: int(combined[c].notna().sum()) for c in present}
    print(f"\nXAUUSD saved: {len(combined)} rows, {len(present)} macro columns")
    print(f"Coverage: {coverage}")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    run()
