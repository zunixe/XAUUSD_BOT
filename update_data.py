import yfinance as yf
import pandas as pd

MACRO_TICKERS = {
    "^VIX": ("VIX_Close", "vix_daily.csv"),
    "^GSPC": ("SPY_Close", "spy_daily.csv"),
    "^TNX": ("US10Y_Close", "us10y_daily.csv"),
    "CL=F": ("OIL_Close", "oil_daily.csv"),
    "DX-Y.NYB": ("DXY_Close", "dxy_daily.csv"),
}

# ========== GC=F DAILY (XAUUSD) ==========
print("Downloading latest GC=F daily data...")
df = yf.download("GC=F", period="5d", interval="1d", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df.reset_index(inplace=True)
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print(f"GC=F daily: {df.index[-1].strftime('%Y-%m-%d')} ${df['Close'].iloc[-1]:.2f}")

old = pd.read_csv("xauusd_daily.csv", parse_dates=["Date"]).set_index("Date")
combined = pd.concat([old, df])
combined = combined[~combined.index.duplicated(keep="last")]
combined.sort_index(inplace=True)

# ========== GC=F 4H (incremental) ==========
print("\nDownloading latest GC=F 4H data...")
try:
    df_4h = yf.download("GC=F", period="5d", interval="4h", progress=False)
    if isinstance(df_4h.columns, pd.MultiIndex):
        df_4h.columns = [c[0] for c in df_4h.columns]
    df_4h.reset_index(inplace=True)
    if "Datetime" in df_4h.columns:
        df_4h.rename(columns={"Datetime": "Date"}, inplace=True)
    df_4h.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df_4h["Date"] = pd.to_datetime(df_4h["Date"], utc=True).dt.tz_localize(None)
    df_4h.set_index("Date", inplace=True)
    df_4h.sort_index(inplace=True)

    old_4h = pd.read_csv("xauusd_4h.csv", parse_dates=["Date"])
    old_4h["Date"] = pd.to_datetime(old_4h["Date"], utc=True).dt.tz_localize(None)
    old_4h.set_index("Date", inplace=True)
    combined_4h = pd.concat([old_4h, df_4h])
    combined_4h = combined_4h[~combined_4h.index.duplicated(keep="last")]
    combined_4h.sort_index(inplace=True)
    combined_4h.to_csv("xauusd_4h.csv")
    print(f"GC=F 4H: {len(combined_4h)} candles, last: {combined_4h.index[-1]}")
    print(f"Last close: {combined_4h['Close'].iloc[-1]:.2f}")
except Exception as e:
    import traceback
    print(f"4H download error: {e}")
    traceback.print_exc()

# ========== MACRO TICKERS (incremental) ==========
for ticker, (colname, csv_file) in MACRO_TICKERS.items():
    print(f"\nDownloading {ticker}...")
    try:
        new = yf.download(ticker, period="5d", interval="1d", progress=False)
        if isinstance(new.columns, pd.MultiIndex):
            new.columns = [c[0] for c in new.columns]
        new = new[["Close"]]
        new.columns = [colname]
        new.sort_index(inplace=True)
        print(f"  Last: {new.index[-1].strftime('%Y-%m-%d')} {new[colname].iloc[-1]:.2f}")

        # Merge with existing file
        try:
            csv_old = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        except FileNotFoundError:
            csv_old = pd.DataFrame()
        all_data = pd.concat([csv_old, new])
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        all_data.sort_index(inplace=True)
        all_data.to_csv(csv_file)

        # Merge into combined gold CSV
        combined[colname] = all_data[colname]
    except Exception as e:
        print(f"  ERROR: {e}")

# Forward fill all macro columns
for c in ["VIX_Close", "SPY_Close", "US10Y_Close", "OIL_Close", "DXY_Close"]:
    if c in combined.columns:
        combined[c] = combined[c].ffill()

combined.to_csv("xauusd_daily.csv")
print(f"\nXAUUSD saved: {len(combined)} rows")
print(f"Coverage: { {c:int(combined[c].notna().sum()) for c in ['VIX_Close','SPY_Close','US10Y_Close','OIL_Close','DXY_Close']} }")
