"""Bulk download 4H XAUUSD data"""
import yfinance as yf
import pandas as pd

print("Downloading GC=F 4H data...")
df = yf.download("GC=F", period="730d", interval="4h", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df.reset_index(inplace=True)
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
df.to_csv("xauusd_4h.csv")
print(f"4H data: {len(df)} candles")
print(f"Range: {df.index[0]} - {df.index[-1]}")
print(f"Last close: ${df['Close'].iloc[-1]:.2f}")
