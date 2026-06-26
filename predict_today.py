"""
XAUUSD Daily Predictor - Load saved model and predict for today
Usage: python predict_today.py
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load model + scaler
artifacts = joblib.load("xauusd_model.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_cols = artifacts["feature_cols"]
best_thresh = artifacts["best_thresh"]
forward_days = artifacts.get("forward_days", 3)

# Load latest data
df = pd.read_csv("xauusd_daily.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

close = df["Close"]
high = df["High"]
low = df["Low"]
open_p = df["Open"]
volume = df["Volume"]

# Feature engineering (same as training)
for p in [5, 10, 20, 50, 100, 200]:
    df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
df["RSI_14"] = 100 - (100 / (1 + rs))

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
df["BB_Upper"] = bb_mid + 2 * bb_std
df["BB_Lower"] = bb_mid - 2 * bb_std
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
df["BB_Pct"] = (close - bb_mid) / (2 * bb_std + 1e-10)

tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
df["ATR_14"] = tr.rolling(14).mean()
df["ATR_Pct"] = df["ATR_14"] / close

df["Body"] = abs(close - open_p)
df["Range"] = high - low
df["Return_1d"] = close.pct_change()
df["Return_5d"] = close.pct_change(5)
df["Return_20d"] = close.pct_change(20)
df["High_Low_Ratio"] = (high - low) / close
df["Vol_20"] = close.pct_change().rolling(20).std()
df["Vol_50"] = close.pct_change().rolling(50).std()

for p in [10, 20, 50]:
    df[f"Dist_High_{p}"] = (high.rolling(p).max() - close) / close
    df[f"Dist_Low_{p}"] = (close - low.rolling(p).min()) / close

df["Volume_Change"] = volume.pct_change()
df["Volume_SMA_20"] = volume.rolling(20).mean()
df["Volume_Ratio"] = volume / (df["Volume_SMA_20"] + 1e-10)
df["DayOfWeek"] = df.index.dayofweek
df["Month"] = df.index.month

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Get latest row
last_row = df.iloc[-1:]
if last_row.isnull().any(axis=1).iloc[0]:
    # find the most recent complete row
    clean = df[feature_cols].dropna()
    last_row = clean.iloc[-1:]
    last_date = clean.index[-1]
else:
    last_date = df.index[-1]

features = last_row[feature_cols].values
features_scaled = scaler.transform(features)
prob = model.predict_proba(features_scaled)[0, 1]

thresh = max(best_thresh, 0.55)

print(f"\n{'='*45}")
print("XAUUSD PREDICTOR - DAILY SIGNAL")
print(f"{'='*45}")
print(f"Tanggal         : {last_date.strftime('%d %B %Y')}")
print(f"Close           : ${last_row['Close'].values[0]:.2f}")
print(f"RSI(14)         : {last_row['RSI_14'].values[0]:.1f}")
print(f"MACD            : {last_row['MACD_Hist'].values[0]:.1f}")
print(f"ATR(14)         : ${last_row['ATR_14'].values[0]:.1f}")
print(f"{'='*45}")
print(f"PREDIKSI {forward_days} HARI: ", end="")

if prob >= thresh:
    print(f"BUY (Bullish)  confidence: {prob:.1%}")
    if prob >= 0.75:
        print("Kategori        : SIGNAL KUAT - high probability")
    else:
        print("Kategori        : SIGNAL MODERAT")
else:
    print(f"SELL (Bearish) confidence: {1-prob:.1%}")
    if prob <= 0.25:
        print("Kategori        : SIGNAL KUAT - high probability")
    else:
        print("Kategori        : SIGNAL MODERAT")

print(f"Threshold       : {thresh:.3f}")
print(f"{'='*45}")
print(f"* Sinyal berlaku untuk arah {forward_days} hari ke depan")
print(f"* Hanya trading jika confidence > threshold")
print(f"* Selalu gunakan stop loss dan risk management")
