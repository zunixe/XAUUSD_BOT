import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=== XAUUSD ML PIPELINE ===\n")

# 1. LOAD DATA
df = pd.read_csv("xauusd_daily.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)
print(f"Data: {len(df)} hari ({df.index.min()} - {df.index.max()})")

close = df["Close"]
high = df["High"]
low = df["Low"]
open_p = df["Open"]
volume = df["Volume"]

# 2. FEATURE ENGINEERING
# Moving Averages
for p in [5, 10, 20, 50, 100, 200]:
    df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
df["RSI_14"] = 100 - (100 / (1 + rs))

# MACD
ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# Bollinger Bands
bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
df["BB_Upper"] = bb_mid + 2 * bb_std
df["BB_Lower"] = bb_mid - 2 * bb_std
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
df["BB_Pct"] = (close - bb_mid) / (2 * bb_std + 1e-10)

# ATR
tr = pd.concat([
    high - low,
    abs(high - close.shift(1)),
    abs(low - close.shift(1))
], axis=1).max(axis=1)
df["ATR_14"] = tr.rolling(14).mean()
df["ATR_Pct"] = df["ATR_14"] / close

# Price Action
df["Body"] = abs(close - open_p)
df["Range"] = high - low
df["Return_1d"] = close.pct_change()
df["Return_5d"] = close.pct_change(5)
df["Return_20d"] = close.pct_change(20)
df["High_Low_Ratio"] = (high - low) / close

# Volatility
df["Vol_20"] = close.pct_change().rolling(20).std()
df["Vol_50"] = close.pct_change().rolling(50).std()

# Support/Resistance distance
for p in [10, 20, 50]:
    df[f"Dist_High_{p}"] = (high.rolling(p).max() - close) / close
    df[f"Dist_Low_{p}"] = (close - low.rolling(p).min()) / close

# Volume
df["Volume_Change"] = volume.pct_change()
df["Volume_SMA_20"] = volume.rolling(20).mean()
df["Volume_Ratio"] = volume / (df["Volume_SMA_20"] + 1e-10)

# Time features
df["DayOfWeek"] = df.index.dayofweek
df["Month"] = df.index.month

# 3. TARGET (3-day forward return - less noise)
forward_days = 3
future_close = df["Close"].shift(-forward_days)
df["Target"] = (future_close > df["Close"] * 1.005).astype(int)  # 0.5% move over 3 days

# 4. CLEAN INFINITE VALUES
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Setelah feature engineering: {len(df)} baris, {len(df.columns)} kolom\n")

# 5. SPLIT DATA
feature_cols = [c for c in df.columns if c not in ["Target", "Close", "High", "Low", "Open", "Volume"]]
X = df[feature_cols].values
y = df["Target"].values

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Fitur: {len(feature_cols)}")

# 6. TRAIN XGBOOST
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Scale features ---
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Handle class imbalance ---
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1

model = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.015,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=50,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# --- Evaluate with threshold tuning ---
y_prob = model.predict_proba(X_test)[:, 1]
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
best_thresh = thresholds[np.argmax(f1_scores[:-1])]
y_pred_opt = (y_prob >= best_thresh).astype(int)
y_pred = model.predict(X_test)

print("\n=== XGBoost RESULTS ===")
print(f"Default threshold (0.5):")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"  Precision: {precision_score(y_test, y_pred):.2%}")
print(f"  Recall   : {recall_score(y_test, y_pred):.2%}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.2%}")
print(f"Optimal threshold ({best_thresh:.3f}):")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_opt):.2%}")
print(f"  Precision: {precision_score(y_test, y_pred_opt):.2%}")
print(f"  Recall   : {recall_score(y_test, y_pred_opt):.2%}")
print(f"  F1 Score : {f1_score(y_test, y_pred_opt):.2%}")
print(f"Confusion Matrix (default):")
print(confusion_matrix(y_test, y_pred))

# 7. FEATURE IMPORTANCE
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})

importance = model.feature_importances_
idx = np.argsort(importance)[::-1][:20]
top_features = [feature_cols[i] for i in idx]
top_importance = importance[idx]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(20), top_importance, align="center")
plt.yticks(range(20), top_features)
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("xauusd_feature_importance.png", dpi=150)
plt.close()
print("\nFeature importance chart saved: xauusd_feature_importance.png")

# 8. BACKTEST
bt = df.iloc[-len(y_pred):].copy()
# Gunakan optimal threshold, min 0.55 untuk filtering sinyal lemah
bt_thresh = max(best_thresh, 0.55)
bt["Signal"] = (y_prob >= bt_thresh).astype(int)
bt["Next_Close"] = bt["Close"].shift(-forward_days)
bt["Return"] = bt["Next_Close"] / bt["Close"] - 1
bt["Strat_Return"] = bt["Signal"] * bt["Return"]
bt.dropna(inplace=True)
bt["Cum_Strat"] = (1 + bt["Strat_Return"]).cumprod()
bt["Cum_BH"] = (1 + bt["Return"]).cumprod()

total_trades = int(bt["Signal"].sum())
win_trades = int(((bt["Signal"] == 1) & (bt["Return"] > 0)).sum())
win_rate = win_trades / total_trades if total_trades > 0 else 0
strategy_ret = bt["Cum_Strat"].iloc[-1] - 1 if len(bt) > 0 else 0
bh_ret = bt["Cum_BH"].iloc[-1] - 1 if len(bt) > 0 else 0
max_dd = min((bt["Strat_Return"].cumsum().cummax() - bt["Strat_Return"].cumsum()).min(), 0) if len(bt) > 0 else 0

print(f"\n{'='*45}")
print(f"BACKTEST RESULTS (Threshold: {bt_thresh:.3f})")
print(f"{'='*45}")
print(f"Period         : {bt.index[0].strftime('%Y-%m-%d')} - {bt.index[-1].strftime('%Y-%m-%d')}")
print(f"Total Trades   : {total_trades}")
print(f"Win Rate       : {win_rate:.2%}")
print(f"Strategy Return: {strategy_ret:.2%}")
print(f"Buy & Hold     : {bh_ret:.2%}")
print(f"Max Drawdown   : {max_dd:.2%}")

# 9. SAVE MODEL + SCALER
import joblib
joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols, "best_thresh": best_thresh, "forward_days": forward_days}, "xauusd_model.pkl")
print("\nModel + scaler saved: xauusd_model.pkl")

# 10. PREDIKSI HARI INI
last_features = df[feature_cols].iloc[-1:].values
last_features_scaled = scaler.transform(last_features)
last_prob = model.predict_proba(last_features_scaled)[0, 1]
thresh_used = max(best_thresh, 0.55)
last_pred = "BUY (Bullish)" if last_prob >= thresh_used else "SELL (Bearish)"

print(f"\n{'='*45}")
print("PREDIKSI 3 HARI KE DEPAN")
print(f"{'='*45}")
print(f"Tanggal         : {df.index[-1].strftime('%d %B %Y')}")
print(f"Close Terakhir  : ${df['Close'].iloc[-1]:.2f}")
print(f"Prediksi Arah   : {last_pred}")
print(f"Confidence      : {last_prob:.1%}")
print(f"Threshold       : {thresh_used:.3f}")

if last_prob >= 0.75:
    print("Kekuatan Sinyal : KUAT")
elif last_prob >= thresh_used:
    print("Kekuatan Sinyal : MODERAT")
else:
    print("Kekuatan Sinyal : LEMAH (skip)")
