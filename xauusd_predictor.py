import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# XAUUSD PREDICTOR - XGBoost + Technical Indicators
# ============================================================

# ---------------------
# 1. AMBIL DATA
# ---------------------
def fetch_xauusd_data(source="csv", filepath="xauusd_daily.csv"):
    """
    Bisa dari CSV atau download via library yfinance / dukascopy-node
    Format minimal: Date, Open, High, Low, Close, Volume
    """
    if source == "csv":
        df = pd.read_csv(filepath, parse_dates=["Date"])
    else:
        # Contoh: download dari Investing.com via pandas
        # pip install yfinance
        import yfinance as yf
        df = yf.download("GC=F", start="2010-01-01", end="2026-06-23")
        df.reset_index(inplace=True)

    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df

# ---------------------
# 2. FEATURE ENGINEERING
# ---------------------
def add_technical_features(df):
    """
    Tambah fitur teknikal: RSI, MACD, BB, EMA, ATR, dll
    """
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"] if "Volume" in data else None

    # --- Moving Averages ---
    for period in [5, 10, 20, 50, 100, 200]:
        data[f"EMA_{period}"] = close.ewm(span=period, adjust=False).mean()
        data[f"SMA_{period}"] = close.rolling(period).mean()

    # --- RSI ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # --- Bollinger Bands ---
    bb_period = 20
    bb_std = 2
    bb_mid = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    data["BB_Upper"] = bb_mid + bb_std * bb_std_val
    data["BB_Lower"] = bb_mid - bb_std * bb_std_val
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / bb_mid
    data["BB_Position"] = (close - bb_mid) / (data["BB_Width"] * bb_mid + 1e-10)

    # --- ATR (Average True Range) ---
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    data["ATR_14"] = tr.rolling(14).mean()

    # --- Price Action ---
    data["Body"] = abs(close - data["Open"])
    data["Upper_Shadow"] = high - close.where(close > data["Open"], data["Open"])
    data["Lower_Shadow"] = close.where(close < data["Open"], data["Open"]) - low
    data["Range"] = high - low
    data["Return_1d"] = close.pct_change()
    data["High_Low_Ratio"] = (high - low) / close

    # --- Volatility ---
    data["Volatility_20"] = close.pct_change().rolling(20).std()
    data["Volatility_50"] = close.pct_change().rolling(50).std()

    # --- Jarak dari Support/Resistance dinamis ---
    for period in [10, 20, 50]:
        data[f"Dist_Recent_High_{period}"] = (high.rolling(period).max() - close) / close
        data[f"Dist_Recent_Low_{period}"] = (close - low.rolling(period).min()) / close

    # --- Volume features ---
    if volume is not None:
        data["Volume_Change"] = volume.pct_change()
        data["Volume_SMA_20"] = volume.rolling(20).mean()
        data["Volume_Ratio"] = volume / (data["Volume_SMA_20"] + 1e-10)

    # --- Seasonal / Time features ---
    data["DayOfWeek"] = data.index.dayofweek
    data["Month"] = data.index.month
    data["Quarter"] = data.index.quarter
    data["DayOfMonth"] = data.index.day

    return data

# ---------------------
# 3. TARGET LABEL
# ---------------------
def create_target(df, forward_period=1, target_type="direction"):
    """
    target_type:
      - "direction" : 1 = naik, 0 = turun (binary classification)
      - "regression": harga future close
    """
    data = df.copy()
    future_price = data["Close"].shift(-forward_period)
    current_price = data["Close"]

    if target_type == "direction":
        data["Target"] = (future_price > current_price * 1.001).astype(int)  # threshold 0.1%
    elif target_type == "regression":
        data["Target"] = future_price / current_price - 1  # return %
    else:
        raise ValueError("target_type must be 'direction' or 'regression'")

    return data

# ---------------------
# 4. PREPARE DATASET
# ---------------------
def prepare_dataset(df, sequence_length=0, test_size=0.2):
    """
    Untuk XGBoost: test_size = 0.2
    Untuk LSTM: set sequence_length > 0
    """
    data = df.dropna().copy()
    feature_cols = [c for c in data.columns if c not in ["Target", "Close", "Open", "High", "Low", "Volume"]]

    X = data[feature_cols].values
    y = data["Target"].values

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if sequence_length > 0:
        # Untuk LSTM/GRU: return sequence
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length:i])
            y_seq.append(y[i])
        split = int(len(X_seq) * (1 - test_size))
        return (np.array(X_seq[:split]), np.array(y_seq[:split]),
                np.array(X_seq[split:]), np.array(y_seq[split:]), feature_cols)
    else:
        return X_train, X_test, y_train, y_test, feature_cols

# ---------------------
# 5. MODEL XGBOOST
# ---------------------
def train_xgboost(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="logloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost Results ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall   : {recall_score(y_test, y_pred):.2%}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.2%}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_pred, y_prob

# ---------------------
# 6. MODEL LSTM
# ---------------------
def train_lstm(X_train, y_train, X_test, y_test, input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

# ---------------------
# 7. FEATURE IMPORTANCE
# ---------------------
def plot_feature_importance(model, feature_names, top_n=20):
    import matplotlib.pyplot as plt

    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_importance = importance[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importance, align="center")
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance - XGBoost")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("xauusd_feature_importance.png", dpi=150)
    plt.show()

# ---------------------
# 8. BACKTEST
# ---------------------
def backtest_strategy(df, predictions, confidence_threshold=0.6):
    """
    Simple backtest: long when probability > threshold
    """
    data = df.iloc[-len(predictions):].copy()
    data["Signal"] = (predictions > confidence_threshold).astype(int)
    data["Return"] = data["Close"].pct_change().shift(-1)
    data["Strategy_Return"] = data["Signal"] * data["Return"]
    data["Cumulative_Return"] = (1 + data["Strategy_Return"]).cumprod()
    data["Buy_Hold"] = (1 + data["Return"]).cumprod()

    total_trades = data["Signal"].sum()
    win_trades = ((data["Signal"] == 1) & (data["Return"] > 0)).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    print("\n=== Backtest Results ===")
    print(f"Total Trades      : {total_trades}")
    print(f"Win Rate          : {win_rate:.2%}")
    print(f"Strategy Return   : {(data['Cumulative_Return'].iloc[-1] - 1):.2%}")
    print(f"Buy & Hold Return : {(data['Buy_Hold'].iloc[-1] - 1):.2%}")
    print(f"Max Drawdown      : {(data['Strategy_Return'].cumsum().cummax() - data['Strategy_Return'].cumsum()).max():.2%}")

    return data

# ---------------------
# 9. PREDIKSI HARI INI
# ---------------------
def predict_today(features, model, feature_names):
    import xgboost as xgb
    df = pd.DataFrame([features], columns=feature_names)
    prob = model.predict_proba(df)[0, 1]
    pred = "BUY (Bullish)" if prob >= 0.5 else "SELL (Bearish)"
    return pred, prob

# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    print("=== XAUUSD PREDICTOR ===")

    # --- Load data ---
    df = fetch_xauusd_data(source="csv", filepath="xauusd_daily.csv")
    print(f"Loaded {len(df)} days of data ({df.index.min()} - {df.index.max()})")

    # --- Feature engineering ---
    df = add_technical_features(df)
    print(f"Features: {len(df.columns)}")

    # --- Create target ---
    df = create_target(df, forward_period=1, target_type="direction")

    # --- Prepare dataset ---
    X_train, X_test, y_train, y_test, features = prepare_dataset(df, sequence_length=0, test_size=0.2)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(features)}")

    # --- Train XGBoost ---
    model, y_pred, y_prob = train_xgboost(X_train, y_train, X_test, y_test)

    # --- Feature importance ---
    plot_feature_importance(model, features, top_n=20)

    # --- Backtest ---
    results = backtest_strategy(df, y_prob, confidence_threshold=0.6)

    # --- Simpan model ---
    import joblib
    joblib.dump(model, "xauusd_xgb_model.pkl")
    print("\nModel saved: xauusd_xgb_model.pkl")
