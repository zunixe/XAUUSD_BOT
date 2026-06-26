"""
Shared trading logic: levels, TP/SL, OHLC evaluation for XAUUSD.
Used by auto_runner.py, telegram_notifier.py, backtest.py
"""
import os, sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "xauusd_journal.db")
CSV_DAILY = os.path.join(BASE_DIR, "xauusd_daily.csv")
CSV_4H = os.path.join(BASE_DIR, "xauusd_4h.csv")


def load_daily(csv_path=None):
    """Load xauusd_daily.csv and return sorted dataframe with Date index."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xauusd_daily.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df


def calculate_levels(df):
    """Calculate ATR, EMAs, RSI, BB, support/resistance from a dataframe."""
    c, h, l = df["Close"], df["High"], df["Low"]
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / (ls + 1e-10))
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return {
        "close": float(c.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "atr": float(tr.rolling(14).mean().iloc[-1]),
        "ema20": float(c.ewm(span=20).mean().iloc[-1]),
        "ema50": float(c.ewm(span=50).mean().iloc[-1]),
        "bb_upper": float((c.rolling(20).mean() + 2 * c.rolling(20).std()).iloc[-1]),
        "bb_lower": float((c.rolling(20).mean() - 2 * c.rolling(20).std()).iloc[-1]),
        "swing_high": float(h.rolling(50).max().iloc[-1]),
        "swing_low": float(l.rolling(50).min().iloc[-1]),
        "low_20": float(l.rolling(20).min().iloc[-1]),
        "high_20": float(h.rolling(20).max().iloc[-1]),
    }


def compute_tp_sl(levels, is_buy, entry):
    """Return (sl, tp1, tp2) from levels dict."""
    atr = levels["atr"]
    if is_buy:
        sl = round(max(levels["low_20"], entry - atr * 1.2), 2)
        tp1 = round(max(entry + atr * 0.8, entry + atr * 0.5), 2)
        tp2 = round(max(entry + atr * 1.8, levels["ema50"]), 2)
    else:
        sl = round(min(levels["high_20"], entry + atr * 1.2), 2)
        tp1 = round(min(entry - atr * 0.8, entry - atr * 0.5), 2)
        tp2 = round(min(entry - atr * 1.8, levels["low_20"]), 2)
    return sl, tp1, tp2


def evaluate_sl_tp(df, entry_idx, entry_price, sl, tp1, tp2, is_buy, max_lookahead=5):
    """
    Scan forward OHLC data from entry_idx.
    Returns (outcome, detail, exit_price, pct).
    """
    end = min(entry_idx + max_lookahead + 1, len(df))
    segment = df.iloc[entry_idx + 1:end]
    for _, row in segment.iterrows():
        high, low = row["High"], row["Low"]
        if is_buy:
            if low <= sl:
                return "LOSS", "SL_HIT", sl, (sl - entry_price) / entry_price * 100
            if high >= tp1:
                d = "TP2_HIT" if high >= tp2 else "TP1_HIT"
                x = tp2 if high >= tp2 else tp1
                return "WIN", d, x, (x - entry_price) / entry_price * 100
        else:
            if high >= sl:
                return "LOSS", "SL_HIT", sl, (entry_price - sl) / entry_price * 100
            if low <= tp1:
                d = "TP2_HIT" if low <= tp2 else "TP1_HIT"
                x = tp2 if low <= tp2 else tp1
                return "WIN", d, x, (entry_price - x) / entry_price * 100

    final_close = float(segment["Close"].iloc[-1]) if len(segment) > 0 else entry_price
    pct = (final_close - entry_price) / entry_price * 100
    if is_buy:
        outcome = "WIN" if pct > 0 else "LOSS"
    else:
        outcome = "WIN" if pct < 0 else "LOSS"
        pct = -pct  # SELL: profit positive when price goes down
    return outcome, "EXPIRED", final_close, pct


# ========== SHARED FEATURE ENGINEERING ==========

def engineer_features(df, return_periods=None, return_prefix="d"):
    """Generate features for any timeframe. return_prefix: 'd'=daily, 'b'=4h bars."""
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    if return_periods is None:
        return_periods = [1, 5, 20] if return_prefix == "d" else [1, 4, 8]
    for p in [5, 10, 20, 50, 100, 200]:
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()
    d = close.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + g / (ls + 1e-10))
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
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / close
    df["Body"] = abs(close - df["Open"])
    df["Range"] = high - low
    for p in return_periods:
        df[f"Return_{p}{return_prefix}"] = close.pct_change(p)
    df["Vol_20"] = close.pct_change().rolling(20).std()
    df["Vol_50"] = close.pct_change().rolling(50).std()
    for p in [10, 20, 50]:
        df[f"Dist_High_{p}"] = (high.rolling(p).max() - close) / close
        df[f"Dist_Low_{p}"] = (close - low.rolling(p).min()) / close
    df["Volume_Change"] = vol.pct_change()
    df["Volume_SMA_20"] = vol.rolling(20).mean()
    df["Volume_Ratio"] = vol / (df["Volume_SMA_20"] + 1e-10)
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    if return_prefix == "b":
        df["Hour"] = df.index.hour
    return df


def engineer_features_daily(df):
    """Full daily features including DXY, macro, FOMC."""
    df = engineer_features(df, return_periods=[1, 5, 20], return_prefix="d")
    close = df["Close"]
    # DXY features
    if "DXY_Close" in df.columns:
        dxy = df["DXY_Close"]
        df["DXY_Return_1d"] = dxy.pct_change()
        df["DXY_Return_5d"] = dxy.pct_change(5)
        df["GOLD_DXY_Corr_20"] = close.pct_change().rolling(20).corr(dxy.pct_change())
        df["DXY_EMA_20"] = dxy.ewm(span=20, adjust=False).mean()
        df["DXY_Dist_EMA20"] = (dxy - df["DXY_EMA_20"]) / dxy
    else:
        for c in ["DXY_Return_1d","DXY_Return_5d","GOLD_DXY_Corr_20","DXY_Dist_EMA20"]:
            df[c] = np.nan
    macro_map = {"VIX_Close":"VIX","SPY_Close":"SPY","US10Y_Close":"US10Y","OIL_Close":"OIL"}
    for col, prefix in macro_map.items():
        if col in df.columns:
            s = df[col]
            df[f"{prefix}_Return_1d"] = s.pct_change()
            df[f"{prefix}_Return_5d"] = s.pct_change(5)
            df[f"{prefix}_EMA_20"] = s.ewm(span=20, adjust=False).mean()
            df[f"{prefix}_Dist_EMA20"] = (s - df[f"{prefix}_EMA_20"]) / (s + 1e-10)
            df[f"GOLD_{prefix}_Corr_20"] = close.pct_change().rolling(20).corr(s.pct_change())
        else:
            for feat in [f"{prefix}_Return_1d",f"{prefix}_Return_5d",f"{prefix}_Dist_EMA20",f"GOLD_{prefix}_Corr_20"]:
                df[feat] = np.nan
    return df


def engineer_features_4h(df):
    """4H features (no macro)."""
    df = engineer_features(df, return_periods=[1, 4, 8], return_prefix="b")
    return df


# ========== SHARED ATR TRIPLE-BARRIER LABELING ==========

def atr_target(df, forward=3, atr_mult=0.8, sl_mult=0.6):
    """ATR-based triple-barrier labeling. Returns 1 if TP hit before SL."""
    close, high, low = df["Close"], df["High"], df["Low"]
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)
    target = pd.Series(0, index=df.index)
    for i in range(len(df) - forward):
        entry = close.iloc[i]
        tp = entry + atr.iloc[i] * atr_mult
        sl = entry - atr.iloc[i] * sl_mult
        future = df.iloc[i + 1:i + forward + 1]
        hit_tp = (future["High"] >= tp).any()
        hit_sl = (future["Low"] <= sl).any()
        if hit_tp and not hit_sl:
            target.iloc[i] = 1
        elif hit_tp and hit_sl:
            mid = (tp + sl) / 2
            target.iloc[i] = 1 if entry < mid else 0
    return target


# ========== SHARED WALK-FORWARD CV ==========

def walk_forward_split(X, n_splits=5, embargo=3, oot_pct=0.15):
    """Purged expanding window walk-forward CV with OOT holdout.
    
    Args:
        X: feature matrix (used only for length)
        n_splits: number of CV folds
        embargo: gap between train and validation to prevent label leakage
        oot_pct: fraction of data reserved as final out-of-time test set
    
    Returns:
        folds: list of (train_idx, val_idx) tuples
        oot: (oot_idx, None) — final holdout set (never touched during CV)
    """
    n = len(X)
    oot_size = int(n * oot_pct)
    train_val_end = n - oot_size
    indices = np.arange(train_val_end)
    
    fold_size = train_val_end // (n_splits + 1)
    folds = []
    for k in range(n_splits):
        train_end = (k + 1) * fold_size
        purged_end = max(train_end - embargo, 0)
        val_start = train_end + embargo
        val_end = min(val_start + fold_size, train_val_end)
        if val_end <= val_start:
            continue
        folds.append((indices[:purged_end], indices[val_start:val_end]))
    
    oot = (np.arange(train_val_end, n), None)
    return folds, oot


# ========== SHARED OPTUNA TUNING ==========

def optuna_tune(X_train, y_train, X_val, y_val, scale, n_trials=100,
                n_estimators_range=(200, 600), default_n_estimators=300,
                sample_weight=None):
    """Hyperparameter tuning with Optuna (fallback to fixed params if unavailable)."""
    try:
        import optuna
    except ImportError:
        optuna = None

    sw_train = sample_weight
    fit_kwargs = {}
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train

    if optuna is None:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=default_n_estimators, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=2.0, scale_pos_weight=scale,
            random_state=42, eval_metric="logloss", early_stopping_rounds=30, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kwargs)
        return model, {"n_estimators": default_n_estimators, "max_depth": 6, "learning_rate": 0.02}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *n_estimators_range),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": scale,
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        from xgboost import XGBClassifier
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kwargs)
        from sklearn.metrics import f1_score
        yp = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, yp)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    from xgboost import XGBClassifier
    model = XGBClassifier(scale_pos_weight=scale, random_state=42,
        eval_metric="logloss", early_stopping_rounds=30, verbosity=0, **best)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kwargs)
    return model, best
