"""
Shared trading logic: levels, TP/SL, OHLC evaluation, feature engineering for XAUUSD.
Used by auto_runner.py, runner_4h.py, telegram_notifier.py, backtest.py
"""
import os, sys
import pandas as pd
import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "xauusd_journal.db")
CSV_DAILY = os.path.join(BASE_DIR, "xauusd_daily.csv")
CSV_4H = os.path.join(BASE_DIR, "xauusd_4h.csv")

try:
    with open(os.path.join(BASE_DIR, "config.yaml")) as f:
        CFG = yaml.safe_load(f)
except Exception as e:
    print(f"[WARN] Failed to load config.yaml: {e}")
    CFG = {}


def load_daily(csv_path=None):
    if csv_path is None:
        csv_path = CSV_DAILY
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df


def calculate_levels(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / (ls + 1e-10))
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return {
        "close": float(c.iloc[-1]), "rsi": float(rsi.iloc[-1]),
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
    atr = levels["atr"]
    if is_buy:
        sl = round(max(levels["low_20"], entry - atr * 1.2), 2)
        tp1 = round(min(entry + atr * 0.8, levels["bb_upper"]), 2)
        tp2 = round(min(entry + atr * 1.8, levels["ema50"]), 2)
    else:
        sl = round(min(levels["high_20"], entry + atr * 1.2), 2)
        tp1 = round(max(entry - atr * 0.8, levels["bb_lower"]), 2)
        tp2 = round(max(entry - atr * 1.8, levels["low_20"]), 2)
    return sl, tp1, tp2


def evaluate_sl_tp(df, entry_idx, entry_price, sl, tp1, tp2, is_buy, max_lookahead=5):
    end = min(entry_idx + max_lookahead + 1, len(df))
    segment = df.iloc[entry_idx + 1:end]
    for _, row in segment.iterrows():
        high, low, open_p = row["High"], row["Low"], row["Open"]
        if is_buy:
            sl_hit, tp_hit = low <= sl, high >= tp1
            if sl_hit and tp_hit:
                if abs(open_p - sl) < abs(open_p - tp1):
                    return "LOSS", "SL_HIT", sl, (sl - entry_price) / entry_price * 100
                d = "TP2_HIT" if high >= tp2 else "TP1_HIT"
                x = tp2 if high >= tp2 else tp1
                return "WIN", d, x, (x - entry_price) / entry_price * 100
            if sl_hit:
                return "LOSS", "SL_HIT", sl, (sl - entry_price) / entry_price * 100
            if tp_hit:
                d = "TP2_HIT" if high >= tp2 else "TP1_HIT"
                x = tp2 if high >= tp2 else tp1
                return "WIN", d, x, (x - entry_price) / entry_price * 100
        else:
            sl_hit, tp_hit = high >= sl, low <= tp1
            if sl_hit and tp_hit:
                if abs(open_p - sl) < abs(open_p - tp1):
                    return "LOSS", "SL_HIT", sl, (entry_price - sl) / entry_price * 100
                d = "TP2_HIT" if low <= tp2 else "TP1_HIT"
                x = tp2 if low <= tp2 else tp1
                return "WIN", d, x, (entry_price - x) / entry_price * 100
            if sl_hit:
                return "LOSS", "SL_HIT", sl, (entry_price - sl) / entry_price * 100
            if tp_hit:
                d = "TP2_HIT" if low <= tp2 else "TP1_HIT"
                x = tp2 if low <= tp2 else tp1
                return "WIN", d, x, (entry_price - x) / entry_price * 100

    final_close = float(segment["Close"].iloc[-1]) if len(segment) > 0 else entry_price
    pct = (final_close - entry_price) / entry_price * 100
    if is_buy:
        outcome = "EXPIRED"
    else:
        outcome = "EXPIRED"
        pct = -pct
    return outcome, "EXPIRED", final_close, pct


# ========== FEATURE ENGINEERING ==========

def compute_adx(high, low, close, period=14):
    """True ADX calculation."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def _add_macro_features(df, close):
    """Add macro/intermarket correlation features."""
    macro_map = {
        "DXY_Close": "DXY", "VIX_Close": "VIX", "SPY_Close": "SPY",
        "US10Y_Close": "US10Y", "OIL_Close": "OIL",
        "TIP_Close": "TIP", "Silver_Close": "Silver",
        "BTC_Close": "BTC", "USDJPY_Close": "USDJPY",
        "EURUSD_Close": "EURUSD", "Copper_Close": "Copper",
    }
    for col, prefix in macro_map.items():
        if col in df.columns:
            s = df[col]
            ret = s.pct_change()
            df[f"{prefix}_Return_1d"] = ret
            df[f"{prefix}_Return_5d"] = s.pct_change(5)
            ema20 = s.ewm(span=20, adjust=False).mean()
            df[f"{prefix}_EMA_20"] = ema20
            df[f"{prefix}_Dist_EMA20"] = (s - ema20) / (s + 1e-10)
            df[f"GOLD_{prefix}_Corr_20"] = close.pct_change().rolling(20).corr(ret)
        else:
            for feat in [f"{prefix}_Return_1d", f"{prefix}_Return_5d",
                         f"{prefix}_Dist_EMA20", f"GOLD_{prefix}_Corr_20"]:
                df[feat] = np.nan

    # Gold-Silver ratio
    if "Silver_Close" in df.columns:
        df["Gold_Silver_Ratio"] = close / df["Silver_Close"]
        gsr_ema = df["Gold_Silver_Ratio"].ewm(span=20).mean()
        df["GSR_Zscore"] = (df["Gold_Silver_Ratio"] - gsr_ema) / df["Gold_Silver_Ratio"].rolling(20).std()
    else:
        df["Gold_Silver_Ratio"] = np.nan
        df["GSR_Zscore"] = np.nan

    # Breakeven inflation
    for col in ["Breakeven_5Y", "Breakeven_10Y"]:
        if col in df.columns:
            df[f"{col}_Change"] = df[col].diff()
        else:
            df[f"{col}_Change"] = np.nan
    if "Breakeven_10Y" in df.columns:
        df["GOLD_Breakeven_Corr_20"] = close.pct_change().rolling(20).corr(df["Breakeven_10Y"].diff())
    else:
        df["GOLD_Breakeven_Corr_20"] = np.nan

    # COT positioning
    for col in ["Spec_Net_Long_Pct", "Spec_Positioning_Change", "Commercial_Net_Short_Pct"]:
        df[col] = df[col] if col in df.columns else np.nan

    # Geopolitical risk
    if "GPR_Index" in df.columns:
        df["GPR_Change_1m"] = df["GPR_Index"].pct_change(20)
        df["GPR_High"] = (df["GPR_Index"] > df["GPR_Index"].rolling(100).quantile(0.8)).astype(int)
    else:
        df["GPR_Change_1m"] = np.nan
        df["GPR_High"] = 0

    return df


def engineer_features(df, return_periods=None, return_prefix="d"):
    """Generate features for any timeframe."""
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    if return_periods is None:
        return_periods = [1, 5, 20] if return_prefix == "d" else [1, 4, 8]

    # EMAs
    for p in [5, 10, 20, 50, 100, 200]:
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    # RSI
    d = close.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + g / (ls + 1e-10))

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
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / close

    # Body/Range
    df["Body"] = abs(close - df["Open"])
    df["Range"] = high - low

    # Returns
    for p in return_periods:
        df[f"Return_{p}{return_prefix}"] = close.pct_change(p)

    # Volatility
    df["Vol_20"] = close.pct_change().rolling(20).std()
    df["Vol_50"] = close.pct_change().rolling(50).std()

    # Distance to highs/lows
    for p in [10, 20, 50]:
        df[f"Dist_High_{p}"] = (high.rolling(p).max() - close) / close
        df[f"Dist_Low_{p}"] = (close - low.rolling(p).min()) / close

    # Volume
    df["Volume_Change"] = vol.pct_change()
    df["Volume_SMA_20"] = vol.rolling(20).mean()
    df["Volume_Ratio"] = vol / (df["Volume_SMA_20"] + 1e-10)

    # True ADX (replaces proxy)
    df["ADX_14"], df["Plus_DI"], df["Minus_DI"] = compute_adx(high, low, close)
    df["Is_Trending"] = (df["ADX_14"] > 25).astype(int)
    df["Trend_Direction"] = np.where(df["Plus_DI"] > df["Minus_DI"], 1, -1)

    # ATR percentile rank (vectorized, faster than lambda)
    atr_vals = df["ATR_14"].values
    pctile = np.full(len(atr_vals), np.nan)
    for i in range(99, len(atr_vals)):
        window = atr_vals[i-99:i+1]
        pctile[i] = (window <= atr_vals[i]).sum() / 100
    df["ATR_Pctile_100"] = pctile

    # Cyclical encoding
    dow = df.index.dayofweek
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * dow / 5)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * dow / 5)
    month = df.index.month
    df["Month_sin"] = np.sin(2 * np.pi * month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month / 12)

    if return_prefix == "b":
        hour = df.index.hour
        df["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
        # Session awareness
        df["Is_Asian_Session"] = ((hour >= 0) & (hour < 8)).astype(int)
        df["Is_London_Session"] = ((hour >= 8) & (hour < 16)).astype(int)
        df["Is_NY_Session"] = ((hour >= 13) & (hour < 21)).astype(int)
        df["Is_London_NY_Overlap"] = ((hour >= 13) & (hour < 16)).astype(int)

    if return_prefix == "d":
        # Gold seasonal features
        df["Is_Q4_Gold_Season"] = ((month >= 10) | (month <= 2)).astype(int)
        df["Is_Month_End"] = (df.index.day >= 25).astype(int)
        df["Is_Quarter_End"] = ((month % 3 == 0) & (df.index.day >= 25)).astype(int)

    return df


def engineer_features_daily(df):
    """Full daily features including macro, regime, COT, GPR."""
    df = engineer_features(df, return_periods=[1, 5, 20], return_prefix="d")
    close = df["Close"]
    df = _add_macro_features(df, close)
    return df


def engineer_features_4h(df):
    """4H features with macro data resampled from daily + daily/weekly trend."""
    df = engineer_features(df, return_periods=[1, 4, 8], return_prefix="b")
    close = df["Close"]

    # Resample daily macro data to 4H
    daily_macro_files = {
        "dxy_daily.csv": "DXY_Close", "tip_daily.csv": "TIP_Close",
        "silver_daily.csv": "Silver_Close", "btc_daily.csv": "BTC_Close",
        "usdjpy_daily.csv": "USDJPY_Close", "eurusd_daily.csv": "EURUSD_Close",
    }
    for csv_file, colname in daily_macro_files.items():
        csv_path = os.path.join(BASE_DIR, csv_file)
        if os.path.exists(csv_path):
            try:
                macro = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
                macro.index = pd.to_datetime(macro.index).tz_localize(None) if macro.index.tz else macro.index
                macro = macro.resample("4h").ffill()
                macro = macro.reindex(df.index, method="ffill")
                if colname in macro.columns:
                    df[colname] = macro[colname]
            except Exception as e:
                print(f"[WARN] 4H macro {csv_file}: {e}")

    # Add macro features for 4H
    df = _add_macro_features(df, close)

    # Multi-timeframe: inject daily trend
    daily_csv = os.path.join(BASE_DIR, "xauusd_daily.csv")
    if os.path.exists(daily_csv):
        try:
            df_daily = pd.read_csv(daily_csv, parse_dates=["Date"], index_col="Date").sort_index()
            daily_trend = get_daily_trend(df_daily)
            df["Daily_Trend"] = daily_trend.reindex(df.index, method="ffill")
            weekly_trend = get_weekly_trend(df_daily)
            df["Weekly_Trend"] = weekly_trend.reindex(df.index, method="ffill")
        except Exception as e:
            print(f"[WARN] 4H trend injection: {e}")
            df["Daily_Trend"] = 0
            df["Weekly_Trend"] = 0
    else:
        df["Daily_Trend"] = 0
        df["Weekly_Trend"] = 0

    # Ensure all expected columns exist
    for c in ["DXY_Return_1b", "DXY_Return_4b", "GOLD_DXY_Corr_8", "DXY_Dist_EMA20",
              "Daily_Trend", "Weekly_Trend"]:
        if c not in df.columns:
            df[c] = np.nan

    return df


# ========== MULTI-TIMEFRAME TREND ==========

def get_daily_trend(df_daily):
    """Calculate daily trend: 1=bullish, 0=neutral, -1=bearish."""
    close = df_daily["Close"]
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    d = close.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / (ls + 1e-10))
    trend = pd.Series(0, index=df_daily.index)
    trend[(close > ema20) & (ema20 > ema50) & (rsi > 50)] = 1
    trend[(close < ema20) & (ema20 < ema50) & (rsi < 50)] = -1
    return trend


def get_weekly_trend(df_daily):
    """Resample to weekly and calculate trend: 1=bullish, -1=bearish."""
    weekly = df_daily.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    close_w = weekly["Close"]
    ema10w = close_w.ewm(span=10).mean()
    trend = pd.Series(0, index=weekly.index)
    trend[close_w > ema10w] = 1
    trend[close_w < ema10w] = -1
    return trend.reindex(df_daily.index, method="ffill")


# ========== SHARED ATR TRIPLE-BARRIER LABELING ==========

def atr_target(df, forward=3, atr_mult=0.8, sl_mult=1.2):
    """3-class ATR labeling: 2=BULLISH, 1=NEUTRAL, 0=BEARISH."""
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(close)
    tr = np.maximum.reduce([high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))])
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14).mean().shift(1).values
    target = np.ones(n, dtype=int)
    for i in range(n - forward):
        if np.isnan(atr[i]):
            continue
        entry = close[i]
        tp = entry + atr[i] * atr_mult
        sl = entry - atr[i] * sl_mult
        fut_high = high[i + 1:i + forward + 1]
        fut_low = low[i + 1:i + forward + 1]
        tp_hit_bars = np.where(fut_high >= tp)[0]
        sl_hit_bars = np.where(fut_low <= sl)[0]
        first_tp = tp_hit_bars[0] if len(tp_hit_bars) > 0 else forward + 1
        first_sl = sl_hit_bars[0] if len(sl_hit_bars) > 0 else forward + 1
        if first_tp < first_sl:
            target[i] = 2
        elif first_sl < first_tp:
            target[i] = 0
    return pd.Series(target, index=df.index)


def atr_target_binary(df, forward=3, atr_mult=0.8, sl_mult=1.2):
    t = atr_target(df, forward, atr_mult, sl_mult)
    return (t == 2).astype(int)


# ========== SHARED WALK-FORWARD CV ==========

def walk_forward_split(X, n_splits=5, embargo=3, oot_pct=0.15):
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
