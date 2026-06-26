"""
XAUUSD Backtest - Walk-forward simulation of the model.
Simulates predictions day-by-day over the test period,
tracks SL/TP hits, and reports performance.
"""
import os, pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def engineer_features(df):
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
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
    df["Return_1d"] = close.pct_change()
    df["Return_5d"] = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)
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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def compute_tp_sl(levels, is_buy, entry):
    atr = levels["atr"]
    if is_buy:
        sl = round(max(levels["low_20"], entry - atr * 1.2), 2)
        tp1 = round(max(entry + atr * 0.8, entry + atr * 0.5), 2)
        tp2 = round(max(entry + atr * 1.8, levels["ema50"]), 2)
    else:
        sl = round(min(levels["ema20"] + atr * 0.8, entry + atr * 1.2), 2)
        tp1 = round(min(entry - atr * 0.8, entry - atr * 0.5), 2)
        tp2 = round(min(entry - atr * 1.8, levels["low_20"]), 2)
    return sl, tp1, tp2

def evaluate_sl_tp(df, entry_idx, entry_price, sl, tp1, tp2, is_buy, max_lookahead=5):
    """
    Scan forward from entry_idx using OHLC to see if SL/TP was hit.
    Returns (outcome, detail, exit_price, pct).
    """
    end = min(entry_idx + max_lookahead + 1, len(df))
    segment = df.iloc[entry_idx + 1:end]
    for i, (_, row) in enumerate(segment.iterrows()):
        high, low = row["High"], row["Low"]
        if is_buy:
            if low <= sl:
                return "LOSS", "SL_HIT", sl, (sl - entry_price) / entry_price * 100
            if high >= tp1:
                detail = "TP2_HIT" if high >= tp2 else "TP1_HIT"
                exit_px = tp2 if high >= tp2 else tp1
                return "WIN", detail, exit_px, (exit_px - entry_price) / entry_price * 100
        else:
            if high >= sl:
                return "LOSS", "SL_HIT", sl, (entry_price - sl) / entry_price * 100
            if low <= tp1:
                detail = "TP2_HIT" if low <= tp2 else "TP1_HIT"
                exit_px = tp2 if low <= tp2 else tp1
                return "WIN", detail, exit_px, (entry_price - exit_px) / entry_price * 100
    # Time expired - check close price
    final_close = float(segment["Close"].iloc[-1]) if len(segment) > 0 else entry_price
    pct = (final_close - entry_price) / entry_price * 100
    if is_buy:
        outcome = "WIN" if pct > 0 else "LOSS"
    else:
        outcome = "WIN" if pct < 0 else "LOSS"
    return outcome, "EXPIRED", final_close, pct


def backtest():
    print("Loading model...")
    arts = joblib.load("xauusd_model.pkl")
    model = arts["model"]
    scaler = arts["scaler"]
    feature_cols = arts["feature_cols"]
    best_thresh = arts["best_thresh"]
    forward_days = arts.get("forward_days", 3)

    print("Loading data...")
    df = pd.read_csv("xauusd_daily.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # Use test portion (last 20%)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()

    print(f"\nTest period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print(f"Test rows: {len(test_df)}")
    print(f"Threshold: {best_thresh:.3f} (min: 0.55)")
    print(f"Forward days: {forward_days}")
    print(f"Simulating {len(test_df)} days...")

    trades = []
    signals = 0
    skipped = 0
    empty_counter = 0

    for idx in range(len(test_df)):
        # Use all data up to current test point
        cutoff = split + idx
        train_window = df.iloc[:cutoff + 1].copy()

        feat_df = engineer_features(train_window)

        last_row = feat_df[feature_cols].dropna().iloc[-1:]
        if last_row.empty:
            empty_counter += 1
            continue

        features_scaled = scaler.transform(last_row.values)
        prob = float(model.predict_proba(features_scaled)[0, 1])
        min_thresh = max(0.55, best_thresh)

        signals += 1
        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx+1}/{len(test_df)} days")

        # Only record when confidence >= threshold (model only predicts bullish)
        if prob < min_thresh:
            skipped += 1
            continue

        entry_price = float(train_window["Close"].iloc[-1])
        entry_date = train_window.index[-1]
        # Calculate levels from window data
        c = train_window["Close"]
        h = train_window["High"]
        l = train_window["Low"]
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        ema50 = float(c.ewm(span=50).mean().iloc[-1])
        low_20 = float(l.rolling(20).min().iloc[-1])

        levels = {"atr": atr, "ema20": ema20, "ema50": ema50, "low_20": low_20}

        sl, tp1, tp2 = compute_tp_sl(levels, True, entry_price)

        outcome, detail, exit_price, pct = evaluate_sl_tp(
            df, cutoff, entry_price, sl, tp1, tp2, True, forward_days + 2
        )

        trades.append({
            "date": entry_date,
            "entry": entry_price,
            "direction": "BUY",
            "confidence": prob,
            "threshold": min_thresh,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "outcome": outcome,
            "detail": detail,
            "exit_price": exit_price,
            "pct": pct,
        })

    results = pd.DataFrame(trades)
    print(f"\nBacktest complete: {len(trades)} trades recorded ({empty_counter} empty rows skipped)")

    total = len(results)
    wins = len(results[results["outcome"] == "WIN"])
    losses = len(results[results["outcome"] == "LOSS"])
    acc = wins / total * 100 if total > 0 else 0
    avg_win = results[results["outcome"] == "WIN"]["pct"].mean() if wins > 0 else 0
    avg_loss = results[results["outcome"] == "LOSS"]["pct"].mean() if losses > 0 else 0
    total_return = results["pct"].sum()
    profit_factor = abs(results[results["pct"] > 0]["pct"].sum() /
                       (results[results["pct"] < 0]["pct"].sum() + 1e-10))

    print("=" * 60)
    print(f"  BACKTEST RESULTS - Test Period")
    print("=" * 60)
    print(f"  Total signals     : {total}")
    print(f"  Wins              : {wins}")
    print(f"  Losses            : {losses}")
    print(f"  Accuracy          : {acc:.1f}%")
    print(f"  Avg Win           : {avg_win:+.2f}%")
    print(f"  Avg Loss          : {avg_loss:+.2f}%")
    print(f"  Total Return      : {total_return:+.2f}%")
    print(f"  Profit Factor     : {profit_factor:.2f}")
    print("=" * 60)

    # --- MONTHLY BREAKDOWN ---
    print(f"\n  MONTHLY PERFORMANCE:")
    print(f"  {'Month':<10} {'Signals':>7} {'Wins':>5} {'Losses':>7} {'Acc':>6} {'Return':>9}")
    print(f"  {'-'*48}")
    results["month"] = results["date"].dt.to_period("M")
    for month, grp in results.groupby("month", sort=True):
        w = len(grp[grp["outcome"] == "WIN"])
        l = len(grp) - w
        a = w / len(grp) * 100 if len(grp) > 0 else 0
        r = grp["pct"].sum()
        print(f"  {str(month):<10} {len(grp):>7} {w:>5} {l:>7} {a:>5.1f}% {r:>+8.2f}%")
    print(f"  {'-'*48}")

    detail_counts = results["detail"].value_counts()
    print(f"\n  Outcome breakdown:")
    for d, c in detail_counts.items():
        print(f"    {d}: {c}")

    # Performance by confidence bracket
    print(f"\n  Performance by confidence (threshold >= {max(0.55, best_thresh):.2f}):")
    for label, lo, hi in [("55-65%", 0.55, 0.65), ("65-75%", 0.65, 0.75),
                           ("75-85%", 0.75, 0.85), ("85%+", 0.85, 1.0)]:
        subset = results[(results["confidence"] >= lo) & (results["confidence"] < hi)]
        if len(subset) > 0:
            w = len(subset[subset["outcome"] == "WIN"])
            ret = subset["pct"].sum()
            print(f"    {label}: {w}/{len(subset)} ({w/len(subset)*100:.0f}%) return: {ret:+.2f}%")

    # Best threshold analysis
    print(f"\n  Threshold optimization (min 5 trades per bracket):")
    best_profit = -999
    best_t = 0.5
    for t in np.arange(0.50, 0.90, 0.025):
        subset = results[results["confidence"] >= t]
        if len(subset) >= 5:
            w = len(subset[subset["outcome"] == "WIN"])
            ret = subset["pct"].sum()
            print(f"    >= {t:.3f}: {w}/{len(subset)} ({w/len(subset)*100:.0f}%) return: {ret:+.2f}%")
            if ret > best_profit:
                best_profit = ret
                best_t = t
                best_wr = w / len(subset)

    print(f"\n  Recommended threshold: {best_t:.3f} (win rate: {best_wr:.0%}, profit: {best_profit:+.2f}%)")
    print("=" * 60)

    # Save detailed results
    results.to_csv("backtest_results.csv", index=False)
    print(f"\n  Detailed results saved to backtest_results.csv")
    print(f"  {len(results)} trades analyzed")


if __name__ == "__main__":
    backtest()
