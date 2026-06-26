"""
XAUUSD 4-Hour Runner
- Separate model, data, DB table for 4H predictions
- Auto-download, train, predict, evaluate
- Telegram outcome notifications + weekly reports
"""
import sys, os, time, subprocess, argparse, json, sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patterns, trading

MODEL_4H = os.path.join(trading.BASE_DIR, "xauusd_model_4h.pkl")
FORWARD_BARS = 8  # ~32 jam (~1.5 hari trading)


def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[4H][{t}] {msg}")
    with open(os.path.join(trading.BASE_DIR, "runner_4h.log"), "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def init_db():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions_4h (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, time TEXT, price REAL,
            predicted_direction TEXT, confidence REAL, threshold REAL,
            target_date TEXT, target_time TEXT, model_version TEXT,
            sl REAL, tp1 REAL, tp2 REAL, entry_realtime REAL,
            outcome TEXT, outcome_detail TEXT, result_pct REAL,
            notified INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def update_data():
    import yfinance as yf
    df = yf.download("GC=F", period="5d", interval="4h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    try:
        old = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date")
        old.index = pd.to_datetime(old.index, utc=True).tz_localize(None)
        combined = pd.concat([old, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    except FileNotFoundError:
        combined = df
    combined.to_csv(trading.CSV_4H)
    log(f"Data updated: {len(combined)} candles")


def _features_4h(df):
    """4H feature engineering (shared core + patterns)."""
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None) if df.index.tz is not None else df.index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = trading.engineer_features_4h(df)
    df = patterns.detect_candlestick(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cols = [c for c in df.columns if c not in ["Close", "High", "Low", "Open", "Volume", "Target"]]
    return df, cols


def retrain():
    log("Retraining 4H model...")
    df = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df["Target"] = trading.atr_target(df, forward=FORWARD_BARS, atr_mult=0.6, sl_mult=0.4)

    # Override ATR labels with real SL/TP outcomes where available
    df["sample_weight"] = 1.0
    try:
        import sqlite3
        con = sqlite3.connect(trading.DB_FILE)
        outcomes = pd.read_sql_query(
            "SELECT date, time, predicted_direction, outcome FROM predictions_4h"
            " WHERE outcome IN ('WIN','LOSS') AND sl IS NOT NULL AND tp1 IS NOT NULL ORDER BY id", con
        )
        con.close()
        n_total = len(df)
        if len(outcomes) >= 5 and len(outcomes) / n_total >= 0.05:
            outcomes["dt"] = pd.to_datetime(outcomes["date"] + " " + outcomes["time"])
            outcomes = outcomes.drop_duplicates("dt", keep="last")
            n_matched = 0
            for _, row in outcomes.iterrows():
                dt = row["dt"]
                if dt in df.index:
                    is_buy = str(row["predicted_direction"]).upper().startswith("BUY")
                    is_win = row["outcome"] == "WIN"
                    label = 1.0 if (is_buy and is_win) or (not is_buy and not is_win) else 0.0
                    df.loc[dt, "Target"] = label
                    df.loc[dt, "sample_weight"] = 2.0
                    n_matched += 1
            if n_matched:
                n_win = int((outcomes["outcome"] == "WIN").sum())
                n_loss = int((outcomes["outcome"] == "LOSS").sum())
                log(f"[LEARN] 4H overrode {n_matched} ATR labels ({n_win}W/{n_loss}L, weight=2x)")
        elif len(outcomes) > 0:
            log(f"[LEARN] 4H skipped override: only {len(outcomes)} real outcomes (<5% of {n_total})")
    except Exception as e:
        log(f"[LEARN] 4H gagal query real outcomes: {e}")

    df, cols = _features_4h(df)
    df.dropna(inplace=True)
    if len(df) < 500:
        log(f"Data terlalu sedikit ({len(df)}), skip")
        return False

    X, y = df[cols].values, df["Target"].values
    sample_weights = df["sample_weight"].values if "sample_weight" in df.columns else None
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, f1_score
    scaler = RobustScaler()
    models, scores, params_list = [], [], []

    folds, oot = trading.walk_forward_split(X, n_splits=3, embargo=FORWARD_BARS)
    thresholds = []
    for k, (train_idx, val_idx) in enumerate(folds):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        sw_t = sample_weights[train_idx] if sample_weights is not None else None
        X_t = scaler.fit_transform(X_t)
        X_v = scaler.transform(X_v)
        neg, pos = (y_t == 0).sum(), (y_t == 1).sum()
        scale = neg / pos if pos > 0 else 1

        model, best_params = trading.optuna_tune(X_t, y_t, X_v, y_v, scale, n_trials=30,
                                                  default_n_estimators=400, sample_weight=sw_t)

        from sklearn.metrics import precision_recall_curve
        y_prob_v = model.predict_proba(X_v)[:, 1]
        precisions, recalls, threshs = precision_recall_curve(y_v, y_prob_v)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_t = float(threshs[np.argmax(f1s[:-1])]) if len(threshs) > 0 else 0.5
        yp = (y_prob_v >= best_t).astype(int)
        acc = accuracy_score(y_v, yp)
        f1 = f1_score(y_v, yp)
        thresholds.append(best_t)
        log(f"  Fold {k+1}: acc={acc:.1%} f1={f1:.1%} thresh={best_t:.3f} | depth={best_params.get('max_depth','?')} lr={best_params.get('learning_rate','?'):.4f}")
        models.append(model)
        scores.append({"acc": acc, "f1": f1})
        params_list.append(best_params)

    best_thresh = float(np.mean(thresholds))
    avg_acc = np.mean([s["acc"] for s in scores])
    fold_weights = np.array([max(s["f1"], 0.01) for s in scores])
    fold_weights = fold_weights / fold_weights.sum()
    ensemble_models = list(zip(models, fold_weights))

    oot_acc = None
    oot_idx, _ = oot
    if len(oot_idx) > 5:
        X_oot, y_oot = X[oot_idx], y[oot_idx]
        X_oot_scaled = scaler.transform(X_oot)
        y_prob_oot = np.zeros(len(oot_idx))
        for m, w in ensemble_models:
            y_prob_oot += w * m.predict_proba(X_oot_scaled)[:, 1]
        y_pred_oot = (y_prob_oot >= best_thresh).astype(int)
        oot_acc = float(accuracy_score(y_oot, y_pred_oot))
        log(f"  OOT holdout ({len(oot_idx)} samples): acc={oot_acc:.1%}")

    import joblib, tempfile
    _data = {"ensemble_models": ensemble_models, "scaler": scaler, "cols": cols,
              "threshold": best_thresh, "forward": FORWARD_BARS,
              "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "samples": len(X), "test_acc": float(avg_acc), "oot_acc": oot_acc}
    _tmp = tempfile.NamedTemporaryFile(delete=False, dir=trading.BASE_DIR, suffix=".pkl")
    try:
        joblib.dump(_data, _tmp.name)
        os.replace(_tmp.name, MODEL_4H)
    except:
        os.unlink(_tmp.name)
        raise
    log(f"Model saved. Avg acc: {avg_acc:.1%}")
    return True


def predict():
    import joblib
    arts = joblib.load(MODEL_4H)
    scaler, cols = arts["scaler"], arts["cols"]
    best_thresh = arts.get("threshold", 0.55)
    forward = arts.get("forward", FORWARD_BARS)
    ensemble_models = arts.get("ensemble_models")
    single_model = arts.get("model")
    if ensemble_models is None and single_model is not None:
        ensemble_models = [(single_model, 1.0)]

    df = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    latest_price = float(df["Close"].iloc[-1])
    latest_time = df.index[-1]
    target_time = latest_time + timedelta(hours=forward * 4)

    # Real-time price
    from telegram_notifier import get_realtime_price
    live = get_realtime_price()
    entry = live["price"] if live else latest_price
    change_str = ""
    if live and live.get("change"):
        change_str = f" ({'+' if live['change'] > 0 else ''}{live['change']:.2f})"

    df_feat, _ = _features_4h(df)
    last_row = df_feat[cols].dropna().iloc[-1:]
    if len(last_row) == 0:
        log("No valid features for prediction")
        return None

    features_scaled = scaler.transform(last_row.values)
    prob = float(sum(w * m.predict_proba(features_scaled)[0, 1] for m, w in ensemble_models))
    min_thresh = max(0.55, best_thresh)
    sell_thresh = 1.0 - min_thresh
    direction = "BUY" if prob >= min_thresh else "SELL" if prob <= sell_thresh else "NO_TRADE"

    log(f"4H Close: ${latest_price:.2f} | Realtime: ${entry:.2f}{change_str} | {direction} | conf: {prob:.1%} | thresh: {min_thresh:.3f}")

    # Save to DB
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()

    # Levels & TP/SL (based on realtime entry)
    sl = tp1 = tp2 = None
    if prob >= min_thresh:
        tr = pd.concat([
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        ], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        sl = round(entry - atr_val * 0.6, 2)
        tp1 = round(entry + atr_val * 0.8, 2)
        tp2 = round(entry + atr_val * 1.5, 2)
    elif prob <= sell_thresh:
        tr = pd.concat([
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        ], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        sl = round(entry + atr_val * 0.6, 2)
        tp1 = round(entry - atr_val * 0.8, 2)
        tp2 = round(entry - atr_val * 1.5, 2)

    c.execute("""
        INSERT INTO predictions_4h (date, time, price, predicted_direction, confidence,
            threshold, target_date, target_time, sl, tp1, tp2, entry_realtime, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (latest_time.strftime("%Y-%m-%d"), latest_time.strftime("%H:%M"),
          latest_price, direction, prob, min_thresh,
          target_time.strftime("%Y-%m-%d"), target_time.strftime("%Y-%m-%d %H:%M"),
          sl, tp1, tp2, entry, f"xgb_4h_v1"))
    pred_id = c.lastrowid
    conn.commit()
    conn.close()
    log(f"Saved to DB (ID: {pred_id})")

    # Telegram — send for BUY, SELL, or skip uncertain middle zone
    if direction != "NO_TRADE":
        from telegram_notifier import send_4h_signal
        send_4h_signal(pred_id, direction, prob, entry, latest_price,
                       target_time.strftime("%Y-%m-%d %H:%M"), min_thresh, sl, tp1, tp2)

    return pred_id


def evaluate():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, date, time, price, sl, tp1, tp2, predicted_direction
        FROM predictions_4h WHERE outcome IS NULL
        ORDER BY id
    """).fetchall()
    if not rows:
        conn.close()
        return 0

    df = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    evaluated = 0
    for pred_id, pred_date, pred_time, entry, sl, tp1, tp2, direction in rows:
        try:
            entry_ts = pd.Timestamp(f"{pred_date} {pred_time}")
            locs = np.where(df.index >= entry_ts)[0]
            if len(locs) == 0:
                continue
            entry_idx = locs[0]
            # Skip if no forward data yet
            if entry_idx + 1 >= len(df):
                continue
            is_buy = direction == "BUY"
            if sl is None or tp1 is None:
                outcome_detail = "NO_SLTP"
                next_close = df.iloc[entry_idx + 1]["Close"] if entry_idx + 1 < len(df) else entry
                pct = (next_close - entry) / entry * 100
                outcome = "WIN" if pct > 0 else "LOSS"
                if not is_buy:
                    pct = -pct
            else:
                max_look = min(FORWARD_BARS + 4, len(df) - entry_idx - 1)
                outcome, detail, exit_price, pct = trading.evaluate_sl_tp(
                    df, entry_idx, entry, sl, tp1, tp2, is_buy, max_lookahead=max_look)
            c.execute("UPDATE predictions_4h SET outcome=?, outcome_detail=?, result_pct=? WHERE id=?",
                      (outcome, detail if sl else outcome_detail, pct, pred_id))
            log(f"  #{pred_id}: {outcome} ({detail if sl else outcome_detail}) {pct:+.2f}%")
            evaluated += 1
        except Exception as e:
            log(f"  #{pred_id}: Error - {e}")

    conn.commit()
    conn.close()

    # Notify outcomes to Telegram
    notify_outcomes()
    return evaluated


def notify_outcomes():
    """Send Telegram notification for recently evaluated 4H signals"""
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, price, predicted_direction, confidence, outcome, outcome_detail, result_pct,
               sl, tp1, tp2
        FROM predictions_4h
        WHERE outcome IS NOT NULL AND notified = 0
    """).fetchall()
    if not rows:
        conn.close()
        return
    for row in rows:
        try:
            from telegram_notifier import send_outcome_notification
            send_outcome_notification(prediction_id=row[0], timeframe="4H",
                direction=row[2], entry=row[1], outcome=row[4],
                detail=row[5], pct=row[6], sl=row[7], tp1=row[8], tp2=row[9])
            c.execute("UPDATE predictions_4h SET notified=1 WHERE id=?", (row[0],))
            log(f"  Outcome notified: #{row[0]} {row[4]}")
        except Exception as e:
            log(f"  Notify error #{row[0]}: {e}")
    conn.commit()
    conn.close()


def weekly_report():
    conn = sqlite3.connect(trading.DB_FILE)
    df = pd.read_sql("SELECT * FROM predictions_4h WHERE outcome IS NOT NULL", conn)
    conn.close()
    if len(df) == 0:
        return

    df["date"] = pd.to_datetime(df["date"])
    now = pd.Timestamp.now()
    week_ago = now - timedelta(days=7)
    week = df[df["date"] >= week_ago]
    if len(week) == 0:
        return

    wins = len(week[week["outcome"] == "WIN"])
    total = len(week)
    acc = wins / total * 100
    ret = week["result_pct"].sum()
    pf = abs(week[week["result_pct"] > 0]["result_pct"].sum() /
             (week[week["result_pct"] < 0]["result_pct"].sum() + 1e-10))

    msg = (
        f"<b>[4H WEEKLY REPORT]</b>\n"
        f"<pre>\n"
        f"  Trades      : {total}\n"
        f"  Win Rate    : {acc:.1f}% ({wins}W / {total-wins}L)\n"
        f"  Total Return: {ret:+.2f}%\n"
        f"  Profit Fact : {pf:.2f}\n"
        f"</pre>"
    )
    from telegram_notifier import send_report
    send_report(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--run", action="store_true", help="Full cycle: update → eval → predict")
    args = parser.parse_args()

    init_db()

    if args.retrain:
        retrain()
    elif args.predict:
        predict()
    elif args.evaluate:
        evaluate()
    elif args.report:
        weekly_report()
    elif args.run:
        update_data()
        evaluate()
        predict()
        weekly_report()
    else:
        print("Usage: python runner_4h.py --run | --retrain | --predict | --evaluate | --report")
