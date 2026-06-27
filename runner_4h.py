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
    exclude = ["Close", "High", "Low", "Open", "Volume", "Target", "sample_weight",
               "DXY_Close", "TIP_Close", "Silver_Close", "BTC_Close", "USDJPY_Close",
               "EURUSD_Close", "Copper_Close"]
    cols = [c for c in df.columns if c not in exclude]
    # Fill NaN in features with 0
    df[cols] = df[cols].fillna(0)
    return df, cols


def retrain():
    log("Retraining 4H model (3-class)...")
    df = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    # 3-class labels aligned with trading: TP=0.8 ATR, SL=1.2 ATR
    df["Target"] = trading.atr_target(df, forward=FORWARD_BARS, atr_mult=0.8, sl_mult=1.2)

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
        if len(outcomes) >= 10 and len(outcomes) / n_total >= 0.05:
            outcomes["dt"] = pd.to_datetime(outcomes["date"] + " " + outcomes["time"])
            outcomes = outcomes.drop_duplicates("dt", keep="last")
            n_matched = 0
            for _, row in outcomes.iterrows():
                dt = row["dt"]
                if dt in df.index:
                    is_buy = str(row["predicted_direction"]).upper().startswith("BUY")
                    is_win = row["outcome"] == "WIN"
                    if is_buy and is_win:
                        df.loc[dt, "Target"] = 2
                    elif not is_buy and not is_win:
                        df.loc[dt, "Target"] = 2
                    elif is_buy and not is_win:
                        df.loc[dt, "Target"] = 0
                    else:
                        df.loc[dt, "Target"] = 0
                    df.loc[dt, "sample_weight"] = 2.0
                    n_matched += 1
            if n_matched:
                log(f"[LEARN] 4H overrode {n_matched} labels (weight=2x)")
        elif len(outcomes) > 0:
            log(f"[LEARN] 4H skipped override: only {len(outcomes)} real outcomes")
    except Exception as e:
        log(f"[LEARN] 4H gagal query real outcomes: {e}")

    df, cols = _features_4h(df)
    df.dropna(subset=["Target"], inplace=True)
    if len(df) < 500:
        log(f"Data terlalu sedikit ({len(df)}), skip")
        return False

    X, y = df[cols].values, df["Target"].values
    sample_weights = df["sample_weight"].values if "sample_weight" in df.columns else None
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    folds, oot = trading.walk_forward_split(X, n_splits=3, embargo=FORWARD_BARS)
    oot_idx, _ = oot
    scaler = RobustScaler()
    scaler.fit(X[:len(X) - len(oot_idx)])

    models, scores = [], []
    n_classes = 3

    for k, (train_idx, val_idx) in enumerate(folds):
        X_t = scaler.transform(X[train_idx])
        X_v = scaler.transform(X[val_idx])
        y_t, y_v = y[train_idx], y[val_idx]
        sw_t = sample_weights[train_idx] if sample_weights is not None else None

        n_val = len(X_v)
        es_end = int(n_val * 0.7)
        X_es, X_eval = X_v[:es_end], X_v[es_end:]
        y_es, y_eval = y_v[:es_end], y_v[es_end:]

        fit_kwargs = {}
        if sw_t is not None:
            fit_kwargs["sample_weight"] = sw_t

        try:
            import optuna
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                    "max_depth": trial.suggest_int("max_depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                    "random_state": 42, "verbosity": 0,
                    "objective": "multi:softprob", "num_class": n_classes,
                }
                from xgboost import XGBClassifier
                m = XGBClassifier(**params)
                m.fit(X_t, y_t, eval_set=[(X_es, y_es)], verbose=False, **fit_kwargs)
                yp = m.predict(X_eval)
                return f1_score(y_eval, yp, average="macro")
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            best_params = study.best_params
        except ImportError:
            best_params = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.02,
                           "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3,
                           "reg_alpha": 0.1, "reg_lambda": 2.0}

        from xgboost import XGBClassifier
        model = XGBClassifier(objective="multi:softprob", num_class=n_classes,
                              random_state=42, verbosity=0, early_stopping_rounds=30, **best_params)
        model.fit(X_t, y_t, eval_set=[(X_es, y_es)], verbose=False, **fit_kwargs)

        yp = model.predict(X_eval)
        acc = accuracy_score(y_eval, yp)
        f1 = f1_score(y_eval, yp, average="macro")
        log(f"  Fold {k+1}: acc={acc:.1%} f1={f1:.1%}")
        models.append(model)
        scores.append({"acc": acc, "f1": f1})

    fold_weights = np.array([max(s["f1"], 0.01) for s in scores])
    fold_weights = fold_weights / fold_weights.sum()
    ensemble_models = list(zip(models, fold_weights))
    best_thresh = 0.55
    avg_acc = np.mean([s["acc"] for s in scores])

    oot_acc = None
    if len(oot_idx) > 5:
        X_oot = scaler.transform(X[oot_idx])
        y_oot = y[oot_idx]
        probs_oot = np.zeros((len(oot_idx), n_classes))
        for m, w in ensemble_models:
            probs_oot += w * m.predict_proba(X_oot)
        y_pred_oot = np.argmax(probs_oot, axis=1)
        oot_acc = float(accuracy_score(y_oot, y_pred_oot))
        log(f"  OOT holdout ({len(oot_idx)} samples): acc={oot_acc:.1%}")

    import joblib, tempfile
    _data = {"ensemble_models": ensemble_models, "scaler": scaler, "cols": cols,
              "threshold": best_thresh, "forward": FORWARD_BARS, "n_classes": n_classes,
              "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "samples": len(X), "test_acc": float(avg_acc), "oot_acc": oot_acc}
    _tmp = tempfile.NamedTemporaryFile(delete=False, dir=trading.BASE_DIR, suffix=".pkl")
    try:
        joblib.dump(_data, _tmp.name)
        os.replace(_tmp.name, MODEL_4H)
    except Exception:
        try: os.unlink(_tmp.name)
        except Exception: pass
        raise
    log(f"Model saved. Avg acc: {avg_acc:.1%}")
    return True


def predict():
    import joblib
    arts = joblib.load(MODEL_4H)
    scaler, cols = arts["scaler"], arts["cols"]
    best_thresh = arts.get("threshold", 0.55)
    forward = arts.get("forward", FORWARD_BARS)
    n_classes = arts.get("n_classes", 3)
    ensemble_models = arts.get("ensemble_models")
    single_model = arts.get("model")
    if ensemble_models is None and single_model is not None:
        ensemble_models = [(single_model, 1.0)]

    df = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    latest_price = float(df["Close"].iloc[-1])
    latest_time = df.index[-1]
    target_time = latest_time + timedelta(hours=forward * 4)

    from telegram_notifier import get_realtime_price
    live = get_realtime_price()
    entry = live["price"] if live else latest_price
    change_str = ""
    if live and live.get("change"):
        change_str = f" ({'+' if live['change'] > 0 else ''}{live['change']:.2f})"

    df_feat, _ = _features_4h(df)
    last_row = df_feat[cols].iloc[-1:]
    if len(last_row) == 0:
        log("No valid features for prediction")
        return None

    features_scaled = scaler.transform(last_row.values)
    min_thresh = max(0.55, best_thresh)

    if n_classes == 3:
        probs = np.zeros(n_classes)
        for m, w in ensemble_models:
            probs += w * m.predict_proba(features_scaled)[0]
        prob_bearish, prob_neutral, prob_bullish = probs
        if prob_bullish >= min_thresh:
            direction = "BUY"
            confidence = prob_bullish
        elif prob_bearish >= min_thresh:
            direction = "SELL"
            confidence = prob_bearish
        else:
            direction = "NO_TRADE"
            confidence = max(prob_bullish, prob_bearish)
        log(f"3-class: bear={prob_bearish:.1%} neutral={prob_neutral:.1%} bull={prob_bullish:.1%}")
    else:
        prob = float(sum(w * m.predict_proba(features_scaled)[0, 1] for m, w in ensemble_models))
        sell_thresh = 1.0 - min_thresh
        direction = "BUY" if prob >= min_thresh else "SELL" if prob <= sell_thresh else "NO_TRADE"
        confidence = prob

    log(f"4H Close: ${latest_price:.2f} | Realtime: ${entry:.2f}{change_str} | {direction} | conf: {confidence:.1%} | thresh: {min_thresh:.3f}")

    # Levels & TP/SL (aligned with compute_tp_sl multipliers)
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift(1)),
        abs(df["Low"] - df["Close"].shift(1))
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(14).mean().iloc[-1]
    sl = tp1 = tp2 = None
    if direction == "BUY":
        sl = round(entry - atr_val * 1.2, 2)
        tp1 = round(entry + atr_val * 0.8, 2)
        tp2 = round(entry + atr_val * 1.8, 2)
    elif direction == "SELL":
        sl = round(entry + atr_val * 1.2, 2)
        tp1 = round(entry - atr_val * 0.8, 2)
        tp2 = round(entry - atr_val * 1.8, 2)

    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions_4h (date, time, price, predicted_direction, confidence,
            threshold, target_date, target_time, sl, tp1, tp2, entry_realtime, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (latest_time.strftime("%Y-%m-%d"), latest_time.strftime("%H:%M"),
          latest_price, direction, confidence, min_thresh,
          target_time.strftime("%Y-%m-%d"), target_time.strftime("%Y-%m-%d %H:%M"),
          sl, tp1, tp2, entry, f"xgb_4h_v3"))
    pred_id = c.lastrowid
    conn.commit()
    conn.close()
    log(f"Saved to DB (ID: {pred_id})")

    if direction != "NO_TRADE":
        from telegram_notifier import send_4h_signal
        send_4h_signal(pred_id, direction, confidence, entry, latest_price,
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
                detail = outcome_detail
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
                      (outcome, detail, pct, pred_id))
            log(f"  #{pred_id}: {outcome} ({detail}) {pct:+.2f}%")
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
