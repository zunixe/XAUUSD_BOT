"""
XAUUSD AUTO RUNNER + LEARNING ENGINE
- Update data + prediksi + catat journal
- Auto-retrain model dari hasil prediksi sebelumnya
- Adaptive threshold (naikkin threshold sampe profit konsisten)
- Target: confidence >= 80% dengan profit konsisten
"""
import sys, os, time, subprocess, argparse, json
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import telegram_notifier
import trading
import patterns

MODEL_FILE = os.path.join(trading.BASE_DIR, "xauusd_model.pkl")
LEARNING_LOG = os.path.join(trading.BASE_DIR, "learning.csv")

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")
    with open(os.path.join(trading.BASE_DIR, "auto_runner.log"), "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def migrate_db():
    """Add columns for SL/TP tracking if missing"""
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    for col, dtype in [("sl", "REAL"), ("tp1", "REAL"), ("tp2", "REAL"),
                        ("entry_realtime", "REAL"), ("outcome_detail", "TEXT")]:
        try:
            c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {dtype}")
            log(f"DB: added column {col}")
        except sqlite3.OperationalError:
            pass
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN notified INTEGER DEFAULT 0")
        log("DB: added column notified")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# ========== FEATURE ENGINEERING ==========
_FOMC_DATES = [
    pd.Timestamp("2024-01-31"), pd.Timestamp("2024-03-20"),
    pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-12"),
    pd.Timestamp("2024-07-31"), pd.Timestamp("2024-09-18"),
    pd.Timestamp("2024-11-07"), pd.Timestamp("2024-12-18"),
    pd.Timestamp("2025-01-29"), pd.Timestamp("2025-03-19"),
    pd.Timestamp("2025-05-07"), pd.Timestamp("2025-06-18"),
    pd.Timestamp("2025-07-30"), pd.Timestamp("2025-09-17"),
    pd.Timestamp("2025-10-29"), pd.Timestamp("2025-12-10"),
    pd.Timestamp("2026-01-28"), pd.Timestamp("2026-03-18"),
    pd.Timestamp("2026-05-06"), pd.Timestamp("2026-06-17"),
    pd.Timestamp("2026-07-29"), pd.Timestamp("2026-09-16"),
    pd.Timestamp("2026-11-04"), pd.Timestamp("2026-12-14"),
]

def _add_fomc_features(df):
    """Add FOMC calendar features to dataframe."""
    fomc_series = pd.Series(0, index=df.index)
    for d in _FOMC_DATES:
        if d in df.index:
            fomc_series.loc[d] = 1
    df["Is_FOMC_Day"] = fomc_series.values
    df["Days_To_FOMC"] = np.nan
    fomc_sorted = sorted(_FOMC_DATES)
    for i, date in enumerate(df.index):
        future_fomc = [d for d in fomc_sorted if d >= date]
        if future_fomc:
            df.loc[date, "Days_To_FOMC"] = (future_fomc[0] - date).days
    df["Week_Before_FOMC"] = ((df["Days_To_FOMC"] >= 0) & (df["Days_To_FOMC"] <= 7)).astype(int)
    df["Week_After_FOMC"] = ((df["Days_To_FOMC"] >= -7) & (df["Days_To_FOMC"] < 0)).astype(int)
    return df

def _engineer_features_full(df):
    """Full daily features (shared core + macro + FOMC + pattern detection)."""
    df = trading.engineer_features_daily(df)
    df = patterns.detect_candlestick(df)
    df = patterns.detect_support_resistance(df)
    df = patterns.detect_double_top_bottom(df)
    df = _add_fomc_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    exclude = ["Target", "Close", "High", "Low", "Open", "Volume",
               "DXY_Close", "VIX_Close", "SPY_Close", "US10Y_Close", "OIL_Close",
               "sample_weight"]
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols

# ========== TRAINING ENGINE (Walk-forward CV + ATR labeling + Optuna) ==========
# Uses shared implementations from trading.py


def retrain_model():
    """Retrain XGBoost with walk-forward CV + ATR labeling + Optuna tuning"""
    log("[LEARN] Retraining model...")
    df = pd.read_csv(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    forward_days = 3
    df["Target"] = trading.atr_target(df, forward=forward_days, atr_mult=0.8, sl_mult=0.6)

    # Override ATR labels with real SL/TP outcomes where available
    df["sample_weight"] = 1.0
    try:
        import sqlite3
        conn = sqlite3.connect(trading.DB_FILE)
        outcomes = pd.read_sql_query(
            "SELECT date, predicted_direction, outcome FROM predictions"
            " WHERE outcome IN ('WIN','LOSS') AND sl IS NOT NULL AND tp1 IS NOT NULL ORDER BY id", conn
        )
        conn.close()
        n_total = len(df)
        if len(outcomes) >= 5 and len(outcomes) / n_total >= 0.05:
            outcomes["date"] = pd.to_datetime(outcomes["date"]).dt.date
            outcomes = outcomes.drop_duplicates("date", keep="last")
            n_applied = 0
            for _, row in outcomes.iterrows():
                dt = row["date"]
                mask = df.index.date == dt
                if mask.any():
                    is_buy = str(row["predicted_direction"]).upper().startswith("BUY")
                    is_win = row["outcome"] == "WIN"
                    label = 1.0 if (is_buy and is_win) or (not is_buy and not is_win) else 0.0
                    df.loc[mask, "Target"] = label
                    df.loc[mask, "sample_weight"] = 2.0
                    n_applied += 1
            n_win = int((outcomes["outcome"] == "WIN").sum())
            n_loss = int((outcomes["outcome"] == "LOSS").sum())
            log(f"[LEARN] Overrode {n_applied} ATR labels with real outcomes ({n_win} WIN / {n_loss} LOSS, weight=2x)")
        elif len(outcomes) > 0:
            log(f"[LEARN] Skipped override: only {len(outcomes)} real outcomes (<5% of {n_total} rows)")
    except Exception as e:
        log(f"[LEARN] Gagal query real outcomes: {e}")

    df, feature_cols = _engineer_features_full(df)
    df.dropna(inplace=True)

    if len(df) < 300:
        log(f"[LEARN] Data terlalu sedikit ({len(df)}), skip retrain")
        return False

    X = df[feature_cols].values
    y = df["Target"].values
    sample_weights = df["sample_weight"].values if "sample_weight" in df.columns else None

    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve

    scaler = RobustScaler()
    folds, oot = trading.walk_forward_split(X, n_splits=4, embargo=3)
    models, scores, thresholds = [], [], []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sample_weights[train_idx] if sample_weights is not None else None
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale = neg / pos if pos > 0 else 1

        model, params = trading.optuna_tune(X_train, y_train, X_val, y_val, scale, n_trials=50,
                                            sample_weight=sw_train)
        y_prob = model.predict_proba(X_val)[:, 1]

        precisions, recalls, threshs = precision_recall_curve(y_val, y_prob)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_t = threshs[np.argmax(f1s[:-1])] if len(threshs) > 0 else 0.5
        y_pred = (y_prob >= best_t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        log(f"  Fold {fold_i + 1}/{len(folds)}: acc={acc:.1%} f1={f1:.1%} threshold={best_t:.3f} | {len(X_train)} train, {len(X_val)} val")
        models.append(model)
        scores.append({"acc": acc, "f1": f1})
        thresholds.append(best_t)

    # Ensemble: weighted average of all fold models
    fold_weights = np.array([max(s["f1"], 0.01) for s in scores])
    fold_weights = fold_weights / fold_weights.sum()
    ensemble_models = list(zip(models, fold_weights))

    # Final threshold: average across folds, adjusted by historical accuracy
    best_thresh = max(np.mean(thresholds), 0.50)

    # --- FEEDBACK LOOP (bidirectional with time decay) ---
    conn = sqlite3.connect(trading.DB_FILE)
    journal_df = pd.read_sql("SELECT * FROM predictions WHERE outcome IS NOT NULL", conn)
    conn.close()

    if len(journal_df) >= 10:
        from datetime import timedelta
        recent_cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        journal_df["date_str"] = journal_df["date"].astype(str).str[:10]
        recent = journal_df[journal_df["date_str"] >= recent_cutoff]
        older = journal_df[journal_df["date_str"] < recent_cutoff]
        recent_wins = int((recent["outcome"] == "WIN").sum())
        older_wins = int((older["outcome"] == "WIN").sum())
        weighted_wins = recent_wins * 2 + older_wins
        weighted_total = len(recent) * 2 + len(older)
        hist_acc = weighted_wins / weighted_total if weighted_total > 0 else 0.5
        log(f"[LEARN] Historical accuracy (time-weighted): {hist_acc:.1%} (recent {len(recent)}, older {len(older)})")
        if hist_acc < 0.45:
            best_thresh = min(best_thresh + 0.10, 0.75)
        elif hist_acc < 0.55:
            best_thresh = min(best_thresh + 0.05, 0.70)
        elif hist_acc >= 0.65:
            best_thresh = max(best_thresh - 0.03, 0.55)

    avg_acc = np.mean([s["acc"] for s in scores])
    avg_f1 = np.mean([s["f1"] for s in scores])

    # OOT evaluation (never touched during training)
    oot_acc = None
    oot_idx, _ = oot
    if len(oot_idx) > 5:
        X_oot, y_oot = X[oot_idx], y[oot_idx]
        X_oot_scaled = scaler.transform(X_oot)
        y_prob_oot = np.zeros(len(oot_idx))
        for m, w in ensemble_models:
            y_prob_oot += w * m.predict_proba(X_oot_scaled)[:, 1]
        y_pred_oot = (y_prob_oot >= best_thresh).astype(int)
        from sklearn.metrics import accuracy_score
        oot_acc = float(accuracy_score(y_oot, y_pred_oot))
        log(f"  OOT holdout ({len(oot_idx)} samples): acc={oot_acc:.1%}")

    import joblib, tempfile
    _data = {"ensemble_models": ensemble_models, "scaler": scaler,
              "feature_cols": feature_cols,
              "best_thresh": float(best_thresh), "forward_days": forward_days,
              "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "train_samples": len(X), "test_acc": float(avg_acc), "f1": float(avg_f1),
              "oot_acc": oot_acc, "params": params, "fold_scores": scores}
    try:
        joblib.dump(_data, MODEL_FILE)
    except Exception:
        _tmp = tempfile.NamedTemporaryFile(delete=False, dir=trading.BASE_DIR, suffix=".pkl")
        try:
            joblib.dump(_data, _tmp.name)
            os.replace(_tmp.name, MODEL_FILE)
        except Exception:
            os.unlink(_tmp.name)
            raise

    log(f"[LEARN] Retrain selesai! Avg acc: {avg_acc:.1%} | Threshold: {best_thresh:.3f} | F1: {avg_f1:.1%}")
    return True

# ========== PREDIKSI & CATAT ==========
def run_prediction_job():
    log("Job: Update data...")
    p = sys.executable
    r = subprocess.run([p, "update_data.py"], capture_output=True, text=True, cwd=trading.BASE_DIR)
    if r.returncode != 0:
        log(f"Gagal update data: {r.stderr[:150]}")
        return None

    import joblib
    artifacts = joblib.load(MODEL_FILE)
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    best_thresh = artifacts["best_thresh"]
    forward_days = artifacts.get("forward_days", 3)
    ensemble_models = artifacts.get("ensemble_models")
    single_model = artifacts.get("model")
    if ensemble_models is None and single_model is not None:
        single_weight = 1.0 / len(single_model) if isinstance(single_model, list) else 1.0
        ensemble_models = [(single_model, 1.0)]

    df = pd.read_csv(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    latest_price = float(df["Close"].iloc[-1])
    latest_date = df.index[-1]

    df, _ = _engineer_features_full(df)
    last_row = df[feature_cols].dropna().iloc[-1:]
    features_scaled = scaler.transform(last_row.values)
    prob = float(sum(w * m.predict_proba(features_scaled)[0, 1] for m, w in ensemble_models))
    # Minimum threshold untuk real trading (hindari sinyal lemah)
    min_thresh = max(0.55, best_thresh)
    sell_thresh = 1.0 - min_thresh
    direction = "BUY (Bullish)" if prob >= min_thresh else "SELL (Bearish)" if prob <= sell_thresh else "NO_TRADE"
    target_date = (latest_date + timedelta(days=forward_days)).strftime("%Y-%m-%d")

    log(f"Hasil: ${latest_price:.2f} | {direction} | confidence: {prob:.1%} | threshold: {min_thresh:.3f} | target: {target_date}")

    # Hitung levels & TP/SL hanya untuk BUY/SELL
    trade_df = trading.load_daily(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"))
    levels = trading.calculate_levels(trade_df)
    sl = tp1 = tp2 = None
    entry_realtime = None
    if direction != "NO_TRADE":
        is_buy = direction.startswith("BUY")
        sl, tp1, tp2 = trading.compute_tp_sl(levels, is_buy, latest_price)

    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (date, price, predicted_direction, confidence, threshold,
                                 target_date, model_version, sl, tp1, tp2, entry_realtime)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (latest_date.strftime("%Y-%m-%d"), latest_price, direction, prob, min_thresh,
          target_date, f"xgb_v2_acc{float(artifacts.get('test_acc',0)):.0%}",
          sl, tp1, tp2, entry_realtime))
    conn.commit()
    pred_id = c.lastrowid
    if sl is not None:
        log(f"Tersimpan ke journal (ID: {pred_id}) | SL: ${sl:.2f} TP1: ${tp1:.2f} TP2: ${tp2:.2f}")
    else:
        log(f"Tersimpan ke journal (ID: {pred_id}) | NO_TRADE (no SL/TP)")

    # Kirim notifikasi Telegram hanya untuk BUY/SELL
    if direction != "NO_TRADE":
        result = telegram_notifier.send_signal(pred_id, direction, prob, latest_price, target_date, min_thresh)
        if result:
            c.execute("UPDATE predictions SET entry_realtime=? WHERE id=?", (result["entry_realtime"], pred_id))
            conn.commit()
    else:
        log(f"Signal lemah ({prob:.1%} di zona {sell_thresh:.2f}-{min_thresh:.2f}), tidak kirim notifikasi")

    conn.close()
    return pred_id

# ========== EVALUASI (OHLC-based SL/TP) ==========
def evaluate_past_predictions():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, date, price, sl, tp1, tp2, predicted_direction
        FROM predictions WHERE outcome IS NULL
        ORDER BY id
    """).fetchall()
    if not rows:
        conn.close()
        return 0
    log(f"Evaluasi {len(rows)} prediksi...")
    df = trading.load_daily(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"))
    evaluated = 0
    for pred_id, pred_date, entry_price, sl, tp1, tp2, direction in rows:
        try:
            entry_idx = df.index.get_loc(pd.Timestamp(pred_date))
            # Skip if no forward data yet
            if entry_idx + 1 >= len(df):
                continue
            is_buy = "BUY" in direction
            if sl is None or tp1 is None:
                actual_close = float(df.iloc[entry_idx + 1:]["Close"].iloc[0])
                pct = (actual_close - entry_price) / entry_price * 100
                outcome = "WIN" if (is_buy and pct > 0) or (not is_buy and pct < 0) else "LOSS"
                if not is_buy:
                    pct = -pct  # SELL: profit when price goes down
                c.execute("UPDATE predictions SET outcome=?, result_pct=?, outcome_detail='NO_SLTP' WHERE id=?",
                          (outcome, pct, pred_id))
                log(f"  #{pred_id}: {outcome} (NO_SLTP) pct: {pct:+.2f}%")
            else:
                outcome, detail, exit_price, pct = trading.evaluate_sl_tp(
                    df, entry_idx, entry_price, sl, tp1, tp2, is_buy, max_lookahead=5
                )
                c.execute("UPDATE predictions SET outcome=?, result_pct=?, outcome_detail=? WHERE id=?",
                          (outcome, pct, detail, pred_id))
                hit = f"{detail} @ ${exit_price:.2f}"
                emoji = "+" if outcome == "WIN" else "-"
                log(f"  #{pred_id}: {outcome} ({hit}) pct: {emoji}{abs(pct):.2f}%")
            evaluated += 1
        except KeyError:
            continue
        except Exception as e:
            log(f"  #{pred_id}: Gagal - {e}")
    conn.commit()
    conn.close()

    if evaluated > 0:
        _notify_daily_outcomes()
    return evaluated


def _notify_daily_outcomes():
    """Send Telegram notification for recently evaluated daily signals"""
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, price, predicted_direction, confidence, outcome, outcome_detail, result_pct,
               sl, tp1, tp2
        FROM predictions WHERE outcome IS NOT NULL AND (notified IS NULL OR notified = 0)
    """).fetchall()
    if not rows:
        conn.close()
        return
    for row in rows:
        try:
            from telegram_notifier import send_outcome_notification
            send_outcome_notification(prediction_id=row[0], timeframe="Daily",
                direction=row[2], entry=row[1], outcome=row[4],
                detail=row[5], pct=row[6], sl=row[7], tp1=row[8], tp2=row[9])
            c.execute("UPDATE predictions SET notified=1 WHERE id=?", (row[0],))
            log(f"  Outcome notified: #{row[0]} {row[4]}")
        except Exception as e:
            log(f"  Notify error #{row[0]}: {e}")
    conn.commit()
    conn.close()


def monthly_report():
    """Send monthly report for daily signals"""
    conn = sqlite3.connect(trading.DB_FILE)
    df = pd.read_sql("SELECT * FROM predictions WHERE outcome IS NOT NULL", conn)
    conn.close()
    if len(df) == 0:
        return
    df["date"] = pd.to_datetime(df["date"])
    now = pd.Timestamp.now()
    month_ago = now - timedelta(days=30)
    month = df[df["date"] >= month_ago]
    if len(month) == 0:
        return
    wins = len(month[month["outcome"] == "WIN"])
    total = len(month)
    acc = wins / total * 100
    ret = month["result_pct"].sum()
    pf = abs(month[month["result_pct"] > 0]["result_pct"].sum() /
             (month[month["result_pct"] < 0]["result_pct"].sum() + 1e-10))
    msg = (
        f"<b>[DAILY MONTHLY REPORT]</b>\n"
        f"<pre>\n"
        f"  Trades      : {total}\n"
        f"  Win Rate    : {acc:.1f}% ({wins}W / {total-wins}L)\n"
        f"  Total Return: {ret:+.2f}%\n"
        f"  Profit Fact : {pf:.2f}\n"
        f"</pre>"
    )
    try:
        from telegram_notifier import send_report
        send_report(msg)
        log("Monthly report sent")
    except Exception as e:
        log(f"Monthly report error: {e}")

# ========== STATISTIK + THRESHOLD OPTIMIZATION ==========
def show_learning_stats():
    conn = sqlite3.connect(trading.DB_FILE)
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id", conn)
    conn.close()

    if len(df) == 0:
        log("Belum ada data di journal")
        return

    total = len(df)
    evaluated = df[df["outcome"].notna()]
    uneval = df[df["outcome"].isna()]

    print(f"\n{'='*55}")
    print(f"  LEARNING REPORT - {datetime.now().strftime('%d %b %Y %H:%M')}")
    print(f"{'='*55}")

    if len(evaluated) > 0:
        wins = len(evaluated[evaluated["outcome"] == "WIN"])
        losses = len(evaluated) - wins
        acc = wins / len(evaluated) * 100
        avg_return = evaluated["result_pct"].mean()
        total_return = evaluated["result_pct"].sum()
        profit_factor = abs(evaluated[evaluated["result_pct"] > 0]["result_pct"].sum() /
                          (evaluated[evaluated["result_pct"] < 0]["result_pct"].sum() + 1e-10))

        print(f"  Total prediksi      : {total}")
        print(f"  Sudah dievaluasi    : {len(evaluated)} ({len(uneval)} pending)")
        print(f"  Akurasi             : {acc:.1f}% ({wins}W / {losses}L)")
        print(f"  Rata-rata return    : {avg_return:+.2f}%")
        print(f"  Total return        : {total_return:+.2f}%")
        print(f"  Profit factor       : {profit_factor:.2f}")
        print(f"{'='*55}")

        # Performance by confidence bracket
        print(f"  PERFORMANCE BY CONFIDENCE:")
        brackets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
        for lo, hi in brackets:
            subset = evaluated[(evaluated["confidence"] >= lo) & (evaluated["confidence"] < hi)]
            if len(subset) >= 3:
                w = len(subset[subset["outcome"] == "WIN"])
                r = subset["result_pct"].mean()
                print(f"    conf {lo:.0%}-{hi:.0%}: {w}/{len(subset)} ({w/len(subset)*100:.0f}%) avg_return: {r:+.2f}%")
    else:
        print(f"  Total prediksi: {total} (belum ada yang dievaluasi)")

    # Jika cukup data, rekomendasi threshold
    if len(evaluated) >= 10:
        best_profit = -999
        best_t = 0.5
        for t in np.arange(0.5, 0.90, 0.025):
            subset = evaluated[evaluated["confidence"] >= t]
            if len(subset) >= 3:
                w = len(subset[subset["outcome"] == "WIN"])
                ret = subset["result_pct"].sum()
                if ret > best_profit:
                    best_profit = ret
                    best_t = t
                    best_wr = w / len(subset)
        print(f"  Threshold rekomendasi : {best_t:.3f} (win rate: {best_wr:.0%}, profit: {best_profit:+.2f}%)")

    print(f"\n  TARGET: Akurasi >= 80%")
    if len(evaluated) >= 10:
        high_conf = evaluated[evaluated["confidence"] >= 0.7]
        if len(high_conf) >= 5:
            w = len(high_conf[high_conf["outcome"] == "WIN"])
            print(f"  Progress: {w}/{len(high_conf)} ({w/len(high_conf)*100:.0f}%) pada confidence >= 70%")
        else:
            print(f"  Progress: butuh lebih banyak data high-confidence (baru {len(high_conf)} prediksi >=70%)")
        print(f"  Estimasi: ~50-100 prediksi terkumpul untuk threshold optimal")
    else:
        print(f"  Progress: baru {len(evaluated)} prediksi dievaluasi (butuh >=10)")
    print(f"{'='*55}")

# ========== AUTO-RETRAIN CHECK ==========
def should_retrain():
    """Retrain jika ada >= 10 evaluasi baru sejak retrain terakhir"""
    try:
        import joblib
        arts = joblib.load(MODEL_FILE)
        last_train = arts.get("train_date", "2000-01-01")
    except Exception:
        last_train = "2000-01-01"

    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    count = c.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE outcome IS NOT NULL AND date > ?
    """, (last_train[:10],)).fetchone()[0]
    conn.close()
    return count >= 10

# ========== DAEMON ==========
def run_daemon(interval_hours=4):
    log(f"AUTO RUNNER started (interval: {interval_hours} jam)")
    # Start Telegram polling thread
    import threading
    try:
        from telegram_notifier import process_commands
        def _loop():
            import time
            while True:
                try:
                    process_commands()
                except Exception:
                    pass
                time.sleep(30)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        log("Telegram polling started")
    except Exception as e:
        log(f"Polling thread error: {e}")
    last_monthly = ""
    while True:
        try:
            evaluated = evaluate_past_predictions()
            if should_retrain() or evaluated >= 10:
                retrain_model()
            run_prediction_job()
            show_learning_stats()

            # Telegram commands
            try:
                from telegram_notifier import process_commands
                process_commands()
            except Exception:
                pass

            # 4H runner
            _run_4h_cycle()

            # Monthly report (once per month)
            month_key = datetime.now().strftime("%Y-%m")
            if month_key != last_monthly:
                monthly_report()
                last_monthly = month_key

            log(f"Tidur {interval_hours} jam...")
            log("-" * 55)
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            log("Stopped.")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(300)


def _run_4h_cycle():
    """Run 4H evaluation + prediction in a subprocess"""
    try:
        p = sys.executable
        script = os.path.join(trading.BASE_DIR, "runner_4h.py")
        # Evaluate
        r = subprocess.run([p, script, "--evaluate"], capture_output=True, text=True, cwd=trading.BASE_DIR, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().split("\n"):
                log(f"[4H] {line.strip()}")
        # Predict
        r2 = subprocess.run([p, script, "--predict"], capture_output=True, text=True, cwd=trading.BASE_DIR, timeout=60)
        if r2.returncode == 0 and r2.stdout.strip():
            for line in r2.stdout.strip().split("\n"):
                log(f"[4H] {line.strip()}")
        # Weekly report
        r3 = subprocess.run([p, script, "--report"], capture_output=True, text=True, cwd=trading.BASE_DIR, timeout=30)
    except subprocess.TimeoutExpired:
        log("[4H] Runner timeout")
    except Exception as e:
        log(f"[4H] Error: {e}")

# ========== WINDOWS TASK ==========
def install_windows_task(interval_hours=4):
    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    task_name = "XAUUSD Learning Engine"
    ps_cmd = (
        f'$a=New-ScheduledTaskAction -Execute "{python_exe}" -Argument "{script_path}" -WorkingDirectory "{trading.BASE_DIR}"; '
        f'$t=New-ScheduledTaskTrigger -Once -At (Get-Date "00:00") -RepetitionInterval (New-TimeSpan -Hours {interval_hours}) -RepetitionDuration (New-TimeSpan -Days 365); '
        f'Register-ScheduledTask -TaskName "{task_name}" -Action $a -Trigger $t -RunLevel Highest -Force'
    )
    r = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
    if r.returncode == 0:
        log(f"Windows Task terinstall: '{task_name}' ({interval_hours} jam)")
    else:
        log(f"Gagal: {r.stderr.strip()}")
    # Run once now
    subprocess.Popen([python_exe, script_path], cwd=trading.BASE_DIR)

def uninstall_windows_task():
    r = subprocess.run('schtasks /delete /tn "XAUUSD Learning Engine" /f', shell=True, capture_output=True, text=True)
    log(f"Task dihapus" if r.returncode == 0 else f"Gagal: {r.stderr}")

# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="Force retrain model")
    parser.add_argument("--stats", action="store_true", help="Tampilkan learning report")
    args = parser.parse_args()

    os.chdir(trading.BASE_DIR)
    migrate_db()

    if args.install:
        install_windows_task(args.interval)
    elif args.uninstall:
        uninstall_windows_task()
    elif args.retrain:
        retrain_model()
        show_learning_stats()
    elif args.stats:
        show_learning_stats()
    elif args.daemon:
        run_daemon(args.interval)
    else:
        print("=" * 55)
        print(f"  XAUUSD LEARNING ENGINE - {datetime.now().strftime('%d %b %Y %H:%M')}")
        print("=" * 55)
        evaled = evaluate_past_predictions()
        if should_retrain() or evaled >= 10:
            retrain_model()
        run_prediction_job()
        show_learning_stats()
        try:
            from telegram_notifier import process_commands
            process_commands()
        except Exception:
            pass
        _run_4h_cycle()
        monthly_report()
        print(f"  Usage: python auto_runner.py --daemon | --install | --retrain | --stats")
        print("=" * 55)
