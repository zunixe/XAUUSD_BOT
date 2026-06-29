"""
XAUUSD AUTO RUNNER + LEARNING ENGINE (V2)
- 3-class model with regime detection, MTF confirmation, risk management
- Heartbeat, model monitoring, graceful shutdown, trailing stops
"""
import sys, os, time, subprocess, argparse, json, signal, shutil
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
_shutdown = False

def _handle_shutdown(signum, frame):
    global _shutdown
    log(f"Received signal {signum}, shutting down...")
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")
    with open(os.path.join(trading.BASE_DIR, "auto_runner.log"), "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def migrate_db():
    """Add columns for SL/TP tracking if missing + DB indexes"""
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
    # Performance indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(outcome)",
        "CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(date)",
        "CREATE INDEX IF NOT EXISTS idx_pred_notified ON predictions(notified)",
        "CREATE INDEX IF NOT EXISTS idx_4h_outcome ON predictions_4h(outcome)",
        "CREATE INDEX IF NOT EXISTS idx_4h_date ON predictions_4h(date)",
        "CREATE INDEX IF NOT EXISTS idx_sim_active ON simulation(active)",
        "CREATE INDEX IF NOT EXISTS idx_trades_sim ON sim_trades(sim_id)",
        "CREATE INDEX IF NOT EXISTS idx_trades_date ON sim_trades(created_at)",
    ]
    for sql in indexes:
        try:
            c.execute(sql)
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
    """Add FOMC calendar features to dataframe using bisect for O(n log k)."""
    import bisect
    fomc_sorted = sorted(_FOMC_DATES)
    last_fomc = fomc_sorted[-1] if fomc_sorted else None
    if last_fomc and pd.Timestamp.now() > last_fomc:
        log(f"[WARN] FOMC dates list expired (last: {last_fomc.date()}). Days_To_FOMC will be NaN for future dates.")

    # Is_FOMC_Day
    fomc_set = set(fomc_sorted)
    df["Is_FOMC_Day"] = df.index.isin(fomc_set).astype(int)

    # Days_To_FOMC using bisect (O(n log k) instead of O(n*k))
    fomc_ts = [d.value for d in fomc_sorted]
    dates_ts = df.index.values.astype("int64")
    days_to = np.full(len(df), np.nan)
    for i, ts in enumerate(dates_ts):
        idx = bisect.bisect_left(fomc_ts, ts)
        if idx < len(fomc_sorted):
            days_to[i] = (fomc_sorted[idx] - df.index[i]).days
    df["Days_To_FOMC"] = days_to
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
               "TIP_Close", "Silver_Close", "BTC_Close", "USDJPY_Close", "EURUSD_Close",
               "Copper_Close", "Breakeven_5Y", "Breakeven_10Y", "GPR_Index",
               "sample_weight"]
    feature_cols = [c for c in df.columns if c not in exclude]
    # Fill NaN in features with 0 (don't drop rows - some macro data has partial coverage)
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols

# ========== TRAINING ENGINE (Walk-forward CV + ATR labeling + Optuna) ==========
# Uses shared implementations from trading.py


def retrain_model():
    """Retrain XGBoost with 3-class labeling, walk-forward CV, single scaler, Optuna tuning."""
    log("[LEARN] Retraining 3-class model...")
    df = pd.read_csv(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    forward_days = 3
    # 3-class labels: 2=BULLISH, 1=NEUTRAL, 0=BEARISH
    # Multipliers aligned with compute_tp_sl: TP=0.8 ATR, SL=1.2 ATR
    df["Target"] = trading.atr_target(df, forward=forward_days, atr_mult=0.8, sl_mult=1.2)

    # Override labels with real outcomes (directional, with time-decay weighting)
    df["sample_weight"] = 1.0
    try:
        conn = sqlite3.connect(trading.DB_FILE)
        outcomes = pd.read_sql_query(
            "SELECT date, predicted_direction, outcome FROM predictions"
            " WHERE outcome IN ('WIN','LOSS') AND sl IS NOT NULL AND tp1 IS NOT NULL ORDER BY id", conn
        )
        conn.close()
        n_total = len(df)
        if len(outcomes) >= 10 and len(outcomes) / n_total >= 0.05:
            outcomes["date"] = pd.to_datetime(outcomes["date"]).dt.date
            outcomes = outcomes.drop_duplicates("date", keep="last")
            n_applied = 0
            now_date = datetime.now().date()
            for _, row in outcomes.iterrows():
                dt = row["date"]
                mask = df.index.date == dt
                if mask.any():
                    is_buy = str(row["predicted_direction"]).upper().startswith("BUY")
                    is_win = row["outcome"] == "WIN"
                    if is_buy and is_win:
                        df.loc[mask, "Target"] = 2
                    elif not is_buy and not is_win:
                        df.loc[mask, "Target"] = 2
                    elif is_buy and not is_win:
                        df.loc[mask, "Target"] = 0
                    else:
                        df.loc[mask, "Target"] = 0
                    # Time-decay: recent outcomes weighted more
                    age_days = (now_date - dt).days
                    weight = max(1.5, 3.0 - age_days * 0.03)  # 3.0 today, 1.5 after 50 days, min 1.5
                    df.loc[mask, "sample_weight"] = weight
                    n_applied += 1
            log(f"[LEARN] Overrode {n_applied} labels with real outcomes (time-decay weight)")
        elif len(outcomes) > 0:
            log(f"[LEARN] Skipped override: only {len(outcomes)} real outcomes")
    except Exception as e:
        log(f"[LEARN] Gagal query real outcomes: {e}")

    df, feature_cols = _engineer_features_full(df)
    df.dropna(subset=["Target"], inplace=True)

    if len(df) < 300:
        log(f"[LEARN] Data terlalu sedikit ({len(df)}), skip retrain")
        return False

    X = df[feature_cols].values
    y = df["Target"].values
    sample_weights = df["sample_weight"].values if "sample_weight" in df.columns else None

    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    folds, oot = trading.walk_forward_split(X, n_splits=4, embargo=3)
    oot_idx, _ = oot

    models, scores = [], []
    n_classes = 3
    scaler = None

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        # Fit scaler ONLY on training data (no leakage)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sample_weights[train_idx] if sample_weights is not None else None

        # Split val: 70% early-stop, 30% eval
        n_val = len(X_val)
        es_end = int(n_val * 0.7)
        X_es, X_eval = X_val[:es_end], X_val[es_end:]
        y_es, y_eval = y_val[:es_end], y_val[es_end:]

        # Class weights for 3-class imbalance
        class_counts = np.bincount(y_train, minlength=3).astype(float)
        scale = class_counts.max() / (class_counts + 1e-10)

        fit_kwargs = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train

        try:
            import optuna
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                    "max_depth": trial.suggest_int("max_depth", 4, 7),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                    "random_state": 42,
                    "verbosity": 0,
                    "objective": "multi:softprob",
                    "num_class": n_classes,
                }
                from xgboost import XGBClassifier
                m = XGBClassifier(**params)
                m.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=False, **fit_kwargs)
                yp = m.predict(X_eval)
                return f1_score(y_eval, yp, average="macro")

            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            best_params = study.best_params
        except ImportError:
            best_params = {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.02,
                           "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3,
                           "reg_alpha": 0.1, "reg_lambda": 2.0}

        from xgboost import XGBClassifier
        model = XGBClassifier(objective="multi:softprob", num_class=n_classes,
                              random_state=42, verbosity=0, early_stopping_rounds=30, **best_params)
        model.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=False, **fit_kwargs)

        y_pred = model.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average="macro")

        log(f"  Fold {fold_i + 1}/{len(folds)}: acc={acc:.1%} f1={f1:.1%} | {len(X_train)} train, {len(X_val)} val")
        models.append(model)
        scores.append({"acc": acc, "f1": f1})

    # Ensemble weights
    fold_weights = np.array([max(s["f1"], 0.01) for s in scores])
    fold_weights = fold_weights / fold_weights.sum()
    ensemble_models = list(zip(models, fold_weights))
    best_thresh = trading.CFG.get("model", {}).get("min_threshold", 0.55)

    # --- FEEDBACK LOOP (min 30 evaluations) ---
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(trading.DB_FILE)
        journal_df = pd.read_sql("SELECT * FROM predictions WHERE outcome IS NOT NULL", conn)
        conn.close()
        if len(journal_df) >= 30:
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
                best_thresh = min(best_thresh + 0.08, 0.70)
            elif hist_acc < 0.55:
                best_thresh = min(best_thresh + 0.03, 0.65)
            elif hist_acc >= 0.65:
                best_thresh = max(best_thresh - 0.02, 0.45)
        else:
            log(f"[LEARN] Feedback loop: butuh 30+ evaluasi (baru {len(journal_df) if 'journal_df' in dir() else 0})")
    except Exception as e:
        log(f"[LEARN] Feedback loop error: {e}")

    avg_acc = np.mean([s["acc"] for s in scores])
    avg_f1 = np.mean([s["f1"] for s in scores])

    # OOT evaluation using last fold's scaler
    oot_acc = None
    if len(oot_idx) > 5 and scaler is not None:
        X_oot = scaler.transform(X[oot_idx])
        y_oot = y[oot_idx]
        # Ensemble 3-class prediction
        probs_oot = np.zeros((len(oot_idx), n_classes))
        for m, w in ensemble_models:
            probs_oot += w * m.predict_proba(X_oot)
        y_pred_oot = np.argmax(probs_oot, axis=1)
        oot_acc = float(accuracy_score(y_oot, y_pred_oot))
        oot_report = classification_report(y_oot, y_pred_oot, target_names=["BEARISH", "NEUTRAL", "BULLISH"], output_dict=True)
        log(f"  OOT holdout ({len(oot_idx)} samples): acc={oot_acc:.1%}")
        log(f"    BULLISH precision: {oot_report['BULLISH']['precision']:.1%} recall: {oot_report['BULLISH']['recall']:.1%}")
        log(f"    BEARISH precision: {oot_report['BEARISH']['precision']:.1%} recall: {oot_report['BEARISH']['recall']:.1%}")

    # Model versioning
    import joblib, tempfile
    version = datetime.now().strftime("%Y%m%d_%H%M")
    versioned_file = os.path.join(trading.BASE_DIR, f"xauusd_model_{version}.pkl")
    _data = {"ensemble_models": ensemble_models, "scaler": scaler,
              "feature_cols": feature_cols, "n_classes": n_classes,
              "best_thresh": float(best_thresh), "forward_days": forward_days,
              "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "train_samples": len(X), "test_acc": float(avg_acc), "f1": float(avg_f1),
              "oot_acc": oot_acc, "fold_scores": scores, "model_version": version}
    _tmp = tempfile.NamedTemporaryFile(delete=False, dir=trading.BASE_DIR, suffix=".pkl")
    try:
        joblib.dump(_data, _tmp.name)
        os.replace(_tmp.name, MODEL_FILE)
        # Save versioned copy
        shutil.copy2(MODEL_FILE, versioned_file)
    except Exception:
        try:
            os.unlink(_tmp.name)
        except Exception:
            pass
        raise

    log(f"[LEARN] Retrain selesai! Avg acc: {avg_acc:.1%} | Threshold: {best_thresh:.3f} | F1: {avg_f1:.1%} | Version: {version}")

    # Feature selection with SHAP (Phase 2.3)
    try:
        from feature_selector import select_features
        selected, importance = select_features(models[-1], X_oot, feature_cols)
        _data["selected_features"] = selected
        _data["feature_importance"] = {k: float(v) for k, v in importance.items()}
    except Exception as e:
        log(f"[SHAP] Feature selection skipped: {e}")

    return True

# ========== PREDIKSI & CATAT ==========
def run_prediction_job():
    log("Job: Update data...")
    p = sys.executable
    r = subprocess.run([p, "update_data.py"], capture_output=True, text=True, cwd=trading.BASE_DIR)
    if r.returncode != 0:
        log(f"Gagal update data: {r.stderr[:500]}")
        return None

    import joblib
    artifacts = joblib.load(MODEL_FILE)
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    best_thresh = artifacts["best_thresh"]
    forward_days = artifacts.get("forward_days", 3)
    n_classes = artifacts.get("n_classes", 3)
    ensemble_models = artifacts.get("ensemble_models")
    single_model = artifacts.get("model")
    if ensemble_models is None and single_model is not None:
        ensemble_models = [(single_model, 1.0)]

    df = pd.read_csv(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    latest_price = float(df["Close"].iloc[-1])
    latest_date = df.index[-1]

    # Compute levels and daily trend from raw df (before feature engineering)
    levels = trading.calculate_levels(df)
    daily_trend = trading.get_daily_trend(df)

    df, _ = _engineer_features_full(df)
    last_row = df[feature_cols].dropna().iloc[-1:]
    features_scaled = scaler.transform(last_row.values)

    # 3-class ensemble prediction: [bearish, neutral, bullish]
    if n_classes == 3:
        probs = np.zeros(n_classes)
        for m, w in ensemble_models:
            probs += w * m.predict_proba(features_scaled)[0]
        prob_bearish, prob_neutral, prob_bullish = probs
        min_thresh = min(trading.CFG.get("model", {}).get("min_threshold", 0.55), best_thresh)
        if prob_bullish >= min_thresh:
            direction = "BUY (Bullish)"
            confidence = prob_bullish
        elif prob_bearish >= min_thresh:
            direction = "SELL (Bearish)"
            confidence = prob_bearish
        else:
            direction = "NO_TRADE"
            confidence = max(prob_bullish, prob_bearish)
        log(f"3-class: bear={prob_bearish:.1%} neutral={prob_neutral:.1%} bull={prob_bullish:.1%}")
    else:
        # Legacy binary model fallback
        prob = float(sum(w * m.predict_proba(features_scaled)[0, 1] for m, w in ensemble_models))
        min_thresh = min(trading.CFG.get("model", {}).get("min_threshold", 0.55), best_thresh)
        sell_thresh = 1.0 - min_thresh
        direction = "BUY (Bullish)" if prob >= min_thresh else "SELL (Bearish)" if prob <= sell_thresh else "NO_TRADE"
        confidence = prob
        prob_bullish = prob
        prob_bearish = 1.0 - prob

    target_date = (latest_date + timedelta(days=forward_days)).strftime("%Y-%m-%d")
    log(f"Hasil: ${latest_price:.2f} | {direction} | conf: {confidence:.1%} | thresh: {min_thresh:.3f} | target: {target_date}")

    # Regime filter (Phase 2.5): skip signals in very low-ADX ranging market
    adx_passed = True
    if direction != "NO_TRADE":
        try:
            adx_val = df["ADX_14"].iloc[-1] if "ADX_14" in df.columns else None
            if adx_val is not None and adx_val < 20:
                log(f"[FILTER] Low ADX ({adx_val:.1f}): ranging market, skipping signal")
                direction = "NO_TRADE"
                adx_passed = False
        except Exception:
            pass

    # Multi-timeframe confirmation (Phase 2.2): disabled — model already uses daily data
    # if direction != "NO_TRADE" and adx_passed:
    #     try:
    #         dt_val = daily_trend.iloc[-1] if len(daily_trend) > 0 else 0
    #         if direction.startswith("BUY") and dt_val == -1:
    #             log(f"[FILTER] BUY rejected: daily trend bearish")
    #             direction = "NO_TRADE"
    #         elif direction.startswith("SELL") and dt_val == 1:
    #             log(f"[FILTER] SELL rejected: daily trend bullish")
    #             direction = "NO_TRADE"
    #     except Exception:
    #         pass

    # SHAP top drivers (Phase 2.4)
    top_features = []
    if direction != "NO_TRADE":
        try:
            from feature_selector import get_top_drivers
            top_features = get_top_drivers(ensemble_models[0][0], features_scaled, feature_cols)
            if top_features:
                log(f"Top drivers: {', '.join(f'{f}({v:+.3f})' for f, v in top_features)}")
        except Exception:
            pass

    # Hitung levels & TP/SL hanya untuk BUY/SELL
    sl = tp1 = tp2 = None
    entry_realtime = None
    if direction != "NO_TRADE":
        is_buy = direction.startswith("BUY")
        sl, tp1, tp2 = trading.compute_tp_sl(levels, is_buy, latest_price)

    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    model_ver = artifacts.get("model_version", "v3")
    c.execute("""
        INSERT INTO predictions (date, price, predicted_direction, confidence, threshold,
                                 target_date, model_version, sl, tp1, tp2, entry_realtime)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (latest_date.strftime("%Y-%m-%d"), latest_price, direction, confidence, min_thresh,
          target_date, f"xgb_{model_ver}_acc{float(artifacts.get('test_acc',0)):.0%}",
          sl, tp1, tp2, entry_realtime))
    conn.commit()
    pred_id = c.lastrowid
    if sl is not None:
        log(f"Tersimpan ke journal (ID: {pred_id}) | SL: ${sl:.2f} TP1: ${tp1:.2f} TP2: ${tp2:.2f}")
    else:
        log(f"Tersimpan ke journal (ID: {pred_id}) | NO_TRADE (no SL/TP)")

    # Kirim notifikasi Telegram hanya untuk BUY/SELL
    if direction != "NO_TRADE":
        result = telegram_notifier.send_signal(pred_id, direction, confidence, latest_price, target_date, min_thresh)
        if result:
            c.execute("UPDATE predictions SET entry_realtime=? WHERE id=?", (result["entry_realtime"], pred_id))
            conn.commit()
    else:
        log(f"Signal lemah (bull={prob_bullish:.1%} bear={prob_bearish:.1%}), tidak kirim notifikasi")

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
            # Skip NO_TRADE — no position to evaluate
            if direction == "NO_TRADE":
                c.execute("UPDATE predictions SET outcome='SKIP', result_pct=0, outcome_detail='NO_TRADE' WHERE id=?",
                          (pred_id,))
                evaluated += 1
                continue
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
                    df, entry_idx, entry_price, sl, tp1, tp2, is_buy, max_lookahead=10
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
        best_wr = 0.0
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
    """Retrain if >= 10 new evaluations OR monthly re-optimization due."""
    try:
        import joblib
        arts = joblib.load(MODEL_FILE)
        last_train = arts.get("train_date", "2000-01-01")
    except Exception:
        last_train = "2000-01-01"

    # Monthly re-optimization
    try:
        days_since = (datetime.now() - datetime.strptime(last_train[:10], "%Y-%m-%d")).days
        if days_since >= 30:
            log(f"[RETRAIN] Monthly re-optimization ({days_since} days since last)")
            return True
    except Exception:
        pass

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
    log(f"AUTO RUNNER V2 started (fast=60s, slow={interval_hours}h)")
    import threading
    try:
        from telegram_notifier import process_commands
        def _loop():
            while not _shutdown:
                try:
                    process_commands()
                except Exception:
                    pass
                time.sleep(5)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        log("Telegram polling started")
    except Exception as e:
        log(f"Polling thread error: {e}")
    last_monthly = ""
    last_heartbeat = datetime.now() - timedelta(hours=7)
    last_slow = datetime.now() - timedelta(hours=interval_hours)
    last_pred_date = None

    while not _shutdown:
        try:
            now = datetime.now()

            # === FAST LOOP (every 60s): daily + 4H prediction + trailing stops ===
            try:
                today = now.strftime("%Y-%m-%d")
                if today != last_pred_date:
                    log("New day detected — generating prediction...")
                    run_prediction_job()
                    last_pred_date = today

                # 4H prediction (check if new 4H candle available)
                _run_4h_cycle()

                # Trailing stops
                from trailing_stop import manage_trailing_stops
                updated = manage_trailing_stops()
                if updated > 0:
                    log(f"[TRAIL] Updated {updated} trailing stops")
            except Exception as e:
                log(f"[FAST] Error: {e}")

            # === SLOW LOOP (every N hours): evaluate + retrain + 4H + risk ===
            if (now - last_slow).total_seconds() >= interval_hours * 3600:
                log("=" * 55)

                # Data validation
                try:
                    from data_validator import validate_and_report
                    _, issues = validate_and_report(os.path.join(trading.BASE_DIR, "xauusd_daily.csv"))
                    if any("STALE" in i or "INVALID" in i for i in issues):
                        log(f"[WARN] Data quality issues: {issues}")
                except Exception as e:
                    log(f"[WARN] Data validation error: {e}")

                # Risk management
                try:
                    from simulation import check_drawdown_limit, check_weekly_loss_limit, get_consecutive_losses
                    dd_ok, dd_pct = check_drawdown_limit()
                    if not dd_ok:
                        log(f"[RISK] Max drawdown breached ({dd_pct:.1%}).")
                        telegram_notifier._send_telegram(f"<b>[RISK]</b> Max drawdown {dd_pct:.1%}.")
                    wl_ok, wl_pnl = check_weekly_loss_limit()
                    if not wl_ok:
                        log(f"[RISK] Weekly loss limit breached (${wl_pnl:.2f}).")
                        telegram_notifier._send_telegram(f"<b>[RISK]</b> Weekly loss ${wl_pnl:.2f}.")
                except Exception as e:
                    log(f"[RISK] Risk check error: {e}")

                # Evaluate + retrain
                evaluated = evaluate_past_predictions()
                if should_retrain() or evaluated >= 10:
                    retrain_model()
                show_learning_stats()

                # 4H runner
                _run_4h_cycle()

                # Model health check
                try:
                    _check_model_health()
                except Exception as e:
                    log(f"[ALERT] Model health check error: {e}")

                # Monthly report
                month_key = now.strftime("%Y-%m")
                if month_key != last_monthly:
                    monthly_report()
                    try:
                        from monte_carlo import run_monte_carlo, format_mc_report
                        mc = run_monte_carlo()
                        if mc:
                            telegram_notifier._send_telegram(f"<b>[MONTE CARLO]</b>\n<pre>{format_mc_report(mc)}</pre>")
                    except Exception as e:
                        log(f"[MC] Monte Carlo error: {e}")
                    last_monthly = month_key

                # Heartbeat (every 30 min)
                if (now - last_heartbeat).total_seconds() > 30 * 60:
                    try:
                        from simulation import get_active_sim, get_consecutive_losses
                        sim = get_active_sim()
                        bal = f"${sim[1]:.2f}" if sim else "N/A"
                        cl = get_consecutive_losses()
                        # Count active positions
                        import sqlite3 as _hb_sql
                        _conn = _hb_sql.connect(trading.DB_FILE)
                        d_active = _conn.execute("SELECT COUNT(*) FROM predictions WHERE outcome IS NULL AND predicted_direction != 'NO_TRADE'").fetchone()[0]
                        h_active = _conn.execute("SELECT COUNT(*) FROM predictions_4h WHERE outcome IS NULL AND predicted_direction != 'NO_TRADE'").fetchone()[0]
                        d_total = _conn.execute("SELECT COUNT(*) FROM predictions WHERE predicted_direction != 'NO_TRADE'").fetchone()[0]
                        h_total = _conn.execute("SELECT COUNT(*) FROM predictions_4h WHERE predicted_direction != 'NO_TRADE'").fetchone()[0]
                        _conn.close()
                        msg = (
                            f"<b>[HEARTBEAT]</b>\n"
                            f"  Status: Alive\n"
                            f"  Balance: {bal}\n"
                            f"  Consec losses: {cl}\n"
                            f"  Daily: {d_active} active / {d_total} total\n"
                            f"  4H: {h_active} active / {h_total} total\n"
                            f"  Next slow: {(now + timedelta(hours=interval_hours)).strftime('%H:%M')}"
                        )
                        telegram_notifier._send_telegram(msg)
                        log(f"[HEARTBEAT] Alive | Balance: {bal} | Consec losses: {cl}")
                        last_heartbeat = now
                    except Exception as e:
                        log(f"[HEARTBEAT] Heartbeat error: {e}")

                last_slow = now
                log("-" * 55)

            # Sleep 60s between fast loop iterations
            for _ in range(6):
                if _shutdown:
                    break
                time.sleep(10)

        except KeyboardInterrupt:
            log("Stopped.")
            break
        except Exception as e:
            log(f"Error: {e}")
            try:
                telegram_notifier._send_telegram(f"<b>[ERROR]</b> {str(e)[:200]}")
            except Exception:
                pass
            time.sleep(60)
    log("Daemon stopped gracefully.")


def _check_model_health():
    """Monitor model accuracy degradation."""
    conn = sqlite3.connect(trading.DB_FILE)
    recent = pd.read_sql("SELECT outcome FROM predictions WHERE outcome IS NOT NULL ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    if len(recent) >= 20:
        acc = (recent["outcome"] == "WIN").sum() / len(recent)
        if acc < 0.40:
            telegram_notifier._send_telegram(
                f"<b>[ALERT]</b> Model accuracy {acc:.1%} over last {len(recent)} predictions. Consider retrain.")


def _run_4h_cycle():
    """Run 4H evaluation + prediction + report in a single subprocess"""
    try:
        p = sys.executable
        script = os.path.join(trading.BASE_DIR, "runner_4h.py")
        r = subprocess.run([p, script, "--run"], capture_output=True, text=True, cwd=trading.BASE_DIR, timeout=180)
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().split("\n"):
                log(f"[4H] {line.strip()}")
        elif r.returncode != 0:
            log(f"[4H] Error: {r.stderr[:1000]}")
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
    import simulation as sim; sim.init_sim_db()

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
