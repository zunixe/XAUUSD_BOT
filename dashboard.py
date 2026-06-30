"""XAUUSD Trading Dashboard — Web monitoring interface."""
import os, sys, json, sqlite3, subprocess, secrets
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, abort
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import trading
import yaml

try:
    with open(os.path.join(BASE_DIR, "config.yaml")) as f:
        CFG = yaml.safe_load(f)
except Exception:
    CFG = {}

app = Flask(__name__)

# Generate or load API key
DASHBOARD_KEY_FILE = os.path.join(BASE_DIR, ".dashboard_key")
def _get_api_key():
    try:
        with open(DASHBOARD_KEY_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        key = secrets.token_urlsafe(32)
        with open(DASHBOARD_KEY_FILE, "w") as f:
            f.write(key)
        return key

API_KEY = _get_api_key()

def _check_auth():
    """Check API key for action endpoints."""
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        abort(403, description="Invalid API key")

# Simple TTL cache
_cache = {}
_cache_ttl = {}

def _cached(key, ttl_seconds, func):
    """Return cached value or compute and cache."""
    now = datetime.now().timestamp()
    if key in _cache and now - _cache_ttl.get(key, 0) < ttl_seconds:
        return _cache[key]
    val = func()
    _cache[key] = val
    _cache_ttl[key] = now
    return val


# ========== HELPERS ==========

def _db_query(sql, params=()):
    conn = sqlite3.connect(trading.DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _db_query_one(sql, params=()):
    conn = sqlite3.connect(trading.DB_FILE)
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return dict(row) if row else None


def _load_model(tf="daily"):
    import joblib
    path = os.path.join(BASE_DIR, "xauusd_model.pkl" if tf == "daily" else "xauusd_model_4h.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _get_live_price():
    try:
        from telegram_notifier import get_realtime_price
        p = get_realtime_price()
        return p if p and p.get("price") else None
    except Exception:
        return None


def _tail_log(filename, n=50):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()[-n:]
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        level = "info"
        if any(k in line for k in ["[RISK]", "Error", "CRITICAL", "[ALERT]"]):
            level = "error"
        elif any(k in line for k in ["[FILTER]", "[WARN]"]):
            level = "warning"
        elif any(k in line for k in ["[HEARTBEAT]", "terkirim", "OK"]):
            level = "success"
        result.append({"text": line, "level": level})
    return result


def _parse_log_timestamp(filename, pattern):
    """Find last log line matching pattern and extract timestamp."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in reversed(f.readlines()):
            if pattern in line:
                try:
                    ts_str = line[1:20]  # [YYYY-MM-DD HH:MM:SS]
                    return ts_str
                except Exception:
                    pass
    return None


def _last_log_time(filename):
    """Get timestamp of last log entry."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in reversed(f.readlines()):
            line = line.strip()
            if line and line.startswith("[") and len(line) > 20:
                return line[1:20]
    return None


# ========== ROUTES ==========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    def _compute():
        import psutil
        bot_alive = False
        for p in psutil.process_iter(["cmdline"]):
            try:
                if p.info["cmdline"] and "auto_runner" in " ".join(p.info["cmdline"]) and "dashboard" not in " ".join(p.info["cmdline"]):
                    bot_alive = True
                    break
            except Exception:
                pass
        live = _get_live_price()
        hb = _parse_log_timestamp("auto_runner.log", "[HEARTBEAT]")
        if not hb:
            hb = _last_log_time("auto_runner.log")
            if hb:
                hb = hb + " (last activity)"
        last_pred = _db_query_one("SELECT date, predicted_direction FROM predictions ORDER BY id DESC LIMIT 1")
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
            last_data = df["Date"].max().strftime("%Y-%m-%d")
            age_days = (datetime.now() - df["Date"].max()).days
        except Exception:
            last_data = "N/A"
            age_days = -1
        return {
            "bot_alive": bot_alive, "live_price": live, "heartbeat": hb,
            "last_prediction": last_pred, "last_data_date": last_data,
            "data_age_days": age_days, "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_key": API_KEY,
        }
    return jsonify(_cached("status", 15, _compute))


@app.route("/api/overview")
def api_overview():
    # Simulation
    sim = _db_query_one("SELECT balance, initial_balance FROM simulation WHERE active=1 ORDER BY id DESC LIMIT 1")

    # Trade stats (daily + 4H combined, exclude SKIP/NO_TRADE)
    daily_stats = _db_query_one("SELECT COUNT(*) as total, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses FROM predictions WHERE outcome NOT IN ('SKIP', 'NO_TRADE')") or {"total": 0, "wins": 0, "losses": 0}
    fourh_stats = _db_query_one("SELECT COUNT(*) as total, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses FROM predictions_4h WHERE outcome NOT IN ('SKIP', 'NO_TRADE')") or {"total": 0, "wins": 0, "losses": 0}

    total = (daily_stats["total"] or 0) + (fourh_stats["total"] or 0)
    wins = (daily_stats["wins"] or 0) + (fourh_stats["wins"] or 0)
    losses = (daily_stats["losses"] or 0) + (fourh_stats["losses"] or 0)

    # Active positions (only BUY/SELL, not NO_TRADE) — same month filter as Telegram
    this_month = datetime.now().strftime("%Y-%m")
    daily_active = _db_query("SELECT id, predicted_direction, confidence, sl, tp1, tp2, entry_realtime, price FROM predictions WHERE date LIKE ? AND outcome IS NULL AND predicted_direction != 'NO_TRADE'", (f"{this_month}%",))
    fourh_active = _db_query("SELECT id, predicted_direction, confidence, sl, tp1, tp2, entry_realtime, price FROM predictions_4h WHERE date LIKE ? AND outcome IS NULL AND predicted_direction != 'NO_TRADE'", (f"{this_month}%",))

    # Pending predictions
    daily_pending = len(daily_active)
    fourh_pending = len(fourh_active)

    # Monthly return
    monthly = _db_query("""
        SELECT result_pct FROM predictions WHERE outcome IS NOT NULL
        AND date >= date('now', '-30 days')
        UNION ALL
        SELECT result_pct FROM predictions_4h WHERE outcome IS NOT NULL
        AND date >= date('now', '-30 days')
    """)
    monthly_return = sum(r["result_pct"] for r in monthly if r["result_pct"]) if monthly else 0

    balance = sim["balance"] if sim else 0
    initial = sim["initial_balance"] if sim else 0

    # Unrealized P&L for active positions
    unrealized_pnl = 0
    live_price = None
    try:
        from telegram_notifier import get_realtime_price
        rp = get_realtime_price()
        if rp and rp.get("price"):
            live_price = rp["price"]
    except Exception:
        pass

    if live_price and balance > 0:
        XAU_USD_PER_MOVE = 1.0
        from simulation import calc_lot_size
        for pos in daily_active + fourh_active:
            entry = pos.get("entry_realtime") or live_price
            sl = pos.get("sl")
            if not entry or not sl:
                continue
            is_buy = "BUY" in (pos.get("predicted_direction") or "")
            lot, _ = calc_lot_size(balance, entry, sl)
            if lot > 0:
                price_diff = (live_price - entry) if is_buy else (entry - live_price)
                unrealized_pnl += lot * price_diff * XAU_USD_PER_MOVE

    effective_balance = round(balance + unrealized_pnl, 2)
    pnl = round(effective_balance - initial, 2) if sim else 0
    ret_pct = round(pnl / initial * 100, 2) if initial else 0

    d_total = daily_stats["total"] or 0
    d_wins = daily_stats["wins"] or 0
    d_losses = daily_stats["losses"] or 0
    f_total = fourh_stats["total"] or 0
    f_wins = fourh_stats["wins"] or 0
    f_losses = fourh_stats["losses"] or 0

    return jsonify({
        "balance": balance, "initial": initial, "pnl": pnl, "return_pct": ret_pct,
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "daily_total": d_total, "daily_wins": d_wins, "daily_losses": d_losses,
        "daily_win_rate": round(d_wins / d_total * 100, 1) if d_total else 0,
        "fourh_total": f_total, "fourh_wins": f_wins, "fourh_losses": f_losses,
        "fourh_win_rate": round(f_wins / f_total * 100, 1) if f_total else 0,
        "daily_pending": daily_pending, "fourh_pending": fourh_pending,
        "active_positions": daily_active + fourh_active,
        "monthly_return": round(monthly_return, 2),
    })


@app.route("/api/signals")
def api_signals():
    daily = _db_query_one("""
        SELECT id, date, price, predicted_direction, confidence, threshold,
               sl, tp1, tp2, entry_realtime, model_version, target_date
        FROM predictions ORDER BY id DESC LIMIT 1
    """)
    fourh = _db_query_one("""
        SELECT id, date, time, price, predicted_direction, confidence, threshold,
               sl, tp1, tp2, entry_realtime, model_version, target_date, target_time
        FROM predictions_4h ORDER BY id DESC LIMIT 1
    """)

    # Regime info
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        from trading import compute_adx
        adx, plus_di, minus_di = compute_adx(df["High"], df["Low"], df["Close"])
        adx_val = float(adx.iloc[-1])
        is_trending = adx_val > 25
        # Daily trend
        from trading import get_daily_trend
        dt = get_daily_trend(df)
        daily_trend_val = int(dt.iloc[-1])
    except Exception:
        adx_val = 0
        is_trending = False
        daily_trend_val = 0

    return jsonify({
        "daily": daily,
        "fourh": fourh,
        "regime": {"adx": round(adx_val, 1), "is_trending": is_trending, "daily_trend": daily_trend_val},
    })


@app.route("/api/equity")
def api_equity():
    sim = _db_query_one("SELECT initial_balance FROM simulation WHERE active=1 ORDER BY id DESC LIMIT 1")
    initial = sim["initial_balance"] if sim else 100

    trades = _db_query("""
        SELECT balance_after, created_at FROM sim_trades ORDER BY id
    """)
    if not trades:
        preds = _db_query("""
            SELECT result_pct, date FROM predictions WHERE outcome IS NOT NULL ORDER BY id
        """)
        if not preds:
            return jsonify({"points": [], "initial": initial, "peak": initial})
        equity = [initial]
        for p in preds:
            equity.append(round(equity[-1] * (1 + (p["result_pct"] or 0) / 100), 2))
        points = [{"date": preds[i]["date"], "balance": equity[i+1]} for i in range(len(preds))]
        return jsonify({"points": points, "initial": initial, "peak": max(equity)})

    points = [{"date": t["created_at"][:10], "balance": t["balance_after"]} for t in trades]
    peak = max(t["balance_after"] for t in trades)
    return jsonify({"points": points, "initial": initial, "peak": peak})


@app.route("/api/risk")
def api_risk():
    try:
        from simulation import check_risk_all
        r = check_risk_all()
    except Exception as e:
        return jsonify({"error": str(e), "dd_ok": False, "wl_ok": False, "consec": -1, "mc_ok": False})

    regime = "unknown"
    try:
        from simulation import get_volatility_regime
        regime = get_volatility_regime()
    except Exception:
        pass

    return jsonify({
        "drawdown_pct": round(r["dd_pct"] * 100, 1),
        "drawdown_ok": r["dd_ok"],
        "drawdown_limit": CFG.get("risk", {}).get("max_drawdown_pct", 0.15) * 100,
        "weekly_pnl": round(r["wl_pnl"], 2),
        "weekly_ok": r["wl_ok"],
        "weekly_limit_pct": CFG.get("risk", {}).get("weekly_loss_limit_pct", 0.05) * 100,
        "consecutive_losses": r["consec"],
        "cooldown_limit": CFG.get("risk", {}).get("cooldown_after_losses", 3),
        "volatility_regime": regime,
        "max_concurrent": CFG.get("risk", {}).get("max_concurrent_positions", 2),
        "current_positions": r["mc_total"],
        "concurrent_ok": r["mc_ok"],
    })


@app.route("/api/model")
def api_model():
    result = {}
    for tf, path in [("daily", "xauusd_model.pkl"), ("4h", "xauusd_model_4h.pkl")]:
        arts = _load_model(tf)
        if arts:
            result[tf] = {
                "version": arts.get("model_version", "N/A"),
                "train_date": arts.get("train_date", "N/A"),
                "test_acc": round(arts.get("test_acc", 0) * 100, 1),
                "oot_acc": round(arts.get("oot_acc", 0) * 100, 1) if arts.get("oot_acc") else None,
                "f1": round(arts.get("f1", 0) * 100, 1),
                "threshold": arts.get("best_thresh", arts.get("threshold", 0.55)),
                "features": len(arts.get("feature_cols", arts.get("cols", []))),
                "train_samples": arts.get("train_samples", arts.get("samples", 0)),
                "n_classes": arts.get("n_classes", 3),
                "fold_scores": arts.get("fold_scores", []),
                "forward_days": arts.get("forward_days", arts.get("forward", 3)),
            }
        else:
            result[tf] = None
    return jsonify(result)


@app.route("/api/trades")
def api_trades():
    daily = _db_query("""
        SELECT id, 'Daily' as timeframe, date, price, predicted_direction as direction,
               confidence, outcome, outcome_detail, result_pct, sl, tp1, tp2
        FROM predictions ORDER BY id DESC LIMIT 200
    """)
    fourh = _db_query("""
        SELECT id, '4H' as timeframe, date || ' ' || time as date, price, predicted_direction as direction,
               confidence, outcome, outcome_detail, result_pct, sl, tp1, tp2
        FROM predictions_4h ORDER BY id DESC LIMIT 200
    """)
    all_trades = sorted(daily + fourh, key=lambda x: x["id"], reverse=True)[:200]
    return jsonify(all_trades)


@app.route("/api/features")
def api_features():
    arts = _load_model("daily")
    if not arts:
        return jsonify({"features": []})
    importance = arts.get("feature_importance", {})
    if not importance:
        return jsonify({"features": []})
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    return jsonify({"features": [{"name": k, "importance": round(v, 4)} for k, v in sorted_features]})


@app.route("/api/montecarlo")
def api_montecarlo():
    def _compute():
        try:
            from monte_carlo import run_monte_carlo
            mc = run_monte_carlo()
            if mc:
                return mc
        except Exception:
            pass
        return None
    return jsonify(_cached("montecarlo", 300, _compute))  # cache 5 min


@app.route("/api/macro")
def api_macro():
    def _compute():
        tickers = {
            "DXY_Close": {"name": "DXY", "file": "dxy_daily.csv"},
            "VIX_Close": {"name": "VIX", "file": "vix_daily.csv"},
            "TIP_Close": {"name": "TIP (Real Yield)", "file": "tip_daily.csv"},
            "Silver_Close": {"name": "Silver", "file": "silver_daily.csv"},
            "BTC_Close": {"name": "Bitcoin", "file": "btc_daily.csv"},
            "OIL_Close": {"name": "Crude Oil", "file": "oil_daily.csv"},
            "SPY_Close": {"name": "S&P 500", "file": "spy_daily.csv"},
            "US10Y_Close": {"name": "US 10Y", "file": "us10y_daily.csv"},
            "Breakeven_10Y": {"name": "Inflation Exp", "file": "breakeven_10y.csv"},
        }
        result = []
        for col, info in tickers.items():
            csv_path = os.path.join(BASE_DIR, info["file"])
            try:
                df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
                if col in df.columns:
                    s = df[col].dropna()
                    if len(s) >= 2:
                        result.append({
                            "name": info["name"],
                            "value": round(float(s.iloc[-1]), 2),
                            "change": round(float(s.iloc[-1] - s.iloc[-2]), 2),
                            "change_pct": round(float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100), 2),
                        })
            except Exception:
                pass
        return result
    return jsonify(_cached("macro", 120, _compute))  # cache 2 min


@app.route("/api/logs")
def api_logs():
    lines = _tail_log("auto_runner.log", 50)
    return jsonify(lines)


@app.route("/api/price_history")
def api_price_history():
    """Return OHLC data for XAUUSD price chart."""
    days = request.args.get("days", 90, type=int)
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "xauusd_daily.csv"), parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        df = df.tail(days)
        points = []
        for _, row in df.iterrows():
            points.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })
        # Add technical levels for last bar
        levels = trading.calculate_levels(df.set_index("Date"))
        # Clean NaN values for JSON serialization
        clean_levels = {}
        for k, v in levels.items():
            if v is not None and not (isinstance(v, float) and (v != v)):  # not NaN
                clean_levels[k] = round(float(v), 2)
        return jsonify({"points": points, "levels": clean_levels})
    except Exception as e:
        return jsonify({"points": [], "levels": {}, "error": str(e)})


@app.route("/api/active")
def api_active():
    """Active positions with SL/TP proximity (matches Telegram /info logic)."""
    this_month = datetime.now().strftime("%Y-%m")
    curr = _get_live_price()
    conn = sqlite3.connect(trading.DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        daily = [dict(r) for r in conn.execute(
            "SELECT id, date, price, predicted_direction, confidence, sl, tp1, tp2, entry_realtime "
            "FROM predictions WHERE date LIKE ? AND outcome IS NULL AND predicted_direction != 'NO_TRADE'",
            (f"{this_month}%",)).fetchall()]
        fourh = [dict(r) for r in conn.execute(
            "SELECT id, date, time, price, predicted_direction, confidence, sl, tp1, tp2, entry_realtime "
            "FROM predictions_4h WHERE date LIKE ? AND outcome IS NULL AND predicted_direction != 'NO_TRADE'",
            (f"{this_month}%",)).fetchall()]
    finally:
        conn.close()

    def _enrich(pos, is_4h=False):
        p = dict(pos)
        entry = p.get("entry_realtime") or p.get("price")
        p["entry"] = entry
        p["is_4h"] = is_4h
        p["near_sl"] = False
        p["near_tp1"] = False
        if curr and entry and p.get("sl"):
            is_buy = "BUY" in (p.get("predicted_direction") or "")
            if (is_buy and curr <= p["sl"]) or (not is_buy and curr >= p["sl"]):
                p["near_sl"] = True
            elif p.get("tp1") and ((is_buy and curr >= p["tp1"]) or (not is_buy and curr <= p["tp1"])):
                p["near_tp1"] = True
        return p

    return jsonify({
        "daily": [_enrich(d) for d in daily],
        "fourh": [_enrich(f, True) for f in fourh],
        "current_price": curr["price"] if curr and isinstance(curr, dict) else curr,
    })


@app.route("/api/action", methods=["POST"])
def api_action():
    _check_auth()
    data = request.get_json()
    action = data.get("action")
    if not action:
        return jsonify({"error": "No action specified"}), 400

    p = sys.executable
    try:
        if action == "retrain_daily":
            r = subprocess.run([p, os.path.join(BASE_DIR, "auto_runner.py"), "--retrain"],
                               capture_output=True, text=True, cwd=BASE_DIR, timeout=600)
            return jsonify({"ok": r.returncode == 0, "output": r.stdout[-500:] if r.stdout else r.stderr[-500:]})
        elif action == "retrain_4h":
            r = subprocess.run([p, os.path.join(BASE_DIR, "runner_4h.py"), "--retrain"],
                               capture_output=True, text=True, cwd=BASE_DIR, timeout=600)
            return jsonify({"ok": r.returncode == 0, "output": r.stdout[-500:] if r.stdout else r.stderr[-500:]})
        elif action == "reset_sim":
            r = subprocess.run([p, "-c", "import simulation as sim; sim.reset_sim(); print('OK')"],
                               capture_output=True, text=True, cwd=BASE_DIR, timeout=30)
            return jsonify({"ok": r.returncode == 0, "output": r.stdout.strip() or r.stderr[-300:]})
        elif action == "start_sim":
            balance = data.get("balance", 100)
            try:
                balance = max(1, min(100000, float(balance)))
            except (ValueError, TypeError):
                balance = 100
            r = subprocess.run([p, "-c", f"import simulation as sim; sim.start_sim({float(balance)}); print('OK')"],
                               capture_output=True, text=True, cwd=BASE_DIR, timeout=30)
            return jsonify({"ok": r.returncode == 0, "output": r.stdout.strip() or r.stderr[-300:]})
        elif action == "evaluate":
            r = subprocess.run([p, "-c",
                "from auto_runner import evaluate_past_predictions; n = evaluate_past_predictions(); print(f'Evaluated {n} predictions')"],
                capture_output=True, text=True, cwd=BASE_DIR, timeout=120)
            return jsonify({"ok": r.returncode == 0, "output": r.stdout.strip()[-500:] if r.returncode == 0 else r.stderr[-500:]})
        elif action == "predict":
            r = subprocess.run([p, "-c",
                "from auto_runner import run_prediction_job; pid = run_prediction_job(); print(f'Prediction ID: {pid}')"],
                capture_output=True, text=True, cwd=BASE_DIR, timeout=120)
            return jsonify({"ok": r.returncode == 0, "output": r.stdout.strip()[-500:] if r.returncode == 0 else r.stderr[-500:]})
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Action timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  XAUUSD Trading Dashboard")
    print(f"  http://localhost:5050")
    print(f"  API Key: {API_KEY}")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5050, debug=False)
