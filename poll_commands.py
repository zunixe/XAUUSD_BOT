"""Continuous daemon: Telegram commands (5s) + signal checks (3-tier: 1s/3s/30s)."""
import sys, os, time, sqlite3, subprocess, pandas as pd, json, urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "poll.log")

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

sys.path.insert(0, BASE)
from telegram_notifier import process_commands, get_realtime_price, send_outcome_notification
import trading

def _new_candle(csv_path, table):
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        latest = df["Date"].iloc[-1]
        if isinstance(latest, str):
            latest = pd.to_datetime(latest)
        if hasattr(latest, 'tzinfo') and latest.tzinfo is not None:
            latest = latest.tz_localize(None)
        conn = sqlite3.connect(trading.DB_FILE)
        if table == "predictions":
            latest_str = latest.strftime("%Y-%m-%d")
            exists = conn.execute("SELECT 1 FROM predictions WHERE date = ?", (latest_str,)).fetchone()
        elif table == "predictions_4h":
            date_str = latest.strftime("%Y-%m-%d")
            time_str = latest.strftime("%H:%M")
            exists = conn.execute("SELECT 1 FROM predictions_4h WHERE date = ? AND time = ?", (date_str, time_str)).fetchone()
        else:
            exists = None
        conn.close()
        return exists is None
    except Exception as e:
        log(f"check error ({table}): {e}")
        return False

def _evaluate_realtime():
    """Evaluate active predictions against current real-time price, close any SL/TP hits."""
    live = get_realtime_price()
    if not live:
        return
    curr = live["price"]
    conn = sqlite3.connect(trading.DB_FILE)
    try:
        c = conn.cursor()
        # Daily
        for row in c.execute("""
            SELECT id, price, predicted_direction, sl, tp1, tp2, outcome_detail, entry_realtime
            FROM predictions WHERE outcome IS NULL AND sl IS NOT NULL
        """).fetchall():
            pred_id, price, direction, sl, tp1, tp2, _, entry_rt = row
            entry = entry_rt or price
            is_buy = "BUY" in (direction or "")
            hit = None
            if (is_buy and curr <= sl) or (not is_buy and curr >= sl):
                pct = (entry - sl) / entry * 100 if not is_buy else (sl - entry) / entry * 100
                hit = ("LOSS", "SL_HIT", sl, pct)
            elif tp2 and ((is_buy and curr >= tp2) or (not is_buy and curr <= tp2)):
                pct = (entry - tp2) / entry * 100 if not is_buy else (tp2 - entry) / entry * 100
                hit = ("WIN", "TP2_HIT", tp2, pct)
            elif tp1 and ((is_buy and curr >= tp1) or (not is_buy and curr <= tp1)):
                pct = (entry - tp1) / entry * 100 if not is_buy else (tp1 - entry) / entry * 100
                hit = ("WIN", "TP1_HIT", tp1, pct)
            if hit:
                outcome, detail, exit_price, pct = hit
                c.execute("UPDATE predictions SET outcome=?, outcome_detail=?, result_pct=? WHERE id=?",
                          (outcome, detail, pct, pred_id))
                conn.commit()
                log(f"REALTIME #{pred_id}: {outcome} ({detail}) pct: {pct:+.2f}% @ ${curr}")
                try:
                    send_outcome_notification(pred_id, "Daily", direction, entry, outcome, detail, pct, sl, tp1, tp2)
                except Exception as e:
                    log(f"send_outcome_notification error #{pred_id}: {e}")
        # 4H
        for row in c.execute("""
            SELECT id, predicted_direction, entry_realtime, sl, tp1, tp2
            FROM predictions_4h WHERE outcome IS NULL AND sl IS NOT NULL AND entry_realtime IS NOT NULL
        """).fetchall():
            pred_id, direction, entry, sl, tp1, tp2 = row
            if not entry or not sl:
                continue
            is_buy = direction == "BUY"
            hit = None
            if (is_buy and curr <= sl) or (not is_buy and curr >= sl):
                pct = (entry - sl) / entry * 100 if not is_buy else (sl - entry) / entry * 100
                hit = ("LOSS", "SL_HIT", sl, pct)
            elif tp2 and ((is_buy and curr >= tp2) or (not is_buy and curr <= tp2)):
                pct = (entry - tp2) / entry * 100 if not is_buy else (tp2 - entry) / entry * 100
                hit = ("WIN", "TP2_HIT", tp2, pct)
            elif tp1 and ((is_buy and curr >= tp1) or (not is_buy and curr <= tp1)):
                pct = (entry - tp1) / entry * 100 if not is_buy else (tp1 - entry) / entry * 100
                hit = ("WIN", "TP1_HIT", tp1, pct)
            if hit:
                outcome, detail, exit_price, pct = hit
                c.execute("UPDATE predictions_4h SET outcome=?, outcome_detail=?, result_pct=? WHERE id=?",
                          (outcome, detail, pct, pred_id))
                conn.commit()
                log(f"REALTIME 4H #{pred_id}: {outcome} ({detail}) pct: {pct:+.2f}% @ ${curr}")
                try:
                    send_outcome_notification(pred_id, "4H", direction, entry, outcome, detail, pct, sl, tp1, tp2)
                except Exception as e:
                    log(f"send_outcome_notification error 4H #{pred_id}: {e}")
    finally:
        conn.close()

last_check = time.time()
TIER3_INTERVAL = 30   # 30s — harga jauh dari SL/TP
TIER2_INTERVAL = 3    # 3s  — harga dalam 0.3% dari SL/TP
TIER1_INTERVAL = 1    # 1s  — harga dalam 0.1% dari SL/TP (sangat dekat)
TIER2_PCT = 0.003     # 0.3% proximity threshold
TIER1_PCT = 0.001     # 0.1% proximity threshold (very close)

def _get_proximity_tier():
    """Return 1 (very close), 2 (close), or 3 (far) from active SL/TP levels."""
    try:
        live = get_realtime_price()
        if not live:
            return 3
        curr = live["price"]
        conn = sqlite3.connect(trading.DB_FILE)
        c = conn.cursor()
        rows = c.execute("""
            SELECT sl, tp1, tp2 FROM predictions WHERE outcome IS NULL AND sl IS NOT NULL
            UNION ALL
            SELECT sl, tp1, tp2 FROM predictions_4h WHERE outcome IS NULL AND sl IS NOT NULL
        """).fetchall()
        conn.close()
        min_pct = 1.0
        for sl, tp1, tp2 in rows:
            for level in (sl, tp1, tp2):
                if level:
                    pct = abs(curr - level) / curr
                    min_pct = min(min_pct, pct)
        if min_pct < TIER1_PCT:
            return 1
        if min_pct < TIER2_PCT:
            return 2
    except Exception:
        pass
    return 3

log("Daemon started")

while True:
    try:
        process_commands()
    except Exception as e:
        log(f"poll error: {e}")

    now = time.time()
    tier = _get_proximity_tier()
    interval = TIER1_INTERVAL if tier == 1 else TIER2_INTERVAL if tier == 2 else TIER3_INTERVAL
    if now - last_check >= interval:
        last_check = now
        _evaluate_realtime()
        daily_new = _new_candle(trading.CSV_DAILY, "predictions")
        fourh_new = _new_candle(trading.CSV_4H, "predictions_4h")
        if daily_new:
            log(f"New daily candle — spawning auto_runner...")
            subprocess.Popen([sys.executable, os.path.join(BASE, "auto_runner.py")], cwd=BASE)
        elif fourh_new:
            log(f"New 4H candle — spawning runner_4h.py...")
            subprocess.Popen([sys.executable, os.path.join(BASE, "runner_4h.py"), "--run"], cwd=BASE)

    time.sleep(5)
