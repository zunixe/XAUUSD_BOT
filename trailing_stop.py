"""Trailing stop management for active positions."""
import sqlite3
import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, "config.yaml")) as f:
        CFG = yaml.safe_load(f)
except Exception:
    CFG = {"trailing_stop": {"breakeven_atr": 0.5, "trail_atr": 1.0, "trail_distance_atr": 0.5}}

import trading


def manage_trailing_stops(atr=None):
    """Check active positions and update trailing stops. Single DB connection."""
    try:
        from telegram_notifier import get_realtime_price
        live = get_realtime_price()
        if not live or not live.get("price"):
            return 0
        current_price = live["price"]
    except Exception:
        return 0

    # Get ATR if not provided
    if atr is None:
        try:
            import pandas as pd
            df = trading.load_daily()
            tr = pd.concat([df["High"] - df["Low"],
                            abs(df["High"] - df["Close"].shift(1)),
                            abs(df["Low"] - df["Close"].shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        except Exception:
            return 0

    ts_cfg = CFG.get("trailing_stop", {})
    breakeven_atr = ts_cfg.get("breakeven_atr", 0.5)
    trail_atr = ts_cfg.get("trail_atr", 1.0)
    trail_distance = ts_cfg.get("trail_distance_atr", 0.5)

    # Single connection for read + write
    conn = sqlite3.connect(trading.DB_FILE)
    try:
        active = conn.execute("""
            SELECT id, date, price, predicted_direction, sl, tp1, tp2, entry_realtime
            FROM predictions WHERE outcome IS NULL AND sl IS NOT NULL
        """).fetchall()

        if not active:
            return 0

        updates = []
        for pred_id, date, price, direction, sl, tp1, tp2, entry in active:
            entry = entry or price
            is_buy = "BUY" in (direction or "")
            profit = (current_price - entry) if is_buy else (entry - current_price)
            profit_atr = profit / (atr + 1e-10)

            new_sl = sl
            if profit_atr >= trail_atr:
                if is_buy:
                    new_sl = round(current_price - atr * trail_distance, 2)
                else:
                    new_sl = round(current_price + atr * trail_distance, 2)
            elif profit_atr >= breakeven_atr:
                if is_buy:
                    new_sl = max(sl, round(entry + 0.30, 2))
                else:
                    new_sl = min(sl, round(entry - 0.30, 2))

            if is_buy and new_sl > sl:
                updates.append((new_sl, pred_id))
            elif not is_buy and new_sl < sl:
                updates.append((new_sl, pred_id))

        if updates:
            conn.executemany("UPDATE predictions SET sl=? WHERE id=?", updates)
            conn.commit()

        return len(updates)
    finally:
        conn.close()
