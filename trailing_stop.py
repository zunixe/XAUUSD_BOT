"""Trailing stop management for active positions."""
import sqlite3
import os
import yaml
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

import trading


def manage_trailing_stops():
    """Check active positions and update trailing stops."""
    conn = sqlite3.connect(trading.DB_FILE)
    active = conn.execute("""
        SELECT id, date, price, predicted_direction, sl, tp1, tp2, entry_realtime
        FROM predictions WHERE outcome IS NULL AND sl IS NOT NULL
    """).fetchall()
    conn.close()

    if not active:
        return 0

    try:
        from telegram_notifier import get_realtime_price
        live = get_realtime_price()
        if not live or not live.get("price"):
            return 0
        current_price = live["price"]
    except Exception:
        return 0

    # Get current ATR
    try:
        import pandas as pd
        df = trading.load_daily()
        tr = pd.concat([df["High"] - df["Low"],
                        abs(df["High"] - df["Close"].shift(1)),
                        abs(df["Low"] - df["Close"].shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
    except Exception:
        return 0

    ts_cfg = CFG["trailing_stop"]
    breakeven_atr = ts_cfg["breakeven_atr"]
    trail_atr = ts_cfg["trail_atr"]
    trail_distance = ts_cfg["trail_distance_atr"]

    updated = 0
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
            updated += 1
        elif not is_buy and new_sl < sl:
            updates.append((new_sl, pred_id))
            updated += 1

    if updates:
        conn = sqlite3.connect(trading.DB_FILE)
        c = conn.cursor()
        c.executemany("UPDATE predictions SET sl=? WHERE id=?", updates)
        conn.commit()
        conn.close()

    return updated
