"""Trailing stop management for active positions (daily + 4H)."""
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
    """Check active positions (daily + 4H) and update trailing stops."""
    try:
        from telegram_notifier import get_realtime_price
        live = get_realtime_price()
        if not live or not live.get("price"):
            return 0
        current_price = live["price"]
    except Exception:
        return 0

    # Load OHLC data for both timeframes
    import pandas as pd
    df_daily = None
    df_4h = None
    try:
        df_daily = trading.load_daily()
        if atr is None:
            tr = pd.concat([df_daily["High"] - df_daily["Low"],
                            abs(df_daily["High"] - df_daily["Close"].shift(1)),
                            abs(df_daily["Low"] - df_daily["Close"].shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
    except Exception:
        pass
    try:
        df_4h = pd.read_csv(trading.CSV_4H, parse_dates=["Date"], index_col="Date").sort_index()
    except Exception:
        pass

    if atr is None:
        return 0

    ts_cfg = CFG.get("trailing_stop", {})
    breakeven_atr = ts_cfg.get("breakeven_atr", 0.5)
    trail_atr = ts_cfg.get("trail_atr", 1.0)
    trail_distance = ts_cfg.get("trail_distance_atr", 0.5)

    conn = sqlite3.connect(trading.DB_FILE)
    try:
        daily_active = conn.execute("""
            SELECT id, date, price, predicted_direction, sl, tp1, tp2, entry_realtime
            FROM predictions WHERE outcome IS NULL AND sl IS NOT NULL
        """).fetchall()
        fourh_active = conn.execute("""
            SELECT id, date, time, price, predicted_direction, sl, tp1, tp2, entry_realtime
            FROM predictions_4h WHERE outcome IS NULL AND sl IS NOT NULL
        """).fetchall()

        if not daily_active and not fourh_active:
            return 0

        daily_updates = []
        fourh_updates = []

        for pred_id, date, price, direction, sl, tp1, tp2, entry in daily_active:
            _process(pred_id, date, direction, sl, tp1, entry, price, current_price, atr,
                     breakeven_atr, trail_atr, trail_distance, df_daily, is_4h=False,
                     daily_updates=daily_updates, time_field=None)

        for row in fourh_active:
            pred_id, date, ctime, price, direction, sl, tp1, tp2, entry = row
            _process(pred_id, date, direction, sl, tp1, entry, price, current_price, atr,
                     breakeven_atr, trail_atr, trail_distance, df_4h, is_4h=True,
                     fourh_updates=fourh_updates, time_field=ctime)

        total = 0
        if daily_updates:
            conn.executemany("UPDATE predictions SET sl=? WHERE id=?", daily_updates)
            total += len(daily_updates)
        if fourh_updates:
            conn.executemany("UPDATE predictions_4h SET sl=? WHERE id=?", fourh_updates)
            total += len(fourh_updates)
        conn.commit()
        return total
    finally:
        conn.close()


def _process(pred_id, date, direction, sl, tp1, entry, price, current_price, atr,
             breakeven_atr, trail_atr, trail_distance, df, is_4h,
             daily_updates=None, fourh_updates=None, time_field=None):
    entry = entry or price
    is_buy = "BUY" in (direction or "")
    profit = (current_price - entry) if is_buy else (entry - current_price)
    profit_atr = profit / (atr + 1e-10)

    new_sl = sl

    # --- TP1 breach check: if TP1 ever hit, move SL to breakeven ---
    if tp1 is not None and df is not None and len(df) > 0:
        try:
            import pandas as pd
            ts = pd.Timestamp(date)
            if is_4h and time_field:
                ts = pd.Timestamp(f"{date} {time_field}")
            entry_idx = df.index.get_loc(ts)
            segment = df.iloc[entry_idx + 1:]
            if len(segment) > 0:
                tp1_hit = segment["High"].max() >= tp1 if is_buy else segment["Low"].min() <= tp1
                if tp1_hit:
                    be_sl = round(entry + 0.30, 2) if is_buy else round(entry - 0.30, 2)
                    if (is_buy and be_sl > sl) or (not is_buy and be_sl < sl):
                        new_sl = max(new_sl, be_sl) if is_buy else min(new_sl, be_sl)
        except (KeyError, Exception):
            pass

    # --- ATR-based trailing ---
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
        updates_list = daily_updates if not is_4h else fourh_updates
        updates_list.append((new_sl, pred_id))
    elif not is_buy and new_sl < sl:
        updates_list = daily_updates if not is_4h else fourh_updates
        updates_list.append((new_sl, pred_id))