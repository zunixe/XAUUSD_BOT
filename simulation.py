"""
XAUUSD Paper Trading Simulation with Risk Management.
- Virtual account, auto lot sizing, P&L tracking
- Drawdown circuit breaker, loss limits, cooldown, position sizing
"""
import sqlite3, json
from datetime import datetime, timedelta
import numpy as np
import trading

import os, yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

RISK_PCT = CFG["risk"]["risk_per_trade"]
MAX_RISK_PCT = CFG["risk"]["max_risk_pct"]
MIN_LOT = 0.00001
XAU_USD_PER_MOVE = 100
SPREAD = CFG["evaluation"]["spread"]


def init_sim_db():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS simulation (
        id INTEGER PRIMARY KEY AUTOINCREMENT, balance REAL NOT NULL,
        initial_balance REAL NOT NULL, active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS sim_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, sim_id INTEGER, prediction_id INTEGER,
        timeframe TEXT, direction TEXT, entry REAL, sl REAL, tp1 REAL, lot_size REAL,
        risk_amount REAL, outcome TEXT, pnl REAL, balance_before REAL, balance_after REAL,
        created_at TEXT DEFAULT (datetime('now')), FOREIGN KEY(sim_id) REFERENCES simulation(id))""")
    conn.commit()
    conn.close()


def get_active_sim():
    conn = sqlite3.connect(trading.DB_FILE)
    row = conn.execute("SELECT id, balance, initial_balance FROM simulation WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row


def start_sim(initial_balance):
    init_sim_db()
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE simulation SET active=0 WHERE active=1")
    c.execute("INSERT INTO simulation (balance, initial_balance) VALUES (?, ?)", (initial_balance, initial_balance))
    conn.commit()
    sim_id = c.lastrowid
    conn.close()
    return sim_id


def reset_sim():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE simulation SET active=0 WHERE active=1")
    conn.commit()
    conn.close()


# ========== RISK MANAGEMENT (Phase 3) ==========

def check_risk_all():
    """All risk checks in a single DB connection. Returns dict with all results."""
    sim = get_active_sim()
    if not sim:
        return {"dd_ok": True, "dd_pct": 0, "wl_ok": True, "wl_pnl": 0,
                "consec": 0, "mc_ok": True, "mc_total": 0}
    sim_id, balance, initial = sim
    conn = sqlite3.connect(trading.DB_FILE)
    try:
        # Drawdown
        peak = conn.execute("SELECT MAX(balance_after) FROM sim_trades WHERE sim_id=?", (sim_id,)).fetchone()[0]
        peak = peak or initial
        drawdown = (peak - balance) / peak if peak > 0 else 0
        dd_limit = CFG["risk"]["max_drawdown_pct"]

        # Weekly loss
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        weekly_pnl = conn.execute(
            "SELECT SUM(pnl) FROM sim_trades WHERE sim_id=? AND created_at >= ?",
            (sim_id, week_ago)).fetchone()[0] or 0
        wl_limit = balance * CFG["risk"]["weekly_loss_limit_pct"]

        # Consecutive losses
        trades = conn.execute(
            "SELECT outcome FROM sim_trades WHERE sim_id=? ORDER BY id DESC LIMIT 10",
            (sim_id,)).fetchall()
        consec = 0
        for (outcome,) in trades:
            if outcome == "LOSS":
                consec += 1
            else:
                break

        # Active positions count
        d_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE outcome IS NULL").fetchone()[0]
        f_count = conn.execute("SELECT COUNT(*) FROM predictions_4h WHERE outcome IS NULL").fetchone()[0]
        mc_total = d_count + f_count
        mc_limit = CFG["risk"]["max_concurrent_positions"]

        return {
            "dd_ok": drawdown < dd_limit, "dd_pct": drawdown,
            "wl_ok": weekly_pnl > -wl_limit, "wl_pnl": weekly_pnl,
            "consec": consec,
            "mc_ok": mc_total < mc_limit, "mc_total": mc_total,
        }
    finally:
        conn.close()


# Keep individual functions for backward compatibility
def check_drawdown_limit():
    r = check_risk_all()
    return r["dd_ok"], r["dd_pct"]

def check_weekly_loss_limit():
    r = check_risk_all()
    return r["wl_ok"], r["wl_pnl"]

def get_consecutive_losses():
    return check_risk_all()["consec"]

def check_max_concurrent():
    r = check_risk_all()
    return r["mc_ok"], r["mc_total"]


def get_active_positions():
    """Get currently active (unevaluated) positions across timeframes."""
    conn = sqlite3.connect(trading.DB_FILE)
    try:
        daily = conn.execute("SELECT id, predicted_direction FROM predictions WHERE outcome IS NULL").fetchall()
        fourh = conn.execute("SELECT id, predicted_direction FROM predictions_4h WHERE outcome IS NULL").fetchall()
    finally:
        conn.close()
    return daily, fourh


def adjust_lot_for_correlation(lot, direction):
    """Reduce lot if same-direction position exists in other timeframe."""
    daily, fourh = get_active_positions()
    same_dir = 0
    for rows in [daily, fourh]:
        for _, dir_str in rows:
            if dir_str and direction[:3] in dir_str:
                same_dir += 1
    if same_dir > 0:
        lot = round(lot * 0.5, 2)
    return max(lot, MIN_LOT)


def get_volatility_regime():
    """Return volatility regime: 'low', 'normal', 'high', 'extreme'."""
    try:
        import pandas as pd
        df = pd.read_csv(trading.CSV_DAILY, parse_dates=["Date"], index_col="Date")
        close, high, low = df["Close"], df["High"], df["Low"]
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        pctile = atr.rank(pct=True).iloc[-1]
        if pctile < 0.2: return "low"
        if pctile < 0.6: return "normal"
        if pctile < 0.85: return "high"
        return "extreme"
    except Exception:
        return "normal"


def calc_lot_size(balance, entry, sl):
    """Calculate lot size with volatility adjustment."""
    sl_distance = abs(entry - sl)
    if sl_distance < entry * 0.0003:
        return 0, 0
    target_risk = balance * RISK_PCT
    max_allowed = balance * MAX_RISK_PCT
    lot = target_risk / (sl_distance * XAU_USD_PER_MOVE)
    max_lot = max_allowed / (sl_distance * XAU_USD_PER_MOVE)
    lot = min(lot, max_lot)

    # Volatility regime adjustment
    regime = get_volatility_regime()
    regime_mult = CFG["volatility_regime"]
    mult = {"low": regime_mult["low_mult"], "normal": regime_mult["normal_mult"],
            "high": regime_mult["high_mult"], "extreme": regime_mult["extreme_mult"]}
    lot = round(lot * mult.get(regime, 1.0), 5)

    if lot < MIN_LOT:
        return 0, 0
    risk_amount = round(lot * sl_distance * XAU_USD_PER_MOVE, 2)
    return lot, risk_amount


def record_trade(prediction_id, timeframe, direction, entry, sl, tp1, outcome, pnl_pct, tp2=None, close_price=None, outcome_detail=None):
    sim = get_active_sim()
    if not sim:
        return None
    sim_id, balance_before, _ = sim
    if outcome in ('NO_SLTP', 'PENDING'):
        return None

    lot, risk_amount = calc_lot_size(balance_before, entry, sl)
    if lot == 0:
        return None

    lot = adjust_lot_for_correlation(lot, direction)

    spread_adj = SPREAD if direction.upper().startswith("BUY") else -SPREAD
    effective_entry = entry + spread_adj

    detail = outcome_detail or ""
    if outcome == 'WIN':
        exit_price = tp2 if "TP2" in detail and tp2 else tp1
    elif outcome == 'LOSS':
        exit_price = sl
    elif outcome == 'EXPIRED':
        exit_price = close_price or entry
    else:
        exit_price = sl

    price_diff = exit_price - effective_entry if direction.upper().startswith("BUY") else effective_entry - exit_price
    pnl = round(lot * price_diff * XAU_USD_PER_MOVE, 2)
    balance_after = round(balance_before + pnl, 2)

    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("""INSERT INTO sim_trades (sim_id, prediction_id, timeframe, direction,
        entry, sl, tp1, lot_size, risk_amount, outcome, pnl, balance_before, balance_after)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sim_id, prediction_id, timeframe, direction, effective_entry, sl, tp1, lot, risk_amount, outcome, pnl, balance_before, balance_after))
    c.execute("UPDATE simulation SET balance=? WHERE id=?", (balance_after, sim_id))
    conn.commit()
    conn.close()
    return {"lot": lot, "pnl": pnl, "balance_before": balance_before, "balance_after": balance_after}


def get_sim_summary():
    sim = get_active_sim()
    if not sim:
        return None
    sim_id, balance, initial = sim
    conn = sqlite3.connect(trading.DB_FILE)
    trades = conn.execute("SELECT outcome, pnl FROM sim_trades WHERE sim_id=? ORDER BY id", (sim_id,)).fetchall()
    conn.close()
    wins = sum(1 for t in trades if t[0] == "WIN")
    losses = sum(1 for t in trades if t[0] == "LOSS")
    return {
        "initial": initial, "balance": balance,
        "pnl": round(balance - initial, 2),
        "return_pct": round((balance - initial) / initial * 100, 2),
        "trades": len(trades), "wins": wins, "losses": losses,
    }


def get_performance_stats():
    """Extended performance analytics (Phase 4)."""
    sim = get_active_sim()
    if not sim:
        return None
    sim_id, balance, initial = sim
    conn = sqlite3.connect(trading.DB_FILE)
    trades = conn.execute("SELECT outcome, pnl, created_at FROM sim_trades WHERE sim_id=? ORDER BY id", (sim_id,)).fetchall()
    conn.close()
    if not trades:
        return None

    pnls = [t[1] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Sharpe ratio (simplified)
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    equity = np.cumsum([initial] + pnls)
    running_max = np.maximum.accumulate(equity)
    max_dd = (equity - running_max) / (running_max + 1e-10)

    return {
        "win_rate": win_rate, "expectancy": expectancy,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": abs(sum(wins) / (sum(losses) + 1e-10)),
        "sharpe": sharpe, "max_dd": float(max_dd.min()),
        "total_trades": len(pnls),
    }
