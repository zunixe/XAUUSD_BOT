"""
XAUUSD Paper Trading Simulation
- Virtual account, auto lot sizing, P&L tracking
- Telegram commands: /start 100, /bal, /reset
"""
import sqlite3, json
from datetime import datetime
import trading

RISK_PCT = 0.01  # 1% risk per trade
MIN_LOT = 0.01
XAU_USD_PER_MOVE = 100  # $100 per $1 price move for 1 standard lot (100 oz)
SPREAD = 0.30  # typical XAUUSD spread in dollars


def init_sim_db():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS simulation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balance REAL NOT NULL,
            initial_balance REAL NOT NULL,
            active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sim_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_id INTEGER,
            prediction_id INTEGER,
            timeframe TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp1 REAL,
            lot_size REAL,
            risk_amount REAL,
            outcome TEXT,
            pnl REAL,
            balance_before REAL,
            balance_after REAL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(sim_id) REFERENCES simulation(id)
        )
    """)
    conn.commit()
    conn.close()


def get_active_sim():
    conn = sqlite3.connect(trading.DB_FILE)
    c = conn.cursor()
    row = c.execute("SELECT id, balance, initial_balance FROM simulation WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row  # (id, balance, initial_balance) or None


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


MAX_RISK_PCT = 0.02  # max 2% risk per trade (safety cap)

def calc_lot_size(balance, entry, sl):
    sl_distance = abs(entry - sl)
    if sl_distance < 1:
        return 0, 0
    target_risk = balance * RISK_PCT
    max_allowed = balance * MAX_RISK_PCT
    lot = target_risk / (sl_distance * XAU_USD_PER_MOVE)
    max_lot = max_allowed / (sl_distance * XAU_USD_PER_MOVE)
    lot = min(lot, max_lot)
    lot = round(lot, 2)
    if lot < MIN_LOT:
        return 0, 0
    risk_amount = round(lot * sl_distance * XAU_USD_PER_MOVE, 2)
    return lot, risk_amount


def record_trade(prediction_id, timeframe, direction, entry, sl, tp1, outcome, pnl_pct, tp2=None, close_price=None):
    sim = get_active_sim()
    if not sim:
        return None
    sim_id, balance_before, _ = sim

    if outcome in ('NO_SLTP', 'PENDING'):
        return None

    lot, risk_amount = calc_lot_size(balance_before, entry, sl)
    if lot == 0:
        return None

    spread_adj = SPREAD if direction.upper().startswith("BUY") else -SPREAD
    effective_entry = entry + spread_adj

    if outcome in ('WIN', 'TP1_HIT'):
        exit_price = tp1
    elif outcome == 'TP2_HIT':
        exit_price = tp2 or tp1
    elif outcome in ('LOSS', 'SL_HIT'):
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
    c.execute("""
        INSERT INTO sim_trades (sim_id, prediction_id, timeframe, direction,
            entry, sl, tp1, lot_size, risk_amount, outcome, pnl,
            balance_before, balance_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (sim_id, prediction_id, timeframe, direction, effective_entry, sl, tp1, lot, risk_amount, outcome, pnl, balance_before, balance_after))
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
    total_pnl = sum(t[1] for t in trades)
    return {
        "initial": initial,
        "balance": balance,
        "pnl": round(balance - initial, 2),
        "return_pct": round((balance - initial) / initial * 100, 2),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
    }
