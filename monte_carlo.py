"""Monte Carlo simulation for trade sequence analysis."""
import numpy as np
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "xauusd_journal.db")


def run_monte_carlo(n_sims=10000, table="predictions"):
    """Simulate trade sequence randomization. Returns dict of statistics."""
    conn = sqlite3.connect(DB_FILE)
    try:
        trades = conn.execute(f"SELECT result_pct FROM {table} WHERE outcome IS NOT NULL").fetchall()
    except Exception:
        conn.close()
        return None
    conn.close()

    if len(trades) < 20:
        return None

    returns = np.array([t[0] / 100 for t in trades])
    results = []

    for _ in range(n_sims):
        shuffled = np.random.choice(returns, size=len(returns), replace=True)
        equity = np.cumprod(1 + shuffled)
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity / running_max - 1
        results.append({
            "return": equity[-1] - 1,
            "max_dd": drawdowns.min(),
        })

    returns_arr = np.array([r["return"] for r in results])
    dd_arr = np.array([r["max_dd"] for r in results])

    return {
        "n_trades": len(returns),
        "median_return": float(np.median(returns_arr)),
        "p5_return": float(np.percentile(returns_arr, 5)),
        "p95_return": float(np.percentile(returns_arr, 95)),
        "median_max_dd": float(np.median(dd_arr)),
        "p5_max_dd": float(np.percentile(dd_arr, 5)),
        "prob_ruin": float((dd_arr < -0.50).mean()),
        "prob_profit": float((returns_arr > 0).mean()),
    }


def format_mc_report(stats):
    """Format Monte Carlo results as string."""
    if not stats:
        return "Monte Carlo: insufficient data (need 20+ trades)"
    return (
        f"Monte Carlo ({stats['n_trades']} trades, 10000 sims):\n"
        f"  Median return: {stats['median_return']:+.1%}\n"
        f"  5th-95th pctl: {stats['p5_return']:+.1%} to {stats['p95_return']:+.1%}\n"
        f"  Median max DD: {stats['median_max_dd']:.1%}\n"
        f"  Prob of ruin:  {stats['prob_ruin']:.1%}\n"
        f"  Prob of profit:{stats['prob_profit']:.1%}"
    )
