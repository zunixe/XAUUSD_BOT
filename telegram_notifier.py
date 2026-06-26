"""
Telegram notifier for XAUUSD trade signals.
Reads credentials from Hermes Agent .env, sends via Telegram Bot API.
Uses shared trading.py for levels/TP/SL.
"""
import os, re, json, time, sys, threading, subprocess, urllib.request, urllib.error, ssl
ssl._create_default_https_context = ssl._create_unverified_context
from datetime import datetime
import yfinance as yf
import trading
import simulation as sim

HERMES_ENV = os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", ".env")
ENTRY_ARROW = "▶"  # indicator for entry line


def _read_env(path):
    creds = {}
    if not os.path.exists(path):
        return creds
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^(TELEGRAM_\w+)=["\']?(.*?)["\']?$', line)
            if m:
                creds[m.group(1)] = m.group(2).strip()
    return creds


def get_realtime_price():
    """Fetch live XAUUSD spot price from Kitco (free, no API key needed) + yfinance fallback"""
    # Try Kitco first
    for attempt in range(2):
        try:
            url = "https://www.kitco.com/gold-price-today-usa/"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            # Multiple extraction strategies
            gold = None
            for pat in [r'"bid":([\d.]+)', r'"ask":([\d.]+)', r'id="sp-bid"[^>]*>([\d.]+)', r'(\d{4}\.\d{2})', r'(\d{3}\d{2}\.\d{2})']:
                matches = re.findall(pat, html)
                for m in matches:
                    val = float(m)
                    if 2000 < val < 10000:
                        gold = val
                        break
                if gold:
                    break
            if gold:
                return {"price": round(gold, 2), "change": 0, "high": 0, "low": 0, "source": "Kitco (spot)"}
            if attempt == 0:
                time.sleep(2)
        except Exception:
            if attempt == 0:
                time.sleep(2)
    # Fallback: yfinance GC=F 1m
    try:
        ticker = yf.Ticker("GC=F")
        hist = ticker.history(period="1d", interval="1m")
        if hist is not None and len(hist) > 0:
            last = hist.iloc[-1]
            return {
                "price": round(last["Close"], 2),
                "change": 0,
                "high": round(hist["High"].max(), 2),
                "low": round(hist["Low"].min(), 2),
                "source": "GC=F (delayed)",
            }
    except Exception:
        pass
    return None


def send_signal(prediction_id, direction, confidence, daily_price, target_date, threshold,
                bot_token=None, chat_id=None):
    """
    Send signal notification to Telegram.
    Returns dict with (sl, tp1, tp2, entry_realtime) for the caller to save.
    """
    if bot_token is None or chat_id is None:
        creds = _read_env(HERMES_ENV)
        bot_token = bot_token or creds.get("TELEGRAM_TRADING_BOT") or creds.get("TELEGRAM_BOT_TOKEN")
        chat_id = chat_id or creds.get("TELEGRAM_TRADING_CHAT") or creds.get("TELEGRAM_HOME_CHANNEL") or creds.get("TELEGRAM_ALLOWED_USERS", "")
    token = bot_token
    chat_id = str(chat_id).strip()

    if not token or not chat_id:
        print("[TELEGRAM] Token atau Chat ID tidak ditemukan")
        return None

    live = get_realtime_price()
    entry = live["price"] if live and live["price"] else daily_price
    change_str = f" ({'+' if live['change'] > 0 else ''}{live['change']})" if live and live.get("change") else ""

    df = trading.load_daily(trading.CSV_DAILY)
    levels = trading.calculate_levels(df)
    is_buy = "BUY" in direction
    sl, tp1, tp2 = trading.compute_tp_sl(levels, is_buy, entry)
    risk = round(abs(entry - sl), 2)
    rr1 = round(abs(tp1 - entry) / risk, 2) if risk > 0 else 0
    rr2 = round(abs(tp2 - entry) / risk, 2) if risk > 0 else 0

    emoji = "BUY" if is_buy else "SELL"
    arrow = "LONG" if is_buy else "SHORT"
    atr_val = levels["atr"]
    rsi_val = levels["rsi"]

    p = lambda s, v: f"  {s:13s}: {v}"
    body = (
        f"{p('Confidence', f'{confidence:.1%} (threshold {threshold:.3f})')}\n"
        f"{p('Daily Close', f'${daily_price:.2f}')}\n"
        f"{p(f'{ENTRY_ARROW} Entry', f'${entry:.2f}{change_str}')}\n"
        f"{p('SL', f'${sl:.2f} (-${risk:.2f})')}\n"
        f"{p('TP1', f'${tp1:.2f} (+${tp1 - entry:.2f}, RR 1:{rr1:.2f})')}\n"
        f"{p('TP2', f'${tp2:.2f} (+${tp2 - entry:.2f}, RR 1:{rr2:.2f})')}\n"
        f"{p('ATR / RSI', f'${atr_val:.1f} / {rsi_val:.0f}')}\n"
        f"{p('Target', target_date)}"
    )
    msg = (
        f"<b>\U0001F525 [XAUUSD SIGNAL] #{prediction_id} — Daily Candle</b>\n"
        f"{emoji} {arrow}\n"
        f"<pre>\n{body}\n</pre>"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                print(f"[TELEGRAM] Signal #{prediction_id} terkirim via real-time price ${entry:.2f}")
            else:
                print(f"[TELEGRAM] Gagal: {result}")
    except urllib.error.URLError as e:
        print(f"[TELEGRAM] Error: {e}")

    return {"sl": sl, "tp1": tp1, "tp2": tp2, "entry_realtime": entry, "levels": levels}


def _send_telegram(msg, bot_token=None, chat_id=None):
    """Low-level send to Telegram"""
    if bot_token is None or chat_id is None:
        creds = _read_env(HERMES_ENV)
        bot_token = bot_token or creds.get("TELEGRAM_TRADING_BOT") or creds.get("TELEGRAM_BOT_TOKEN")
        chat_id = chat_id or creds.get("TELEGRAM_TRADING_CHAT") or creds.get("TELEGRAM_HOME_CHANNEL", "")
    chat_id = str(chat_id).strip()
    if not bot_token or not chat_id:
        print("[TELEGRAM] Missing credentials")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception:
        return False


def send_4h_signal(prediction_id, direction, confidence, entry, close_4h, target, threshold, sl, tp1, tp2):
    """Send 4H candle signal to Telegram"""
    is_buy = direction == "BUY"
    risk = round(abs(entry - sl), 2) if sl else 0
    rr1 = round(abs(tp1 - entry) / risk, 2) if risk > 0 else 0
    rr2 = round(abs(tp2 - entry) / risk, 2) if risk > 0 else 0
    emoji = "BUY" if is_buy else "SELL"
    arrow = "LONG" if is_buy else "SHORT"

    p = lambda s, v: f"  {s:13s}: {v}"
    body = (
        f"{p('Confidence', f'{confidence:.1%} (threshold {threshold:.3f})')}\n"
        f"{p('4H Close', f'${close_4h:.2f}')}\n"
        f"{p(f'{ENTRY_ARROW} Entry', f'${entry:.2f}')}\n"
        f"{p('SL', f'${sl:.2f} (-${risk:.2f})')}\n"
        f"{p('TP1', f'${tp1:.2f} (+${tp1 - entry:.2f}, RR 1:{rr1:.2f})')}\n"
        f"{p('TP2', f'${tp2:.2f} (+${tp2 - entry:.2f}, RR 1:{rr2:.2f})')}\n"
        f"{p('Target', target)}"
    )
    msg = (
        f"<b>\U0001F535 [XAUUSD SIGNAL] #{prediction_id} — 4H Candle</b>\n"
        f"{emoji} {arrow}\n"
        f"<pre>\n{body}\n</pre>"
    )
    ok = _send_telegram(msg)
    print(f"[TELEGRAM] 4H Signal #{prediction_id} {'terkirim' if ok else 'gagal'} (${entry:.2f})")


def send_outcome_notification(prediction_id, timeframe, direction, entry, outcome,
                               detail, pct, sl=None, tp1=None, tp2=None):
    """Notify when a signal hits TP/SL or expires + sim P&L"""
    sim_line = ""
    pnl_display = None
    try:
        r = sim.record_trade(prediction_id, timeframe, direction, entry, sl, tp1, outcome, pct)
        if r:
            sim_line = f"\n  Sim: {r['lot']} lot | P&L: ${r['pnl']:+.2f} | Bal: ${r['balance_after']:.2f}"
            pnl_display = r['pnl']
    except Exception as e:
        print(f"[TELEGRAM] sim record_trade error: {e}")

    if outcome == "WIN":
        if "TP2" in (detail or ""):
            icon = "\U0001F7E2"
            label = f"TP2 HIT! +{pct:.2f}%"
        elif "TP1" in (detail or ""):
            icon = "\U0001F7E2"
            label = f"TP1 HIT! +{pct:.2f}%"
        else:
            icon = "\U0001F7E2"
            label = f"WIN +{pct:.2f}%"
        pnl_str = f"${pnl_display:+.2f}" if pnl_display is not None else f"${(pct/100)*entry:.2f}"
        line = f"  Profit: {pnl_str} ({pct:+.2f}%)"
    else:
        if "SL" in (detail or ""):
            icon = "\U0001F534"
            label = f"SL HIT! {pct:.2f}%"
        elif "EXPIRED" in (detail or ""):
            icon = "\U0001F7E1"
            label = f"EXPIRED {pct:.2f}%"
        else:
            icon = "\U0001F534"
            label = f"LOSS {pct:.2f}%"
        pnl_str = f"${pnl_display:+.2f}" if pnl_display is not None else f"-${abs(pct/100)*entry:.2f}"
        line = f"  Loss: {pnl_str} ({pct:+.2f}%)"

    msg = (
        f"<b>{icon} [CLOSED] #{prediction_id} — {timeframe}</b>\n"
        f"  Direction: {direction}\n"
        f"  Entry: ${entry:.2f}\n"
        f"  Result: {label}\n"
        f"{line}"
        f"{sim_line}"
    )
    ok = _send_telegram(msg)
    print(f"[TELEGRAM] Outcome #{prediction_id}: {outcome} ({detail}) {'ok' if ok else 'gagal'}")


def send_report(msg):
    """Send a report (weekly/monthly) to Telegram"""
    ok = _send_telegram(msg)
    print(f"[TELEGRAM] Report {'terkirim' if ok else 'gagal'}")


_retrain_result = {"running": False, "result": "", "chat_id": None, "token": None}

def _retrain_worker():
    try:
        r1 = subprocess.run([sys.executable, os.path.join(trading.BASE_DIR, "auto_runner.py"), "--retrain"],
                            capture_output=True, text=True, timeout=300, cwd=trading.BASE_DIR)
        r2 = subprocess.run([sys.executable, os.path.join(trading.BASE_DIR, "runner_4h.py"), "--retrain"],
                            capture_output=True, text=True, timeout=300, cwd=trading.BASE_DIR)
        daily_out = [l for l in r1.stdout.split("\n") if "Retrain selesai" in l or "Fold " in l or "acc=" in l][-3:]
        fourh_out = [l for l in r2.stdout.split("\n") if "Fold " in l or "avg acc" in l.lower() or "saved" in l.lower()][-3:]
        lines = ["✅ Retrain selesai!"]
        if daily_out:
            lines.append("Daily:")
            for l in daily_out:
                lines.append(f"  {l.strip()}")
        if fourh_out:
            lines.append("4H:")
            for l in fourh_out:
                lines.append(f"  {l.strip()}")
        _retrain_result["result"] = "\n".join(lines)
    except subprocess.TimeoutExpired:
        _retrain_result["result"] = "❌ Retrain timeout (>5 menit)"
    except Exception as e:
        _retrain_result["result"] = f"❌ Retrain error: {e}"
    finally:
        _retrain_result["running"] = False
        # Send async notification to Telegram
        if _retrain_result["token"] and _retrain_result["chat_id"]:
            try:
                payload = json.dumps({"chat_id": _retrain_result["chat_id"], "text": _retrain_result["result"],
                    "parse_mode": "HTML"}).encode()
                url = f"https://api.telegram.org/bot{_retrain_result['token']}/sendMessage"
                req = urllib.request.Request(url, data=payload, method="POST",
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                pass


def process_commands():
    """Check Telegram for simulation commands and process them"""
    creds = _read_env(HERMES_ENV)
    token = creds.get("TELEGRAM_TRADING_BOT") or creds.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if not data.get("ok"):
            return
        for update in data["result"]:
            msg = update.get("message", {})
            text = (msg.get("text") or "").strip()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if not text or not chat_id:
                continue
            reply = None
            if text.startswith("/start"):
                parts = text.split()
                try:
                    amount = float(parts[1])
                except Exception:
                    amount = 100
                sim.start_sim(amount)
                s = sim.get_sim_summary()
                reply = f"✅ Simulasi dimulai dengan ${amount:.2f}\n  Balance: ${s['balance']:.2f}"
            elif text == "/bal":
                s = sim.get_sim_summary()
                if s:
                    reply = (f"💰 <b>Simulation</b>\n"
                             f"  Balance : ${s['balance']:.2f}\n"
                             f"  P&L     : {s['pnl']:+.2f} ({s['return_pct']:+.2f}%)\n"
                             f"  Trades  : {s['trades']} ({s['wins']}W / {s['losses']}L)")
                else:
                    reply = "❌ Belum ada simulasi. Ketik /start 100"
            elif text == "/reset":
                sim.reset_sim()
                reply = "🔄 Simulasi direset."
            elif text == "/price":
                live = get_realtime_price()
                if live:
                    reply = (f"💰 <b>XAUUSD Real-time Price</b>\n"
                             f"  Price : ${live['price']:.2f}\n"
                             f"  Source: {live['source']}")
                else:
                    reply = "❌ Gagal ambil harga real-time"
            elif text == "/info":
                try:
                    import sqlite3
                    this_month = datetime.now().strftime("%Y-%m")
                    conn = sqlite3.connect(trading.DB_FILE)
                    c = conn.cursor()
                    # Evaluated outcomes
                    daily_eval = c.execute(
                        "SELECT outcome, COUNT(*) FROM predictions WHERE date LIKE ? AND outcome IN ('WIN','LOSS') GROUP BY outcome",
                        (f"{this_month}%",)).fetchall()
                    fourh_eval = c.execute(
                        "SELECT outcome, COUNT(*) FROM predictions_4h WHERE date LIKE ? AND outcome IN ('WIN','LOSS') GROUP BY outcome",
                        (f"{this_month}%",)).fetchall()
                    # Active (unevaluated) — check vs real-time price
                    live = get_realtime_price()
                    curr = live["price"] if live else None
                    daily_active = c.execute(
                        "SELECT id, price, predicted_direction, sl, tp1, tp2, entry_realtime FROM predictions WHERE date LIKE ? AND outcome IS NULL",
                        (f"{this_month}%",)).fetchall()
                    fourh_active = c.execute(
                        "SELECT id, predicted_direction, entry_realtime, sl, tp1 FROM predictions_4h WHERE date LIKE ? AND outcome IS NULL",
                        (f"{this_month}%",)).fetchall()
                    # Count hits
                    d_hit_sl = d_hit_tp = 0
                    for r in daily_active:
                        _, price, direction, sl, tp1, tp2, entry_rt = r
                        entry = entry_rt or price
                        if not sl or not curr: continue
                        is_buy = "BUY" in (direction or "")
                        if (is_buy and curr <= sl) or (not is_buy and curr >= sl):
                            d_hit_sl += 1
                        elif tp1 and ((is_buy and curr >= tp1) or (not is_buy and curr <= tp1)):
                            d_hit_tp += 1
                    f_hit_sl = f_hit_tp = 0
                    for r in fourh_active:
                        _, direction, entry_rt, sl, tp1 = r
                        if not sl or not curr or not entry_rt: continue
                        is_buy = direction == "BUY"
                        if (is_buy and curr <= sl) or (not is_buy and curr >= sl):
                            f_hit_sl += 1
                        elif tp1 and ((is_buy and curr >= tp1) or (not is_buy and curr <= tp1)):
                            f_hit_tp += 1
                    # Build reply
                    lines = [f"📊 <b>Monthly — {this_month}</b>"]
                    d_w = sum(c for o,c in daily_eval if o=="WIN")
                    d_l = sum(c for o,c in daily_eval if o=="LOSS")
                    f_w = sum(c for o,c in fourh_eval if o=="WIN")
                    f_l = sum(c for o,c in fourh_eval if o=="LOSS")
                    total_w = d_w + f_w
                    total_l = d_l + f_l
                    evaluated = total_w + total_l
                    if evaluated > 0:
                        lines.append(f"  Evaluated: {total_w}W / {total_l}L ({total_w/evaluated*100:.0f}%)")
                    # Active with real-time hits
                    total_active = len(daily_active) + len(fourh_active)
                    total_hit_sl = d_hit_sl + f_hit_sl
                    total_hit_tp = d_hit_tp + f_hit_tp
                    lines.append(f"  Active: {total_active} pending")
                    if total_hit_sl > 0 or total_hit_tp > 0:
                        lines.append(f"  Real-time:")
                        if total_hit_sl > 0: lines.append(f"    SL hit : {total_hit_sl}")
                        if total_hit_tp > 0: lines.append(f"    TP hit : {total_hit_tp}")
                    if curr:
                        lines.append(f"  Current: ${curr:.2f}")
                    # Recent closed trades with exit price
                    c2 = conn.cursor()
                    closed = c2.execute("""
                        SELECT id, predicted_direction, outcome, outcome_detail, result_pct, sl, tp1, tp2, date
                        FROM predictions WHERE outcome IS NOT NULL AND date LIKE ?
                        ORDER BY id DESC LIMIT 5
                    """, (f"{this_month}%",)).fetchall()
                    if closed:
                        lines.append(f"\n  Closed Daily:")
                        for r in closed:
                            pid, direc, outcome, detail, pct, sl, tp1, tp2, d = r
                            exit_p = {"SL_HIT": sl, "TP1_HIT": tp1, "TP2_HIT": tp2}.get(detail, "?")
                            icon = "+" if outcome == "WIN" else "-"
                            exit_str = f" @ ${exit_p:.2f}" if isinstance(exit_p, float) else ""
                            short_dir = direc[:4] if direc else "?"
                            lines.append(f"    #{pid} {short_dir} {icon}{abs(pct):.1f}%{exit_str}")
                    c3 = conn.cursor()
                    closed4 = c3.execute("""
                        SELECT id, predicted_direction, outcome, outcome_detail, result_pct, sl, tp1, date
                        FROM predictions_4h WHERE outcome IS NOT NULL AND date LIKE ?
                        ORDER BY id DESC LIMIT 5
                    """, (f"{this_month}%",)).fetchall()
                    if closed4:
                        lines.append(f"\n  Closed 4H:")
                        for r in closed4:
                            pid, direc, outcome, detail, pct, sl, tp1, d = r
                            exit_p = {"SL_HIT": sl, "TP1_HIT": tp1}.get(detail, "?")
                            icon = "+" if outcome == "WIN" else "-"
                            exit_str = f" @ ${exit_p:.2f}" if isinstance(exit_p, float) else ""
                            short_dir = direc[:4] if direc else "?"
                            lines.append(f"    #{pid} {short_dir} {icon}{abs(pct):.1f}%{exit_str}")
                    conn.close()
                    reply = "\n".join(lines)
                except Exception as e:
                    reply = f"❌ Info error: {e}"
            elif text == "/retrain":
                if _retrain_result["running"]:
                    reply = "⏳ Retrain sedang berjalan..."
                else:
                    _retrain_result["running"] = True
                    _retrain_result["token"] = token
                    _retrain_result["chat_id"] = chat_id
                    t = threading.Thread(target=_retrain_worker, daemon=True)
                    t.start()
                    reply = "⏳ Retrain dimulai, akan dikirim notifikasi setelah selesai..."
            if reply:
                keyboard = {
                    "keyboard": [["/bal", "/price"], ["/start 100"], ["/retrain", "/info"], ["/reset"]],
                    "resize_keyboard": True,
                    "one_time_keyboard": False
                }
                url_send = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = json.dumps({
                    "chat_id": chat_id, "text": reply,
                    "parse_mode": "HTML",
                    "reply_markup": keyboard
                }).encode()
                try:
                    r = urllib.request.urlopen(urllib.request.Request(url_send, data=payload,
                        method="POST", headers={"Content-Type": "application/json"}), timeout=10)
                    ok = json.loads(r.read()).get("ok", False)
                except Exception as send_err:
                    ok = False
                    with open(os.path.join(trading.BASE_DIR, "poll.log"), "a") as f:
                        f.write(f"[{datetime.now()}] send failed: {send_err}\n")
                with open(os.path.join(trading.BASE_DIR, "poll.log"), "a") as f:
                    f.write(f"[{datetime.now()}] cmd '{text}' from {chat_id}: reply sent={ok}\n")
        # Mark all as processed
        if data["result"]:
            last_id = data["result"][-1]["update_id"]
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getUpdates?offset={last_id + 1}", timeout=5)
    except Exception as e:
        with open(os.path.join(trading.BASE_DIR, "poll.log"), "a") as f:
            f.write(f"[{datetime.now()}] process_commands error: {e}\n")


if __name__ == "__main__":
    send_signal(999, "BUY (Bullish)", 0.75, 4210.80, "2026-06-26", 0.55)
