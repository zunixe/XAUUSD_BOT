"""
XAUUSD LIVE PRICE - Auto-refresh every 5 detik
Cara pakai: python live_xauusd.py
Stop: Ctrl+C
"""
import time, os, json, sys
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

def get_live_price():
    """Get XAUUSD price from multiple sources, return dict"""
    result = {"price": 0, "change": 0, "change_pct": 0, "high": 0, "low": 0, "source": "-", "time": ""}

    # Source 1: Yahoo Finance - Gold Futures (GC=F) - 1m interval
    try:
        ticker = yf.Ticker("GC=F")
        hist = ticker.history(period="1d", interval="1m")
        if hist is not None and len(hist) > 0:
            last = hist.iloc[-1]
            prev_close = ticker.info.get("regularMarketPreviousClose", 0)
            if prev_close:
                result["price"] = round(last["Close"], 2)
                result["change"] = round(last["Close"] - prev_close, 2)
                result["change_pct"] = round((last["Close"] - prev_close) / prev_close * 100, 2)
            result["high"] = round(hist["High"].max(), 2)
            result["low"] = round(hist["Low"].min(), 2)
            result["source"] = "GC=F"
            result["time"] = datetime.now().strftime("%H:%M:%S")
            return result
    except:
        pass

    # Source 2: TwelveData free tier (no API key needed for basic)
    try:
        r = requests.get("https://api.twelvedata.com/price?symbol=GC=F&apikey=demo", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if "price" in data:
                result["price"] = round(float(data["price"]), 2)
                result["source"] = "TwelveData"
                result["time"] = datetime.now().strftime("%H:%M:%S")
                return result
    except:
        pass

    # Source 3: Scrape dari situs gold price
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers, timeout=5)
        # Note: goldapi requires key, this is just backup
    except:
        pass

    return result


def get_daily_data():
    """Get daily OHLC for context"""
    try:
        df = yf.download("GC=F", period="5d", interval="1d", progress=False)
        if df is not None and len(df) > 0:
            df = df.iloc[-5:]
            return df
    except:
        return None
    return None


def print_header():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 56)
    print(f"  XAUUSD GOLD SPOT - REAL TIME PRICE TRACKER     ({datetime.now().strftime('%H:%M:%S')})")
    print("=" * 56)


def main():
    last_price = 0
    peak = 0
    valley = float("inf")
    session_high = 0
    session_low = float("inf")

    print_header()
    print("  Connecting to market data...")
    time.sleep(1)

    while True:
        try:
            print_header()
            now = datetime.now()

            # Get price
            data = get_live_price()

            if data["price"] and data["price"] > 0:
                price = data["price"]
                session_high = max(session_high, price)
                session_low = min(session_low, price)

                # Direction arrow
                if last_price > 0:
                    direction = "▲" if price > last_price else "▼" if price < last_price else "─"
                else:
                    direction = "─"

                change_str = ""
                if data["change"] != 0:
                    sign = "+" if data["change"] > 0 else ""
                    change_str = f"{sign}{data['change']:.2f} ({sign}{data['change_pct']:.2f}%)"

                print(f"  Live Price    : ${price:>8.2f}  {direction}")
                print(f"  Change        : {change_str}")
                print(f"  Source        : {data['source']}")
                print(f"  Update        : {data['time']}")
                print(f"  Session       : {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 56)
                print(f"  Session High  : ${session_high:>8.2f}")
                print(f"  Session Low   : ${session_low:>8.2f}")

                if data["high"] and data["low"]:
                    print(f"  Day Range     : ${data['low']:<8.2f} - ${data['high']:>8.2f}")

                # Market status
                hour = now.hour
                # Gold market: Sunday 6pm - Friday 5pm ET (roughly 23:00-21:00 UTC)
                print("-" * 56)

                # Daily chart - get latest daily candle for context
                daily = get_daily_data()
                if daily is not None:
                    print(f"  LATEST 5 DAYS:")
                    print(f"  {'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8}")
                    for idx, row in daily.iterrows():
                        d = idx.strftime("%b %d")
                        o = row["Open"]
                        h = row["High"]
                        l = row["Low"]
                        c = row["Close"]
                        arrow = "▲" if c > o else "▼"
                        print(f"  {d:<12} {o:>8.2f} {h:>8.2f} {l:>8.2f} {c:>8.2f} {arrow}")

                last_price = price
            else:
                print("  Waiting for data...")
                print("  (Market might be closed on weekends)")

            print(f"\n  {'─'*48}")
            print(f"  Auto-refresh every 10s | Ctrl+C to exit")
            time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n  Stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
