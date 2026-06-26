import pandas as pd, numpy as np

df = pd.read_csv("xauusd_daily.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

c, h, l = df["Close"], df["High"], df["Low"]

# RSI
d = c.diff()
g = d.where(d > 0, 0).rolling(14).mean()
ls = (-d.where(d < 0, 0)).rolling(14).mean()
rsi = 100 - 100 / (1 + g / (ls + 1e-10))

# ATR
tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

# EMA
ema20 = c.ewm(span=20).mean()
ema50 = c.ewm(span=50).mean()
ema200 = c.ewm(span=200).mean()

# BB
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()

# Pivot Points harian
open_p = df["Open"]
high_p = h
low_p = l
close_p = c

# Fibonacci retracement dari swing terakhir
swing_high = h.rolling(50).max()
swing_low = l.rolling(50).min()
range_fib = swing_high - swing_low
fib_382 = swing_high - range_fib * 0.382
fib_500 = swing_high - range_fib * 0.5
fib_618 = swing_high - range_fib * 0.618

print("=" * 55)
print("  XAUUSD TECHNICAL LEVELS - 22 JUNE 2026")
print("=" * 55)
print(f"  Close Terakhir    : ${c.iloc[-1]:>8.2f}")
print(f"  Open Hari Ini     : ${open_p.iloc[-1]:>8.2f}")
print(f"  High Hari Ini     : ${h.iloc[-1]:>8.2f}")
print(f"  Low Hari Ini      : ${l.iloc[-1]:>8.2f}")
print("-" * 55)
print(f"  RSI(14)           : {rsi.iloc[-1]:>8.1f}  {'(oversold)' if rsi.iloc[-1] < 35 else '(netral)' if rsi.iloc[-1] < 65 else '(overbought)'}")
print(f"  ATR(14)           : ${atr.iloc[-1]:>8.1f}")
print("-" * 55)
print("  MOVING AVERAGES:")
print(f"  EMA 20            : ${ema20.iloc[-1]:>8.2f}")
print(f"  EMA 50            : ${ema50.iloc[-1]:>8.2f}")
print(f"  EMA 200           : ${ema200.iloc[-1]:>8.2f}")
print("-" * 55)
print("  BOLLINGER BANDS:")
print(f"  Upper Band        : ${(bb_mid + 2*bb_std).iloc[-1]:>8.2f}")
print(f"  Middle Band       : ${bb_mid.iloc[-1]:>8.2f}")
print(f"  Lower Band        : ${(bb_mid - 2*bb_std).iloc[-1]:>8.2f}")
print("-" * 55)
print("  SUPPORT & RESISTANCE:")
print(f"  R2                : ${(swing_high.iloc[-1]):>8.2f}  (50-day high)")
print(f"  R1                : ${(ema20.iloc[-1]):>8.2f}  (EMA 20)")
print(f"  Pivot             : ${c.iloc[-1]:>8.2f}  (current)")
print(f"  S1                : ${(ema50.iloc[-1]):>8.2f}  (EMA 50)")
print(f"  S2                : ${(l.rolling(20).min().iloc[-1]):>8.2f}  (20-day low)")
print(f"  S3                : ${(l.rolling(50).min().iloc[-1]):>8.2f}  (50-day low)")
print("-" * 55)
print("  FIBONACCI (50-day swing):")
print(f"  0.0%  (low)      : ${swing_low.iloc[-1]:>8.2f}")
print(f"  38.2%             : ${fib_382.iloc[-1]:>8.2f}")
print(f"  50.0%             : ${fib_500.iloc[-1]:>8.2f}")
print(f"  61.8%             : ${fib_618.iloc[-1]:>8.2f}")
print(f"  100%  (high)     : ${swing_high.iloc[-1]:>8.2f}")
print("=" * 55)

# RECOMMENDATION
prediction_bullish = True  # dari model kita
print()
print("=" * 55)
print("  REKOMENDASI TRADING")
print("=" * 55)

entry = c.iloc[-1]
bb_lower_val = (bb_mid - 2 * bb_std).iloc[-1]
sup_20_val = l.rolling(20).min().iloc[-1]
sup_50_val = l.rolling(50).min().iloc[-1]
atr_val = atr.iloc[-1]

if prediction_bullish:
    # SL di bawah recent low atau BB Lower, mana yang lebih dekat
    sl = max(sup_20_val - atr_val * 0.3, bb_lower_val - atr_val * 0.2)
    tp1 = min(ema20.iloc[-1], fib_618.iloc[-1])  # ~$4,347
    tp2 = ema50.iloc[-1]  # ~$4,492
    risk = abs(entry - sl)
    rr1 = abs(tp1 - entry) / risk if risk > 0 else 0
    rr2 = abs(tp2 - entry) / risk if risk > 0 else 0
    print(f"  ARAH        : BUY (LONG)")
    print(f"  ENTRY       : ${entry:.2f}")
    print(f"  STOP LOSS   : ${sl:.2f}  (${risk:.2f} risk)")
    print(f"  TP 1        : ${tp1:.2f}  (+${tp1-entry:.2f}, RR 1:{rr1:.2f})")
    print(f"  TP 2        : ${tp2:.2f}  (+${tp2-entry:.2f}, RR 1:{rr2:.2f})")
else:
    sl = min(ema20.iloc[-1] + atr_val * 0.5, swing_high.iloc[-1])
    tp1 = ema50.iloc[-1]
    tp2 = sup_20_val
    risk = abs(sl - entry)
    rr1 = abs(entry - tp1) / risk if risk > 0 else 0
    rr2 = abs(entry - tp2) / risk if risk > 0 else 0
    print(f"  ARAH        : SELL (SHORT)")
    print(f"  ENTRY       : ${entry:.2f}")
    print(f"  STOP LOSS   : ${sl:.2f}  (${risk:.2f} risk)")
    print(f"  TP 1        : ${tp1:.2f}  (-${entry-tp1:.2f}, RR 1:{rr1:.2f})")
    print(f"  TP 2        : ${tp2:.2f}  (-${entry-tp2:.2f}, RR 1:{rr2:.2f})")

print(f"  LOT SIZE    : 0.01 per $100 modal")

print("-" * 55)
print(f"  RISIKO/TRADE: ${risk:.2f} per lot")
print(f"  MODAL MIN   : ${risk * 100:.0f} (untuk risiko 1% per trade)")
print("=" * 55)
print()
print("  CATATAN:")
print("  - Harga bisa tembus SL karena volatilitas tinggi (ATR $105)")
print("  - Konfirmasi entry: candle bullish close di atas EMA 20")
print("  - Jika harga turun < $4,000, sinyal buy menjadi invalid")
print("=" * 55)
