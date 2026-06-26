"""
Chart pattern detection for XAUUSD
- Candlestick patterns (single & multi-bar)
- Support/resistance proximity & breakouts
- Price action features
"""
import pandas as pd
import numpy as np


def detect_candlestick(df):
    """Add candlestick pattern features (binary + ratio)"""
    df = df.copy()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    body = abs(c - o)
    rng = h - l + 1e-10
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    df["Body_Ratio"] = body / rng
    df["Upper_Wick_Ratio"] = upper / rng
    df["Lower_Wick_Ratio"] = lower / rng
    df["Candle_Color"] = (c > o).astype(int)

    # Doji
    df["Is_Doji"] = (body / rng < 0.1).astype(int)

    # Hammer: long lower wick >= 2x body, upper wick < body
    df["Is_Hammer"] = ((lower >= 2 * body) & (upper < body)).astype(int)

    # Shooting Star: long upper wick >= 2x body, lower wick < body
    df["Is_Shooting_Star"] = ((upper >= 2 * body) & (lower < body)).astype(int)

    # Pin Bar: max wick >= 3x body
    df["Is_Pin_Bar"] = ((np.maximum(upper, lower) >= 3 * body) & (body > 0)).astype(int)

    # Marubozu: very small wicks
    df["Is_Marubozu"] = ((upper / rng < 0.05) & (lower / rng < 0.05) & (body / rng > 0.8)).astype(int)

    # Engulfing
    prev_c = c.shift(1)
    prev_o = o.shift(1)
    df["Is_Bullish_Engulfing"] = (
        (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)
    ).astype(int)
    df["Is_Bearish_Engulfing"] = (
        (c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)
    ).astype(int)

    # Inside Bar
    df["Is_Inside_Bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    return df


def detect_support_resistance(df, lookback=50):
    """Add distance to S/R levels + breakout flags"""
    df = df.copy()
    h, l, c = df["High"], df["Low"], df["Close"]

    recent_high = h.rolling(lookback).max()
    recent_low = l.rolling(lookback).min()

    df["Dist_Resistance_Pct"] = (recent_high - c) / c
    df["Dist_Support_Pct"] = (c - recent_low) / c

    df["Near_Resistance"] = (df["Dist_Resistance_Pct"] < 0.005).astype(int)
    df["Near_Support"] = (df["Dist_Support_Pct"] < 0.005).astype(int)

    df["Breakout_High"] = (c > recent_high.shift(1)).astype(int)
    df["Breakout_Low"] = (c < recent_low.shift(1)).astype(int)

    # Touch count (vectorized with sliding window)
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(df)
    touch_l = np.zeros(n, dtype=int)
    touch_h = np.zeros(n, dtype=int)
    if n > lookback:
        l_wins = sliding_window_view(l.values[:n-1], lookback)
        h_wins = sliding_window_view(h.values[:n-1], lookback)
        lo = l.values[lookback:] * 0.997
        hi = l.values[lookback:] * 1.003
        ho = h.values[lookback:] * 0.997
        hih = h.values[lookback:] * 1.003
        touch_l[lookback:] = ((l_wins >= lo[:, None]) & (l_wins <= hi[:, None])).sum(axis=1)
        touch_h[lookback:] = ((h_wins >= ho[:, None]) & (h_wins <= hih[:, None])).sum(axis=1)
    df["Touch_Count_High"] = touch_h
    df["Touch_Count_Low"] = touch_l

    return df


def detect_double_top_bottom(df, lookback=30, tolerance=0.005):
    """Detect double top/bottom pattern (binary feature)"""
    df = df.copy()
    h, l, c = df["High"], df["Low"], df["Close"]

    df["Is_Double_Top"] = 0
    df["Is_Double_Bottom"] = 0
    h_arr = h.values
    l_arr = l.values

    for i in range(lookback * 2, len(df)):
        curr_h = h_arr[i]
        curr_l = l_arr[i]

        # Find local peak indices (uses full h_arr with boundary checks matching original)
        peaks = []
        for j in range(1, lookback - 1):
            idx = i - lookback + j
            if not (h_arr[idx] > h_arr[idx - 1] and h_arr[idx] > h_arr[idx + 1]):
                continue
            if idx > i - lookback + 2 and h_arr[idx] <= h_arr[idx - 2]:
                continue
            if idx < i - 2 and h_arr[idx] <= h_arr[idx + 2]:
                continue
            peaks.append((idx, h_arr[idx]))

        # Check for double top: two peaks within tolerance
        if len(peaks) >= 2:
            for p1_idx in range(len(peaks)):
                for p2_idx in range(p1_idx + 1, len(peaks)):
                    if abs(peaks[p1_idx][1] - peaks[p2_idx][1]) / peaks[p1_idx][1] < tolerance:
                        mid_min = l_arr[peaks[p1_idx][0]:peaks[p2_idx][0]].min()
                        drop_depth = (peaks[p1_idx][1] - mid_min) / peaks[p1_idx][1]
                        if drop_depth > 0.01 and curr_h <= peaks[p1_idx][1]:
                            df.loc[df.index[i], "Is_Double_Top"] = 1

        # Double Bottom (same logic, for lows)
        troughs = []
        for j in range(1, lookback - 1):
            idx = i - lookback + j
            if l_arr[idx] < l_arr[idx - 1] and l_arr[idx] < l_arr[idx + 1]:
                troughs.append((idx, l_arr[idx]))

        if len(troughs) >= 2:
            for t1_idx in range(len(troughs)):
                for t2_idx in range(t1_idx + 1, len(troughs)):
                    if abs(troughs[t1_idx][1] - troughs[t2_idx][1]) / troughs[t1_idx][1] < tolerance:
                        mid_max = h_arr[troughs[t1_idx][0]:troughs[t2_idx][0]].max()
                        rise = (mid_max - troughs[t1_idx][1]) / troughs[t1_idx][1]
                        if rise > 0.01 and curr_l >= troughs[t1_idx][1]:
                            df.loc[df.index[i], "Is_Double_Bottom"] = 1

    return df
