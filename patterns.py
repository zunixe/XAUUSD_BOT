"""
Chart pattern detection for XAUUSD
- Candlestick patterns (single & multi-bar)
- Support/resistance proximity & breakouts
- Price action features
"""
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def detect_candlestick(df):
    """Add candlestick pattern features (binary + ratio)"""
    df = df.copy()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    body = abs(c - o)
    rng = (h - l).clip(lower=1e-10)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    df["Body_Ratio"] = body / rng
    df["Upper_Wick_Ratio"] = upper / rng
    df["Lower_Wick_Ratio"] = lower / rng
    df["Candle_Color"] = (c > o).astype(int)

    # Only compute pattern flags where range is meaningful (> $0.10)
    valid = (h - l) > 0.10

    df["Is_Doji"] = (valid & (body / rng < 0.1)).astype(int)
    df["Is_Hammer"] = (valid & (lower >= 2 * body) & (upper < body)).astype(int)
    df["Is_Shooting_Star"] = (valid & (upper >= 2 * body) & (lower < body)).astype(int)
    df["Is_Pin_Bar"] = (valid & (np.maximum(upper, lower) >= 3 * body) & (body > 0)).astype(int)
    df["Is_Marubozu"] = (valid & (upper / rng < 0.05) & (lower / rng < 0.05) & (body / rng > 0.8)).astype(int)

    prev_c = c.shift(1)
    prev_o = o.shift(1)
    df["Is_Bullish_Engulfing"] = (
        (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)
    ).astype(int)
    df["Is_Bearish_Engulfing"] = (
        (c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)
    ).astype(int)
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
    n = len(df)
    touch_l = np.zeros(n, dtype=int)
    touch_h = np.zeros(n, dtype=int)
    if n > lookback:
        l_wins = sliding_window_view(l.values[:n - 1], lookback)
        h_wins = sliding_window_view(h.values[:n - 1], lookback)
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
    """Detect double top/bottom pattern using vectorized numpy (O(n*k^2) worst case)."""
    df = df.copy()
    h_arr = df["High"].values.astype(float)
    l_arr = df["Low"].values.astype(float)
    n = len(df)

    double_top = np.zeros(n, dtype=int)
    double_bottom = np.zeros(n, dtype=int)

    for i in range(lookback * 2, n):
        window_h = h_arr[i - lookback:i]
        window_l = l_arr[i - lookback:i]
        k = len(window_h)

        # Find local peaks in window: h[j] > h[j-1] and h[j] > h[j+1]
        if k < 3:
            continue
        peak_mask = (
            (window_h[1:-1] > window_h[:-2]) &
            (window_h[1:-1] > window_h[2:])
        )
        peak_indices = np.where(peak_mask)[0] + 1  # offset by 1 due to slicing
        peak_vals = window_h[peak_indices]

        trough_mask = (
            (window_l[1:-1] < window_l[:-2]) &
            (window_l[1:-1] < window_l[2:])
        )
        trough_indices = np.where(trough_mask)[0] + 1
        trough_vals = window_l[trough_indices]

        curr_h = h_arr[i]
        curr_l = l_arr[i]

        # Double top: two peaks within tolerance
        if len(peak_indices) >= 2:
            for p1 in range(len(peak_indices)):
                for p2 in range(p1 + 1, len(peak_indices)):
                    v1, v2 = peak_vals[p1], peak_vals[p2]
                    if v1 <= 0:
                        continue
                    if abs(v1 - v2) / v1 >= tolerance:
                        continue
                    idx1, idx2 = peak_indices[p1], peak_indices[p2]
                    if idx1 >= idx2:
                        continue
                    mid_slice = window_l[idx1:idx2 + 1]
                    if len(mid_slice) == 0:
                        continue
                    mid_min = mid_slice.min()
                    drop_depth = (v1 - mid_min) / v1
                    if drop_depth > 0.01 and curr_h <= v1:
                        double_top[i] = 1
                        break
                if double_top[i]:
                    break

        # Double bottom: two troughs within tolerance
        if len(trough_indices) >= 2:
            for t1 in range(len(trough_indices)):
                for t2 in range(t1 + 1, len(trough_indices)):
                    v1, v2 = trough_vals[t1], trough_vals[t2]
                    if v1 <= 0:
                        continue
                    if abs(v1 - v2) / v1 >= tolerance:
                        continue
                    idx1, idx2 = trough_indices[t1], trough_indices[t2]
                    if idx1 >= idx2:
                        continue
                    mid_slice = window_h[idx1:idx2 + 1]
                    if len(mid_slice) == 0:
                        continue
                    mid_max = mid_slice.max()
                    rise = (mid_max - v1) / v1
                    if rise > 0.01 and curr_l >= v1:
                        double_bottom[i] = 1
                        break
                if double_bottom[i]:
                    break

    df["Is_Double_Top"] = double_top
    df["Is_Double_Bottom"] = double_bottom
    return df
