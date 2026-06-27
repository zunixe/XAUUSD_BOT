"""Data quality validation for XAUUSD data."""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_daily(df):
    """Check data quality. Returns list of issue strings."""
    issues = []

    # 1. Missing bars (gaps > 5 days for daily, accounting for weekends)
    if len(df) > 1:
        date_diffs = df.index.to_series().diff().dt.days
        gaps = date_diffs[date_diffs > 5]
        for date, gap in gaps.items():
            issues.append(f"DATA_GAP: {int(gap)} days gap ending {date.date()}")

    # 2. Duplicate entries
    dups = df.index[df.index.duplicated()]
    if len(dups) > 0:
        issues.append(f"DUPLICATES: {len(dups)} duplicate dates")

    # 3. Price anomalies (>8% move in 1 bar - gold rarely moves this much daily)
    if "Close" in df.columns:
        returns = df["Close"].pct_change().abs()
        spikes = returns[returns > 0.08]
        for date, ret in spikes.items():
            issues.append(f"PRICE_SPIKE: {ret:.1%} move on {date.date()}")

    # 4. Stale data (last bar > 3 days old, accounting for weekends)
    if len(df) > 0:
        last_date = df.index[-1]
        age_days = (pd.Timestamp.now() - last_date).days
        if age_days > 3:
            issues.append(f"STALE_DATA: last bar is {age_days} days old ({last_date.date()})")

    # 5. Zero/negative prices
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            bad = (df[col] <= 0).sum()
            if bad > 0:
                issues.append(f"INVALID_PRICE: {bad} zero/negative {col} values")

    # 6. OHLC consistency
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        bad_hl = (df["High"] < df["Low"]).sum()
        if bad_hl > 0:
            issues.append(f"OHLC_ERROR: {bad_hl} bars where High < Low")

    # 7. NaN check
    nan_counts = df[["Open", "High", "Low", "Close"]].isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            issues.append(f"NaN: {count} missing {col} values")

    return issues


def validate_and_report(csv_path):
    """Validate CSV and print report. Returns (df, issues)."""
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    issues = validate_daily(df)

    if issues:
        print(f"[DATA] {len(issues)} issues in {os.path.basename(csv_path)}:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"[DATA] {os.path.basename(csv_path)}: OK ({len(df)} rows)")

    return df, issues


def check_staleness(csv_path, max_age_days=3):
    """Check if data is stale. Returns (is_fresh, age_days)."""
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    if len(df) == 0:
        return False, 999
    last_date = df.index[-1]
    age = (pd.Timestamp.now() - last_date).days
    return age <= max_age_days, age


if __name__ == "__main__":
    for csv in ["xauusd_daily.csv", "xauusd_4h.csv"]:
        path = os.path.join(BASE_DIR, csv)
        if os.path.exists(path):
            validate_and_report(path)
