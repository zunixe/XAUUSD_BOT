"""COT (Commitments of Traders) data from CFTC for gold futures."""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COT_CSV = os.path.join(BASE_DIR, "cot_weekly.csv")

# CFTC Traders in Financial Futures - Gold (contract: 088691)
COT_URL = "https://www.cftc.gov/dea/futures/financial_lf.htm"


def download_cot():
    """Download latest COT data from CFTC website."""
    try:
        tables = pd.read_html(COT_URL, match="GOLD")
        if not tables:
            print("[COT] No gold table found")
            return False
        df = tables[0]
        # Parse columns - CFTC format varies, try common layouts
        df.columns = [str(c).strip() for c in df.columns]

        # Try to find relevant columns
        date_col = None
        long_col = None
        short_col = None
        oi_col = None

        for c in df.columns:
            cl = c.lower()
            if "date" in cl or "as of" in cl:
                date_col = c
            elif "long" in cl and "non" in cl and "comm" in cl:
                long_col = c
            elif "short" in cl and "non" in cl and "comm" in cl:
                short_col = c
            elif "open" in cl and "interest" in cl:
                oi_col = c

        if not all([date_col, long_col, short_col]):
            print("[COT] Could not parse CFTC columns")
            return False

        result = pd.DataFrame()
        result["Date"] = pd.to_datetime(df[date_col])
        result["NonComm_Long"] = pd.to_numeric(df[long_col], errors="coerce")
        result["NonComm_Short"] = pd.to_numeric(df[short_col], errors="coerce")
        if oi_col:
            result["Open_Interest"] = pd.to_numeric(df[oi_col], errors="coerce")
        result.dropna(subset=["NonComm_Long", "NonComm_Short"], inplace=True)
        result.sort_values("Date", inplace=True)
        result.set_index("Date", inplace=True)

        # Calculate derived features
        result["Spec_Net_Long"] = result["NonComm_Long"] - result["NonComm_Short"]
        if "Open_Interest" in result.columns:
            result["Spec_Net_Long_Pct"] = result["Spec_Net_Long"] / result["Open_Interest"] * 100
            result["Commercial_Net_Short_Pct"] = -result["Spec_Net_Long_Pct"]  # inverse
        else:
            result["Spec_Net_Long_Pct"] = result["Spec_Net_Long"]
            result["Commercial_Net_Short_Pct"] = -result["Spec_Net_Long"]

        result["Spec_Positioning_Change"] = result["Spec_Net_Long_Pct"].diff()

        # Save
        result[["Spec_Net_Long_Pct", "Spec_Positioning_Change", "Commercial_Net_Short_Pct"]].to_csv(COT_CSV)
        print(f"[COT] Saved {len(result)} weeks to {COT_CSV}")
        return True
    except Exception as e:
        print(f"[COT] Download error: {e}")
        return False


def load_cot_features():
    """Load COT features for merging into main dataframe."""
    if not os.path.exists(COT_CSV):
        return None
    try:
        df = pd.read_csv(COT_CSV, parse_dates=["Date"], index_col="Date")
        return df
    except Exception:
        return None
