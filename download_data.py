"""
Download XAUUSD historical data dari berbagai sumber.
Pilih salah satu metode yang paling cocok.
"""
import pandas as pd

# ========== METODE 1: YAHOO FINANCE (GC=F futures) ==========
def from_yahoo(save_path="xauusd_daily.csv"):
    """
    pip install yfinance
    Data: Gold Futures (GC=F) dari Yahoo Finance
    Kelebihan: gratis, update otomatis
    """
    import yfinance as yf
    print("Downloading from Yahoo Finance (GC=F)...")
    df = yf.download("GC=F", start="2000-01-01", end="2026-06-23")
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.rename(columns={"Date": "Date", "Open": "Open", "High": "High",
                       "Low": "Low", "Close": "Close", "Volume": "Volume"}, inplace=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")
    return df

# ========== METODE 2: INVESTING.COM ==========
def from_investing(save_path="xauusd_daily.csv"):
    """
    pip install investing-com
    Data: XAU/USD spot langsung
    """
    from investing_com import Investing
    client = Investing()
    df = client.get_historical(
        pair_id="8830",  # XAU/USD pair ID
        period="MAX",
        return_df=True
    )
    df.to_csv(save_path)
    print(f"Saved to {save_path}")
    return df

# ========== METODE 3: KAGGLE DATASET ==========
def from_kaggle(save_path="xauusd_daily.csv"):
    """
    Download dataset dari Kaggle:
    https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024

    Manual: download dari Kaggle, simpan sebagai xauusd_daily.csv
    """
    print("Manual step:")
    print("1. Buka https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024")
    print("2. Download file XAU_daily_data.csv")
    print(f"3. Rename dan simpan sebagai {save_path}")

# ========== METODE 4: DUKASCOPY (via Node.js) ==========
def from_dukascopy(save_path="xauusd_daily.csv"):
    """
    Butuh Node.js:
    npm install -g dukascopy-node
    dukascopy-node -i xauusd -from 2000-01-01 -to 2026-06-23 -t d1 -f csv
    """
    print("Run di terminal:")
    print("npx dukascopy-node -i xauusd -from 2000-01-01 -to 2026-06-23 -t d1 -f csv")
    print(f"Kemudian rename output CSV ke {save_path}")

# ========== GENERATE SAMPLE DATA (jika belum ada data) ==========
def generate_sample_data(save_path="xauusd_daily.csv", n_days=1000):
    """
    Buat sample data untuk testing pipeline
    """
    np.random.seed(42)
    dates = pd.date_range(end="2026-06-23", periods=n_days, freq="D")
    price = 2000
    prices = []
    for _ in range(n_days):
        step = np.random.randn() * 15
        price += step
        prices.append(max(price, 1000))

    df = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": [p + abs(np.random.randn() * 10) for p in prices],
        "Low": [p - abs(np.random.randn() * 10) for p in prices],
        "Close": [p + np.random.randn() * 5 for p in prices],
        "Volume": np.random.randint(1000, 100000, n_days)
    })
    df.to_csv(save_path, index=False)
    print(f"Generated sample data: {save_path}")
    return df

if __name__ == "__main__":
    import numpy as np

    print("Pilih sumber data:")
    print("1. Yahoo Finance (GC=F) - RECOMMENDED")
    print("2. Generate Sample Data (testing)")
    pilihan = input("Pilihan (1/2): ").strip()

    if pilihan == "1":
        from_yahoo()
    else:
        generate_sample_data()
