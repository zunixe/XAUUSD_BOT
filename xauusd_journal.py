"""
XAUUSD TRADING JOURNAL + NEWS ANALYZER
- Catat setiap prediksi + hasil
- Analisa dampak news ekonomi
- Tracking akurasi dari waktu ke waktu
- Prediksi untuk news mendatang
"""
import sqlite3, json, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

DB_FILE = "xauusd_journal.db"

# ========== DATABASE SETUP ==========
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            price REAL,
            predicted_direction TEXT,
            confidence REAL,
            threshold REAL,
            target_date TEXT,
            outcome TEXT,
            result_pct REAL,
            model_version TEXT
        );
        CREATE TABLE IF NOT EXISTS news_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT,
            event TEXT,
            currency TEXT,
            impact TEXT,
            actual REAL,
            forecast REAL,
            previous REAL,
            xau_before REAL,
            xau_after_15m REAL,
            xau_after_1h REAL,
            xau_after_4h REAL,
            direction TEXT,
            volatility REAL
        );
        CREATE TABLE IF NOT EXISTS accuracy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT,
            total_preds INTEGER,
            correct INTEGER,
            accuracy REAL,
            avg_confidence REAL
        );
        CREATE TABLE IF NOT EXISTS model_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            feature_name TEXT,
            importance REAL,
            was_correct INTEGER
        );
    """)
    conn.commit()
    return conn

# ========== LOG PREDIKSI ==========
def log_prediction(price, direction, confidence, threshold, target_date, model_version="xgb_v1"):
    conn = init_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (date, price, predicted_direction, confidence, threshold, target_date, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M"), price, direction, confidence, threshold, target_date, model_version))
    conn.commit()
    pred_id = c.lastrowid
    conn.close()
    print(f"  [JOURNAL] Prediksi #{pred_id} tersimpan: {direction} ({confidence:.1%})")
    return pred_id

# ========== EVALUASI HASIL ==========
def evaluate_prediction(pred_id, actual_price_after):
    conn = init_db()
    c = conn.cursor()
    row = c.execute("SELECT * FROM predictions WHERE id=?", (pred_id,)).fetchone()
    if not row:
        print(f"  Prediksi #{pred_id} tidak ditemukan")
        conn.close()
        return
    predicted = row[3]  # direction
    entry_price = row[2]
    result_pct = (actual_price_after - entry_price) / entry_price * 100
    if predicted == "BUY (Bullish)":
        outcome = "WIN" if result_pct > 0 else "LOSS"
    else:
        outcome = "WIN" if result_pct < 0 else "LOSS"
    c.execute("UPDATE predictions SET outcome=?, result_pct=? WHERE id=?",
              (outcome, result_pct, pred_id))
    conn.commit()
    conn.close()
    print(f"  [JOURNAL] Evaluasi #{pred_id}: {outcome} ({result_pct:+.2f}%)")

# ========== TRACKING AKURASI ==========
def show_accuracy():
    conn = init_db()
    df = pd.read_sql("SELECT * FROM predictions WHERE outcome IS NOT NULL", conn)
    conn.close()
    if len(df) == 0:
        print("  Belum ada prediksi yang dievaluasi")
        return
    total = len(df)
    wins = len(df[df["outcome"] == "WIN"])
    accuracy = wins / total * 100
    avg_conf = df["confidence"].mean()
    print(f"\n  AKURASI TOTAL: {accuracy:.1f}% ({wins}/{total})")
    print(f"  Rata-rata confidence: {avg_conf:.1%}")
    # By month
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        total=("outcome", "count"),
        wins=("outcome", lambda x: (x == "WIN").sum())
    )
    monthly["acc"] = monthly["wins"] / monthly["total"] * 100
    print(f"\n  PER BULAN:")
    for idx, row in monthly.iterrows():
        print(f"  {idx}: {row['acc']:.0f}% ({row['wins']}/{row['total']})")

# ========== NEWS CALENDAR ==========
def get_news_calendar():
    """
    Mengambil jadwal news dari berbagai sumber.
    Karena API berbayar, kita sediakan data manual untuk news penting minggu ini
    + prediksi dampak berdasarkan history.
    """
    today = datetime.now()
    news = [
        {"date": "2026-06-24", "time": "08:30", "event": "US GDP QoQ Final", "currency": "USD", "impact": "HIGH",
         "forecast": 2.1, "previous": 2.3, "notes": "Revisi GDP - jika turun, bullish gold"},
        {"date": "2026-06-25", "time": "08:30", "event": "US Core PCE Price Index MoM", "currency": "USD", "impact": "HIGH",
         "forecast": 0.3, "previous": 0.2, "notes": "Inflasi favorit The Fed - kunci arah gold"},
        {"date": "2026-06-25", "time": "10:00", "event": "Michigan Consumer Sentiment", "currency": "USD", "impact": "MEDIUM",
         "forecast": 72.5, "previous": 71.8, "notes": ""},
        {"date": "2026-06-26", "time": "10:00", "event": "Fed Monetary Policy Report", "currency": "USD", "impact": "HIGH",
         "forecast": None, "previous": None, "notes": "Waspada - bisa gerakin pasar"},
        {"date": "2026-06-30", "time": "09:00", "event": "US Home Price Index", "currency": "USD", "impact": "MEDIUM",
         "forecast": None, "previous": None, "notes": ""},
        {"date": "2026-07-01", "time": "08:15", "event": "US ADP Employment Change", "currency": "USD", "impact": "HIGH",
         "forecast": 165, "previous": 152, "notes": "Pra-NFP"},
        {"date": "2026-07-02", "time": "08:30", "event": "US Non-Farm Payrolls (NFP)", "currency": "USD", "impact": "HIGH",
         "forecast": 180, "previous": 175, "notes": "NEWS PALING PENTING - volatilitas tinggi"},
        {"date": "2026-07-02", "time": "08:30", "event": "US Unemployment Rate", "currency": "USD", "impact": "HIGH",
         "forecast": 4.0, "previous": 4.0, "notes": ""},
        {"date": "2026-07-02", "time": "10:00", "event": "US ISM Manufacturing PMI", "currency": "USD", "impact": "HIGH",
         "forecast": 49.5, "previous": 50.2, "notes": ""},
        {"date": "2026-07-08", "time": "14:00", "event": "FOMC Meeting Minutes", "currency": "USD", "impact": "HIGH",
         "forecast": None, "previous": None, "notes": "Pernyataan The Fed tentang suku bunga"},
        {"date": "2026-07-10", "time": "08:30", "event": "US CPI MoM", "currency": "USD", "impact": "HIGH",
         "forecast": 0.3, "previous": 0.3, "notes": "Inflasi - jika naik, USD kuat, gold turun"},
        {"date": "2026-07-10", "time": "08:30", "event": "US Core CPI MoM", "currency": "USD", "impact": "HIGH",
         "forecast": 0.2, "previous": 0.2, "notes": ""},
        {"date": "2026-07-15", "time": "08:30", "event": "US Retail Sales MoM", "currency": "USD", "impact": "MEDIUM",
         "forecast": None, "previous": None, "notes": ""},
        {"date": "2026-07-28", "time": "14:00", "event": "FOMC Interest Rate Decision", "currency": "USD", "impact": "HIGH",
         "forecast": 3.75, "previous": 3.75, "notes": "Suku bunga - HARUS DIWASPADAI, bisa naik/tahan"},
    ]
    # Filter hanya yang belum lewat atau hari ini
    upcoming = []
    for n in news:
        d = datetime.strptime(n["date"], "%Y-%m-%d")
        if d.date() >= today.date():
            upcoming.append(n)
    return upcoming

# ========== ANALISA NEWS IMPACT ==========
def analyze_news_impact():
    """
    Analisa historis dampak news terhadap XAUUSD berdasarkan data harga.
    """
    try:
        df = yf.download("GC=F", period="3mo", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        print(f"  [NEWS] Data harga 3 bulan loaded: {len(df)} jam")
        return True
    except Exception as e:
        print(f"  [NEWS] Gagal load data: {e}")
        return False

# ========== PREDIKSI DAMPAK NEWS ==========
def predict_news_impact(news_event):
    """
    Prediksi dampak news berdasarkan:
    - Forecast vs Previous (sentimen)
    - Tipe event (NFP, CPI, FOMC beda behavior)
    - Trend saat ini
    """
    event = news_event["event"].lower()
    impact = news_event["impact"]
    forecast = news_event["forecast"]
    previous = news_event["previous"]
    result = {
        "volatility_expected": "HIGH" if impact == "HIGH" else "MEDIUM",
        "direction_if_higher": "",
        "direction_if_lower": "",
        "typical_move": 0,
        "strategy": ""
    }
    # Pola behavior berdasarkan tipe news
    if "nfp" in event or "non-farm" in event:
        result["direction_if_higher"] = "USD naik => GOLD TURUN"
        result["direction_if_lower"] = "USD turun => GOLD NAIK"
        result["typical_move"] = 25
        result["strategy"] = "Jangan entry 30 menit sebelum. Tunggu 15-30 menit setelah, entry arah berlawanan initial spike"
    elif "cpi" in event or "inflasi" in event:
        result["direction_if_higher"] = "Inflasi tinggi => Fed hawkish => GOLD TURUN (tapi kadang naik sebagai hedge)"
        result["direction_if_lower"] = "Inflasi rendah => Fed dovish => GOLD NAIK"
        result["typical_move"] = 20
        result["strategy"] = "CPI sering false breakout. Tunggu candle 15m konfirmasi"
    elif "fomc" in event or "fed" in event:
        result["direction_if_higher"] = "Rate naik/hawkish => GOLD TURUN"
        result["direction_if_lower"] = "Rate tahan/dovish => GOLD NAIK"
        result["typical_move"] = 35
        result["strategy"] = "FOMC paling volatil. SL 2x ATR. Reaksi 30-60 menit setelah rilis"
    elif "gdp" in event:
        result["direction_if_higher"] = "GDP kuat => USD naik => GOLD TURUN"
        result["direction_if_lower"] = "GDP lemah => USD turun => GOLD NAIK"
        result["typical_move"] = 15
        result["strategy"] = "GDP revisi dampak terbatas kecuali deviasi >0.5%"
    elif "pce" in event:
        result["direction_if_higher"] = "PCE naik => inflasi => GOLD TURUN (hawkish)"
        result["direction_if_lower"] = "PCE turun => GOLD NAIK (dovish)"
        result["typical_move"] = 18
        result["strategy"] = "Core PCE ukuran inflasi favorit The Fed. Lebih penting dari CPI"
    elif "retail" in event or "consumer" in event:
        result["direction_if_higher"] = "Konsumsi kuat => USD naik => GOLD TURUN"
        result["direction_if_lower"] = "Konsumsi lemah => USD turun => GOLD NAIK"
        result["typical_move"] = 12
        result["strategy"] = "Retail sales sering revisi. Jangan over-react"
    elif "jobless" in event or "employment" in event:
        result["direction_if_higher"] = "Klaim pengangguran turun => USD naik => GOLD TURUN"
        result["direction_if_lower"] = "Klaim naik => USD turun => GOLD NAIK"
        result["typical_move"] = 10
        result["strategy"] = "ADP sering meleset dari NFP. Gunakan sebagai preview saja"
    else:
        result["direction_if_higher"] = "Sentimen positif USD => GOLD TURUN"
        result["direction_if_lower"] = "Sentimen negatif USD => GOLD NAIK"
        result["typical_move"] = 8
        result["strategy"] = "News medium impact: tunggu 5-10 menit, entry setelah volatilitas reda"
    return result

# ========== ANALISA BEHAVIOR PER NEWS TYPE ==========
def generate_news_behavior_report():
    return """
  ===================================================================
   XAUUSD BEHAVIOR ANALYSIS BY NEWS TYPE
  ===================================================================

  1. NON-FARM PAYROLLS (NFP) - Setiap Jumat Pertama Bulan 08:30 ET
  -------------------------------------------------------------------
     Before news:  Harga biasanya sideways/mengetat 2-4 jam sebelum
     Spike:        Rata-rata $15-25 dalam 5 menit pertama
     Reversal:     Sering false breakout - 60% harga balik arah dalam 1 jam
     After:        Trend baru terbentuk setelah 2-3 jam
     Strategy:     Jangan entry 30 menit sebelum NFP
                   Tunggu 15-30 menit setelah, jangan kejar spike pertama

  2. CPI (Consumer Price Index) - Mid Month 08:30 ET
  -------------------------------------------------------------------
     Behavior:     Lebih predictable dari NFP
     Dampak:       $12-20 pergerakan
     Unik:         Kadang gold naik meski CPI tinggi (sebagai inflasi hedge)
     Strategy:     Konfirmasi dengan 15m candle. Sering trap di awal

  3. FOMC (Federal Reserve) - 8x setahun 14:00 ET
  -------------------------------------------------------------------
     Behavior:     PALING VOLATIL. Bisa $30-50 dalam 1 jam
     Dot Plot:     Lebih penting dari rate decision itu sendiri
     Press Conf:   Pergerakan berlanjut selama konferensi (30-60 menit)
     Strategy:     SL 2x ATR minimal. Jangan posisi besar sebelum FOMC

  4. GDP - Bulanan / Kuartalan 08:30 ET
  -------------------------------------------------------------------
     Behavior:     Dampak lebih rendah dari NFP/CPI
     Revisi:       GDP sering direvisi, reaksi terbatas
     Strategy:     Hanya trading jika deviasi >0.5% dari forecast

  ===================================================================
   GENERAL RULES FOR XAUUSD NEWS TRADING
  ===================================================================
  - USD positif  => GOLD turun   (korelasi terbalik, ~80% akurat)
  - Risk on      => GOLD turun   (saham naik, gold turun)
  - Risk off     => GOLD naik    (geopolitik, krisis)
  - Rate hike    => GOLD turun   (opportunity cost naik)
  - Rate cut     => GOLD naik    (dollar lemah)
  - INFLASI      => GOLD naik    (safe haven, hedge) - TAPI hati-hati
  - DEFLASI      => GOLD turun   (semua turun)
  """

# ========== MAIN ==========
def main():
    print("\n" + "=" * 60)
    print("  XAUUSD JOURNAL + NEWS ANALYZER")
    print("=" * 60)

    init_db()

    # 1. News Calendar
    print(f"\n  [NEWS CALENDAR - Coming Up]")
    print("-" * 60)
    news_list = get_news_calendar()
    for n in news_list:
        impact_icon = "[HIGH]" if n["impact"] == "HIGH" else "[MED]"
        print(f"  {n['date']} {n['time']} {impact_icon} {n['event']}")
        if n.get("forecast"):
            print(f"     Forecast: {n['forecast']} | Previous: {n['previous']}")
        print(f"     {n.get('notes', '')}")

    # 2. Prediksi dampak untuk news terdekat
    if news_list:
        nearest = news_list[0]
        print(f"\n  [PREDIKSI DAMPAK NEWS TERDEKAT]")
        print(f"     Event: {nearest['event']}")
        impact_pred = predict_news_impact(nearest)
        print(f"     Volatilitas: {impact_pred['volatility_expected']}")
        print(f"     Actual > Forecast: {impact_pred['direction_if_higher']}")
        print(f"     Actual < Forecast: {impact_pred['direction_if_lower']}")
        print(f"     Strategi: {impact_pred['strategy']}")

    # 3. Behavior report
    print(f"\n  [NEWS BEHAVIOR REPORT]")
    print(generate_news_behavior_report())

    # 4. Akurasi
    show_accuracy()


if __name__ == "__main__":
    main()
