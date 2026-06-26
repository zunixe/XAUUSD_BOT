# XAUUSD AI Trading System

Sistem prediksi XAUUSD (Gold) berbasis Machine Learning dengan continuous learning, paper trading, dan Telegram bot.

---

## Daftar Isi

1. [Arsitektur Sistem](#1-arsitektur-sistem)
2. [File Structure](#2-file-structure)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Labeling & Target](#5-labeling--target)
6. [Training & Validasi](#6-training--validasi)
7. [Signal Generation (Trigger)](#7-signal-generation-trigger)
8. [Real-Time Price & Evaluation](#8-real-time-price--evaluation)
9. [Paper Trading Simulation](#9-paper-trading-simulation)
10. [Telegram Bot](#10-telegram-bot)
11. [Continuous Learning (Feedback Loop)](#11-continuous-learning-feedback-loop)
12. [Cara Menjalankan](#12-cara-menjalankan)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    XAUUSD AI Trading System                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Data    │───▶│  Feature     │───▶│  ML Model        │   │
│  │  Source  │    │  Engineering │    │  (XGBoost)       │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│       │                                      │               │
│       ▼                                      ▼               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Kitco   │    │  Patterns    │    │  Ensemble        │   │
│  │  yfinance│    │  (S/R,       │    │  (3-4 fold       │   │
│  │  Yahoo   │    │   candle)    │    │   models)        │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                                               │               │
│                                               ▼               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Telegram │◀───│  Daemon      │◀───│  Signal          │   │
│  │  Bot     │    │  (poll.py)   │    │  Generator       │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│       │                 │                       │             │
│       ▼                 ▼                       ▼             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  User    │    │  Real-time   │    │  Paper Trading   │   │
│  │  Chat    │    │  Evaluation  │    │  Simulation      │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Alur Data

1. **Data Collection**: `update_data.py` download data OHLCV harian + 4 jam dari yfinance, plus data makro (DXY, VIX, SPY, US10Y, OIL)
2. **Feature Engineering**: `trading.py` generate 60+ fitur teknikal + makro
3. **Training**: `auto_runner.py` (daily) / `runner_4h.py` (4H) train model XGBoost
4. **Prediction**: Model menghasilkan probabilitas BUY/SELL
5. **Signal**: Jika confidence melewati threshold, sinyal dikirim ke Telegram
6. **Evaluation**: Harga real-time dicek terhadap SL/TP, outcome dicatat
7. **Feedback**: Outcome real digunakan untuk retrain model (continuous learning)

---

## 2. File Structure

| File | Fungsi |
|------|--------|
| `trading.py` | **Core library** — feature engineering, ATR labeling, Optuna tuning, walk-forward CV, centralized paths |
| `auto_runner.py` | **Daily runner** — retrain model daily, prediksi, evaluasi historis, show stats |
| `runner_4h.py` | **4H runner** — retrain model 4 jam, prediksi, evaluasi OHLC |
| `poll_commands.py` | **Daemon** — polling Telegram (5s) + evaluasi real-time (60s) + deteksi candle baru |
| `telegram_notifier.py` | **Telegram bot** — semua command handling, notifikasi, price fetching |
| `simulation.py` | **Paper trading** — virtual account, lot sizing, P&L tracking |
| `patterns.py` | **Pattern detection** — candlestick patterns, S/R, double top/bottom |
| `update_data.py` | **Data downloader** — download OHLCV + makro data |
| `xauusd_journal.db` | **Database SQLite** — predictions, simulation, outcomes |
| `xauusd_model.pkl` | **Model daily** — XGBoost ensemble + scaler + threshold |
| `xauusd_model_4h.pkl` | **Model 4H** — XGBoost ensemble + scaler + threshold |

---

## 3. Machine Learning Pipeline

### Model: XGBoost Ensemble

Sistem menggunakan **ensemble dari beberapa fold models** (bukan single model):

- **Daily**: 4 fold models, bobot berdasarkan F1 score
- **4H**: 3 fold models, bobot berdasarkan F1 score

Prediksi = weighted average dari semua fold models:
```
prob = Σ (weight_i × model_i.predict_proba(X))
```

### Hyperparameter Tuning: Optuna

Setiap fold di-tune menggunakan **Optuna** dengan:
- **Daily**: 50 trials per fold
- **4H**: 30 trials per fold
- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Pruner**: MedianPruner (stop trial jelek lebih awal)
- **Metric**: F1 score pada validation set

Parameter yang di-tune:
| Parameter | Range |
|-----------|-------|
| `n_estimators` | 200-600 |
| `max_depth` | 4-10 |
| `learning_rate` | 0.005-0.05 (log scale) |
| `subsample` | 0.6-0.95 |
| `colsample_bytree` | 0.5-0.9 |
| `min_child_weight` | 1-5 |
| `reg_alpha` | 0.0-1.0 |
| `reg_lambda` | 0.5-5.0 |

### Walk-Forward CV (Purged)

Validasi menggunakan **purged expanding window walk-forward**:

```
Fold 1: [====train====]---embargo---[==val==]
Fold 2: [========train========]---embargo---[==val==]
Fold 3: [============train============]---embargo---[==val==]
OOT:    [================================]---[=holdout=]
```

- **Embargo**: 3 bar (daily) / 8 bar (4H) — mencegah label leakage
- **OOT**: 15% data terakhir, tidak pernah disentuh saat training
- **n_splits**: 4 (daily) / 3 (4H)

### Sample Weight

- **ATR labels**: weight = 1.0
- **Real outcomes** (dari SL/TP aktual): weight = 2.0
- Real outcomes hanya di-override jika ≥ 5% dari total data dan ≥ 5 samples

---

## 4. Feature Engineering

### Price Features
| Feature | Deskripsi |
|---------|-----------|
| `EMA_5/10/20/50/100/200` | Exponential Moving Average berbagai periode |
| `RSI_14` | Relative Strength Index (14 bar) |
| `MACD/Signal/Histogram` | MACD (12,26,9) |
| `BB_Upper/Lower/Width/PctB` | Bollinger Bands (20,2) |
| `ATR_14` | Average True Range (14 bar) |
| `ATR_Pct` | ATR sebagai % harga |
| `Return_1/5/20` | Return 1/5/20 bar |
| `Vol_20/50` | Rolling volatility |
| `Dist_High/Low_10/20/50` | Jarak ke high/low tertinggi/terendah |

### Volume Features
| Feature | Deskripsi |
|---------|-----------|
| `Volume_Change` | Perubahan volume |
| `Volume_SMA_20` | Volume SMA 20 |
| `Volume_Ratio` | Volume / Volume_SMA_20 |

### Calendar Features
| Feature | Deskripsi |
|---------|-----------|
| `DayOfWeek` | Hari dalam minggu (0-4) |
| `Month` | Bulan (1-12) |
| `Hour` | Jam (khusus 4H) |

### Macro Features (Makroekonomi)
| Feature | Deskripsi |
|---------|-----------|
| `DXY_Return_1d/5d` | Dollar Index return |
| `VIX_Return_1d/5d` | VIX (fear index) return |
| `SPY_Return_1d/5d` | S&P 500 return |
| `US10Y_Return_1d/5d` | US 10-Year Treasury yield |
| `OIL_Return_1d/5d` | Crude Oil return |
| `DXY_EMA20_Dist` | Jarak DXY ke EMA20 |
| `Gold_DXY_Corr_20` | Korelasi Gold-DXY 20 bar |
| `Gold_VIX_Corr_20` | Korelasi Gold-VIX 20 bar |

### FOMC Features
| Feature | Deskripsi |
|---------|-----------|
| `Is_FOMC_Day` | Apakah hari ini FOMC |
| `Days_To_FOMC` | Jumlah hari menuju FOMC berikutnya |
| `Week_Before_FOMC` | 7 hari sebelum FOMC |
| `Week_After_FOMC` | 7 hari setelah FOMC |

### Pattern Features (dari `patterns.py`)
| Feature | Deskripsi |
|---------|-----------|
| `Body_Ratio` | Rasio body terhadap range candle |
| `Upper/Lower_Wick_Ratio` | Rasio wick atas/bawah |
| `Is_Doji/Hammer/Shooting_Star` | Pola candlestick |
| `Is_Pin_Bar/Marubozu` | Pola candlestick lanjutan |
| `Is_Bullish/Bearish_Engulfing` | Pola engulfing |
| `Is_Inside_Bar` | Inside bar |
| `Dist_Resistance/Support_Pct` | Jarak ke resistance/support |
| `Near_Resistance/Support` | Dekat resistance/support (< 0.5%) |
| `Breakout_High/Low` | Breakout dari high/low |
| `Touch_Count_High/Low` | Jumlah sentuhan di level |
| `Is_Double_Top/Bottom` | Pola double top/bottom |

---

## 5. Labeling & Target

### ATR Triple-Barrier Labeling

Target (label) dihitung menggunakan **ATR-based triple-barrier method**:

```python
entry = Close[i]
atr = TR.rolling(14).mean().shift(1)[i]  # shifted, no future leak
tp = entry + atr * atr_mult              # take profit level
sl = entry - atr * sl_mult               # stop loss level

# Forward scan (3 bar ke depan)
for bar in future_bars:
    if bar.High >= tp → Target = 1 (BUY win)
    if bar.Low <= sl  → Target = 0 (SELL win)
    # Jika keduanya kena → cek mana yang kena duluan
    # Jika tidak kena → Target berdasarkan close terakhir
```

Parameter:
| Parameter | Daily | 4H |
|-----------|-------|-----|
| `forward` | 3 bar (3 hari) | 8 bar (32 jam) |
| `atr_mult` (TP) | 0.8 | 0.6 |
| `sl_mult` (SL) | 0.6 | 0.4 |

### Real Outcome Override

Jika ada data real outcome (dari evaluasi SL/TP aktual), label ATR di-override:
- **WIN** → Target = 1 (jika BUY) atau 0 (jika SELL)
- **LOSS** → Target = 0 (jika BUY) atau 1 (jika SELL)
- **Weight**: 2x (lebih dipercaya daripada ATR label)
- **Syarat**: minimal 5 samples DAN ≥ 5% dari total data

---

## 6. Training & Validasi

### Alur Training (Daily)

```
1. Load data daily (xauusd_daily.csv)
2. Hitung ATR target (triple-barrier)
3. Override dengan real outcomes (jika ada)
4. Feature engineering (60+ fitur)
5. Walk-forward split (4 folds + OOT)
6. Untuk setiap fold:
   a. Scale dengan RobustScaler
   b. Tune hyperparameter dengan Optuna (50 trials)
   c. Hitung F1-optimal threshold dari PR curve
   d. Simpan model dan threshold
7. Ensemble: weighted average dari semua fold models
8. OOT evaluation (never touched during training)
9. Feedback loop: adjust threshold berdasarkan historis
10. Simpan model ke xauusd_model.pkl
```

### Alur Training (4H)

Sama dengan daily, tapi:
- 3 folds (bukan 4)
- 30 Optuna trials (bukan 50)
- Embargo = 8 bar (bukan 3)
- Data dari xauusd_4h.csv

---

## 7. Signal Generation (Trigger)

### Kondisi Sinyal

Sinyal dikirim ke Telegram jika **semua kondisi** terpenuhi:

#### Daily
| Kondisi | Threshold |
|---------|-----------|
| **BUY** | `prob ≥ max(0.55, best_thresh)` |
| **SELL** | `prob ≤ 1 - max(0.55, best_thresh)` |
| **NO_TRADE** | Di antara BUY dan SELL |

#### 4H
| Kondisi | Threshold |
|---------|-----------|
| **BUY** | `prob ≥ max(0.55, best_thresh)` |
| **SELL** | `prob ≤ 1 - max(0.55, best_thresh)` |
| **NO_TRADE** | Di antara BUY dan SELL |

### Contoh

| Confidence | Threshold 0.55 | Hasil |
|------------|---------------|-------|
| 0.72 | ≥ 0.55 | **BUY signal** ✅ |
| 0.30 | ≤ 0.45 | **SELL signal** ✅ |
| 0.50 | 0.45-0.55 | **NO_TRADE** ❌ (tidak kirim) |

### SL/TP Calculation

SL/TP dihitung dari ATR dan level teknikal:

**BUY:**
```python
sl  = max(low_20, entry - atr * 1.2)
tp1 = max(entry + atr * 0.8, entry + atr * 0.5)
tp2 = max(entry + atr * 1.8, ema50)
```

**SELL:**
```python
sl  = min(high_20, entry + atr * 1.2)
tp1 = min(entry - atr * 0.8, entry - atr * 0.5)
tp2 = min(entry - atr * 1.8, low_20)
```

---

## 8. Real-Time Price & Evaluation

### Harga Real-Time

Harga diambil dari **Kitco** (web scraping) dengan fallback ke **yfinance**:
- Multi-pattern regex untuk menangkap berbagai format harga
- 2 retry attempts jika gagal
- Fallback ke GC=F (Gold Futures) 1-minute data

### Evaluasi Real-Time

Daemon mengecek SL/TP setiap **60 detik** (normal) atau **5 detik** (saat harga dekat level):

```python
# Cek apakah harga dekat SL/TP (dalam 0.5%)
def _is_near_sltp():
    for setiap prediksi aktif:
        untuk setiap level (sl, tp1, tp2):
            if abs(harga - level) / harga < 0.005:
                return True  # → check setiap 5 detik
    return False  # → check setiap 60 detik
```

Urutan pengecekan (prioritas):
1. **SL** → Jika harga menyentuh SL, catat LOSS
2. **TP2** → Jika harga menyentuh TP2, catat WIN (TP2_HIT)
3. **TP1** → Jika harga menyentuh TP1, catat WIN (TP1_HIT)

### Evaluasi OHLC (Batch)

Untuk prediksi yang belum dievaluasi real-time, dilakukan evaluasi OHLC batch:
- Scan bar forward dari entry point
- First-touch logic: yang kena duluan yang dihitung
- Jika tidak kena dalam 5 bar → EXPIRED (lihat close terakhir)

---

## 9. Paper Trading Simulation

### Virtual Account

- **Starting balance**: $100 (default, bisa diubah via `/start`)
- **Risk per trade**: 1% dari balance
- **Max risk cap**: 2% dari balance (safety)

### Lot Sizing

```python
lot = (balance * 0.01) / (sl_distance * 100)
```

- `XAU_USD_PER_MOVE = 100` → 1 lot = 100 oz, $1 move = $100 P&L
- **Minimum lot**: 0.01
- **Jika lot < 0.01** atau risk > 2% → tidak trade (return None)

### P&L Calculation

```python
# BUY: exit > entry = profit, exit < entry = loss
price_diff = exit_price - effective_entry

# SELL: entry > exit = profit, entry < exit = loss
price_diff = effective_entry - exit_price

pnl = lot * price_diff * 100
```

### Spread Adjustment

- Spread XAUUSD: $0.30 (default)
- BUY entry = price + $0.30
- SELL entry = price - $0.30

### Outcome Handling

| Outcome | Exit Price |
|---------|-----------|
| WIN / TP1_HIT | TP1 |
| TP2_HIT | TP2 |
| LOSS / SL_HIT | SL |
| EXPIRED | Close price terakhir |
| NO_TRADE | Tidak ada trade |

---

## 10. Telegram Bot

### Commands

| Command | Fungsi |
|---------|--------|
| `/start` atau `/start 100` | Mulai paper trading dengan balance awal |
| `/bal` | Lihat balance dan statistik |
| `/reset` | Reset simulasi |
| `/info` | Info prediksi terakhir |
| `/retrain` | Trigger retrain manual (async) |
| `/stats` | Lihat learning statistics |

### Signal Notification

Format sinyal yang dikirim ke Telegram:

```
🔔 XAUUSD SINYAL BUY

Harga: $4,060.50
Confidence: 72.3%
Threshold: 55.0%
Target: 2026-06-29

SL: $4,045.20
TP1: $4,078.80
TP2: $4,095.50
```

### Outcome Notification

Setelah evaluasi:

```
✅ WIN (TP1_HIT)
Prediksi: BUY @ $4,060.50
Exit: $4,078.80
P&L: +$18.30 (+0.45%)
Balance: $118.30
```

---

## 11. Continuous Learning (Feedback Loop)

### Mekanisme

```
Prediksi → Evaluasi → Simpan Outcome → Retrain → Model Baru
    ↑                                                    │
    └────────────────────────────────────────────────────┘
```

### Kapan Retrain?

- **Daily**: otomatis saat candle baru terdeteksi
- **4H**: otomatis saat candle baru terdeteksi
- **Manual**: via `/retrain` command
- **Syarat**: ≥ 10 evaluasi baru sejak retrain terakhir

### Feedback Loop (Threshold Adjustment)

Threshold diadjust berdasarkan performa historis dengan **time decay**:

```python
# Outcomes 30 hari terakhir = 2x weight
weighted_acc = (recent_wins * 2 + older_wins) / (recent_count * 2 + older_count)

if weighted_acc < 0.45:    threshold += 0.10  (max 0.75)  # sangat buruk
elif weighted_acc < 0.55:  threshold += 0.05  (max 0.70)  # buruk
elif weighted_acc >= 0.65: threshold -= 0.03  (min 0.55)  # bagus → turunkan
# 0.55-0.64: tidak ada perubahan (stabil)
```

Sifat:
- **Konservatif**: naik agresif (+0.05/+0.10), turun pelan (-0.03)
- **Time decay**: outcome terbaru lebih berpengaruh
- **Bisa turun**: threshold tidak stuck di 0.75 selamanya

---

## 12. Cara Menjalankan

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost optuna joblib yfinance
```

### Start Daemon (Background)

Daemon otomatis berjalan saat Windows login via VBS:
```
C:\Users\zaini\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\XAUUSD_Daemon.vbs
```

Atau manual:
```bash
python poll_commands.py
```

### Manual Commands

```bash
# Download/update data
python update_data.py

# Train model daily
python auto_runner.py

# Train model 4H
python runner_4h.py --run

# Predict (daily)
python auto_runner.py --predict

# Show learning stats
python auto_runner.py --stats
```

### Logs

- `poll.log` — daemon log
- `auto_runner.log` — daily training log
- `runner_4h.log` — 4H training log

---

## 13. Troubleshooting

### Daemon tidak jalan

```powershell
# Cek apakah daemon berjalan
Get-Process python | Where-Object {$_.CommandLine -like "*poll_commands*"}

# Restart daemon
Stop-Process -Name python -Force
Start-Process "C:\Users\zaini\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -ArgumentList "poll_commands.py" -WorkingDirectory "C:\Users\zaini\OneDrive\Documents\Project\Trading XAUUSD"
```

### Model tidak ada / error load

```bash
# Retrain dari awal
python auto_runner.py
python runner_4h.py --run
```

### Telegram tidak merespons

1. Cek `poll.log` untuk error
2. Pastikan token bot benar di `.env`
3. Pastikan chat_id benar

### Database locked

```bash
# Restart daemon (release connection)
# Atau hapus file lock:
del xauusd_journal.db-journal
```

---

## Catatan Teknis

### SSL Issue

Karena Hermes venv tidak punya CA certificates, SSL verification dimatikan:
```python
ssl._create_default_https_context = ssl._create_unverified_context
```

Ini diperlukan untuk Telegram API dan yfinance. Tidak berpengaruh pada keamanan lokal.

### Model File Compatibility

Model file (.pkl) berisi:
- `ensemble_models`: list of (model, weight) tuples
- `scaler`: RobustScaler yang sudah fit
- `feature_cols`: nama kolom fitur
- `best_thresh`: threshold optimal
- `fold_scores`: akurasi per fold
- `oot_acc`: out-of-time accuracy

Model lama (single model) masih bisa di-load — sistem auto-detect format.

---

*Last updated: 2026-06-26*
