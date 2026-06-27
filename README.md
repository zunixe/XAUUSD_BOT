# XAUUSD AI Trading System

Sistem prediksi XAUUSD (Gold) berbasis Machine Learning dengan **3-class classification**, continuous learning, paper trading, dan Telegram bot.

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
│  │  Source  │    │  Engineering │    │  (XGBoost 3cls)  │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│       │                                      │               │
│       ▼                                      ▼               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Kitco   │    │  Patterns    │    │  Ensemble        │   │
│  │  yfinance│    │  (S/R,       │    │  (4 fold models  │   │
│  │  Yahoo   │    │   candle,    │    │   + Optuna)      │   │
│  │          │    │   regime)    │    │                  │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                                               │               │
│                                               ▼               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Telegram │◀───│  Daemon      │◀───│  Signal          │   │
│  │  Bot     │    │  (launchd)   │    │  Generator       │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│       │                 │                       │             │
│       ▼                 ▼                       ▼             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  User    │    │  OHLC Eval   │    │  Paper Trading   │   │
│  │  Chat    │    │  (open-based │    │  Simulation      │   │
│  │          │    │   heuristic) │    │                  │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Alur Data

1. **Data Collection**: `update_data.py` download data OHLCV harian + 4 jam dari yfinance, plus data makro (DXY, VIX, SPY, US10Y, OIL)
2. **Feature Engineering**: `trading.py` generate 70+ fitur teknikal + makro + regime + pola
3. **Training**: `auto_runner.py` (daily) / `runner_4h.py` (4H) train model XGBoost 3-class
4. **Prediction**: Model menghasilkan probabilitas BEARISH / NEUTRAL / BULLISH
5. **Signal**: Jika probabilitas BULLISH atau BEARISH melewati threshold, sinyal dikirim ke Telegram
6. **Evaluation**: Harga real-time dicek terhadap SL/TP dengan open-price heuristic, outcome dicatat
7. **Feedback**: Outcome real digunakan untuk retrain model (continuous learning, min 30 evaluasi)

---

## 2. File Structure

| File | Fungsi |
|------|--------|
| `trading.py` | **Core library** — feature engineering, ATR 3-class labeling, Optuna tuning, walk-forward CV, TP/SL evaluation |
| `auto_runner.py` | **Daily runner + daemon** — retrain 3-class model, prediksi, evaluasi historis, Telegram polling, model versioning |
| `runner_4h.py` | **4H runner** — retrain 3-class model 4 jam, prediksi, evaluasi OHLC |
| `telegram_notifier.py` | **Telegram bot** — command handling, notifikasi, price fetching, persistent offset |
| `simulation.py` | **Paper trading** — virtual account, lot sizing, P&L tracking dengan outcome detail |
| `patterns.py` | **Pattern detection** — candlestick patterns, S/R, double top/bottom (vectorized) |
| `update_data.py` | **Data downloader** — download OHLCV + makro data (guarded `__main__`) |
| `.env` | **Credentials** — Telegram bot token dan chat ID |
| `.tg_offset` | **Telegram offset** — persistent getUpdates offset |
| `xauusd_journal.db` | **Database SQLite** — predictions, predictions_4h, simulation, outcomes |
| `xauusd_model.pkl` | **Model daily** — XGBoost 3-class ensemble + scaler + threshold |
| `xauusd_model_4h.pkl` | **Model 4H** — XGBoost 3-class ensemble + scaler + threshold |
| `xauusd_model_YYYYMMDD_HHMM.pkl` | **Model versioned** — backup model per retrain |

---

## 3. Machine Learning Pipeline

### Model: XGBoost 3-Class Ensemble

Sistem menggunakan **3-class classification** (BEARISH / NEUTRAL / BULLISH) dengan **ensemble dari fold models**:

- **Daily**: 4 fold models, bobot berdasarkan macro F1 score
- **4H**: 3 fold models, bobot berdasarkan macro F1 score

Prediksi = weighted average dari semua fold models:
```
probs = Σ (weight_i × model_i.predict_proba(X))
# probs = [prob_bearish, prob_neutral, prob_bullish]
```

Keunggulan 3-class vs binary:
- Model bisa mengenali pola **bearish secara eksplisit** (bukan cuma "bukan bullish")
- NO_TRADE zone terjadi ketika baik bearish maupun bullish tidak confident
- Sinyal SELL lebih reliable karena model dilatih untuk mengenali downside

### Hyperparameter Tuning: Optuna

Setiap fold di-tune menggunakan **Optuna** dengan:
- **Daily**: 30 trials per fold
- **4H**: 20 trials per fold
- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Metric**: macro F1 score pada evaluasi subset (bukan early-stop subset)

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
- **Scaler**: fit **sekali** di semua non-OOT data (konsisten untuk semua fold)

### Validation Split (Anti-Overfitting)

Setiap fold validation di-split:
- **70%** → early stopping (passed ke `eval_set`)
- **30%** → evaluasi F1 (Optuna objective)

Ini mencegah data leakage di mana early stopping dan evaluasi pakai data yang sama.

### Sample Weight

- **ATR labels**: weight = 1.0
- **Real outcomes** (dari SL/TP aktual): weight = 2.0
- Real outcomes hanya di-override jika ≥ 10 samples DAN ≥ 5% dari total data

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

### Regime Features (NEW)
| Feature | Deskripsi |
|---------|-----------|
| `ATR_Pctile_100` | ATR percentile rank dalam 100 bar terakhir (deteksi volatilitas regime) |
| `ADX_Proxy` | Proxy ADX — trend strength (0-1) |

### Calendar Features (Cyclical Encoding)
| Feature | Deskripsi |
|---------|-----------|
| `DayOfWeek_sin/cos` | Hari dalam minggu, encoded sebagai sin/cos (周期性) |
| `Month_sin/cos` | Bulan, encoded sebagai sin/cos |
| `Hour_sin/cos` | Jam (khusus 4H), encoded sebagai sin/cos |

Cyclical encoding memastikan Desember (12) dan Januari (1) dianggap berdekatan, bukan berjauhan.

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

### Macro Features 4H (NEW)
Model 4H sekarang juga menggunakan data DXY yang di-resample dari daily:
| Feature | Deskripsi |
|---------|-----------|
| `DXY_Return_1b/4b` | DXY return per 1/4 bar 4H |
| `DXY_Dist_EMA20` | Jarak DXY ke EMA20 |
| `GOLD_DXY_Corr_8` | Korelasi Gold-DXY 8 bar 4H |

### FOMC Features
| Feature | Deskripsi |
|---------|-----------|
| `Is_FOMC_Day` | Apakah hari ini FOMC |
| `Days_To_FOMC` | Jumlah hari menuju FOMC berikutnya (bisect, O(n log k)) |
| `Week_Before_FOMC` | 7 hari sebelum FOMC |
| `Week_After_FOMC` | 7 hari setelah FOMC |

### Pattern Features (dari `patterns.py`)
| Feature | Deskripsi |
|---------|-----------|
| `Body_Ratio` | Rasio body terhadap range candle |
| `Upper/Lower_Wick_Ratio` | Rasio wick atas/bawah |
| `Is_Doji/Hammer/Shooting_Star` | Pola candlestick (hanya jika range > $0.10) |
| `Is_Pin_Bar/Marubozu` | Pola candlestick lanjutan |
| `Is_Bullish/Bearish_Engulfing` | Pola engulfing |
| `Is_Inside_Bar` | Inside bar |
| `Dist_Resistance/Support_Pct` | Jarak ke resistance/support |
| `Near_Resistance/Support` | Dekat resistance/support (< 0.5%) |
| `Breakout_High/Low` | Breakout dari high/low |
| `Touch_Count_High/Low` | Jumlah sentuhan di level |
| `Is_Double_Top/Bottom` | Pola double top/bottom (vectorized) |

---

## 5. Labeling & Target

### ATR Triple-Barrier Labeling (3-Class)

Target (label) dihitung menggunakan **ATR-based triple-barrier method** dengan **3 kelas**:

```python
entry = Close[i]
atr = TR.rolling(14).mean().shift(1)[i]  # shifted, no future leak
tp = entry + atr * 0.8                    # take profit level
sl = entry - atr * 1.2                    # stop loss level

# Forward scan (3 bar ke depan untuk daily)
first_tp = bar pertama di mana High >= tp
first_sl = bar pertama di mana Low <= sl

if first_tp < first_sl → Target = 2 (BULLISH)
elif first_sl < first_tp → Target = 0 (BEARISH)
else → Target = 1 (NEUTRAL)
```

Parameter (aligned dengan trading):
| Parameter | Daily | 4H |
|-----------|-------|-----|
| `forward` | 3 bar (3 hari) | 8 bar (32 jam) |
| `atr_mult` (TP) | 0.8 | 0.8 |
| `sl_mult` (SL) | 1.2 | 1.2 |

Label multipliers sekarang **identik** dengan trading SL/TP multipliers (sebelumnya berbeda).

### Real Outcome Override

Jika ada data real outcome (dari evaluasi SL/TP aktual), label ATR di-override:
- **WIN + BUY** → Target = 2 (BULLISH)
- **WIN + SELL** → Target = 2 (BULLISH — prediksi benar)
- **LOSS + BUY** → Target = 0 (BEARISH)
- **LOSS + SELL** → Target = 0 (BEARISH — prediksi benar)
- **Weight**: 2x (lebih dipercaya daripada ATR label)
- **Syarat**: minimal 10 samples DAN ≥ 5% dari total data

---

## 6. Training & Validasi

### Alur Training (Daily)

```
1. Load data daily (xauusd_daily.csv)
2. Hitung ATR target 3-class (triple-barrier, aligned multipliers)
3. Override dengan real outcomes (jika ada, min 10 samples)
4. Feature engineering (70+ fitur termasuk regime + cyclical)
5. Walk-forward split (4 folds + OOT)
6. Fit scaler SEKALI di semua non-OOT data
7. Untuk setiap fold:
   a. Transform dengan scaler (fit sudah dilakukan)
   b. Split val: 70% early-stop, 30% evaluasi
   c. Tune hyperparameter dengan Optuna (30 trials)
   d. Train model multi:softprob (3-class)
   e. Hitung macro F1 pada evaluasi subset
   f. Simpan model dan score
8. Ensemble: weighted average dari semua fold models
9. OOT evaluation dengan classification report per kelas
10. Feedback loop: adjust threshold (min 30 evaluasi)
11. Simpan model ke xauusd_model.pkl + versi timestamped
```

### Alur Training (4H)

Sama dengan daily, tapi:
- 3 folds (bukan 4)
- 20 Optuna trials (bukan 30)
- Embargo = 8 bar (bukan 3)
- Data dari xauusd_4h.csv (termasuk DXY resampled)

---

## 7. Signal Generation (Trigger)

### Kondisi Sinyal (3-Class)

Sinyal dikirim ke Telegram jika probabilitas kelas tertentu melewati threshold:

#### Daily & 4H
| Kondisi | Threshold |
|---------|-----------|
| **BUY** | `prob_bullish ≥ max(0.55, best_thresh)` |
| **SELL** | `prob_bearish ≥ max(0.55, best_thresh)` |
| **NO_TRADE** | Kedua prob di bawah threshold |

### Contoh

| prob_bearish | prob_neutral | prob_bullish | Threshold 0.55 | Hasil |
|-------------|-------------|-------------|---------------|-------|
| 15% | 13% | 72% | ≥ 0.55 | **BUY** (bull 72%) |
| 65% | 20% | 15% | ≥ 0.55 | **SELL** (bear 65%) |
| 30% | 40% | 30% | < 0.55 | **NO_TRADE** |

### SL/TP Calculation

SL/TP dihitung dari ATR dan level teknikal:

**BUY:**
```python
sl  = max(low_20, entry - atr * 1.2)
tp1 = max(entry + atr * 0.8, bb_upper)
tp2 = max(entry + atr * 1.8, ema50)
```

**SELL:**
```python
sl  = min(high_20, entry + atr * 1.2)
tp1 = min(entry - atr * 0.8, bb_lower)
tp2 = min(entry - atr * 1.8, low_20)
```

---

## 8. Real-Time Price & Evaluation

### Harga Real-Time

Harga diambil dari **Kitco** (web scraping) dengan fallback ke **yfinance**:
- Multi-pattern regex untuk menangkap berbagai format harga
- 2 retry attempts jika gagal
- SSL per-request (tidak disabled global)
- Fallback ke GC=F (Gold Futures) 1-minute data

### Evaluasi OHLC (Batch)

Untuk prediksi yang belum dievaluasi, dilakukan evaluasi OHLC batch:
- Scan bar forward dari entry point (**max 10 bar**, sebelumnya 5)
- **Open-price heuristic**: Jika TP dan SL kena di bar yang sama, gunakan open price untuk tebak mana yang kena duluan
  - Open dekat SL → SL kena duluan
  - Open dekat TP → TP kena duluan
- Jika tidak kena dalam 10 bar → EXPIRED (lihat close terakhir)

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
- **Minimum SL distance**: 0.03% dari harga (bukan absolut $1)
- **Jika lot < 0.01** → tidak trade (return None)

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

| Outcome | Detail | Exit Price |
|---------|--------|-----------|
| WIN | TP1_HIT | TP1 |
| WIN | TP2_HIT | TP2 |
| LOSS | SL_HIT | SL |
| EXPIRED | — | Close price terakhir |
| NO_TRADE | — | Tidak ada trade |

P&L simulation sekarang menggunakan **outcome_detail** (TP1 vs TP2) untuk exit price yang akurat.

---

## 10. Telegram Bot

### Commands

| Command | Fungsi |
|---------|--------|
| `/start` atau `/start 100` | Mulai paper trading dengan balance awal |
| `/bal` | Lihat balance dan statistik |
| `/price` | Harga XAUUSD real-time |
| `/info` | Info prediksi dan performa bulanan |
| `/retrain` | Trigger retrain manual (async, hasil dikirim ke Telegram) |
| `/reset` | Reset simulasi |

### Polling

- Polling Telegram setiap **5 detik**
- Offset disimpan persistent di `.tg_offset` (tidak replay perintah lama)
- Setiap command di-handle terpisah (satu error tidak blok command lain)

### Signal Notification

Format sinyal yang dikirim ke Telegram:

```
🔥 [XAUUSD SIGNAL] #4 — Daily Candle
BUY LONG
  Confidence : 55.1% (threshold 0.550)
  Daily Close: $4044.40
  ▶ Entry    : $4044.40
  SL         : $3963.30 (-$81.10)
  TP1        : $4604.65 (+$560.25, RR 1:6.91)
  TP2        : $4427.78 (+$383.38, RR 1:4.73)
  ATR / RSI  : $67.6 / 52
  Target     : 2026-06-29
```

### Outcome Notification

```
🟢 [CLOSED] #4 — Daily
  Direction: BUY (Bullish)
  Entry: $4044.40
  Result: TP1 HIT! +0.45%
  Profit: $18.30 (+0.45%)
  Sim: 0.03 lot | P&L: +$18.30 | Bal: $118.30
```

---

## 11. Continuous Learning (Feedback Loop)

### Mekanisme

```
Prediksi → Evaluasi → Simpan Outcome → Retrain → Model Baru (3-class)
    ↑                                                    │
    └────────────────────────────────────────────────────┘
```

### Kapan Retrain?

- **Daemon**: otomatis setiap 4 jam (update data + prediksi + evaluasi + retrain jika perlu)
- **Manual**: via `/retrain` command di Telegram
- **Syarat**: ≥ 10 evaluasi baru sejak retrain terakhir ATAU `should_retrain()` true

### Feedback Loop (Threshold Adjustment)

Threshold diadjust berdasarkan performa historis dengan **time decay**:

```python
# Minimal 30 evaluasi sebelum adjust (sebelumnya 10)
# Outcomes 30 hari terakhir = 2x weight
weighted_acc = (recent_wins * 2 + older_wins) / (recent_count * 2 + older_count)

if weighted_acc < 0.45:    threshold += 0.08  (max 0.70)  # sangat buruk
elif weighted_acc < 0.55:  threshold += 0.03  (max 0.65)  # buruk
elif weighted_acc >= 0.65: threshold -= 0.02  (min 0.55)  # bagus → turunkan
# 0.55-0.64: tidak ada perubahan (stabil)
```

Sifat:
- **Konservatif**: naik agresif, turun pelan
- **Time decay**: outcome terbaru lebih berpengaruh
- **Min 30 evaluasi**: mencegah over-reaksi ke sample kecil
- **Bisa turun**: threshold tidak stuck di high selamanya

### Model Versioning

Setiap retrain menyimpan model dengan timestamp:
```
xauusd_model.pkl              → model aktif (terbaru)
xauusd_model_20260627_0727.pkl → backup versi
```

---

## 12. Cara Menjalankan

### macOS (Daemon via launchd)

Daemon sudah terinstall dan auto-start saat login:

```bash
# Cek status
launchctl list | grep xauusd

# Stop
launchctl unload ~/Library/LaunchAgents/com.xauusd.bot.plist

# Start
launchctl load ~/Library/LaunchAgents/com.xauusd.bot.plist

# Lihat log live
tail -f ~/Documents/XAUUSD_bot/auto_runner.log
```

### Prerequisites

```bash
# Buat virtual environment
uv venv .venv --python 3.11

# Install dependencies
uv pip install pandas numpy scikit-learn xgboost optuna joblib yfinance matplotlib seaborn

# Install libomp (untuk XGBoost di macOS)
brew install libomp
```

### Manual Commands

```bash
# Download/update data
.venv/bin/python update_data.py

# Train model daily (3-class)
.venv/bin/python auto_runner.py --retrain

# Train model 4H (3-class)
.venv/bin/python runner_4h.py --retrain

# Run single cycle (update + predict + evaluate)
.venv/bin/python auto_runner.py

# Show learning stats
.venv/bin/python auto_runner.py --stats

# Daemon mode
.venv/bin/python auto_runner.py --daemon --interval 4
```

### Logs

- `auto_runner.log` — daily training + daemon log
- `runner_4h.log` — 4H training log
- `poll.log` — Telegram command log
- `launchd_stdout.log` / `launchd_stderr.log` — daemon stdout/stderr

---

## 13. Troubleshooting

### Daemon tidak jalan

```bash
# Cek status
launchctl list | grep xauusd
ps aux | grep auto_runner | grep -v grep

# Restart
launchctl unload ~/Library/LaunchAgents/com.xauusd.bot.plist
launchctl load ~/Library/LaunchAgents/com.xauusd.bot.plist
```

### Model tidak ada / error load

```bash
# Retrain dari awal
.venv/bin/python auto_runner.py --retrain
.venv/bin/python runner_4h.py --retrain
```

### Telegram tidak merespons

1. Cek `poll.log` untuk error
2. Pastikan token bot benar di `.env`
3. Pastikan chat_id benar
4. Offset tersimpan di `.tg_offset` — hapus file ini jika ada masalah replay

### Database locked

```bash
# Restart daemon (release connection)
launchctl unload ~/Library/LaunchAgents/com.xauusd.bot.plist
launchctl load ~/Library/LaunchAgents/com.xauusd.bot.plist
```

### XGBoost libomp error

```bash
brew install libomp
```

---

## Catatan Teknis

### SSL

SSL verification dilakukan per-request dengan context terpisah (tidak disabled global). Setiap HTTP request ke Telegram API, Kitco, dan yfinance menggunakan SSL context sendiri yang fleksibel.

### Model File Compatibility

Model file (.pkl) berisi:
- `ensemble_models`: list of (model, weight) tuples — 3-class XGBoost
- `scaler`: RobustScaler yang sudah fit di semua non-OOT data
- `feature_cols`: nama kolom fitur (70+ fitur)
- `n_classes`: 3 (BEARISH, NEUTRAL, BULLISH)
- `best_thresh`: threshold optimal (min 0.55)
- `model_version`: timestamp versi
- `fold_scores`: akurasi per fold
- `oot_acc`: out-of-time accuracy

Model lama (binary/single model) masih bisa di-load — sistem auto-detect format.

### Database Schema

**predictions** (daily):
- id, date, price, predicted_direction, confidence, threshold, target_date, model_version, sl, tp1, tp2, entry_realtime, outcome, outcome_detail, result_pct, notified

**predictions_4h**:
- id, date, time, price, predicted_direction, confidence, threshold, target_date, target_time, model_version, sl, tp1, tp2, entry_realtime, outcome, outcome_detail, result_pct, notified

**simulation**:
- id, balance, initial_balance, active, created_at

**sim_trades**:
- id, sim_id, prediction_id, timeframe, direction, entry, sl, tp1, lot_size, risk_amount, outcome, pnl, balance_before, balance_after, created_at

---

*Last updated: 2026-06-27*
