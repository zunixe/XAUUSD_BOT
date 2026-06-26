"""
XAUUSD MASTER DASHBOARD
Jalankan semua analisa dalam satu perintah:
  1. Update data realtime
  2. Prediksi ML + catat ke journal
  3. Analisa teknikal + level
  4. News calendar + prediksi dampak
  5. Evaluasi akurasi history
"""
import subprocess, sys, os

scripts = [
    ("UPDATE DATA", "python update_data.py"),
    ("PREDIKSI ML", "python predict_today.py"),
    ("LEVEL TEKNIKAL", "python levels.py"),
    ("JOURNAL + NEWS", "python xauusd_journal.py"),
]

print("=" * 58)
print(f"  XAUUSD MASTER DASHBOARD - {__import__('datetime').datetime.now().strftime('%d %B %Y %H:%M')}")
print("=" * 58)

for name, cmd in scripts:
    print(f"\n{'─'*58}")
    print(f"  [{name}]")
    print(f"{'─'*58}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

print("=" * 58)
print("  SELESAI - Semua analisa telah dijalankan")
print("=" * 58)
print("  File penting:")
print("  - xauusd_journal.db   : database history prediksi")
print("  - xauusd_daily.csv    : data harga harian")
print("  - xauusd_model.pkl    : model ML terlatih")
print("  - analysis/           : folder hasil analisa")
print("=" * 58)
