"""
start.py -- BIST Algo Trading Baslangic Scripti
-------------------------------------------------
Kullanim:
  python3.12 start.py          # normal mod (loop calisir)
  python3.12 start.py --once   # tek tick test modu
"""

import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
PY   = sys.executable
ONCE = "--once" in sys.argv

STEPS = [
    ("Veri guncelleniyor...  (fetch_data.py)",   "fetch_data.py",   []),
    ("Indiktorler hesaplaniyor...  (indicators.py)", "indicators.py", []),
]

print("=" * 55)
print("BIST LOOP TRADER -- Baslangic")
print("=" * 55)

# 1. fetch + indicators
for label, script, extra_args in STEPS:
    print(f"\n[START] {label}")
    result = subprocess.run(
        [PY, str(BASE / script)] + extra_args,
        capture_output=False,   # terminalde canli gorunsun
    )
    if result.returncode != 0:
        print(f"\n[HATA] {script} basarisiz oldu (kod {result.returncode}). Devam edilemiyor.")
        sys.exit(1)
    print(f"[OK]    {script} tamamlandi.")

# 2. loop_trader
print("\n[START] Trader loop baslatiliyor...  (loop_trader.py)")
print("=" * 55)

trader_args = [PY, str(BASE / "loop_trader.py")]
if ONCE:
    trader_args.append("--once")

subprocess.run(trader_args)
