"""
debug_db.py — veritabanını kontrol et
py -3.12 debug_db.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "trade_data.db"
conn = sqlite3.connect(DB_PATH)

# Tablolar
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tablolar:", [t[0] for t in tables])

# ohlcv satır sayısı
n = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
print(f"ohlcv toplam satır: {n}")

# İlk 3 satır ham görünüm
print("\nohlcv ilk 3 satır (ham):")
rows = conn.execute("SELECT * FROM ohlcv LIMIT 3").fetchall()
for r in rows:
    print(" ", r)

# Sütun isimleri
cols = conn.execute("PRAGMA table_info(ohlcv)").fetchall()
print("\nohlcv sütunları:")
for c in cols:
    print(" ", c)

# GARAN için birkaç satır
print("\nGARAN son 3 satır:")
rows = conn.execute(
    "SELECT * FROM ohlcv WHERE symbol='GARAN' ORDER BY date DESC LIMIT 3"
).fetchall()
for r in rows:
    print(" ", r)

conn.close()