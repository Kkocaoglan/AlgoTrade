"""
fetch_data.py  (v3 — HGDG sütun formatı düzeltildi)
Kullanım:  py -3.12 fetch_data.py
"""

import sqlite3, time
from datetime import datetime, timedelta, date as _date
from pathlib import Path
import pandas as pd

try:
    from isyatirimhisse import fetch_stock_data
    HAS_ISY = True
except ImportError:
    HAS_ISY = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

DB_PATH  = Path(__file__).parent / "trade_data.db"
START    = "01-01-2022"
END      = datetime.today().strftime("%d-%m-%Y")
START_YF = "2022-01-01"

SYMBOLS = [
    "YKBNK","AKBNK","ISCTR","GARAN",
    "TUPRS","PETKM",
    "TAVHL","FROTO",
    "TCELL","ASELS",
    "BIMAS","MGROS",
    "ENKAI","EKGYO",
    # Expanded universe (added 2026-04-24)
    "THYAO","EREGL",
    "KCHOL","SAHOL",
    "SISE","TOASO",
    "ARCLK","VESTL",
    "KRDMD",
    "PGSUS","ODAS",
    "GUBRF","CIMSA",
    "LOGO","NETAS",
]

# Vol proxy symbols — also in trading universe; kept here for clarity
VOL_PROXY_SYMBOLS: list[str] = []  # THYAO/EREGL now in main SYMBOLS

YF_MAP = {s: s+".IS" for s in SYMBOLS}

# isyatirimhisse'nin HGDG_ sütun haritası
ISY_COLS = {
    "HGDG_TARIH"   : "date",
    "HGDG_KAPANIS" : "close",
    "HGDG_AOF"     : "open",   # ağırlıklı ortalama fiyat → open proxy
    "HGDG_MIN"     : "low",
    "HGDG_MAX"     : "high",
    "HGDG_HACIM"   : "volume",
}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS ohlcv (
        symbol TEXT, date TEXT, open REAL, high REAL,
        low REAL, close REAL, volume REAL, source TEXT,
        PRIMARY KEY (symbol, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetch_log (
        symbol TEXT, fetched_at TEXT, rows INTEGER, source TEXT, status TEXT)""")
    conn.commit()
    return conn

def save_df(conn, symbol, df, source):
    """DataFrame → SQLite. df'in index'i DatetimeIndex olmalı."""
    if df is None or df.empty:
        return 0
    today = _date.today()
    saved = 0
    for dt, row in df.iterrows():
        try:
            row_date = pd.Timestamp(dt).date()
            if row_date > today:
                # Reject any future-dated row — prevents corrupt isyatirimhisse
                # data (e.g. May-December 2026 rows) from polluting the DB.
                continue
            conn.execute(
                "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)",
                (symbol,
                 row_date.strftime("%Y-%m-%d"),
                 _f(row, "open"),  _f(row, "high"),
                 _f(row, "low"),   _f(row, "close"),
                 _f(row, "volume"), source))
            saved += 1
        except Exception:
            pass
    conn.commit()
    return saved

def _f(row, col):
    """Satırdan güvenli float al."""
    v = row.get(col) if hasattr(row, "get") else row[col] if col in row.index else None
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(str(v).replace(",", ".").strip())
    except Exception:
        return None

def fetch_isy(sym):
    if not HAS_ISY:
        return None
    df = None
    for attempt in range(2):
        try:
            if attempt == 0:
                df = fetch_stock_data(symbols=sym, start_date=START,
                                      end_date=END, exchange="0",
                                      frequency="1d", return_type="pandas")
            else:
                df = fetch_stock_data(sym, START, END)
            if df is not None and not df.empty:
                break
        except Exception as e:
            if attempt == 1:
                print(f"  [ISY hata] {e}")

    if df is None or df.empty:
        return None

    df = df.copy()

    # HGDG_ formatını standart isime çevir
    df = df.rename(columns=ISY_COLS)

    # Tarih → index
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
        df["date"] = pd.to_datetime(df["date"], format="mixed",
                                     dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
    else:
        try:
            df.index = pd.to_datetime(df.index.astype(str), format="mixed",
                                       dayfirst=True, errors="coerce")
            df = df[df.index.notna()]
        except Exception as e:
            print(f"  [ISY tarih hatası] {e}")
            return None

    # Eksik sütunları ekle
    for col in ("open","high","low","close","volume"):
        if col not in df.columns:
            df[col] = float("nan")
        # Scalar değilse (DataFrame vs Series karışıklığı) düzelt
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",","."),
                                 errors="coerce")

    # open yoksa close'u kopyala
    if df["open"].isna().all():
        df["open"] = df["close"]

    df = df.sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    return df[["open","high","low","close","volume"]]

def fetch_yf(sym):
    if not HAS_YF:
        return None
    ticker = YF_MAP.get(sym, sym+".IS")
    try:
        # end is exclusive in yfinance — use tomorrow so today's bar is included
        end_yf = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=START_YF,
                         end=end_yf,
                         interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        for col in ("open","high","low","close","volume"):
            if col not in df.columns:
                df[col] = float("nan")
        df = df[["open","high","low","close","volume"]]
        df = df[df["close"].notna() & (df["close"] > 0)]
        return df if not df.empty else None
    except Exception as e:
        print(f"  [YF hata] {e}")
        return None

def purge_future_rows(conn) -> int:
    """
    Delete ohlcv rows dated after today — these are corrupt rows from
    a previous bad isyatirimhisse fetch (e.g. May-December 2026 rows
    stored while today is April 2026).  Returns number of rows deleted.
    """
    today_str = _date.today().isoformat()
    cur = conn.execute("DELETE FROM ohlcv WHERE date > ?", (today_str,))
    conn.commit()
    return cur.rowcount


def fetch_all():
    print("="*60)
    print("BIST Veri Çekici v3")
    print(f"Donem     : {START_YF} -> bugun")
    print(f"Hisseler  : {', '.join(SYMBOLS)}")
    print(f"Veritabanı: {DB_PATH}")
    print("="*60)

    conn    = get_db()

    # Remove any rows that slipped in with a future date
    deleted = purge_future_rows(conn)
    if deleted:
        print(f"  [TEMIZLIK] {deleted} gelecek-tarihli satir silindi (bozuk kaynak verisi)")

    results = []

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"\n[{i:2}/{len(SYMBOLS)}] {sym}...")
        df, source = None, None

        raw = fetch_isy(sym)
        if raw is not None and len(raw) > 50:
            df, source = raw, "isyatirimhisse"
            print(f"  isyatirimhisse: {len(df)} satır OK")
        else:
            print(f"  isyatirimhisse yetersiz -> yfinance...")
            ydf = fetch_yf(sym)
            if ydf is not None:
                df, source = ydf, "yfinance"
                print(f"  yfinance: {len(df)} satır OK")
            else:
                print(f"  [FAIL] {sym}")
                conn.execute("INSERT INTO fetch_log VALUES (?,?,?,?,?)",
                             (sym, datetime.now().isoformat(), 0, "none", "FAIL"))
                conn.commit()
                results.append({"symbol":sym,"rows":0,"source":"FAIL"})
                continue

        saved = save_df(conn, sym, df, source)
        conn.execute("INSERT INTO fetch_log VALUES (?,?,?,?,?)",
                     (sym, datetime.now().isoformat(), saved, source, "OK"))
        conn.commit()
        results.append({"symbol":sym,"rows":saved,"source":source})
        print(f"  Kaydedildi: {saved} satır")
        time.sleep(0.5)

    # -- Vol proxy symbols (THYAO, EREGL) — yfinance only, not in trading universe --
    if VOL_PROXY_SYMBOLS:
        print(f"\n[VOL PROXY] {', '.join(VOL_PROXY_SYMBOLS)} guncelleniyor...")
        for sym in VOL_PROXY_SYMBOLS:
            conn2 = get_db()
            ydf = fetch_yf(sym)
            if ydf is not None:
                saved = save_df(conn2, sym, ydf, "yfinance")
                conn2.execute("INSERT INTO fetch_log VALUES (?,?,?,?,?)",
                              (sym, datetime.now().isoformat(), saved, "yfinance", "OK"))
                conn2.commit()
                results.append({"symbol": sym, "rows": saved, "source": "yfinance(vol_proxy)"})
                print(f"  {sym}: {saved} satır OK")
            else:
                conn2.execute("INSERT INTO fetch_log VALUES (?,?,?,?,?)",
                              (sym, datetime.now().isoformat(), 0, "none", "FAIL"))
                conn2.commit()
                results.append({"symbol": sym, "rows": 0, "source": "FAIL"})
                print(f"  {sym}: [FAIL]")
            conn2.close()
            time.sleep(0.5)

    conn.close()
    return results

def print_summary(results):
    print("\n"+"="*60)
    print("ÖZET")
    print("="*60)
    ok  = [r for r in results if r["source"] != "FAIL"]
    bad = [r for r in results if r["source"] == "FAIL"]
    for r in ok:
        print(f"  OK   {r['symbol']:<8} {r['rows']:>4} satır  ({r['source']})")
    for r in bad:
        print(f"  FAIL {r['symbol']:<8}")
    print(f"\nToplam: {len(ok)}/{len(results)} başarılı")
    if ok:
        conn = sqlite3.connect(DB_PATH)
        print("\nSon kapanış fiyatları:")
        for r in ok:
            row = conn.execute(
                "SELECT date, close FROM ohlcv "
                "WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (r["symbol"],)).fetchone()
            if row:
                print(f"  {r['symbol']:<8} {row[1]:>10.2f} TL   ({row[0]})")
        conn.close()

if __name__ == "__main__":
    t0  = time.time()
    res = fetch_all()
    print_summary(res)
    print(f"\nSüre: {time.time()-t0:.1f}s")
    print("\nSonraki adım: py -3.12 indicators.py")