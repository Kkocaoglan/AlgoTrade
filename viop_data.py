"""
viop_data.py -- XU030 Vadeli Islemler (Futures) veri cekici
------------------------------------------------------------
Completely independent from the spot system (ohlcv table untouched).
Uses viop_ohlcv table in trade_data.db.

Kullanim: python3.12 viop_data.py
"""

import sqlite3
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent / "trade_data.db"

# Ticker priority order for XU030 futures
VIOP_TICKERS = ["XU030.IS", "XU030F.IS", "^XU030"]
VIOP_PROXY   = "GARAN.IS"   # high correlation with BIST-30 -- fallback
VIOP_SYMBOL  = "XU030F"     # canonical symbol stored in DB


# ── DB setup ─────────────────────────────────────────────────────────
def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS viop_ohlcv (
            symbol  TEXT NOT NULL,
            date    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  REAL,
            source  TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()


# ── Fetch helpers ────────────────────────────────────────────────────
def _download(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download via yfinance; apply period='2d' retry for known empty-return bug."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False)
        if df.empty:
            # Known yfinance bug: some tickers return nothing with period='1y'
            # but do return data with period='2d'
            df = yf.download(ticker, period="2d", auto_adjust=True,
                             progress=False)
        return df
    except Exception:
        return pd.DataFrame()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance and lowercase."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["close"])
    return df


# ── Main fetch ───────────────────────────────────────────────────────
def fetch_viop_data() -> bool:
    """
    Fetch XU030 futures OHLCV; store in viop_ohlcv.
    Returns True on success.
    """
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)

    df          = pd.DataFrame()
    used_ticker = None
    used_source = None

    # Try official tickers first
    for ticker in VIOP_TICKERS:
        print(f"  Deneniyor: {ticker}...")
        df = _download(ticker)
        if not df.empty:
            used_ticker = ticker
            used_source = "yfinance"
            print(f"  OK: {ticker} -- {len(df)} gun")
            break
        print(f"  FAIL: {ticker} bos dondü")

    # Fallback: GARAN as BIST-30 proxy (high correlation)
    if df.empty:
        print(f"  Tum XU030 tickerlari basarisiz -- GARAN proxy kullaniliyor...")
        df = _download(VIOP_PROXY)
        if not df.empty:
            used_ticker = VIOP_PROXY
            used_source = "yfinance_proxy_GARAN"
            print(f"  PROXY OK: GARAN -- {len(df)} gun")
        else:
            print("  HATA: Hicbir kaynak calismadi. Veri cekilemedi.")
            conn.close()
            return False

    df = _normalise(df)

    rows_inserted = 0
    for date_idx, row in df.iterrows():
        date_str = date_idx.strftime("%Y-%m-%d")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO viop_ohlcv
                    (symbol, date, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                VIOP_SYMBOL, date_str,
                float(row.get("open",   0) or 0),
                float(row.get("high",   0) or 0),
                float(row.get("low",    0) or 0),
                float(row.get("close",  0)),
                float(row.get("volume", 0) or 0),
                used_source,
            ))
            rows_inserted += 1
        except Exception as e:
            print(f"  Satir hatasi {date_str}: {e}")

    conn.commit()
    conn.close()
    print(f"\n  viop_ohlcv guncellendi: {rows_inserted} satir "
          f"({VIOP_SYMBOL} / {used_ticker})")
    return True


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("VIOP Veri Cekici -- XU030 Futures")
    print("Tablo: viop_ohlcv (spot ohlcv'ye dokunulmadi)")
    print("=" * 60)

    ok = fetch_viop_data()

    if ok:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date), MAX(close) "
            "FROM viop_ohlcv WHERE symbol=?",
            (VIOP_SYMBOL,)
        ).fetchone()
        conn.close()
        count, min_d, max_d, last_close = row
        print(f"\n  Dogrulama: {count} gun  |  {min_d} -- {max_d}  "
              f"|  Son kapanis: {last_close:.2f}")
        print("\n[VIOP] Veri cekme tamamlandi.")
    else:
        print("\n[VIOP] Veri cekme basarisiz -- internet baglantisinizi kontrol edin.")
