"""
viop_indicators.py -- VIOP teknik indikatorler
------------------------------------------------
Reads from viop_ohlcv, writes to viop_indicators.
Completely independent: does NOT touch the indicators table.

Kullanim: python3.12 viop_indicators.py
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH     = Path(__file__).parent / "trade_data.db"
VIOP_SYMBOL = "XU030F"


# ── DB setup ─────────────────────────────────────────────────────────
def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS viop_indicators (
            symbol      TEXT NOT NULL,
            date        TEXT NOT NULL,
            close       REAL,
            ema8        REAL,
            ema21       REAL,
            rsi14       REAL,
            atr14       REAL,
            macd_line   REAL,
            macd_signal REAL,
            macd_hist   REAL,
            bb_upper    REAL,
            bb_mid      REAL,
            bb_lower    REAL,
            bb_width    REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()


# ── Indicator helpers ────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger(close: pd.Series, window: int = 20):
    """Returns (upper, mid, lower, width_pct)."""
    mid   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = ((upper - lower) / (mid + 1e-9)) * 100
    return upper, mid, lower, width


def _safe(series: pd.Series, idx) -> float | None:
    try:
        val = float(series.at[idx])
        return None if np.isnan(val) else val
    except Exception:
        return None


# ── Compute & store ──────────────────────────────────────────────────
def compute_viop_indicators(sym: str = VIOP_SYMBOL) -> int:
    """
    Load viop_ohlcv for sym, compute indicators, write to viop_indicators.
    Returns number of rows written.
    """
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)

    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume "
        "FROM viop_ohlcv WHERE symbol=? ORDER BY date",
        conn, params=(sym,)
    )
    if df.empty:
        print(f"  [{sym}] viop_ohlcv bos -- once viop_data.py calistirin.")
        conn.close()
        return 0

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c = df["close"]

    ema8    = _ema(c, 8)
    ema21   = _ema(c, 21)
    rsi14   = _rsi(c, 14)
    atr14   = _atr(df["high"], df["low"], c, 14)

    # MACD 12/26/9
    ema12       = _ema(c, 12)
    ema26       = _ema(c, 26)
    macd_line   = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist   = macd_line - macd_signal

    bb_upper, bb_mid, bb_lower, bb_width = _bollinger(c, 20)

    rows_inserted = 0
    for idx in df.index:
        date_str = idx.strftime("%Y-%m-%d")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO viop_indicators
                    (symbol, date, close, ema8, ema21, rsi14, atr14,
                     macd_line, macd_signal, macd_hist,
                     bb_upper, bb_mid, bb_lower, bb_width)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sym, date_str,
                _safe(c,          idx),
                _safe(ema8,       idx),
                _safe(ema21,      idx),
                _safe(rsi14,      idx),
                _safe(atr14,      idx),
                _safe(macd_line,  idx),
                _safe(macd_signal,idx),
                _safe(macd_hist,  idx),
                _safe(bb_upper,   idx),
                _safe(bb_mid,     idx),
                _safe(bb_lower,   idx),
                _safe(bb_width,   idx),
            ))
            rows_inserted += 1
        except Exception as e:
            print(f"  Satir hatasi {date_str}: {e}")

    conn.commit()
    conn.close()
    return rows_inserted


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("VIOP Indikatorler -- XU030 Futures")
    print("Tablo: viop_indicators (indicators tablosuna dokunulmadi)")
    print("=" * 60)

    n = compute_viop_indicators()

    if n > 0:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""
            SELECT date, close, ema8, ema21, rsi14, atr14,
                   macd_line, macd_signal, bb_upper, bb_lower
            FROM viop_indicators WHERE symbol=?
            ORDER BY date DESC LIMIT 1
        """, (VIOP_SYMBOL,)).fetchone()
        conn.close()

        if row:
            d, cl, e8, e21, rsi, atr, ml, ms, bbu, bbl = row
            print(f"\n  Son gun ({d}):")
            print(f"    Kapanis  : {cl:.2f}")
            print(f"    EMA8/21  : {e8:.2f} / {e21:.2f}")
            print(f"    RSI14    : {rsi:.1f}")
            print(f"    ATR14    : {atr:.2f}")
            print(f"    MACD     : {ml:.4f} | Signal: {ms:.4f}")
            print(f"    BB       : {bbu:.2f} / {bbl:.2f}")

        print(f"\n  viop_indicators: {n} satir guncellendi.")
        print("\n[VIOP] Indikatorler hesaplandi.")
    else:
        print("\n[VIOP] Indikator hesaplama basarisiz -- veri yok.")
