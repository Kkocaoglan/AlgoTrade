"""
indicators.py  (v2 — parse_dates fix)
Kullanım: py -3.12 indicators.py
"""

import sqlite3, time, json
from datetime import date as _date
from pathlib import Path
import pandas as pd
import numpy as np

DB_PATH = Path(__file__).parent / "trade_data.db"
RESULTS = Path(__file__).parent / "results"

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

def load_sentiment_scores() -> dict:
    """Load sentiment_scores.json produced by news_filter.py. Returns {} if missing."""
    path = RESULTS / "sentiment_scores.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            symbol TEXT, date TEXT,
            close REAL,
            ema8 REAL, ema21 REAL, ema50 REAL, ema200 REAL, sma20 REAL,
            rsi14 REAL,
            macd_line REAL, macd_signal REAL, macd_hist REAL,
            atr14 REAL,
            bb_upper REAL, bb_mid REAL, bb_lower REAL, bb_width REAL,
            obv REAL, vol_ratio REAL,
            mtf_trend INTEGER,
            rsi_zone TEXT, bb_zone TEXT,
            above_ema200 INTEGER, golden_cross INTEGER,
            sentiment_score REAL DEFAULT 0.0,
            news_count INTEGER DEFAULT 0,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            date TEXT PRIMARY KEY,
            usdtry REAL,
            brent REAL,
            tcmb_rate REAL DEFAULT 43.0
        )
    """)
    conn.commit()
    # Migration: add new columns to existing databases that lack them
    for col, defn in [
        ("sentiment_score",    "REAL DEFAULT 0.0"),
        ("news_count",         "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE indicators ADD COLUMN {col} {defn}")
            conn.commit()
        except Exception:
            pass   # column already exists

    # Migration: add new macro_data columns (cross-asset + derived features)
    macro_new_cols = [
        ("vix",               "REAL"),
        ("eem",               "REAL"),
        ("gold_usd",          "REAL"),
        ("dxy",               "REAL"),
        ("sp500",             "REAL"),
        ("stoxx50",           "REAL"),
        ("usdtry_1d_ret",     "REAL"),
        ("usdtry_5d_ret",     "REAL"),
        ("usdtry_20d_zscore", "REAL"),
        ("usdtry_above_20ma", "INTEGER"),
        ("brent_1d_ret",      "REAL"),
        ("brent_5d_ret",      "REAL"),
        ("vix_level",         "REAL"),
        ("vix_5d_zscore",     "REAL"),
        ("gold_try_ratio",    "REAL"),
        ("dxy_5d_ret",        "REAL"),
        ("sp500_overnight_ret","REAL"),
        ("stoxx50_am_ret",    "REAL"),
        ("em_5d_ret",         "REAL"),
        ("macro_risk_score",  "REAL"),
    ]
    for col, defn in macro_new_cols:
        try:
            conn.execute(f"ALTER TABLE macro_data ADD COLUMN {col} {defn}")
            conn.commit()
        except Exception:
            pass   # column already exists
    return conn


def fetch_macro_data(conn) -> int:
    """
    Fetch 8 cross-asset tickers from yfinance, compute 12 derived macro features,
    and store all into macro_data table (pre-computed with ffill so BIST calendar
    gaps are handled correctly — no more fillna(0) causing false zero-returns).

    Raw tickers: TRY=X, BZ=F, ^VIX, EEM, GC=F, DX-Y.NYB, ^GSPC, ^STOXX50E
    Derived (12): usdtry_1d_ret, usdtry_5d_ret, usdtry_20d_zscore, usdtry_above_20ma,
                  vix_level, vix_5d_zscore, gold_try_ratio, dxy_5d_ret,
                  sp500_overnight_ret, stoxx50_am_ret, em_5d_ret, macro_risk_score
                  + brent_1d_ret, brent_5d_ret (pre-computed for ml_train.py)

    TCMB rate hardcoded at 43.0 — update manually after each TCMB meeting.
    Returns number of rows saved.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance yok, macro veri atlanıyor")
        return 0

    TCMB_RATE = 43.0  # Last known rate — update manually

    TICKERS = {
        "TRY=X":     "usdtry",
        "BZ=F":      "brent",
        "^VIX":      "vix",
        "EEM":       "eem",
        "GC=F":      "gold_usd",
        "DX-Y.NYB":  "dxy",
        "^GSPC":     "sp500",
        "^STOXX50E": "stoxx50",
    }

    raw_series = {}
    for ticker, col in TICKERS.items():
        try:
            raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  {ticker}: veri yok")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index)
            if hasattr(raw.index, "tz") and raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
            close_col = next((c for c in ["Close", "close"] if c in raw.columns), None)
            if close_col is None:
                print(f"  {ticker}: Close sutunu bulunamadi")
                continue
            s = raw[close_col].dropna()
            raw_series[col] = s
            print(f"  {ticker} ({col}): {len(s)} gun indirildi")
        except Exception as e:
            print(f"  {ticker} indirilemedi: {e}")

    if not raw_series:
        return 0

    # Build combined DataFrame; forward-fill across full calendar (including weekends/holidays)
    # so that when ml_train.py reindexes to BIST dates and calls ffill(), values are always present
    macro_df = pd.DataFrame(raw_series)
    full_idx = pd.date_range(macro_df.index.min(), macro_df.index.max(), freq="D")
    macro_df = macro_df.reindex(full_idx).ffill().bfill()

    # Convenience: pull each series (NaN series if not fetched)
    def _col(name):
        return macro_df[name] if name in macro_df.columns else pd.Series(np.nan, index=macro_df.index)

    usdtry = _col("usdtry")
    brent  = _col("brent")
    vix    = _col("vix")
    eem    = _col("eem")
    gold   = _col("gold_usd")
    dxy    = _col("dxy")
    sp500  = _col("sp500")
    stoxx  = _col("stoxx50")

    # ── Derived features ──────────────────────────────────────────
    # USDTRY
    macro_df["usdtry_1d_ret"]     = usdtry.pct_change(1)
    macro_df["usdtry_5d_ret"]     = usdtry.pct_change(5)
    usdtry_20ma  = usdtry.rolling(20, min_periods=5).mean()
    usdtry_20std = usdtry.rolling(20, min_periods=5).std()
    macro_df["usdtry_above_20ma"] = (usdtry > usdtry_20ma).astype(int)
    macro_df["usdtry_20d_zscore"] = (usdtry - usdtry_20ma) / (usdtry_20std + 1e-9)

    # Brent
    macro_df["brent_1d_ret"] = brent.pct_change(1)
    macro_df["brent_5d_ret"] = brent.pct_change(5)

    # VIX
    macro_df["vix_level"]    = vix / 100.0            # normalize ~[0,1]
    vix_5ma  = vix.rolling(5, min_periods=2).mean()
    vix_5std = vix.rolling(5, min_periods=2).std()
    macro_df["vix_5d_zscore"] = (vix - vix_5ma) / (vix_5std + 1e-9)

    # Gold in TRY 5-day return (gold_usd × usdtry = effective gold_try)
    gold_try = gold * usdtry
    macro_df["gold_try_ratio"] = gold_try.pct_change(5)

    # DXY
    macro_df["dxy_5d_ret"] = dxy.pct_change(5)

    # S&P 500 overnight return (US close → BIST next open proxy)
    macro_df["sp500_overnight_ret"] = sp500.pct_change(1)

    # STOXX50 AM return (European session open proxy for BIST)
    macro_df["stoxx50_am_ret"] = stoxx.pct_change(1)

    # MSCI EM 5-day return
    macro_df["em_5d_ret"] = eem.pct_change(5)

    # Composite macro risk score
    # Negative = risk-off (bad for BIST): rising VIX, rising USDTRY, rising DXY
    macro_df["macro_risk_score"] = (
        macro_df["vix_5d_zscore"].fillna(0)    * -0.3 +
        macro_df["usdtry_20d_zscore"].fillna(0) * -0.4 +
        macro_df["dxy_5d_ret"].fillna(0)        * -0.3
    )

    # ── Save to DB ────────────────────────────────────────────────
    def _fv(series_or_val, idx):
        """Safe float extractor — returns None if NaN."""
        try:
            v = float(series_or_val.iloc[idx] if hasattr(series_or_val, "iloc") else series_or_val)
            return None if (v != v) else v   # NaN check
        except Exception:
            return None

    saved = 0
    rows = list(macro_df.itertuples())
    for i, row in enumerate(rows):
        date_str = str(row.Index)[:10]

        def fv(col):
            v = getattr(row, col, None)
            if v is None: return None
            try:
                f = float(v)
                return None if (f != f) else f
            except Exception:
                return None

        # usdtry_above_20ma is int (0/1); keep as int for DB
        uam = fv("usdtry_above_20ma")
        uam_int = int(uam) if uam is not None else None

        try:
            conn.execute("""
                INSERT OR REPLACE INTO macro_data (
                    date, usdtry, brent, tcmb_rate,
                    vix, eem, gold_usd, dxy, sp500, stoxx50,
                    usdtry_1d_ret, usdtry_5d_ret, usdtry_20d_zscore, usdtry_above_20ma,
                    brent_1d_ret, brent_5d_ret,
                    vix_level, vix_5d_zscore,
                    gold_try_ratio, dxy_5d_ret,
                    sp500_overnight_ret, stoxx50_am_ret,
                    em_5d_ret, macro_risk_score
                ) VALUES (
                    ?,?,?,?,
                    ?,?,?,?,?,?,
                    ?,?,?,?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,?
                )
            """, (
                date_str,
                fv("usdtry"), fv("brent"), TCMB_RATE,
                fv("vix"), fv("eem"), fv("gold_usd"), fv("dxy"), fv("sp500"), fv("stoxx50"),
                fv("usdtry_1d_ret"), fv("usdtry_5d_ret"), fv("usdtry_20d_zscore"), uam_int,
                fv("brent_1d_ret"), fv("brent_5d_ret"),
                fv("vix_level"), fv("vix_5d_zscore"),
                fv("gold_try_ratio"), fv("dxy_5d_ret"),
                fv("sp500_overnight_ret"), fv("stoxx50_am_ret"),
                fv("em_5d_ret"), fv("macro_risk_score"),
            ))
            saved += 1
        except Exception as e:
            pass   # skip individual row errors silently

    conn.commit()
    return saved

# ── İndikatör fonksiyonları ───────────────────────────────────

def ema(s, p):    return s.ewm(span=p, adjust=False).mean()
def sma(s, p):    return s.rolling(p).mean()

def rsi(s, p=14):
    d    = s.diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s, fast=12, slow=26, sig=9):
    line   = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    return line, signal, line - signal

def atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def bollinger(s, p=20, k=2.0):
    mid   = sma(s, p)
    sigma = s.rolling(p).std()
    upper = mid + k * sigma
    lower = mid - k * sigma
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    return upper, mid, lower, width

def obv_calc(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()

def vol_ratio(v, p=20):
    return v / v.rolling(p).mean().replace(0, np.nan)

def mtf_score(row):
    score = 0
    pairs = [("ema8","ema21"), ("ema21","ema50"), ("ema50","ema200")]
    for a, b in pairs:
        if pd.notna(row.get(a)) and pd.notna(row.get(b)):
            score += 1 if row[a] > row[b] else -1
    return 1 if score >= 2 else (-1 if score <= -2 else 0)

def rsi_zone(v):
    if pd.isna(v):   return "unknown"
    if v < 30:        return "oversold"
    if v > 70:        return "overbought"
    return "neutral"

def bb_zone(w, series):
    if pd.isna(w): return "unknown"
    p20 = series.quantile(0.20)
    p80 = series.quantile(0.80)
    if w < p20:    return "squeeze"
    if w > p80:    return "expand"
    return "normal"

# ── Hesapla & kaydet ─────────────────────────────────────────

def compute(conn, sym, sentiment_scores=None):
    # TEXT olarak oku, sonra manuel parse et
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume "
        "FROM ohlcv WHERE symbol=? ORDER BY date",
        conn, params=(sym,)
    )
    if df.empty or len(df) < 30:
        return 0

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")

    for col in ("open","high","low","close","volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ema8"]   = ema(c, 8)
    df["ema21"]  = ema(c, 21)
    df["ema50"]  = ema(c, 50)
    df["ema200"] = ema(c, 200)
    df["sma20"]  = sma(c, 20)
    df["rsi14"]  = rsi(c, 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["atr14"]  = atr(h, l, c, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_width"] = bollinger(c)
    df["obv"]      = obv_calc(c, v)
    df["vol_ratio"]= vol_ratio(v, 20)

    df["mtf_trend"]    = df.apply(mtf_score, axis=1)
    df["rsi_zone"]     = df["rsi14"].apply(rsi_zone)
    df["bb_zone"]      = df["bb_width"].apply(lambda x: bb_zone(x, df["bb_width"]))
    df["above_ema200"] = (c > df["ema200"]).astype(int)
    df["golden_cross"] = (df["ema50"] > df["ema200"]).astype(int)

    df = df.reset_index()
    saved = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO indicators VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sym,
                str(row["date"])[:10],
                _f(row,"close"),
                _f(row,"ema8"),  _f(row,"ema21"),
                _f(row,"ema50"), _f(row,"ema200"), _f(row,"sma20"),
                _f(row,"rsi14"),
                _f(row,"macd_line"), _f(row,"macd_signal"), _f(row,"macd_hist"),
                _f(row,"atr14"),
                _f(row,"bb_upper"), _f(row,"bb_mid"),
                _f(row,"bb_lower"), _f(row,"bb_width"),
                _f(row,"obv"),   _f(row,"vol_ratio"),
                int(row["mtf_trend"]),
                str(row["rsi_zone"]),
                str(row["bb_zone"]),
                int(row["above_ema200"]),
                int(row["golden_cross"]),
                0.0,   # sentiment_score — updated below for latest row
                0,     # news_count — updated below for latest row
            ))
            saved += 1
        except Exception:
            pass
    conn.commit()

    # Apply current-day sentiment to the latest row only
    if sentiment_scores:
        entry = sentiment_scores.get(sym, {})
        if isinstance(entry, dict):
            score = entry.get("score", 0.0)
            count = entry.get("count",  0)
        else:
            score, count = float(entry), 0
        if score != 0.0 or count != 0:
            latest_date = str(df["date"].max())[:10]
            try:
                conn.execute("""
                    UPDATE indicators SET sentiment_score=?, news_count=?
                    WHERE symbol=? AND date=?
                """, (score, count, sym, latest_date))
                conn.commit()
            except Exception:
                pass

    return saved

def _f(row, col):
    v = row.get(col) if hasattr(row,"get") else None
    if v is None: return None
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except Exception:
        return None

# ── Özet çıktılar ─────────────────────────────────────────────

def print_table(conn):
    print("\n" + "="*68)
    print(f"  {'SEM':<8} {'FİYAT':>8} {'RSI':>6} {'TREND':>7} {'BB_ZONE':<10} {'EMA200':>8} {'GOLDEN':>6}")
    print("  " + "-"*65)
    for sym in SYMBOLS:
        row = conn.execute("""
            SELECT close, rsi14, mtf_trend, bb_zone, ema200, golden_cross
            FROM indicators WHERE symbol=? ORDER BY date DESC LIMIT 1
        """, (sym,)).fetchone()
        if not row: continue
        c, r, t, bz, e200, gc = row
        tstr = "YUKARI" if t==1 else " ASAGI" if t==-1 else " YATAY"
        print(f"  {sym:<8} {c:>8.2f} {r:>6.1f} {tstr:>7} {(bz or ''):.<10} {(e200 or 0):>8.2f} {'ALTIN' if gc else '':>6}")

def print_signals(conn):
    print("\n" + "="*68)
    print("SİNYAL ÖZETİ — bugünün sinyalleri")
    print("="*68)
    long_c, short_c, squeeze = [], [], []

    for sym in SYMBOLS:
        row = conn.execute("""
            SELECT mtf_trend, rsi_zone, bb_zone, rsi14,
                   macd_hist, above_ema200, close, atr14
            FROM indicators WHERE symbol=? ORDER BY date DESC LIMIT 1
        """, (sym,)).fetchone()
        if not row: continue
        trend, rz, bz, rsi_v, mh, ab200, price, atr_v = row

        if trend == 1 and ab200 == 1 and rsi_v and rsi_v < 65:
            long_c.append((sym, price, rsi_v, atr_v))
        if trend == -1 and ab200 == 0 and rsi_v and rsi_v > 35:
            short_c.append((sym, price, rsi_v, atr_v))
        if rz == "oversold":
            long_c.append((sym+"*", price, rsi_v, atr_v))
        if rz == "overbought":
            short_c.append((sym+"*", price, rsi_v, atr_v))
        if bz == "squeeze":
            squeeze.append((sym, price, atr_v))

    print(f"\n  LONG adayları  ({len(long_c)}) — * RSI aşırı satım:")
    for s,p,r,a in long_c[:6]:
        stop = round(p - 2*(a or 0), 2)
        print(f"    {s:<12} {p:>8.2f} TL   RSI:{r:>5.1f}   Stop: {stop:.2f}")

    print(f"\n  SHORT adayları ({len(short_c)}) — * RSI aşırı alım:")
    for s,p,r,a in short_c[:6]:
        stop = round(p + 2*(a or 0), 2)
        print(f"    {s:<12} {p:>8.2f} TL   RSI:{r:>5.1f}   Stop: {stop:.2f}")

    print(f"\n  BB Sıkışması   ({len(squeeze)}) — kırılım yakın:")
    for s,p,a in squeeze:
        print(f"    {s:<12} {p:>8.2f} TL   ATR: {(a or 0):.2f}")

# ── Entry ─────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("="*68)
    print("İndikatör Hesaplayıcı v2")
    print(f"Veritabanı: {DB_PATH}")
    print("="*68)

    conn  = get_db()

    # Purge any future-dated rows from the indicators table
    today_str = _date.today().isoformat()
    cur = conn.execute("DELETE FROM indicators WHERE date > ?", (today_str,))
    conn.commit()
    if cur.rowcount:
        print(f"  [TEMIZLIK] {cur.rowcount} gelecek-tarihli indikatör satiri silindi")

    print("\nMakro veri indiriliyor (USDTRY, Brent, VIX, EEM, Gold, DXY, SP500, STOXX50)...")
    n_macro = fetch_macro_data(conn)
    print(f"  macro_data: {n_macro} satir kaydedildi (12 turetilmis ozellik dahil)")

    sentiment_scores = load_sentiment_scores()
    if sentiment_scores:
        print(f"Duygu skorlari yuklendi: {len(sentiment_scores)} sembol")
    else:
        print("Duygu skoru yok (once news_filter.py calistirin)")

    total = 0
    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i:2}/{len(SYMBOLS)}] {sym}...", end=" ", flush=True)
        n = compute(conn, sym, sentiment_scores=sentiment_scores)
        print(f"{n} satır")
        total += n

    print(f"\nToplam: {total} satır  |  Süre: {time.time()-t0:.1f}s")

    print_table(conn)
    print_signals(conn)
    conn.close()
    print("\nSonraki adım: py -3.12 backtest.py")