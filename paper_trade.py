"""
paper_trade.py — BIST Paper Trading Engine
────────────────────────────────────────────────────────────────
Kullanim: py -3.12 paper_trade.py [--run | --status | --close-all]

Strateji: XGBoost sinyalleri (güven > %68) + kural bazli filtreleme
  - Her gün çalistir: mevcut sinyalleri al, pozisyonlari güncelle
  - SQLite'a pozisyon, trade ve equity geçmisi yaz

Sermaye: 100,000 TL
Max pozisyon: 6 esit zamanli (her biri max %15 = 15,000 TL)
Stop loss: 2x ATR
Hedef: 1.2x ATR (gerçekçi hedef ~%6)
Confidence filtresi: > %60
────────────────────────────────────────────────────────────────
"""

import sqlite3, pickle, sys, time, os
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from regime_hmm import compute_historical_regime_features

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from telegram_bot import (
        send_telegram_alert,
        alert_stop_proximity,
        alert_tuprs_urgent,
        alert_signals as _tg_alert_signals,
    )
    HAS_TG = True
except Exception:
    HAS_TG = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from zoneinfo import ZoneInfo
    TZ_IST = ZoneInfo("Europe/Istanbul")
except Exception:
    TZ_IST = None

DB_PATH    = Path(__file__).parent / "trade_data.db"
MODELS     = Path(__file__).parent / "models"
MODEL_PATH = MODELS / "xgb_model.pkl"
LGB_MODEL_PATH = MODELS / "lgb_model.pkl"
CAT_MODEL_PATH = MODELS / "cat_model.pkl"
META_MODEL_PATH = MODELS / "meta_lr.pkl"
MODEL_BUNDLE_PATH = MODELS / "model_bundle.pkl"
RESULTS    = Path(__file__).parent / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CAPITAL      = 100_000
MAX_POS      = 6
MAX_POS_PCT  = 0.15       # tek pozisyon max %15
COMMISSION   = 0.002      # %0.2
SLIPPAGE     = 0.001      # %0.1
COMMISSION_ENTRY = COMMISSION / 2
COMMISSION_EXIT  = COMMISSION / 2
SLIPPAGE_ENTRY   = SLIPPAGE
SLIPPAGE_EXIT    = SLIPPAGE
ENTRY_ATR_SLIP_MULT = 0.00
EXIT_ATR_SLIP_MULT  = 0.00
USE_GAP_AWARE_EXITS = True
CONF_THRESH  = 68.0       # min %68 confidence (raised from 60 for quality)
BUY_THRESHOLD_DEFAULT  = CONF_THRESH / 100.0
SELL_THRESHOLD_DEFAULT = 1.0 - BUY_THRESHOLD_DEFAULT
ATR_STOP     = 2.0        # stop = entry - ATR_STOP * atr
ATR_TARGET   = 1.2        # hedef = entry + ATR_TARGET * atr (gerçekçi ~%6)
MIN_SIZE_TL  = 1_000

RISK_PER_TRADE_PCT = 0.01   # pozisyon basina risk: sermayenin %1'i (ATR sizing)
ATR_TRAIL_PARTIAL  = 1.0    # kismi cikis sonrasi ATR tabanli trailing carpani
TIME_STOP_DAYS     = 5      # max islem gunu: kar yoksa kapat
TIME_STOP_MIN_PCT  = 1.0    # zaman durusu icin minimum kar esigi (%)

# Watch mode
WATCH_INTERVAL     = 300    # saniye (5 dakika)
MARKET_OPEN_H      = 10
MARKET_OPEN_M      = 5
MARKET_CLOSE_H     = 17
MARKET_CLOSE_M     = 55
INTRADAY_ALERT_PCT = 1.5    # % hareket uyarisi

# ANSI renk kodlari (Windows 10+ destekler)
if os.name == "nt":
    os.system("")  # Windows ANSI aktif et
CLR_RED    = "\033[91m"
CLR_GREEN  = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_CYAN   = "\033[96m"
CLR_BOLD   = "\033[1m"
CLR_RESET  = "\033[0m"

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

SECTOR_MAP = {
    "YKBNK": "bank",         "AKBNK": "bank",
    "ISCTR": "bank",         "GARAN": "bank",
    "TUPRS": "energy",       "PETKM": "energy",
    "TAVHL": "aviation",     "FROTO": "auto",
    "TCELL": "telecom",      "ASELS": "defense",
    "BIMAS": "retail",       "MGROS": "retail",
    "ENKAI": "construction", "EKGYO": "realestate",
    # Expanded universe (added 2026-04-24)
    "THYAO": "aviation",     "EREGL": "steel",
    "KCHOL": "holding",      "SAHOL": "holding",
    "SISE":  "glass",        "TOASO": "auto",
    "ARCLK": "consumer",     "VESTL": "consumer",
    "KRDMD": "steel",
    "PGSUS": "aviation",     "ODAS":  "energy",
    "GUBRF": "chemicals",    "CIMSA": "cement",
    "LOGO":  "technology",   "NETAS": "technology",
}

STRATEGY_NAMES = {
    "ML_AL":  "XGBoost AL",
    "ML_SAT": "XGBoost SAT",
    "S1":     "MTF Momentum",
    "S2":     "RSI MeanRev",
    "S3":     "BB Breakout",
}

# ── Veritabani kurulumu ───────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_positions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT,
        strategy    TEXT,
        entry_date  TEXT,
        entry_price REAL,
        size_tl     REAL,
        is_long     INTEGER,
        stop_price  REAL,
        target_price REAL,
        confidence  REAL,
        signal_prob REAL,
        regime      TEXT,
        entry_note  TEXT,
        status      TEXT DEFAULT 'open'
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT,
        strategy    TEXT,
        entry_date  TEXT,
        exit_date   TEXT,
        entry_price REAL,
        exit_price  REAL,
        size_tl     REAL,
        is_long     INTEGER,
        pnl         REAL,
        pct_return  REAL,
        exit_reason TEXT,
        signal_prob REAL,
        regime      TEXT,
        entry_note  TEXT,
        exit_note   TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_equity (
        date            TEXT PRIMARY KEY,
        cash            REAL,
        positions_value REAL,
        total_equity    REAL,
        daily_pnl       REAL,
        open_positions  INTEGER
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        event_date  TEXT,
        event_type  TEXT,
        symbol      TEXT,
        strategy    TEXT,
        details     TEXT
    )""")
    conn.commit()
    # Sutun gecis (eski DB'ler icin — hata olursa sutun zaten vardir)
    try:
        conn.execute(
            "ALTER TABLE paper_positions ADD COLUMN partial_exit_done INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    for stmt in [
        "ALTER TABLE paper_positions ADD COLUMN signal_prob REAL",
        "ALTER TABLE paper_positions ADD COLUMN regime TEXT",
        "ALTER TABLE paper_positions ADD COLUMN entry_note TEXT",
        "ALTER TABLE paper_trades ADD COLUMN signal_prob REAL",
        "ALTER TABLE paper_trades ADD COLUMN regime TEXT",
        "ALTER TABLE paper_trades ADD COLUMN entry_note TEXT",
        "ALTER TABLE paper_trades ADD COLUMN exit_note TEXT",
    ]:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass
    return conn

# ── Mevcut nakit hesapla ──────────────────────────────────────
def get_cash(conn):
    """Nakit = sermaye - yatirilan + realize edilmis P&L"""
    row = conn.execute(
        "SELECT SUM(size_tl) FROM paper_positions WHERE status='open'"
    ).fetchone()
    invested = row[0] or 0.0
    row2 = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM paper_trades"
    ).fetchone()
    realized_pnl = row2[0] or 0.0
    return CAPITAL - invested + realized_pnl

def get_open_positions(conn):
    rows = conn.execute(
        "SELECT id, symbol, strategy, entry_date, entry_price, "
        "size_tl, is_long, stop_price, target_price, confidence, "
        "COALESCE(signal_prob, 0), COALESCE(regime, ''), COALESCE(entry_note, ''), "
        "COALESCE(partial_exit_done, 0) "
        "FROM paper_positions WHERE status='open'"
    ).fetchall()
    cols = ["id","symbol","strategy","entry_date","entry_price",
            "size_tl","is_long","stop_price","target_price","confidence",
            "signal_prob","regime","entry_note",
            "partial_exit_done"]
    return [dict(zip(cols, r)) for r in rows]


def log_event(conn, event_date, event_type, symbol=None, strategy=None, details=""):
    conn.execute(
        "INSERT INTO paper_events (event_date, event_type, symbol, strategy, details) VALUES (?,?,?,?,?)",
        (event_date, event_type, symbol, strategy, details),
    )
    conn.commit()

# ── Canli fiyat çek (watch mode için) ────────────────────────
def fetch_live_prices(symbols):
    """Guncel fiyat + gunun acilis fiyati (yfinance, BIST intraday).

    fix 2026-04-24: fast_info.last_price (batch) BIST icin acilis fiyatini
    donduruyor — gercek intraday fiyati degil. Tamamen kaldirildi.

    Yeni strateji (fast_info yok):
      1. Birincil : period="1d"/interval="1m"  — en son 1dk bar kapanis
      2. Fallback : period="2d"/interval="5m"  — eksik semboller icin
    Her iki kaynak da bugunun gercek intraday hareketini yansitiyor (max ~15dk gecikme).
    """
    import datetime as _dt

    prices, opens = {}, {}
    if not HAS_YF:
        return prices, opens

    today = _dt.date.today()

    def _extract(raw, sym_list):
        """Bir yf.download sonucundan {sym: close} ve {sym: open} cikar."""
        p, o = {}, {}
        if raw is None or raw.empty:
            return p, o
        for sym in sym_list:
            sym_is = sym + ".IS"
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df_s = raw[sym_is].dropna(how="all")
                else:
                    # Tek sembol: raw direkt o sembolun verisi
                    df_s = raw.dropna(how="all")
                if df_s.empty:
                    continue
                df_s.columns = [c.lower() for c in df_s.columns]
                # Bugünün satirlari; yoksa son mevcut seans
                today_rows = df_s[[i.date() == today for i in df_s.index]]
                session = today_rows if not today_rows.empty else df_s
                if session.empty:
                    continue
                p[sym] = float(session["close"].iloc[-1])  # son bar kapanis
                o[sym] = float(session["open"].iloc[0])    # ilk bar acilis
            except Exception:
                pass
        return p, o

    # 1. Birincil: 1dk barlar — daha guncel kapanis fiyati
    tickers_is = [s + ".IS" for s in symbols]
    try:
        raw1m = yf.download(
            tickers_is, period="1d", interval="1m",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        prices, opens = _extract(raw1m, symbols)
    except Exception:
        pass

    # 2. Fallback: 5dk barlar — 1m'de eksik kalan semboller icin
    missing = [s for s in symbols if s not in prices]
    if missing:
        missing_is = [s + ".IS" for s in missing]
        try:
            raw5m = yf.download(
                missing_is, period="2d", interval="5m",
                progress=False, auto_adjust=True, group_by="ticker",
            )
            p5, o5 = _extract(raw5m, missing)
            for sym in missing:
                if sym in p5:
                    prices[sym] = p5[sym]
                if sym not in opens and sym in o5:
                    opens[sym] = o5[sym]
        except Exception:
            pass

    return prices, opens

# ── Makro günlük degisim (Brent, USDTRY) ─────────────────────
def fetch_macro_daily_change():
    """Brent ve USDTRY günlük yüzde degisimini yfinance ile çek."""
    result = {"brent_chg": None, "usdtry_chg": None}
    if not HAS_YF:
        return result
    for ticker, key in [("BZ=F", "brent_chg"), ("USDTRY=X", "usdtry_chg")]:
        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                closes = df["close"].dropna()
                if len(closes) >= 2:
                    result[key] = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
        except Exception:
            pass
    return result

# ── Fiyat ve indikatörleri çek ────────────────────────────────
def get_latest(conn, sym):
    row = conn.execute("""
        SELECT o.open, o.high, o.low, o.close, i.atr14, i.rsi14, i.mtf_trend,
               i.above_ema200, i.macd_hist, i.bb_lower, i.bb_upper,
               i.bb_width, i.vol_ratio, o.date
        FROM ohlcv o
        JOIN indicators i ON o.symbol=i.symbol AND o.date=i.date
        WHERE o.symbol=?
        ORDER BY o.date DESC LIMIT 1
    """, (sym,)).fetchone()
    if not row:
        return None
    keys = ["open","high","low","close","atr14","rsi14","mtf_trend","above_ema200",
            "macd_hist","bb_lower","bb_upper","bb_width","vol_ratio","date"]
    return dict(zip(keys, row))

def _side_slippage_pct(price, atr, side):
    base = SLIPPAGE_ENTRY if side == "entry" else SLIPPAGE_EXIT
    atr_mult = ENTRY_ATR_SLIP_MULT if side == "entry" else EXIT_ATR_SLIP_MULT
    atr_pct = (atr / price) if price and atr else 0.0
    return max(0.0, base + atr_pct * atr_mult)

def apply_fill_price(raw_price, is_long, side, atr=0.0):
    slip_pct = _side_slippage_pct(raw_price, atr, side)
    if side == "entry":
        adverse_up = is_long
    else:
        adverse_up = not is_long
    return raw_price * (1 + slip_pct if adverse_up else 1 - slip_pct)

def calc_position_pnl(entry_price, exit_price, is_long, size_tl):
    entry_cost = size_tl * COMMISSION_ENTRY
    exit_cost  = size_tl * COMMISSION_EXIT
    if is_long:
        ret = (exit_price - entry_price) / entry_price
    else:
        ret = (entry_price - exit_price) / entry_price
    return size_tl * ret - entry_cost - exit_cost, ret * 100

def resolve_daily_exit(latest, pos):
    """Resolve gap-aware stop/target exits from daily OHLC."""
    if not latest:
        return None, None

    open_p = latest["open"]
    high_p = latest["high"]
    low_p = latest["low"]
    close_p = latest["close"]
    stop_p = pos["stop_price"]
    target_p = pos["target_price"]

    if not USE_GAP_AWARE_EXITS:
        if pos["is_long"] and close_p <= stop_p:
            return close_p, "stop_loss"
        if not pos["is_long"] and close_p >= stop_p:
            return close_p, "stop_loss"
        if pos["is_long"] and close_p >= target_p:
            return close_p, "target_hit"
        if not pos["is_long"] and close_p <= target_p:
            return close_p, "target_hit"
        return None, None

    if pos["is_long"]:
        if open_p <= stop_p:
            return open_p, "stop_gap"
        if low_p <= stop_p:
            return stop_p, "stop_loss"
        if open_p >= target_p:
            return open_p, "target_gap"
        if high_p >= target_p:
            return target_p, "target_hit"
    else:
        if open_p >= stop_p:
            return open_p, "stop_gap"
        if high_p >= stop_p:
            return stop_p, "stop_loss"
        if open_p <= target_p:
            return open_p, "target_gap"
        if low_p <= target_p:
            return target_p, "target_hit"
    return None, None

def resolve_daily_partial_exit(latest, pos):
    """Resolve half-target partial exits from daily OHLC."""
    if not latest or pos["partial_exit_done"]:
        return None, None, None

    entry = pos["entry_price"]
    target = pos["target_price"]
    halfway = entry + (target - entry) * 0.5

    open_p = latest["open"]
    high_p = latest["high"]
    low_p = latest["low"]

    if pos["is_long"]:
        if open_p >= halfway:
            return open_p, halfway, "partial_gap"
        if high_p >= halfway:
            return halfway, halfway, "partial_exit"
    else:
        if open_p <= halfway:
            return open_p, halfway, "partial_gap"
        if low_p <= halfway:
            return halfway, halfway, "partial_exit"
    return None, halfway, None

def resolve_intraday_barrier(level, is_long, cur_price, day_open, hit_type):
    """Resolve a delayed intraday barrier conservatively, honoring opening gaps."""
    if level is None:
        return None, None

    if is_long:
        if hit_type == "adverse":
            if day_open <= level:
                return day_open, "gap"
            if cur_price <= level:
                return cur_price, "touch"
        else:
            if day_open >= level:
                return day_open, "gap"
            if cur_price >= level:
                return cur_price, "touch"
    else:
        if hit_type == "adverse":
            if day_open >= level:
                return day_open, "gap"
            if cur_price >= level:
                return cur_price, "touch"
        else:
            if day_open <= level:
                return day_open, "gap"
            if cur_price <= level:
                return cur_price, "touch"
    return None, None

# ── XGBoost modeli yükle ──────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        print(f"[HATA] Model bulunamadi: {MODEL_PATH}")
        print("  Once ml_train.py calistir: py -3.12 ml_train.py")
        return None, None, None, None

    import ml_train as _ml_train_mod

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and hasattr(_ml_train_mod, name):
                return getattr(_ml_train_mod, name)
            return super().find_class(module, name)

    if MODEL_BUNDLE_PATH.exists():
        with open(MODEL_BUNDLE_PATH, "rb") as f:
            bundle = _SafeUnpickler(f).load()
        with open(MODEL_PATH, "rb") as f:
            xgb_model = _SafeUnpickler(f).load()
        lgb_model = None
        cat_model = None
        meta_lr = None
        if LGB_MODEL_PATH.exists():
            with open(LGB_MODEL_PATH, "rb") as f:
                lgb_model = _SafeUnpickler(f).load()
        if CAT_MODEL_PATH.exists():
            with open(CAT_MODEL_PATH, "rb") as f:
                cat_model = _SafeUnpickler(f).load()
        if META_MODEL_PATH.exists():
            with open(META_MODEL_PATH, "rb") as f:
                meta_lr = _SafeUnpickler(f).load()

        model = _ml_train_mod.EnsembleModel(
            xgb_model,
            lgb_model,
            cat_model,
            meta_lr=meta_lr,
            stacking_enabled=bundle.get("stacking_enabled", False),
            meta_extra_features=bundle.get("meta_extra_features", True),
        )
        return (
            model,
            bundle["features"],
            bundle.get("calibrator"),
            bundle.get("thresholds", {"buy": BUY_THRESHOLD_DEFAULT, "sell": SELL_THRESHOLD_DEFAULT}),
        )

    # Legacy fallback: old single-file bundle
    with open(MODEL_PATH, "rb") as f:
        obj = _SafeUnpickler(f).load()
    return (
        obj["model"],
        obj["features"],
        obj.get("calibrator"),
        obj.get("thresholds", {"buy": BUY_THRESHOLD_DEFAULT, "sell": SELL_THRESHOLD_DEFAULT}),
    )


def calibrated_buy_prob(model, calibrator, X):
    raw_prob = model.predict_proba(X.fillna(0))[:, 1]
    if calibrator is None:
        return np.clip(raw_prob, 0.0, 1.0)
    return np.clip(np.asarray(calibrator.predict(raw_prob), dtype=float), 0.0, 1.0)


def classify_prob(prob_buy, thresholds):
    buy_th = thresholds.get("buy", BUY_THRESHOLD_DEFAULT)
    sell_th = thresholds.get("sell", SELL_THRESHOLD_DEFAULT)
    if prob_buy >= buy_th:
        return 1, prob_buy
    if prob_buy <= sell_th:
        return -1, 1.0 - prob_buy
    return 0, max(prob_buy, 1.0 - prob_buy)

# ── Feature üret (ml_train.py ile ayni logic) ─────────────────
def make_features_single(df, sym=None, market_data=None, macro_df=None, regime_df=None):
    """Son N güne ait DataFrame'den feature vektörü üret."""
    f = pd.DataFrame(index=df.index)
    c = df["close"]

    for n in [1, 3, 5, 10, 20]:
        f[f"ret_{n}d"] = c.pct_change(n)

    f["rsi"]         = df["rsi14"] / 100
    f["rsi_delta"]   = df["rsi14"].diff(3) / 100
    f["rsi_above50"] = (df["rsi14"] > 50).astype(int)

    f["macd_hist"]   = df["macd_hist"] / (c + 1e-9)
    f["macd_cross"]  = (
        (df["macd_line"] > df["macd_signal"]) &
        (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    f["macd_above0"] = (df["macd_hist"] > 0).astype(int)

    f["ema_8_21"]    = (df["ema8"]  / df["ema21"]  - 1)
    f["ema_21_50"]   = (df["ema21"] / df["ema50"]  - 1)
    f["ema_50_200"]  = (df["ema50"] / df["ema200"] - 1)
    f["price_ema200"]= (c / df["ema200"] - 1)
    f["above_ema200"]= df["above_ema200"].astype(int)
    f["golden_cross"]= df["golden_cross"].astype(int)
    f["mtf_trend"]   = df["mtf_trend"]

    bb_range = df["bb_upper"] - df["bb_lower"]
    f["bb_pos"]    = (c - df["bb_lower"]) / (bb_range + 1e-9)
    f["bb_width"]  = df["bb_width"] / 100
    f["bb_squeeze"]= (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)).astype(int)
    f["above_upper"]= (c > df["bb_upper"]).astype(int)
    f["below_lower"]= (c < df["bb_lower"]).astype(int)

    f["atr_pct"]   = df["atr14"] / (c + 1e-9)
    f["atr_ratio"] = df["atr14"] / df["atr14"].rolling(20).mean()

    f["vol_ratio"]  = df["vol_ratio"].clip(0, 5)
    f["vol_trend"]  = df["volume"].pct_change(5)
    f["obv_slope"]  = df["obv"].diff(5) / (df["obv"].abs().rolling(5).mean() + 1e-9)

    f["high_low_pct"]= (df["high"] - df["low"]) / (c + 1e-9)
    f["close_pos"]   = (c - df["low"]) / ((df["high"] - df["low"]) + 1e-9)
    f["gap"]         = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-9)

    for n in [10, 20]:
        roll_std = c.pct_change().rolling(n).std()
        roll_ret = c.pct_change(n)
        f[f"trend_str_{n}"] = roll_ret / (roll_std * np.sqrt(n) + 1e-9)

    f["above_sma20"] = (c > df["ema21"]).astype(int)
    f["consec_up"]   = (c > c.shift(1)).astype(int).rolling(5).sum()
    f["consec_down"] = (c < c.shift(1)).astype(int).rolling(5).sum()

    # ── 11. 52-haftalık yüksek/düşük uzaklığı ──
    f["dist_52w_high"] = (c / c.rolling(252, min_periods=20).max() - 1)
    f["dist_52w_low"]  = (c / c.rolling(252, min_periods=20).min() - 1)

    # ── 12. Haftanın günü ──
    f["day_of_week"]   = df.index.dayofweek / 4.0

    # ── 13. Makro / çapraz-hisse özellikler ──
    if market_data is not None:
        tcell_close = market_data.get("TCELL")
        if tcell_close is not None:
            f["tcell_ret_1d"] = tcell_close.pct_change(1).reindex(df.index).fillna(0)
        else:
            f["tcell_ret_1d"] = 0.0

        mkt_rets = [close_s.pct_change(1).reindex(df.index)
                    for close_s in market_data.values()]
        if mkt_rets:
            mkt_ret   = pd.concat(mkt_rets, axis=1).mean(axis=1)
            stock_ret = c.pct_change(1)
            cov20     = stock_ret.rolling(20).cov(mkt_ret)
            var20     = mkt_ret.rolling(20).var()
            f["mkt_beta_20"] = (cov20 / (var20 + 1e-9)).clip(-3, 3).fillna(1.0)
        else:
            f["mkt_beta_20"] = 1.0

        my_sector = SECTOR_MAP.get(sym, "other")
        sec_rets  = [close_s.pct_change(5).reindex(df.index)
                     for s, close_s in market_data.items()
                     if s != sym and SECTOR_MAP.get(s) == my_sector]
        if sec_rets:
            f["sector_mom_5d"] = pd.concat(sec_rets, axis=1).mean(axis=1).fillna(0)
        else:
            f["sector_mom_5d"] = 0.0
    else:
        f["tcell_ret_1d"]  = 0.0
        f["mkt_beta_20"]   = 1.0
        f["sector_mom_5d"] = 0.0

    # ── Macro features (pre-computed in macro_data DB by indicators.py) ──
    if macro_df is not None and not macro_df.empty:
        import numpy as _np
        mac = macro_df.reindex(df.index).ffill()

        def _mac(col, default=0.0):
            if col in mac.columns:
                return mac[col].fillna(default)
            return _np.float64(default)

        brent_mult = 1.5 if sym in ("TUPRS", "PETKM") else 1.0
        f["usdtry_1d_return"]    = _mac("usdtry_1d_ret",      0.0)
        f["usdtry_5d_return"]    = _mac("usdtry_5d_ret",      0.0)
        f["usdtry_above_20ma"]   = _mac("usdtry_above_20ma",  0).astype(int)
        f["usdtry_20d_zscore"]   = _mac("usdtry_20d_zscore",  0.0)
        f["brent_1d_return"]     = _mac("brent_1d_ret",  0.0) * brent_mult
        f["brent_5d_return"]     = _mac("brent_5d_ret",  0.0) * brent_mult
        f["tcmb_rate"]           = _mac("tcmb_rate",          43.0)
        f["vix_level"]           = _mac("vix_level",          0.0)
        f["vix_5d_zscore"]       = _mac("vix_5d_zscore",      0.0)
        f["gold_try_ratio"]      = _mac("gold_try_ratio",     0.0)
        f["dxy_5d_ret"]          = _mac("dxy_5d_ret",         0.0)
        f["sp500_overnight_ret"] = _mac("sp500_overnight_ret", 0.0)
        f["stoxx50_am_ret"]      = _mac("stoxx50_am_ret",     0.0)
        f["em_5d_ret"]           = _mac("em_5d_ret",          0.0)
        f["macro_risk_score"]    = _mac("macro_risk_score",   0.0)
    else:
        f["usdtry_1d_return"]    = 0.0
        f["usdtry_5d_return"]    = 0.0
        f["usdtry_above_20ma"]   = 0
        f["usdtry_20d_zscore"]   = 0.0
        f["brent_1d_return"]     = 0.0
        f["brent_5d_return"]     = 0.0
        f["tcmb_rate"]           = 43.0
        f["vix_level"]           = 0.0
        f["vix_5d_zscore"]       = 0.0
        f["gold_try_ratio"]      = 0.0
        f["dxy_5d_ret"]          = 0.0
        f["sp500_overnight_ret"] = 0.0
        f["stoxx50_am_ret"]      = 0.0
        f["em_5d_ret"]           = 0.0
        f["macro_risk_score"]    = 0.0

    if regime_df is not None and not regime_df.empty:
        reg = regime_df.reindex(df.index).ffill()
        f["hmm_prob_bull"] = pd.to_numeric(reg.get("hmm_prob_bull", 1.0 / 3.0), errors="coerce").fillna(1.0 / 3.0)
        f["hmm_prob_bear"] = pd.to_numeric(reg.get("hmm_prob_bear", 1.0 / 3.0), errors="coerce").fillna(1.0 / 3.0)
        f["hmm_prob_range"] = pd.to_numeric(reg.get("hmm_prob_range", 1.0 / 3.0), errors="coerce").fillna(1.0 / 3.0)
        f["hmm_days_in_state"] = pd.to_numeric(reg.get("hmm_days_in_state", 0.0), errors="coerce").fillna(0.0)
    else:
        f["hmm_prob_bull"] = 1.0 / 3.0
        f["hmm_prob_bear"] = 1.0 / 3.0
        f["hmm_prob_range"] = 1.0 / 3.0
        f["hmm_days_in_state"] = 0.0

    return f

def get_ml_signals(conn, model, feature_names, calibrator=None, thresholds=None):
    """Tüm hisseler için ML sinyali üret."""
    FETCH_SQL = """
        SELECT o.date, o.open, o.high, o.low, o.close, o.volume,
               i.ema8, i.ema21, i.ema50, i.ema200,
               i.rsi14, i.macd_line, i.macd_signal, i.macd_hist,
               i.atr14, i.bb_upper, i.bb_mid, i.bb_lower, i.bb_width,
               i.obv, i.vol_ratio, i.mtf_trend,
               i.above_ema200, i.golden_cross
        FROM ohlcv o
        JOIN indicators i ON o.symbol=i.symbol AND o.date=i.date
        WHERE o.symbol=? ORDER BY o.date DESC LIMIT 300
    """
    # 1. Tüm hisse verilerini önceden yükle (makro özellikler için)
    all_dfs = {}
    for sym in SYMBOLS:
        df = pd.read_sql(FETCH_SQL, conn, params=(sym,))
        if df.empty or len(df) < 30:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        all_dfs[sym] = df

    # 2. Piyasa verisi (çapraz-hisse özellikler için)
    market_data = {sym: df["close"] for sym, df in all_dfs.items()}
    regime_df = compute_historical_regime_features(DB_PATH)

    # 3. Her hisse için sinyal üret
    signals = []
    for sym, df in all_dfs.items():
        X = make_features_single(df, sym=sym, market_data=market_data, regime_df=regime_df)
        last = X.fillna(0).iloc[[-1]]
        for feat in feature_names:
            if feat not in last.columns:
                last[feat] = 0
        last = last[feature_names]

        prob_buy = calibrated_buy_prob(model, calibrator, last)[0]
        pred, conf_prob = classify_prob(
            prob_buy,
            thresholds or {"buy": BUY_THRESHOLD_DEFAULT, "sell": SELL_THRESHOLD_DEFAULT},
        )
        conf = conf_prob * 100

        latest = get_latest(conn, sym)
        if not latest:
            continue

        signals.append({
            "symbol":     sym,
            "signal":     pred,
            "confidence": conf,
            "prob_buy":   prob_buy,
            "price":      latest["close"],
            "atr":        latest["atr14"] or 0,
            "rsi":        latest["rsi14"] or 50,
            "mtf_trend":  latest["mtf_trend"] or 0,
            "data_date":  latest["date"],
        })
    return signals

# ── Pozisyon kapat ────────────────────────────────────────────
def close_position(conn, pos, exit_price, reason, today_str, exit_note=None):
    pnl, pct = calc_position_pnl(
        pos["entry_price"], exit_price, pos["is_long"], pos["size_tl"]
    )

    conn.execute("""
        INSERT INTO paper_trades
        (symbol, strategy, entry_date, exit_date, entry_price,
         exit_price, size_tl, is_long, pnl, pct_return, exit_reason,
         signal_prob, regime, entry_note, exit_note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (pos["symbol"], pos["strategy"], pos["entry_date"], today_str,
          pos["entry_price"], exit_price, pos["size_tl"],
          pos["is_long"], pnl, pct, reason,
          pos.get("signal_prob"), pos.get("regime"), pos.get("entry_note"),
          exit_note or reason))
    conn.execute(
        "UPDATE paper_positions SET status='closed' WHERE id=?",
        (pos["id"],))
    conn.commit()
    log_event(
        conn, today_str, "EXIT", pos["symbol"], pos["strategy"],
        exit_note or f"{reason} @ {exit_price:.2f} pnl {pnl:+.0f} TL"
    )
    return pnl

# ── Pozisyonun %50'sini kapat ────────────────────────────────
def partial_close_position(conn, pos, exit_price, today_str):
    """Yarı-hedef seviyesinde pozisyonun %50'sini kapat.
    - paper_trades'e half_size ile yeni kayit ekler (exit_reason='partial_exit')
    - paper_positions'da size_tl yarıya iner, stop breakeven'a tasınır,
      partial_exit_done=1 olarak isaretlenir.
    Donus: realize edilen P&L (TL)
    """
    half_size = pos["size_tl"] / 2.0
    pnl, pct = calc_position_pnl(
        pos["entry_price"], exit_price, pos["is_long"], half_size
    )

    conn.execute("""
        INSERT INTO paper_trades
        (symbol, strategy, entry_date, exit_date, entry_price,
         exit_price, size_tl, is_long, pnl, pct_return, exit_reason,
         signal_prob, regime, entry_note, exit_note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (pos["symbol"], pos["strategy"], pos["entry_date"], today_str,
          pos["entry_price"], exit_price, half_size,
          pos["is_long"], pnl, pct, "partial_exit",
          pos.get("signal_prob"), pos.get("regime"), pos.get("entry_note"),
          "Half target reached; remaining size kept with breakeven stop"))

    new_stop = round(pos["entry_price"], 2)
    conn.execute("""
        UPDATE paper_positions
        SET size_tl=?, stop_price=?, partial_exit_done=1
        WHERE id=?
    """, (half_size, new_stop, pos["id"]))
    conn.commit()
    log_event(
        conn, today_str, "PARTIAL_EXIT", pos["symbol"], pos["strategy"],
        f"Half target reached @ {exit_price:.2f}; stop moved to breakeven {new_stop:.2f}"
    )
    return pnl

# ── Yeni pozisyon aç ──────────────────────────────────────────
def open_position(conn, symbol, strategy, price, is_long, size_tl,
                  stop_price, target_price, confidence, today_str,
                  signal_prob=0.0, regime="", entry_note=""):
    conn.execute("""
        INSERT INTO paper_positions
        (symbol, strategy, entry_date, entry_price, size_tl,
         is_long, stop_price, target_price, confidence, signal_prob,
         regime, entry_note, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'open')
    """, (symbol, strategy, today_str, price, size_tl,
          1 if is_long else 0, stop_price, target_price, confidence,
          signal_prob, regime, entry_note))
    conn.commit()
    log_event(conn, today_str, "ENTRY", symbol, strategy, entry_note)

# ── Mevcut pozisyon hedeflerini yeniden hesapla ───────────────
def recalculate_open_targets(conn):
    """ATR sabiti değiştiğinde açık pozisyon hedeflerini güncelle.
    ATR, mevcut stop_price'dan geri türetilir: atr = |entry - stop| / ATR_STOP
    """
    open_pos = get_open_positions(conn)
    updated = 0
    for pos in open_pos:
        entry = pos["entry_price"]
        stop  = pos["stop_price"]
        # ATR = |entry - stop| / ATR_STOP
        atr_implied = abs(entry - stop) / ATR_STOP
        if atr_implied <= 0:
            continue
        if pos["is_long"]:
            new_target = entry + ATR_TARGET * atr_implied
        else:
            new_target = entry - ATR_TARGET * atr_implied
        conn.execute(
            "UPDATE paper_positions SET target_price=? WHERE id=?",
            (round(new_target, 2), pos["id"])
        )
        updated += 1
    conn.commit()
    if updated:
        print(f"  {updated} pozisyonun hedef fiyati guncellendi "
              f"(ATR×{ATR_TARGET} formülü)")

# ── Günlük kosu ───────────────────────────────────────────────
def run_daily(conn, model, feature_names, calibrator=None, thresholds=None):
    today_str = date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"Paper Trade Gunu: {today_str}")
    print(f"{'='*60}")

    # Makro verileri bir kez çek (sektor filtresi için)
    print("Makro veriler aliniyor (Brent, USDTRY)...", end=" ", flush=True)
    macro = fetch_macro_daily_change()
    brent_chg   = macro["brent_chg"]
    usdtry_chg  = macro["usdtry_chg"]
    brent_str  = f"{brent_chg:+.2f}%" if brent_chg is not None else "N/A"
    usdtry_str = f"{usdtry_chg:+.2f}%" if usdtry_chg is not None else "N/A"
    print(f"Brent:{brent_str}  USDTRY:{usdtry_str}")

    # Hedef fiyatları yeni ATR_TARGET ile güncelle
    recalculate_open_targets(conn)

    # ── 1. Acik pozisyonlari güncelle (stop / hedef / ML_exit) ──
    open_pos = get_open_positions(conn)
    closed_today = []
    total_closed_pnl = 0

    print(f"\n[1] Acik Pozisyon Kontrolu ({len(open_pos)} pozisyon)")
    for pos in open_pos:
        sym    = pos["symbol"]
        latest = get_latest(conn, sym)
        if not latest:
            continue
        price = latest["close"]
        atr   = latest["atr14"] or 0

        # P&L hesapla (trailing + zaman durusu icin)
        entry = pos["entry_price"]
        if pos["is_long"]:
            unrealized_pct = (price - entry) / entry * 100
        else:
            unrealized_pct = (entry - price) / entry * 100

        # ── Trailing stop ────────────────────────────────────
        # Kismi cikis yapıldıysa: ATR×1.0 dinamik trail
        # Yapılmadıysa: sabit esikler (breakeven @+2%, +1% @+4%)
        if pos["partial_exit_done"] and atr > 0:
            if pos["is_long"]:
                new_stop = round(price - ATR_TRAIL_PARTIAL * atr, 2)
            else:
                new_stop = round(price + ATR_TRAIL_PARTIAL * atr, 2)
            trail_label = f"ATR×{ATR_TRAIL_PARTIAL} trail"
        elif unrealized_pct > 4.0:
            new_stop    = round(entry * (1.01 if pos["is_long"] else 0.99), 2)
            trail_label = "trail +1%"
        elif unrealized_pct > 2.0:
            new_stop    = round(entry, 2)
            trail_label = "breakeven"
        else:
            new_stop = None

        if new_stop is not None:
            if pos["is_long"] and new_stop > pos["stop_price"]:
                conn.execute("UPDATE paper_positions SET stop_price=? WHERE id=?",
                             (new_stop, pos["id"]))
                conn.commit()
                print(f"  [TRAIL] {sym} LONG: stop "
                      f"{pos['stop_price']:.2f} -> {new_stop:.2f} ({trail_label})")
                pos["stop_price"] = new_stop
            elif not pos["is_long"] and new_stop < pos["stop_price"]:
                conn.execute("UPDATE paper_positions SET stop_price=? WHERE id=?",
                             (new_stop, pos["id"]))
                conn.commit()
                print(f"  [TRAIL] {sym} SHORT: stop "
                      f"{pos['stop_price']:.2f} -> {new_stop:.2f} ({trail_label})")
                pos["stop_price"] = new_stop

        # ── Kismi cikis: hedefin %50'sine ulasildiysa ────────
        if not pos["partial_exit_done"]:
            partial_raw_px, halfway, partial_reason = resolve_daily_partial_exit(latest, pos)
            if partial_raw_px is not None:
                partial_fill_px = apply_fill_price(
                    partial_raw_px, pos["is_long"], "exit", atr
                )
                partial_pnl = partial_close_position(conn, pos, partial_fill_px, today_str)
                total_closed_pnl += partial_pnl
                dir_str = "LONG" if pos["is_long"] else "SHORT"
                print(f"  [KISMI CIKIS] {sym} {dir_str}  "
                      f"Yari-hedef:{halfway:.2f}  Fiyat:{partial_fill_px:.2f}  "
                      f"P&L:{partial_pnl:+,.0f} TL  "
                      f"(Kalan %50 breakeven stop ile devam)")
                closed_today.append((sym, partial_pnl, partial_reason))
                # Yerel kopyayi guncelle (devam eden stop check icin)
                pos["size_tl"]          /= 2
                pos["stop_price"]        = round(entry, 2)
                pos["partial_exit_done"] = 1

        # ── Zaman durdurmasi: >5 islem gunu, kar <%1 ─────────
        entry_dt     = date.fromisoformat(pos["entry_date"])
        delta_days   = (date.today() - entry_dt).days
        trading_days = sum(1 for i in range(delta_days)
                           if (entry_dt + timedelta(days=i + 1)).weekday() < 5)

        reason = None

        raw_exit, barrier_reason = resolve_daily_exit(latest, pos)
        if raw_exit is not None:
            reason = barrier_reason
        elif (trading_days > TIME_STOP_DAYS
              and abs(unrealized_pct) < TIME_STOP_MIN_PCT):
            reason = "time_stop"

        if reason:
            raw_px = raw_exit if raw_exit is not None else price
            fill_px = apply_fill_price(raw_px, pos["is_long"], "exit", atr)
            exit_note = (
                f"Exit via {reason}; entry {pos['entry_price']:.2f}, "
                f"fill {fill_px:.2f}, stop {pos['stop_price']:.2f}, "
                f"target {pos['target_price']:.2f}"
            )
            pnl = close_position(conn, pos, fill_px, reason, today_str, exit_note)
            total_closed_pnl += pnl
            closed_today.append((sym, pnl, reason))
            dir_str = "LONG" if pos["is_long"] else "SHORT"
            pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
            print(f"  KAPANDI: {sym} {dir_str}  {reason}  "
                  f"Giris:{pos['entry_price']:.2f} Cikis:{fill_px:.2f}  "
                  f"P&L: {pnl_str} TL")
        else:
            dir_str = "LONG" if pos["is_long"] else "SHORT"
            unrealized = (price / pos["entry_price"] - 1) * 100
            if not pos["is_long"]:
                unrealized *= -1
            print(f"  ACIK: {sym} {dir_str}  "
                  f"Giris:{pos['entry_price']:.2f}  "
                  f"Simdiki:{price:.2f}  "
                  f"Gerçeklesm.: {unrealized:+.1f}%  "
                  f"Stop:{pos['stop_price']:.2f}")

    # ── 2. Yeni sinyalleri al ────────────────────────────────
    print(f"\n[2] ML Sinyal Taramasi")
    thresholds = thresholds or {"buy": BUY_THRESHOLD_DEFAULT, "sell": SELL_THRESHOLD_DEFAULT}
    signals = get_ml_signals(conn, model, feature_names, calibrator, thresholds)
    if HAS_TG:
        _tg_alert_signals(signals)

    open_pos = get_open_positions(conn)   # kapatmalardan sonra güncelle
    open_syms = {p["symbol"] for p in open_pos}
    cash = get_cash(conn)

    print(f"  Mevcut nakit: {cash:,.0f} TL  |  "
          f"Acik pozisyon: {len(open_pos)}/{MAX_POS}")

    new_positions = 0
    new_open_pos  = []  # bu seans açilan pozisyonlar (korelasyon için)
    for sig in sorted(signals, key=lambda x: x["confidence"], reverse=True):
        sym  = sig["symbol"]
        conf = sig["confidence"]
        pred = sig["signal"]
        price= sig["price"]
        atr  = sig["atr"]

        if sym in open_syms:
            continue
        if len(open_pos) + new_positions >= MAX_POS:
            break
        if pred == 0:
            continue
        if atr <= 0:
            continue

        is_long = (pred == 1)
        sector  = SECTOR_MAP.get(sym, "other")

        # ── Sektor-makro filtresi ──────────────────────────────
        if not is_long and sector == "energy":
            if brent_chg is not None and brent_chg > 0:
                print(f"  MACRO BLOCK: {sym} SHORT engellendi — "
                      f"Brent yükseliyor ({brent_chg:+.2f}%)")
                log_event(conn, today_str, "BLOCK", sym, "ML_SAT",
                          f"Macro block: Brent {brent_chg:+.2f}% against energy short")
                continue
        if not is_long and sector == "bank":
            if usdtry_chg is not None and usdtry_chg < 0:
                print(f"  MACRO BLOCK: {sym} SHORT engellendi — "
                      f"USDTRY düsiyor, TL güçleniyor ({usdtry_chg:+.2f}%)")
                log_event(conn, today_str, "BLOCK", sym, "ML_SAT",
                          f"Macro block: USDTRY {usdtry_chg:+.2f}% against bank short")
                continue

        # ── Korelasyon / sektör limiti ─────────────────────────
        all_pos   = open_pos + new_open_pos
        sec_count = sum(1 for p in all_pos
                        if SECTOR_MAP.get(p["symbol"], "other") == sector)
        lng_count = sum(1 for p in all_pos if p["is_long"])

        if sec_count >= 2:
            print(f"  KOREL BLOCK: {sym} — {sector} sektorunde max 2 siniri "
                  f"({sec_count} acik)")
            log_event(conn, today_str, "BLOCK", sym, None,
                      f"Sector exposure limit: {sector} already {sec_count} open")
            continue
        if is_long and lng_count >= 4:
            print(f"  KOREL BLOCK: {sym} — max 4 LONG siniri doldu "
                  f"({lng_count} LONG acik)")
            log_event(conn, today_str, "BLOCK", sym, None,
                      f"Long exposure limit reached: {lng_count} long positions")
            continue
        if sector == "energy" and any(p["symbol"] == "TUPRS" for p in all_pos):
            print(f"  KOREL BLOCK: {sym} — TUPRS zaten acik, "
                  f"ikinci enerji pozisyonu engellendi")
            log_event(conn, today_str, "BLOCK", sym, None,
                      "Energy concentration block: TUPRS already open")
            continue

        # ── ATR volatilite bazli pozisyon boyutu (1% risk) ────
        # risk_tl = 1,000 TL  |  stop_dist = ATR × 2  |  size = (risk/stop)*price
        risk_tl   = CAPITAL * RISK_PER_TRADE_PCT        # 1,000 TL
        stop_dist = ATR_STOP * atr                      # fiyat birimi cinsinden
        shares    = risk_tl / stop_dist if stop_dist > 0 else 0
        size_tl   = shares * price
        size_tl   = min(size_tl, CAPITAL * MAX_POS_PCT) # max %15 = 15,000 TL
        size_tl   = max(0, size_tl)
        if size_tl < MIN_SIZE_TL or cash < size_tl:
            log_event(conn, today_str, "BLOCK", sym, None,
                      f"Sizing/cash block: size {size_tl:,.0f} TL, cash {cash:,.0f} TL")
            continue

        entry_fill = apply_fill_price(price, is_long, "entry", atr)
        stop   = entry_fill - ATR_STOP   * atr if is_long else entry_fill + ATR_STOP   * atr
        target = entry_fill + ATR_TARGET * atr if is_long else entry_fill - ATR_TARGET * atr

        # Kural bazli ek filtre: AL icin RSI < 70, SAT icin RSI > 30
        rsi = sig["rsi"]
        if is_long  and rsi > 72:
            log_event(conn, today_str, "BLOCK", sym, None,
                      f"RSI filter block: long RSI {rsi:.1f} > 72")
            continue
        if not is_long and rsi < 28:
            log_event(conn, today_str, "BLOCK", sym, None,
                      f"RSI filter block: short RSI {rsi:.1f} < 28")
            continue

        strategy = "ML_AL" if is_long else "ML_SAT"
        regime = "YUKARI" if sig["mtf_trend"] == 1 else "ASAGI" if sig["mtf_trend"] == -1 else "YATAY"
        prob_buy = sig.get("prob_buy", 0.5)
        entry_note = (
            f"Entry via {'BUY' if is_long else 'SELL'} signal; "
            f"p_buy={prob_buy:.3f}, conf={conf:.1f}%, rsi={rsi:.1f}, "
            f"regime={regime}, atr={atr:.2f}, size={size_tl:,.0f} TL, "
            f"stop={stop:.2f}, target={target:.2f}"
        )
        open_position(conn, sym, strategy, entry_fill, is_long,
                      size_tl, stop, target, conf, today_str,
                      signal_prob=prob_buy, regime=regime, entry_note=entry_note)

        dir_str = "LONG" if is_long else "SHORT"
        print(f"  YENi POZISYON: {sym} {dir_str}  "
              f"Giris:{entry_fill:.2f}  Boyut:{size_tl:,.0f} TL  "
              f"Güven:{conf:.1f}%  Stop:{stop:.2f}  Hedef:{target:.2f}  "
              f"(ATR:{atr:.2f}, risk:1%)")
        open_syms.add(sym)
        new_open_pos.append({"symbol": sym, "is_long": is_long})
        cash -= size_tl
        new_positions += 1

    if new_positions == 0:
        print("  Bugun sinyal yok (filtreden gecen yok veya pozisyon dolu)")
        log_event(conn, today_str, "INFO", None, None,
                  "No new positions opened today after filters and limits")

    # ── 3. Equity hesapla ────────────────────────────────────
    open_pos = get_open_positions(conn)
    pos_value = 0
    for pos in open_pos:
        latest = get_latest(conn, pos["symbol"])
        if latest:
            price = latest["close"]
            if pos["is_long"]:
                pos_value += pos["size_tl"] * (price / pos["entry_price"])
            else:
                pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]
                pos_value += pos["size_tl"] * (1 + pnl_pct)
        else:
            pos_value += pos["size_tl"]

    cash_now = get_cash(conn)
    total_equity = cash_now + pos_value

    # Önceki günden günlük P&L
    prev = conn.execute(
        "SELECT total_equity FROM paper_equity ORDER BY date DESC LIMIT 1"
    ).fetchone()
    daily_pnl = total_equity - (prev[0] if prev else CAPITAL)

    conn.execute("""
        INSERT OR REPLACE INTO paper_equity
        (date, cash, positions_value, total_equity, daily_pnl, open_positions)
        VALUES (?,?,?,?,?,?)
    """, (today_str, cash_now, pos_value, total_equity, daily_pnl, len(open_pos)))
    conn.commit()

    total_pnl = total_equity - CAPITAL
    print(f"\n{'='*60}")
    print(f"PORTFOY OZETI — {today_str}")
    print(f"  Baslangic sermayesi : {CAPITAL:>12,.0f} TL")
    print(f"  Nakit               : {cash_now:>12,.0f} TL")
    print(f"  Pozisyon degeri     : {pos_value:>12,.0f} TL")
    print(f"  Toplam equity       : {total_equity:>12,.0f} TL")
    print(f"  Gunlük P&L          : {daily_pnl:>+12,.0f} TL")
    print(f"  Toplam P&L          : {total_pnl:>+12,.0f} TL  "
          f"({total_pnl/CAPITAL*100:+.2f}%)")
    print(f"  Kapanan bugün       : {len(closed_today)} trade")
    if closed_today:
        for sym, pnl, reason in closed_today:
            print(f"    {sym}  P&L: {pnl:+,.0f} TL  ({reason})")
    print(f"{'='*60}")

# ── Durum raporu ──────────────────────────────────────────────
def print_status(conn):
    print(f"\n{'='*65}")
    print("PAPER TRADE DURUMU")
    print(f"{'='*65}")

    # Acik pozisyonlar
    open_pos = get_open_positions(conn)
    print(f"\nAcik Pozisyonlar ({len(open_pos)}):")
    live_pos_value = 0
    if open_pos:
        print(f"  {'SEM':<8} {'YON':<6} {'GIRIS':>8} {'SIMDI':>8} "
              f"{'BOYUT':>8} {'P&L TL':>9} {'P&L%':>6} {'STOP':>8} {'HEDEF':>8}")
        print("  " + "-"*80)
        for pos in open_pos:
            latest = get_latest(conn, pos["symbol"])
            cur_price = latest["close"] if latest else pos["entry_price"]
            entry     = pos["entry_price"]
            size_tl   = pos["size_tl"]

            # Dogru P&L hesabi
            if pos["is_long"]:
                pnl_tl  = (cur_price - entry) / entry * size_tl
                pnl_pct = (cur_price - entry) / entry * 100
                pos_cur_val = size_tl * (cur_price / entry)
            else:
                pnl_tl  = (entry - cur_price) / entry * size_tl
                pnl_pct = (entry - cur_price) / entry * 100
                pos_cur_val = size_tl * (1 + (entry - cur_price) / entry)
            live_pos_value += pos_cur_val

            dir_str = "LONG" if pos["is_long"] else "SHORT"
            pnl_col = CLR_GREEN if pnl_tl >= 0 else CLR_RED
            print(f"  {pos['symbol']:<8} {dir_str:<6} {entry:>8.2f} {cur_price:>8.2f} "
                  f"{size_tl:>8,.0f} "
                  f"{pnl_col}{pnl_tl:>+9,.0f}{CLR_RESET} "
                  f"{pnl_col}{pnl_pct:>+5.1f}%{CLR_RESET} "
                  f"{pos['stop_price']:>8.2f} {pos['target_price']:>8.2f}")

            # Stop yakinlik uyarisi (>%50 stop'a gitmis)
            if pos["is_long"]:
                stop_dist    = entry - pos["stop_price"]
                toward_stop  = entry - cur_price
            else:
                stop_dist    = pos["stop_price"] - entry
                toward_stop  = cur_price - entry

            if stop_dist > 0:
                proximity = toward_stop / stop_dist
                if proximity > 0.50:
                    pct_toward = proximity * 100
                    print(f"  {CLR_YELLOW}  [!] STOP YAKINI — stop'a %{pct_toward:.0f} gidildi"
                          f" ({pos['symbol']} {dir_str}){CLR_RESET}")
    else:
        print("  (acik pozisyon yok)")

    # Telegram: stop proximity + TUPRS urgent
    if HAS_TG and open_pos:
        alert_stop_proximity(open_pos, threshold_pct=50.0)
        alert_tuprs_urgent(open_pos)

    # Canli equity hesapla (paper_equity tablosuna bagli degil)
    live_cash       = get_cash(conn)
    live_equity     = live_cash + live_pos_value
    live_total_pnl  = live_equity - CAPITAL
    realized_row    = conn.execute(
        "SELECT COALESCE(SUM(pnl),0) FROM paper_trades").fetchone()
    realized_pnl    = realized_row[0] or 0.0
    unrealized_pnl  = live_pos_value - sum(
        p["size_tl"] for p in open_pos) if open_pos else 0

    print(f"\nPortfoy Ozeti (canli hesap):")
    print(f"  Baslangic sermayesi : {CAPITAL:>12,.0f} TL")
    print(f"  Nakit (realize+kalan): {live_cash:>12,.0f} TL")
    print(f"  Acik pozisyon degeri: {live_pos_value:>12,.0f} TL")
    print(f"  Realize edilmis P&L : {realized_pnl:>+12,.0f} TL")
    print(f"  Realize edilmemis   : {unrealized_pnl:>+12,.0f} TL")
    eq_col = CLR_GREEN if live_total_pnl >= 0 else CLR_RED
    print(f"  Toplam Equity       : {live_equity:>12,.0f} TL")
    print(f"  Toplam P&L          : {eq_col}{live_total_pnl:>+12,.0f} TL  "
          f"({live_total_pnl/CAPITAL*100:>+.2f}%){CLR_RESET}")

    # Son 10 trade
    trades = conn.execute("""
        SELECT symbol, strategy, entry_date, exit_date,
               entry_price, exit_price, size_tl, is_long,
               pnl, pct_return, exit_reason
        FROM paper_trades ORDER BY exit_date DESC LIMIT 10
    """).fetchall()
    print(f"\nSon 10 Trade:")
    if trades:
        print(f"  {'SEM':<8} {'STRATEJI':<12} {'GIRIS':>10} {'CIKIS':>10} "
              f"{'YON':<6} {'P&L':>9} {'%':>7} {'NEDEN':<12}")
        print("  " + "-"*72)
        for t in trades:
            sym, strat, ed, xd, ep, xp, sz, lon, pnl, pct, reason = t
            dir_s = "LONG" if lon else "SHORT"
            pnl_c = CLR_GREEN if pnl >= 0 else CLR_RED
            print(f"  {sym:<8} {strat:<12} {ed:>10} {xd:>10} "
                  f"{dir_s:<6} {pnl_c}{pnl:>+9,.0f}{CLR_RESET} "
                  f"{pnl_c}{pct:>+6.1f}%{CLR_RESET} {reason:<12}")
    else:
        print("  (henuz kapanan trade yok)")

    # Strateji bazli P&L
    strat_pnl = conn.execute("""
        SELECT strategy, COUNT(*) trades,
               SUM(pnl) total_pnl,
               SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins,
               AVG(pct_return) avg_pct
        FROM paper_trades GROUP BY strategy
    """).fetchall()
    if strat_pnl:
        print(f"\nStrateji Bazli (kapali tradeler):")
        print(f"  {'STRATEJI':<14} {'TRADE':>6} {'KAZANMA%':>9} "
              f"{'TOPLAM P&L':>12} {'ORT %':>7}")
        print("  " + "-"*52)
        for row in strat_pnl:
            strat, n, tp, wins, avg_pct = row
            wr = wins/n*100 if n else 0
            print(f"  {strat:<14} {n:>6}  {wr:>8.0f}% "
                  f"{tp:>+12,.0f} TL {avg_pct:>+6.1f}%")
    print(f"\n{'='*65}")

# ── Intraday izleme (watch) modu ─────────────────────────────
def watch_mode(conn):
    """Her 5 dakikada bir pozisyonlari izle, stop/hedef kontrolü yap."""
    today_str = date.today().isoformat()
    log_path  = RESULTS / f"intraday_{today_str}.txt"

    def log(msg):
        now_s = datetime.now(TZ_IST).strftime("%H:%M:%S") if TZ_IST else \
                datetime.now().strftime("%H:%M:%S")
        line = f"[{now_s}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def is_market_open():
        now = datetime.now(TZ_IST) if TZ_IST else datetime.now()
        ot  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
        ct  = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
        return ot <= now <= ct

    def secs_to_open():
        now = datetime.now(TZ_IST) if TZ_IST else datetime.now()
        ot  = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
        diff = (ot - now).total_seconds()
        return max(0, diff)

    print(f"\n{'='*65}")
    print(f"INTRADAY IZLEME — {today_str}")
    print(f"Aralik: {MARKET_OPEN_H:02d}:{MARKET_OPEN_M:02d}–"
          f"{MARKET_CLOSE_H:02d}:{MARKET_CLOSE_M:02d} Istanbul")
    print(f"Guncelleme: {WATCH_INTERVAL//60} dakikada bir  |  Log: {log_path}")
    print(f"Cikis: Ctrl+C")
    print(f"{'='*65}\n")

    try:
        while True:
            now = datetime.now(TZ_IST) if TZ_IST else datetime.now()

            if secs_to_open() > 0:
                wait = int(secs_to_open())
                print(f"Borsa acilisi bekleniyor... {wait//60} dakika {wait%60} saniye")
                time.sleep(min(60, wait + 1))
                continue

            if not is_market_open():
                print("Borsa kapandi. Izleme tamamlandi.")
                log("Izleme borsa kapanisiyla sona erdi.")
                break

            open_pos = get_open_positions(conn)
            if not open_pos:
                log("Acik pozisyon yok, bekleniyor...")
                time.sleep(WATCH_INTERVAL)
                continue

            syms = [p["symbol"] for p in open_pos]
            log(f"Fiyatlar guncelleniyor... ({len(syms)} sembol, yfinance 15dk gecikmeli)")
            live_px, day_opens = fetch_live_prices(syms)

            # Ekrania temizle ve tabloyu yeniden ciz
            os.system("cls" if os.name == "nt" else "clear")
            ts_now = now.strftime("%H:%M:%S")
            print(f"INTRADAY IZLEME  {today_str} {ts_now}  (yfinance ~15dk gecikmeli)")
            print("="*75)

            closed_in_loop = []
            total_unreal   = 0.0
            rows_to_print  = []

            for pos in open_pos:
                sym    = pos["symbol"]
                entry  = pos["entry_price"]
                size   = pos["size_tl"]
                is_lng = pos["is_long"]
                cur    = live_px.get(sym) or (get_latest(conn, sym) or {}).get("close", entry)

                # P&L
                if is_lng:
                    pnl_tl  = (cur - entry) / entry * size
                    pnl_pct = (cur - entry) / entry * 100
                else:
                    pnl_tl  = (entry - cur) / entry * size
                    pnl_pct = (entry - cur) / entry * 100
                total_unreal += pnl_tl

                # Intragün hareket (güne göre)
                day_open   = day_opens.get(sym, entry)
                intra_pct  = (cur - day_open) / day_open * 100 if day_open else 0.0

                # Trailing stop + kismi cikis (watch mode)
                if is_lng:
                    unreal_pct_w = (cur - entry) / entry * 100
                else:
                    unreal_pct_w = (entry - cur) / entry * 100

                # ATR: DB'den çek (ATR trail ve kismi cikis icin gerekli)
                latest_w = get_latest(conn, sym)
                atr_w    = (latest_w["atr14"] or 0) if latest_w else 0

                if pos["partial_exit_done"] and atr_w > 0:
                    new_stop_w  = (round(cur - ATR_TRAIL_PARTIAL * atr_w, 2)
                                   if is_lng else
                                   round(cur + ATR_TRAIL_PARTIAL * atr_w, 2))
                    trail_lbl_w = f"ATR×{ATR_TRAIL_PARTIAL} trail"
                elif unreal_pct_w > 4.0:
                    new_stop_w  = round(entry * (1.01 if is_lng else 0.99), 2)
                    trail_lbl_w = "trail +1%"
                elif unreal_pct_w > 2.0:
                    new_stop_w  = round(entry, 2)
                    trail_lbl_w = "breakeven"
                else:
                    new_stop_w = None

                if new_stop_w is not None:
                    if is_lng and new_stop_w > pos["stop_price"]:
                        conn.execute(
                            "UPDATE paper_positions SET stop_price=? WHERE id=?",
                            (new_stop_w, pos["id"]))
                        conn.commit()
                        log(f"[TRAIL] {sym} LONG: stop "
                            f"{pos['stop_price']:.2f} -> {new_stop_w:.2f} ({trail_lbl_w})")
                        pos = dict(pos); pos["stop_price"] = new_stop_w
                    elif not is_lng and new_stop_w < pos["stop_price"]:
                        conn.execute(
                            "UPDATE paper_positions SET stop_price=? WHERE id=?",
                            (new_stop_w, pos["id"]))
                        conn.commit()
                        log(f"[TRAIL] {sym} SHORT: stop "
                            f"{pos['stop_price']:.2f} -> {new_stop_w:.2f} ({trail_lbl_w})")
                        pos = dict(pos); pos["stop_price"] = new_stop_w

                # Kismi cikis kontrolü (watch mode'da da tetiklenebilir)
                if not pos["partial_exit_done"]:
                    halfway_w = entry + (pos["target_price"] - entry) * 0.5
                    partial_raw_w, partial_hit_kind = resolve_intraday_barrier(
                        halfway_w, is_lng, cur, day_open, "favorable"
                    )

                    if partial_raw_w is not None:
                        fill_cur = apply_fill_price(partial_raw_w, is_lng, "exit", atr_w)
                        p_pnl = partial_close_position(conn, pos, fill_cur, today_str)
                        msg_tag = "partial_gap" if partial_hit_kind == "gap" else "partial_exit"
                        msg   = (f"[KISMI CIKIS] {sym} {'LONG' if is_lng else 'SHORT'} "
                                 f"yari-hedef:{halfway_w:.2f} fiyat:{fill_cur:.2f} "
                                 f"P&L:{p_pnl:+.0f} TL ({msg_tag}, kalan %50 breakeven)")
                        log(msg)
                        closed_in_loop.append(msg)
                        pos = dict(pos)
                        pos["size_tl"]          /= 2
                        pos["stop_price"]        = round(entry, 2)
                        pos["partial_exit_done"] = 1

                # Stop yakinligi
                if is_lng:
                    stop_dist   = entry - pos["stop_price"]
                    toward_stop = entry - cur
                else:
                    stop_dist   = pos["stop_price"] - entry
                    toward_stop = cur - entry
                proximity = (toward_stop / stop_dist * 100) if stop_dist > 0 else 0

                # Durum etiketleri
                flags = ""
                if abs(intra_pct) >= INTRADAY_ALERT_PCT:
                    flags += f" {CLR_YELLOW}[!]{abs(intra_pct):.1f}% INTRAGUN{CLR_RESET}"
                if proximity > 50:
                    flags += f" {CLR_RED}[!] STOP YAKINI %{proximity:.0f}{CLR_RESET}"
                    if HAS_TG:
                        alert_stop_proximity([pos], threshold_pct=50.0)
                        if sym == "TUPRS" and not is_lng:
                            alert_tuprs_urgent([pos])

                dir_s   = "LONG" if is_lng else "SHORT"
                pc      = CLR_GREEN if pnl_tl >= 0 else CLR_RED
                ic      = CLR_GREEN if intra_pct >= 0 else CLR_RED
                rows_to_print.append(
                    f"{CLR_BOLD}{sym:<8}{CLR_RESET} {dir_s:<6} "
                    f"G:{entry:>8.2f} S:{cur:>8.2f}  "
                    f"P&L:{pc}{pnl_tl:>+8.0f}TL({pnl_pct:>+.1f}%){CLR_RESET}  "
                    f"Gun:{ic}{intra_pct:>+.1f}%{CLR_RESET}{flags}"
                )

                # Önemli hareket logu
                if abs(intra_pct) >= INTRADAY_ALERT_PCT:
                    log(f"[ALERT] {sym} {dir_s} intragün {intra_pct:+.2f}% | "
                        f"Simdi:{cur:.2f} | P&L:{pnl_tl:+.0f} TL")

                # Stop / hedef kontrolü
                stop_raw, stop_kind = resolve_intraday_barrier(
                    pos["stop_price"], is_lng, cur, day_open, "adverse"
                )
                target_raw, target_kind = resolve_intraday_barrier(
                    pos["target_price"], is_lng, cur, day_open, "favorable"
                )

                trigger = None
                raw_exit = None
                if stop_raw is not None:
                    trigger = "stop_gap" if stop_kind == "gap" else "stop_loss"
                    raw_exit = stop_raw
                elif target_raw is not None:
                    trigger = "target_gap" if target_kind == "gap" else "target_hit"
                    raw_exit = target_raw

                if trigger:
                    fill_cur = apply_fill_price(raw_exit, is_lng, "exit", atr_w)
                    pnl_c = close_position(conn, pos, fill_cur, trigger, today_str)
                    msg = (f"[AUTO-CLOSE] {sym} {dir_s} {trigger} | "
                           f"Giris:{entry:.2f} Cikis:{fill_cur:.2f} | P&L:{pnl_c:+.0f} TL")
                    log(msg)
                    closed_in_loop.append(msg)

            for r in rows_to_print:
                print(r)
            print("-"*75)
            eq_col = CLR_GREEN if total_unreal >= 0 else CLR_RED
            print(f"TOPLAM ACIK P&L: {eq_col}{total_unreal:+,.0f} TL{CLR_RESET}  |  "
                  f"Pozisyon: {len(open_pos)-len(closed_in_loop)}  |  "
                  f"Sonraki guncelleme: {WATCH_INTERVAL//60} dakika sonra")
            if closed_in_loop:
                print(f"\nBu turda kapatilan:")
                for m in closed_in_loop:
                    print(f"  {m}")
            print("="*75)

            time.sleep(WATCH_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n{CLR_YELLOW}Izleme durduruldu (Ctrl+C){CLR_RESET}")
        log("Izleme kullanici tarafindan durduruldu.")

# ── Tüm pozisyonlari kapat ────────────────────────────────────
def close_all(conn):
    open_pos = get_open_positions(conn)
    if not open_pos:
        print("Acik pozisyon yok.")
        return
    today_str = date.today().isoformat()
    total = 0
    for pos in open_pos:
        latest = get_latest(conn, pos["symbol"])
        price  = latest["close"] if latest else pos["entry_price"]
        fill_px = apply_fill_price(price, pos["is_long"], "exit", (latest or {}).get("atr14", 0))
        pnl    = close_position(conn, pos, fill_px, "manual_close", today_str)
        total += pnl
        print(f"  Kapatildi: {pos['symbol']}  P&L: {pnl:+,.0f} TL")
    print(f"\nToplam realize P&L: {total:+,.0f} TL")

# ── Entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"

    conn = get_db()

    if mode == "--status":
        print_status(conn)
        conn.close()

    elif mode == "--watch":
        watch_mode(conn)
        conn.close()

    elif mode == "--close-all":
        confirm = input("Tum pozisyonlari kapat? (evet/hayir): ").strip().lower()
        if confirm == "evet":
            close_all(conn)
        else:
            print("Iptal edildi.")
        conn.close()

    else:  # --run (varsayilan)
        print("XGBoost modeli yukleniyor...", end=" ", flush=True)
        model, feature_names, calibrator, thresholds = load_model()
        if model is None:
            conn.close()
            sys.exit(1)
        print("OK")

        run_daily(conn, model, feature_names, calibrator, thresholds)
        print_status(conn)
        conn.close()
        print("\nSonraki adim: py -3.12 daily_report.py")
