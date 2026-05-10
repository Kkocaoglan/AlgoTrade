"""
viop_trader.py -- VIOP (BIST Futures) Paper Trading Loop
----------------------------------------------------------
Completely independent from the spot system (loop_trader.py untouched).
Separate DB tables: viop_ohlcv, viop_indicators, viop_orders, viop_positions.

Kullanim:
  python3.12 viop_trader.py           # production 60s loop
  python3.12 viop_trader.py --once    # single tick (safe test)
"""

import math
import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Config ─────────────────────────────────────────────────────────────
VIOP_CAPITAL         = 20_000       # TL
VIOP_LEVERAGE        = 2            # 1:2 max
VIOP_MAX_POSITIONS   = 2
VIOP_MARGIN_PCT      = 0.15         # %15 teminat gereksinimi
VIOP_STOP_PCT        = 0.02         # %2 stop loss (kaldiracli baz)
VIOP_TARGET_PCT      = 0.03         # %3 hedef (kaldiracli baz)
VIOP_KILL_DAILY_LOSS = 0.05         # gunluk %5 kayip -> dur
VIOP_TICK_SECONDS    = 60
VIOP_SYMBOL          = "XU030F"
ISTANBUL_TZ          = ZoneInfo("Europe/Istanbul")

DB_PATH = Path(__file__).parent / "trade_data.db"

# ── Imports from existing modules (read-only, not modified) ────────────
try:
    from telegram_bot import send_telegram_alert
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False
    def send_telegram_alert(msg, **kw): return False  # noqa: E731

from logger import algo_log
from viop_oms import viop_oms


# ── DB tables ────────────────────────────────────────────────────────
def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS viop_positions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT NOT NULL,
            direction    TEXT NOT NULL,
            entry_date   TEXT,
            entry_price  REAL,
            contracts    INTEGER,
            size_tl      REAL,
            stop_price   REAL,
            target_price REAL,
            status       TEXT DEFAULT 'open',
            exit_date    TEXT,
            exit_price   REAL,
            pnl          REAL,
            exit_reason  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS viop_daily_pnl (
            date        TEXT PRIMARY KEY,
            pnl         REAL    DEFAULT 0.0,
            trades_open INTEGER DEFAULT 0,
            trades_close INTEGER DEFAULT 0
        )
    """)
    conn.commit()


# ── Time helpers ─────────────────────────────────────────────────────
def _now_ist():
    return datetime.now(ISTANBUL_TZ)


def _in_market():
    """True during 10:00-18:00 Istanbul time."""
    t = _now_ist()
    return (10, 0) <= (t.hour, t.minute) < (18, 0)


# ── DB helpers ───────────────────────────────────────────────────────
def _get_latest_indicators() -> dict | None:
    """Return most recent row from viop_indicators as a dict."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""
            SELECT v.date, v.close,
                   i.ema8, i.ema21, i.rsi14, i.atr14,
                   i.macd_line, i.macd_signal, i.macd_hist,
                   i.bb_upper, i.bb_lower
            FROM viop_ohlcv v
            JOIN viop_indicators i ON v.symbol = i.symbol AND v.date = i.date
            WHERE v.symbol = ?
            ORDER BY v.date DESC LIMIT 1
        """, (VIOP_SYMBOL,)).fetchone()
        conn.close()
        if row is None:
            return None
        keys = ["date", "close", "ema8", "ema21", "rsi14", "atr14",
                "macd_line", "macd_signal", "macd_hist", "bb_upper", "bb_lower"]
        return dict(zip(keys, row))
    except Exception as e:
        algo_log.system(f"[VIOP] _get_latest_indicators error: {e}")
        return None


def _get_open_positions() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    rows = conn.execute(
        "SELECT id, symbol, direction, entry_date, entry_price, "
        "contracts, size_tl, stop_price, target_price, "
        "status, exit_date, exit_price, pnl, exit_reason "
        "FROM viop_positions WHERE status='open' ORDER BY entry_date"
    ).fetchall()
    conn.close()
    cols = ["id", "symbol", "direction", "entry_date", "entry_price",
            "contracts", "size_tl", "stop_price", "target_price",
            "status", "exit_date", "exit_price", "pnl", "exit_reason"]
    return [dict(zip(cols, r)) for r in rows]


def _get_today_pnl() -> float:
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    row = conn.execute(
        "SELECT pnl FROM viop_daily_pnl WHERE date=?", (today,)
    ).fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def _add_daily_pnl(delta: float):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    conn.execute("""
        INSERT INTO viop_daily_pnl (date, pnl) VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET pnl = pnl + excluded.pnl
    """, (today, delta))
    conn.commit()
    conn.close()


# ── Position sizing ──────────────────────────────────────────────────
def _compute_contracts(price: float) -> int:
    """
    contracts = floor(VIOP_CAPITAL * VIOP_LEVERAGE * 0.30 / price)
    Uses max 30% of leveraged capital per position.
    """
    if price <= 0:
        return 0
    return max(0, math.floor(VIOP_CAPITAL * VIOP_LEVERAGE * 0.30 / price))


# ── Signal logic ─────────────────────────────────────────────────────
def _check_signals(ind: dict) -> str | None:
    """
    LONG  signal: RSI < 40  AND price < BB lower  AND MACD bullish cross
    SHORT signal: RSI > 65  AND price > BB upper  AND MACD bearish cross
    Returns "LONG", "SHORT", or None.
    """
    if ind is None:
        return None
    rsi = ind.get("rsi14")
    cl  = ind.get("close")
    bbu = ind.get("bb_upper")
    bbl = ind.get("bb_lower")
    mh  = ind.get("macd_hist")
    ml  = ind.get("macd_line")
    ms  = ind.get("macd_signal")

    if None in (rsi, cl, bbu, bbl, mh, ml, ms):
        return None

    macd_bullish = mh > 0 and ml > ms
    macd_bearish = mh < 0 and ml < ms

    if rsi < 40 and cl < bbl and macd_bullish:
        return "LONG"
    if rsi > 65 and cl > bbu and macd_bearish:
        return "SHORT"
    return None


# ── Open / close positions ───────────────────────────────────────────
def _open_position(ind: dict, direction: str):
    price     = ind["close"]
    contracts = _compute_contracts(price)
    if contracts == 0:
        algo_log.system(f"[VIOP] Sozlesme hesaplanamadi (fiyat={price}), acilmiyor")
        print(f"  [VIOP] Sozlesme = 0 @ {price:.2f} -- atlanıyor")
        return

    size_tl = contracts * price
    if direction == "LONG":
        stop_price   = price * (1 - VIOP_STOP_PCT)
        target_price = price * (1 + VIOP_TARGET_PCT)
        oms_dir      = "LONG_OPEN"
    else:
        stop_price   = price * (1 + VIOP_STOP_PCT)
        target_price = price * (1 - VIOP_TARGET_PCT)
        oms_dir      = "SHORT_OPEN"

    order = viop_oms.create_order(
        symbol=VIOP_SYMBOL, direction=oms_dir,
        size_tl=size_tl, contracts=contracts,
        price=price, reason=f"{direction} signal"
    )
    if order["status"] != "FILLED":
        algo_log.system(f"[VIOP] Emir reddedildi: {order.get('rejected_reason')}")
        print(f"  [VIOP] Emir reddedildi: {order.get('rejected_reason')}")
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    conn.execute("""
        INSERT INTO viop_positions
            (symbol, direction, entry_date, entry_price, contracts, size_tl,
             stop_price, target_price, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
    """, (VIOP_SYMBOL, direction, now_str, price, contracts, size_tl,
          stop_price, target_price))
    conn.commit()
    conn.close()

    sign = "+" if direction == "LONG" else ""
    msg = (f"\U0001f4ca [VIOP] {direction} ACILDI: {VIOP_SYMBOL} @ {price:.2f} "
           f"| Stop: {stop_price:.2f} | Hedef: {target_price:.2f} "
           f"| {contracts} sozlesme | {size_tl:.0f} TL")
    send_telegram_alert(msg)
    algo_log.trade(f"[VIOP] {direction} ACILDI: {VIOP_SYMBOL} @ {price:.2f} "
                   f"stop={stop_price:.2f} target={target_price:.2f} "
                   f"contracts={contracts} size={size_tl:.0f}TL")
    print(f"  [VIOP] {direction} ACILDI @ {price:.2f}  "
          f"Stop:{stop_price:.2f}  Hedef:{target_price:.2f}  {contracts} sozlesme")


def _close_position(pos: dict, cur_price: float, reason: str):
    direction = pos["direction"]
    entry     = pos["entry_price"]
    contracts = pos["contracts"]
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pnl = (cur_price - entry) * contracts if direction == "LONG" \
          else (entry - cur_price) * contracts

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE viop_positions
        SET status='closed', exit_date=?, exit_price=?, pnl=?, exit_reason=?
        WHERE id=?
    """, (now_str, cur_price, pnl, reason, pos["id"]))
    conn.commit()
    conn.close()

    oms_dir = "LONG_CLOSE" if direction == "LONG" else "SHORT_CLOSE"
    viop_oms.create_order(
        symbol=VIOP_SYMBOL, direction=oms_dir,
        size_tl=pos["size_tl"], contracts=contracts,
        price=cur_price, reason=reason
    )
    _add_daily_pnl(pnl)

    sign = "+" if pnl >= 0 else ""
    msg = (f"\U0001f4ca [VIOP] KAPATILDI: {reason} "
           f"| P&L: {sign}{pnl:.2f} TL "
           f"| {VIOP_SYMBOL} {direction} {entry:.2f}->{cur_price:.2f}")
    send_telegram_alert(msg)
    algo_log.trade(f"[VIOP] KAPANDI: {VIOP_SYMBOL} {direction} "
                   f"entry={entry:.2f} exit={cur_price:.2f} "
                   f"pnl={sign}{pnl:.2f}TL reason={reason}")
    print(f"  [VIOP] KAPATILDI ({reason})  P&L: {sign}{pnl:.2f} TL")


# ── Exit checks ──────────────────────────────────────────────────────
def _check_exits(cur_price: float):
    for pos in _get_open_positions():
        direction    = pos["direction"]
        stop_price   = pos["stop_price"]
        target_price = pos["target_price"]
        if direction == "LONG":
            if cur_price <= stop_price:
                _close_position(pos, cur_price, "stop_loss")
            elif cur_price >= target_price:
                _close_position(pos, cur_price, "target_hit")
        else:  # SHORT
            if cur_price >= stop_price:
                _close_position(pos, cur_price, "stop_loss")
            elif cur_price <= target_price:
                _close_position(pos, cur_price, "target_hit")


# ── Daily loss guard ─────────────────────────────────────────────────
def _check_daily_loss_limit() -> bool:
    """Return True if daily loss limit breached (trading should stop)."""
    daily_pnl = _get_today_pnl()
    threshold = -(VIOP_CAPITAL * VIOP_KILL_DAILY_LOSS)
    if daily_pnl <= threshold:
        msg = (f"\U0001f4ca [VIOP] GUNLUK ZARAR LIMITI -- islemler durduruldu "
               f"(Gunluk P&L: {daily_pnl:.2f} TL, limit: {threshold:.2f} TL)")
        send_telegram_alert(msg)
        algo_log.risk(f"[VIOP] GUNLUK ZARAR LIMITI: pnl={daily_pnl:.2f} limit={threshold:.2f}")
        print(f"\n  [VIOP] GUNLUK ZARAR LIMITI ({daily_pnl:.2f} TL <= {threshold:.2f} TL) -- dur")
        return True
    return False


# ── Health check (startup only) ──────────────────────────────────────
def _print_health_check():
    conn = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    ind_count = conn.execute(
        "SELECT COUNT(*) FROM viop_indicators WHERE symbol=?",
        (VIOP_SYMBOL,)
    ).fetchone()[0]
    open_pos = conn.execute(
        "SELECT COUNT(*) FROM viop_positions WHERE status='open'"
    ).fetchone()[0]
    conn.close()

    print("=" * 60)
    print(f"[VIOP HEALTH] Sermaye     : {VIOP_CAPITAL:,} TL")
    print(f"[VIOP HEALTH] Kaldirac    : 1:{VIOP_LEVERAGE}")
    print(f"[VIOP HEALTH] Max pozisyon: {VIOP_MAX_POSITIONS}")
    print("[VIOP HEALTH] Spot sistem : BAGIMSIZ (bu sistem ona dokunmaz)")
    print("[VIOP HEALTH] DB tablolari: viop_ohlcv, viop_indicators, "
          "viop_orders, viop_positions")
    print(f"[VIOP HEALTH] viop_indicators: {ind_count} satir")
    print(f"[VIOP HEALTH] Acik pozisyon  : {open_pos} / {VIOP_MAX_POSITIONS}")
    print("=" * 60)


# ── Single tick ──────────────────────────────────────────────────────
def run_tick(tick_num: int):
    ts  = _now_ist().strftime("%H:%M:%S")
    ind = _get_latest_indicators()

    cur_price = ind["close"] if ind else None
    positions = _get_open_positions()
    daily_pnl = _get_today_pnl()

    print(f"\n[{ts}] VIOP Tick #{tick_num}  |  "
          f"Acik:{len(positions)}/{VIOP_MAX_POSITIONS}  |  "
          f"Gunluk P&L: {daily_pnl:+.2f} TL")

    if cur_price is None:
        print("  [VIOP] Fiyat verisi yok -- "
              "viop_data.py ve viop_indicators.py calistirin")
        return

    rsi = ind.get("rsi14") or 0.0
    bbl = ind.get("bb_lower") or 0.0
    bbu = ind.get("bb_upper") or 0.0
    print(f"  Fiyat:{cur_price:.2f}  RSI:{rsi:.1f}  "
          f"BB:{bbl:.2f}/{bbu:.2f}")

    # 1. Daily loss guard
    if _check_daily_loss_limit():
        return

    # 2. Exit checks on open positions
    _check_exits(cur_price)

    # 3. Re-fetch after potential exits
    positions = _get_open_positions()

    # 4. New entry signal check
    if len(positions) < VIOP_MAX_POSITIONS:
        signal = _check_signals(ind)
        if signal:
            open_dirs = {p["direction"] for p in positions}
            if signal not in open_dirs:
                print(f"  [VIOP] Sinyal: {signal}")
                _open_position(ind, signal)
            else:
                print(f"  [VIOP] {signal} zaten acik -- atlaniyor")
        else:
            print(f"  [VIOP] Sinyal yok (RSI={rsi:.1f}, "
                  f"BB low={bbl:.2f}, BB up={bbu:.2f})")
    else:
        print(f"  [VIOP] Max pozisyon dolu ({VIOP_MAX_POSITIONS})")

    # 5. OMS summary
    viop_oms.summary()


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ONCE_MODE = "--once" in sys.argv

    print("=" * 60)
    print("BIST VIOP TRADER v1  |  py -3.12 viop_trader.py")
    print(f"Baslangic : {_now_ist().strftime('%Y-%m-%d %H:%M:%S')} Istanbul")
    print(f"Sermaye   : {VIOP_CAPITAL:,} TL | "
          f"Kaldirac: 1:{VIOP_LEVERAGE} | "
          f"Max Pozisyon: {VIOP_MAX_POSITIONS}")
    print("=" * 60)
    mode_str = "--once (tek iterasyon)" if ONCE_MODE else "uretim (60s dongu)"
    print(f"MODE: {mode_str}")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    conn.close()

    algo_log.system(
        f"[VIOP] Trader v1 baslatildi | "
        f"capital={VIOP_CAPITAL} leverage={VIOP_LEVERAGE} "
        f"max_pos={VIOP_MAX_POSITIONS}"
    )

    _print_health_check()

    tick = 0
    try:
        while True:
            tick += 1
            if _in_market():
                run_tick(tick)
            else:
                ts = _now_ist().strftime("%H:%M:%S")
                print(f"[{ts}] [VIOP] Piyasa kapali, bekleniyor...")

            if ONCE_MODE:
                print("\n[--once] Tamamlandi. Cikiliyor.")
                break

            time.sleep(VIOP_TICK_SECONDS)

    except KeyboardInterrupt:
        print("\n[VIOP] Kullanici tarafindan durduruldu.")
        algo_log.system("[VIOP] Trader kullanici tarafindan durduruldu")
