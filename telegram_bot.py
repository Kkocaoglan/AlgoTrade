"""
telegram_bot.py -- BIST Algo Telegram Alert Bot
--------------------------------------------------
Kullanim: py -3.12 telegram_bot.py        (startup + test)
Import  : from telegram_bot import send_telegram_alert, alert_stop_proximity,
                                   alert_signals, alert_daily_summary
"""

import os, sys, json, sqlite3, struct
from datetime import date, datetime
from pathlib import Path
from urllib import request as urequest, error as uerror

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load .env if present (python-dotenv optional)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# -- Config -----------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("TELEGRAM_CHAT_ID", "")
API_BASE       = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

DB_PATH = Path(__file__).parent / "trade_data.db"
CAPITAL = 100_000

# Deduplication: track which (symbol, alert_type) combos were sent this session
_sent: set = set()


# =====================================================================
# BASE SEND
# =====================================================================
def send_telegram_alert(message: str, parse_mode: str = "HTML") -> bool:
    """
    Send a Telegram message to the configured chat.
    Returns True on success, False on any error.
    Safe to call from any module.
    """
    url  = f"{API_BASE}/sendMessage"
    data = json.dumps({
        "chat_id":    CHAT_ID,
        "text":       message,
        "parse_mode": parse_mode,
    }).encode("utf-8")
    req = urequest.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urequest.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except uerror.URLError as e:
        print(f"[Telegram] Baglanti hatasi: {e}")
        return False
    except Exception as e:
        print(f"[Telegram] Beklenmedik hata: {e}")
        return False


# =====================================================================
# DB HELPERS  (read-only, minimal queries)
# =====================================================================
def _db():
    return sqlite3.connect(DB_PATH)


def _get_open_positions() -> list[dict]:
    conn = _db()
    rows = conn.execute("""
        SELECT symbol, strategy, entry_date, entry_price,
               size_tl, is_long, stop_price, target_price,
               COALESCE(confidence, 0)
        FROM paper_positions WHERE status='open'
    """).fetchall()
    conn.close()
    cols = ["symbol", "strategy", "entry_date", "entry_price",
            "size_tl", "is_long", "stop_price", "target_price", "confidence"]
    positions = []
    for r in rows:
        pos = dict(zip(cols, r))
        c = pos["confidence"]
        if isinstance(c, (bytes, bytearray)):
            c = struct.unpack("<f", c)[0]
        pos["confidence"] = float(c)
        positions.append(pos)
    return positions


def _get_latest_price(symbol: str) -> float | None:
    conn = _db()
    row = conn.execute(
        "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    conn.close()
    return float(row[0]) if row else None


def _get_equity() -> dict:
    conn = _db()
    row = conn.execute(
        "SELECT total_equity, daily_pnl, cash "
        "FROM paper_equity ORDER BY date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return {"equity": float(row[0]), "daily_pnl": float(row[1]), "cash": float(row[2])}
    return {"equity": float(CAPITAL), "daily_pnl": 0.0, "cash": float(CAPITAL)}


def _stop_proximity(pos: dict, cur_price: float) -> float:
    """Return % of the way from entry toward stop (0..100+). Negative = moving away."""
    entry = pos["entry_price"]
    stop  = pos["stop_price"]
    if pos["is_long"]:
        stop_dist   = entry - stop
        toward_stop = entry - cur_price
    else:
        stop_dist   = stop - entry
        toward_stop = cur_price - entry
    if stop_dist <= 0:
        return 0.0
    return (toward_stop / stop_dist) * 100.0


# =====================================================================
# ALERT 1 — STOP PROXIMITY  (>50% threshold)
# =====================================================================
def alert_stop_proximity(
    positions: list[dict] | None = None,
    threshold_pct: float = 50.0,
    dedupe: bool = True,
) -> int:
    """
    Send stop-proximity alerts for all positions that have moved
    more than threshold_pct% of the way from entry toward stop.

    Args:
        positions: list of position dicts (reads DB if None)
        threshold_pct: warn when proximity exceeds this value (default 50%)
        dedupe: skip if already alerted this session (avoids watch-mode spam)

    Returns:
        Number of alerts sent.
    """
    if positions is None:
        positions = _get_open_positions()

    sent = 0
    for pos in positions:
        sym     = pos["symbol"]
        cur     = _get_latest_price(sym)
        if cur is None:
            continue

        prox = _stop_proximity(pos, cur)
        if prox < threshold_pct:
            continue

        dedup_key = (sym, "stop_prox")
        if dedupe and dedup_key in _sent:
            continue

        entry   = pos["entry_price"]
        stop    = pos["stop_price"]
        target  = pos["target_price"]
        dir_str = "LONG" if pos["is_long"] else "SHORT"

        if pos["is_long"]:
            pnl_pct = (cur - entry) / entry * 100
            rem_tl  = abs(cur - stop)
        else:
            pnl_pct = (entry - cur) / entry * 100
            rem_tl  = abs(stop - cur)

        total_dist = abs(entry - stop)
        rem_pct    = (rem_tl / total_dist * 100) if total_dist > 0 else 100.0
        is_tuprs   = (sym == "TUPRS" and not pos["is_long"])
        emoji      = "🚨" if (prox >= 80 or is_tuprs) else "⚠️"

        msg = (
            f"{emoji} <b>STOP YAKINI — {sym} {dir_str}</b>\n"
            f"\n"
            f"Giris  : {entry:.2f} TL\n"
            f"Simdi  : {cur:.2f} TL\n"
            f"Stop   : {stop:.2f} TL\n"
            f"Hedef  : {target:.2f} TL\n"
            f"P&amp;L    : {pnl_pct:+.1f}%\n"
            f"Stop'a : %{prox:.0f} gidildi\n"
            f"Kalan  : {rem_tl:.2f} TL (%{rem_pct:.1f} uzakta)"
        )
        if prox >= 90:
            msg += "\n\n<b>KRITIK: Stop seviyesi cok yakin, pozisyonu gozden gecir!</b>"

        if send_telegram_alert(msg):
            _sent.add(dedup_key)
            sent += 1
            print(f"[Telegram] Stop proximity alarmı: {sym} {dir_str} %{prox:.0f}")

    return sent


# =====================================================================
# ALERT 2 — TUPRS SHORT URGENT  (<10% to stop)
# =====================================================================
def alert_tuprs_urgent(positions: list[dict] | None = None) -> bool:
    """
    Dedicated TUPRS SHORT urgent warning when less than 10% of
    the stop range remains. Called from paper_trade and watch mode.
    """
    if positions is None:
        positions = _get_open_positions()

    for pos in positions:
        if pos["symbol"] != "TUPRS" or pos["is_long"]:
            continue

        cur = _get_latest_price("TUPRS")
        if cur is None:
            return False

        entry      = pos["entry_price"]
        stop       = pos["stop_price"]
        rem_tl     = abs(stop - cur)
        total_dist = abs(stop - entry)
        rem_pct    = (rem_tl / total_dist * 100) if total_dist > 0 else 100.0
        prox       = _stop_proximity(pos, cur)

        if rem_pct > 10.0:
            return False

        dedup_key = ("TUPRS", "urgent")
        if dedup_key in _sent:
            return True  # already sent this session

        pnl_pct = (entry - cur) / entry * 100

        msg = (
            f"🚨 <b>ACIL: TUPRS SHORT — STOP SON DERECE YAKIN!</b>\n"
            f"\n"
            f"Giris        : {entry:.2f} TL\n"
            f"Simdi        : {cur:.2f} TL\n"
            f"Stop         : {stop:.2f} TL\n"
            f"P&amp;L          : {pnl_pct:+.1f}%\n"
            f"Stop'a       : %{prox:.0f} gidildi\n"
            f"Kalan mesafe : {rem_tl:.2f} TL (%{rem_pct:.1f})\n"
            f"\n"
            f"Manuel kapama:\n"
            f"<code>py -3.12 paper_trade.py --close-all</code>"
        )
        ok = send_telegram_alert(msg)
        if ok:
            _sent.add(dedup_key)
            print("[Telegram] TUPRS SHORT acil uyarisi gonderildi")
        return ok

    return False


# =====================================================================
# ALERT 3 — NEW BUY/SELL SIGNALS
# =====================================================================

# ATR multipliers (mirror paper_trade.py constants)
_ATR_STOP   = 2.0
_ATR_TARGET = 1.2

# Signal strength thresholds
STRONG_THRESHOLD = 64.0   # >= 64%: STRONG — tradeable
WEAK_THRESHOLD   = 50.0   # 50-64%: WEAK   — informational


def _trend_label(mtf_trend: int) -> str:
    if mtf_trend == 1:
        return "YUKARI ^"
    if mtf_trend == -1:
        return "ASAGI v"
    return "YATAY -"


def _build_signal_msg(s: dict) -> str:
    """
    Build a per-signal Telegram message.

    STRONG (>= 64%): full card with calculated stop & target.
    WEAK   (50-64%): warning card, no stop/target.
    """
    sym       = s["symbol"]
    conf      = s["confidence"]
    price     = s["price"]
    atr       = s.get("atr", 0.0)
    rsi       = s.get("rsi", 0.0)
    trend     = _trend_label(s.get("mtf_trend", 0))
    is_long   = s["signal"] in (1, "AL")
    direction = "AL" if is_long else "SAT"

    if conf >= STRONG_THRESHOLD:
        if is_long:
            stop   = price - _ATR_STOP   * atr
            target = price + _ATR_TARGET * atr
        else:
            stop   = price + _ATR_STOP   * atr
            target = price - _ATR_TARGET * atr

        return (
            f"🟢 <b>GUCLU SINYAL: {sym} {direction}</b>\n"
            f"💰 Fiyat: {price:.2f} TL\n"
            f"📊 Guven: {conf:.1f}%\n"
            f"📈 RSI: {rsi:.0f} | Trend: {trend}\n"
            f"🎯 Hedef: {target:.2f} TL\n"
            f"🛑 Stop: {stop:.2f} TL"
        )
    else:
        return (
            f"🟡 <b>ZAYIF SINYAL: {sym} {direction}</b>\n"
            f"💰 Fiyat: {price:.2f} TL\n"
            f"📊 Guven: {conf:.1f}% ⚠️ Esik altinda (&lt;%64)\n"
            f"📈 RSI: {rsi:.0f} | Trend: {trend}\n"
            f"⚠️ Bu sinyal guven esiginin altinda, dikkatli ol!"
        )


def alert_signals(signals: list[dict]) -> int:
    """
    Send per-signal Telegram alerts for both STRONG and WEAK signals.

    STRONG (>= 64%): green card with target & stop.
    WEAK   (50-64%): yellow warning card, no target/stop.

    Args:
        signals: list from get_ml_signals() — dicts with keys:
                 symbol, signal (1=AL / -1=SAT / 0=BEKLE),
                 confidence, price, atr, rsi, mtf_trend

    Returns:
        Number of messages sent.
    """
    def is_directional(s):
        return s["signal"] in (1, "AL", -1, "SAT")

    actionable = sorted(
        [s for s in signals if is_directional(s) and s["confidence"] >= WEAK_THRESHOLD],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    if not actionable:
        return 0

    sent = 0
    strong_count = 0
    weak_count   = 0

    for s in actionable:
        msg = _build_signal_msg(s)
        if send_telegram_alert(msg):
            sent += 1
            is_long   = s["signal"] in (1, "AL")
            direction = "AL" if is_long else "SAT"
            if s["confidence"] >= STRONG_THRESHOLD:
                strong_count += 1
                print(f"[Telegram] GUCLU sinyal: {s['symbol']} {direction} {s['confidence']:.1f}%")
            else:
                weak_count += 1
                print(f"[Telegram] ZAYIF sinyal: {s['symbol']} {direction} {s['confidence']:.1f}%")

    if sent:
        print(f"[Telegram] Toplam: {strong_count} guclu + {weak_count} zayif sinyal gonderildi")
    return sent


# =====================================================================
# ALERT 4 — DAILY SUMMARY (09:30)
# =====================================================================
def alert_daily_summary(positions: list[dict] | None = None) -> bool:
    """
    Send the morning portfolio summary. Call from daily_report.py main.
    Includes: equity, daily P&L, all open positions, stop warnings.
    """
    if positions is None:
        positions = _get_open_positions()

    eq      = _get_equity()
    equity  = eq["equity"]
    d_pnl   = eq["daily_pnl"]
    cash    = eq["cash"]
    tot_pnl = equity - CAPITAL

    d_sign  = "+" if d_pnl   >= 0 else ""
    t_sign  = "+" if tot_pnl >= 0 else ""
    d_emoji = "📈" if d_pnl >= 0 else "📉"

    lines = [
        f"{d_emoji} <b>BIST Algo — Gunluk Ozet</b>",
        f"📅 {date.today().isoformat()}",
        "",
        "💰 <b>Portfoy Durumu</b>",
        f"  Equity   : {equity:>10,.0f} TL",
        f"  Nakit    : {cash:>10,.0f} TL",
        f"  Gun P&amp;L  : {d_sign}{d_pnl:>+10,.0f} TL",
        f"  Toplam   : {t_sign}{tot_pnl:>+10,.0f} TL "
        f"({t_sign}{tot_pnl / CAPITAL * 100:.2f}%)",
        "",
        f"📋 <b>Acik Pozisyonlar ({len(positions)})</b>",
    ]

    warnings = []
    for pos in positions:
        sym     = pos["symbol"]
        cur     = _get_latest_price(sym) or pos["entry_price"]
        entry   = pos["entry_price"]
        stop    = pos["stop_price"]
        dir_str = "LONG" if pos["is_long"] else "SHORT"

        if pos["is_long"]:
            pnl_pct = (cur - entry) / entry * 100
        else:
            pnl_pct = (entry - cur) / entry * 100

        prox = _stop_proximity(pos, cur)
        warn = " ⚠️" if prox > 50 else ""
        if prox > 50:
            warnings.append(f"  ⚠️ {sym} {dir_str}: stop'a %{prox:.0f} gidildi")

        pnl_emoji = "+" if pnl_pct >= 0 else ""
        lines.append(
            f"  {sym:<6} {dir_str:<5}  {pnl_emoji}{pnl_pct:.1f}%"
            f"  Cur:{cur:.2f}  Stop:{stop:.2f}{warn}"
        )

    if warnings:
        lines += ["", "⚠️ <b>Stop Uyarilari:</b>"] + warnings

    # TUPRS SHORT critical check
    for pos in positions:
        if pos["symbol"] == "TUPRS" and not pos["is_long"]:
            cur        = _get_latest_price("TUPRS") or pos["entry_price"]
            rem_tl     = abs(pos["stop_price"] - cur)
            total_dist = abs(pos["stop_price"] - pos["entry_price"])
            rem_pct    = (rem_tl / total_dist * 100) if total_dist > 0 else 100
            if rem_pct < 10:
                lines += [
                    "",
                    "🚨 <b>ACIL: TUPRS SHORT stop cok yakin!</b>",
                    f"  Kalan: {rem_tl:.2f} TL (%{rem_pct:.1f})",
                ]

    ok = send_telegram_alert("\n".join(lines))
    if ok:
        print("[Telegram] Gunluk ozet gonderildi")
    return ok


# =====================================================================
# CLI ENTRY — test + startup message
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Telegram Bot Baslangic Testi")
    print("=" * 50)

    # 1. Startup ping
    print("\n[1] Startup mesaji gonderiliyor...")
    ok = send_telegram_alert("🤖 BIST Algo Bot aktif")
    print(f"    {'OK' if ok else 'HATA'}")

    if not ok:
        print("\nHATA: Telegram API'ye ulasilamiyor.")
        print("Token / Chat ID'yi kontrol et.")
        sys.exit(1)

    # 2. Signal format test — dummy signals for both strength levels
    print("\n[2] Sinyal format testi (dummy veriler)...")

    dummy_signals = [
        # STRONG signal (>= 64%) — full card with stop & target
        {
            "symbol":    "GARAN",
            "signal":    "AL",        # or 1
            "confidence": 72.4,
            "price":     127.10,
            "atr":        4.71,       # ATR14
            "rsi":        38.5,
            "mtf_trend":  0,          # YATAY
        },
        # WEAK signal (50-64%) — yellow warning, no stop/target
        {
            "symbol":    "YKBNK",
            "signal":    "AL",
            "confidence": 58.2,
            "price":      34.34,
            "atr":         1.15,
            "rsi":        40.1,
            "mtf_trend":   1,         # YUKARI
        },
        # STRONG SAT signal
        {
            "symbol":    "TUPRS",
            "signal":    "SAT",       # or -1
            "confidence": 66.8,
            "price":     260.00,
            "atr":         9.20,
            "rsi":        57.1,
            "mtf_trend":   1,         # YUKARI
        },
    ]

    n_sent = alert_signals(dummy_signals)
    print(f"    {n_sent} mesaj gonderildi")

    # 3. Daily summary
    print("\n[3] Gunluk ozet gonderiliyor...")
    alert_daily_summary()

    # 4. Stop proximity check
    print("\n[4] Stop proximity kontrol...")
    positions = _get_open_positions()
    if positions:
        n = alert_stop_proximity(positions, threshold_pct=50.0, dedupe=False)
        print(f"    {n} uyari gonderildi")
    else:
        print("    Acik pozisyon yok")

    # 5. TUPRS urgent
    print("\n[5] TUPRS SHORT kontrol...")
    alert_tuprs_urgent(positions)

    print("\n" + "=" * 50)
    print("Test tamamlandi.")
    print("=" * 50)
