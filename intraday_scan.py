"""
intraday_scan.py -- BIST Intraday Signal Scanner
-------------------------------------------------
Task Scheduler tarafindan calistirilir. sleep() KULLANILMAZ.

Kullanim:
  py -3.12 intraday_scan.py            # intraday scan (10:30-18:00)
  py -3.12 intraday_scan.py --morning  # morning summary (10:00)

Her calistirildiginda:
  - ML sinyallerini kontrol eder
  - Yeni sinyal varsa (bugun daha gonderilmemisse) Telegram'a yollar
  - Acik pozisyonlarda intragun >3% dusus varsa uyari gonderir
  - Durum logs/signals_today.json dosyasina kaydedilir (gece sifirlanir)
"""

import sys, json, sqlite3, struct
from datetime import date, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -- Paths ---------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
LOGS_DIR    = BASE_DIR / "logs"
SIGNALS_LOG = LOGS_DIR / "signals_today.json"
LOGS_DIR.mkdir(exist_ok=True)

IS_MORNING = "--morning" in sys.argv

# -- Import project modules ----------------------------------------------
# paper_trade.py is importable since its main block is under __name__
from paper_trade import (
    get_db,
    get_open_positions,
    load_model,
    get_ml_signals,
    fetch_live_prices,
    SYMBOLS,
    BUY_THRESHOLD_DEFAULT,
    SELL_THRESHOLD_DEFAULT,
)
from telegram_bot import (
    send_telegram_alert,
    STRONG_THRESHOLD,   # 64.0
    WEAK_THRESHOLD,     # 50.0
)

# ATR multipliers (mirrors paper_trade.py)
_ATR_STOP   = 2.0
_ATR_TARGET = 1.2


# =====================================================================
# SIGNAL LOG  (auto-reset at midnight)
# =====================================================================

def load_today_log() -> dict:
    """Load today's sent-signals log. Returns fresh dict if date changed."""
    today = date.today().isoformat()
    if SIGNALS_LOG.exists():
        try:
            data = json.loads(SIGNALS_LOG.read_text(encoding="utf-8"))
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "signals": {}, "drops": {}}


def save_today_log(data: dict) -> None:
    SIGNALS_LOG.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def _sig_key(sym: str, direction: str) -> str:
    return f"{sym}_{direction}"


# =====================================================================
# HELPERS
# =====================================================================

def _trend_label(mtf_trend: int) -> str:
    if mtf_trend == 1:
        return "YUKARI ^"
    if mtf_trend == -1:
        return "ASAGI v"
    return "YATAY -"


def _is_buy(s: dict) -> bool:
    return s["signal"] in (1, "AL")


def _is_sell(s: dict) -> bool:
    return s["signal"] in (-1, "SAT")


def _direction(s: dict) -> str:
    return "AL" if _is_buy(s) else "SAT"


def _build_intraday_msg(s: dict, now_str: str) -> str:
    """Build a 📊 GUN ICI message for a single signal."""
    sym       = s["symbol"]
    conf      = s["confidence"]
    price     = s["price"]
    atr       = s.get("atr", 0.0)
    rsi       = s.get("rsi", 0.0)
    trend     = _trend_label(s.get("mtf_trend", 0))
    direction = _direction(s)
    is_long   = _is_buy(s)

    header = (
        f"📊 <b>GUN ICI SINYAL: {sym} {direction}</b>\n"
        f"🕐 Saat: {now_str}\n"
    )

    if conf >= STRONG_THRESHOLD:
        stop   = price - _ATR_STOP   * atr if is_long else price + _ATR_STOP   * atr
        target = price + _ATR_TARGET * atr if is_long else price - _ATR_TARGET * atr
        body = (
            f"🟢 <b>GUCLU SINYAL</b>\n"
            f"💰 Fiyat: {price:.2f} TL\n"
            f"📊 Guven: {conf:.1f}%\n"
            f"📈 RSI: {rsi:.0f} | Trend: {trend}\n"
            f"🎯 Hedef: {target:.2f} TL\n"
            f"🛑 Stop: {stop:.2f} TL"
        )
    else:
        body = (
            f"🟡 <b>ZAYIF SINYAL</b>\n"
            f"💰 Fiyat: {price:.2f} TL\n"
            f"📊 Guven: {conf:.1f}% ⚠️ Esik altinda (&lt;%64)\n"
            f"📈 RSI: {rsi:.0f} | Trend: {trend}\n"
            f"⚠️ Bu sinyal guven esiginin altinda, dikkatli ol!"
        )

    return header + body


# =====================================================================
# MORNING SCAN  (--morning flag, 10:00)
# =====================================================================

def run_morning(conn, model, feature_names, calibrator, thresholds) -> None:
    """
    Send 🌅 SABAH summary with ALL signals for the day.
    Marks strong signals as sent so intraday scans skip them.
    """
    now_str = datetime.now().strftime("%H:%M")
    log     = load_today_log()
    signals = get_ml_signals(conn, model, feature_names, calibrator, thresholds)

    strong = sorted(
        [s for s in signals if (_is_buy(s) or _is_sell(s)) and s["confidence"] >= STRONG_THRESHOLD],
        key=lambda x: x["confidence"], reverse=True,
    )
    weak = sorted(
        [s for s in signals if (_is_buy(s) or _is_sell(s))
         and WEAK_THRESHOLD <= s["confidence"] < STRONG_THRESHOLD],
        key=lambda x: x["confidence"], reverse=True,
    )

    lines = [
        "🌅 <b>SABAH TARAMASI</b>",
        f"📅 {date.today().isoformat()} | {now_str}",
        f"Evren: {len(SYMBOLS)} hisse",
        "",
    ]

    if strong:
        lines.append(f"🟢 <b>Guclu Sinyaller ({len(strong)}):</b>")
        for s in strong:
            d = _direction(s)
            lines.append(
                f"  <b>{s['symbol']}</b> {d}  Guven: {s['confidence']:.1f}%"
                f"  Fiyat: {s['price']:.2f} TL  RSI: {s.get('rsi',0):.0f}"
            )
        lines.append("")

    if weak:
        lines.append(f"🟡 <b>Zayif Sinyaller ({len(weak)}):</b>")
        for s in weak:
            d = _direction(s)
            lines.append(
                f"  {s['symbol']} {d}  Guven: {s['confidence']:.1f}%"
                f"  Fiyat: {s['price']:.2f} TL"
            )
        lines.append("")

    if not strong and not weak:
        lines.append("Bugun yuksek guvenli sinyal yok.")
        lines.append("Piyasa bekleme modunda.")

    ok = send_telegram_alert("\n".join(lines))
    print(f"[Sabah] Ozet gonderildi: {len(strong)} guclu + {len(weak)} zayif  {'OK' if ok else 'HATA'}")

    # Mark strong signals as sent so intraday doesn't duplicate them
    for s in strong:
        key = _sig_key(s["symbol"], _direction(s))
        log["signals"][key] = now_str

    save_today_log(log)


# =====================================================================
# INTRADAY SCAN  (every 30 min, 10:30-18:00)
# =====================================================================

def run_intraday(conn, model, feature_names, calibrator, thresholds) -> int:
    """
    1. ML signals — send only NEW ones (not in today's log).
    2. Open positions — send drop warning if intraday move >3%.
    Returns total alerts sent.
    """
    now_str  = datetime.now().strftime("%H:%M")
    log      = load_today_log()
    sent     = 0

    # -- 1. New ML signals -----------------------------------------------
    signals = get_ml_signals(conn, model, feature_names, calibrator, thresholds)

    new_sigs = [
        s for s in signals
        if (_is_buy(s) or _is_sell(s))
        and s["confidence"] >= WEAK_THRESHOLD
        and _sig_key(s["symbol"], _direction(s)) not in log["signals"]
    ]
    new_sigs.sort(key=lambda x: x["confidence"], reverse=True)

    for s in new_sigs:
        key = _sig_key(s["symbol"], _direction(s))
        msg = _build_intraday_msg(s, now_str)
        if send_telegram_alert(msg):
            log["signals"][key] = now_str
            sent += 1
            strength = "GUCLU" if s["confidence"] >= STRONG_THRESHOLD else "ZAYIF"
            print(f"[Intraday] {strength} sinyal: {s['symbol']} {_direction(s)} {s['confidence']:.1f}%")

    # -- 2. Open position drop warnings ----------------------------------
    open_pos = get_open_positions(conn)
    if open_pos:
        live_prices, day_opens = fetch_live_prices(SYMBOLS)

        for pos in open_pos:
            sym      = pos["symbol"]
            cur      = live_prices.get(sym)
            day_open = day_opens.get(sym)

            if cur is None or not day_open or day_open == 0:
                continue

            intra_pct = (cur - day_open) / day_open * 100

            # "drop" = adverse intraday move for this direction
            drop = -intra_pct if pos["is_long"] else intra_pct

            if drop < 3.0:
                continue

            drop_key = f"drop_{sym}"
            if drop_key in log["drops"]:
                continue  # already warned today

            entry   = pos["entry_price"]
            stop    = pos["stop_price"]
            dir_str = "LONG" if pos["is_long"] else "SHORT"
            pnl_pct = (
                (cur - entry) / entry * 100 if pos["is_long"]
                else (entry - cur) / entry * 100
            )

            # Stop proximity for context
            if pos["is_long"]:
                prox = (entry - cur) / max(entry - stop, 1e-9) * 100
            else:
                prox = (cur - entry) / max(stop - entry, 1e-9) * 100
            prox = max(0.0, prox)

            msg = (
                f"⚠️ <b>POZISYON UYARI: {sym} {dir_str}</b>\n"
                f"🕐 Saat: {now_str}\n"
                f"\n"
                f"Intragun hareket : -{drop:.1f}% (gun acilisindan)\n"
                f"Giris            : {entry:.2f} TL\n"
                f"Simdi            : {cur:.2f} TL\n"
                f"Stop             : {stop:.2f} TL\n"
                f"Pozisyon P&amp;L     : {pnl_pct:+.1f}%\n"
                f"Stop'a yakinlik  : %{prox:.0f}\n"
                f"\n"
                f"Pozisyonu gozden gecir!"
            )
            if send_telegram_alert(msg):
                log["drops"][drop_key] = now_str
                sent += 1
                print(f"[Intraday] Drop uyarisi: {sym} {dir_str} -{drop:.1f}%")

    save_today_log(log)

    # Quiet status line (for scheduler logs)
    if sent == 0:
        n_pos = len(open_pos) if open_pos else 0
        print(f"[Intraday] {now_str} — Yeni sinyal yok | {n_pos} pozisyon izleniyor")

    return sent


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":
    print("=" * 55)
    mode = "SABAH TARAMASI" if IS_MORNING else "INTRADAY TARAMA"
    print(f"BIST {mode}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    conn = get_db()
    print("Model yukleniyor...", end=" ", flush=True)
    model, feature_names, calibrator, thresholds = load_model()

    if model is None:
        print("HATA: Model bulunamadi. Once ml_train.py calistir.")
        conn.close()
        sys.exit(1)
    print("OK")

    if IS_MORNING:
        run_morning(conn, model, feature_names, calibrator, thresholds)
    else:
        n = run_intraday(conn, model, feature_names, calibrator, thresholds)
        print(f"Toplam {n} alert gonderildi.")

    conn.close()
    print("Tamamlandi.")
