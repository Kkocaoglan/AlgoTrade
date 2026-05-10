"""
viop_journal.py -- VIOP Gunluk Ozet (EOD Summary)
---------------------------------------------------
Run separately at 18:30 or call run_viop_journal() from viop_trader.py.
Prints summary and sends Telegram alert.
Does NOT touch any spot-system tables.

Kullanim: python3.12 viop_journal.py
"""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DB_PATH     = Path(__file__).parent / "trade_data.db"
ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")

try:
    from telegram_bot import send_telegram_alert
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False
    def send_telegram_alert(msg, **kw): return False  # noqa: E731

from logger import algo_log


def run_viop_journal(target_date: str | None = None):
    """
    Compute and print the daily VIOP summary.
    target_date: 'YYYY-MM-DD' string; defaults to today.
    """
    today = target_date or date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)

    # Confirm VIOP tables exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name LIKE 'viop_%'"
    ).fetchall()}
    if "viop_positions" not in tables:
        print("[VIOP JOURNAL] viop_positions tablosu yok -- "
              "once viop_trader.py calistirin.")
        conn.close()
        return

    # Positions opened today
    opened = conn.execute(
        "SELECT COUNT(*) FROM viop_positions WHERE DATE(entry_date)=?",
        (today,)
    ).fetchone()[0]

    # Positions closed today
    closed_rows = conn.execute(
        "SELECT pnl, exit_reason "
        "FROM viop_positions "
        "WHERE DATE(exit_date)=? AND status='closed'",
        (today,)
    ).fetchall()
    closed_count = len(closed_rows)
    daily_pnl    = sum(r[0] for r in closed_rows if r[0] is not None)

    # Currently open positions
    open_now = conn.execute(
        "SELECT COUNT(*) FROM viop_positions WHERE status='open'"
    ).fetchone()[0]

    # Cumulative P&L (all closed trades ever)
    total_row = conn.execute(
        "SELECT SUM(pnl) FROM viop_positions WHERE status='closed'"
    ).fetchone()
    total_pnl = float(total_row[0]) if total_row[0] is not None else 0.0

    conn.close()

    sign_d = "+" if daily_pnl >= 0 else ""
    sign_t = "+" if total_pnl >= 0 else ""

    lines = [
        f"\U0001f4ca VIOP GUNLUK OZET ({today})",
        f"Acilan pozisyon  : {opened}",
        f"Kapanan pozisyon : {closed_count}",
        f"Gunluk P&L       : {sign_d}{daily_pnl:.2f} TL",
        f"Toplam P&L       : {sign_t}{total_pnl:.2f} TL",
        f"Acik pozisyon    : {open_now}",
    ]

    print("\n" + "=" * 50)
    for line in lines:
        print(line)
    print("=" * 50)

    send_telegram_alert("\n".join(lines))
    algo_log.system(
        f"[VIOP] Gunluk ozet: acilan={opened} kapanan={closed_count} "
        f"gunluk_pnl={sign_d}{daily_pnl:.2f} "
        f"toplam_pnl={sign_t}{total_pnl:.2f} "
        f"acik={open_now}"
    )


if __name__ == "__main__":
    t = datetime.now(ISTANBUL_TZ)
    print(f"VIOP Gunluk Ozet -- {t.strftime('%Y-%m-%d %H:%M')} Istanbul")
    run_viop_journal()
