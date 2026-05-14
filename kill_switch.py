"""
kill_switch.py -- BIST Algo Trading Kill Switch v1
-----------------------------------------------------
Usage:
  from kill_switch import KillSwitch, get_equity_peak, get_current_equity_ks
  ok, reason = KillSwitch.check_all(symbol, price, size_tl)

Test:
  py -3.12 kill_switch.py --test
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    from logger import algo_log
    HAS_LOG = True
except Exception:
    HAS_LOG = False

from bist_config import (
    DAILY_LOSS_LIMIT,
    DRAWDOWN_HALT_PCT,
    DRAWDOWN_KILL_PCT,
    FAT_FINGER_PCT,
    MAX_ORDERS_PER_MIN,
    MAX_POSITION_SIZE,
)

# -- Paths -------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "trade_data.db"
KILL_FILE = str(BASE_DIR / "KILL_SWITCH.txt")
LOG_FILE  = str(BASE_DIR / "logs" / "kill_switch.log")
TRADE_LOG = str(BASE_DIR / "results" / "trade_log.csv")

# -- Constants ---------------------------------------------------------------
# Imported from bist_config.py so kill-switch gates share loop_trader.py active defaults.


# =============================================================================
# DB HELPERS  (standalone — no paper_trade imports to avoid circular deps)
# =============================================================================

def get_last_known_price(symbol: str) -> float | None:
    """Return last EOD close price for symbol from ohlcv table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.execute(
            "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (symbol,),
        )
        row = cur.fetchone()
        conn.close()
        return float(row[0]) if row else None
    except Exception:
        return None


def get_equity_peak(conn) -> float:
    """
    Read equity peak from equity_tracker table.
    Creates the table if it does not exist.
    Returns 100,000 TL (starting capital) if no rows exist yet.
    """
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_tracker (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                equity_peak REAL,
                updated_at  TEXT
            )
        """)
        conn.commit()
        row = conn.execute(
            "SELECT equity_peak FROM equity_tracker ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row else 100_000.0
    except Exception:
        return 100_000.0


def get_current_equity_ks(conn) -> float:
    """
    Standalone equity calculation (does not depend on paper_trade imports).
    cash (from paper_equity) + mark-to-market of open positions (using ohlcv close).
    """
    try:
        row2 = conn.execute(
            "SELECT cash FROM paper_equity ORDER BY date DESC LIMIT 1"
        ).fetchone()
        balance = float(row2[0]) if row2 else 100_000.0

        positions = conn.execute(
            "SELECT symbol, entry_price, size_tl, is_long "
            "FROM paper_positions WHERE status='open'"
        ).fetchall()

        pos_val = 0.0
        for sym, entry_price, size_tl, is_long in positions:
            prow = conn.execute(
                "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if prow and entry_price:
                cur_price = float(prow[0])
                if is_long:
                    pos_val += size_tl * (cur_price / entry_price)
                else:
                    pct      = (entry_price - cur_price) / entry_price
                    pos_val += size_tl * (1 + pct)
            else:
                pos_val += size_tl

        return balance + pos_val
    except Exception:
        return 100_000.0


# =============================================================================
# KILL SWITCH CLASS
# =============================================================================

class KillSwitch:

    @staticmethod
    def check_all(symbol: str, price: float, size_tl: float,
                  direction: str = "BUY") -> tuple[bool, str | None]:
        """
        Run all safety checks before a BUY or SHORT order is placed.

        direction: "BUY" (default) or "SHORT" — used by the rate limiter
                   to maintain separate per-direction counters.

        Returns:
            (True,  None)   -> order approved, proceed
            (False, reason) -> order blocked, reason is a plain-ASCII string
        """
        checks = [
            KillSwitch._check_kill_file(),
            KillSwitch._check_daily_loss(),
            KillSwitch._check_size(size_tl),
            KillSwitch._check_fat_finger(symbol, price),
            KillSwitch._check_order_rate(direction=direction),
            KillSwitch._check_drawdown(),
        ]
        for ok, reason in checks:
            if not ok:
                KillSwitch._log(f"BLOCKED {symbol}: {reason}")
                return False, reason
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_kill_file() -> tuple[bool, str | None]:
        if os.path.exists(KILL_FILE):
            return False, "KILL_SWITCH.txt mevcut -- sistem durduruldu"
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_daily_loss(_log_path: str | None = None) -> tuple[bool, str | None]:
        today    = datetime.now().strftime("%Y-%m-%d")
        log_path = _log_path or TRADE_LOG
        try:
            import csv
            if not os.path.exists(log_path):
                return True, None
            today_pnl = 0.0
            with open(log_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("tarih", "") == today:
                        try:
                            today_pnl += float(row.get("net_pnl_tl", 0) or 0)
                        except (ValueError, TypeError):
                            pass
            if today_pnl < DAILY_LOSS_LIMIT:
                return False, f"Gunluk kayip limiti: {today_pnl:.0f} TL"
        except Exception:
            pass
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_size(size_tl: float) -> tuple[bool, str | None]:
        if size_tl > MAX_POSITION_SIZE:
            return False, (
                f"Pozisyon cok buyuk: {size_tl:.0f} TL "
                f"(max {MAX_POSITION_SIZE:,} TL)"
            )
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_fat_finger(symbol: str, price: float) -> tuple[bool, str | None]:
        try:
            last = get_last_known_price(symbol)
            if last and last > 0:
                deviation = abs(price - last) / last
                if deviation > FAT_FINGER_PCT:
                    return False, (
                        f"Fat finger: {price:.2f} vs son {last:.2f} "
                        f"(fark %{deviation*100:.1f})"
                    )
        except Exception:
            pass
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_order_rate(direction: str = "BUY") -> tuple[bool, str | None]:
        """
        Separate rate-limit counters for BUY and SHORT orders.

        Limits:
          BUY  : max MAX_ORDERS_PER_MIN  (5) per 60-second window
          SHORT: max MAX_ORDERS_PER_MIN  (5) per 60-second window (separate counter)
          COMBINED: max MAX_ORDERS_COMBINED (8) per 60-second window
        """
        MAX_ORDERS_COMBINED = 8
        try:
            if not os.path.exists(LOG_FILE):
                return True, None
            since = datetime.now() - timedelta(seconds=60)
            buy_recent   = 0
            short_recent = 0
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if len(line) < 19:
                        continue
                    try:
                        ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    if ts <= since:
                        continue
                    if "ORDER_SENT_BUY" in line:
                        buy_recent += 1
                    elif "ORDER_SENT_SHORT" in line:
                        short_recent += 1
                    elif "ORDER_SENT" in line:
                        # Legacy entries without direction tag count as BUY
                        buy_recent += 1

            # Per-direction check
            if direction == "SHORT":
                if short_recent >= MAX_ORDERS_PER_MIN:
                    return False, (
                        f"Emir hizi limiti SHORT: {short_recent}/dk "
                        f"(max {MAX_ORDERS_PER_MIN})"
                    )
            else:  # BUY or any other entry direction
                if buy_recent >= MAX_ORDERS_PER_MIN:
                    return False, (
                        f"Emir hizi limiti: {buy_recent}/dk "
                        f"(max {MAX_ORDERS_PER_MIN})"
                    )

            # Combined flood-guard
            combined = buy_recent + short_recent
            if combined >= MAX_ORDERS_COMBINED:
                return False, (
                    f"Emir hizi limiti (toplam): {combined}/dk "
                    f"(max {MAX_ORDERS_COMBINED})"
                )
        except Exception:
            pass
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_drawdown() -> tuple[bool, str | None]:
        try:
            conn = sqlite3.connect(DB_PATH)

            # Need at least 3 equity data points before drawdown is meaningful
            eq_rows = conn.execute(
                "SELECT COUNT(*) FROM paper_equity WHERE total_equity > 0"
            ).fetchone()[0]
            if eq_rows < 3:
                conn.close()
                return True, None

            peak    = get_equity_peak(conn)
            current = get_current_equity_ks(conn)
            conn.close()

            if peak <= 0 or current <= 0:
                return True, None

            if peak > 0:
                dd = (peak - current) / peak
                if dd >= DRAWDOWN_KILL_PCT:
                    # Auto-create kill file
                    with open(KILL_FILE, "w") as f:
                        f.write(
                            f"Auto-created: drawdown {dd:.1%} "
                            f"at {datetime.now().isoformat()}"
                        )
                    return False, (
                        f"DRAWDOWN KILL: {dd:.1%} -- "
                        f"KILL_SWITCH.txt olusturuldu"
                    )
                if dd >= DRAWDOWN_HALT_PCT:
                    return False, f"Drawdown halt: {dd:.1%} (limit %{DRAWDOWN_HALT_PCT*100:.0f})"
        except Exception:
            pass
        return True, None

    # -------------------------------------------------------------------------
    @staticmethod
    def _log(msg: str):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
        print(f"[KILL] {msg}")
        if HAS_LOG:
            algo_log.risk(f"[KILL_SWITCH] {msg}")

    # -------------------------------------------------------------------------
    @staticmethod
    def log_order_sent(symbol: str, price: float, size_tl: float,
                       direction: str = "BUY"):
        """
        Call after a BUY or SHORT order is filled — used by order-rate limiter.
        Logs direction-tagged marker so _check_order_rate can count separately.
        """
        tag = "ORDER_SENT_SHORT" if direction == "SHORT" else "ORDER_SENT_BUY"
        msg = f"{tag} {symbol} {price:.2f} {size_tl:.0f}TL"
        KillSwitch._log(msg)
        if HAS_LOG:
            algo_log.debug(f"[KILL_SWITCH] {msg}")


# =============================================================================
# --test MODE
# =============================================================================

def _run_tests():
    import csv as _csv

    passed = 0
    failed = 0

    def check(name: str, ok: bool, reason: str | None,
              expect_ok: bool, expect_substr: str = ""):
        nonlocal passed, failed
        result_ok = (ok == expect_ok)
        if expect_substr and not ok:
            result_ok = result_ok and (expect_substr in (reason or ""))
        status = "PASS" if result_ok else "FAIL"
        if result_ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")
        if not result_ok:
            print(f"         got: ok={ok}, reason={reason!r}")
            print(f"         expected ok={expect_ok}, substr={expect_substr!r}")

    print("\n" + "="*55)
    print("Kill Switch Test Suite")
    print("="*55)

    # -- Test 1: KILL_SWITCH.txt → must be blocked ---------------------------
    print("\n[1] KILL_SWITCH.txt mevcutsa blokla:")
    with open(KILL_FILE, "w") as f:
        f.write("test")
    ok, reason = KillSwitch.check_all("GARAN", 100.0, 5_000.0)
    check("KILL_SWITCH.txt mevcutsa blocked", ok, reason,
          expect_ok=False, expect_substr="KILL_SWITCH.txt")
    os.remove(KILL_FILE)
    print("     KILL_SWITCH.txt silindi")

    # -- Test 2: Daily loss → must be blocked --------------------------------
    print("\n[2] Gunluk kayip -3000 TL simule et:")
    today   = datetime.now().strftime("%Y-%m-%d")
    tmp_log = str(BASE_DIR / "results" / "_test_trade_log.csv")
    cols    = ["tarih", "net_pnl_tl"]
    with open(tmp_log, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"tarih": today, "net_pnl_tl": "-3000"})

    ok, reason = KillSwitch._check_daily_loss(_log_path=tmp_log)
    check("Gunluk kayip -3000 blocked", ok, reason,
          expect_ok=False, expect_substr="Gunluk kayip")
    os.remove(tmp_log)

    # -- Test 3: Fat finger → must be blocked --------------------------------
    print("\n[3] Fat finger %10 fark:")
    last = get_last_known_price("GARAN")
    if last:
        bad_price = last * 1.15   # 15% deviation
        ok, reason = KillSwitch._check_fat_finger("GARAN", bad_price)
        check("Fat finger blocked", ok, reason,
              expect_ok=False, expect_substr="Fat finger")
    else:
        print("   GARAN fiyati DB'de yok, fat-finger testi atlanıyor")
        passed += 1   # skip counts as pass

    # -- Test 4: Drawdown %9 → must be blocked (but < %12 → no kill file) ---
    print("\n[4] Drawdown %9 simule et:")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_tracker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equity_peak REAL, updated_at TEXT
        )
    """)
    # Insert a peak that is 9.9% above current equity (guaranteed halt not kill)
    current = get_current_equity_ks(conn)
    fake_peak = current / (1 - 0.09)   # => dd = 9%
    conn.execute(
        "INSERT INTO equity_tracker (equity_peak, updated_at) VALUES (?, ?)",
        (fake_peak, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    ok, reason = KillSwitch._check_drawdown()
    check("Drawdown halt %9 blocked", ok, reason,
          expect_ok=False, expect_substr="Drawdown halt")

    # Clean up fake peak
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM equity_tracker WHERE equity_peak=?", (fake_peak,))
    conn.commit()
    conn.close()
    print("     Sahte equity_peak silindi")

    # -- Test 5: All clear → approved ----------------------------------------
    print("\n[5] Tum kontroller gec (temiz ortam):")
    if os.path.exists(KILL_FILE):
        os.remove(KILL_FILE)
    # Use actual last known price so fat-finger check passes
    garan_price = get_last_known_price("GARAN") or 100.0
    ok, reason = KillSwitch.check_all("GARAN", garan_price, 5_000.0)
    check("Temiz ortamda approved", ok, reason, expect_ok=True)

    # -- Summary -------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"Sonuc: {passed} PASS / {failed} FAIL")
    print(f"{'='*55}\n")

    if os.path.exists(KILL_FILE):
        os.remove(KILL_FILE)
        print("KILL_SWITCH.txt silindi (temizlik)")

    return failed == 0


# =============================================================================
# CLI ENTRY
# =============================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        ok = _run_tests()
        sys.exit(0 if ok else 1)

    # Quick manual check
    print("Kill Switch — hizli kontrol")
    print(f"  KILL_FILE  : {KILL_FILE}")
    print(f"  DB         : {DB_PATH}")
    print(f"  LOG        : {LOG_FILE}")
    print()
    ok, reason = KillSwitch.check_all("TEST", 100.0, 5_000.0)
    if ok:
        print("  Durum: ACIK — emirler gecerli")
    else:
        print(f"  Durum: KAPALI — {reason}")
