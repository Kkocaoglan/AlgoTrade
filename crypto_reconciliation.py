"""
crypto_reconciliation.py -- Crypto pozisyon ve emir tutarlilik kontrolu
------------------------------------------------------------------------
Crypto modulu icin DB consistency checks. Hicbir seyi otomatik duzeltmez
-- sadece raporlar. HIGH severity sorunlar Telegram ile bildirilir.

Kontroller:
  1. GHOST_POSITION  : crypto_positions open ama emir dolumu yok
  2. MISSING_FILL    : crypto_orders FILLED ama pozisyon yok
  3. SIZE_MISMATCH   : pozisyon amount_usdt vs order amount_usdt > 1 USDT
  4. NEGATIVE_BALANCE: kapanmis pozisyonlarin pnl toplami tutarsiz
  5. ORPHAN_ORDER    : emir 30+ dakika 'open' statusunde kalmis
  6. PNL_MISMATCH    : closed pnl_usdt toplami vs journal toplami > 0.5 USDT

Kullanim:
  from crypto_reconciliation import crypto_recon
  is_ok, issues = crypto_recon.run()

  python3.12 crypto_reconciliation.py --report

Isolation: Only imports telegram_bot, logger, sqlite3. No spot/VIOP imports.
"""

import sys
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from logger import algo_log
    HAS_LOG = True
except Exception:
    HAS_LOG = False

BASE_DIR    = Path(__file__).parent
DB_PATH     = str(BASE_DIR / "trade_data.db")
RECON_LOG   = str(BASE_DIR / "logs" / "crypto_reconciliation.log")
RESULTS_DIR = BASE_DIR / "results"

# Tolerances
SIZE_TOL_USDT    = 1.0    # USDT -- position vs order size difference
PNL_TOL_USDT     = 0.5    # USDT -- closed pnl vs journal total
ORPHAN_MINUTES   = 30     # minutes -- order stuck in 'open' status


class CryptoReconciliation:

    DB_PATH = DB_PATH

    def run(self) -> tuple[bool, list[dict]]:
        """
        Run all crypto consistency checks.

        Returns: (is_ok: bool, issues: list[dict])
        """
        issues = []

        issues += self._check_ghost_positions()
        issues += self._check_missing_fills()
        issues += self._check_size_mismatch()
        issues += self._check_negative_balance()
        issues += self._check_orphan_orders()
        issues += self._check_pnl_mismatch()

        self._log_issues(issues)
        self._save_report(issues)

        if issues:
            self._send_telegram_alert(issues)
        else:
            now = datetime.now(timezone.utc).strftime("%H:%M")
            print(f"[CRYPTO RECON] OK -- {now} UTC tutarsizlik yok")

        return len(issues) == 0, issues

    # =========================================================================
    # CHECK 1 -- GHOST_POSITION: open position with no matching filled order
    # =========================================================================

    def _check_ghost_positions(self) -> list[dict]:
        """
        An open position should have at least one filled buy/sell order
        linked via position_id or matching symbol+entry time.
        """
        issues = []
        try:
            conn = sqlite3.connect(self.DB_PATH)

            positions = conn.execute(
                "SELECT id, symbol, entry_date, amount_usdt "
                "FROM crypto_positions WHERE status = 'open'"
            ).fetchall()

            for pos_id, symbol, entry_date, amount_usdt in positions:
                # Check for a matching filled order by position_id
                order = conn.execute(
                    "SELECT order_id FROM crypto_orders "
                    "WHERE position_id = ? AND status = 'FILLED'",
                    (pos_id,),
                ).fetchone()

                if order is None:
                    # Fallback: check by symbol + similar timestamp (within 5s)
                    order = conn.execute(
                        "SELECT order_id FROM crypto_orders "
                        "WHERE symbol = ? AND status = 'FILLED' "
                        "AND side = 'buy' "
                        "AND abs(julianday(filled_at) - julianday(?)) < 5.0/86400.0",
                        (symbol, entry_date),
                    ).fetchone()

                if order is None:
                    issues.append({
                        "type":     "GHOST_POSITION",
                        "severity": "HIGH",
                        "msg":      f"{symbol} pos_id={pos_id} acik ama eslesen FILLED emir yok "
                                    f"(entry={entry_date}, size={amount_usdt:.2f} USDT)",
                        "action":   "crypto_oms DB kontrol et, pozisyonu dogrula",
                    })

            conn.close()
        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg": f"_check_ghost_positions hatasi: {e}",
                "action": "DB kontrol et",
            })
        return issues

    # =========================================================================
    # CHECK 2 -- MISSING_FILL: filled order with no corresponding position
    # =========================================================================

    def _check_missing_fills(self) -> list[dict]:
        """
        A FILLED buy order (order_role='entry' or side='buy') should have
        a matching crypto_positions row.
        """
        issues = []
        try:
            conn = sqlite3.connect(self.DB_PATH)

            # Find filled entry orders without a matching position
            orders = conn.execute("""
                SELECT o.order_id, o.symbol, o.amount_usdt, o.filled_at, o.position_id
                FROM crypto_orders o
                WHERE o.status = 'FILLED'
                  AND (o.order_role = 'entry' OR (o.side = 'buy' AND o.order_role IS NULL))
                  AND o.position_id IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM crypto_positions p
                    WHERE p.id = o.position_id
                  )
            """).fetchall()

            for order_id, symbol, amount_usdt, filled_at, position_id in orders:
                issues.append({
                    "type":     "MISSING_FILL",
                    "severity": "HIGH",
                    "msg":      f"{symbol} order={order_id} FILLED ama pozisyon satiri yok "
                                f"(pos_id={position_id}, size={amount_usdt:.2f} USDT, "
                                f"filled={filled_at})",
                    "action":   "crypto_positions tablosunu kontrol et",
                })

            conn.close()
        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg": f"_check_missing_fills hatasi: {e}",
                "action": "DB kontrol et",
            })
        return issues

    # =========================================================================
    # CHECK 3 -- SIZE_MISMATCH: position amount_usdt vs order amount_usdt > 1
    # =========================================================================

    def _check_size_mismatch(self) -> list[dict]:
        """
        For open positions, compare position.amount_usdt with
        the corresponding entry order.amount_usdt.
        """
        issues = []
        try:
            conn = sqlite3.connect(self.DB_PATH)

            rows = conn.execute("""
                SELECT p.id, p.symbol, p.amount_usdt, o.amount_usdt, o.order_id
                FROM crypto_positions p
                JOIN crypto_orders o ON o.position_id = p.id
                WHERE p.status = 'open'
                  AND o.status = 'FILLED'
                  AND (o.order_role = 'entry' OR (o.side = 'buy' AND o.order_role IS NULL))
            """).fetchall()

            for pos_id, symbol, pos_usdt, order_usdt, order_id in rows:
                diff = abs(float(pos_usdt) - float(order_usdt))
                if diff > SIZE_TOL_USDT:
                    issues.append({
                        "type":     "SIZE_MISMATCH",
                        "severity": "MEDIUM",
                        "msg":      f"{symbol} pos_id={pos_id} size fark: "
                                    f"position={pos_usdt:.2f} vs order={order_usdt:.2f} "
                                    f"(diff={diff:.2f} USDT)",
                        "action":   "Hangi degerin dogru oldugunu dogrula",
                    })

            conn.close()
        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg": f"_check_size_mismatch hatasi: {e}",
                "action": "DB kontrol et",
            })
        return issues

    # =========================================================================
    # CHECK 4 -- NEGATIVE_BALANCE: cumulative realized PnL inconsistency
    # =========================================================================

    def _check_negative_balance(self) -> list[dict]:
        """
        Sum of all closed positions' pnl_usdt should be internally consistent:
        no individual closed position should have NULL pnl_usdt, and the total
        should match the sum of (exit_price - entry_price) * amount_coin.
        """
        issues = []
        try:
            conn = sqlite3.connect(self.DB_PATH)

            # Check for NULL pnl on closed positions
            null_pnl = conn.execute(
                "SELECT COUNT(*) FROM crypto_positions "
                "WHERE status = 'closed' AND pnl_usdt IS NULL"
            ).fetchone()[0]

            if null_pnl > 0:
                issues.append({
                    "type":     "NEGATIVE_BALANCE",
                    "severity": "HIGH",
                    "msg":      f"{null_pnl} kapanmis pozisyon(lar) pnl_usdt = NULL",
                    "action":   "crypto_positions tablosunda pnl hesapla",
                })

            # Cross-check: sum(pnl_usdt) vs recomputed from prices
            rows = conn.execute("""
                SELECT id, symbol, direction, entry_price, exit_price,
                       amount_coin, pnl_usdt
                FROM crypto_positions
                WHERE status = 'closed' AND pnl_usdt IS NOT NULL
                      AND exit_price IS NOT NULL
            """).fetchall()

            for pos_id, symbol, direction, entry_p, exit_p, coins, stored_pnl in rows:
                if direction == "long":
                    computed = (float(exit_p) - float(entry_p)) * float(coins)
                else:
                    computed = (float(entry_p) - float(exit_p)) * float(coins)
                diff = abs(computed - float(stored_pnl))
                if diff > SIZE_TOL_USDT:
                    issues.append({
                        "type":     "NEGATIVE_BALANCE",
                        "severity": "MEDIUM",
                        "msg":      f"{symbol} pos_id={pos_id} pnl tutarsiz: "
                                    f"stored={float(stored_pnl):.4f} vs "
                                    f"computed={computed:.4f} (diff={diff:.4f} USDT)",
                        "action":   "pnl_usdt degerini yeniden hesapla",
                    })

            conn.close()
        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg": f"_check_negative_balance hatasi: {e}",
                "action": "DB kontrol et",
            })
        return issues

    # =========================================================================
    # CHECK 5 -- ORPHAN_ORDER: order stuck in 'open' for > 30 minutes
    # =========================================================================

    def _check_orphan_orders(self) -> list[dict]:
        """
        In paper mode, all orders should be instantly FILLED or REJECTED.
        Any order stuck in non-terminal status for > ORPHAN_MINUTES is orphaned.
        """
        issues = []
        try:
            conn = sqlite3.connect(self.DB_PATH)

            rows = conn.execute(f"""
                SELECT order_id, symbol, side, amount_usdt, created_at
                FROM crypto_orders
                WHERE status NOT IN ('FILLED', 'REJECTED', 'CANCELLED')
                  AND created_at < datetime('now', '-{ORPHAN_MINUTES} minutes')
            """).fetchall()

            for order_id, symbol, side, amount_usdt, created_at in rows:
                issues.append({
                    "type":     "ORPHAN_ORDER",
                    "severity": "LOW",
                    "msg":      f"{symbol} {side} emir {ORPHAN_MINUTES}+ dk 'open' kaldi: "
                                f"order={order_id}, size={amount_usdt:.2f} USDT "
                                f"(created={created_at})",
                    "action":   "Emri CANCELLED olarak guncelle",
                })

            conn.close()
        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg": f"_check_orphan_orders hatasi: {e}",
                "action": "DB kontrol et",
            })
        return issues

    # =========================================================================
    # CHECK 6 -- PNL_MISMATCH: closed pnl sum vs daily journal total
    # =========================================================================

    def _check_pnl_mismatch(self) -> list[dict]:
        """
        Compare sum of closed crypto_positions.pnl_usdt (today UTC)
        with what crypto_journal recorded. Tolerance: PNL_TOL_USDT.
        """
        issues = []
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            conn = sqlite3.connect(self.DB_PATH)

            # Sum of closed positions today
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl_usdt), 0) FROM crypto_positions "
                "WHERE status = 'closed' AND exit_date LIKE ?",
                (f"{today_utc}%",),
            ).fetchone()
            closed_pnl = float(row[0]) if row else 0.0

            # Check if crypto_journal table exists
            has_journal = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='crypto_journal'"
            ).fetchone()

            if has_journal:
                jr = conn.execute(
                    "SELECT COALESCE(SUM(pnl_usdt), 0) FROM crypto_journal "
                    "WHERE date = ?",
                    (today_utc,),
                ).fetchone()
                journal_pnl = float(jr[0]) if jr else 0.0

                diff = abs(closed_pnl - journal_pnl)
                if diff > PNL_TOL_USDT and journal_pnl != 0.0:
                    issues.append({
                        "type":     "PNL_MISMATCH",
                        "severity": "LOW",
                        "msg":      f"Gunluk PnL tutarsiz ({today_utc}): "
                                    f"closed_positions={closed_pnl:.4f} USDT vs "
                                    f"journal={journal_pnl:.4f} USDT "
                                    f"(diff={diff:.4f} USDT)",
                        "action":   "crypto_journal.py ile yeniden hesapla",
                    })

            conn.close()
        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg": f"_check_pnl_mismatch hatasi: {e}",
                "action": "DB kontrol et",
            })
        return issues

    # =========================================================================
    # LOGGING / REPORTING
    # =========================================================================

    def _log_issues(self, issues: list[dict]) -> None:
        os.makedirs(os.path.dirname(RECON_LOG), exist_ok=True)
        with open(RECON_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC ===\n")
            if not issues:
                f.write("OK -- tutarsizlik yok\n")
                if HAS_LOG:
                    algo_log.system("[CRYPTO RECON] OK -- no inconsistency found")
                return
            for issue in issues:
                f.write(
                    f"[{issue['severity']}] {issue['type']}: {issue['msg']}\n"
                    f"  -> {issue['action']}\n"
                )
        for issue in issues:
            print(f"[CRYPTO RECON] [{issue['severity']}] {issue['type']}: {issue['msg']}")
            if HAS_LOG:
                log_msg = (
                    f"[CRYPTO RECON] [{issue['severity']}] {issue['type']}: "
                    f"{issue['msg']} | action: {issue['action']}"
                )
                if issue["severity"] == "HIGH":
                    algo_log.risk(log_msg)
                else:
                    algo_log.system(log_msg)

    def _send_telegram_alert(self, issues: list[dict]) -> None:
        """Send Telegram alert for HIGH severity issues only."""
        high = [i for i in issues if i["severity"] == "HIGH"]
        if not high:
            return
        try:
            from telegram_bot import send_telegram_alert
            lines = ["[!] CRYPTO RECONCILIATION UYARISI"]
            for i in high:
                lines.append(f"  {i['type']}: {i['msg']}")
            send_telegram_alert("\n".join(lines))
        except Exception:
            pass

    def _save_report(self, issues: list[dict]) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = RESULTS_DIR / f"crypto_reconciliation_{today}.txt"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now(timezone.utc):%H:%M:%S} UTC ===\n")
            if not issues:
                f.write("OK -- tutarsizlik yok\n")
                return
            for i in issues:
                f.write(
                    f"[{i['severity']}] {i['type']}: {i['msg']}\n"
                    f"  Aksiyon: {i['action']}\n"
                )


# =============================================================================
# Global instance
# =============================================================================
crypto_recon = CryptoReconciliation()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if "--report" in sys.argv:
        print("=" * 55)
        print("CRYPTO RECONCILIATION REPORT")
        print("=" * 55)
        is_ok, issues = crypto_recon.run()
        print()
        print(f"Sonuc: {'TEMIZ' if is_ok else f'{len(issues)} SORUN TESPIT EDILDI'}")
        print("=" * 55)
    else:
        print("Kullanim: python3.12 crypto_reconciliation.py --report")
        sys.exit(0)
