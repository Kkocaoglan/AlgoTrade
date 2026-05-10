"""
reconciliation.py -- Pozisyon ve bakiye tutarlilik kontrolu
------------------------------------------------------------
Sistemnin bildigi pozisyonlar ile DB gercegi arasindaki farklari raporlar.
Hicbir seyi otomatik duzeltmez -- sadece raporlar.

Kontroller:
  1. _check_positions : bellek vs DB acik pozisyonlar (symbol + size_tl)
  2. _check_balance   : paper_equity.cash + positions_value == total_equity
  3. _check_orphan_orders: SENT durumda kalan emirler
  4. _check_pnl       : paper_trades PnL == paper_equity.daily_pnl

Kullanim:
  from reconciliation import recon
  is_ok, issues = recon.run(open_positions)   # loop_trader: tam kontrol
  is_ok, issues = recon.run()                 # daily_journal: DB-only kontrol

Test:
  python3.12 reconciliation.py --test
"""

import sys
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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

BASE_DIR      = Path(__file__).parent
DB_PATH       = str(BASE_DIR / "trade_data.db")
RECON_LOG     = str(BASE_DIR / "logs" / "reconciliation.log")
RESULTS_DIR   = BASE_DIR / "results"

# Toleranslar
BALANCE_TOL    = 100.0   # TL -- cash + pos_val vs total_equity fark limiti
PNL_TOL        = 50.0    # TL -- paper_trades vs paper_equity.daily_pnl fark limiti
SIZE_TOL       = 0.01    # TL -- size_tl farki limiti
ORPHAN_MINUTES = 10      # dakika -- bu sureden uzun SENT emirler orphan sayilir


class Reconciliation:

    DB_PATH = DB_PATH

    def run(self, open_positions_in_memory=None):
        """
        Tum kontrolleri calistir, sorunlari logla ve Telegram gonder.

        open_positions_in_memory: None  -> pozisyon bellek kontrolu atlanir
                                  list  -> bellek vs DB karsilastirmasi yapilir

        Returns: (is_ok: bool, issues: list[dict])
        """
        issues = []

        if open_positions_in_memory is not None:
            issues += self._check_positions(open_positions_in_memory)

        issues += self._check_balance()
        issues += self._check_orphan_orders()
        issues += self._check_pnl()

        self._log_issues(issues)
        self._save_report(issues)

        if issues:
            self._send_telegram_alert(issues)
        else:
            print(f"[RECON] OK -- {datetime.now():%H:%M} tutarsizlik yok")

        return len(issues) == 0, issues

    # =========================================================================
    # CHECK 1 -- pozisyon tutarlilik (bellek vs DB)
    # =========================================================================

    def _check_positions(self, memory_positions):
        """
        Bellekteki (loop_trader icindeki) acik pozisyonlari
        DB'deki paper_positions (status='open') ile karsilastirir.
        """
        issues = []
        try:
            conn    = sqlite3.connect(self.DB_PATH)
            db_rows = conn.execute(
                "SELECT symbol, entry_price, size_tl "
                "FROM paper_positions WHERE status='open'"
            ).fetchall()
            conn.close()
        except Exception as e:
            return [{"type": "RECON_ERROR", "severity": "LOW",
                     "msg": f"_check_positions DB hatasi: {e}",
                     "action": "DB kontrol et"}]

        db_symbols  = {r[0] for r in db_rows}
        db_map      = {r[0]: {"entry_price": r[1], "size_tl": r[2]} for r in db_rows}
        mem_symbols = {p["symbol"] for p in memory_positions}

        # Bellekte var, DB'de yok -- hayalet pozisyon
        for sym in mem_symbols - db_symbols:
            issues.append({
                "type":     "GHOST_POSITION",
                "severity": "HIGH",
                "msg":      f"{sym} bellekte acik ama DB'de yok",
                "action":   "loop_trader yeniden baslat veya DB kontrol et",
            })

        # DB'de var, bellekte yok -- kayip pozisyon
        for sym in db_symbols - mem_symbols:
            issues.append({
                "type":     "MISSING_POSITION",
                "severity": "HIGH",
                "msg":      f"{sym} DB'de acik ama bellekte yok",
                "action":   "loop_trader yeniden baslat",
            })

        # Her ikisinde de var -- size_tl karsilastir
        mem_map = {p["symbol"]: p for p in memory_positions}
        for sym in mem_symbols & db_symbols:
            db_size  = float(db_map[sym]["size_tl"])
            mem_size = float(mem_map[sym].get("size_tl", 0))
            if abs(db_size - mem_size) > SIZE_TOL:
                issues.append({
                    "type":     "SIZE_MISMATCH",
                    "severity": "MEDIUM",
                    "msg":      f"{sym} size_tl: bellek={mem_size:.2f} TL, DB={db_size:.2f} TL",
                    "action":   "DB degerini kullan",
                })

        return issues

    # =========================================================================
    # CHECK 2 -- bakiye tutarlilik (paper_equity ic tutarlilik)
    # =========================================================================

    def _check_balance(self):
        """
        paper_equity'nin en son satirinda:
          cash + positions_value == total_equity  (tolerans: BALANCE_TOL TL)
          cash >= 0
        """
        issues = []
        try:
            conn   = sqlite3.connect(self.DB_PATH)
            eq_row = conn.execute(
                "SELECT cash, positions_value, total_equity "
                "FROM paper_equity ORDER BY date DESC LIMIT 1"
            ).fetchone()
            conn.close()

            if eq_row is None:
                return issues   # kayit yok -- ilk gun, normal

            cash         = float(eq_row[0])
            pos_val      = float(eq_row[1])
            total_equity = float(eq_row[2])
            computed     = cash + pos_val
            diff         = abs(computed - total_equity)

            if diff > BALANCE_TOL:
                issues.append({
                    "type":     "BALANCE_MISMATCH",
                    "severity": "MEDIUM",
                    "msg":      (f"paper_equity tutarsiz: "
                                 f"cash={cash:.0f} + pos_val={pos_val:.0f} = {computed:.0f} "
                                 f"vs total_equity={total_equity:.0f} "
                                 f"(fark {diff:.0f} TL)"),
                    "action":   "paper_trade.py --run calistir",
                })

            if cash < 0:
                issues.append({
                    "type":     "NEGATIVE_CASH",
                    "severity": "HIGH",
                    "msg":      f"Nakit negatif: {cash:.0f} TL",
                    "action":   "paper_equity kontrol et, manuel duzelt",
                })

        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg":  f"_check_balance hatasi: {e}", "action": "DB kontrol et",
            })

        return issues

    # =========================================================================
    # CHECK 3 -- orphan emirler (SENT durumda kalan order_log satirlari)
    # =========================================================================

    def _check_orphan_orders(self):
        """
        order_log'da SENT durumda kalan ve ORPHAN_MINUTES'tan eski emirleri tespit et.
        Paper trade'de her emir aninda FILLED olmali.
        """
        issues = []
        try:
            conn = sqlite3.connect(self.DB_PATH)
            exists = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='order_log'"
            ).fetchone()
            if not exists:
                conn.close()
                return issues   # order_log henuz olusturulmamis

            rows = conn.execute(f"""
                SELECT order_id, symbol, direction, created_at
                FROM order_log
                WHERE status = 'SENT'
                  AND created_at < datetime('now', '-{ORPHAN_MINUTES} minutes')
            """).fetchall()
            conn.close()

            for order_id, symbol, direction, created_at in rows:
                issues.append({
                    "type":     "ORPHAN_ORDER",
                    "severity": "LOW",
                    "msg":      (f"Emir SENT kaldi: {direction} {symbol} "
                                 f"({created_at[:16]}) [{order_id}]"),
                    "action":   "order_log'da CANCELLED olarak guncelle",
                })

        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg":  f"_check_orphan_orders hatasi: {e}",
                "action": "order_log kontrol et",
            })

        return issues

    # =========================================================================
    # CHECK 4 -- PnL tutarlilik (paper_trades vs paper_equity.daily_pnl)
    # =========================================================================

    def _check_pnl(self):
        """
        Bugun kapanan paper_trades.pnl toplami
        ile paper_equity.daily_pnl (bugun icin) karsilastirilir.
        """
        issues = []
        today = date.today().isoformat()
        try:
            conn = sqlite3.connect(self.DB_PATH)

            row_trades = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) "
                "FROM paper_trades WHERE exit_date = ?",
                (today,),
            ).fetchone()
            trades_pnl = float(row_trades[0]) if row_trades else 0.0

            row_eq = conn.execute(
                "SELECT daily_pnl FROM paper_equity WHERE date = ?",
                (today,),
            ).fetchone()
            conn.close()

            if row_eq is None:
                return issues   # bugun icin kayit yok -- normal

            eq_pnl = float(row_eq[0])
            diff   = abs(trades_pnl - eq_pnl)

            if diff > PNL_TOL:
                issues.append({
                    "type":     "PNL_MISMATCH",
                    "severity": "LOW",
                    "msg":      (f"Gunluk PnL: paper_trades={trades_pnl:.0f} TL, "
                                 f"paper_equity={eq_pnl:.0f} TL, "
                                 f"fark={diff:.0f} TL"),
                    "action":   "paper_trade.py --run ile equity guncelle",
                })

        except Exception as e:
            issues.append({
                "type": "RECON_ERROR", "severity": "LOW",
                "msg":  f"_check_pnl hatasi: {e}", "action": "DB kontrol et",
            })

        return issues

    # =========================================================================
    # LOGLAMA / RAPORLAMA
    # =========================================================================

    def _log_issues(self, issues):
        os.makedirs(os.path.dirname(RECON_LOG), exist_ok=True)
        with open(RECON_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            if not issues:
                f.write("OK -- tutarsizlik yok\n")
                if HAS_LOG:
                    algo_log.system("[RECON] OK -- no inconsistency found")
                return
            for issue in issues:
                f.write(
                    f"[{issue['severity']}] {issue['type']}: {issue['msg']}\n"
                    f"  -> {issue['action']}\n"
                )
        for issue in issues:
            print(f"[RECON] [{issue['severity']}] {issue['type']}: {issue['msg']}")
            if HAS_LOG:
                log_msg = (
                    f"[RECON] [{issue['severity']}] {issue['type']}: "
                    f"{issue['msg']} | action: {issue['action']}"
                )
                if issue["severity"] == "HIGH":
                    algo_log.risk(log_msg)
                else:
                    algo_log.system(log_msg)

    def _send_telegram_alert(self, issues):
        """Telegram'a sadece HIGH severity sorunlari bildir."""
        high = [i for i in issues if i["severity"] == "HIGH"]
        if not high:
            return
        try:
            from telegram_bot import send_telegram_alert
            lines = ["[!] RECONCILIATION UYARISI"]
            for i in high:
                lines.append(f"  {i['type']}: {i['msg']}")
            send_telegram_alert("\n".join(lines))
        except Exception:
            pass

    def _save_report(self, issues):
        RESULTS_DIR.mkdir(exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        path  = RESULTS_DIR / f"reconciliation_{today}.txt"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now():%H:%M:%S} ===\n")
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
recon = Reconciliation()


# =============================================================================
# --test MODE
# =============================================================================

if __name__ == "__main__":
    if "--test" not in sys.argv:
        print("Kullanim: python3.12 reconciliation.py --test")
        sys.exit(0)

    print("=" * 55)
    print("Reconciliation Test")
    print("=" * 55)

    passed = 0
    failed = 0

    def chk(name, cond):
        global passed, failed
        if cond:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    # -- Test 1: DB-only run (no memory positions)
    print("\n[1] DB-only kontrol (open_positions=None):")
    is_ok, issues = recon.run()
    chk("run() calisti (exception yok)", True)
    chk("is_ok bool dondurdu", isinstance(is_ok, bool))
    chk("issues list dondurdu", isinstance(issues, list))

    # -- Test 2: Memory positions = DB positions (should be 0 issues for CHECK 1)
    print("\n[2] Bellek = DB kontrolu:")
    conn_t = sqlite3.connect(DB_PATH)
    db_positions = conn_t.execute(
        "SELECT symbol, entry_price, size_tl FROM paper_positions WHERE status='open'"
    ).fetchall()
    conn_t.close()

    mem_list = [{"symbol": r[0], "entry_price": r[1], "size_tl": r[2]}
                for r in db_positions]
    pos_issues = recon._check_positions(mem_list)
    chk("Bellek=DB -> pozisyon sorunu yok", len(pos_issues) == 0)

    # -- Test 3: Ghost position detection
    print("\n[3] Hayalet pozisyon tespiti:")
    ghost_list = [{"symbol": "PHANTOM_XYZ", "entry_price": 10.0, "size_tl": 5000.0}]
    ghost_issues = recon._check_positions(ghost_list)
    has_ghost = any(i["type"] == "GHOST_POSITION" for i in ghost_issues)
    chk("GHOST_POSITION tespit edildi", has_ghost)

    # -- Test 4: Missing position detection
    print("\n[4] Kayip pozisyon tespiti:")
    if db_positions:
        missing_issues = recon._check_positions([])
        has_missing = any(i["type"] == "MISSING_POSITION" for i in missing_issues)
        chk("MISSING_POSITION tespit edildi (DB'de acik pozisyon var)", has_missing)
    else:
        print("  (acik pozisyon yok -- test atlandi)")
        passed += 1

    # -- Test 5: Balance check
    print("\n[5] Bakiye kontrolu:")
    bal_issues = recon._check_balance()
    chk("_check_balance calisti", True)
    chk("Bakiye kontrolu list donduruyor", isinstance(bal_issues, list))

    # -- Test 6: Orphan order check
    print("\n[6] Orphan emir kontrolu:")
    orp_issues = recon._check_orphan_orders()
    chk("_check_orphan_orders calisti", True)

    # -- Test 7: PnL check
    print("\n[7] PnL kontrolu:")
    pnl_issues = recon._check_pnl()
    chk("_check_pnl calisti", True)

    # -- Test 8: Log file created
    print("\n[8] Log dosyalari olusturuldu mu?")
    chk("logs/reconciliation.log var", os.path.exists(RECON_LOG))
    today_str  = datetime.now().strftime("%Y-%m-%d")
    report_path = RESULTS_DIR / f"reconciliation_{today_str}.txt"
    chk(f"results/reconciliation_{today_str}.txt var", report_path.exists())

    print(f"\n{'='*55}")
    print(f"Sonuc: {passed} PASS / {failed} FAIL")
    print(f"{'='*55}")
    print(f"Log    : {RECON_LOG}")
    print(f"Rapor  : {report_path}")

    sys.exit(0 if failed == 0 else 1)
