"""
viop_oms.py -- VIOP Order Management System
---------------------------------------------
Completely separate from the spot OMS (order_log table untouched).
Uses viop_orders table in trade_data.db.

Directions : LONG_OPEN | SHORT_OPEN | LONG_CLOSE | SHORT_CLOSE
Status     : NEW -> FILLED | REJECTED
"""

import sqlite3
import uuid
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "trade_data.db"

VALID_DIRECTIONS = frozenset({"LONG_OPEN", "SHORT_OPEN", "LONG_CLOSE", "SHORT_CLOSE"})


# ── DB setup ─────────────────────────────────────────────────────────
def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS viop_orders (
            order_id        TEXT PRIMARY KEY,
            symbol          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            size_tl         REAL,
            contracts       INTEGER,
            entry_price     REAL,
            exit_price      REAL,
            status          TEXT DEFAULT 'NEW',
            created_at      TEXT,
            filled_at       TEXT,
            reason          TEXT,
            rejected_reason TEXT
        )
    """)
    conn.commit()


# ── OMS class ────────────────────────────────────────────────────────
class ViopOMS:
    """
    Minimal order lifecycle for VIOP positions.
    Paper mode: orders fill immediately if validations pass.

    NEW --> FILLED | REJECTED
    """

    def __init__(self):
        conn = sqlite3.connect(DB_PATH)
        _ensure_table(conn)
        conn.close()

    # ── Public API ───────────────────────────────────────────────────
    def create_order(self, symbol: str, direction: str,
                     size_tl: float, contracts: int,
                     price: float, reason: str = "") -> dict:
        """
        Create and immediately fill (paper) a VIOP order.
        Returns order dict with status FILLED or REJECTED.
        """
        order_id = str(uuid.uuid4())[:12]
        now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Validation
        if direction not in VALID_DIRECTIONS:
            return self._reject(order_id, symbol, direction, size_tl,
                                contracts, price, reason,
                                f"Invalid direction: {direction}", now)

        if contracts <= 0:
            return self._reject(order_id, symbol, direction, size_tl,
                                contracts, price, reason,
                                "contracts <= 0", now)

        if price <= 0:
            return self._reject(order_id, symbol, direction, size_tl,
                                contracts, price, reason,
                                "price <= 0", now)

        # All OK -- fill immediately (paper)
        filled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        _ensure_table(conn)
        conn.execute("""
            INSERT INTO viop_orders
                (order_id, symbol, direction, size_tl, contracts,
                 entry_price, status, created_at, filled_at, reason)
            VALUES (?, ?, ?, ?, ?, ?, 'FILLED', ?, ?, ?)
        """, (order_id, symbol, direction, size_tl, contracts,
              price, now, filled_at, reason))
        conn.commit()
        conn.close()

        return {
            "order_id": order_id, "symbol": symbol,
            "direction": direction, "size_tl": size_tl,
            "contracts": contracts, "price": price,
            "status": "FILLED", "created_at": now,
            "filled_at": filled_at, "reason": reason,
        }

    def update_exit(self, order_id: str, exit_price: float,
                    reason: str = "") -> bool:
        """Record exit price on an existing order (audit trail for closes)."""
        filled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        _ensure_table(conn)
        conn.execute(
            "UPDATE viop_orders SET exit_price=?, filled_at=?, reason=? WHERE order_id=?",
            (exit_price, filled_at, reason, order_id)
        )
        conn.commit()
        conn.close()
        return True

    def summary(self):
        """Print today's order summary."""
        today = date.today().isoformat()
        conn  = sqlite3.connect(DB_PATH)
        _ensure_table(conn)
        rows = conn.execute("""
            SELECT direction, status, contracts, entry_price, reason
            FROM viop_orders
            WHERE DATE(created_at) = ?
            ORDER BY created_at
        """, (today,)).fetchall()
        conn.close()

        if not rows:
            print("[VIOP OMS] Bugun emir yok")
            return
        print(f"[VIOP OMS] Bugun {len(rows)} emir:")
        for r in rows:
            print(f"  {r[0]:<14} {r[1]:<10} {r[2]} sozlesme @ {r[3]:.2f}  {r[4]}")

    # ── Internal ────────────────────────────────────────────────────
    def _reject(self, order_id, symbol, direction, size_tl,
                contracts, price, reason, reject_reason, now) -> dict:
        conn = sqlite3.connect(DB_PATH)
        _ensure_table(conn)
        conn.execute("""
            INSERT INTO viop_orders
                (order_id, symbol, direction, size_tl, contracts,
                 entry_price, status, created_at, reason, rejected_reason)
            VALUES (?, ?, ?, ?, ?, ?, 'REJECTED', ?, ?, ?)
        """, (order_id, symbol, direction, size_tl, contracts,
              price, now, reason, reject_reason))
        conn.commit()
        conn.close()
        return {
            "order_id": order_id, "symbol": symbol,
            "direction": direction, "size_tl": size_tl,
            "contracts": contracts, "price": price,
            "status": "REJECTED", "rejected_reason": reject_reason,
        }


# Module-level singleton -- import this everywhere
viop_oms = ViopOMS()
