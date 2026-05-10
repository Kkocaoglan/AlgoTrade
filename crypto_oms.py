"""
crypto_oms.py — Paper OMS for Crypto Module.

Manages crypto_positions and crypto_orders tables in trade_data.db.
Completely isolated from oms.py (spot) and viop_oms.py (VIOP).

DB tables created on first import:
  crypto_positions — open/closed positions
  crypto_orders    — order log
"""

import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DB_PATH = Path(__file__).parent / "trade_data.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _ensure_tables() -> None:
    """Create crypto_positions and crypto_orders if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS crypto_positions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol       TEXT    NOT NULL,
        direction    TEXT    NOT NULL,   -- 'long' or 'short'
        entry_date   TEXT    NOT NULL,
        entry_price  REAL    NOT NULL,
        amount_usdt  REAL    NOT NULL,
        amount_coin  REAL    NOT NULL,
        stop_price   REAL    NOT NULL,
        target_price REAL    NOT NULL,
        status       TEXT    NOT NULL DEFAULT 'open',  -- 'open' / 'closed'
        exit_date    TEXT,
        exit_price   REAL,
        pnl_usdt     REAL,
        exit_reason  TEXT
    );

    CREATE TABLE IF NOT EXISTS crypto_orders (
        order_id       TEXT PRIMARY KEY,
        symbol         TEXT NOT NULL,
        side           TEXT NOT NULL,    -- 'buy' or 'sell'
        amount_usdt    REAL NOT NULL,
        amount_coin    REAL NOT NULL,
        price          REAL NOT NULL,
        status         TEXT NOT NULL,    -- 'FILLED' / 'REJECTED'
        paper_mode     INTEGER NOT NULL, -- 1=paper, 0=live
        created_at     TEXT NOT NULL,
        filled_at      TEXT,
        ccxt_order_id  TEXT
    );
    """
    with _get_conn() as conn:
        conn.executescript(ddl)
        _ensure_column(conn, "crypto_positions", "signal_id", "signal_id TEXT")
        _ensure_column(conn, "crypto_positions", "entry_reason", "entry_reason TEXT")
        _ensure_column(conn, "crypto_positions", "parent_position_id", "parent_position_id INTEGER")
        _ensure_column(conn, "crypto_positions", "closed_fraction", "closed_fraction REAL")
        _ensure_column(conn, "crypto_positions", "partial_exit_done", "partial_exit_done INTEGER DEFAULT 0")
        _ensure_column(conn, "crypto_orders", "signal_id", "signal_id TEXT")
        _ensure_column(conn, "crypto_orders", "position_id", "position_id INTEGER")
        _ensure_column(conn, "crypto_orders", "order_role", "order_role TEXT")


# Ensure tables exist at import time
_ensure_tables()


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class CryptoOMS:
    """
    Order and position manager for the Crypto module.

    All methods work against trade_data.db (crypto_positions + crypto_orders).
    Thread-safe for single-process use; not multi-process safe.
    """

    # ── BUY / SHORT open ─────────────────────────────────────────────────────

    def create_buy(
        self,
        symbol: str,
        amount_usdt: float,
        price: float,
        stop_pct: float,
        target_pct: float,
        direction: str = "long",
        signal_id: str | None = None,
        entry_reason: str | None = None,
    ) -> dict:
        """
        Open a new position (long or short).

        For long:  stop = entry*(1-stop_pct), target = entry*(1+target_pct)
        For short: stop = entry*(1+stop_pct), target = entry*(1-target_pct)

        Returns position dict on success, or dict with status='REJECTED'.
        """
        if price <= 0 or amount_usdt <= 0:
            return {"status": "REJECTED", "reason": "invalid price or amount"}

        amount_coin = round(amount_usdt / price, 8)
        now = datetime.now(timezone.utc).isoformat()

        if direction == "long":
            stop_price   = round(price * (1 - stop_pct), 8)
            target_price = round(price * (1 + target_pct), 8)
        else:
            stop_price   = round(price * (1 + stop_pct), 8)
            target_price = round(price * (1 - target_pct), 8)

        order_id = str(uuid.uuid4())

        with _get_conn() as conn:
            # Insert order log
            conn.execute(
                """INSERT INTO crypto_orders
                   (order_id, symbol, side, amount_usdt, amount_coin, price,
                    status, paper_mode, created_at, filled_at, signal_id, position_id, order_role)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (order_id, symbol, "buy", amount_usdt, amount_coin, price,
                 "FILLED", 1, now, now, signal_id, None, "entry"),
            )
            # Insert position
            cursor = conn.execute(
                """INSERT INTO crypto_positions
                   (symbol, direction, entry_date, entry_price, amount_usdt,
                    amount_coin, stop_price, target_price, status, signal_id, entry_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, direction, now, price, amount_usdt,
                 amount_coin, stop_price, target_price, "open", signal_id, entry_reason),
            )
            pos_id = cursor.lastrowid
            conn.execute(
                "UPDATE crypto_orders SET position_id=? WHERE order_id=?",
                (pos_id, order_id),
            )

        return {
            "status":       "FILLED",
            "position_id":  pos_id,
            "order_id":     order_id,
            "symbol":       symbol,
            "direction":    direction,
            "entry_price":  price,
            "amount_usdt":  amount_usdt,
            "amount_coin":  amount_coin,
            "stop_price":   stop_price,
            "target_price": target_price,
            "entry_date":   now,
            "signal_id":    signal_id,
            "entry_reason": entry_reason,
        }

    # ── SELL / close ──────────────────────────────────────────────────────────

    def create_sell(
        self,
        symbol: str,
        position_id: int,
        price: float,
        reason: str = "manual",
        signal_id: str | None = None,
        close_fraction: float = 1.0,
    ) -> dict:
        """
        Close an open position.

        Calculates P&L, updates crypto_positions, logs to crypto_orders.
        Returns result dict with pnl_usdt.
        """
        now = datetime.now(timezone.utc).isoformat()
        close_fraction = float(close_fraction)
        if close_fraction <= 0 or close_fraction > 1:
            return {"status": "REJECTED", "reason": f"invalid close_fraction={close_fraction}"}

        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM crypto_positions WHERE id=? AND status='open'",
                (position_id,),
            ).fetchone()

            if row is None:
                return {"status": "REJECTED", "reason": f"position {position_id} not found or already closed"}

            pos = dict(row)
            position_signal_id = signal_id or pos.get("signal_id")
            if close_fraction >= 0.999999:
                close_amount_usdt = float(pos["amount_usdt"])
                close_amount_coin = float(pos["amount_coin"])
                remaining_amount_usdt = 0.0
                remaining_amount_coin = 0.0
                is_partial = False
            else:
                close_amount_usdt = round(float(pos["amount_usdt"]) * close_fraction, 8)
                close_amount_coin = round(float(pos["amount_coin"]) * close_fraction, 8)
                remaining_amount_usdt = round(float(pos["amount_usdt"]) - close_amount_usdt, 8)
                remaining_amount_coin = round(float(pos["amount_coin"]) - close_amount_coin, 8)
                if close_amount_usdt <= 0 or close_amount_coin <= 0 or remaining_amount_usdt <= 0 or remaining_amount_coin <= 0:
                    return {"status": "REJECTED", "reason": "partial close leaves invalid size"}
                is_partial = True

            pnl = self.calc_pnl({
                "entry_price": pos["entry_price"],
                "amount_coin": close_amount_coin,
                "direction": pos["direction"],
            }, price)

            order_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO crypto_orders
                   (order_id, symbol, side, amount_usdt, amount_coin, price,
                    status, paper_mode, created_at, filled_at, signal_id, position_id, order_role)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (order_id, symbol, "sell", close_amount_usdt, close_amount_coin,
                 price, "FILLED", 1, now, now, position_signal_id, position_id, "exit"),
            )
            if is_partial:
                conn.execute(
                    """INSERT INTO crypto_positions
                       (symbol, direction, entry_date, entry_price, amount_usdt, amount_coin,
                        stop_price, target_price, status, exit_date, exit_price, pnl_usdt,
                        exit_reason, signal_id, entry_reason, parent_position_id, closed_fraction, partial_exit_done)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        pos["symbol"], pos["direction"], pos["entry_date"], pos["entry_price"],
                        close_amount_usdt, close_amount_coin, pos["stop_price"], pos["target_price"],
                        "closed", now, price, pnl, reason, position_signal_id, pos.get("entry_reason"),
                        position_id, close_fraction, 1,
                    ),
                )
                conn.execute(
                    """UPDATE crypto_positions
                       SET amount_usdt=?, amount_coin=?, partial_exit_done=1
                       WHERE id=?""",
                    (remaining_amount_usdt, remaining_amount_coin, position_id),
                )
            else:
                conn.execute(
                    """UPDATE crypto_positions
                       SET status='closed', exit_date=?, exit_price=?, pnl_usdt=?, exit_reason=?, closed_fraction=1.0
                       WHERE id=?""",
                    (now, price, pnl, reason, position_id),
                )

        return {
            "status":      "FILLED",
            "position_id": position_id,
            "order_id":    order_id,
            "symbol":      symbol,
            "exit_price":  price,
            "pnl_usdt":    pnl,
            "exit_reason": reason,
            "signal_id":   position_signal_id,
            "is_partial":  is_partial,
            "closed_fraction": close_fraction,
            "closed_amount_usdt": close_amount_usdt,
            "remaining_amount_usdt": remaining_amount_usdt,
        }

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        """Return list of dicts for all open crypto positions."""
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM crypto_positions WHERE status='open' ORDER BY entry_date"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_today_closed(self, tz_name: str = "UTC") -> list[dict]:
        """Return positions closed today for the requested timezone."""
        today = datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
        return self.get_closed_for_date(today, tz_name)

    def get_closed_for_date(self, date_str: str, tz_name: str = "UTC") -> list[dict]:
        """Return positions whose exit timestamp falls on date_str in tz_name."""
        target_tz = ZoneInfo(tz_name)
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM crypto_positions
                   WHERE status='closed' AND exit_date IS NOT NULL
                   ORDER BY exit_date DESC""",
            ).fetchall()
        selected: list[dict] = []
        for row in rows:
            exit_dt = _parse_iso_dt(row["exit_date"])
            if exit_dt is None:
                continue
            if exit_dt.astimezone(target_tz).strftime("%Y-%m-%d") == date_str:
                selected.append(dict(row))
        return selected

    def get_closed_positions_since(self, since_iso: str) -> list[dict]:
        """Return closed positions with exit_date >= since_iso."""
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM crypto_positions
                   WHERE status='closed' AND exit_date IS NOT NULL AND exit_date >= ?
                   ORDER BY exit_date DESC""",
                (since_iso,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cumulative_pnl(self) -> float:
        """Total P&L across all closed crypto positions."""
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT SUM(pnl_usdt) as total FROM crypto_positions WHERE status='closed'"
            ).fetchone()
        return row["total"] or 0.0

    # ── P&L calc ──────────────────────────────────────────────────────────────

    def calc_pnl(self, position: dict, current_price: float) -> float:
        """
        Calculate unrealised or realised P&L in USDT.

        For long:  (current - entry) * amount_coin
        For short: (entry - current) * amount_coin
        """
        entry  = position["entry_price"]
        coins  = position["amount_coin"]
        if position["direction"] == "long":
            return round((current_price - entry) * coins, 4)
        else:
            return round((entry - current_price) * coins, 4)
