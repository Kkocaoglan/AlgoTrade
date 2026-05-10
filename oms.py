"""
oms.py -- Order Management System (emir yasam dongusu)
--------------------------------------------------------
Tum emirleri kayit altina alir: NEW -> SENT -> FILLED/PARTIAL/REJECTED/CANCELLED

BUY emirleri: Kill Switch kontrolunden gecer.
SELL emirleri: Kill Switch atlanir (cikis emirleri her zaman gecerlidir).

Kullanim:
  from oms import oms, OrderStatus

Test:
  python3.12 oms.py --test
"""

import sys
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).parent
DB_PATH  = str(BASE_DIR / "trade_data.db")


class OrderStatus(Enum):
    NEW       = "NEW"        # olusturuldu
    SENT      = "SENT"       # gonderildi
    FILLED    = "FILLED"     # gerceklesti
    PARTIAL   = "PARTIAL"    # kismen gerceklesti
    REJECTED  = "REJECTED"   # reddedildi
    CANCELLED = "CANCELLED"  # iptal edildi


class Order:
    def __init__(self, symbol, direction, price, size_tl, reason=""):
        self.order_id        = f"{symbol}_{datetime.now():%Y%m%d%H%M%S%f}"
        self.symbol          = symbol
        self.direction       = direction   # BUY / SELL
        self.price           = price
        self.size_tl         = size_tl
        self.shares          = round(size_tl / price, 4) if price > 0 else 0
        self.reason          = reason
        self.status          = OrderStatus.NEW
        self.filled_price    = None
        self.filled_shares   = None
        self.filled_at       = None
        self.rejected_reason = None
        self.created_at      = datetime.now()


class OMS:

    DB_PATH = DB_PATH

    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_log (
                order_id         TEXT PRIMARY KEY,
                symbol           TEXT,
                direction        TEXT,
                requested_price  REAL,
                requested_shares REAL,
                size_tl          REAL,
                filled_price     REAL,
                filled_shares    REAL,
                status           TEXT,
                reason           TEXT,
                rejected_reason  TEXT,
                created_at       TEXT,
                filled_at        TEXT
            )
        """)
        conn.commit()
        conn.close()

    def create_order(self, symbol, direction, price, size_tl, reason=""):
        """
        BUY / SHORT : Kill Switch kontrolu yap; gerekirse REJECTED dondur.
        SELL / COVER: Kill Switch atlanir -- cikis emirleri her zaman gecerlidir.
        """
        if direction in ("BUY", "SHORT"):
            from kill_switch import KillSwitch
            ok, block_reason = KillSwitch.check_all(symbol, price, size_tl,
                                                     direction=direction)
            if not ok:
                print(f"[OMS] BLOCKED {direction} {symbol}: {block_reason}")
                order = Order(symbol, direction, price, size_tl, reason)
                order.status = OrderStatus.REJECTED
                order.rejected_reason = block_reason
                self._save(order)
                return order

        order = Order(symbol, direction, price, size_tl, reason)
        order.status = OrderStatus.SENT
        self._save(order)
        print(f"[OMS] {direction} {symbol} @ {price:.2f} | "
              f"{size_tl:.0f} TL | {order.shares:.4f} adet | "
              f"ID:{order.order_id}")
        return order

    def fill_order(self, order, filled_price, filled_shares=None):
        """Paper trade: aninda tam dolum."""
        order.filled_price  = filled_price
        order.filled_shares = filled_shares or order.shares
        order.filled_at     = datetime.now()

        fill_ratio = order.filled_shares / order.shares if order.shares > 0 else 1
        if fill_ratio >= 0.95:
            order.status = OrderStatus.FILLED
        elif fill_ratio > 0:
            order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.REJECTED
            order.rejected_reason = "Zero fill"

        self._save(order)
        print(f"[OMS] FILLED {order.direction} {order.symbol} "
              f"@ {filled_price:.2f} | {order.filled_shares:.4f} adet "
              f"| status:{order.status.value}")

        # BUY / SHORT doldurulmasi: kill switch emir hizi loguna yaz
        if order.direction in ("BUY", "SHORT") and order.status == OrderStatus.FILLED:
            from kill_switch import KillSwitch
            KillSwitch.log_order_sent(order.symbol, filled_price, order.size_tl,
                                      direction=order.direction)

        return order

    def reject_order(self, order, reason):
        order.status = OrderStatus.REJECTED
        order.rejected_reason = reason
        self._save(order)
        print(f"[OMS] REJECTED {order.direction} {order.symbol}: {reason}")
        return order

    def cancel_order(self, order, reason=""):
        order.status = OrderStatus.CANCELLED
        order.rejected_reason = reason
        self._save(order)
        print(f"[OMS] CANCELLED {order.direction} {order.symbol}: {reason}")
        return order

    def _save(self, order):
        conn = sqlite3.connect(self.DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO order_log VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            order.order_id,
            order.symbol,
            order.direction,
            order.price,
            order.shares,
            order.size_tl,
            order.filled_price,
            order.filled_shares,
            order.status.value,
            order.reason,
            order.rejected_reason,
            order.created_at.isoformat(),
            order.filled_at.isoformat() if order.filled_at else None,
        ))
        conn.commit()
        conn.close()

    def get_todays_orders(self):
        today = datetime.now().strftime("%Y-%m-%d")
        conn  = sqlite3.connect(self.DB_PATH)
        import pandas as pd
        df = pd.read_sql(f"""
            SELECT * FROM order_log
            WHERE created_at LIKE '{today}%'
            ORDER BY created_at DESC
        """, conn)
        conn.close()
        return df

    def get_fill_rate_today(self):
        df = self.get_todays_orders()
        if df.empty:
            return None
        filled = len(df[df["status"] == "FILLED"])
        total  = len(df[df["status"].isin(["FILLED", "PARTIAL", "REJECTED"])])
        return filled / total if total > 0 else None

    def summary(self):
        df = self.get_todays_orders()
        if df.empty:
            print("[OMS] Bugun emir yok")
            return
        n_filled   = len(df[df["status"] == "FILLED"])
        n_rejected = len(df[df["status"] == "REJECTED"])
        fill_rate  = self.get_fill_rate_today()
        fr_str     = f"{fill_rate:.0%}" if fill_rate is not None else "N/A"
        print(f"[OMS] Bugun: {len(df)} emir | "
              f"Filled:{n_filled} | "
              f"Rejected:{n_rejected} | "
              f"Fill rate:{fr_str}")


# =============================================================================
# Global instance -- loop_trader.py tarafindan import edilir
# =============================================================================
oms = OMS()


# =============================================================================
# --test MODE
# =============================================================================

if __name__ == "__main__":
    if "--test" not in sys.argv:
        print("Kullanim: python3.12 oms.py --test")
        sys.exit(0)

    import sqlite3 as _sq

    print("=" * 50)
    print("OMS Test Suite")
    print("=" * 50)

    _passed = 0
    _failed = 0

    def _chk(name, cond):
        global _passed, _failed
        if cond:
            print(f"  [PASS] {name}")
            _passed += 1
        else:
            print(f"  [FAIL] {name}")
            _failed += 1

    # -- Test 1: BUY order create -> SENT or REJECTED (depends on kill switch env)
    print("\n[1] BUY emri olustur:")
    o1 = oms.create_order("GARAN", "BUY", 100.0, 5_000.0, reason="test_buy")
    _chk("BUY order SENT or REJECTED", o1.status in (OrderStatus.SENT, OrderStatus.REJECTED))
    _chk("order_id set", bool(o1.order_id))
    _chk("symbol correct", o1.symbol == "GARAN")

    # -- Test 2: SELL order always SENT (no kill switch block)
    print("\n[2] SELL emri olustur (her zaman SENT):")
    o2 = oms.create_order("GARAN", "SELL", 102.0, 5_000.0, reason="test_sell")
    _chk("SELL order always SENT", o2.status == OrderStatus.SENT)

    # -- Test 3: fill_order -> FILLED
    print("\n[3] fill_order -> FILLED:")
    oms.fill_order(o2, 102.0)
    _chk("SELL fill -> FILLED", o2.status == OrderStatus.FILLED)
    _chk("filled_price correct", o2.filled_price == 102.0)
    _chk("filled_at set", o2.filled_at is not None)

    # -- Test 4: DB persistence
    print("\n[4] DB kaydi kontrol:")
    conn = _sq.connect(DB_PATH)
    row = conn.execute(
        "SELECT status FROM order_log WHERE order_id = ?", (o2.order_id,)
    ).fetchone()
    conn.close()
    _chk("DB'de FILLED kaydi var", row is not None and row[0] == "FILLED")

    # -- Test 5: summary()
    print("\n[5] summary() calistir:")
    oms.summary()
    _chk("summary hata vermedi", True)

    print(f"\n{'='*50}")
    print(f"Sonuc: {_passed} PASS / {_failed} FAIL")
    print(f"{'='*50}\n")

    # Cleanup test rows
    conn = _sq.connect(DB_PATH)
    conn.execute("DELETE FROM order_log WHERE reason LIKE 'test_%'")
    conn.commit()
    conn.close()
    print("Test emirleri DB'den silindi.")

    sys.exit(0 if _failed == 0 else 1)
