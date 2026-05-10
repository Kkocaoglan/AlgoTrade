"""
outcome_filler.py — Fill outcome_3d / actual_return_3d in signals_log
----------------------------------------------------------------------
Run daily after market close (18:30).

For every row in signals_log where outcome_3d IS NULL and
signal_date <= today - 3 calendar days:

  1. Find the signal price from ohlcv (close on signal_date).
  2. Find the close price 3 trading days later (first available ohlcv row
     whose date >= signal_date + 3 calendar days).
  3. actual_return_3d = (price_3d - price_signal) / price_signal
  4. outcome_3d = 1 if actual_return_3d >= 0.02 else 0

Usage:
  python3.12 outcome_filler.py           # fill pending rows, print report
  python3.12 outcome_filler.py --dry-run # show what would be updated, no writes
"""

import sqlite3
import sys
from datetime import date, timedelta

DB_PATH        = "trade_data.db"
HORIZON_DAYS   = 3        # calendar days after signal
TARGET_RETURN  = 0.02     # +2% → outcome_3d = 1


def _get_close(cur: sqlite3.Cursor, symbol: str, on_or_after: str) -> tuple:
    """Return (date_str, close) for the first ohlcv row >= on_or_after."""
    cur.execute(
        """SELECT date, close FROM ohlcv
           WHERE symbol=? AND date >= ?
           ORDER BY date ASC LIMIT 1""",
        (symbol, on_or_after),
    )
    row = cur.fetchone()
    return row if row else (None, None)


def _signal_close(cur: sqlite3.Cursor, symbol: str, signal_date: str) -> float | None:
    """Return close price on the signal date (or the nearest prior date)."""
    sig_day = signal_date[:10]
    cur.execute(
        """SELECT close FROM ohlcv
           WHERE symbol=? AND date <= ?
           ORDER BY date DESC LIMIT 1""",
        (symbol, sig_day),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def run(dry_run: bool = False) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    cutoff = (date.today() - timedelta(days=HORIZON_DAYS)).isoformat()

    cur.execute(
        """SELECT id, symbol, signal_date
           FROM signals_log
           WHERE outcome_3d IS NULL
             AND date(signal_date) <= ?""",
        (cutoff,),
    )
    pending = cur.fetchall()

    filled    = 0
    skipped   = 0
    outcomes  = {1: 0, 0: 0}

    for row in pending:
        row_id      = row["id"]
        symbol      = row["symbol"]
        signal_date = row["signal_date"]
        sig_day     = signal_date[:10]

        price_entry = _signal_close(cur, symbol, signal_date)
        if price_entry is None or price_entry == 0:
            skipped += 1
            continue

        future_day = (
            date.fromisoformat(sig_day) + timedelta(days=HORIZON_DAYS)
        ).isoformat()
        future_date_str, price_future = _get_close(cur, symbol, future_day)

        if price_future is None:
            skipped += 1
            continue

        actual_return = (float(price_future) - float(price_entry)) / float(price_entry)
        outcome       = 1 if actual_return >= TARGET_RETURN else 0

        outcomes[outcome] += 1

        if dry_run:
            print(
                f"  DRY  id={row_id:<6} {symbol:<8} {sig_day} "
                f"entry={price_entry:.2f}  future({future_date_str})={price_future:.2f} "
                f"ret={actual_return:+.2%}  outcome={outcome}"
            )
        else:
            cur.execute(
                """UPDATE signals_log
                   SET outcome_3d=?, actual_return_3d=?
                   WHERE id=?""",
                (outcome, round(actual_return, 6), row_id),
            )
            filled += 1

    if not dry_run:
        conn.commit()
    conn.close()

    return {
        "pending":  len(pending),
        "filled":   filled,
        "skipped":  skipped,
        "outcome_1": outcomes[1],
        "outcome_0": outcomes[0],
    }


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv

    print("=" * 55)
    print(f"outcome_filler.py  {'(DRY RUN)' if dry else ''}")
    print(f"Cutoff: signals on or before {(date.today() - timedelta(days=HORIZON_DAYS)).isoformat()}")
    print("=" * 55)

    result = run(dry_run=dry)

    print(f"\nPending rows found : {result['pending']}")
    if dry:
        print("(no writes — dry run)")
    else:
        print(f"Filled             : {result['filled']}")
        print(f"Skipped (no price) : {result['skipped']}")
        print(f"Outcome=1 (hit)    : {result['outcome_1']}")
        print(f"Outcome=0 (miss)   : {result['outcome_0']}")
        if result["filled"] > 0:
            base_rate = result["outcome_1"] / result["filled"] * 100
            print(f"Base rate (+2% hit): {base_rate:.1f}%")
    print("Done.")
