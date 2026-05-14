"""
bist_data_quality.py -- BIST price-series and entry sanity checks.

These helpers are intentionally lightweight and read-only. They protect live
entry paths from corporate-action discontinuities such as bedelsiz/split days
where yesterday's DB close and today's executable price live on different
price bases.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


DB_PATH = Path(__file__).parent / "trade_data.db"

# A one-day move this large is treated as a data/corporate-action event until
# the symbol is refreshed and indicators are recomputed on a consistent basis.
MAX_DAILY_PRICE_JUMP_PCT = 0.35

# Live/entry price vs latest DB close mismatch. BIMAS %100 bedelsiz produced
# roughly 418 vs 813, so 25% catches split-like basis breaks without catching
# ordinary high-volatility days.
MAX_ENTRY_DB_DEVIATION_PCT = 0.25


@dataclass(frozen=True)
class DataQualityResult:
    ok: bool
    code: str
    detail: str
    db_close: float | None = None
    db_date: str | None = None
    deviation_pct: float | None = None


def _recent_closes(symbol: str, limit: int = 3) -> list[tuple[str, float]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """
            SELECT date, close
            FROM ohlcv
            WHERE symbol=?
            ORDER BY date DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
        conn.close()
    except Exception:
        return []

    out: list[tuple[str, float]] = []
    for date_str, close in rows:
        try:
            px = float(close)
        except (TypeError, ValueError):
            continue
        if px > 0:
            out.append((str(date_str), px))
    return out


def get_latest_db_close(symbol: str) -> tuple[str | None, float | None]:
    rows = _recent_closes(symbol, limit=1)
    if not rows:
        return None, None
    return rows[0]


def check_recent_close_jump(symbol: str) -> DataQualityResult:
    rows = _recent_closes(symbol, limit=2)
    if len(rows) < 2:
        return DataQualityResult(True, "OK", "insufficient close history")

    latest_date, latest_close = rows[0]
    prev_date, prev_close = rows[1]
    jump = abs(latest_close - prev_close) / prev_close
    if jump > MAX_DAILY_PRICE_JUMP_PCT:
        return DataQualityResult(
            False,
            "CORP_ACTION_CLOSE_JUMP",
            (
                f"{symbol} close jump {jump:.1%}: "
                f"{prev_date}={prev_close:.2f} -> {latest_date}={latest_close:.2f}"
            ),
            db_close=latest_close,
            db_date=latest_date,
            deviation_pct=jump,
        )
    return DataQualityResult(True, "OK", "recent close continuity OK")


def check_entry_price(symbol: str, entry_price: float) -> DataQualityResult:
    try:
        entry = float(entry_price)
    except (TypeError, ValueError):
        return DataQualityResult(False, "BAD_ENTRY_PRICE", f"{symbol} entry price invalid")

    if entry <= 0:
        return DataQualityResult(False, "BAD_ENTRY_PRICE", f"{symbol} entry price <= 0")

    db_date, db_close = get_latest_db_close(symbol)
    if db_close is None or db_close <= 0:
        return DataQualityResult(True, "OK", "no DB close available")

    deviation = abs(entry - db_close) / db_close
    if deviation > MAX_ENTRY_DB_DEVIATION_PCT:
        return DataQualityResult(
            False,
            "CORP_ACTION_PRICE_BASIS",
            (
                f"{symbol} entry/db basis break {deviation:.1%}: "
                f"entry={entry:.2f} vs db_close({db_date})={db_close:.2f}. "
                "Likely corporate action or stale adjusted/unadjusted data."
            ),
            db_close=db_close,
            db_date=db_date,
            deviation_pct=deviation,
        )

    jump_result = check_recent_close_jump(symbol)
    if not jump_result.ok:
        return jump_result

    return DataQualityResult(True, "OK", "entry price continuity OK")


def format_data_quality_block(result: DataQualityResult) -> str:
    return f"{result.code}: {result.detail}"


def _all_symbols() -> list[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
        conn.close()
    except Exception:
        return []
    return [str(row[0]) for row in rows if row and row[0]]


def audit_recent_jumps(symbols: list[str] | None = None) -> list[tuple[str, DataQualityResult]]:
    syms = symbols or _all_symbols()
    rows = []
    for sym in syms:
        result = check_recent_close_jump(sym)
        if not result.ok:
            rows.append((sym, result))
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit BIST price continuity / corporate-action breaks.")
    parser.add_argument("symbols", nargs="*", help="Optional symbols. Defaults to all DB symbols.")
    args = parser.parse_args()

    rows = audit_recent_jumps(args.symbols or None)
    print("=" * 72)
    print("BIST DATA QUALITY / CORPORATE ACTION AUDIT")
    print("=" * 72)
    print(f"DB       : {DB_PATH}")
    print(f"Threshold: daily close jump > {MAX_DAILY_PRICE_JUMP_PCT:.0%}")
    print()
    if not rows:
        print("No recent close jumps above threshold.")
        return 0

    for sym, result in rows:
        print(f"{sym:<8} {result.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
