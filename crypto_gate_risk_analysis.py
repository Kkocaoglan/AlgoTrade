"""
crypto_gate_risk_analysis.py -- Crypto gate ablation and risk-pct analysis.

Read-only analysis tool. Uses crypto_signal_journal to show which gates block
signals and how closed-trade P&L would scale under alternate CRYPTO_RISK_PCT
values. It does not mutate DB or result files.

Usage:
  py -3.12 crypto_gate_risk_analysis.py
  py -3.12 crypto_gate_risk_analysis.py --risk-pcts 0.01,0.02,0.03
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from crypto_config import CRYPTO_CAPITAL_USDT, CRYPTO_RISK_PCT

DB_PATH = Path(__file__).parent / "trade_data.db"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _load_rows(conn: sqlite3.Connection) -> list[dict]:
    if not _table_exists(conn, "crypto_signal_journal"):
        return []
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT signal_ts, closed_at, symbol, side, tier, mtf_score, ml_probability,
               composite_score, score_components, threshold, regime, risk_decision,
               risk_reason, exit_reason, position_size_usdt, gross_pnl,
               net_pnl_estimate
        FROM crypto_signal_journal
        ORDER BY signal_ts
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _max_drawdown(pnls: list[float], capital: float) -> float:
    equity = capital
    peak = capital
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _risk_reason_key(reason: str | None) -> str:
    if not reason:
        return "unknown"
    return str(reason).split(":", 1)[0]


def _score_component_keys(row: dict) -> list[str]:
    raw = row.get("score_components")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    keys = []
    for key, value in payload.items():
        try:
            if float(value) != 0.0:
                keys.append(key)
        except Exception:
            if value:
                keys.append(key)
    return keys


def _closed_trade_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("closed_at") or row.get("exit_reason"):
            if row.get("net_pnl_estimate") is not None:
                out.append(row)
    return out


def _print_gate_ablation(rows: list[dict]) -> None:
    total = len(rows)
    accepted = [r for r in rows if r.get("risk_decision") == "accepted"]
    rejected = [r for r in rows if r.get("risk_decision") != "accepted"]
    reason_counts = Counter(_risk_reason_key(r.get("risk_reason")) for r in rejected)

    print("=" * 88)
    print("CRYPTO GATE ABLATION / BLOCKER REPORT")
    print("=" * 88)
    print(f"Signals        : {total}")
    print(f"Accepted       : {len(accepted)} ({len(accepted)/total:.1%})" if total else "Accepted       : 0")
    print(f"Rejected       : {len(rejected)} ({len(rejected)/total:.1%})" if total else "Rejected       : 0")
    print()

    print("Top blocking gates:")
    if reason_counts:
        for reason, count in reason_counts.most_common(15):
            print(f"  {reason:<36} {count:>5}  ({count/max(len(rejected), 1):>6.1%} of rejects)")
    else:
        print("  no rejected signals")
    print()

    by_tier = defaultdict(list)
    by_side = defaultdict(list)
    for row in rows:
        by_tier[row.get("tier") or "UNKNOWN"].append(row)
        by_side[row.get("side") or "NONE"].append(row)

    print("Acceptance by tier:")
    for tier, tier_rows in sorted(by_tier.items()):
        acc = sum(1 for r in tier_rows if r.get("risk_decision") == "accepted")
        print(f"  {tier:<10} {acc:>4}/{len(tier_rows):<4} ({acc/len(tier_rows):>6.1%})")
    print()

    print("Acceptance by side:")
    for side, side_rows in sorted(by_side.items()):
        acc = sum(1 for r in side_rows if r.get("risk_decision") == "accepted")
        print(f"  {side:<10} {acc:>4}/{len(side_rows):<4} ({acc/len(side_rows):>6.1%})")
    print()

    component_counts = Counter()
    for row in accepted:
        component_counts.update(_score_component_keys(row))
    print("Non-zero score components on accepted signals:")
    if component_counts:
        for key, count in component_counts.most_common(12):
            print(f"  {key:<36} {count:>5}")
    else:
        print("  no component payloads found")


def _print_risk_pct_analysis(rows: list[dict], risk_pcts: list[float]) -> None:
    closed = _closed_trade_rows(rows)
    base_pnls = [float(r.get("net_pnl_estimate") or 0.0) for r in closed]

    print()
    print("=" * 88)
    print("CRYPTO RISK_PCT SCALING ANALYSIS")
    print("=" * 88)
    print(f"Closed trades with net P&L : {len(base_pnls)}")
    print(f"Current CRYPTO_RISK_PCT    : {CRYPTO_RISK_PCT:.2%}")
    print("Note: scales realized P&L linearly; fees/slippage/liquidity are not re-simulated.")
    print()
    if not base_pnls:
        print("No closed crypto_signal_journal rows with net_pnl_estimate yet.")
        return

    print(f"{'risk%':>7} {'net_pnl':>12} {'avg_pnl':>10} {'win%':>8} {'max_dd%':>9} {'end_equity':>12}")
    print("-" * 68)
    for pct in risk_pcts:
        scale = pct / CRYPTO_RISK_PCT if CRYPTO_RISK_PCT else 0.0
        pnls = [pnl * scale for pnl in base_pnls]
        net = sum(pnls)
        wins = sum(1 for pnl in pnls if pnl > 0)
        max_dd = _max_drawdown(pnls, CRYPTO_CAPITAL_USDT)
        print(
            f"{pct*100:>6.1f}% {net:>12.2f} {net/len(pnls):>10.2f} "
            f"{wins/len(pnls)*100:>7.1f}% {max_dd*100:>8.1f}% "
            f"{CRYPTO_CAPITAL_USDT + net:>12.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze crypto gates and risk sizing from signal journal.")
    parser.add_argument("--risk-pcts", default="0.01,0.02,0.03,0.04", help="Comma-separated risk pct values")
    args = parser.parse_args()
    risk_pcts = [float(x.strip()) for x in args.risk_pcts.split(",") if x.strip()]

    conn = sqlite3.connect(DB_PATH)
    rows = _load_rows(conn)
    conn.close()

    if not rows:
        print("No crypto_signal_journal rows found. Run crypto_trader.py long enough to populate the journal.")
        return 0

    _print_gate_ablation(rows)
    _print_risk_pct_analysis(rows, risk_pcts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
