"""
crypto_journal.py — Signal journal and daily summary for Crypto Module.

Usage:
  python3.12 crypto_journal.py
  python3.12 crypto_journal.py --date 2026-05-01
"""

import argparse
import csv
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "trade_data.db"
RESULTS_DIR = Path(__file__).parent / "results"
SIGNAL_EXPORT_TEMPLATE = "crypto_signal_journal_{date}.csv"
SUMMARY_EXPORT_PATH = RESULTS_DIR / "crypto_daily_journal_summary.csv"
MAJORAFTER_STATE_PATH = RESULTS_DIR / "majorafter_state.json"
CONFIDENCE_BUCKETS = [
    (0.00, 0.55, "0.00-0.55"),
    (0.55, 0.65, "0.55-0.65"),
    (0.65, 0.75, "0.65-0.75"),
    (0.75, 0.85, "0.75-0.85"),
    (0.85, 1.01, "0.85-1.00"),
]

DEFAULT_COST_CONFIG = {
    "enabled": True,
    "order_type": "market",
    "order_modes": {
        "market": {"fee_bps": 10.0, "spread_bps": 2.0, "slippage_bps": 5.0},
        "limit": {"fee_bps": 10.0, "spread_bps": 0.5, "slippage_bps": 1.5},
    },
}

try:
    from telegram_bot import send_telegram_alert
except Exception:
    def send_telegram_alert(msg, **kw):  # type: ignore[no-redef]
        pass

from logger import algo_log


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def ensure_journal_schema() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    try:
        from crypto_oms import CryptoOMS  # local import to bootstrap base tables
        CryptoOMS()
    except Exception:
        pass
    ddl = """
    CREATE TABLE IF NOT EXISTS crypto_signal_journal (
        signal_id TEXT PRIMARY KEY,
        signal_ts TEXT NOT NULL,
        decision_ts TEXT,
        closed_at TEXT,
        symbol TEXT NOT NULL,
        side TEXT,
        tier TEXT,
        mtf_score REAL,
        ml_probability REAL,
        composite_score REAL,
        score_components TEXT,
        threshold REAL,
        regime TEXT,
        risk_decision TEXT,
        risk_reason TEXT,
        entry_reason TEXT,
        exit_reason TEXT,
        entry_price REAL,
        exit_price REAL,
        position_size_usdt REAL,
        position_id INTEGER,
        entry_order_id TEXT,
        exit_order_id TEXT,
        order_status TEXT,
        gross_pnl REAL,
        net_pnl_estimate REAL,
        slippage_estimate REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS crypto_daily_journal (
        date TEXT PRIMARY KEY,
        signal_count INTEGER NOT NULL,
        reject_count INTEGER NOT NULL,
        trade_count INTEGER NOT NULL,
        closed_trade_count INTEGER NOT NULL,
        win_rate REAL NOT NULL,
        avg_win REAL NOT NULL,
        avg_loss REAL NOT NULL,
        max_drawdown REAL NOT NULL,
        confidence_bucket_json TEXT NOT NULL,
        generated_at TEXT NOT NULL
    );
    """
    with _get_conn() as conn:
        conn.executescript(ddl)
        _ensure_column(conn, "crypto_positions", "signal_id", "signal_id TEXT")
        _ensure_column(conn, "crypto_positions", "entry_reason", "entry_reason TEXT")
        _ensure_column(conn, "crypto_orders", "signal_id", "signal_id TEXT")
        _ensure_column(conn, "crypto_orders", "position_id", "position_id INTEGER")
        _ensure_column(conn, "crypto_orders", "order_role", "order_role TEXT")
        _ensure_column(conn, "crypto_signal_journal", "composite_score", "composite_score REAL")
        _ensure_column(conn, "crypto_signal_journal", "score_components", "score_components TEXT")


def make_signal_id(symbol: str, side: str | None = None, signal_ts: str | None = None) -> str:
    ts = signal_ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sym = symbol.replace("/", "").lower()
    side_token = (side or "none").lower()
    return f"sig_{ts}_{sym}_{side_token}_{uuid.uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_signal(payload: dict) -> None:
    ensure_journal_schema()
    now = _utc_now_iso()
    existing = {}
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM crypto_signal_journal WHERE signal_id=?",
            (payload["signal_id"],),
        ).fetchone()
        if row is not None:
            existing = dict(row)
    row = {
        "signal_id": payload["signal_id"],
        "signal_ts": payload.get("signal_ts", existing.get("signal_ts")) or now,
        "decision_ts": payload.get("decision_ts", existing.get("decision_ts")),
        "closed_at": payload.get("closed_at", existing.get("closed_at")),
        "symbol": payload.get("symbol", existing.get("symbol")),
        "side": payload.get("side", existing.get("side")),
        "tier": payload.get("tier", existing.get("tier")),
        "mtf_score": payload.get("mtf_score", existing.get("mtf_score")),
        "ml_probability": payload.get("ml_probability", existing.get("ml_probability")),
        "composite_score": payload.get("composite_score", existing.get("composite_score")),
        "score_components": payload.get("score_components", existing.get("score_components")),
        "threshold": payload.get("threshold", existing.get("threshold")),
        "regime": payload.get("regime", existing.get("regime")),
        "risk_decision": payload.get("risk_decision", existing.get("risk_decision")),
        "risk_reason": payload.get("risk_reason", existing.get("risk_reason")),
        "entry_reason": payload.get("entry_reason", existing.get("entry_reason")),
        "exit_reason": payload.get("exit_reason", existing.get("exit_reason")),
        "entry_price": payload.get("entry_price", existing.get("entry_price")),
        "exit_price": payload.get("exit_price", existing.get("exit_price")),
        "position_size_usdt": payload.get("position_size_usdt", existing.get("position_size_usdt")),
        "position_id": payload.get("position_id", existing.get("position_id")),
        "entry_order_id": payload.get("entry_order_id", existing.get("entry_order_id")),
        "exit_order_id": payload.get("exit_order_id", existing.get("exit_order_id")),
        "order_status": payload.get("order_status", existing.get("order_status")),
        "gross_pnl": payload.get("gross_pnl", existing.get("gross_pnl")),
        "net_pnl_estimate": payload.get("net_pnl_estimate", existing.get("net_pnl_estimate")),
        "slippage_estimate": payload.get("slippage_estimate", existing.get("slippage_estimate")),
        "created_at": payload.get("created_at", existing.get("created_at")) or now,
        "updated_at": now,
    }
    if not row["symbol"]:
        raise ValueError(f"signal journal row requires symbol for signal_id={payload['signal_id']}")
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    assignments = ", ".join(
        f"{col}=excluded.{col}" for col in cols
        if col not in {"signal_id", "created_at"}
    )
    sql = f"""
        INSERT INTO crypto_signal_journal ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(signal_id) DO UPDATE SET {assignments}
    """
    with _get_conn() as conn:
        conn.execute(sql, [row[col] for col in cols])


def record_signal(
    signal_id: str,
    signal_ts: str,
    symbol: str,
    side: str | None,
    tier: str | None,
    mtf_score: float | None,
    ml_probability: float | None,
    composite_score: float | None = None,
    score_components: dict | None = None,
    threshold: float | None = None,
    regime: str | None = None,
    risk_decision: str = "",
    risk_reason: str = "",
    entry_reason: str | None = None,
) -> None:
    _upsert_signal({
        "signal_id": signal_id,
        "signal_ts": signal_ts,
        "decision_ts": _utc_now_iso(),
        "symbol": symbol,
        "side": side,
        "tier": tier,
        "mtf_score": mtf_score,
        "ml_probability": ml_probability,
        "composite_score": composite_score,
        "score_components": json.dumps(score_components, ensure_ascii=True, sort_keys=True) if score_components else None,
        "threshold": threshold,
        "regime": regime,
        "risk_decision": risk_decision,
        "risk_reason": risk_reason,
        "entry_reason": entry_reason,
        "order_status": "FILLED" if risk_decision == "accepted" else "REJECTED",
    })


def mark_signal_entry(
    signal_id: str,
    position_id: int,
    entry_order_id: str,
    entry_price: float,
    position_size_usdt: float,
    entry_reason: str,
) -> None:
    _upsert_signal({
        "signal_id": signal_id,
        "position_id": position_id,
        "entry_order_id": entry_order_id,
        "entry_price": entry_price,
        "position_size_usdt": position_size_usdt,
        "entry_reason": entry_reason,
        "risk_decision": "accepted",
        "order_status": "FILLED",
    })


def mark_signal_exit(
    signal_id: str,
    exit_reason: str,
    exit_order_id: str,
    exit_price: float,
    gross_pnl: float,
    net_pnl_estimate: float,
    slippage_estimate: float,
) -> None:
    _upsert_signal({
        "signal_id": signal_id,
        "closed_at": _utc_now_iso(),
        "exit_reason": exit_reason,
        "exit_order_id": exit_order_id,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "net_pnl_estimate": net_pnl_estimate,
        "slippage_estimate": slippage_estimate,
    })


def _load_cost_snapshot() -> dict:
    try:
        from crypto_ml import _per_side_cost_bps  # type: ignore[attr-defined]
        return dict(_per_side_cost_bps(None, DEFAULT_COST_CONFIG))
    except Exception:
        order_type = DEFAULT_COST_CONFIG["order_type"]
        mode = DEFAULT_COST_CONFIG["order_modes"][order_type]
        return {
            "enabled": bool(DEFAULT_COST_CONFIG["enabled"]),
            "order_type": order_type,
            "fee_bps": float(mode["fee_bps"]),
            "spread_bps": float(mode["spread_bps"]),
            "slippage_bps": float(mode["slippage_bps"]),
            "per_side_total_bps": float(mode["fee_bps"] + mode["spread_bps"] + mode["slippage_bps"]),
        }


def estimate_trade_costs(amount_usdt: float) -> dict[str, float]:
    cost = _load_cost_snapshot()
    total_cost_rate = (float(cost["per_side_total_bps"]) * 2.0) / 10000.0
    slippage_rate = (float(cost["slippage_bps"]) * 2.0) / 10000.0
    return {
        "total_cost_rate": total_cost_rate,
        "total_cost_usdt": float(amount_usdt) * total_cost_rate,
        "slippage_rate": slippage_rate,
        "slippage_usdt": float(amount_usdt) * slippage_rate,
    }


def _confidence_bucket(prob: float | None) -> str:
    if prob is None:
        return "unknown"
    p = float(prob)
    for lower, upper, label in CONFIDENCE_BUCKETS:
        if lower <= p < upper:
            return label
    return "unknown"


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for value in values:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _fetch_signal_rows(date_str: str) -> list[dict]:
    ensure_journal_schema()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM crypto_signal_journal
               WHERE signal_ts LIKE ?
               ORDER BY signal_ts, symbol""",
            (f"{date_str}%",),
        ).fetchall()
    return [dict(r) for r in rows]


def _load_majorafter_snapshot(date_str: str) -> dict:
    if not MAJORAFTER_STATE_PATH.exists():
        return {}
    try:
        state = json.loads(MAJORAFTER_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if state.get("date") != date_str:
        return {}
    return state


def _exit_reason_summary(closed_rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for row in closed_rows:
        reason = str(row.get("exit_reason") or "unknown")
        info = summary.setdefault(reason, {
            "count": 0,
            "wins": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
        })
        net_val = float(row.get("net_pnl_estimate") or 0.0)
        info["count"] += 1
        info["wins"] += int(net_val > 0)
        info["gross_pnl"] += float(row.get("gross_pnl") or 0.0)
        info["net_pnl"] += net_val
    for info in summary.values():
        count = int(info["count"])
        info["win_rate"] = (int(info["wins"]) / count) if count else 0.0
        info["avg_net_pnl"] = (float(info["net_pnl"]) / count) if count else 0.0
    return summary


def _coin_pnl_summary(closed_rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for row in closed_rows:
        symbol = str(row.get("symbol") or "UNKNOWN")
        info = summary.setdefault(symbol, {
            "count": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
        })
        info["count"] += 1
        info["gross_pnl"] += float(row.get("gross_pnl") or 0.0)
        info["net_pnl"] += float(row.get("net_pnl_estimate") or 0.0)
    return dict(sorted(summary.items(), key=lambda item: item[1]["net_pnl"], reverse=True))


def _majorafter_analytics(date_str: str) -> dict:
    state = _load_majorafter_snapshot(date_str)
    if not state:
        return {}
    shadow_closed = list(state.get("shadow_closed_positions", []))
    shadow_open = list(state.get("shadow_open_positions", []))
    pnls = [float(row.get("pnl_usdt") or 0.0) for row in shadow_closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    coin_summary: dict[str, dict] = {}
    exit_reason_summary: dict[str, dict] = {}
    for row in shadow_closed:
        coin = str(row.get("symbol") or "UNKNOWN")
        coin_info = coin_summary.setdefault(coin, {"count": 0, "pnl_usdt": 0.0})
        coin_info["count"] += 1
        coin_info["pnl_usdt"] += float(row.get("pnl_usdt") or 0.0)
        reason = str(row.get("exit_reason") or "unknown")
        reason_info = exit_reason_summary.setdefault(reason, {"count": 0, "pnl_usdt": 0.0})
        reason_info["count"] += 1
        reason_info["pnl_usdt"] += float(row.get("pnl_usdt") or 0.0)
    locked_equity = float(state.get("locked_equity") or 0.0)
    shadow_total = sum(pnls)
    return {
        "locked": bool(state.get("locked")),
        "start_equity": float(state.get("start_equity") or 0.0),
        "target_equity": float(state.get("target_equity") or 0.0),
        "locked_equity": locked_equity,
        "shadow_closed_count": len(shadow_closed),
        "shadow_open_count": len(shadow_open),
        "shadow_win_rate": (len(wins) / len(shadow_closed)) if shadow_closed else 0.0,
        "shadow_avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "shadow_avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "shadow_total_pnl": shadow_total,
        "shadow_total_pct": (shadow_total / locked_equity) if locked_equity else 0.0,
        "coin_pnl": dict(sorted(coin_summary.items(), key=lambda item: item[1]["pnl_usdt"], reverse=True)),
        "exit_reason_pnl": exit_reason_summary,
    }


def generate_daily_summary(date_str: str) -> dict:
    rows = _fetch_signal_rows(date_str)
    signal_count = len(rows)
    reject_count = sum(1 for row in rows if row.get("risk_decision") != "accepted")
    trade_rows = [row for row in rows if row.get("position_id") is not None]
    trade_count = len(trade_rows)
    closed_rows = [row for row in trade_rows if row.get("exit_reason")]
    net_pnls = [float(row.get("net_pnl_estimate") or 0.0) for row in closed_rows]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p < 0]
    win_rate = (len(wins) / len(closed_rows)) if closed_rows else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    running = 0.0
    equity_curve: list[float] = []
    for row in sorted(closed_rows, key=lambda item: item.get("closed_at") or item.get("updated_at") or ""):
        running += float(row.get("net_pnl_estimate") or 0.0)
        equity_curve.append(running)

    bucket_summary: dict[str, dict] = {}
    for row in trade_rows:
        bucket = _confidence_bucket(row.get("ml_probability"))
        info = bucket_summary.setdefault(bucket, {
            "signals": 0,
            "closed_trades": 0,
            "wins": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
        })
        info["signals"] += 1
        if row.get("exit_reason"):
            net_val = float(row.get("net_pnl_estimate") or 0.0)
            info["closed_trades"] += 1
            info["wins"] += int(net_val > 0)
            info["gross_pnl"] += float(row.get("gross_pnl") or 0.0)
            info["net_pnl"] += net_val
    for bucket, info in bucket_summary.items():
        closed_count = int(info["closed_trades"])
        info["win_rate"] = (int(info["wins"]) / closed_count) if closed_count else 0.0
        info["avg_net_pnl"] = (float(info["net_pnl"]) / closed_count) if closed_count else 0.0
    exit_reason_performance = _exit_reason_summary(closed_rows)
    coin_performance = _coin_pnl_summary(closed_rows)
    majorafter = _majorafter_analytics(date_str)

    return {
        "date": date_str,
        "signal_count": signal_count,
        "reject_count": reject_count,
        "trade_count": trade_count,
        "closed_trade_count": len(closed_rows),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": _max_drawdown(equity_curve),
        "confidence_bucket_performance": bucket_summary,
        "exit_reason_performance": exit_reason_performance,
        "coin_performance": coin_performance,
        "majorafter": majorafter,
        "rows": rows,
    }


def _write_signal_csv(date_str: str, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / SIGNAL_EXPORT_TEMPLATE.format(date=date_str)
    fieldnames = [
        "signal_id", "signal_ts", "symbol", "side", "tier", "mtf_score",
        "ml_probability", "composite_score", "score_components", "threshold",
        "regime", "risk_decision", "risk_reason",
        "entry_reason", "exit_reason", "entry_price", "exit_price",
        "gross_pnl", "net_pnl_estimate", "slippage_estimate",
        "position_size_usdt", "position_id", "entry_order_id", "exit_order_id",
        "order_status", "closed_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _upsert_daily_summary(summary: dict) -> None:
    ensure_journal_schema()
    bucket_json = json.dumps(summary["confidence_bucket_performance"], ensure_ascii=True, sort_keys=True)
    now = _utc_now_iso()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO crypto_daily_journal
            (date, signal_count, reject_count, trade_count, closed_trade_count,
             win_rate, avg_win, avg_loss, max_drawdown, confidence_bucket_json, generated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(date) DO UPDATE SET
                signal_count=excluded.signal_count,
                reject_count=excluded.reject_count,
                trade_count=excluded.trade_count,
                closed_trade_count=excluded.closed_trade_count,
                win_rate=excluded.win_rate,
                avg_win=excluded.avg_win,
                avg_loss=excluded.avg_loss,
                max_drawdown=excluded.max_drawdown,
                confidence_bucket_json=excluded.confidence_bucket_json,
                generated_at=excluded.generated_at
            """,
            (
                summary["date"],
                summary["signal_count"],
                summary["reject_count"],
                summary["trade_count"],
                summary["closed_trade_count"],
                summary["win_rate"],
                summary["avg_win"],
                summary["avg_loss"],
                summary["max_drawdown"],
                bucket_json,
                now,
            ),
        )


def _export_daily_summary_csv(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    row = {
        "date": summary["date"],
        "signal_count": summary["signal_count"],
        "reject_count": summary["reject_count"],
        "trade_count": summary["trade_count"],
        "closed_trade_count": summary["closed_trade_count"],
        "win_rate": summary["win_rate"],
        "avg_win": summary["avg_win"],
        "avg_loss": summary["avg_loss"],
        "max_drawdown": summary["max_drawdown"],
        "confidence_bucket_json": json.dumps({
            "confidence_buckets": summary["confidence_bucket_performance"],
            "exit_reason_performance": summary["exit_reason_performance"],
            "coin_performance": summary["coin_performance"],
            "majorafter": summary["majorafter"],
        }, ensure_ascii=True, sort_keys=True),
    }
    fieldnames = list(row.keys())
    existing: list[dict] = []
    if SUMMARY_EXPORT_PATH.exists():
        with SUMMARY_EXPORT_PATH.open("r", newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    updated = False
    for idx, existing_row in enumerate(existing):
        if existing_row.get("date") == summary["date"]:
            existing[idx] = {key: str(value) for key, value in row.items()}
            updated = True
            break
    if not updated:
        existing.append({key: str(value) for key, value in row.items()})
    existing.sort(key=lambda item: item.get("date", ""))
    with SUMMARY_EXPORT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def run_journal(date_str: str | None = None) -> None:
    ensure_journal_schema()
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = generate_daily_summary(date_str)
    signal_csv_path = _write_signal_csv(date_str, summary["rows"])
    _upsert_daily_summary(summary)
    _export_daily_summary_csv(summary)

    print()
    print(f"CRYPTO JOURNAL ({date_str})")
    print("-" * 60)
    print(f"Signals: {summary['signal_count']}")
    print(f"Rejected: {summary['reject_count']}")
    print(f"Trades opened: {summary['trade_count']}")
    print(f"Closed trades: {summary['closed_trade_count']}")
    print(f"Win rate: {summary['win_rate']:.1%}")
    print(f"Avg win: {summary['avg_win']:+.2f} USDT")
    print(f"Avg loss: {summary['avg_loss']:+.2f} USDT")
    print(f"Max drawdown: {summary['max_drawdown']:.2f} USDT")
    print(f"Signal CSV: {signal_csv_path}")
    print(f"Summary CSV: {SUMMARY_EXPORT_PATH}")
    if summary["confidence_bucket_performance"]:
        print("Confidence buckets:")
        for bucket, info in sorted(summary["confidence_bucket_performance"].items()):
            print(
                f"  {bucket}: signals={info['signals']} closed={info['closed_trades']} "
                f"win_rate={info['win_rate']:.1%} avg_net={info['avg_net_pnl']:+.2f}"
            )
    if summary["exit_reason_performance"]:
        print("Exit reasons:")
        for reason, info in sorted(summary["exit_reason_performance"].items()):
            print(
                f"  {reason}: count={info['count']} win_rate={info['win_rate']:.1%} "
                f"avg_net={info['avg_net_pnl']:+.2f}"
            )
    if summary["majorafter"]:
        ma = summary["majorafter"]
        print(
            f"MajorAfter: locked={ma.get('locked')} "
            f"shadow_closed={ma.get('shadow_closed_count', 0)} "
            f"shadow_open={ma.get('shadow_open_count', 0)} "
            f"shadow_total={ma.get('shadow_total_pnl', 0.0):+.2f} USDT"
        )
    print("-" * 60)

    msg = (
        f"[CRYPTO JOURNAL] {date_str} | "
        f"signals={summary['signal_count']} rejected={summary['reject_count']} "
        f"trades={summary['trade_count']} closed={summary['closed_trade_count']} "
        f"win_rate={summary['win_rate']:.1%} max_dd={summary['max_drawdown']:.2f} USDT"
    )
    send_telegram_alert(msg)
    algo_log.system(f"{msg} | csv={signal_csv_path.name}")

    # Run reconciliation after journal (00:00 UTC trigger)
    try:
        from crypto_reconciliation import crypto_recon
        crypto_recon.run()
    except Exception as _recon_exc:
        print(f"[CRYPTO JOURNAL] Reconciliation hatasi (atlanıyor): {_recon_exc}")


ensure_journal_schema()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto signal journal")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    args = parser.parse_args()
    run_journal(args.date)
