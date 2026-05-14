"""
bist_live_wf_sim.py -- Live-rule BIST signal simulator.

Read-only analysis tool. Uses signals_log + OHLCV/indicators to replay logged
model signals with loop_trader-style sizing, gates, stops, targets, trailing,
and model-flip exits.

Usage:
  py -3.12 bist_live_wf_sim.py
  py -3.12 bist_live_wf_sim.py --thresholds 0.60,0.65,0.70
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bist_config import (
    ATR_TARGET_MULT,
    BUY_THRESHOLD,
    MAX_POS,
    MAX_POSITION_SIZE,
    MIN_CASH_RESERVE,
    MODEL_FLIP_THRESHOLD,
    RISK_PER_TRADE,
    SHORT_SL_PCT,
    SHORT_THRESHOLD,
    SL_PCT,
    TOTAL_CAPITAL,
    TRAIL_BE_PCT,
    TRAIL_DIST_PCT,
    TRAIL_START_PCT,
)

DB_PATH = Path(__file__).parent / "trade_data.db"
COMMISSION_ROUND_TRIP = 0.002
ENTRY_SLIPPAGE = 0.001
EXIT_SLIPPAGE = 0.001
MIN_SIZE_TL = 1_000


@dataclass
class Position:
    symbol: str
    direction: str
    entry_date: pd.Timestamp
    entry_price: float
    size_tl: float
    stop: float
    target: float
    hwm: float
    lwm: float


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _load_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = _table_columns(conn, "signals_log")
    required = {"symbol", "prob_buy"}
    if not required.issubset(cols):
        return pd.DataFrame()

    wanted = [
        "ts", "signal_date", "symbol", "prob_buy", "prob_sell", "regime",
        "gate_result", "reason_blocked", "trade_opened",
    ]
    select_cols = [col for col in wanted if col in cols]
    df = pd.read_sql(
        f"SELECT {', '.join(select_cols)} FROM signals_log ORDER BY COALESCE(signal_date, ts), symbol",
        conn,
    )
    if df.empty:
        return df
    date_col = "signal_date" if "signal_date" in df.columns else "ts"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["prob_buy"] = pd.to_numeric(df["prob_buy"], errors="coerce")
    if "prob_sell" in df.columns:
        df["prob_sell"] = pd.to_numeric(df["prob_sell"], errors="coerce")
    else:
        df["prob_sell"] = 1.0 - df["prob_buy"]
    if "gate_result" not in df.columns:
        df["gate_result"] = "PASS"
    df = df.dropna(subset=["date", "symbol", "prob_buy"])
    return df


def _load_market(conn: sqlite3.Connection, symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ohlcv = pd.read_sql(
            """
            SELECT date, open, high, low, close
            FROM ohlcv
            WHERE symbol=?
            ORDER BY date
            """,
            conn,
            params=(sym,),
        )
        ind = pd.read_sql(
            """
            SELECT date, atr14
            FROM indicators
            WHERE symbol=?
            ORDER BY date
            """,
            conn,
            params=(sym,),
        )
        if ohlcv.empty:
            continue
        df = pd.merge(ohlcv, ind, on="date", how="left")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "atr14"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[sym] = df
    return frames


def _fill_price(raw_price: float, direction: str, side: str) -> float:
    is_long = direction == "long"
    adverse_up = is_long if side == "entry" else not is_long
    slip = ENTRY_SLIPPAGE if side == "entry" else EXIT_SLIPPAGE
    return raw_price * (1.0 + slip if adverse_up else 1.0 - slip)


def _pnl(pos: Position, exit_price: float) -> float:
    if pos.direction == "long":
        gross = pos.size_tl * ((exit_price - pos.entry_price) / pos.entry_price)
    else:
        gross = pos.size_tl * ((pos.entry_price - exit_price) / pos.entry_price)
    return gross - pos.size_tl * COMMISSION_ROUND_TRIP


def _position_size(price: float, atr: float, cash: float) -> float:
    if price <= 0 or atr <= 0:
        return 0.0
    risk_tl = TOTAL_CAPITAL * RISK_PER_TRADE
    stop_dist = atr * ATR_TARGET_MULT
    size_by_risk = risk_tl / (stop_dist / price)
    return max(0.0, min(size_by_risk, MAX_POSITION_SIZE, cash - MIN_CASH_RESERVE))


def _next_bar(frames: dict[str, pd.DataFrame], symbol: str, signal_date: pd.Timestamp):
    df = frames.get(symbol)
    if df is None:
        return None, None
    later = df.index[df.index > signal_date]
    if len(later) == 0:
        return None, None
    dt = later[0]
    return dt, df.loc[dt]


def _close_position(pos: Position, raw_exit: float, reason: str, date: pd.Timestamp) -> dict:
    exit_price = _fill_price(raw_exit, pos.direction, "exit")
    pnl = _pnl(pos, exit_price)
    return {
        "symbol": pos.symbol,
        "direction": pos.direction,
        "entry_date": pos.entry_date,
        "exit_date": date,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "size_tl": pos.size_tl,
        "pnl": pnl,
        "return_pct": pnl / pos.size_tl if pos.size_tl else 0.0,
        "reason": reason,
    }


def _check_exit(pos: Position, row: pd.Series, date: pd.Timestamp, prob_buy: float | None):
    open_p = float(row["open"])
    high_p = float(row["high"])
    low_p = float(row["low"])
    close_p = float(row["close"])

    if pos.direction == "long":
        pos.hwm = max(pos.hwm, high_p)
        effective_stop = pos.stop
        profit = (pos.hwm - pos.entry_price) / pos.entry_price
        if profit >= TRAIL_START_PCT:
            effective_stop = max(effective_stop, pos.hwm * (1.0 - TRAIL_DIST_PCT))
        elif profit >= TRAIL_BE_PCT:
            effective_stop = max(effective_stop, pos.entry_price)
        if open_p <= effective_stop:
            return _close_position(pos, open_p, "stop_gap", date)
        if low_p <= effective_stop:
            return _close_position(pos, effective_stop, "stop_loss", date)
        if high_p >= pos.target:
            return _close_position(pos, pos.target, "target_hit", date)
        if prob_buy is not None and prob_buy < MODEL_FLIP_THRESHOLD:
            return _close_position(pos, close_p, "model_flip", date)
    else:
        pos.lwm = min(pos.lwm, low_p)
        effective_stop = pos.stop
        profit = (pos.entry_price - pos.lwm) / pos.entry_price
        if profit >= TRAIL_START_PCT:
            effective_stop = min(effective_stop, pos.lwm * (1.0 + TRAIL_DIST_PCT))
        elif profit >= TRAIL_BE_PCT:
            effective_stop = min(effective_stop, pos.entry_price)
        if open_p >= effective_stop:
            return _close_position(pos, open_p, "stop_gap", date)
        if high_p >= effective_stop:
            return _close_position(pos, effective_stop, "stop_loss", date)
        if low_p <= pos.target:
            return _close_position(pos, pos.target, "target_hit", date)
        if prob_buy is not None and prob_buy > (1.0 - MODEL_FLIP_THRESHOLD):
            return _close_position(pos, close_p, "model_flip", date)
    return None


def simulate(signals: pd.DataFrame, frames: dict[str, pd.DataFrame], buy_threshold: float) -> dict:
    dates = sorted({dt for df in frames.values() for dt in df.index})
    if signals.empty or not dates:
        return {"threshold": buy_threshold, "trades": [], "equity": [TOTAL_CAPITAL]}

    signal_by_date = {
        dt: day.sort_values("prob_buy", ascending=False)
        for dt, day in signals.groupby("date")
    }
    prob_map = {
        (row.date, row.symbol): float(row.prob_buy)
        for row in signals.itertuples(index=False)
    }

    cash = float(TOTAL_CAPITAL)
    positions: dict[str, Position] = {}
    pending: list[tuple[pd.Timestamp, str, str]] = []
    trades: list[dict] = []
    equity_curve: list[float] = []

    for dt in dates:
        # Execute entries scheduled from prior signal day at today's open.
        still_pending = []
        for entry_dt, sym, direction in pending:
            if entry_dt != dt or sym in positions or len(positions) >= MAX_POS:
                still_pending.append((entry_dt, sym, direction))
                continue
            row = frames.get(sym, pd.DataFrame()).loc[dt] if sym in frames and dt in frames[sym].index else None
            if row is None:
                continue
            atr = float(row.get("atr14") or 0.0)
            raw_entry = float(row["open"])
            entry = _fill_price(raw_entry, direction, "entry")
            size = _position_size(entry, atr, cash)
            if size < MIN_SIZE_TL:
                continue
            cash -= size
            stop = entry * (1.0 - SL_PCT) if direction == "long" else entry * (1.0 + SHORT_SL_PCT)
            target = entry + atr * ATR_TARGET_MULT if direction == "long" else entry - atr * ATR_TARGET_MULT
            positions[sym] = Position(sym, direction, dt, entry, size, stop, target, entry, entry)
        pending = still_pending

        # Exits.
        for sym, pos in list(positions.items()):
            if sym not in frames or dt not in frames[sym].index:
                continue
            prob_buy = prob_map.get((dt, sym))
            exit_trade = _check_exit(pos, frames[sym].loc[dt], dt, prob_buy)
            if exit_trade:
                cash += pos.size_tl + float(exit_trade["pnl"])
                trades.append(exit_trade)
                del positions[sym]

        # Schedule next-bar entries from today's logged signals.
        day_signals = signal_by_date.get(dt)
        if day_signals is not None:
            for row in day_signals.itertuples(index=False):
                if len(positions) + len(pending) >= MAX_POS:
                    break
                gate_result = str(getattr(row, "gate_result", "") or "").upper()
                if gate_result == "BLOCK":
                    continue
                sym = row.symbol
                if sym in positions or any(p[1] == sym for p in pending):
                    continue
                direction = None
                if float(row.prob_buy) >= buy_threshold:
                    direction = "long"
                elif float(row.prob_sell) >= SHORT_THRESHOLD:
                    direction = "short"
                if direction is None:
                    continue
                next_dt, _ = _next_bar(frames, sym, dt)
                if next_dt is not None:
                    pending.append((next_dt, sym, direction))

        marked = cash
        for sym, pos in positions.items():
            if sym in frames and dt in frames[sym].index:
                close_p = float(frames[sym].loc[dt, "close"])
                if pos.direction == "long":
                    marked += pos.size_tl * (close_p / pos.entry_price)
                else:
                    marked += pos.size_tl * (1.0 + (pos.entry_price - close_p) / pos.entry_price)
            else:
                marked += pos.size_tl
        equity_curve.append(marked)

    return {
        "threshold": buy_threshold,
        "trades": trades,
        "equity": equity_curve,
        "open_positions": len(positions),
        "pending_entries": len(pending),
    }


def _max_drawdown(equity: list[float]) -> float:
    peak = None
    max_dd = 0.0
    for value in equity:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd


def _summarize(result: dict) -> dict:
    trades = result["trades"]
    equity = result["equity"]
    pnl_values = [float(t["pnl"]) for t in trades]
    pnl = sum(pnl_values)
    wins_list = [p for p in pnl_values if p > 0]
    losses_list = [p for p in pnl_values if p <= 0]
    wins = len(wins_list)
    gross_win = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    return {
        "threshold": result["threshold"],
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "net_pnl": pnl,
        "final_equity": equity[-1] if equity else TOTAL_CAPITAL,
        "max_drawdown": _max_drawdown(equity),
        "avg_pnl": pnl / len(trades) if trades else 0.0,
        "avg_win": gross_win / len(wins_list) if wins_list else 0.0,
        "avg_loss": -gross_loss / len(losses_list) if losses_list else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "expectancy": pnl / len(trades) if trades else 0.0,
        "open_positions": int(result.get("open_positions", 0)),
        "pending_entries": int(result.get("pending_entries", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay BIST signals_log with live-loop style rules.")
    parser.add_argument("--thresholds", default=str(BUY_THRESHOLD), help="Comma-separated BUY thresholds")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    conn = sqlite3.connect(DB_PATH)
    signals = _load_signals(conn)
    frames = _load_market(conn, sorted(signals["symbol"].dropna().unique().tolist()) if not signals.empty else [])
    conn.close()

    print("=" * 88)
    print("BIST LIVE-RULE WALK-FORWARD STYLE SIMULATOR")
    print("=" * 88)
    print(f"Signals loaded : {len(signals)}")
    print(f"Symbols loaded : {len(frames)}")
    print("Entry model    : next trading day open, adverse slippage, loop_trader sizing")
    print("Exit model     : gap-aware stop, ATR target, trailing, model flip when signal exists")
    print()
    if signals.empty or not frames:
        print("No usable signals_log/OHLCV data found. Run loop_trader long enough to populate signals_log.")
        return 0

    rows = [_summarize(simulate(signals, frames, thr)) for thr in thresholds]
    print(
        f"{'thr':>6} {'trades':>7} {'open':>5} {'pend':>5} {'win%':>8} "
        f"{'net_pnl':>12} {'exp/tr':>10} {'pf':>7} {'avg_win':>10} "
        f"{'avg_loss':>10} {'max_dd%':>9} {'final_eq':>12}"
    )
    print("-" * 122)
    for row in rows:
        pf = row["profit_factor"]
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(
            f"{row['threshold']:>6.2f} {row['trades']:>7} {row['open_positions']:>5} "
            f"{row['pending_entries']:>5} {row['win_rate']*100:>7.1f}% "
            f"{row['net_pnl']:>12.2f} {row['expectancy']:>10.2f} "
            f"{pf_str:>7} {row['avg_win']:>10.2f} {row['avg_loss']:>10.2f} "
            f"{row['max_drawdown']*100:>8.1f}% {row['final_equity']:>12.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
