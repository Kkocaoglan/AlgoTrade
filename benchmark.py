"""
benchmark.py — System vs XU100 vs 29-Stock Buy & Hold comparison
-----------------------------------------------------------------
Usage:
  python3.12 benchmark.py              # full report + chart
  python3.12 benchmark.py --no-plot   # skip matplotlib output

Outputs:
  - Formatted comparison table (stdout)
  - results/benchmark_comparison.png
  - Verdict: BEATING BENCHMARK or UNDERPERFORMING
"""

import sqlite3
import sys
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH       = "trade_data.db"
START_DATE    = "2026-04-08"          # paper trading inception
TOTAL_CAPITAL = 100_000.0             # TL
RESULTS_DIR   = Path("results")
OUTPUT_PNG    = RESULTS_DIR / "benchmark_comparison.png"
PROXY_SYMS    = ["GARAN", "THYAO", "EREGL"]   # XU100 fallback
TRADING_DAYS_PER_YEAR = 252
# Risk-free rate set to 0 for strategy-vs-strategy Sharpe comparison.
# Using TCMB 43% would unfairly penalise the system for holding cash
# at 0% in paper mode (cash earns nothing in the paper engine).
RISK_FREE_DAILY = 0.0

UNIVERSE = [
    "YKBNK","AKBNK","ISCTR","GARAN","TUPRS","PETKM","TAVHL","FROTO",
    "TCELL","ASELS","BIMAS","MGROS","ENKAI","EKGYO","THYAO","EREGL",
    "KCHOL","SAHOL","SISE","TOASO","ARCLK","VESTL","KRDMD","PGSUS",
    "ODAS","GUBRF","CIMSA","LOGO","NETAS",
]


# =============================================================================
# Helpers
# =============================================================================

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_sharpe(daily_rets: pd.Series) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    excess = daily_rets - RISK_FREE_DAILY
    std    = excess.std()
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(cum_series: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.08 = 8%)."""
    roll_max = cum_series.cummax()
    dd       = (cum_series - roll_max) / roll_max
    return float(abs(dd.min()))


def _annualise(total_ret: float, n_days: int) -> float:
    if n_days <= 0:
        return float("nan")
    return float((1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


# =============================================================================
# 1. XU100 index
# =============================================================================

def fetch_xu100(start: str, end: str) -> pd.Series:
    """Return daily close series for XU100.IS.  Falls back to proxy."""
    print("Fetching XU100.IS from yfinance ...", end=" ", flush=True)
    try:
        raw = yf.download("XU100.IS", start=start, end=end, progress=False)
        closes = raw["Close"].squeeze().dropna()
        if len(closes) >= 5:
            print(f"OK ({len(closes)} days)")
            closes.index = pd.to_datetime(closes.index).normalize()
            return closes.rename("XU100")
        print("too few rows — switching to proxy")
    except Exception as e:
        print(f"error ({e}) — switching to proxy")

    # Fallback: equal-weight GARAN+THYAO+EREGL from DB
    print(f"Using proxy ({'+'.join(PROXY_SYMS)}) ...", end=" ", flush=True)
    conn = _db_conn()
    frames = []
    for sym in PROXY_SYMS:
        df = pd.read_sql_query(
            "SELECT date, close FROM ohlcv WHERE symbol=? AND date >= ? ORDER BY date",
            conn, params=(sym, start),
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        frames.append(df["close"].rename(sym))
    conn.close()
    proxy = pd.concat(frames, axis=1).mean(axis=1).rename("XU100")
    print(f"OK ({len(proxy)} days)")
    return proxy


# =============================================================================
# 2. 29-Stock equal-weight buy & hold
# =============================================================================

def fetch_29stock_bah(start: str, end: str) -> tuple:
    """Return (daily_returns Series, cum_equity Series) for equal-weight B&H."""
    print("Building 29-stock equal-weight B&H ...", end=" ", flush=True)
    conn = _db_conn()
    frames = []
    for sym in UNIVERSE:
        df = pd.read_sql_query(
            "SELECT date, close FROM ohlcv WHERE symbol=? AND date >= ? ORDER BY date",
            conn, params=(sym, start),
        )
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        frames.append(df["close"].rename(sym))
    conn.close()

    prices = pd.concat(frames, axis=1, sort=True).ffill()
    # Daily portfolio return = mean of individual daily returns (equal weight)
    daily_rets = prices.pct_change().mean(axis=1).dropna()
    cum_equity = TOTAL_CAPITAL * (1 + daily_rets).cumprod()
    print(f"OK ({len(daily_rets)} days, {len(frames)}/29 symbols)")
    return daily_rets, cum_equity


# =============================================================================
# 3. System equity curve (reconstructed from paper_trades)
# =============================================================================

def build_system_equity(start: str, end: str) -> tuple:
    """
    Reconstruct daily equity from paper_trades PnL.

    Strategy:
    - Start with TOTAL_CAPITAL on start_date.
    - For each calendar day, accumulate PnL from trades that closed that day.
    - paper_equity table has only 1 row (2026-04-14) so we always reconstruct.

    Returns:
      (daily_equity Series, trade_markers list of (date, equity, label))
    """
    print("Reconstructing system equity curve ...", end=" ", flush=True)
    conn = _db_conn()

    trades = pd.read_sql_query(
        """SELECT entry_date, exit_date, symbol, pnl, size_tl, is_long
           FROM paper_trades ORDER BY exit_date""",
        conn,
    )
    conn.close()

    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"]).dt.normalize()

    # Build date index from start_date to today
    today   = date.today()
    all_dates = pd.date_range(start=start, end=today.isoformat(), freq="B")  # business days

    equity = TOTAL_CAPITAL
    equity_series = {}
    for d in all_dates:
        day_trades = trades[trades["exit_date"] == d]
        equity += day_trades["pnl"].sum()
        equity_series[d] = equity

    equity_s = pd.Series(equity_series, name="System")

    # Daily returns
    daily_rets = equity_s.pct_change().dropna()

    # Trade markers: (date, equity_value, "BUY SYM" or "SELL SYM")
    markers = []
    for _, t in trades.iterrows():
        direction = "BUY" if t["is_long"] else "SHORT"
        markers.append((
            t["entry_date"],
            equity_series.get(t["entry_date"], TOTAL_CAPITAL),
            f"{direction} {t['symbol']}",
        ))
        markers.append((
            t["exit_date"],
            equity_series.get(t["exit_date"], TOTAL_CAPITAL),
            f"EXIT {t['symbol']}",
        ))

    print(f"OK ({len(equity_series)} days, {len(trades)} trades)")
    return daily_rets, equity_s, markers


# =============================================================================
# 4. Metrics calculation
# =============================================================================

def compute_metrics(
    daily_rets: pd.Series,
    cum_equity: pd.Series,
    initial: float = TOTAL_CAPITAL,
    trades_df=None,
) -> dict:
    cum_arr = cum_equity.dropna()
    if cum_arr.empty:
        return {}

    total_ret = float(cum_arr.iloc[-1] / initial - 1)
    n_days    = (cum_arr.index[-1] - cum_arr.index[0]).days
    ann_ret   = _annualise(total_ret, n_days)
    mdd       = _max_drawdown(cum_arr)
    sharpe    = _safe_sharpe(daily_rets.dropna())

    win_rate = float("nan")
    n_trades = 0
    if trades_df is not None and not trades_df.empty:
        n_trades = len(trades_df)
        win_rate = (trades_df["pnl"] > 0).mean()

    return {
        "total_ret": total_ret,
        "ann_ret":   ann_ret,
        "mdd":       mdd,
        "sharpe":    sharpe,
        "win_rate":  win_rate,
        "n_trades":  n_trades,
    }


# =============================================================================
# 5. Print comparison table
# =============================================================================

def _fmt(val, fmt=".2%", na="N/A") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return na
    if fmt == ".2%":
        return f"{val:.2%}"
    if fmt == ".3f":
        return f"{val:.3f}"
    if fmt == "d":
        return str(int(val))
    return str(val)


def print_table(sys_m: dict, xu100_m: dict, bah_m: dict) -> None:
    rows = [
        ("Total Return %",  "total_ret", ".2%"),
        ("Ann. Return %",   "ann_ret",   ".2%"),
        ("Max Drawdown %",  "mdd",       ".2%"),
        ("Sharpe Ratio",    "sharpe",    ".3f"),
        ("Win Rate",        "win_rate",  ".2%"),
        ("Trades",          "n_trades",  "d"),
    ]
    hdr = f"{'Metric':<20}  {'Our System':>14}  {'XU100 B&H':>14}  {'29-Stock B&H':>14}"
    sep = "-" * len(hdr)
    print()
    print(sep)
    print(hdr)
    print(sep)
    for label, key, fmt in rows:
        sv = _fmt(sys_m.get(key, float("nan")), fmt)
        xv = _fmt(xu100_m.get(key, float("nan")), fmt)
        bv = _fmt(bah_m.get(key, float("nan")), fmt)
        print(f"{label:<20}  {sv:>14}  {xv:>14}  {bv:>14}")
    print(sep)
    print()


# =============================================================================
# 6. Plot equity curves
# =============================================================================

def plot_curves(
    sys_equity:    pd.Series,
    xu100_prices:  pd.Series,
    bah_equity:    pd.Series,
    trade_markers: list,
    save_path:     Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"BIST Algo System vs Benchmarks\n(Paper trading since {START_DATE})",
        fontsize=13, fontweight="bold",
    )

    # Normalise all series to 100 at start
    def _norm(s):
        s = s.dropna()
        return s / s.iloc[0] * 100

    sys_n   = _norm(sys_equity)
    xu_n    = _norm(xu100_prices)
    bah_n   = _norm(bah_equity)

    # Align to common dates
    common = sys_n.index.intersection(xu_n.index).intersection(bah_n.index)

    ax1.plot(sys_n.loc[common], color="#2196F3", lw=2.0, label="Our System", zorder=3)
    ax1.plot(xu_n.loc[common],  color="#FF9800", lw=1.5, label="XU100",      zorder=2)
    ax1.plot(bah_n.loc[common], color="#4CAF50", lw=1.5, label="29-Stock EW B&H", zorder=2,
             linestyle="--")
    ax1.axhline(100, color="grey", lw=0.8, linestyle=":")
    ax1.set_ylabel("Normalised Value (base=100)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Trade markers — entries green ^, exits red v (on system line)
    entries = [(d, v) for (d, v, l) in trade_markers if l.startswith(("BUY","SHORT"))]
    exits   = [(d, v) for (d, v, l) in trade_markers if l.startswith("EXIT")]

    if entries:
        ed = pd.DatetimeIndex([e[0] for e in entries])
        ev = pd.Series([sys_n.get(d, float("nan")) for d in ed], index=ed)
        ev = ev.dropna()
        ax1.scatter(ev.index, ev.values, marker="^", color="#1565C0",
                    s=60, zorder=5, label="Entry", alpha=0.8)

    if exits:
        xd = pd.DatetimeIndex([e[0] for e in exits])
        xv = pd.Series([sys_n.get(d, float("nan")) for d in xd], index=xd)
        xv = xv.dropna()
        ax1.scatter(xv.index, xv.values, marker="v", color="#B71C1C",
                    s=60, zorder=5, label="Exit", alpha=0.8)

    ax1.legend(loc="upper left", framealpha=0.9)

    # Drawdown subplot — system only
    roll_max = sys_n.cummax()
    dd_pct   = (sys_n - roll_max) / roll_max * 100
    ax2.fill_between(sys_n.index, dd_pct, 0, color="#2196F3", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {save_path}")


# =============================================================================
# 7. Verdict
# =============================================================================

def print_verdict(sys_m: dict, xu100_m: dict, bah_m: dict) -> None:
    sys_s  = sys_m.get("sharpe", float("nan"))
    xu_s   = xu100_m.get("sharpe", float("nan"))
    bah_s  = bah_m.get("sharpe", float("nan"))

    beats_xu100 = (not math.isnan(sys_s)) and (not math.isnan(xu_s)) and sys_s > xu_s
    beats_bah   = (not math.isnan(sys_s)) and (not math.isnan(bah_s)) and sys_s > bah_s

    sys_ret  = sys_m.get("total_ret", float("nan"))
    xu_ret   = xu100_m.get("total_ret", float("nan"))
    bah_ret  = bah_m.get("total_ret", float("nan"))
    alpha_xu = sys_ret - xu_ret   if not math.isnan(xu_ret)  else float("nan")
    alpha_bah= sys_ret - bah_ret  if not math.isnan(bah_ret) else float("nan")

    print("=" * 55)
    if beats_xu100 and beats_bah:
        print("  VERDICT: ✓ BEATING BOTH BENCHMARKS")
    elif beats_xu100:
        print("  VERDICT: ✓ BEATING XU100  |  behind 29-stock EW")
    elif beats_bah:
        print("  VERDICT: ✓ BEATING 29-STOCK EW  |  behind XU100")
    else:
        print("  VERDICT: ✗ UNDERPERFORMING — review signal quality")
    print()
    if not math.isnan(alpha_xu):
        print(f"  Alpha vs XU100     : {alpha_xu:+.2%}")
    if not math.isnan(alpha_bah):
        print(f"  Alpha vs 29-stk EW : {alpha_bah:+.2%}")
    print(f"  System Sharpe      : {_fmt(sys_s, '.3f')}")
    print(f"  XU100 Sharpe       : {_fmt(xu_s,  '.3f')}")
    print(f"  29-Stock EW Sharpe : {_fmt(bah_s, '.3f')}")
    print()
    # Practical guidance
    if not beats_xu100:
        sys_mdd = sys_m.get("mdd", float("nan"))
        xu_mdd  = xu100_m.get("mdd", float("nan"))
        if not math.isnan(sys_mdd) and not math.isnan(xu_mdd) and sys_mdd < xu_mdd:
            print("  Note: System has lower drawdown than XU100 — risk-adjusted")
            print("        performance may improve as trade count grows.")
        else:
            print("  Action: Win rate < 58% target. Need >= 30 trades for")
            print("          robust signal evaluation (current:", sys_m.get("n_trades", "?"), "trades).")
    print("=" * 55)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    no_plot = "--no-plot" in sys.argv
    today   = date.today().isoformat()
    end     = (date.today() + timedelta(days=1)).isoformat()

    print(f"\nBenchmark Comparison  |  {START_DATE} → {today}")
    print("=" * 55)

    # ── data fetch ─────────────────────────────────────────────────────────
    xu100_prices = fetch_xu100(START_DATE, end)

    bah_daily, bah_equity = fetch_29stock_bah(START_DATE, today)

    sys_daily, sys_equity, trade_markers = build_system_equity(START_DATE, today)

    # ── load trades for win-rate ──────────────────────────────────────────
    conn   = _db_conn()
    trades = pd.read_sql_query("SELECT pnl, is_long FROM paper_trades", conn)
    conn.close()

    # ── metrics ────────────────────────────────────────────────────────────
    # XU100: normalise to TOTAL_CAPITAL
    xu100_eq = TOTAL_CAPITAL * xu100_prices / xu100_prices.iloc[0]
    xu100_dr = xu100_prices.pct_change().dropna().rename("XU100")

    # For system Sharpe: exclude flat days (cash sitting, equity unchanged).
    # These dilute std toward zero and produce extreme Sharpe values.
    sys_daily_active = sys_daily[sys_daily != 0]
    sys_m   = compute_metrics(sys_daily_active, sys_equity,  TOTAL_CAPITAL, trades)
    xu100_m = compute_metrics(xu100_dr,         xu100_eq)
    bah_m   = compute_metrics(bah_daily,         bah_equity)

    # ── print results ──────────────────────────────────────────────────────
    print_table(sys_m, xu100_m, bah_m)

    # ── plot ───────────────────────────────────────────────────────────────
    if not no_plot:
        try:
            plot_curves(sys_equity, xu100_prices, bah_equity, trade_markers, OUTPUT_PNG)
        except Exception as e:
            print(f"Plot error (non-fatal): {e}")

    # ── verdict ────────────────────────────────────────────────────────────
    print_verdict(sys_m, xu100_m, bah_m)


if __name__ == "__main__":
    main()
