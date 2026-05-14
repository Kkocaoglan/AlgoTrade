"""
portfolio_risk.py -- Portfolio Heat & Correlation Gate
-------------------------------------------------------
Calculates total portfolio risk exposure and blocks new entries that would
push heat above MAX_PORTFOLIO_HEAT or that are highly correlated to existing
open positions.

Usage:
  python3.12 portfolio_risk.py --report     # print current state
  python3.12 portfolio_risk.py --check GARAN 1000   # check if symbol+risk_tl is allowed
"""

import sys
import sqlite3
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from bist_config import (
    CORR_THRESHOLD,
    CORR_WINDOW_DAYS,
    MAX_PORTFOLIO_HEAT,
    TOTAL_CAPITAL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Imported from bist_config.py so risk gates share loop_trader.py active defaults.


class PortfolioRisk:
    DB_PATH = "trade_data.db"

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------
    def _get_conn(self):
        return sqlite3.connect(self.DB_PATH)

    # ------------------------------------------------------------------
    def get_open_positions(self) -> list[dict]:
        """
        Fetch all open positions from paper_positions.
        Attaches current_price from the most recent ohlcv close for each symbol.
        Returns list of dicts: {symbol, entry_price, current_price, stop_loss, size_tl}
        """
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "SELECT symbol, entry_price, stop_price, size_tl "
                "FROM paper_positions WHERE status = 'open'"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        positions = []
        for symbol, entry_price, stop_price, size_tl in rows:
            current_price = self._get_latest_close(symbol)
            positions.append({
                "symbol":        symbol,
                "entry_price":   float(entry_price),
                "current_price": float(current_price) if current_price else float(entry_price),
                "stop_loss":     float(stop_price),
                "size_tl":       float(size_tl),
            })
        return positions

    def _get_latest_close(self, symbol: str) -> float | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT close FROM ohlcv WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            return float(row[0]) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def calc_portfolio_heat(self, positions: list) -> float:
        """
        Total portfolio heat = sum of per-position risk / TOTAL_CAPITAL.
        Per-position risk = (entry - stop) / entry * size_tl   (downside TL).
        Returns a float, e.g. 0.043 means 4.3%.
        """
        if not positions:
            return 0.0
        total_risk = 0.0
        for p in positions:
            entry  = p["entry_price"]
            stop   = p["stop_loss"]
            size   = p["size_tl"]
            if entry <= 0:
                continue
            risk_tl   = max(0.0, (entry - stop) / entry * size)
            total_risk += risk_tl
        return total_risk / TOTAL_CAPITAL

    # ------------------------------------------------------------------
    def get_price_history(self, symbols: list, days: int = CORR_WINDOW_DAYS) -> pd.DataFrame:
        """
        Fetch last `days` trading days of close prices from ohlcv.
        Returns a DataFrame pivoted: index=date, columns=symbol, values=close.
        """
        if not symbols:
            return pd.DataFrame()

        cutoff = (datetime.now() - timedelta(days=days + 10)).date().isoformat()
        conn   = self._get_conn()
        try:
            placeholders = ",".join("?" * len(symbols))
            query = (
                f"SELECT symbol, date, close FROM ohlcv "
                f"WHERE symbol IN ({placeholders}) AND date >= ? "
                f"ORDER BY date"
            )
            params = symbols + [cutoff]
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        pivot = df.pivot(index="date", columns="symbol", values="close")
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index().tail(days)
        return pivot

    # ------------------------------------------------------------------
    def calc_correlation_matrix(self, symbols: list) -> pd.DataFrame:
        """
        Compute pairwise return correlation for `symbols` over CORR_WINDOW_DAYS.
        Returns correlation DataFrame (symbols x symbols).
        Empty DataFrame if fewer than 2 symbols or insufficient data.
        """
        if len(symbols) < 2:
            return pd.DataFrame()

        price_df = self.get_price_history(symbols)
        if price_df.empty or price_df.shape[0] < 5:
            return pd.DataFrame()

        # Keep only requested symbols present in the data
        available = [s for s in symbols if s in price_df.columns]
        if len(available) < 2:
            return pd.DataFrame()

        returns = price_df[available].pct_change().dropna(how="all")
        corr    = returns.corr()
        return corr

    # ------------------------------------------------------------------
    def check_new_position_allowed(
        self, new_symbol: str, new_risk_tl: float
    ) -> tuple[bool, str]:
        """
        Gate a proposed new entry against two rules:
          1. Portfolio heat after entry must not exceed MAX_PORTFOLIO_HEAT.
          2. Correlation to any open position must be below CORR_THRESHOLD.

        Args:
            new_symbol  : ticker being considered (e.g. "GARAN")
            new_risk_tl : expected TL at risk for the new trade
                          (typically: size_tl * stop_dist / entry_price)

        Returns:
            (True,  "OK")          — allowed
            (False, reason_string) — blocked
        """
        positions = self.get_open_positions()

        # --- Rule 1: Portfolio heat cap ------------------------------------
        current_heat   = self.calc_portfolio_heat(positions)
        new_heat_delta = new_risk_tl / TOTAL_CAPITAL
        projected_heat = current_heat + new_heat_delta

        if projected_heat > MAX_PORTFOLIO_HEAT:
            msg = (
                f"Portfolio heat cap: current={current_heat:.1%} "
                f"+ new={new_heat_delta:.1%} = {projected_heat:.1%} "
                f"> max={MAX_PORTFOLIO_HEAT:.0%}"
            )
            return False, msg

        # --- Rule 2: Correlation to open positions -------------------------
        open_syms = [p["symbol"] for p in positions]
        if open_syms:
            all_syms = list(set(open_syms + [new_symbol]))
            corr_df  = self.calc_correlation_matrix(all_syms)

            if not corr_df.empty and new_symbol in corr_df.columns:
                for existing_sym in open_syms:
                    if existing_sym not in corr_df.index:
                        continue
                    c = corr_df.loc[existing_sym, new_symbol]
                    if pd.notna(c) and abs(c) >= CORR_THRESHOLD:
                        msg = (
                            f"High correlation to {existing_sym}: "
                            f"corr={c:.2f} >= threshold={CORR_THRESHOLD:.2f}"
                        )
                        return False, msg

        return True, "OK"

    # ------------------------------------------------------------------
    def get_risk_report(self) -> dict:
        """
        Full risk snapshot for the current portfolio.

        Returns dict with keys:
          total_heat        : float (e.g. 0.043)
          positions_risk    : list of {symbol, risk_tl, risk_pct, heat_contribution}
          corr_warnings     : list of {sym1, sym2, correlation} for pairs above threshold
          allowed_new_position : bool (True if heat budget remains for another trade)
        """
        positions = self.get_open_positions()

        # Per-position risk
        pos_risk = []
        for p in positions:
            entry = p["entry_price"]
            stop  = p["stop_loss"]
            size  = p["size_tl"]
            if entry <= 0:
                continue
            risk_tl      = max(0.0, (entry - stop) / entry * size)
            risk_pct     = risk_tl / TOTAL_CAPITAL * 100
            pos_risk.append({
                "symbol":            p["symbol"],
                "entry_price":       p["entry_price"],
                "stop_loss":         p["stop_loss"],
                "current_price":     p["current_price"],
                "size_tl":           p["size_tl"],
                "risk_tl":           round(risk_tl, 2),
                "risk_pct":          round(risk_pct, 2),
                "heat_contribution": round(risk_tl / TOTAL_CAPITAL, 4),
            })

        total_heat = sum(p["heat_contribution"] for p in pos_risk)

        # Correlation matrix + warnings
        open_syms    = [p["symbol"] for p in positions]
        corr_df      = self.calc_correlation_matrix(open_syms) if len(open_syms) >= 2 else pd.DataFrame()
        corr_warnings = []

        if not corr_df.empty:
            syms = corr_df.columns.tolist()
            for i, s1 in enumerate(syms):
                for s2 in syms[i + 1:]:
                    c = corr_df.loc[s1, s2]
                    if pd.notna(c) and abs(c) >= CORR_THRESHOLD:
                        corr_warnings.append({
                            "sym1":        s1,
                            "sym2":        s2,
                            "correlation": round(float(c), 3),
                        })

        # Remaining heat budget
        remaining_heat_tl = max(0.0, (MAX_PORTFOLIO_HEAT - total_heat) * TOTAL_CAPITAL)
        allowed_new       = total_heat < MAX_PORTFOLIO_HEAT

        return {
            "total_heat":            round(total_heat, 4),
            "total_heat_pct":        round(total_heat * 100, 2),
            "heat_budget_used_pct":  round(total_heat / MAX_PORTFOLIO_HEAT * 100, 1),
            "remaining_heat_tl":     round(remaining_heat_tl, 2),
            "positions_risk":        pos_risk,
            "corr_matrix":           corr_df,
            "corr_warnings":         corr_warnings,
            "allowed_new_position":  allowed_new,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(pr: PortfolioRisk):
    report = pr.get_risk_report()

    print("=" * 60)
    print("PORTFOLIO RISK REPORT")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    total_h = report["total_heat"]
    heat_pct = report["total_heat_pct"]
    budget   = report["heat_budget_used_pct"]
    remaining = report["remaining_heat_tl"]

    bar_len  = 20
    filled   = int(budget / 100 * bar_len)
    bar      = "#" * filled + "." * (bar_len - filled)

    print(f"\nPORTFOLIO HEAT: {heat_pct:.2f}%  (max {MAX_PORTFOLIO_HEAT*100:.0f}%)")
    print(f"  [{bar}]  {budget:.1f}% of budget used")
    print(f"  Remaining budget: {remaining:,.0f} TL")
    if not report["allowed_new_position"]:
        print("  *** HEAT CAP REACHED — no new positions allowed ***")

    print(f"\nPOZISYON RISK DETAYI ({len(report['positions_risk'])} acik pozisyon):")
    if report["positions_risk"]:
        hdr = f"  {'Symbol':<8}  {'Entry':>8}  {'Stop':>8}  {'Current':>8}  {'Size TL':>9}  {'Risk TL':>8}  {'Risk %':>6}  {'Heat':>6}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for p in report["positions_risk"]:
            cur_vs_entry = (p["current_price"] - p["entry_price"]) / p["entry_price"] * 100
            print(
                f"  {p['symbol']:<8}  "
                f"{p['entry_price']:>8.2f}  "
                f"{p['stop_loss']:>8.2f}  "
                f"{p['current_price']:>7.2f} ({cur_vs_entry:+.1f}%)  "
                f"{p['size_tl']:>9,.0f}  "
                f"{p['risk_tl']:>8,.0f}  "
                f"{p['risk_pct']:>5.2f}%  "
                f"{p['heat_contribution']*100:>5.2f}%"
            )
    else:
        print("  (no open positions)")

    print(f"\nKORELASYON MATRISI ({CORR_WINDOW_DAYS} gunluk fiyat verisi):")
    corr_df = report["corr_matrix"]
    if not corr_df.empty:
        pd.set_option("display.float_format", "{:.3f}".format)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(corr_df.to_string(float_format="{:.3f}".format))
    else:
        print("  (< 2 open positions — correlation not applicable)")

    print(f"\nKORELASYON UYARILARI (> {CORR_THRESHOLD}):")
    if report["corr_warnings"]:
        for w in report["corr_warnings"]:
            print(f"  {w['sym1']} <-> {w['sym2']}  corr={w['correlation']:.3f}  *** HIGH CORRELATION ***")
    else:
        print("  (no high-correlation pairs)")

    print("=" * 60)


def _check_symbol(pr: PortfolioRisk, symbol: str, risk_tl: float):
    allowed, reason = pr.check_new_position_allowed(symbol, risk_tl)
    status = "ALLOWED" if allowed else "BLOCKED"
    print(f"check_new_position_allowed({symbol}, risk_tl={risk_tl:.0f} TL)")
    print(f"  Result : {status}")
    if not allowed:
        print(f"  Reason : {reason}")
    else:
        report = pr.get_risk_report()
        proj   = report["total_heat"] + risk_tl / TOTAL_CAPITAL
        print(f"  Projected heat after entry: {proj*100:.2f}%  "
              f"(max {MAX_PORTFOLIO_HEAT*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Risk Module")
    parser.add_argument("--report", action="store_true",
                        help="Print current portfolio heat and correlation report")
    parser.add_argument("--check", nargs=2, metavar=("SYMBOL", "RISK_TL"),
                        help="Check if a new position is allowed: --check GARAN 1000")
    args = parser.parse_args()

    pr = PortfolioRisk()

    if args.report:
        _print_report(pr)
    elif args.check:
        symbol  = args.check[0].upper()
        risk_tl = float(args.check[1])
        _check_symbol(pr, symbol, risk_tl)
        print()
        _print_report(pr)
    else:
        parser.print_help()
