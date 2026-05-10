"""
volatility_regime.py -- BIST Volatility Regime Filter
------------------------------------------------------
Computes realized volatility from ohlcv data and classifies the current
market regime as NORMAL / HIGH_VOL / EXTREME.

XU100 is not in the ohlcv table — uses an equal-weighted average of daily
returns from GARAN + THYAO + EREGL as a BIST100 proxy.

Usage:
  python3.12 volatility_regime.py --report     # print current regime
  python3.12 volatility_regime.py --history 10  # print last N days of vol
"""

import sys
import sqlite3
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIX_WINDOW             = 20      # trading days for realized vol calculation
HIGH_VOL_THRESHOLD     = 0.025   # daily vol >= 2.5% → HIGH_VOL
EXTREME_VOL_THRESHOLD  = 0.040   # daily vol >= 4.0% → EXTREME

# BIST100 proxy — equal-weighted average of daily returns
# (XU100 not stored in ohlcv; these are always present in the 14-stock universe)
BIST100_SYMBOL  = "XU100"       # checked first; fallback below if absent
PROXY_SYMBOLS   = ["GARAN", "THYAO", "EREGL"]   # fallback


class VolatilityRegime:
    DB_PATH = "trade_data.db"

    def _get_conn(self):
        return sqlite3.connect(self.DB_PATH)

    # ------------------------------------------------------------------
    def _proxy_available(self) -> bool:
        """Return True if XU100 has enough rows in ohlcv."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol = ?",
                (BIST100_SYMBOL,),
            ).fetchone()
            return (row[0] or 0) >= VIX_WINDOW + 2
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _fetch_proxy_returns(self, window: int) -> pd.Series:
        """
        Build a daily return series for the BIST100 proxy.

        If XU100 rows exist: use XU100 directly.
        Otherwise: load GARAN + THYAO + EREGL, compute daily pct_change per
        symbol, then average them (equal-weighted) per day.

        Returns a pd.Series of daily returns (no NaN, most recent last).
        """
        needed = window + 5   # a few extra rows for pct_change NaN drop

        if self._proxy_available():
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT date, close FROM ohlcv "
                    "WHERE symbol = ? ORDER BY date DESC LIMIT ?",
                    (BIST100_SYMBOL, needed),
                ).fetchall()
            finally:
                conn.close()

            dates  = [r[0] for r in reversed(rows)]
            closes = [r[1] for r in reversed(rows)]
            s      = pd.Series(closes, index=dates, dtype=float, name="xu100")
            return s.pct_change().dropna()

        # ---- Proxy: average daily returns of GARAN + THYAO + EREGL ----
        conn = self._get_conn()
        try:
            placeholders = ",".join("?" * len(PROXY_SYMBOLS))
            rows = conn.execute(
                f"SELECT symbol, date, close FROM ohlcv "
                f"WHERE symbol IN ({placeholders}) "
                f"ORDER BY date DESC LIMIT ?",
                PROXY_SYMBOLS + [needed * len(PROXY_SYMBOLS)],
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return pd.Series(dtype=float)

        df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        pivot       = df.pivot(index="date", columns="symbol", values="close").sort_index()
        returns     = pivot.pct_change()

        # Equal-weighted average return per day
        avg_ret = returns.mean(axis=1).dropna()
        return avg_ret

    # ------------------------------------------------------------------
    def get_realized_vol(self, symbol: str = "XU100", window: int = VIX_WINDOW) -> float:
        """
        Compute realized daily volatility (not annualized).

        Uses XU100 if present in ohlcv; falls back to GARAN+THYAO+EREGL proxy.
        The `symbol` param is accepted for API compatibility but ignored when
        XU100 is absent (proxy is always used in that case).

        Returns float, e.g. 0.018 means daily vol of 1.8%.
        Returns 0.0 on error or insufficient data.
        """
        try:
            returns = self._fetch_proxy_returns(window)
            if len(returns) < window:
                return 0.0
            return float(returns.iloc[-window:].std())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def get_regime(self) -> str:
        """
        Classify current volatility regime.
        Returns: "NORMAL", "HIGH_VOL", or "EXTREME"
        """
        vol = self.get_realized_vol()
        if vol >= EXTREME_VOL_THRESHOLD:
            return "EXTREME"
        if vol >= HIGH_VOL_THRESHOLD:
            return "HIGH_VOL"
        return "NORMAL"

    # ------------------------------------------------------------------
    def get_position_size_multiplier(self) -> float:
        """
        Size scaling factor for the current regime:
          NORMAL   → 1.0  (full size)
          HIGH_VOL → 0.6  (40% reduction)
          EXTREME  → 0.0  (no new entries)
        """
        regime = self.get_regime()
        return {"NORMAL": 1.0, "HIGH_VOL": 0.6, "EXTREME": 0.0}[regime]

    # ------------------------------------------------------------------
    def get_regime_report(self) -> dict:
        """
        Snapshot of current volatility regime.

        Returns dict with keys:
          regime, vol, vol_pct, multiplier, message
        """
        vol    = self.get_realized_vol()
        regime = self.get_regime()
        mult   = self.get_position_size_multiplier()

        if regime == "EXTREME":
            msg = f"EXTREME VOL: Tum yeni islemler durduruldu (gunluk vol: {vol*100:.2f}%)"
        elif regime == "HIGH_VOL":
            msg = f"HIGH_VOL: Pozisyon boyutlari %40 kucultuldu (gunluk vol: {vol*100:.2f}%)"
        else:
            msg = f"NORMAL: Normal islem (gunluk vol: {vol*100:.2f}%)"

        return {
            "regime":      regime,
            "vol":         round(vol, 6),
            "vol_pct":     round(vol * 100, 3),
            "multiplier":  mult,
            "message":     msg,
        }

    # ------------------------------------------------------------------
    def get_vol_history(self, days: int = 5) -> list[dict]:
        """
        Compute rolling realized vol for each of the last `days` trading days.
        Each entry: {date, vol_pct, regime}
        """
        try:
            returns = self._fetch_proxy_returns(VIX_WINDOW + days + 5)
            if len(returns) < VIX_WINDOW + 1:
                return []

            results = []
            # Walk back through the most recent `days` return dates
            for i in range(days, 0, -1):
                idx  = len(returns) - i
                if idx < VIX_WINDOW:
                    continue
                window_returns = returns.iloc[idx - VIX_WINDOW: idx]
                vol  = float(window_returns.std())
                date = returns.index[idx - 1]

                if vol >= EXTREME_VOL_THRESHOLD:
                    regime = "EXTREME"
                elif vol >= HIGH_VOL_THRESHOLD:
                    regime = "HIGH_VOL"
                else:
                    regime = "NORMAL"

                results.append({
                    "date":    str(date)[:10],
                    "vol_pct": round(vol * 100, 3),
                    "regime":  regime,
                })
            return results
        except Exception:
            return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(vr: VolatilityRegime):
    report = vr.get_regime_report()
    using  = BIST100_SYMBOL if vr._proxy_available() else ("+".join(PROXY_SYMBOLS) + " (proxy)")

    regime   = report["regime"]
    vol_pct  = report["vol_pct"]
    mult     = report["multiplier"]

    regime_icons = {"NORMAL": "✅", "HIGH_VOL": "⚠️ ", "EXTREME": "🚨"}
    icon         = regime_icons.get(regime, "")

    print("=" * 60)
    print("VOLATILITY REGIME REPORT")
    print(f"Tarih  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Kaynak : {using}")
    print("=" * 60)
    print(f"\nREJIM  : {icon} {regime}")
    print(f"Vol    : {vol_pct:.3f}%  (gunluk realized vol, {VIX_WINDOW}-gun pencere)")
    print(f"Esik   : HIGH_VOL >= {HIGH_VOL_THRESHOLD*100:.1f}%  |  "
          f"EXTREME >= {EXTREME_VOL_THRESHOLD*100:.1f}%")
    print(f"Carp   : x{mult:.1f}  (pozisyon boyutu carpani)")
    print(f"\nMesaj  : {report['message']}")

    print(f"\nSON 5 GUN VOL GECMISI:")
    history = vr.get_vol_history(5)
    if history:
        print(f"  {'Tarih':<12}  {'Vol %':>6}  {'Rejim'}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*10}")
        for h in history:
            bar = "#" * min(int(h["vol_pct"] / 0.5), 20)
            print(f"  {h['date']:<12}  {h['vol_pct']:>5.3f}%  {h['regime']:<10}  {bar}")
    else:
        print("  (Yeterli veri yok)")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility Regime Filter")
    parser.add_argument("--report", action="store_true",
                        help="Print current regime and last 5 days vol history")
    parser.add_argument("--history", type=int, metavar="N", default=0,
                        help="Print vol history for last N days")
    args = parser.parse_args()

    vr = VolatilityRegime()

    if args.history:
        _print_report.__wrapped__ = True
        report = vr.get_regime_report()
        print(f"Rejim: {report['regime']}  Vol: {report['vol_pct']:.3f}%  Carp: x{report['multiplier']}")
        print(f"\nSon {args.history} gun vol gecmisi:")
        history = vr.get_vol_history(args.history)
        if history:
            print(f"  {'Tarih':<12}  {'Vol %':>6}  {'Rejim'}")
            for h in history:
                print(f"  {h['date']:<12}  {h['vol_pct']:>5.3f}%  {h['regime']}")
        else:
            print("  (Yeterli veri yok)")
    elif args.report:
        _print_report(vr)
    else:
        _print_report(vr)
