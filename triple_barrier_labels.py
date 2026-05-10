"""
Triple-barrier labeling helpers for BIST ML training.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "trade_data.db"


def compute_triple_barrier_label_details(
    symbol_df,
    atr_col="atr14",
    pt_sl_ratio=2.0,
    sl_ratio=1.0,
    vertical_bars=5,
):
    """
    Return per-row triple-barrier outcomes and resolution bars.

    Output columns:
      - label: {1: upper barrier, -1: lower barrier, 0: vertical barrier}
      - resolution_bars: bars until the label resolved (1..vertical_bars)
    """
    if symbol_df.empty:
        return pd.DataFrame(columns=["label", "resolution_bars"])

    df = symbol_df.sort_index().copy()
    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=float)
    dates = df.index

    labels = {}
    resolution_bars = {}

    last_valid = max(len(closes) - vertical_bars, 0)
    for i in range(last_valid):
        entry = closes[i]
        atr = atrs[i]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr) or atr < 0:
            continue

        pt = entry * (1.0 + pt_sl_ratio * atr / entry)
        sl = entry * (1.0 - sl_ratio * atr / entry)

        label = 0
        resolved = vertical_bars
        for j in range(1, vertical_bars + 1):
            price = closes[i + j]
            if not np.isfinite(price):
                continue
            if price >= pt:
                label = 1
                resolved = j
                break
            if price <= sl:
                label = -1
                resolved = j
                break

        labels[dates[i]] = label
        resolution_bars[dates[i]] = resolved

    return pd.DataFrame(
        {
            "label": pd.Series(labels, dtype=float),
            "resolution_bars": pd.Series(resolution_bars, dtype=float),
        }
    )


def compute_triple_barrier_labels(
    symbol_df,
    atr_col="atr14",
    pt_sl_ratio=2.0,
    sl_ratio=1.0,
    vertical_bars=5,
):
    """
    symbol_df: DataFrame with columns [date, close, atr14] indexed by date
    pt_sl_ratio: profit-take = pt_sl_ratio * ATR / close
    sl_ratio: stop-loss = sl_ratio * ATR / close
    vertical_bars: max holding period in trading days
    Returns: Series with labels {1: BUY, -1: SELL, 0: NEUTRAL} indexed by date
    """
    details = compute_triple_barrier_label_details(
        symbol_df,
        atr_col=atr_col,
        pt_sl_ratio=pt_sl_ratio,
        sl_ratio=sl_ratio,
        vertical_bars=vertical_bars,
    )
    return details["label"]


def compute_sample_uniqueness_weights(label_details, vertical_bars=5):
    """
    Simple overlap proxy:
      w_i = 1 / min(vertical_bars, bars_until_label_resolved_i)
      then normalized by the mean.
    """
    if label_details.empty:
        return pd.Series(dtype=float)

    resolved = (
        pd.to_numeric(label_details["resolution_bars"], errors="coerce")
        .clip(lower=1, upper=vertical_bars)
        .astype(float)
    )
    weights = 1.0 / resolved
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight <= 0:
        return weights
    return weights / mean_weight


def _load_symbol_frame(symbol):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            """
            SELECT o.date, o.close, i.atr14
            FROM ohlcv o
            JOIN indicators i ON o.symbol = i.symbol AND o.date = i.date
            WHERE o.symbol = ?
            ORDER BY o.date
            """,
            conn,
            params=(symbol,),
        )
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["atr14"] = pd.to_numeric(df["atr14"], errors="coerce")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test triple-barrier labels on one symbol.")
    parser.add_argument("symbol", nargs="?", default="GARAN")
    args = parser.parse_args()

    symbol_df = _load_symbol_frame(args.symbol)
    details = compute_triple_barrier_label_details(symbol_df)
    weights = compute_sample_uniqueness_weights(details)

    if details.empty:
        print(f"{args.symbol}: no labelable rows")
    else:
        buy_rate = (details["label"] == 1).mean() * 100
        neutral_rate = (details["label"] == 0).mean() * 100
        sell_rate = (details["label"] == -1).mean() * 100
        print(f"{args.symbol}: {len(details)} rows labeled")
        print(
            f"  BUY={buy_rate:.2f}%  NEUTRAL={neutral_rate:.2f}%  SELL={sell_rate:.2f}%"
        )
        print(
            f"  Avg uniqueness weight={weights.mean():.4f}  "
            f"min={weights.min():.4f}  max={weights.max():.4f}"
        )
        print("  Last 5 labels:")
        print(details.tail(5).to_string())
