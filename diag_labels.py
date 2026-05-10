"""
diag_labels.py — Label distribution diagnostic.
READ-ONLY. Does not train, does not save anything.

Prints current label config, OLD positive_rate (triple_barrier v1),
and NEW positive_rate (max_high_rise 8h/3%). Checks whether new rate
falls in target range 0.25 – 0.40.

Usage:
    python3.12 diag_labels.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from crypto_ml import (
    _fetch_ohlcv_all,
    _build_maxhigh_outcomes,
    _build_tradeable_outcomes,
    LABEL_CONFIG,
)
from crypto_stream import CryptoStream

SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT",
    "AVAX/USDT","SUI/USDT","DOT/USDT","ONDO/USDT","FET/USDT",
]

OLD_CONFIG = {
    "name": "tradeable_triple_barrier_v1",
    "mode": "triple_barrier",
    "horizon_bars": 6,
    "fallback_mode": "net_return",
    "fallback_min_net_return": 0.003,
    "use_atr_targets": True,
    "target_atr_mult": 1.25,
    "stop_atr_mult": 0.90,
    "min_target_gain": 0.006,
    "min_stop_loss": 0.005,
    "intrabar_conflict_policy": "stop_first",
}

TARGET_LOW  = 0.20
TARGET_HIGH = 0.40


def label_stats(y: pd.Series, name: str) -> dict:
    y_valid = y.dropna()
    n_total = len(y_valid)
    n_pos   = int(y_valid.sum())
    n_neg   = n_total - n_pos
    pos_rate = n_pos / n_total if n_total > 0 else float("nan")
    return {"name": name, "n_total": n_total, "n_pos": n_pos, "n_neg": n_neg, "pos_rate": pos_rate}


def print_stats(s: dict) -> None:
    flag = ""
    if s["pos_rate"] > TARGET_HIGH:
        flag = "  ← TOO HIGH (overfitting risk)"
    elif s["pos_rate"] < TARGET_LOW:
        flag = "  ← TOO LOW (rare signal)"
    else:
        flag = "  ← OK"
    print(
        f"  {s['name']:<40}  pos={s['n_pos']:>5}/{s['n_total']:<6}  "
        f"rate={s['pos_rate']:.1%}{flag}"
    )


def main():
    print("\n=== LABEL DISTRIBUTION DIAGNOSTIC ===")
    print(f"  ACTIVE config: {LABEL_CONFIG['name']}")
    print(f"  mode={LABEL_CONFIG['mode']}  horizon={LABEL_CONFIG['horizon_bars']}h  "
          f"min_gain={LABEL_CONFIG.get('min_gain', '?')}")
    print(f"  Target pos_rate: {TARGET_LOW:.0%} – {TARGET_HIGH:.0%}")
    print()

    print("Connecting to Binance …")
    stream = CryptoStream()
    raw = _fetch_ohlcv_all(stream)

    old_all, new_3pct, new_4pct, new_25pct = [], [], [], []

    for sym in SYMBOLS:
        df = raw.get(sym)
        if df is None or len(df) < 30:
            continue
        # OLD
        old_out = _build_tradeable_outcomes(df, OLD_CONFIG)
        old_all.append(old_out["label"])
        # NEW 3%
        new_out_3 = _build_maxhigh_outcomes(df, {**LABEL_CONFIG, "min_gain": 0.03})
        new_3pct.append(new_out_3["label"])
        # NEW 4% (fallback if pos_rate > 0.40)
        new_out_4 = _build_maxhigh_outcomes(df, {**LABEL_CONFIG, "min_gain": 0.04})
        new_4pct.append(new_out_4["label"])
        # NEW 2.5% (fallback if pos_rate < 0.20)
        new_out_25 = _build_maxhigh_outcomes(df, {**LABEL_CONFIG, "min_gain": 0.025})
        new_25pct.append(new_out_25["label"])

    old_combined   = pd.concat(old_all,   ignore_index=True)
    new_3_combined = pd.concat(new_3pct,  ignore_index=True)
    new_4_combined = pd.concat(new_4pct,  ignore_index=True)
    new_25_combined = pd.concat(new_25pct, ignore_index=True)

    print("── Per-symbol breakdown (new 3% config) ──────────────────────────────────")
    for sym in SYMBOLS:
        df = raw.get(sym)
        if df is None or len(df) < 30:
            continue
        out = _build_maxhigh_outcomes(df, {**LABEL_CONFIG, "min_gain": 0.03})
        y   = out["label"].dropna()
        pos_rate = y.mean() if len(y) > 0 else float("nan")
        print(f"  {sym:<12}  n={len(y):>4}  pos_rate={pos_rate:.1%}")

    print()
    print("── Combined across all symbols ───────────────────────────────────────────")
    print_stats(label_stats(old_combined,    "OLD: triple_barrier 6h  (ATR, stop_first)"))
    print_stats(label_stats(new_3_combined,  "NEW: max_high_rise  8h  3% threshold"))
    print_stats(label_stats(new_4_combined,  "ALT: max_high_rise  8h  4% threshold"))
    print_stats(label_stats(new_25_combined, "ALT: max_high_rise  8h  2.5% threshold"))

    pos_rate_3 = new_3_combined.dropna().mean()
    print()
    print("── Recommendation ────────────────────────────────────────────────────────")
    if pos_rate_3 > TARGET_HIGH:
        print(f"  pos_rate {pos_rate_3:.1%} > {TARGET_HIGH:.0%} → raise threshold to 4%")
        print(f"  ACTION: change LABEL_CONFIG['min_gain'] from 0.03 to 0.04")
    elif pos_rate_3 < TARGET_LOW:
        print(f"  pos_rate {pos_rate_3:.1%} < {TARGET_LOW:.0%} → lower threshold to 2.5%")
        print(f"  ACTION: change LABEL_CONFIG['min_gain'] from 0.03 to 0.025")
    else:
        print(f"  pos_rate {pos_rate_3:.1%} is in target range [{TARGET_LOW:.0%}, {TARGET_HIGH:.0%}] — 3% threshold is GOOD")
        print(f"  LABEL_CONFIG is correct. Ready to retrain when approved.")

    print()
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()
