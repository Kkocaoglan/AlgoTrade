"""
diag_features.py — Feature shape + NaN rate diagnostic.
READ-ONLY. Does not train, does not save anything.

Usage:
    python3.12 diag_features.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from crypto_ml import (
    _fetch_ohlcv_all,
    _build_features,
    _build_cross_sectional_relative_strength,
    _drop_constant_like_features,
    _fetch_fg_history,
    FEATURE_NAMES,
)
from crypto_stream import CryptoStream

SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT",
    "AVAX/USDT","SUI/USDT","DOT/USDT","ONDO/USDT","FET/USDT",
]

def main():
    print("\n=== FEATURE SHAPE DIAGNOSTIC ===")
    print(f"FEATURE_NAMES count (pre-drop): {len(FEATURE_NAMES)}")

    print("Connecting to Binance …")
    stream = CryptoStream()
    raw = _fetch_ohlcv_all(stream)
    print(f"Fetched {len(raw)} keyed DataFrames")

    print("Fetching Fear & Greed history …")
    fg_dict = _fetch_fg_history()
    print(f"  F&G history: {len(fg_dict)} days")

    print("Building cross-sectional relative strength …")
    rs_map = _build_cross_sectional_relative_strength(raw)

    btc_df = raw.get("BTC/USDT")
    all_parts = []

    for sym in SYMBOLS:
        df1h = raw.get(sym)
        if df1h is None or len(df1h) < 20:
            print(f"  SKIP {sym}: insufficient 1h data")
            continue
        df_5m   = raw.get(f"{sym}_5m")
        btc_ref = btc_df if sym != "BTC/USDT" else None
        try:
            feat_df = _build_features(
                df1h,
                btc_ref,
                fg_dict,
                df_5m,
                relative_strength_df=rs_map.get(sym),
            )
            feat_df["__symbol"] = sym
            all_parts.append(feat_df)
            print(f"  {sym}: {len(feat_df)} feature rows, {feat_df.shape[1]-1} cols")
        except Exception as e:
            import traceback
            print(f"  ERROR {sym}: {e}")
            traceback.print_exc()

    if not all_parts:
        print("No feature data — aborting.")
        return

    combined = pd.concat(all_parts, ignore_index=True)
    print(f"\nCombined rows: {len(combined)}")

    # ── Raw X ────────────────────────────────────────────────────────────────────
    existing_cols = [c for c in FEATURE_NAMES if c in combined.columns]
    missing_cols  = [c for c in FEATURE_NAMES if c not in combined.columns]
    X_raw = combined[existing_cols]

    print(f"\nX.shape BEFORE drop: {X_raw.shape}")
    if missing_cols:
        print(f"  Missing from df: {missing_cols}")

    # ── NaN rates ────────────────────────────────────────────────────────────────
    nan_rates = X_raw.isna().mean().sort_values(ascending=False)
    HIGH_NAN = 0.20
    print(f"\n{'Feature':<30} {'NaN%':>7}  Flag")
    print("-" * 46)
    survivors = []
    dropped_nan = []
    for feat, rate in nan_rates.items():
        flag = "HIGH" if rate > HIGH_NAN else "ok"
        print(f"  {feat:<28} {rate*100:>6.1f}%  {flag}")
        if rate > HIGH_NAN:
            dropped_nan.append(feat)
        else:
            survivors.append(feat)

    print(f"\nSurvive NaN≤{HIGH_NAN*100:.0f}%: {len(survivors)}")
    print(f"Dropped NaN>{HIGH_NAN*100:.0f}%: {len(dropped_nan)}")
    if dropped_nan:
        print(f"  Dropped list: {dropped_nan}")

    # ── _drop_constant_like_features ─────────────────────────────────────────────
    X_surv = X_raw[survivors]
    try:
        # Compute per-feature constant ratios inline
        feature_stats = {
            col: {"constant_ratio": float(X_surv[col].value_counts(dropna=False, normalize=True).iloc[0])}
            for col in survivors
        }
        X_after, dropped_const = _drop_constant_like_features(X_surv, feature_stats)
        print(f"\nX.shape AFTER _drop_constant_like_features: {X_after.shape}")
        if dropped_const:
            print(f"  Also dropped (near-constant): {dropped_const}")
    except Exception as e:
        import traceback
        print(f"\n_drop_constant_like_features error: {e}")
        traceback.print_exc()
        print(f"Expected X.shape ~ ({len(combined)}, {len(survivors)})")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
