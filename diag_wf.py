"""
diag_wf.py — Walk-forward fold size diagnostic.
READ-ONLY. Does not train, does not save anything.

Shows:
  - Combined dataset shape
  - TimeSeriesSplit fold sizes (train / test / date ranges)
  - Confirms min train > 400 and min test > 80

Usage:
    python3.12 diag_wf.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from crypto_ml import (
    _fetch_ohlcv_all,
    _build_features,
    _build_cross_sectional_relative_strength,
    _build_outcomes,
    _fetch_fg_history,
    _symbol_encode,
    FEATURE_NAMES,
    LABEL_CONFIG,
    N_FOLDS,
    WF_GAP_BARS,
    MIN_TRAIN_SIZE,
    MIN_TEST_SIZE,
    WARMUP_BARS,
)
from crypto_stream import CryptoStream

SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT",
    "AVAX/USDT","SUI/USDT","DOT/USDT","ONDO/USDT","FET/USDT",
]


def main():
    print("\n=== WALK-FORWARD FOLD SIZE DIAGNOSTIC ===")
    print(f"  N_FOLDS={N_FOLDS}  WF_GAP_BARS={WF_GAP_BARS}")
    print(f"  MIN_TRAIN={MIN_TRAIN_SIZE}  MIN_TEST={MIN_TEST_SIZE}")

    print("\nConnecting to Binance …")
    stream = CryptoStream()
    raw = _fetch_ohlcv_all(stream)

    print("Fetching Fear & Greed …")
    fg_dict = _fetch_fg_history()

    print("Building cross-sectional RS …")
    rs_map = _build_cross_sectional_relative_strength(raw)

    btc_df = raw.get("BTC/USDT")
    all_parts = []

    for sym in SYMBOLS:
        df1h = raw.get(sym)
        if df1h is None or len(df1h) < 20:
            continue
        df_5m   = raw.get(f"{sym}_5m")
        btc_ref = btc_df if sym != "BTC/USDT" else None
        try:
            X = _build_features(df1h, btc_ref, fg_dict, df_5m,
                                 relative_strength_df=rs_map.get(sym))
            X["symbol_encoded"] = _symbol_encode(sym)
            outcomes = _build_outcomes(df1h, LABEL_CONFIG)
            y = outcomes["label"]

            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].iloc[WARMUP_BARS:]
            y = y.iloc[:min_len].iloc[WARMUP_BARS:]

            valid = y.notna()
            X = X[valid]
            y = y[valid]

            part = X.copy()
            part["__timestamp"] = pd.to_datetime(df1h.loc[X.index, "timestamp"], utc=True)
            part["__symbol"]    = sym
            part["__y"]         = y.astype(float).values
            all_parts.append(part)
        except Exception as e:
            print(f"  ERROR {sym}: {e}")

    if not all_parts:
        print("No data — aborting.")
        return

    combined = pd.concat(all_parts, ignore_index=True)
    combined = combined.sort_values(["__timestamp", "__symbol"]).reset_index(drop=True)

    ts_combined = combined["__timestamp"]
    X_combined  = combined[FEATURE_NAMES].fillna(0)
    y_combined  = combined["__y"]

    valid_mask = y_combined.notna()
    X_valid    = X_combined.loc[valid_mask].reset_index(drop=True)
    y_valid    = y_combined.loc[valid_mask].reset_index(drop=True)
    ts_valid   = ts_combined.loc[valid_mask].reset_index(drop=True)
    sym_valid  = combined["__symbol"].loc[valid_mask].reset_index(drop=True)

    n_syms        = int(sym_valid.nunique())
    effective_gap = WF_GAP_BARS * n_syms

    print(f"\n── Combined dataset ──────────────────────────────────────────────────")
    print(f"  Total rows      : {len(X_valid)}")
    print(f"  Unique symbols  : {n_syms}  → effective row gap = {WF_GAP_BARS}×{n_syms} = {effective_gap}")
    print(f"  Date range      : {ts_valid.iloc[0].strftime('%Y-%m-%d')} → {ts_valid.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Feature count   : {X_valid.shape[1]}")
    print(f"  Positive rate   : {y_valid.mean():.1%}")

    # ── Try n_splits=5 first, fall back to 4 ─────────────────────────────────────
    for n_splits in [N_FOLDS, N_FOLDS - 1]:
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=effective_gap)
        folds = list(splitter.split(X_valid))
        min_train = min(len(tr) for tr, _ in folds)
        min_test  = min(len(te) for _, te in folds)

        print(f"\n── TimeSeriesSplit(n_splits={n_splits}, gap={effective_gap}) ───────────────")
        print(f"  {'Fold':<6} {'Train rows':>12} {'Test rows':>11}  Train date range → Test date range")
        print(f"  {'────':<6} {'──────────':>12} {'─────────':>11}  {'──────────────────────────────────────────────'}")

        ok = True
        for i, (tr_idx, te_idx) in enumerate(folds, start=1):
            tr_start = ts_valid.iloc[tr_idx[0]].strftime("%Y-%m-%d")
            tr_end   = ts_valid.iloc[tr_idx[-1]].strftime("%Y-%m-%d")
            te_start = ts_valid.iloc[te_idx[0]].strftime("%Y-%m-%d")
            te_end   = ts_valid.iloc[te_idx[-1]].strftime("%Y-%m-%d")
            flag = ""
            if len(tr_idx) < MIN_TRAIN_SIZE:
                flag = " ← TRAIN TOO SMALL"
                ok = False
            if len(te_idx) < MIN_TEST_SIZE:
                flag += " ← TEST TOO SMALL"
                ok = False
            print(f"  Fold {i:<2} {len(tr_idx):>12,} {len(te_idx):>11,}  {tr_start}→{tr_end} | {te_start}→{te_end}{flag}")

        print(f"\n  Min train fold  : {min_train:,}  (need > {MIN_TRAIN_SIZE})  {'✓ OK' if min_train > MIN_TRAIN_SIZE else '✗ FAIL'}")
        print(f"  Min test fold   : {min_test:,}  (need > {MIN_TEST_SIZE})  {'✓ OK' if min_test > MIN_TEST_SIZE else '✗ FAIL'}")

        if ok:
            print(f"\n  → n_splits={n_splits} is VALID — all folds meet size requirements.")
            print(f"  → LABEL_CONFIG={LABEL_CONFIG['name']}")
            print(f"  → Ready to retrain when approved.")
            break
        else:
            print(f"\n  → n_splits={n_splits} has undersized folds — trying {n_splits-1} …")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
