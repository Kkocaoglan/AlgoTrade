"""
optimize_threshold.py — Cost-sensitive BUY threshold optimizer
---------------------------------------------------------------
Loads walk-forward validation predictions produced by ml_train.py
(results/wf_predictions.csv) and finds the probability threshold that
maximises expected P&L per trade given realistic BIST transaction costs.

Assumptions
  avg_win  = +2.0%   (label = price rises >= +2% in 3 days)
  avg_loss =  1.5%   (stop-loss cap — paid when prediction is wrong)
  cost     = 25 bps  (0.0025, round-trip: commission + slippage, Tier 1)

  E[PnL/trade] = precision*(avg_win - cost) - (1-precision)*(avg_loss + cost)
  Total E[PnL] = E[PnL/trade] * n_trades

  t* = argmax Total_E[PnL]  subject to  n_trades >= MIN_TRADES

Output
  Prints full threshold table + chosen t*.
  Saves results/optimal_threshold.json with {"buy_threshold": t*, ...}.

Usage
  py -3.12 optimize_threshold.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────
RESULTS  = Path(__file__).parent / "results"
WF_CSV   = RESULTS / "wf_predictions.csv"
OPT_JSON = RESULTS / "optimal_threshold.json"

# ── Cost parameters ───────────────────────────────────────────
AVG_WIN    = 0.020    # +2.0% average win (label target)
AVG_LOSS   = 0.015    # 1.5% average loss magnitude (stop-loss cap)
COST       = 0.0025   # 25 bps round-trip (Tier 1 BIST stocks)
MIN_TRADES = 10       # minimum trades required for a threshold to be eligible
FOLD_MIN_TRADES = 5   # per-fold minimum (looser, fewer samples per fold)


# ── Per-threshold metric computation ─────────────────────────
def compute_metrics(df: pd.DataFrame, t: float) -> dict:
    """
    For a given threshold t, compute precision, recall, trade count,
    and cost-sensitive expected P&L metrics.

    NaN actual_label rows: counted as "not a buy" — they reduce precision
    when the model fires a BUY, because the trade happened but the stock
    didn't reach the +2% target.
    """
    predicted_buy  = df["prob_buy"] >= t
    actual_is_buy  = df["actual_label"] == 1      # NaN == 1 evaluates False → safe

    n_trades  = int(predicted_buy.sum())
    tp        = int((predicted_buy & actual_is_buy).sum())
    n_buy_lbl = int(actual_is_buy.sum())

    precision = tp / n_trades    if n_trades  > 0 else 0.0
    recall    = tp / n_buy_lbl   if n_buy_lbl > 0 else 0.0

    # Expected P&L per trade — cost deducted on both win and loss sides
    epnl_per_trade = precision * (AVG_WIN - COST) - (1.0 - precision) * (AVG_LOSS + COST)
    total_epnl     = epnl_per_trade * n_trades

    return {
        "threshold":      round(float(t), 2),
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "n_trades":       n_trades,
        "epnl_per_trade": round(epnl_per_trade, 6),
        "total_epnl":     round(total_epnl, 4),
    }


def find_optimal(df: pd.DataFrame, thresholds, min_trades: int) -> dict | None:
    """Return the row with highest total_epnl among those with n_trades >= min_trades."""
    rows = pd.DataFrame([compute_metrics(df, t) for t in thresholds])
    eligible = rows[rows["n_trades"] >= min_trades]
    if eligible.empty:
        return None
    return eligible.loc[eligible["total_epnl"].idxmax()].to_dict()


# ── Main ─────────────────────────────────────────────────────
def main():
    if not WF_CSV.exists():
        print(f"ERROR: {WF_CSV} not found.")
        print("  Run py -3.12 ml_train.py first to generate walk-forward predictions.")
        return

    df = pd.read_csv(WF_CSV)
    print("=" * 72)
    print("Cost-Sensitive BUY Threshold Optimizer")
    print("=" * 72)
    print(f"  Input   : {WF_CSV.name}  ({len(df):,} rows)")
    print(f"  Symbols : {df['symbol'].nunique()}")
    print(f"  Folds   : {df['fold'].nunique()}")
    n_labeled = int(df["actual_label"].notna().sum())
    label_counts = df["actual_label"].value_counts(dropna=False).to_dict()
    print(f"  Labels  : {label_counts}")
    print(f"  Cost params: avg_win={AVG_WIN*100:.1f}%  avg_loss={AVG_LOSS*100:.1f}%  "
          f"cost={COST*10000:.0f}bps")
    print()

    # ── Full threshold sweep ──────────────────────────────────
    thresholds = np.round(np.arange(0.50, 0.905, 0.01), 2)
    all_rows   = [compute_metrics(df, t) for t in thresholds]
    results    = pd.DataFrame(all_rows)

    # Table header
    col_w  = 10
    hdr = (f"{'Threshold':>{col_w}} | {'Precision':>9} | {'Recall':>7} | "
           f"{'N_trades':>8} | {'E[PnL/trade]':>13} | {'Total E[PnL]':>13}")
    print(hdr)
    print("-" * len(hdr))

    for _, row in results.iterrows():
        eligible_marker = " *" if row["n_trades"] >= MIN_TRADES else "  "
        print(
            f"  {row['threshold']:>6.2f}    | "
            f"{row['precision']:>8.3f}  | "
            f"{row['recall']:>6.3f}  | "
            f"{int(row['n_trades']):>8}  | "
            f"{row['epnl_per_trade']:>13.5f}  | "
            f"{row['total_epnl']:>13.4f}"
            f"{eligible_marker}"
        )

    print(f"\n  (* = eligible: n_trades >= {MIN_TRADES})")

    # ── Global optimum ────────────────────────────────────────
    print()
    eligible = results[results["n_trades"] >= MIN_TRADES]
    if eligible.empty:
        print(f"ERROR: No threshold has >= {MIN_TRADES} trades. Defaulting to 0.65.")
        t_star, prec_star, n_star = 0.65, 0.0, 0
        epnl_star, total_epnl_star = 0.0, 0.0
    else:
        best = eligible.loc[eligible["total_epnl"].idxmax()]
        t_star         = float(best["threshold"])
        prec_star      = float(best["precision"])
        n_star         = int(best["n_trades"])
        epnl_star      = float(best["epnl_per_trade"])
        total_epnl_star = float(best["total_epnl"])

    print("=" * 72)
    print(f"  Optimal threshold  : {t_star:.2f}")
    print(f"  Precision          : {prec_star:.3f}")
    print(f"  N_trades           : {n_star}")
    print(f"  E[PnL/trade]       : {epnl_star:+.5f}  ({epnl_star*100:+.3f}%)")
    print(f"  Total E[PnL]       : {total_epnl_star:+.4f}  ({total_epnl_star*100:+.2f}%)")
    print("=" * 72)

    # ── Fold-by-fold stability analysis ──────────────────────
    fold_stars = []
    if "fold" in df.columns:
        print()
        print("Fold-level optimal thresholds:")
        print(f"  {'Fold':>5} | {'t*':>6} | {'Precision':>9} | {'N_trades':>8} | {'Total E[PnL]':>13}")
        print("  " + "-" * 50)
        for fold_id in sorted(df["fold"].unique()):
            fold_df   = df[df["fold"] == fold_id]
            fold_best = find_optimal(fold_df, thresholds, min_trades=FOLD_MIN_TRADES)
            if fold_best is not None:
                fold_stars.append(fold_best["threshold"])
                print(
                    f"  {int(fold_id):>5} | {fold_best['threshold']:>6.2f} | "
                    f"{fold_best['precision']:>8.3f}  | "
                    f"{int(fold_best['n_trades']):>8}  | "
                    f"{fold_best['total_epnl']:>13.4f}"
                )
            else:
                print(f"  {int(fold_id):>5} | {'N/A':>6} | {'—':>9} | {'—':>8} | {'—':>13}")

    if fold_stars:
        thr_std  = float(np.std(fold_stars))
        thr_mean = float(np.mean(fold_stars))
        print()
        print(f"  Fold t* values : {[round(t, 2) for t in fold_stars]}")
        print(f"  Mean           : {thr_mean:.3f}")
        print(f"  Std            : {thr_std:.3f}")
        if thr_std > 0.05:
            print(f"  WARNING: high threshold instability (std={thr_std:.3f} > 0.05)")
            print("           Consider using the default 0.65 or collecting more data.")
        else:
            print(f"  Stability      : OK (std <= 0.05)")

    # ── Save JSON ─────────────────────────────────────────────
    output = {
        "buy_threshold":          t_star,
        "precision_at_threshold": prec_star,
        "n_trades":               n_star,
        "epnl_per_trade":         round(epnl_star, 6),
        "total_epnl":             round(total_epnl_star, 4),
        "computed_at":            datetime.now().isoformat(timespec="seconds"),
        "avg_win_pct":            AVG_WIN * 100,
        "avg_loss_pct":           AVG_LOSS * 100,
        "cost_bps":               COST * 10000,
        "min_trades":             MIN_TRADES,
        "fold_thresholds":        fold_stars,
        "fold_threshold_std":     round(float(np.std(fold_stars)) if fold_stars else 0.0, 4),
    }
    RESULTS.mkdir(exist_ok=True)
    with open(OPT_JSON, "w") as fh:
        json.dump(output, fh, indent=2)
    print()
    print(f"Saved: {OPT_JSON}")
    print(f"  loop_trader.py will load this threshold at next startup.")
    print(f"  To override, delete the file (falls back to BUY_THRESHOLD=0.65).")


if __name__ == "__main__":
    main()
