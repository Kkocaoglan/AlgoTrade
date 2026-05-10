"""
model_analysis.py — Deep dive into the BIST ML ensemble
---------------------------------------------------------
Tasks:
  1. Feature importance across XGB / LGB / CatBoost
  2. Threshold calibration curve
  3. Label sensitivity (±threshold, ±horizon)
  4. Regime-specific precision
  5. Error analysis (false positives / false negatives)

Output: results/model_analysis_report.txt  +  results/model_analysis_plots.png
"""

import pickle, sqlite3, sys, math, warnings, io
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_score, recall_score, f1_score

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH     = "trade_data.db"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_PATH = RESULTS_DIR / "model_analysis_report.txt"
PLOT_PATH   = RESULTS_DIR / "model_analysis_plots.png"

# ── config ────────────────────────────────────────────────────────────────────
CURRENT_THRESHOLD = 0.65
LABEL_THRESHOLD   = 0.02    # +2.0% in 3 days
LABEL_HORIZON     = 3       # days
PROXY_SYMS        = ["GARAN", "THYAO", "EREGL"]
VOL_WINDOW        = 20
HIGH_VOL_THR      = 0.025
EXTREME_VOL_THR   = 0.04

# ── output buffer (writes to both stdout and report file) ─────────────────────
_lines = []

def pr(*args, **kw):
    s = " ".join(str(a) for a in args)
    print(s, **{k: v for k, v in kw.items() if k != "end"})
    _lines.append(s)

def section(title: str):
    bar = "=" * 65
    pr(f"\n{bar}")
    pr(f"  {title}")
    pr(bar)


# =============================================================================
# Data loading
# =============================================================================

def load_models():
    pr("Loading models ...", end=" ")
    with open("models/xgb_model.pkl", "rb") as f: xgb = pickle.load(f)
    with open("models/lgb_model.pkl", "rb") as f: lgb = pickle.load(f)
    with open("models/cat_model.pkl", "rb") as f: cat = pickle.load(f)
    with open("models/model_bundle.pkl","rb") as f: bundle = pickle.load(f)
    pr("OK")
    return xgb, lgb, cat, bundle


def load_wf_predictions() -> pd.DataFrame:
    df = pd.read_csv("results/wf_predictions.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["actual_label", "prob_buy"]).reset_index(drop=True)
    df["actual_label"] = df["actual_label"].astype(int)
    return df


def load_ohlcv(start: str = "2022-01-01") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT symbol, date, close FROM ohlcv WHERE date >= ? ORDER BY symbol, date",
        conn, params=(start,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_feature_matrix(bundle) -> pd.DataFrame:
    """
    Reconstruct the feature matrix used during training.
    Imports make_features() from ml_train to stay DRY.
    Falls back to a minimal reconstruction if import fails.
    """
    try:
        sys.path.insert(0, ".")
        import ml_train as mlt
        pr("Building feature matrix via ml_train.make_features ...", end=" ")
        conn = sqlite3.connect(DB_PATH)
        all_syms = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol", conn
        )["symbol"].tolist()
        frames = []
        for sym in all_syms:
            df_raw = pd.read_sql_query(
                "SELECT * FROM ohlcv WHERE symbol=? ORDER BY date",
                conn, params=(sym,),
            )
            df_ind = pd.read_sql_query(
                "SELECT * FROM indicators WHERE symbol=? ORDER BY date",
                conn, params=(sym,),
            )
            if df_raw.empty or df_ind.empty:
                continue
            try:
                df_raw["date"] = pd.to_datetime(df_raw["date"])
                df_ind["date"] = pd.to_datetime(df_ind["date"])
                merged = df_raw.merge(df_ind, on=["symbol","date"], how="inner",
                                      suffixes=("","_ind"))
                merged.set_index("date", inplace=True)
                feat_df = mlt.make_features(merged, sym=sym)
                feat_df["symbol"] = sym
                feat_df["date"] = feat_df.index
                frames.append(feat_df)
            except Exception as e:
                pass
        conn.close()
        if frames:
            full = pd.concat(frames, ignore_index=True)
            pr(f"OK ({len(full)} rows, {len(all_syms)} symbols)")
            return full
    except Exception as e:
        pr(f"make_features import failed ({e}), using minimal reconstruction")

    # Minimal fallback
    return pd.DataFrame()


# =============================================================================
# TASK 1 — Feature Importance
# =============================================================================

def task1_feature_importance(xgb, lgb, cat, bundle) -> pd.DataFrame:
    section("TASK 1 — FEATURE IMPORTANCE DEEP DIVE")
    features = bundle["features"]

    # ── XGB: gain importance ────────────────────────────────────────────────
    xgb_gain = xgb.get_booster().get_score(importance_type="gain")
    xgb_imp  = np.array([float(xgb_gain.get(f, 0.0)) for f in features])
    xgb_imp  = xgb_imp / xgb_imp.sum() * 100 if xgb_imp.sum() > 0 else xgb_imp

    # ── LGB: split importance ────────────────────────────────────────────────
    lgb_imp_raw = lgb.feature_importances_
    lgb_imp = lgb_imp_raw / lgb_imp_raw.sum() * 100 if lgb_imp_raw.sum() > 0 else lgb_imp_raw

    # ── CatBoost: PredictionValuesChange ────────────────────────────────────
    cat_imp_raw = cat.get_feature_importance()
    cat_imp = cat_imp_raw / cat_imp_raw.sum() * 100 if cat_imp_raw.sum() > 0 else cat_imp_raw

    imp_df = pd.DataFrame({
        "feature": features,
        "xgb_gain%": xgb_imp,
        "lgb_split%": lgb_imp,
        "cat_pvc%": cat_imp,
    })
    imp_df["mean%"] = imp_df[["xgb_gain%","lgb_split%","cat_pvc%"]].mean(axis=1)
    imp_df["rank_xgb"] = imp_df["xgb_gain%"].rank(ascending=False).astype(int)
    imp_df["rank_lgb"] = imp_df["lgb_split%"].rank(ascending=False).astype(int)
    imp_df["rank_cat"] = imp_df["cat_pvc%"].rank(ascending=False).astype(int)
    imp_df["rank_max_delta"] = (
        imp_df[["rank_xgb","rank_lgb","rank_cat"]].max(axis=1) -
        imp_df[["rank_xgb","rank_lgb","rank_cat"]].min(axis=1)
    )
    imp_df = imp_df.sort_values("mean%", ascending=False).reset_index(drop=True)

    pr("\n── Top 15 features (mean importance across all 3 models) ──")
    pr(f"{'#':<3} {'Feature':<25} {'XGB%':>7} {'LGB%':>7} {'CAT%':>7} {'Mean%':>7} {'RankDelta':>10}")
    pr("-" * 72)
    for i, row in imp_df.head(15).iterrows():
        pr(f"{i+1:<3} {row['feature']:<25} "
           f"{row['xgb_gain%']:>6.2f}  {row['lgb_split%']:>6.2f}  "
           f"{row['cat_pvc%']:>6.2f}  {row['mean%']:>6.2f}  "
           f"{int(row['rank_max_delta']):>9}")

    # ── Consensus top 10 (all three rank <= 15) ─────────────────────────────
    consensus = imp_df[
        (imp_df["rank_xgb"] <= 15) &
        (imp_df["rank_lgb"] <= 15) &
        (imp_df["rank_cat"] <= 15)
    ].head(10)
    pr(f"\n── Consensus top features (all 3 models agree, rank ≤15) ──")
    for _, row in consensus.iterrows():
        pr(f"  {row['feature']:<28} XGB={row['rank_xgb']:>2} LGB={row['rank_lgb']:>2} CAT={row['rank_cat']:>2}")

    # ── Near-zero importance (removal candidates) ────────────────────────────
    near_zero = imp_df[imp_df["mean%"] < 0.20].sort_values("mean%")
    pr(f"\n── Near-zero importance (mean% < 0.20) — removal candidates ──")
    for _, row in near_zero.iterrows():
        pr(f"  {row['feature']:<28} mean={row['mean%']:.3f}%  "
           f"XGB={row['xgb_gain%']:.3f}% LGB={row['lgb_split%']:.3f}% CAT={row['cat_pvc%']:.3f}%")
    if near_zero.empty:
        pr("  None found (all features contribute > 0.20%)")

    # ── Correlation check on feature importances ─────────────────────────────
    pr(f"\n── Model agreement (rank correlation) ──")
    from scipy.stats import spearmanr
    rc_xl, _  = spearmanr(imp_df["rank_xgb"], imp_df["rank_lgb"])
    rc_xc, _  = spearmanr(imp_df["rank_xgb"], imp_df["rank_cat"])
    rc_lc, _  = spearmanr(imp_df["rank_lgb"], imp_df["rank_cat"])
    pr(f"  Spearman rank corr XGB↔LGB : {rc_xl:.3f}")
    pr(f"  Spearman rank corr XGB↔CAT : {rc_xc:.3f}")
    pr(f"  Spearman rank corr LGB↔CAT : {rc_lc:.3f}")

    # ── Feature-to-feature correlation (from DB) ─────────────────────────────
    pr(f"\n── Highly correlated feature pairs (>0.85) from training data ──")
    try:
        conn = sqlite3.connect(DB_PATH)
        raw = pd.read_sql_query(
            "SELECT * FROM indicators ORDER BY symbol, date", conn
        )
        conn.close()
        available_feat_cols = [f for f in features if f in raw.columns]
        if len(available_feat_cols) >= 2:
            corr_matrix = raw[available_feat_cols].dropna().corr().abs()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    c = corr_matrix.iloc[i, j]
                    if c > 0.85:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], c))
            high_corr.sort(key=lambda x: -x[2])
            for f1, f2, c in high_corr[:15]:
                pr(f"  {f1:<25} ↔ {f2:<25}  corr={c:.3f}")
            if not high_corr:
                pr("  No pairs with correlation > 0.85 found in available indicator columns")
        else:
            pr("  Insufficient overlapping columns for correlation check")
    except Exception as e:
        pr(f"  Correlation check failed: {e}")

    return imp_df


# =============================================================================
# TASK 2 — Threshold Calibration
# =============================================================================

def task2_threshold_calibration(wf: pd.DataFrame) -> None:
    section("TASK 2 — THRESHOLD CALIBRATION ANALYSIS")

    thresholds  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    y_true      = wf["actual_label"].values
    y_prob      = wf["prob_buy"].values
    total_dates = len(wf)
    base_rate   = y_true.mean()

    pr(f"\nDataset: {total_dates} WF predictions | Base rate: {base_rate:.3f}")
    pr(f"\n{'Threshold':>10} {'Signals':>8} {'Signals%':>9} {'Precision':>10} "
       f"{'Recall':>8} {'F1':>7} {'Lift':>6}")
    pr("-" * 65)

    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        n_sig  = y_pred.sum()
        sig_pct= n_sig / total_dates * 100
        if n_sig > 0:
            prec   = precision_score(y_true, y_pred, zero_division=0)
            rec    = recall_score(y_true, y_pred, zero_division=0)
            f1     = f1_score(y_true, y_pred, zero_division=0)
            lift   = prec / base_rate if base_rate > 0 else 0
        else:
            prec = rec = f1 = lift = 0.0
        rows.append({"thr": thr, "n": n_sig, "pct": sig_pct,
                     "prec": prec, "rec": rec, "f1": f1, "lift": lift})
        marker = " ← current" if thr == CURRENT_THRESHOLD else ""
        pr(f"{thr:>10.2f} {n_sig:>8} {sig_pct:>8.1f}% {prec:>10.3f} "
           f"{rec:>8.3f} {f1:>7.3f} {lift:>6.2f}x{marker}")

    # Optimal threshold by F1
    best = max(rows, key=lambda x: x["f1"])
    pr(f"\n  Optimal by F1 : threshold={best['thr']:.2f}  "
       f"(F1={best['f1']:.3f}, Precision={best['prec']:.3f}, Signals/day={best['pct']:.1f}%)")

    # Optimal threshold by Precision ≥ 0.75
    prec_thresh = [r for r in rows if r["prec"] >= 0.75]
    if prec_thresh:
        best_prec = min(prec_thresh, key=lambda x: x["thr"])
        pr(f"  Prec ≥ 0.75  : threshold={best_prec['thr']:.2f}  "
           f"({best_prec['n']} signals, {best_prec['pct']:.1f}%)")

    # Verdict on current threshold
    curr = next(r for r in rows if r["thr"] == CURRENT_THRESHOLD)
    pr(f"\n  Current thr {CURRENT_THRESHOLD:.2f}: precision={curr['prec']:.3f}  "
       f"recall={curr['rec']:.3f}  F1={curr['f1']:.3f}  "
       f"signals={curr['n']} ({curr['pct']:.1f}%)")
    if best["thr"] != CURRENT_THRESHOLD:
        delta_f1 = best["f1"] - curr["f1"]
        pr(f"  Recommendation: raise threshold to {best['thr']:.2f} "
           f"(+{delta_f1:.3f} F1, -{curr['pct'] - best['pct']:.1f}% fewer signals)")
    else:
        pr("  Current threshold is optimal by F1.")


# =============================================================================
# TASK 3 — Label Sensitivity Analysis
# =============================================================================

def task3_label_sensitivity(wf: pd.DataFrame, ohlcv: pd.DataFrame) -> None:
    section("TASK 3 — LABEL SENSITIVITY ANALYSIS")

    # Build future-return lookup: symbol, date → list of next N closes
    pr("Building future-return lookup from ohlcv ...", end=" ")
    pivot = ohlcv.pivot(index="date", columns="symbol", values="close").sort_index()
    pr(f"OK ({len(pivot)} dates)")

    configs = [
        ("Current: +2.0% / 3d", 0.020, 3),
        ("Alt A:   +1.5% / 3d", 0.015, 3),
        ("Alt B:   +2.5% / 3d", 0.025, 3),
        ("Alt C:   +2.0% / 5d", 0.020, 5),
    ]

    pr(f"\n{'Config':<25} {'Base rate':>10} {'Prec@0.65':>10} "
       f"{'Prec@0.70':>10} {'Prec@0.75':>10} {'Signals@0.65':>13}")
    pr("-" * 80)

    for label, ret_thr, horizon in configs:
        # Recompute actual label for each WF row
        new_labels = []
        for _, row in wf.iterrows():
            sym  = row["symbol"]
            dt   = row["date"]
            if sym not in pivot.columns:
                new_labels.append(np.nan)
                continue
            try:
                idx     = pivot.index.get_loc(dt)
                future  = pivot.index[idx + 1: idx + horizon + 1]
                if len(future) == 0:
                    new_labels.append(np.nan)
                    continue
                cur_close = pivot.loc[dt, sym]
                fut_max   = pivot.loc[future, sym].max()
                if pd.isna(cur_close) or pd.isna(fut_max) or cur_close == 0:
                    new_labels.append(np.nan)
                    continue
                new_labels.append(1.0 if (fut_max - cur_close) / cur_close >= ret_thr else 0.0)
            except Exception:
                new_labels.append(np.nan)

        wf_tmp = wf.copy()
        wf_tmp["new_label"] = new_labels
        wf_tmp = wf_tmp.dropna(subset=["new_label"])
        y_true = wf_tmp["new_label"].values
        y_prob = wf_tmp["prob_buy"].values

        base = y_true.mean()
        results = []
        for thr in [0.65, 0.70, 0.75]:
            y_pred = (y_prob >= thr).astype(int)
            prec   = precision_score(y_true, y_pred, zero_division=0)
            n_sig  = y_pred.sum()
            results.append((prec, n_sig))

        pr(f"{label:<25} {base:>10.3f} "
           f"{results[0][0]:>10.3f} {results[1][0]:>10.3f} {results[2][0]:>10.3f} "
           f"{results[0][1]:>13}")

    pr("\n  Note: Using EXISTING model predictions (prob_buy) relabelled under each config.")
    pr("  This measures label sensitivity WITHOUT retraining — shows how precision")
    pr("  changes if the target definition shifts, holding model weights constant.")
    pr("  Full retrain would require ~6-fold walk-forward per config (~30 min each).")


# =============================================================================
# TASK 4 — Regime-Specific Analysis
# =============================================================================

def task4_regime_analysis(wf: pd.DataFrame, ohlcv: pd.DataFrame) -> None:
    section("TASK 4 — REGIME-SPECIFIC PRECISION ANALYSIS")

    pr("Computing vol regime for each date ...", end=" ")

    # Build proxy returns: GARAN + THYAO + EREGL equal-weight
    proxy = ohlcv[ohlcv["symbol"].isin(PROXY_SYMS)].copy()
    proxy_pivot = proxy.pivot(index="date", columns="symbol", values="close")
    proxy_ret   = proxy_pivot.pct_change()
    proxy_daily = proxy_ret.mean(axis=1)

    # Rolling 20-day std (non-annualised) → regime
    rolling_vol = proxy_daily.rolling(VOL_WINDOW).std()
    def _regime(v):
        if pd.isna(v):   return "UNKNOWN"
        if v >= EXTREME_VOL_THR: return "EXTREME"
        if v >= HIGH_VOL_THR:    return "HIGH_VOL"
        return "NORMAL"
    regime_map = rolling_vol.apply(_regime)
    pr(f"OK ({len(regime_map)} dates)")

    # Merge regime into WF predictions
    wf2 = wf.copy()
    wf2["regime"] = wf2["date"].map(regime_map).fillna("UNKNOWN")

    pr(f"\n{'Regime':<12} {'N rows':>8} {'Base rate':>10} {'Prec@0.65':>10} "
       f"{'Prec@0.70':>10} {'Signals':>8} {'Rec@0.65':>9}")
    pr("-" * 70)

    regime_order = ["NORMAL", "HIGH_VOL", "EXTREME", "UNKNOWN"]
    regime_results = {}
    for reg in regime_order:
        sub = wf2[wf2["regime"] == reg]
        if len(sub) < 10:
            continue
        y_true = sub["actual_label"].values
        y_prob = sub["prob_buy"].values
        base   = y_true.mean()
        p65 = precision_score(y_true, (y_prob>=0.65).astype(int), zero_division=0)
        p70 = precision_score(y_true, (y_prob>=0.70).astype(int), zero_division=0)
        r65 = recall_score(y_true, (y_prob>=0.65).astype(int), zero_division=0)
        n65 = (y_prob >= 0.65).sum()
        regime_results[reg] = {"prec65": p65, "prec70": p70, "base": base, "n": len(sub)}
        pr(f"{reg:<12} {len(sub):>8} {base:>10.3f} {p65:>10.3f} {p70:>10.3f} "
           f"{n65:>8} {r65:>9.3f}")

    pr("\n  Regime date distribution:")
    for reg in regime_order:
        sub = wf2[wf2["regime"] == reg]
        if len(sub) == 0: continue
        pct = len(sub) / len(wf2) * 100
        pr(f"    {reg:<12} {len(sub):>5} rows ({pct:.1f}%)")

    # Recommendation
    pr("\n  Regime-specific threshold recommendations:")
    for reg, res in regime_results.items():
        prec65 = res["prec65"]
        if prec65 >= 0.80:
            pr(f"    {reg:<12} → threshold=0.65 OK (prec={prec65:.3f} ≥ 0.80)")
        elif prec65 >= 0.70:
            pr(f"    {reg:<12} → consider raising to 0.70 (prec65={prec65:.3f})")
        else:
            pr(f"    {reg:<12} → CAUTION: prec65={prec65:.3f} < 0.70, raise to 0.75+")


# =============================================================================
# TASK 5 — Error Analysis
# =============================================================================

def task5_error_analysis(wf: pd.DataFrame, feat_df: pd.DataFrame) -> None:
    section("TASK 5 — ERROR ANALYSIS (FALSE POSITIVES / FALSE NEGATIVES)")

    y_true = wf["actual_label"].values
    y_prob = wf["prob_buy"].values
    y_pred = (y_prob >= CURRENT_THRESHOLD).astype(int)

    FP_mask = (y_pred == 1) & (y_true == 0)   # predicted BUY, was wrong
    FN_mask = (y_pred == 0) & (y_true == 1)   # missed good trade
    TP_mask = (y_pred == 1) & (y_true == 1)   # correct BUY
    TN_mask = (y_pred == 0) & (y_true == 0)   # correct skip

    pr(f"\nAt threshold={CURRENT_THRESHOLD}:")
    pr(f"  True Positives (TP) : {TP_mask.sum():>5}")
    pr(f"  False Positives (FP): {FP_mask.sum():>5}  ← model said BUY, stock didn't move +2%")
    pr(f"  False Negatives (FN): {FN_mask.sum():>5}  ← stock moved +2%, model didn't signal")
    pr(f"  True Negatives (TN) : {TN_mask.sum():>5}")

    # Probability distribution of errors
    fp_probs = y_prob[FP_mask]
    fn_probs = y_prob[FN_mask]
    pr(f"\n  FP prob distribution: mean={fp_probs.mean():.3f} "
       f"min={fp_probs.min():.3f} max={fp_probs.max():.3f} "
       f"pct>{CURRENT_THRESHOLD:.2f}={fp_probs.mean():.1%}")
    pr(f"  FN prob distribution: mean={fn_probs.mean():.3f} "
       f"min={fn_probs.min():.3f} max={fn_probs.max():.3f} "
       f"(all < {CURRENT_THRESHOLD:.2f})")

    # FP prob bucket analysis
    pr(f"\n  False Positive breakdown by prob bucket:")
    pr(f"  {'Bucket':<15} {'FP count':>9} {'TP count':>9} {'Precision':>10}")
    buckets = [(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,1.01)]
    for lo, hi in buckets:
        mask_b = (y_prob >= lo) & (y_prob < hi)
        fp_b   = ((y_pred == 1) & (y_true == 0) & mask_b).sum()
        tp_b   = ((y_pred == 1) & (y_true == 1) & mask_b).sum()
        tot_b  = fp_b + tp_b
        prec_b = tp_b / tot_b if tot_b > 0 else 0.0
        pr(f"  {lo:.2f}-{hi:.2f}       {fp_b:>9}  {tp_b:>9}  {prec_b:>10.3f}")

    # Symbol error rates
    pr(f"\n  FP rate by symbol (top 10 worst):")
    sym_stats = {}
    for sym in wf["symbol"].unique():
        sub = wf[wf["symbol"] == sym]
        yt  = sub["actual_label"].values
        yp  = (sub["prob_buy"].values >= CURRENT_THRESHOLD).astype(int)
        n_pred = yp.sum()
        if n_pred < 3:
            continue
        fp_s = ((yp == 1) & (yt == 0)).sum()
        tp_s = ((yp == 1) & (yt == 1)).sum()
        prec_s = tp_s / n_pred if n_pred > 0 else 0
        sym_stats[sym] = {"fp": fp_s, "tp": tp_s, "n": n_pred, "prec": prec_s}

    sorted_sym = sorted(sym_stats.items(), key=lambda x: x[1]["prec"])
    pr(f"  {'Symbol':<10} {'Signals':>8} {'TP':>6} {'FP':>6} {'Precision':>10}")
    for sym, st in sorted_sym[:10]:
        pr(f"  {sym:<10} {st['n']:>8} {st['tp']:>6} {st['fp']:>6} {st['prec']:>10.3f}")

    pr(f"\n  Best precision symbols (top 5):")
    for sym, st in sorted(sorted_sym, key=lambda x: -x[1]["prec"])[:5]:
        pr(f"  {sym:<10} prec={st['prec']:.3f}  n_signals={st['n']}")

    # Feature-level error analysis (if feature matrix available)
    if not feat_df.empty:
        pr(f"\n── Feature values on False Positives vs True Positives ──")
        common_dates_syms = wf[FP_mask][["date","symbol"]].copy()
        fp_rows = feat_df.merge(common_dates_syms, on=["date","symbol"], how="inner")
        tp_df   = wf[TP_mask][["date","symbol"]].copy()
        tp_rows = feat_df.merge(tp_df, on=["date","symbol"], how="inner")

        top_feats = ["ret_20d","trend_str_20","above_upper","price_ema200",
                     "bb_pos","atr_ratio","rsi","vol_ratio","mtf_trend","ret_5d"]
        avail = [f for f in top_feats if f in feat_df.columns]

        if avail and not fp_rows.empty and not tp_rows.empty:
            pr(f"  {'Feature':<25} {'FP mean':>9} {'TP mean':>9} {'Diff':>8}")
            pr("  " + "-" * 55)
            for col in avail:
                fp_mean = fp_rows[col].mean()
                tp_mean = tp_rows[col].mean()
                diff    = fp_mean - tp_mean
                pr(f"  {col:<25} {fp_mean:>9.4f} {tp_mean:>9.4f} {diff:>+8.4f}")
        else:
            pr("  (Feature matrix not available or no overlap — run with full DB)")

    # False negative time-of-day / temporal analysis
    pr(f"\n  False Negatives: which symbols and when were good trades missed?")
    fn_wf = wf[FN_mask].copy()
    fn_by_sym = fn_wf.groupby("symbol").size().sort_values(ascending=False)
    pr(f"  FN by symbol (top 10):")
    for sym, cnt in fn_by_sym.head(10).items():
        sub = fn_wf[fn_wf["symbol"] == sym]
        avg_prob = sub["prob_buy"].mean()
        pr(f"    {sym:<10} {cnt:>4} missed  avg_prob={avg_prob:.3f}")

    fn_probs_hist = pd.cut(fn_probs, bins=[0,.3,.4,.5,.55,.60,.65]).value_counts().sort_index()
    pr(f"\n  FN prob distribution (all < {CURRENT_THRESHOLD:.2f}):")
    for bucket, cnt in fn_probs_hist.items():
        pr(f"    {str(bucket):<20} {cnt:>5} ({cnt/len(fn_probs)*100:.1f}%)")

    pr(f"\n  Key insight: FNs with prob in [0.55, 0.65) are 'near-miss' signals.")
    near_miss = ((fn_probs >= 0.55) & (fn_probs < 0.65)).sum()
    pr(f"  Near-miss FNs (prob 0.55-0.65): {near_miss} "
       f"({near_miss/len(fn_probs)*100:.1f}% of all FNs)")
    pr(f"  Lowering threshold to 0.60 would capture these at cost of lower precision.")


# =============================================================================
# Plotting
# =============================================================================

def make_plots(imp_df: pd.DataFrame, wf: pd.DataFrame,
               ohlcv: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Feature importance comparison (top 15) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    top15 = imp_df.head(15)
    x     = np.arange(len(top15))
    w     = 0.26
    ax1.bar(x - w, top15["xgb_gain%"],  width=w, label="XGB gain",   color="#1565C0", alpha=0.85)
    ax1.bar(x,     top15["lgb_split%"], width=w, label="LGB split",  color="#2E7D32", alpha=0.85)
    ax1.bar(x + w, top15["cat_pvc%"],   width=w, label="CAT PVC",    color="#C62828", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top15["feature"], rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Importance %")
    ax1.set_title("Feature Importance — Top 15 (XGB gain | LGB split | CatBoost PVC)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # ── Plot 2: Threshold calibration curve ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    y_true = wf["actual_label"].values
    y_prob = wf["prob_buy"].values
    precs, recs, sigs = [], [], []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        sigs.append(y_pred.sum() / len(y_true) * 100)
    ax2.plot(thresholds, precs, "b-o", label="Precision", lw=2)
    ax2.plot(thresholds, recs,  "g-s", label="Recall",    lw=2)
    ax2.axvline(CURRENT_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"Current ({CURRENT_THRESHOLD})")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision / Recall vs Threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.bar(thresholds, sigs, width=0.03, alpha=0.25, color="orange", label="Signal %")
    ax2b.set_ylabel("Signal frequency %", color="orange")

    # ── Plot 3: Prob distribution (TP vs FP vs FN) ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    tp_p = y_prob[(y_prob >= CURRENT_THRESHOLD) & (y_true == 1)]
    fp_p = y_prob[(y_prob >= CURRENT_THRESHOLD) & (y_true == 0)]
    fn_p = y_prob[(y_prob <  CURRENT_THRESHOLD) & (y_true == 1)]
    bins = np.linspace(0, 1, 25)
    ax3.hist(tp_p, bins=bins, alpha=0.7, color="#2E7D32", label=f"TP n={len(tp_p)}")
    ax3.hist(fp_p, bins=bins, alpha=0.7, color="#C62828", label=f"FP n={len(fp_p)}")
    ax3.hist(fn_p, bins=bins, alpha=0.5, color="#F57F17", label=f"FN n={len(fn_p)}")
    ax3.axvline(CURRENT_THRESHOLD, color="black", linestyle="--", lw=1.5)
    ax3.set_xlabel("prob_buy")
    ax3.set_ylabel("Count")
    ax3.set_title("Probability Distribution — TP / FP / FN")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ── Plot 4: Symbol precision heatmap ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    sym_prec = []
    for sym in sorted(wf["symbol"].unique()):
        sub    = wf[wf["symbol"] == sym]
        yt, yp = sub["actual_label"].values, sub["prob_buy"].values
        ypred  = (yp >= CURRENT_THRESHOLD).astype(int)
        n_pred = ypred.sum()
        if n_pred >= 3:
            p = precision_score(yt, ypred, zero_division=0)
            sym_prec.append((sym, p, n_pred))
    sym_prec.sort(key=lambda x: x[1])
    syms  = [s[0] for s in sym_prec]
    precs = [s[1] for s in sym_prec]
    colors = ["#C62828" if p < 0.5 else "#F57F17" if p < 0.7 else "#2E7D32" for p in precs]
    ax4.barh(syms, precs, color=colors, alpha=0.85)
    ax4.axvline(CURRENT_THRESHOLD, color="black", linestyle="--", lw=1, alpha=0.5)
    ax4.axvline(y_true.mean(), color="grey", linestyle=":", lw=1, alpha=0.7, label="base rate")
    ax4.set_xlabel("Precision @ 0.65")
    ax4.set_title("Per-Symbol Precision (green≥0.70, red<0.50)")
    ax4.legend(fontsize=8)
    ax4.grid(axis="x", alpha=0.3)

    # ── Plot 5: Regime precision bar ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    proxy = ohlcv[ohlcv["symbol"].isin(PROXY_SYMS)].pivot(
        index="date", columns="symbol", values="close"
    ).pct_change().mean(axis=1).rolling(VOL_WINDOW).std()
    def _reg(v):
        if pd.isna(v): return "UNKNOWN"
        if v >= EXTREME_VOL_THR: return "EXTREME"
        if v >= HIGH_VOL_THR: return "HIGH_VOL"
        return "NORMAL"
    regime_map = proxy.apply(_reg)
    wf2 = wf.copy()
    wf2["regime"] = wf2["date"].map(regime_map).fillna("UNKNOWN")
    reg_prec = {}
    reg_n    = {}
    for reg in ["NORMAL","HIGH_VOL","EXTREME","UNKNOWN"]:
        sub = wf2[wf2["regime"] == reg]
        if len(sub) < 10: continue
        yt, yp = sub["actual_label"].values, sub["prob_buy"].values
        ypred  = (yp >= CURRENT_THRESHOLD).astype(int)
        if ypred.sum() < 3: continue
        reg_prec[reg] = precision_score(yt, ypred, zero_division=0)
        reg_n[reg]    = ypred.sum()
    bar_colors = {"NORMAL":"#2E7D32","HIGH_VOL":"#F57F17","EXTREME":"#C62828","UNKNOWN":"grey"}
    regs  = list(reg_prec.keys())
    pvals = [reg_prec[r] for r in regs]
    bclrs = [bar_colors.get(r,"grey") for r in regs]
    bars  = ax5.bar(regs, pvals, color=bclrs, alpha=0.85)
    for bar, n in zip(bars, [reg_n[r] for r in regs]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"n={n}", ha="center", va="bottom", fontsize=9)
    ax5.axhline(y_true.mean(), color="grey", linestyle=":", lw=1.5, label="base rate")
    ax5.set_ylabel("Precision @ 0.65")
    ax5.set_title("Precision by Volatility Regime")
    ax5.set_ylim(0, 1.0)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    plt.suptitle("BIST ML Ensemble — Model Analysis Report", fontsize=14, fontweight="bold")
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pr(f"\nPlots saved → {PLOT_PATH}")


# =============================================================================
# main
# =============================================================================

def main():
    pr(f"BIST ML Ensemble Analysis  —  {date.today().isoformat()}")
    pr(f"Model: XGBoost + LightGBM + CatBoost  |  BUY_THRESHOLD={CURRENT_THRESHOLD}")

    xgb, lgb, cat, bundle = load_models()
    wf      = load_wf_predictions()
    ohlcv   = load_ohlcv()
    feat_df = load_feature_matrix(bundle)

    imp_df = task1_feature_importance(xgb, lgb, cat, bundle)
    task2_threshold_calibration(wf)
    task3_label_sensitivity(wf, ohlcv)
    task4_regime_analysis(wf, ohlcv)
    task5_error_analysis(wf, feat_df)

    # Save report
    section("SUMMARY")
    y_true = wf["actual_label"].values
    y_prob = wf["prob_buy"].values
    y_pred = (y_prob >= CURRENT_THRESHOLD).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    pr(f"  WF dataset: {len(wf)} rows | {len(wf['symbol'].unique())} symbols | "
       f"6 folds | base rate={y_true.mean():.3f}")
    pr(f"  @ threshold {CURRENT_THRESHOLD}: precision={prec:.3f}  recall={rec:.3f}  "
       f"signals={y_pred.sum()} ({y_pred.sum()/len(wf)*100:.1f}%)")
    pr(f"  See {PLOT_PATH} for visual analysis")

    try:
        make_plots(imp_df, wf, ohlcv)
    except Exception as e:
        pr(f"Plot generation failed (non-fatal): {e}")

    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")
    pr(f"\nReport written → {REPORT_PATH}")


if __name__ == "__main__":
    main()
