"""
meta_labeler.py -- Secondary meta-labeling classifier for BUY signal filtering.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

from regime_hmm import compute_historical_regime_features

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "trade_data.db"
RESULTS = BASE_DIR / "results"
MODELS = BASE_DIR / "models"
WF_PATH = RESULTS / "wf_predictions.csv"
META_MODEL_PATH = MODELS / "meta_labeler.pkl"

CANDIDATE_MIN_PROB = 0.50
DEFAULT_PRIMARY_THRESHOLD = 0.65
META_THRESHOLD = 0.55
PURGE_DAYS = 3
META_MIN_CLOSED_TRADES = 30

META_FEATURES = [
    "primary_prob",
    "primary_prob_xgb",
    "primary_prob_lgb",
    "primary_prob_cat",
    "model_disagreement",
    "hmm_prob_bull",
    "hmm_prob_bear",
    "hmm_days_in_state",
    "amihud_20",
    "mfi_14",
    "volume_surge",
    "vol_pressure_z",
    "rolling_win_rate_10",
    "rolling_pnl_5",
    "sentiment_score",
    "day_of_week",
    "days_since_last_trade",
]


def load_primary_threshold() -> float:
    opt_path = RESULTS / "optimal_threshold.json"
    if not opt_path.exists():
        return DEFAULT_PRIMARY_THRESHOLD
    try:
        with open(opt_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return float(payload.get("buy_threshold", DEFAULT_PRIMARY_THRESHOLD))
    except Exception:
        return DEFAULT_PRIMARY_THRESHOLD


def load_wf_predictions(path: Path = WF_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"WF predictions not found: {path}")
    wf = pd.read_csv(path)
    if wf.empty:
        raise ValueError("wf_predictions.csv is empty")

    if "prob_buy" in wf.columns and "primary_prob" not in wf.columns:
        wf = wf.rename(columns={"prob_buy": "primary_prob"})
    if "xgb_prob" in wf.columns:
        wf["primary_prob_xgb"] = pd.to_numeric(wf["xgb_prob"], errors="coerce")
    if "lgb_prob" in wf.columns:
        wf["primary_prob_lgb"] = pd.to_numeric(wf["lgb_prob"], errors="coerce")
    if "cat_prob" in wf.columns:
        wf["primary_prob_cat"] = pd.to_numeric(wf["cat_prob"], errors="coerce")

    wf["date"] = pd.to_datetime(wf["date"]).dt.normalize()
    wf["primary_prob"] = pd.to_numeric(wf["primary_prob"], errors="coerce").clip(0.0, 1.0)

    missing_base_cols = [
        col for col in ("primary_prob_xgb", "primary_prob_lgb", "primary_prob_cat")
        if col not in wf.columns
    ]
    if missing_base_cols:
        print(
            "  [WARN] wf_predictions.csv eski formatta; "
            "base model probability kolonlari eksik. "
            "Bu turda primary_prob fallback kullanilacak."
        )
    for col in missing_base_cols:
        wf[col] = wf["primary_prob"]
    if "model_disagreement" not in wf.columns:
        wf["model_disagreement"] = wf[
            ["primary_prob_xgb", "primary_prob_lgb", "primary_prob_cat"]
        ].std(axis=1, ddof=0)
    else:
        wf["model_disagreement"] = pd.to_numeric(wf["model_disagreement"], errors="coerce")

    return wf


def build_meta_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        verbose=-1,
    )


def load_symbol_history(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    query = """
        SELECT o.date, o.open, o.high, o.low, o.close, o.volume,
               COALESCE(i.sentiment_score, 0.0) AS sentiment_score
        FROM ohlcv o
        JOIN indicators i ON o.symbol=i.symbol AND o.date=i.date
        WHERE o.symbol=?
        ORDER BY o.date
    """
    try:
        df = pd.read_sql(query, conn, params=(symbol,))
    except Exception:
        fallback_query = """
            SELECT o.date, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv o
            WHERE o.symbol=?
            ORDER BY o.date
        """
        df = pd.read_sql(fallback_query, conn, params=(symbol,))
        df["sentiment_score"] = 0.0

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).clip(lower=0.0)

    ret_1d = close.pct_change()
    dollar_volume = (close * volume).replace(0.0, np.nan)
    amihud_20 = (ret_1d.abs() / (dollar_volume + 1e-12)).rolling(20, min_periods=5).mean() * 1e6

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    tp_delta = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_delta > 0, 0.0)
    neg_flow = raw_money_flow.where(tp_delta < 0, 0.0).abs()
    pos_sum = pos_flow.rolling(14, min_periods=5).sum()
    neg_sum = neg_flow.rolling(14, min_periods=5).sum()
    money_ratio = pos_sum / (neg_sum + 1e-9)
    mfi_14 = 100.0 - (100.0 / (1.0 + money_ratio))

    volume_mean_20 = volume.rolling(20, min_periods=5).mean()
    volume_surge = volume / (volume_mean_20 + 1e-9)

    signed_pressure = np.sign(close.diff().fillna(0.0)) * volume_surge.fillna(1.0)
    pressure_mean = signed_pressure.rolling(20, min_periods=5).mean()
    pressure_std = signed_pressure.rolling(20, min_periods=5).std()
    vol_pressure_z = (signed_pressure - pressure_mean) / (pressure_std + 1e-9)

    return pd.DataFrame(
        {
            "amihud_20": amihud_20.replace([np.inf, -np.inf], np.nan),
            "mfi_14": mfi_14.replace([np.inf, -np.inf], np.nan),
            "volume_surge": volume_surge.replace([np.inf, -np.inf], np.nan),
            "vol_pressure_z": vol_pressure_z.replace([np.inf, -np.inf], np.nan),
        },
        index=df.index,
    )


def load_trade_history(conn: sqlite3.Connection) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    trades = pd.read_sql(
        """
        SELECT symbol, exit_date, pnl
        FROM paper_trades
        WHERE exit_date IS NOT NULL
        ORDER BY exit_date
        """,
        conn,
    )
    if trades.empty:
        empty = pd.DataFrame(columns=["exit_date", "pnl"])
        return {}, empty

    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.normalize()
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    by_symbol = {
        sym: grp[["exit_date", "pnl"]].reset_index(drop=True)
        for sym, grp in trades.groupby("symbol")
    }
    portfolio = trades[["exit_date", "pnl"]].reset_index(drop=True)
    return by_symbol, portfolio


def summarize_trade_context(trades_df: pd.DataFrame, as_of_date: pd.Timestamp) -> tuple[float, float, float]:
    if trades_df.empty:
        return 0.5, 0.0, 999.0

    dt_arr = trades_df["exit_date"].to_numpy(dtype="datetime64[ns]")
    pnl_arr = trades_df["pnl"].to_numpy(dtype=float)
    cutoff = np.datetime64(pd.Timestamp(as_of_date).normalize())
    idx = int(np.searchsorted(dt_arr, cutoff, side="left"))
    if idx <= 0:
        return 0.5, 0.0, 999.0

    hist_dates = dt_arr[:idx]
    hist_pnl = pnl_arr[:idx]
    last10 = hist_pnl[-10:]
    last5 = hist_pnl[-5:]
    win_rate = float((last10 > 0).mean()) if len(last10) else 0.5
    pnl_5 = float(last5.sum()) if len(last5) else 0.0
    days_since = float((pd.Timestamp(as_of_date) - pd.Timestamp(hist_dates[-1])).days)
    return win_rate, pnl_5, days_since


def build_trade_context_features(
    meta_df: pd.DataFrame,
    trades_by_symbol: dict[str, pd.DataFrame],
    portfolio_trades: pd.DataFrame,
) -> pd.DataFrame:
    win_rates: list[float] = []
    pnl_5_vals: list[float] = []
    days_since_vals: list[float] = []

    for row in meta_df.itertuples(index=False):
        sym_trades = trades_by_symbol.get(row.symbol, pd.DataFrame(columns=["exit_date", "pnl"]))
        sym_wr, sym_pnl5, days_since = summarize_trade_context(sym_trades, row.date)
        if len(sym_trades[sym_trades["exit_date"] < row.date]) < 3:
            pf_wr, pf_pnl5, _pf_days = summarize_trade_context(portfolio_trades, row.date)
            sym_wr = pf_wr
            sym_pnl5 = pf_pnl5
        win_rates.append(sym_wr)
        pnl_5_vals.append(sym_pnl5)
        days_since_vals.append(days_since)

    meta_df["rolling_win_rate_10"] = win_rates
    meta_df["rolling_pnl_5"] = pnl_5_vals
    meta_df["days_since_last_trade"] = days_since_vals
    return meta_df


def build_meta_training_frame(
    db_path: Path = DB_PATH,
    wf_path: Path = WF_PATH,
    candidate_min_prob: float = CANDIDATE_MIN_PROB,
) -> pd.DataFrame:
    wf = load_wf_predictions(wf_path)
    wf = wf.loc[wf["primary_prob"] >= candidate_min_prob].copy()
    if wf.empty:
        raise ValueError("No candidate BUY rows found at primary_prob >= 0.50")

    conn = sqlite3.connect(db_path)
    try:
        regime_df = compute_historical_regime_features(db_path, n_seeds=1)[
            ["hmm_prob_bull", "hmm_prob_bear", "hmm_days_in_state"]
        ].copy()
        regime_df["date"] = regime_df.index.normalize()
        trades_by_symbol, portfolio_trades = load_trade_history(conn)

        frames = []
        for symbol, grp in wf.groupby("symbol"):
            hist = load_symbol_history(conn, symbol)
            if hist.empty:
                continue
            micro = compute_microstructure_features(hist)
            hist = hist.join(micro)
            hist["actual_return_3d"] = hist["close"].shift(-3) / hist["close"] - 1.0
            hist = hist.join(regime_df.set_index("date"), how="left")
            hist = hist.reset_index().rename(columns={"index": "date"})

            use_cols = [
                "date",
                "sentiment_score",
                "amihud_20",
                "mfi_14",
                "volume_surge",
                "vol_pressure_z",
                "hmm_prob_bull",
                "hmm_prob_bear",
                "hmm_days_in_state",
                "actual_return_3d",
            ]
            merged = grp.merge(hist[use_cols], on="date", how="left")
            frames.append(merged)
    finally:
        conn.close()

    if not frames:
        raise ValueError("Could not join any symbol history to wf_predictions")

    meta_df = pd.concat(frames, ignore_index=True)
    meta_df["day_of_week"] = meta_df["date"].dt.dayofweek.astype(float)
    meta_df = build_trade_context_features(meta_df, trades_by_symbol, portfolio_trades)
    meta_df["y_meta"] = (pd.to_numeric(meta_df["actual_return_3d"], errors="coerce") >= 0.0).astype(int)

    defaults = {
        "hmm_prob_bull": 1.0 / 3.0,
        "hmm_prob_bear": 1.0 / 3.0,
        "hmm_days_in_state": 0.0,
        "amihud_20": 0.0,
        "mfi_14": 50.0,
        "volume_surge": 1.0,
        "vol_pressure_z": 0.0,
        "rolling_win_rate_10": 0.5,
        "rolling_pnl_5": 0.0,
        "sentiment_score": 0.0,
        "day_of_week": 0.0,
        "days_since_last_trade": 999.0,
        "model_disagreement": 0.0,
    }
    for col, default in defaults.items():
        meta_df[col] = pd.to_numeric(meta_df.get(col, default), errors="coerce").fillna(default)

    for col in ("primary_prob_xgb", "primary_prob_lgb", "primary_prob_cat"):
        meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce").fillna(meta_df["primary_prob"])
    meta_df["primary_prob"] = pd.to_numeric(meta_df["primary_prob"], errors="coerce").fillna(0.5)
    meta_df["model_disagreement"] = pd.to_numeric(meta_df["model_disagreement"], errors="coerce").fillna(
        meta_df[["primary_prob_xgb", "primary_prob_lgb", "primary_prob_cat"]].std(axis=1, ddof=0)
    )

    meta_df = meta_df.dropna(subset=["date", "symbol", "actual_return_3d"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    return meta_df


def purged_walk_forward_eval(
    meta_df: pd.DataFrame,
    primary_threshold: float | None = None,
    meta_threshold: float = META_THRESHOLD,
    n_splits: int = 6,
) -> dict:
    primary_threshold = float(primary_threshold or load_primary_threshold())
    unique_dates = np.array(sorted(meta_df["date"].unique()))
    if len(unique_dates) < (n_splits + 1):
        raise ValueError(f"Not enough unique dates for {n_splits} folds")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rows: list[dict] = []
    oof_meta_prob = pd.Series(np.nan, index=meta_df.index, dtype=float)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates[train_idx]
        test_dates = unique_dates[test_idx]
        if len(train_dates) <= PURGE_DAYS or len(test_dates) == 0:
            continue

        purged_train_dates = train_dates[:-PURGE_DAYS]
        train_mask = meta_df["date"].isin(purged_train_dates)
        test_mask = meta_df["date"].isin(test_dates)
        train_df = meta_df.loc[train_mask].copy()
        test_df = meta_df.loc[test_mask].copy()
        if len(train_df) < 50 or len(test_df) < 10 or train_df["y_meta"].nunique() < 2:
            continue

        model = build_meta_model()
        model.fit(train_df[META_FEATURES], train_df["y_meta"])
        meta_prob = model.predict_proba(test_df[META_FEATURES])[:, 1]
        oof_meta_prob.loc[test_df.index] = meta_prob

        primary_pass = test_df["primary_prob"] >= primary_threshold
        meta_pass = primary_pass & (meta_prob >= meta_threshold)
        primary_hits = test_df.loc[primary_pass, "y_meta"]
        meta_hits = test_df.loc[meta_pass, "y_meta"]

        fold_rows.append(
            {
                "fold": fold_id,
                "primary_signals": int(primary_pass.sum()),
                "meta_signals": int(meta_pass.sum()),
                "blocked": int(primary_pass.sum() - meta_pass.sum()),
                "primary_precision": float(primary_hits.mean()) if len(primary_hits) else np.nan,
                "meta_precision": float(meta_hits.mean()) if len(meta_hits) else np.nan,
            }
        )

    if not fold_rows:
        raise ValueError("Meta-labeler evaluation produced no valid folds")

    eval_df = pd.DataFrame(fold_rows)
    eval_mask = (meta_df["primary_prob"] >= primary_threshold) & oof_meta_prob.notna()
    pass_mask = eval_mask & (oof_meta_prob >= meta_threshold)

    summary = {
        "folds": eval_df,
        "primary_threshold": primary_threshold,
        "meta_threshold": meta_threshold,
        "signals_before": int(eval_mask.sum()),
        "signals_after": int(pass_mask.sum()),
        "blocked": int(eval_mask.sum() - pass_mask.sum()),
        "primary_precision": float(meta_df.loc[eval_mask, "y_meta"].mean()) if eval_mask.any() else np.nan,
        "meta_precision": float(meta_df.loc[pass_mask, "y_meta"].mean()) if pass_mask.any() else np.nan,
        "oof_meta_prob": oof_meta_prob,
    }
    return summary


def fit_final_meta_model(meta_df: pd.DataFrame) -> dict:
    model = build_meta_model()
    model.fit(meta_df[META_FEATURES], meta_df["y_meta"])
    artifact = {
        "model": model,
        "features": META_FEATURES,
        "candidate_min_prob": CANDIDATE_MIN_PROB,
        "meta_threshold": META_THRESHOLD,
        "primary_threshold": load_primary_threshold(),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    META_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, META_MODEL_PATH)
    return artifact


def load_meta_model(path: Path = META_MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    return joblib.load(path)


def predict_meta_probability(artifact: dict, features: pd.DataFrame) -> np.ndarray:
    model = artifact["model"]
    feats = artifact.get("features", META_FEATURES)
    X = features.copy()
    for col in feats:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feats].fillna(0.0)
    return model.predict_proba(X)[:, 1]


def build_live_meta_feature_vector(
    conn: sqlite3.Connection,
    signal: dict,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    signal_date = pd.Timestamp(as_of_date or signal.get("data_date") or date.today().isoformat()).normalize()
    symbol = signal["symbol"]
    trades_by_symbol, portfolio_trades = load_trade_history(conn)
    sym_trades = trades_by_symbol.get(symbol, pd.DataFrame(columns=["exit_date", "pnl"]))
    win_rate, pnl_5, days_since = summarize_trade_context(sym_trades, signal_date)
    if len(sym_trades[sym_trades["exit_date"] < signal_date]) < 3:
        pf_wr, pf_pnl5, _pf_days = summarize_trade_context(portfolio_trades, signal_date)
        win_rate = pf_wr
        pnl_5 = pf_pnl5

    row = {
        "primary_prob": float(signal.get("prob_buy", 0.5)),
        "primary_prob_xgb": float(signal.get("primary_prob_xgb", signal.get("prob_buy", 0.5))),
        "primary_prob_lgb": float(signal.get("primary_prob_lgb", signal.get("prob_buy", 0.5))),
        "primary_prob_cat": float(signal.get("primary_prob_cat", signal.get("prob_buy", 0.5))),
        "model_disagreement": float(signal.get("model_disagreement", 0.0)),
        "hmm_prob_bull": float(signal.get("hmm_prob_bull", 1.0 / 3.0)),
        "hmm_prob_bear": float(signal.get("hmm_prob_bear", 1.0 / 3.0)),
        "hmm_days_in_state": float(signal.get("hmm_days_in_state", 0.0)),
        "amihud_20": float(signal.get("amihud_20", 0.0)),
        "mfi_14": float(signal.get("mfi_14", 50.0)),
        "volume_surge": float(signal.get("volume_surge", 1.0)),
        "vol_pressure_z": float(signal.get("vol_pressure_z", 0.0)),
        "rolling_win_rate_10": float(win_rate),
        "rolling_pnl_5": float(pnl_5),
        "sentiment_score": float(signal.get("sentiment_score", 0.0)),
        "day_of_week": float(signal_date.dayofweek),
        "days_since_last_trade": float(days_since),
    }
    return pd.DataFrame([row], columns=META_FEATURES)


def print_eval_summary(summary: dict) -> None:
    folds = summary["folds"]
    print("META LABELER EVAL")
    print(
        f"Primary threshold={summary['primary_threshold']:.2f}  "
        f"Meta threshold={summary['meta_threshold']:.2f}"
    )
    for row in folds.itertuples(index=False):
        p_prec = 0.0 if np.isnan(row.primary_precision) else row.primary_precision * 100.0
        m_prec = 0.0 if np.isnan(row.meta_precision) else row.meta_precision * 100.0
        print(
            f"  Fold {row.fold}: before={row.primary_signals} after={row.meta_signals} "
            f"blocked={row.blocked} primary_prec={p_prec:.1f}% meta_prec={m_prec:.1f}%"
        )
    p_total = 0.0 if np.isnan(summary["primary_precision"]) else summary["primary_precision"] * 100.0
    m_total = 0.0 if np.isnan(summary["meta_precision"]) else summary["meta_precision"] * 100.0
    print(
        f"\nSignals before/after : {summary['signals_before']} -> {summary['signals_after']} "
        f"(blocked {summary['blocked']})"
    )
    print(f"Precision before     : {p_total:.1f}%")
    print(f"Precision after      : {m_total:.1f}%")
    print(f"Delta                : {m_total - p_total:+.1f} pp")


def run_train() -> int:
    meta_df = build_meta_training_frame()
    summary = purged_walk_forward_eval(meta_df)
    artifact = fit_final_meta_model(meta_df)
    print_eval_summary(summary)
    print(f"\nSaved meta-labeler: {META_MODEL_PATH.name}")
    print(f"Training rows      : {len(meta_df)}")
    print(f"Trained at         : {artifact['trained_at']}")
    return 0


def run_eval() -> int:
    meta_df = build_meta_training_frame()
    summary = purged_walk_forward_eval(meta_df)
    print_eval_summary(summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.train:
        return run_train()
    if args.eval:
        return run_eval()

    parser.error("Use --train or --eval")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
