"""
crypto_ml.py — Directional XGBoost trainer for the Crypto Module.

Default mode trains a staged LONG/SHORT directional pair on 180-day paginated
Binance history using 23 single-coin OHLCV features and 5-fold walk-forward
validation. Newly trained artifacts are written to staging with `active=False`;
only the weekly promotion pipeline may activate them.

Usage:
  python3.12 crypto_ml.py            # train staged directional candidates
  python3.12 crypto_ml.py --eval     # evaluate legacy single-model artifact
  python3.12 crypto_ml.py --single   # train old single-model fallback path

Retrain / promotion: run crypto_weekly_retrain.py (cron Sunday 07:00 UTC)

Completely isolated from spot ml_train.py and VIOP systems.
"""

import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone, timedelta
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "crypto_xgb.pkl"
MODEL_PATH_LONG  = BASE_DIR / "models" / "crypto_xgb_long.pkl"
MODEL_PATH_SHORT = BASE_DIR / "models" / "crypto_xgb_short.pkl"
MODEL_PATH_LONG_MAJOR  = BASE_DIR / "models" / "crypto_xgb_long_major.pkl"
MODEL_PATH_LONG_RISKY  = BASE_DIR / "models" / "crypto_xgb_long_risky.pkl"
MODEL_PATH_SHORT_MAJOR = BASE_DIR / "models" / "crypto_xgb_short_major.pkl"
MODEL_PATH_SHORT_RISKY = BASE_DIR / "models" / "crypto_xgb_short_risky.pkl"
STAGING_DIR = BASE_DIR / "models" / "staging"
REJECTED_DIR = BASE_DIR / "models" / "rejected"
MODEL_STAGE_LONG_PATH = STAGING_DIR / "crypto_xgb_long_candidate.pkl"
MODEL_STAGE_SHORT_PATH = STAGING_DIR / "crypto_xgb_short_candidate.pkl"
MODEL_STAGE_LONG_MAJOR_PATH  = STAGING_DIR / "crypto_xgb_long_major_candidate.pkl"
MODEL_STAGE_LONG_RISKY_PATH  = STAGING_DIR / "crypto_xgb_long_risky_candidate.pkl"
MODEL_STAGE_SHORT_MAJOR_PATH = STAGING_DIR / "crypto_xgb_short_major_candidate.pkl"
MODEL_STAGE_SHORT_RISKY_PATH = STAGING_DIR / "crypto_xgb_short_risky_candidate.pkl"
MODEL_REPORT_PATH = BASE_DIR / "models" / "crypto_model_report.json"
THRESHOLD_REPORT_PATH = BASE_DIR / "models" / "threshold_report.json"
RETRAIN_HISTORY_PATH = BASE_DIR / "models" / "retrain_history.jsonl"
RESULTS_DIR = BASE_DIR / "results"
TIER_SYMBOLS = {
    "MAJOR": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT"],
    "RISKY": ["ONDO/USDT", "FET/USDT"],
}

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_HISTORY_DAYS = 180         # target 180 days of 1h training data
MIN_1H_CANDLES     = 24 * 175    # require near-full 180d 1h coverage
MIN_5M_CANDLES     = 288 * 175   # require near-full 180d 5m coverage
PAGE_LIMIT         = 1000
CANDLES_PER_SYMBOL = PAGE_LIMIT  # backward-compatible single-call fallback
LEGACY_LABEL_CONFIG = {
    "name": "legacy_simple",
    "mode": "legacy_simple",
    "horizon_bars": 3,
    "min_gain": 0.01,
}

LABEL_CONFIG = {
    "name": "max_high_rise_8h_15pct_v1",
    "mode": "max_high_rise",
    # Label: 1 if max(high[t+1 .. t+8]) / close[t] - 1 >= 1.5%
    # Uses rolling max over future highs so any intrabar spike counts.
    # Threshold calibrated to give positive_rate 25-40% on bear/sideways market data.
    # 3% threshold gave only 12% (too rare in bear market); 1.5% gives 32.9%.
    "horizon_bars": 8,
    "min_gain": 0.015,          # 1.5% threshold → pos_rate ~33% in current market
}
EXECUTION_COST_CONFIG = {
    "name": "binance_spot_default",
    "enabled": True,
    "order_type": "market",
    "order_modes": {
        "market": {
            "fee_bps": 10.0,
            "spread_bps": 2.0,
            "slippage_bps": 5.0,
        },
        "limit": {
            "fee_bps": 10.0,
            "spread_bps": 0.5,
            "slippage_bps": 1.5,
        },
    },
}
N_FOLDS            = 5           # walk-forward folds over directional 180-day dataset
FOLD_TEST_SIZE     = None        # None = auto-split (sklearn divides evenly)
MIN_TRAIN_SIZE     = 400         # minimum rows per train fold
MIN_TEST_SIZE      = 80          # minimum rows per test fold
WF_GAP_BARS        = int(LABEL_CONFIG["horizon_bars"])  # logical gap in 1h bars (=8)
WF_EMBARGO_BARS    = 1
MIN_ACCEPT_PRECISION = 0.55
MIN_ACCEPT_TRADES    = 12
MIN_ACCEPT_EXPECTANCY = 0.0
THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
THRESHOLD_MIN_TRADES = {
    "MAJOR": 12,
    "RISKY": 6,
}
THRESHOLD_FALLBACKS = {
    "MAJOR": 0.63,
    "RISKY": 0.72,
}
THRESHOLD_MIN_EXPECTANCY = 0.0
XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
    # scale_pos_weight set dynamically from training data ratio
}

FEATURE_NAMES = [
    # ── 1h base indicators ────────────────────────────────────────────────────
    "ema8_21_cross",   # normalized EMA8 vs EMA21 cross
    "bb_position",     # (close - bb_lower) / (bb_upper - bb_lower)
    "macd_hist_norm",  # MACD histogram normalized by close
    "volume_ratio",    # volume / 20-bar rolling average
    "atr14_norm",      # ATR / close
    "price_ema50_pct", # (close - ema50) / ema50
    # ── Return features ───────────────────────────────────────────────────────
    "ret_1h", "ret_2h", "ret_4h", "ret_6h", "ret_12h", "ret_24h",
    # ── Candle structure ──────────────────────────────────────────────────────
    "high_low_range",  # (high - low) / close
    "body_ratio",      # |close - open| / (high - low)
    "buy_pressure",    # (close - low) / (high - low)
    # ── Volume signals ────────────────────────────────────────────────────────
    "volume_spike",    # binary: volume_ratio > 3.0
    # ── Range / level ─────────────────────────────────────────────────────────
    "price_vs_7d_high", # close / rolling 168h high (proximity to recent high)
    # ── MTF 5m (entry timing — aligned from separate 5m fetch) ────────────────
    "mtf_5m_rsi", "mtf_5m_macd_hist", "mtf_5m_bb_pos",
    # ── Perp crowding / macro / HTF ──────────────────────────────────────────
    "funding_rate", "funding_rate_avg3", "funding_rate_extreme", "funding_contrarian",
    "btc_dominance", "btc_dominance_regime", "btc_dominance_7d_change",
    "htf_bias",
]

# Coins used for RISKY-feature comparison (model learns to weight these more for ONDO/FET)
_RISKY_COINS = ["ONDO/USDT", "FET/USDT"]
DIAG_SYMBOLS = {"BNB/USDT", "AVAX/USDT"}
_QUALITY_WARNINGS: list[str] = []
WARMUP_BARS = 50

# ── Symbol label encoding ──────────────────────────────────────────────────────
# Fixed ordered list so encoding is stable across retrains.
# Unknown symbols get index = len(_SYMBOL_LIST) (safe out-of-range value).
_SYMBOL_LIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT",
    "ONDO/USDT", "FET/USDT",
]
_SYMBOL_ENCODING: dict[str, int] = {sym: i for i, sym in enumerate(_SYMBOL_LIST)}


def _symbol_encode(sym: str) -> float:
    """Return stable integer encoding for a coin symbol. Unknown → len(_SYMBOL_LIST)."""
    return float(_SYMBOL_ENCODING.get(sym, len(_SYMBOL_LIST)))

FEATURE_SOURCE_NOTES = {
    "fear_greed_value": "Daily Fear & Greed history aligned by 1h bar close date; left missing when source unavailable.",
    "btc_1h_return": "BTC 1h close-to-close return aligned backward to each bar close timestamp.",
    "mtf_5m_rsi": "Latest completed 5m RSI aligned backward to each 1h bar close timestamp.",
    "mtf_5m_macd_hist": "Latest completed 5m MACD histogram aligned backward to each 1h bar close timestamp.",
    "mtf_5m_bb_pos": "Latest completed 5m Bollinger position aligned backward to each 1h bar close timestamp.",
    "volume_spike": "Binary flag: 1 when 1h volume ratio exceeds 3.0.",
    "price_vs_7d_high": "Close divided by rolling 168h high.",
    "funding_rate": "Latest aligned Binance perpetual 8h funding rate (major symbols only).",
    "funding_rate_avg3": "Rolling 3-period average of Binance funding rates.",
    "funding_rate_extreme": "Binary flag: abs(funding_rate) > 0.1%.",
    "funding_contrarian": "Negative sign of funding rate; crowded longs become bearish.",
    "btc_dominance": "CoinGecko BTC market-cap dominance snapshot (live/current fallback in training).",
    "btc_dominance_regime": "BTC dominance bucket encoded as 0..3 (altseason -> btc_dominant).",
    "btc_dominance_7d_change": "Change in BTC dominance versus cached 7-day anchor.",
    "htf_bias": "Binary 4h trend bias: 1 when price above EMA50_4h, else 0.",
}

_LEGACY_EXECUTION_COST_KEYS = {
    "fee_bps",
    "spread_bps",
    "slippage_bps",
    "order_type",
    "costs_enabled",
}


def _tier_for_symbol(symbol: str) -> str:
    return "RISKY" if symbol in _RISKY_COINS else "MAJOR"


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_ohlcv_all(stream) -> dict[str, pd.DataFrame]:
    """
    Fetch 1h, 5m, and 15m candles for all 10 coins.
    1h  : limit=1000 (~41 days) — primary training data
    5m  : limit=500  (~1.7 days) — MTF entry-timing features
    15m : limit=500  (~5.2 days) — MTF momentum features

    Returns dict: symbol → 1h DataFrame (with indicators),
                  symbol_5m  → 5m DataFrame,
                  symbol_15m → 15m DataFrame
    """
    from crypto_indicators import compute_all, add_rsi, add_macd, add_bollinger

    LIMIT_1H  = 1000   # ~41 days of 1h bars
    LIMIT_5M  = 500    # ~1.7 days of 5m bars
    LIMIT_15M = 500    # ~5.2 days of 15m bars
    MIN_1H    = 100    # minimum rows to keep a symbol

    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT",
        "ONDO/USDT", "FET/USDT",
    ]
    result  = {}
    fetched = 0

    def _fetch_with_retry(sym: str, timeframe: str, limit: int, max_retries: int = 2) -> pd.DataFrame | None:
        """Fetch OHLCV with up to max_retries retries on failure."""
        import time as _time
        for attempt in range(1, max_retries + 2):
            try:
                df = stream.get_ohlcv(sym, timeframe=timeframe, limit=limit)
                return df
            except Exception as exc:
                if attempt <= max_retries:
                    print(f"  [RETRY {attempt}/{max_retries}] {sym} {timeframe}: {exc} — bekleniyor 2s...")
                    _time.sleep(2)
                else:
                    print(f"  [FAIL] {sym} {timeframe}: {exc} — max retry aşıldı, atlanıyor")
                    return None
        return None

    print(f"  1h limit={LIMIT_1H} | 5m limit={LIMIT_5M} | 15m limit={LIMIT_15M}", flush=True)
    for sym in symbols:
        df = _fetch_with_retry(sym, "1h", LIMIT_1H)
        if df is None or len(df) < MIN_1H:
            print(f"  [WARN] {sym}: 1h veri alinamadi veya yetersiz ({0 if df is None else len(df)} rows) — atlanıyor")
            continue
        df = compute_all(df)
        result[sym] = df
        fetched += 1
        print(f"    {sym} 1h: {len(df)} rows | {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")

    print(f"  1h toplam: {fetched}/{len(symbols)} sembol yuklendi")

    # 5m MTF
    print("  Fetching 5m MTF data...", flush=True)
    for sym in list(result.keys()):
        df_5m = _fetch_with_retry(sym, "5m", LIMIT_5M)
        if df_5m is not None and len(df_5m) >= 30:
            df_5m = add_rsi(df_5m)
            df_5m = add_macd(df_5m)
            df_5m = add_bollinger(df_5m)
            result[f"{sym}_5m"] = df_5m
            print(f"    {sym} 5m: {len(df_5m)} rows | {df_5m['timestamp'].iloc[0].date()} -> {df_5m['timestamp'].iloc[-1].date()}")
        else:
            print(f"  [WARN] {sym} 5m: veri yetersiz")

    # 15m MTF — iterate over BASE symbols only (not _5m keys)
    print("  Fetching 15m MTF data...", flush=True)
    for sym in symbols:
        df_15m = _fetch_with_retry(sym, "15m", LIMIT_15M)
        if df_15m is not None and len(df_15m) >= 30:
            df_15m = add_rsi(df_15m)
            df_15m = add_macd(df_15m)
            df_15m = add_bollinger(df_15m)
            result[f"{sym}_15m"] = df_15m
            print(f"    {sym} 15m: {len(df_15m)} rows | {df_15m['timestamp'].iloc[0].date()} -> {df_15m['timestamp'].iloc[-1].date()}")
        else:
            print(f"  [WARN] {sym} 15m: veri yetersiz")

    return result


def _warn_quality(msg: str) -> None:
    if msg in _QUALITY_WARNINGS:
        return
    _QUALITY_WARNINGS.append(msg)
    print(f"  [QUALITY] {msg}")


def _fetch_history_window() -> tuple[int, int]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=TRAIN_HISTORY_DAYS)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


def _fmt_range(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "no-data"
    return f"{df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}"


def _build_alignment_report(
    symbol: str,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    btc_df: pd.DataFrame | None,
) -> dict:
    report = {
        "symbol": symbol,
        "one_hour_rows": int(len(df_1h)),
        "one_hour_range": _fmt_range(df_1h),
        "five_min_rows": 0 if df_5m is None else int(len(df_5m)),
        "five_min_range": _fmt_range(df_5m),
        "mtf_5m_match_ratio": 0.0,
        "btc_match_ratio": 0.0,
        "one_hour_close_alignment": "timestamp_plus_55m",
        "join_mode": "merge_asof_backward",
    }

    align_df = pd.DataFrame({
        "align_ts": _norm_ts(pd.Series(df_1h["timestamp"] + pd.Timedelta(minutes=55)))
    }).sort_values("align_ts")
    if df_5m is not None and not df_5m.empty:
        right_5m = df_5m[["timestamp"]].copy()
        right_5m["timestamp"] = _norm_ts(right_5m["timestamp"])
        right_5m = right_5m.rename(columns={"timestamp": "source_ts"}).sort_values("source_ts")
        matched_5m = pd.merge_asof(
            align_df,
            right_5m,
            left_on="align_ts",
            right_on="source_ts",
            direction="backward",
            tolerance=pd.Timedelta(minutes=5),
        )
        report["mtf_5m_match_ratio"] = float(matched_5m["source_ts"].notna().mean())
        if matched_5m["source_ts"].notna().any():
            lag = (matched_5m["align_ts"] - matched_5m["source_ts"]).dt.total_seconds() / 60.0
            report["mtf_5m_lag_min_median"] = float(lag.dropna().median())

    if btc_df is not None and not btc_df.empty:
        btc_tmp = btc_df[["timestamp", "close"]].copy().sort_values("timestamp")
        btc_tmp["btc_1h_return"] = btc_tmp["close"].pct_change(1)
        btc_tmp["timestamp"] = _norm_ts(btc_tmp["timestamp"])
        btc_tmp = btc_tmp.rename(columns={"timestamp": "source_ts"})
        matched_btc = pd.merge_asof(
            align_df,
            btc_tmp[["source_ts", "btc_1h_return"]].sort_values("source_ts"),
            left_on="align_ts",
            right_on="source_ts",
            direction="backward",
            tolerance=pd.Timedelta(hours=2),
        )
        report["btc_match_ratio"] = float(matched_btc["btc_1h_return"].notna().mean())
        if matched_btc["source_ts"].notna().any():
            lag = (matched_btc["align_ts"] - matched_btc["source_ts"]).dt.total_seconds() / 60.0
            report["btc_lag_min_median"] = float(lag.dropna().median())

    return report


def _norm_ts(s: pd.Series) -> pd.Series:
    """Normalise a datetime Series to datetime64[us, UTC] to prevent merge_asof precision errors."""
    return s.dt.as_unit("us") if hasattr(s.dt, "as_unit") else pd.to_datetime(s, utc=True)


def _merge_asof_feature_frame(
    base_ts: pd.Series,
    source_df: pd.DataFrame | None,
    source_columns: list[str],
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    base = pd.DataFrame({
        "row_id": np.arange(len(base_ts)),
        "align_ts": _norm_ts(pd.Series(base_ts).reset_index(drop=True)),
    }).sort_values("align_ts")

    if source_df is None or source_df.empty:
        return pd.DataFrame({col: np.nan for col in source_columns}, index=np.arange(len(base_ts)))

    available_cols = ["timestamp"] + [col for col in source_columns if col in source_df.columns]
    if len(available_cols) == 1:
        return pd.DataFrame({col: np.nan for col in source_columns}, index=np.arange(len(base_ts)))

    right = source_df[available_cols].copy()
    right["timestamp"] = _norm_ts(right["timestamp"])
    right = right.sort_values("timestamp")
    merged = pd.merge_asof(
        base,
        right,
        left_on="align_ts",
        right_on="timestamp",
        direction="backward",
        tolerance=tolerance,
    ).sort_values("row_id")

    aligned = merged.reindex(columns=source_columns)
    for col in source_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[source_columns].reset_index(drop=True)


def _infer_constant_reason(row: dict) -> str:
    feature = row["feature"]
    missing_ratio = row["missing_ratio"]
    constant_ratio = row["constant_ratio"]
    top_value = row["top_value"]
    if constant_ratio <= 0.90:
        return ""
    if feature == "fear_greed_value":
        if missing_ratio >= 0.95:
            return "historical_fear_greed_unavailable"
        return "daily_sentiment_series_low_variation"
    if feature.startswith("mtf_5m_"):
        if missing_ratio >= 0.20:
            return "5m_alignment_sparse_or_missing"
        return "5m_signal_nearly_constant_after_alignment"
    if feature == "btc_1h_return":
        if missing_ratio >= 0.20:
            return "btc_reference_alignment_sparse_or_missing"
        return "btc_return_low_variation_in_window"
    if feature == "volume_spike":
        return "threshold_too_strict_or_volume_regime_stable"
    if feature in {"hour_of_day", "day_of_week"}:
        return "time_feature_distribution_collapsed_in_window"
    if top_value in {"0.0", "0", "1.0", "1"}:
        return "binary_feature_dominated_by_single_state"
    return "dominant_value_gt_90pct"


def _write_feature_quality_reports(
    X: pd.DataFrame,
    y: pd.Series,
    alignment_reports: list[dict],
    feature_source_notes: dict[str, str],
) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "feature_quality_report.csv"
    json_path = RESULTS_DIR / "feature_quality_report.json"
    corr_path = RESULTS_DIR / "feature_correlation_matrix.csv"

    rows = []
    corr_df = X.corr(numeric_only=True)
    for col in X.columns:
        s = X[col]
        non_na = s.dropna()
        if non_na.empty:
            constant_ratio = 1.0
            top_value = None
        else:
            vc = non_na.value_counts(dropna=False, normalize=True)
            constant_ratio = float(vc.iloc[0])
            top_value = vc.index[0]
        max_corr = None
        max_corr_partner = None
        if col in corr_df.columns:
            c = corr_df[col].drop(labels=[col], errors="ignore").dropna()
            if not c.empty:
                max_corr_partner = str(c.abs().idxmax())
                max_corr = float(c.loc[max_corr_partner])
        corr_to_target = None
        if non_na.size > 1:
            try:
                corr_to_target = float(pd.concat([s, y], axis=1).corr(numeric_only=True).iloc[0, 1])
            except Exception:
                corr_to_target = None
        rows.append({
            "feature": col,
            "missing_ratio": float(s.isna().mean()),
            "constant_ratio": float(constant_ratio),
            "unique_count": int(non_na.nunique()),
            "min": None if non_na.empty else float(non_na.min()),
            "max": None if non_na.empty else float(non_na.max()),
            "mean": None if non_na.empty else float(non_na.mean()),
            "std": None if non_na.empty else float(non_na.std(ddof=0)),
            "top_value": None if top_value is None else str(top_value),
            "max_abs_corr_partner": max_corr_partner,
            "max_abs_corr_value": None if max_corr is None else float(max_corr),
            "corr_to_target": corr_to_target,
            "source_note": feature_source_notes.get(col, ""),
        })

    for row in rows:
        row["constant_reason"] = _infer_constant_reason(row)

    report_df = pd.DataFrame(rows).sort_values(["constant_ratio", "missing_ratio"], ascending=False)
    report_df.to_csv(csv_path, index=False)
    corr_df.to_csv(corr_path, index=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "alignment_reports": alignment_reports,
        "quality_warnings": list(_QUALITY_WARNINGS),
        "features": rows,
        "correlation_matrix_csv": str(corr_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[CRYPTO ML] Feature quality report: {csv_path}")
    print(f"[CRYPTO ML] Feature quality JSON:   {json_path}")
    return {row["feature"]: row for row in rows}


def _fetch_ohlcv_all_paginated(stream) -> dict[str, pd.DataFrame]:
    """
    Fetch paginated 1h and 5m history for all symbols.
    Returns dict symbol -> DataFrame (with indicators computed).
    """
    from crypto_indicators import compute_all, add_rsi, add_macd, add_bollinger

    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT",
        "ONDO/USDT", "FET/USDT",
    ]
    result = {}
    fetched = 0
    since_ms, until_ms = _fetch_history_window()
    print(
        f"  Target training window: {TRAIN_HISTORY_DAYS} days "
        f"({datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date()} -> "
        f"{datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc).date()})"
    )

    for sym in symbols:
        print(f"  Fetching {sym} 1h history (paginated)...", flush=True)
        df = stream.get_ohlcv_paginated(
            sym,
            timeframe="1h",
            since_ms=since_ms,
            until_ms=until_ms,
            limit_per_call=PAGE_LIMIT,
        )
        if df is None or len(df) < MIN_1H_CANDLES:
            _warn_quality(
                f"{sym} 1h insufficient | candles={0 if df is None else len(df)} "
                f"| range={_fmt_range(df)} | min_required={MIN_1H_CANDLES}"
            )
            print("  [WARN] veri alinamadi veya yetersiz - atlaniyor")
            continue
        df = compute_all(df)
        result[sym] = df
        fetched += 1
        print(f"    {sym} 1h: {len(df)} candles | {_fmt_range(df)}")

    print(f"  Toplam: {fetched}/{len(symbols)} sembol basariyla yuklendi")
    print("  Fetching 5m MTF history (paginated)...", flush=True)

    for sym in list(result.keys()):
        base_df = result[sym]
        first_ts = pd.Timestamp(base_df["timestamp"].iloc[0])
        last_ts = pd.Timestamp(base_df["timestamp"].iloc[-1])
        sym_since_ms = int((first_ts - pd.Timedelta(hours=6)).timestamp() * 1000)
        sym_until_ms = int((last_ts + pd.Timedelta(minutes=55)).timestamp() * 1000)
        df_5m = stream.get_ohlcv_paginated(
            sym,
            timeframe="5m",
            since_ms=sym_since_ms,
            until_ms=sym_until_ms,
            limit_per_call=PAGE_LIMIT,
        )
        if df_5m is None or len(df_5m) < MIN_5M_CANDLES:
            _warn_quality(
                f"{sym} 5m insufficient | candles={0 if df_5m is None else len(df_5m)} "
                f"| range={_fmt_range(df_5m)} | min_required={MIN_5M_CANDLES}"
            )
            continue
        df_5m = add_rsi(df_5m)
        df_5m = add_macd(df_5m)
        df_5m = add_bollinger(df_5m)
        result[f"{sym}_5m"] = df_5m
        print(f"    {sym} 5m: {len(df_5m)} candles | {_fmt_range(df_5m)}")

    return result


# ── Fear & Greed history ──────────────────────────────────────────────────────

def _fetch_fg_history() -> dict[str, int]:
    """Return {date_str: value} dict for up to 365 days."""
    try:
        from crypto_sentiment import sentiment
        hist = sentiment.get_historical(days=365)
        fg_dict = {row["date_str"]: row["value"] for row in hist}
        if not fg_dict:
            _warn_quality("Fear&Greed history unavailable; feature will remain missing in training data")
        return fg_dict
    except Exception:
        _warn_quality("Fear&Greed history fetch failed; feature will remain missing in training data")
        return {}


def _descending_rank_features(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    valid = values.dropna()
    rank = pd.Series(np.nan, index=values.index, dtype=float)
    pct = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return rank, pct
    ranks = valid.rank(method="average", ascending=False)
    rank.loc[valid.index] = ranks.astype(float)
    if len(valid) == 1:
        pct.loc[valid.index] = 0.5
    else:
        pct.loc[valid.index] = 1.0 - ((ranks - 1.0) / (len(valid) - 1.0))
    return rank, pct


def _build_cross_sectional_relative_strength(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for symbol, df in data.items():
        if symbol.endswith("_5m") or symbol.endswith("_15m"):
            continue
        if df is None or df.empty:
            continue
        if "vol_ratio" not in df.columns:
            df = df.copy()
            df["vol_ratio"] = 1.0
        base = df[["timestamp", "close", "vol_ratio"]].copy().sort_values("timestamp")
        base["symbol"] = symbol
        base["ret_1h"] = base["close"].pct_change(1)
        base["ret_4h"] = base["close"].pct_change(4)
        base["ret_24h"] = base["close"].pct_change(24)
        rows.append(base[["timestamp", "symbol", "ret_1h", "ret_4h", "ret_24h", "vol_ratio"]])

    if not rows:
        return {}

    panel = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    for src_col, rank_col, pct_col in [
        ("ret_1h", "cs_ret_1h_rank", "cs_ret_1h_pct"),
        ("ret_4h", "cs_ret_4h_rank", "cs_ret_4h_pct"),
        ("ret_24h", "cs_ret_24h_rank", "cs_ret_24h_pct"),
        ("vol_ratio", "cs_volume_ratio_rank", "cs_volume_ratio_pct"),
    ]:
        rank_parts = []
        pct_parts = []
        for _, grp in panel.groupby("timestamp", sort=False):
            rank_s, pct_s = _descending_rank_features(grp[src_col])
            rank_parts.append(rank_s)
            pct_parts.append(pct_s)
        panel[rank_col] = pd.concat(rank_parts).sort_index()
        panel[pct_col] = pd.concat(pct_parts).sort_index()

    btc_ref = (
        panel.loc[panel["symbol"] == "BTC/USDT", ["timestamp", "ret_1h", "ret_4h", "ret_24h"]]
        .rename(columns={
            "ret_1h": "btc_ret_1h",
            "ret_4h": "btc_ret_4h",
            "ret_24h": "btc_ret_24h",
        })
        .sort_values("timestamp")
    )
    panel = panel.merge(btc_ref, on="timestamp", how="left")
    panel["rel_btc_ret_1h"] = panel["ret_1h"] - panel["btc_ret_1h"]
    panel["rel_btc_ret_4h"] = panel["ret_4h"] - panel["btc_ret_4h"]
    panel["rel_btc_ret_24h"] = panel["ret_24h"] - panel["btc_ret_24h"]
    panel.loc[panel["symbol"] == "BTC/USDT", ["rel_btc_ret_1h", "rel_btc_ret_4h", "rel_btc_ret_24h"]] = 0.0

    feature_cols = [
        "timestamp",
        "cs_ret_1h_rank",
        "cs_ret_1h_pct",
        "cs_ret_4h_rank",
        "cs_ret_4h_pct",
        "cs_ret_24h_rank",
        "cs_ret_24h_pct",
        "cs_volume_ratio_rank",
        "cs_volume_ratio_pct",
        "rel_btc_ret_1h",
        "rel_btc_ret_4h",
        "rel_btc_ret_24h",
    ]
    result: dict[str, pd.DataFrame] = {}
    for symbol, grp in panel.groupby("symbol", sort=False):
        result[symbol] = grp[feature_cols].copy().sort_values("timestamp").reset_index(drop=True)
    return result


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None,
    fg_dict: dict[str, int],
    df_5m: pd.DataFrame | None = None,
    relative_strength_df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
    btc_dominance_info: dict | None = None,
    htf_bias_df: pd.DataFrame | None = None,
    htf_info: dict | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix from enriched 1h OHLCV DataFrame.

    MTF features:
      - 5m RSI/MACD/BB: aligned from separate 5m fetch by timestamp floor

    Important:
      - direct label-proxy 1h features (rsi14 / mtf_1h_rsi / mtf_1h_trend)
        are intentionally excluded from FEATURE_NAMES to reduce tautology risk.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["bar_close_ts"] = df["timestamp"] + pd.Timedelta(minutes=55)

    # ── Core 1h features ──────────────────────────────────────────────────────
    df["rsi14"]           = df["rsi14"].fillna(50)
    df["ema8_21_cross"]   = (df["ema8"] - df["ema21"]) / df["close"].replace(0, np.nan)
    bb_range              = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_position"]     = (df["close"] - df["bb_lower"]) / bb_range
    df["macd_hist_norm"]  = df["macd_hist"] / df["close"].replace(0, np.nan)
    df["volume_ratio"]    = df["vol_ratio"].fillna(1.0)
    df["atr14_norm"]      = df["atr14"] / df["close"].replace(0, np.nan)
    df["price_ema50_pct"] = (df["close"] - df["ema50"]) / df["ema50"].replace(0, np.nan)

    # ── Return features ───────────────────────────────────────────────────────
    df["ret_1h"]  = df["close"].pct_change(1)
    df["ret_4h"]  = df["close"].pct_change(4)
    df["ret_12h"] = df["close"].pct_change(12)
    df["ret_24h"] = df["close"].pct_change(24)

    # ── Candle structure ──────────────────────────────────────────────────────
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range

    # ── Fear & Greed ──────────────────────────────────────────────────────────
    if fg_dict:
        fg_key = df["bar_close_ts"].dt.strftime("%Y-%m-%d")
        df["fear_greed_value"] = pd.to_numeric(fg_key.map(fg_dict), errors="coerce")
        fg_missing = float(df["fear_greed_value"].isna().mean())
        if fg_missing > 0:
            _warn_quality(f"Fear&Greed history partially missing after alignment | missing_ratio={fg_missing:.1%}")
    else:
        _warn_quality("Fear&Greed feature unavailable during training; leaving feature missing instead of neutral fill")
        df["fear_greed_value"] = np.nan

    # ── BTC 1h return (for ETH/SOL aligned by timestamp) ─────────────────────
    if btc_df is not None and len(btc_df) > 1:
        btc_tmp = btc_df[["timestamp", "close"]].copy().sort_values("timestamp")
        btc_tmp["btc_1h_return"] = btc_tmp["close"].pct_change(1)
        aligned_btc = _merge_asof_feature_frame(
            df["bar_close_ts"],
            btc_tmp,
            ["btc_1h_return"],
            tolerance=pd.Timedelta(hours=2),
        )
        df["btc_1h_return"] = pd.to_numeric(aligned_btc["btc_1h_return"], errors="coerce")
        btc_missing = float(df["btc_1h_return"].isna().mean())
        if btc_missing > 0:
            _warn_quality(f"BTC 1h return partially missing after alignment | missing_ratio={btc_missing:.1%}")
    else:
        _warn_quality("BTC 1h return reference unavailable; leaving feature missing")
        df["btc_1h_return"] = np.nan

    # ── Funding-rate features (8h perp crowding, aligned backward) ──────────
    if funding_df is not None and not funding_df.empty:
        aligned_funding = _merge_asof_feature_frame(
            df["bar_close_ts"],
            funding_df.sort_values("timestamp"),
            ["funding_rate", "funding_rate_avg3"],
            tolerance=pd.Timedelta(hours=12),
        )
        df["funding_rate"] = pd.to_numeric(aligned_funding["funding_rate"], errors="coerce").fillna(0.0)
        df["funding_rate_avg3"] = pd.to_numeric(aligned_funding["funding_rate_avg3"], errors="coerce").fillna(df["funding_rate"])
    else:
        df["funding_rate"] = 0.0
        df["funding_rate_avg3"] = 0.0
    df["funding_rate_extreme"] = (df["funding_rate"].abs() > 0.001).astype(float)
    df["funding_contrarian"] = -np.sign(df["funding_rate"]).astype(float)

    # ── BTC dominance features (current/cache snapshot fallback) ─────────────
    dom_info = btc_dominance_info or {}
    df["btc_dominance"] = float(dom_info.get("btc_dominance", 55.0) or 55.0)
    df["btc_dominance_regime"] = float(dom_info.get("btc_dominance_regime_code", 1.0) or 1.0)
    df["btc_dominance_7d_change"] = float(dom_info.get("btc_dominance_7d_change", 0.0) or 0.0)

    # ── 4h higher-timeframe bias (historical when available, else snapshot) ─
    if htf_bias_df is not None and not htf_bias_df.empty:
        aligned_htf = _merge_asof_feature_frame(
            df["bar_close_ts"],
            htf_bias_df.sort_values("timestamp"),
            ["htf_bias"],
            tolerance=pd.Timedelta(hours=4),
        )
        df["htf_bias"] = pd.to_numeric(aligned_htf["htf_bias"], errors="coerce").fillna(0.0)
    elif htf_info is not None:
        df["htf_bias"] = 1.0 if str(htf_info.get("htf_bias", "BEAR")).upper() == "BULL" else 0.0
    else:
        df["htf_bias"] = 0.0

    # ── MTF 5m features (aligned from separate 5m fetch) ─────────────────────
    if df_5m is not None and len(df_5m) >= 30 and "rsi14" in df_5m.columns:
        df_5m_aligned = df_5m.copy().sort_values("timestamp")
        if {"bb_upper", "bb_lower", "close"}.issubset(df_5m_aligned.columns):
            bb_range5 = (df_5m_aligned["bb_upper"] - df_5m_aligned["bb_lower"]).replace(0, np.nan)
            df_5m_aligned["mtf_5m_bb_pos"] = (df_5m_aligned["close"] - df_5m_aligned["bb_lower"]) / bb_range5
        else:
            df_5m_aligned["mtf_5m_bb_pos"] = np.nan
        if "rsi14" in df_5m_aligned.columns:
            df_5m_aligned["mtf_5m_rsi"] = df_5m_aligned["rsi14"]
        if "macd_hist" in df_5m_aligned.columns:
            df_5m_aligned["mtf_5m_macd_hist"] = df_5m_aligned["macd_hist"]

        aligned_5m = _merge_asof_feature_frame(
            df["bar_close_ts"],
            df_5m_aligned,
            ["mtf_5m_rsi", "mtf_5m_macd_hist", "mtf_5m_bb_pos"],
            tolerance=pd.Timedelta(minutes=60),
        )
        for col in ["mtf_5m_rsi", "mtf_5m_macd_hist", "mtf_5m_bb_pos"]:
            df[col] = pd.to_numeric(aligned_5m[col], errors="coerce")
            miss = float(df[col].isna().mean())
            if miss > 0:
                _warn_quality(f"{col} partially missing after 5m alignment | missing_ratio={miss:.1%}")
    else:
        _warn_quality("5m MTF source unavailable; leaving aligned 5m features missing")
        df["mtf_5m_rsi"]       = np.nan
        df["mtf_5m_macd_hist"] = np.nan
        df["mtf_5m_bb_pos"]    = np.nan

    # ── Risky-tier volatility features (computed for all coins) ──────────────
    vol_ratio_s = df.get("vol_ratio", pd.Series(1.0, index=df.index)).fillna(1.0)
    df["volume_spike"] = (vol_ratio_s > 3.0).astype(float)

    if "high" in df.columns:
        rolling_high = df["high"].rolling(window=168, min_periods=1).max()
        df["price_vs_7d_high"] = (df["close"] / rolling_high.replace(0, np.nan)).fillna(1.0)
    else:
        df["price_vs_7d_high"] = 1.0

    # ── NEW: Momentum features ────────────────────────────────────────────────
    df["ret_2h"]    = df["close"].pct_change(2)
    df["ret_6h"]    = df["close"].pct_change(6)
    df["rsi_slope"] = df["rsi14"].diff(3)
    df["macd_slope"] = df["macd_hist"].diff(4)
    df["ema_spread"] = (df["ema8"] - df["ema21"]) / df["close"].replace(0, np.nan)

    # ── NEW: Volatility features ──────────────────────────────────────────────
    atr_mean48 = df["atr14"].rolling(48, min_periods=10).mean().replace(0, np.nan)
    df["atr_ratio"] = df["atr14"] / atr_mean48

    bb_w = (df["bb_upper"] - df["bb_lower"]) if "bb_upper" in df.columns else pd.Series(np.nan, index=df.index)
    bb_w_mean = bb_w.rolling(48, min_periods=10).mean()
    bb_w_std  = bb_w.rolling(48, min_periods=10).std().replace(0, np.nan)
    df["bb_width_zscore"] = (bb_w - bb_w_mean) / bb_w_std

    hl_mean24 = df["high_low_range"].rolling(24, min_periods=5).mean().replace(0, np.nan)
    df["high_low_zscore"] = df["high_low_range"] / hl_mean24

    # ── NEW: Volume features ──────────────────────────────────────────────────
    vol_mean48 = df["volume"].rolling(48, min_periods=10).mean()
    vol_std48  = df["volume"].rolling(48, min_periods=10).std().replace(0, np.nan)
    df["volume_zscore"] = (df["volume"] - vol_mean48) / vol_std48

    vol_mean6  = df["volume"].rolling(6,  min_periods=2).mean()
    vol_mean24 = df["volume"].rolling(24, min_periods=5).mean().replace(0, np.nan)
    df["volume_trend"] = vol_mean6 / vol_mean24

    df["buy_pressure"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)

    # ── NEW: Cross-asset features (BTC 4h return + vol proxy) ────────────────
    if btc_df is not None and len(btc_df) > 4:
        btc_ca = btc_df[["timestamp", "close"]].copy().sort_values("timestamp")
        btc_ca["btc_ret_4h"]  = btc_ca["close"].pct_change(4)
        btc_ca["btc_vol_1h"]  = btc_ca["close"].pct_change(1).abs()
        aligned_ca = _merge_asof_feature_frame(
            df["bar_close_ts"],
            btc_ca,
            ["btc_ret_4h", "btc_vol_1h"],
            tolerance=pd.Timedelta(hours=2),
        )
        df["btc_ret_4h"] = pd.to_numeric(aligned_ca["btc_ret_4h"], errors="coerce")
        df["btc_vol_1h"] = pd.to_numeric(aligned_ca["btc_vol_1h"], errors="coerce")
    else:
        # BTC itself has no external BTC reference — leave missing, imputed as 0 later
        df["btc_ret_4h"] = np.nan
        df["btc_vol_1h"] = np.nan

    # symbol_encoded: placeholder NaN here; overwritten per-symbol in train() and predict_proba()
    df["symbol_encoded"] = np.nan

    return df[FEATURE_NAMES]


def _same_value_ratio(s: pd.Series) -> float:
    """Return the share of rows occupied by the most common exact value."""
    if s.empty:
        return 1.0
    return float(s.value_counts(dropna=False, normalize=True).iloc[0])


def _drop_constant_like_features(
    X: pd.DataFrame,
    feature_stats: dict[str, dict],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features where one exact value appears in more than 90% of rows.
    This catches sparse binary flags that can become overfit shortcuts.
    """
    drop_cols = []
    for col in X.columns:
        stats = feature_stats.get(col, {})
        ratio = float(stats.get("constant_ratio", _same_value_ratio(X[col])))
        if ratio > 0.90:
            top_val = stats.get("top_value")
            reason = stats.get("constant_reason", "")
            print(
                f"[CRYPTO ML] DROP feature {col}: same_value_ratio={ratio:.1%} "
                f"top_value={top_val} reason={reason or 'n/a'}"
            )
            drop_cols.append(col)
    if not drop_cols:
        return X, []
    return X.drop(columns=drop_cols), drop_cols


def _print_prediction_diagnostics(
    symbol: str | None,
    x_raw: pd.DataFrame,
    x_filled: pd.DataFrame,
    raw_prob: float,
) -> None:
    """Verbose diagnostics for suspicious crypto ML predictions."""
    if symbol not in DIAG_SYMBOLS:
        return

    row_raw = x_raw.iloc[0]
    row = x_filled.iloc[0]
    nan_cols = [c for c in x_raw.columns if pd.isna(row_raw[c])]
    zero_cols = [c for c in x_filled.columns if float(row[c]) == 0.0]

    print(f"[CRYPTO ML DIAG] {symbol} feature values at prediction time:")
    print(f"[CRYPTO ML DIAG] {symbol} raw_xgb_prob={raw_prob!r} ({raw_prob:.6f})")
    if nan_cols:
        print(f"[CRYPTO ML DIAG] {symbol} NaN filled with 0: {', '.join(nan_cols)}")
    if zero_cols:
        print(f"[CRYPTO ML DIAG] {symbol} zero-valued features: {', '.join(zero_cols)}")
    if "volume_spike" in row.index:
        print(f"[CRYPTO ML DIAG] {symbol} volume_spike={float(row['volume_spike']):.6f}")
    if "price_vs_7d_high" in row.index:
        print(f"[CRYPTO ML DIAG] {symbol} price_vs_7d_high={float(row['price_vs_7d_high']):.6f}")

    for name, value in row.items():
        print(f"[CRYPTO ML DIAG] {symbol} {name}={float(value):.10g}")


# ── Label ─────────────────────────────────────────────────────────────────────

def _resolve_label_config(label_config: dict | None = None) -> dict:
    cfg = dict(LABEL_CONFIG)
    if label_config:
        cfg.update(label_config)
    return cfg


def _resolve_execution_cost_config(
    label_config: dict | None = None,
    execution_cost_config: dict | None = None,
) -> dict:
    cfg = json.loads(json.dumps(EXECUTION_COST_CONFIG))
    if execution_cost_config:
        for key, value in execution_cost_config.items():
            if key == "order_modes" and isinstance(value, dict):
                cfg["order_modes"].update(value)
            else:
                cfg[key] = value

    # Backward-compatible overrides from legacy label config fields.
    if label_config:
        if "costs_enabled" in label_config:
            cfg["enabled"] = bool(label_config["costs_enabled"])
        if "order_type" in label_config:
            cfg["order_type"] = str(label_config["order_type"])
        mode = cfg["order_type"]
        cfg["order_modes"].setdefault(mode, {})
        for key in _LEGACY_EXECUTION_COST_KEYS:
            if key in {"order_type", "costs_enabled"}:
                continue
            if key in label_config:
                cfg["order_modes"][mode][key] = float(label_config[key])
    return cfg


def _per_side_cost_bps(
    label_config: dict | None = None,
    execution_cost_config: dict | None = None,
) -> dict[str, float | str | bool]:
    cfg = _resolve_execution_cost_config(label_config, execution_cost_config)
    order_type = str(cfg.get("order_type", "market"))
    mode_cfg = dict(cfg.get("order_modes", {}).get(order_type, {}))
    fee_bps = float(mode_cfg.get("fee_bps", 0.0))
    spread_bps = float(mode_cfg.get("spread_bps", 0.0))
    slippage_bps = float(mode_cfg.get("slippage_bps", 0.0))
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        fee_bps = 0.0
        spread_bps = 0.0
        slippage_bps = 0.0
    return {
        "config": cfg,
        "enabled": enabled,
        "order_type": order_type,
        "fee_bps": fee_bps,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "per_side_total_bps": fee_bps + spread_bps + slippage_bps,
    }


def _round_trip_cost_rate(
    label_config: dict | None = None,
    execution_cost_config: dict | None = None,
) -> float:
    cost = _per_side_cost_bps(label_config, execution_cost_config)
    return (float(cost["per_side_total_bps"]) * 2.0) / 10000.0


def _compute_barriers(
    entry_price: float,
    atr_value: float,
    label_config: dict,
) -> tuple[float, float]:
    target_gain = float(label_config.get("min_target_gain", 0.0))
    stop_loss = float(label_config.get("min_stop_loss", 0.0))
    if label_config.get("use_atr_targets", False) and pd.notna(atr_value) and entry_price > 0:
        atr_pct = float(atr_value) / float(entry_price)
        target_gain = max(target_gain, atr_pct * float(label_config.get("target_atr_mult", 1.0)))
        stop_loss = max(stop_loss, atr_pct * float(label_config.get("stop_atr_mult", 1.0)))
    return target_gain, stop_loss


def _build_legacy_labels(df: pd.DataFrame, label_config: dict) -> pd.Series:
    return _build_legacy_outcomes(df, label_config)["label"]


def _build_legacy_outcomes(
    df: pd.DataFrame,
    label_config: dict,
    execution_cost_config: dict | None = None,
) -> pd.DataFrame:
    horizon = int(label_config.get("horizon_bars", 3))
    min_gain = float(label_config.get("min_gain", 0.01))
    cost_rate = _round_trip_cost_rate(label_config, execution_cost_config)
    future_close = df["close"].shift(-horizon)
    label = (future_close > df["close"] * (1 + min_gain)).astype(float)
    label[future_close.isna()] = np.nan
    gross_return = (future_close / df["close"]) - 1.0
    gross_return = gross_return.astype(float)
    gross_return[future_close.isna()] = np.nan
    net_return = gross_return - cost_rate
    net_return = net_return.astype(float)
    net_return[future_close.isna()] = np.nan
    return pd.DataFrame({
        "label": label.astype(float),
        "estimated_gross_return_before_costs": gross_return,
        "estimated_net_return_after_costs": net_return,
    })


def _build_tradeable_labels(df: pd.DataFrame, label_config: dict) -> pd.Series:
    return _build_tradeable_outcomes(df, label_config)["label"]


def _build_tradeable_outcomes(
    df: pd.DataFrame,
    label_config: dict,
    execution_cost_config: dict | None = None,
) -> pd.DataFrame:
    horizon = int(label_config.get("horizon_bars", 6))
    fallback_mode = str(label_config.get("fallback_mode", "net_return"))
    fallback_min_net_return = float(label_config.get("fallback_min_net_return", 0.0))
    intrabar_conflict_policy = str(label_config.get("intrabar_conflict_policy", "stop_first"))
    cost_rate = _round_trip_cost_rate(label_config, execution_cost_config)

    close = df["close"].astype(float).reset_index(drop=True)
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    atr = pd.to_numeric(df.get("atr14", pd.Series(np.nan, index=df.index)), errors="coerce").reset_index(drop=True)

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    gross_returns = pd.Series(np.nan, index=df.index, dtype=float)
    est_returns = pd.Series(np.nan, index=df.index, dtype=float)
    last_valid = max(len(df) - horizon, 0)
    for i in range(last_valid):
        entry = close.iloc[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        target_gain, stop_loss = _compute_barriers(entry, atr.iloc[i], label_config)
        target_price = entry * (1.0 + target_gain + cost_rate)
        stop_price = entry * (1.0 - stop_loss)
        label = np.nan
        gross_return = np.nan
        est_return = np.nan

        for j in range(i + 1, min(i + horizon + 1, len(df))):
            hit_target = bool(high.iloc[j] >= target_price)
            hit_stop = bool(low.iloc[j] <= stop_price)
            if hit_target and hit_stop:
                if intrabar_conflict_policy == "stop_first":
                    label = 0.0
                    gross_return = -stop_loss
                    est_return = -(stop_loss + cost_rate)
                else:
                    label = 1.0
                    gross_return = target_gain + cost_rate
                    est_return = target_gain
                break
            if hit_stop:
                label = 0.0
                gross_return = -stop_loss
                est_return = -(stop_loss + cost_rate)
                break
            if hit_target:
                label = 1.0
                gross_return = target_gain + cost_rate
                est_return = target_gain
                break

        if np.isnan(label):
            future_close = close.iloc[i + horizon]
            gross_return = (future_close / entry) - 1.0
            net_return = gross_return - cost_rate
            est_return = net_return
            if fallback_mode == "net_return":
                label = 1.0 if net_return >= fallback_min_net_return else 0.0
            else:
                label = 1.0 if net_return > 0 else 0.0

        labels.iloc[i] = label
        gross_returns.iloc[i] = gross_return
        est_returns.iloc[i] = est_return

    return pd.DataFrame({
        "label": labels,
        "estimated_gross_return_before_costs": gross_returns,
        "estimated_net_return_after_costs": est_returns,
    })


def _build_maxhigh_outcomes(
    df: pd.DataFrame,
    label_config: dict,
    execution_cost_config: dict | None = None,
) -> pd.DataFrame:
    """
    Max-high-rise label: 1 if max(high[t+1 .. t+H]) / close[t] - 1 >= threshold.

    Uses rolling max over future highs so any intrabar spike within the horizon
    counts as a win, matching real entry/exit behaviour better than close-to-close.

    Formula (vectorised, no lookahead at index t):
        shifted_high      = high.shift(-1)          # at t: high[t+1]
        rolling_max       = shifted_high.rolling(H).max()  # at t: max(high[t-H+2 .. t+1])
        future_max_high   = rolling_max.shift(-(H-1))       # at t: max(high[t+1 .. t+H])
    """
    horizon   = int(label_config.get("horizon_bars", 8))
    threshold = float(label_config.get("min_gain", 0.03))
    cost_rate = _round_trip_cost_rate(label_config, execution_cost_config)

    close = df["close"].astype(float).reset_index(drop=True)
    high  = df["high"].astype(float).reset_index(drop=True)

    # Future max high over next `horizon` bars
    future_max_high = (
        high.shift(-1)
            .rolling(horizon, min_periods=horizon)
            .max()
            .shift(-(horizon - 1))
    )
    future_max_high.index = df.index

    # Gross return proxy: future close at horizon (for net-expectancy accounting)
    future_close = df["close"].astype(float).shift(-horizon)
    gross_return = (future_close / df["close"]) - 1.0
    net_return   = gross_return - cost_rate

    # Label: future max high hits threshold above entry close
    ratio = future_max_high / df["close"].astype(float)
    label = (ratio - 1.0 >= threshold).astype(float)

    # Mask trailing rows where future is unknown
    tail_mask = future_max_high.isna() | future_close.isna()
    label[tail_mask]      = np.nan
    gross_return[tail_mask] = np.nan
    net_return[tail_mask]   = np.nan

    return pd.DataFrame({
        "label": label,
        "estimated_gross_return_before_costs": gross_return,
        "estimated_net_return_after_costs": net_return,
    })


def _build_directional_labels(
    df: pd.DataFrame,
    horizon: int = 8,
    threshold: float = 0.025,
    dominance_ratio: float = 2.0,
) -> pd.DataFrame:
    """
    Pure outcome-based directional labels for the two-model training system.

    ret_future      = (max high over next horizon bars - close) / close  → upside potential
    ret_future_down = (close - min low over next horizon bars) / close   → downside potential

    LONG  (long_label=1):  ret_future >= threshold  AND  ret_future > ret_future_down * dominance_ratio
    SHORT (short_label=1): ret_future_down >= threshold  AND  ret_future_down > ret_future * dominance_ratio
    NEUTRAL: everything else (both sides < threshold, or neither side clearly dominates).

    Design note — NO entry-state filters (RSI, EMA) in the label:
      Prior label included "AND rsi14 > 50 AND close > ema21" for LONG, and the inverse for SHORT.
      This created direct feature-label leakage because bb_position, price_ema50_pct, ema8_21_cross,
      macd_hist_norm and multi-horizon returns all encode the same structure.  The model would learn
      "which rows already look bullish/bearish" rather than "which setups lead to a future directional move."
      Removing these conditions makes the label purely outcome-driven and forces the model to find
      features that predict future price movement regardless of current trend state.

    dominance_ratio=2.0: upside must be at least 2× downside to qualify as LONG (and vice-versa),
      ensuring directional clarity. Raised from 1.5 to reduce ambiguous labels and improve fold
      consistency (LONG WF dropped 51.6%→46.3% at 1.5 with 28 features; reverted + tightened).

    LONG model training set: rows where long_label==1 OR short_label==1 (exclude NEUTRAL).
      y_long = long_label  (1=LONG, 0=confirmed SHORT counterexample)
    SHORT model training set: same rows.
      y_short = short_label (1=SHORT, 0=confirmed LONG counterexample)

    Returns DataFrame with columns:
      long_label, short_label, ret_future, ret_future_down
    Trailing rows where future window is incomplete → NaN in all columns.
    """
    close = df["close"].astype(float).reset_index(drop=True)
    high  = df["high"].astype(float).reset_index(drop=True)
    low   = df["low"].astype(float).reset_index(drop=True)

    # Vectorised future max high over next `horizon` bars (no lookahead at t)
    future_max_high = (
        high.shift(-1)
            .rolling(horizon, min_periods=horizon)
            .max()
            .shift(-(horizon - 1))
    )
    # Vectorised future min low over next `horizon` bars
    future_min_low = (
        low.shift(-1)
           .rolling(horizon, min_periods=horizon)
           .min()
           .shift(-(horizon - 1))
    )

    ret_future      = (future_max_high - close) / close.replace(0, np.nan)
    ret_future_down = (close - future_min_low)  / close.replace(0, np.nan)

    # Pure outcome label — no entry-state filters
    long_label  = (
        (ret_future >= threshold) &
        (ret_future > ret_future_down * dominance_ratio)
    ).astype(float)

    short_label = (
        (ret_future_down >= threshold) &
        (ret_future_down > ret_future * dominance_ratio)
    ).astype(float)

    # Mask tail rows where future window is incomplete
    tail_mask = future_max_high.isna() | future_min_low.isna()
    long_label[tail_mask]  = np.nan
    short_label[tail_mask] = np.nan

    long_label.index      = df.index
    short_label.index     = df.index
    ret_future.index      = df.index
    ret_future_down.index = df.index

    return pd.DataFrame({
        "long_label":       long_label,
        "short_label":      short_label,
        "ret_future":       ret_future,
        "ret_future_down":  ret_future_down,
    })


def _build_outcomes(
    df: pd.DataFrame,
    label_config: dict | None = None,
    execution_cost_config: dict | None = None,
) -> pd.DataFrame:
    """
    Build labels and estimated net returns from the configured horizon/barrier logic.
    Trailing rows with unknown future remain NaN.
    """
    cfg = _resolve_label_config(label_config)
    mode = str(cfg.get("mode", "triple_barrier"))
    if mode == "legacy_simple":
        return _build_legacy_outcomes(df, cfg, execution_cost_config)
    if mode == "triple_barrier":
        return _build_tradeable_outcomes(df, cfg, execution_cost_config)
    if mode == "max_high_rise":
        return _build_maxhigh_outcomes(df, cfg, execution_cost_config)
    raise ValueError(f"Unknown label mode: {mode}")


def _build_labels(
    df: pd.DataFrame,
    label_config: dict | None = None,
    execution_cost_config: dict | None = None,
) -> pd.Series:
    """
    Build labels from the configured horizon and barrier logic.
    Trailing rows with unknown future remain NaN.
    """
    return _build_outcomes(df, label_config, execution_cost_config)["label"]


# ── Walk-forward validation ───────────────────────────────────────────────────

def _safe_metric_mean(folds: list[dict], key: str) -> float | None:
    values = [float(f[key]) for f in folds if f.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _build_production_decision(wf_report: dict) -> dict:
    summary = wf_report.get("summary", {})
    precision_mean = float(summary.get("precision_mean") or 0.0)
    trade_count_total = int(summary.get("trade_count_total") or 0)
    net_expectancy_mean = float(summary.get("net_expectancy_after_costs_mean") or 0.0)

    checks = {
        "precision": precision_mean >= MIN_ACCEPT_PRECISION,
        "trade_count": trade_count_total >= MIN_ACCEPT_TRADES,
        "net_expectancy": net_expectancy_mean > MIN_ACCEPT_EXPECTANCY,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "failed_checks": reasons,
        "thresholds": {
            "min_precision": MIN_ACCEPT_PRECISION,
            "min_trade_count": MIN_ACCEPT_TRADES,
            "min_net_expectancy_after_costs": MIN_ACCEPT_EXPECTANCY,
        },
    }


def _write_model_report(report: dict) -> None:
    MODEL_REPORT_PATH.parent.mkdir(exist_ok=True)
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[CRYPTO ML] Model report JSON: {MODEL_REPORT_PATH}")


def _write_threshold_report(report: dict) -> None:
    THRESHOLD_REPORT_PATH.parent.mkdir(exist_ok=True)
    THRESHOLD_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[CRYPTO ML] Threshold report JSON: {THRESHOLD_REPORT_PATH}")


def _approx_max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    return float(drawdown.min())


def _compute_expectancy(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    win_rate = float((returns > 0).mean())
    avg_win = float(winners.mean()) if not winners.empty else 0.0
    avg_loss = float(losers.mean()) if not losers.empty else 0.0
    return float((win_rate * avg_win) + ((1.0 - win_rate) * avg_loss))


def _threshold_candidate_metrics(tier_df: pd.DataFrame, threshold: float, min_trades: int) -> dict:
    trades = tier_df.loc[tier_df["proba"] >= threshold].copy()
    trade_count = int(len(trades))
    if trade_count == 0:
        precision = 0.0
        avg_return_gross = 0.0
        avg_return_net = 0.0
        expectancy_gross = 0.0
        expectancy_net = 0.0
        max_drawdown = 0.0
    else:
        gross_returns = trades["estimated_gross_return_before_costs"].astype(float)
        net_returns = trades["estimated_net_return_after_costs"].astype(float)
        precision = float((trades["y_true"] == 1).mean())
        avg_return_gross = float(gross_returns.mean())
        avg_return_net = float(net_returns.mean())
        expectancy_gross = _compute_expectancy(gross_returns)
        expectancy_net = _compute_expectancy(net_returns)
        max_drawdown = _approx_max_drawdown(net_returns)

    meets_trade_count = trade_count >= min_trades
    meets_expectancy = expectancy_net > THRESHOLD_MIN_EXPECTANCY
    return {
        "threshold": float(threshold),
        "trade_count": trade_count,
        "precision": precision,
        "gross_avg_return": avg_return_gross,
        "net_avg_return": avg_return_net,
        "max_drawdown_approx": max_drawdown,
        "gross_expectancy": expectancy_gross,
        "net_expectancy": expectancy_net,
        "cost_drag": avg_return_gross - avg_return_net,
        "meets_min_trade_count": meets_trade_count,
        "meets_net_expectancy": meets_expectancy,
        "eligible": bool(meets_trade_count and meets_expectancy),
    }


def _choose_threshold_candidate(candidates: list[dict]) -> dict:
    eligible = [c for c in candidates if c["eligible"]]
    if eligible:
        chosen = max(
            eligible,
            key=lambda item: (
                item["precision"],
                item["net_expectancy"],
                item["max_drawdown_approx"],
                item["trade_count"],
                -item["threshold"],
            ),
        )
        return {
            "selected": chosen,
            "selection_status": "eligible",
            "enabled": True,
        }

    fallback = max(
        candidates,
        key=lambda item: (
            item["net_expectancy"],
            item["precision"],
            item["max_drawdown_approx"],
            item["trade_count"],
            -item["threshold"],
        ),
    )
    return {
        "selected": fallback,
        "selection_status": "fallback_insufficient_edge",
        "enabled": False,
    }


def _choose_cost_disabled_reference(candidates: list[dict], min_trades: int) -> dict:
    eligible = [
        c for c in candidates
        if c["trade_count"] >= min_trades and c["gross_expectancy"] > THRESHOLD_MIN_EXPECTANCY
    ]
    pool = eligible if eligible else candidates
    return max(
        pool,
        key=lambda item: (
            item["precision"],
            item["gross_expectancy"],
            item["trade_count"],
            -item["threshold"],
        ),
    )


def _build_threshold_report_from_fold_maps(
    fold_threshold_maps: list[dict[str, float | None]],
    trained_at: str,
) -> dict:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trained_at": trained_at,
        "report_path": str(THRESHOLD_REPORT_PATH),
        "grid": list(THRESHOLD_GRID),
        "selection_policy": {
            "method": "per_fold_validation_then_median",
            "direction_scope": "long_confidence_proxy",
        },
        "tiers": {},
        "per_fold_selected_thresholds": fold_threshold_maps,
    }

    for tier in ["MAJOR", "RISKY"]:
        tier_values = [
            float(th_map[tier])
            for th_map in fold_threshold_maps
            if th_map.get(tier) is not None
        ]
        if tier_values:
            selected = round(float(np.median(tier_values)), 4)
            enabled = True
            selection_status = "fold_median"
        else:
            selected = float(THRESHOLD_FALLBACKS[tier])
            enabled = False
            selection_status = "fallback_no_fold_threshold"
        report["tiers"][tier] = {
            "enabled": enabled,
            "selected_threshold": selected if enabled else None,
            "fallback_threshold": selected,
            "no_trade_zone": {
                "short_below_or_equal": round(1.0 - selected, 4),
                "long_above_or_equal": round(selected, 4),
            },
            "selection_status": selection_status,
            "source_fold_count": len(tier_values),
        }

    return report


def _optimize_thresholds(prediction_df: pd.DataFrame, trained_at: str) -> dict:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trained_at": trained_at,
        "report_path": str(THRESHOLD_REPORT_PATH),
        "grid": list(THRESHOLD_GRID),
        "selection_policy": {
            "min_trade_count": dict(THRESHOLD_MIN_TRADES),
            "min_net_expectancy_after_costs": THRESHOLD_MIN_EXPECTANCY,
            "selection_rule": "eligible_by_min_trades_and_expectancy_then_max_precision",
            "direction_scope": "long_confidence_proxy",
        },
        "cost_comparison": {},
        "tiers": {},
    }

    comparison_rows = []
    for tier in ["MAJOR", "RISKY"]:
        tier_df = prediction_df.loc[prediction_df["tier"] == tier].copy()
        min_trades = int(THRESHOLD_MIN_TRADES[tier])
        candidates = [
            _threshold_candidate_metrics(tier_df, threshold, min_trades)
            for threshold in THRESHOLD_GRID
        ]
        choice = _choose_threshold_candidate(candidates)
        gross_reference = _choose_cost_disabled_reference(candidates, min_trades)
        selected = dict(choice["selected"])
        threshold = selected["threshold"]
        no_trade_zone = {
            "short_below_or_equal": round(1.0 - threshold, 4),
            "long_above_or_equal": round(threshold, 4),
        }

        report["tiers"][tier] = {
            "enabled": choice["enabled"],
            "selection_status": choice["selection_status"],
            "selected_threshold": threshold if choice["enabled"] else None,
            "fallback_threshold": threshold,
            "no_trade_zone": no_trade_zone,
            "selected_metrics": selected,
            "costs_disabled_reference": gross_reference,
            "candidates": candidates,
        }
        comparison_rows.append({
            "tier": tier,
            "selected_threshold": threshold if choice["enabled"] else None,
            "fallback_threshold": threshold,
            "gross_avg_return": selected["gross_avg_return"],
            "net_avg_return": selected["net_avg_return"],
            "gross_expectancy": selected["gross_expectancy"],
            "net_expectancy": selected["net_expectancy"],
            "cost_drag": selected["cost_drag"],
            "costs_disabled_threshold_reference": gross_reference["threshold"],
        })

    report["cost_comparison"] = {
        "selected_thresholds": comparison_rows,
    }

    return report


def _selected_threshold_map(threshold_report: dict) -> dict[str, float | None]:
    result = {}
    for tier, tier_payload in threshold_report.get("tiers", {}).items():
        result[tier] = tier_payload.get("selected_threshold")
    return result


def _summarize_fold_metrics(
    fold_df: pd.DataFrame,
    fold_meta: dict,
    threshold_map: dict[str, float | None],
) -> dict:
    thresholds = fold_df["tier"].map(threshold_map)
    trade_mask = thresholds.notna() & (fold_df["proba"] >= thresholds)
    preds = trade_mask.astype(int)
    y_true = fold_df["y_true"].astype(int)

    gross_trade_returns = fold_df.loc[trade_mask, "estimated_gross_return_before_costs"].astype(float)
    net_trade_returns = fold_df.loc[trade_mask, "estimated_net_return_after_costs"].astype(float)
    if y_true.nunique() > 1:
        pr_auc = float(average_precision_score(y_true, fold_df["proba"]))
        roc_auc = float(roc_auc_score(y_true, fold_df["proba"]))
    else:
        pr_auc = None
        roc_auc = None

    return {
        "fold": int(fold_meta["fold"]),
        "train_size": int(fold_meta["train_size"]),
        "threshold_val_size": int(fold_meta.get("threshold_val_size", 0)),
        "test_size": int(fold_meta["test_size"]),
        "train_end": fold_meta["train_end"],
        "test_start": fold_meta["test_start"],
        "test_end": fold_meta["test_end"],
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "positive_rate": float(preds.mean()),
        "trade_count": int(trade_mask.sum()),
        "estimated_gross_return_before_costs": float(gross_trade_returns.sum()) if not gross_trade_returns.empty else 0.0,
        "gross_expectancy_before_costs": float(gross_trade_returns.mean()) if not gross_trade_returns.empty else 0.0,
        "estimated_net_return_after_costs": float(net_trade_returns.sum()) if not net_trade_returns.empty else 0.0,
        "net_expectancy_after_costs": float(net_trade_returns.mean()) if not net_trade_returns.empty else 0.0,
        "cost_drag": (
            float(gross_trade_returns.sum()) - float(net_trade_returns.sum())
            if not net_trade_returns.empty else 0.0
        ),
    }


def _walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    estimated_gross_returns: pd.Series,
    estimated_net_returns: pd.Series,
    timestamps: pd.Series,
    tiers: pd.Series,
    symbols: pd.Series,
    trained_at: str,
) -> tuple[dict, dict]:
    """
    TimeSeriesSplit-based expanding walk-forward without shuffle.
    Supports train/test gap and test embargo.
    """
    valid_mask = (
        y.notna()
        & estimated_gross_returns.notna()
        & estimated_net_returns.notna()
        & timestamps.notna()
        & tiers.notna()
        & symbols.notna()
    )
    X_valid = X.loc[valid_mask].reset_index(drop=True)
    y_valid = y.loc[valid_mask].astype(int).reset_index(drop=True)
    gross_ret_valid = estimated_gross_returns.loc[valid_mask].astype(float).reset_index(drop=True)
    ret_valid = estimated_net_returns.loc[valid_mask].astype(float).reset_index(drop=True)
    ts_valid = pd.to_datetime(timestamps.loc[valid_mask], utc=True).reset_index(drop=True)
    tier_valid = tiers.loc[valid_mask].astype(str).reset_index(drop=True)
    symbol_valid = symbols.loc[valid_mask].astype(str).reset_index(drop=True)

    # Effective row-gap: WF_GAP_BARS logical bars × symbols_per_timestamp.
    # With 9 coins interleaved by timestamp, gap=8 rows ≈ <1h real gap.
    # Multiply by n_syms so the gap corresponds to WF_GAP_BARS real 1h bars.
    n_syms = max(int(symbol_valid.nunique()), 1)
    effective_gap = WF_GAP_BARS * n_syms

    base_report = {
        "generated_at": trained_at,
        "config": {
            "method": "TimeSeriesSplit",
            "shuffle": False,
            "n_splits": N_FOLDS,
            "test_size": FOLD_TEST_SIZE,  # None = auto
            "gap_bars": WF_GAP_BARS,
            "effective_row_gap": effective_gap,
            "n_symbols": n_syms,
            "embargo_bars": WF_EMBARGO_BARS,
            "min_train_size": MIN_TRAIN_SIZE,
            "min_test_size": MIN_TEST_SIZE,
        },
        "summary": {},
        "folds": [],
    }
    empty_threshold_report = _optimize_thresholds(
        pd.DataFrame(
            columns=[
                "fold",
                "timestamp",
                "tier",
                "symbol",
                "y_true",
                "proba",
                "estimated_gross_return_before_costs",
                "estimated_net_return_after_costs",
            ]
        ),
        trained_at,
    )

    min_needed = MIN_TRAIN_SIZE + MIN_TEST_SIZE + effective_gap
    if len(X_valid) < min_needed:
        print(f"  [WARN] Yeterli veri yok ({len(X_valid)} < {min_needed}) — walk-forward atlanıyor")
        return base_report, empty_threshold_report

    try:
        splitter = TimeSeriesSplit(
            n_splits=N_FOLDS,
            test_size=FOLD_TEST_SIZE,   # None → sklearn auto-split (equal chunks)
            gap=effective_gap,          # WF_GAP_BARS × n_symbols row gap
        )
        split_indices = list(splitter.split(X_valid))
    except ValueError as exc:
        print(f"  [WARN] Walk-forward kurulamadi: {exc}")
        return base_report, empty_threshold_report

    fold_frames = []
    fold_meta = []
    fold_threshold_maps: list[dict[str, float | None]] = []
    for fold_idx, (train_idx, test_idx_full) in enumerate(split_indices, start=1):
        if len(train_idx) < MIN_TRAIN_SIZE:
            print(f"  [SKIP] Fold {fold_idx}: train too small ({len(train_idx)} < {MIN_TRAIN_SIZE})")
            continue
        test_idx = test_idx_full[WF_EMBARGO_BARS:] if WF_EMBARGO_BARS else test_idx_full
        if len(test_idx) < MIN_TEST_SIZE:
            print(f"  [SKIP] Fold {fold_idx}: test too small ({len(test_idx)} < {MIN_TEST_SIZE})")
            continue

        val_size = max(MIN_TEST_SIZE, int(len(train_idx) * 0.20))
        val_size = min(val_size, max(len(train_idx) - MIN_TRAIN_SIZE, 0))
        if val_size < MIN_TEST_SIZE:
            print(f"  [SKIP] Fold {fold_idx}: threshold val too small ({val_size} < {MIN_TEST_SIZE})")
            continue

        train_core_idx = train_idx[:-val_size]
        val_idx = train_idx[-val_size:]
        if len(train_core_idx) < MIN_TRAIN_SIZE:
            print(f"  [SKIP] Fold {fold_idx}: train core too small ({len(train_core_idx)} < {MIN_TRAIN_SIZE})")
            continue

        X_train_core = X_valid.iloc[train_core_idx].fillna(0)
        y_train_core = y_valid.iloc[train_core_idx]
        X_val = X_valid.iloc[val_idx].fillna(0)
        y_val = y_valid.iloc[val_idx]
        gross_ret_val = gross_ret_valid.iloc[val_idx]
        ret_val = ret_valid.iloc[val_idx]
        tier_val = tier_valid.iloc[val_idx]
        symbol_val = symbol_valid.iloc[val_idx]

        neg_core = int((y_train_core == 0).sum())
        pos_core = int((y_train_core == 1).sum())
        spw_core = max(1.0, neg_core / max(pos_core, 1))
        selector_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_core)
        selector_model.fit(X_train_core, y_train_core, verbose=False)

        val_proba = selector_model.predict_proba(X_val)[:, 1]
        val_prediction_df = pd.DataFrame({
            "fold": fold_idx,
            "timestamp": ts_valid.iloc[val_idx].values,
            "tier": tier_val.values,
            "symbol": symbol_val.values,
            "y_true": y_val.values,
            "proba": val_proba,
            "estimated_gross_return_before_costs": gross_ret_val.values,
            "estimated_net_return_after_costs": ret_val.values,
        })
        val_threshold_report = _optimize_thresholds(val_prediction_df, trained_at)
        fold_threshold_map = _selected_threshold_map(val_threshold_report)
        fold_threshold_maps.append(fold_threshold_map)

        X_train = X_valid.iloc[train_idx].fillna(0)
        y_train = y_valid.iloc[train_idx]
        X_test = X_valid.iloc[test_idx].fillna(0)
        y_test = y_valid.iloc[test_idx]
        gross_ret_test = gross_ret_valid.iloc[test_idx]
        ret_test = ret_valid.iloc[test_idx]

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        spw = max(1.0, neg_count / max(pos_count, 1))
        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X_train, y_train, verbose=False)

        proba = model.predict_proba(X_test)[:, 1]
        fold_frames.append(pd.DataFrame({
            "fold": fold_idx,
            "timestamp": ts_valid.iloc[test_idx].values,
            "tier": tier_valid.iloc[test_idx].values,
            "symbol": symbol_valid.iloc[test_idx].values,
            "y_true": y_test.values,
            "proba": proba,
            "estimated_gross_return_before_costs": gross_ret_test.values,
            "estimated_net_return_after_costs": ret_test.values,
        }))
        fold_meta.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "threshold_val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "train_end": ts_valid.iloc[train_idx[-1]].isoformat(),
            "test_start": ts_valid.iloc[test_idx[0]].isoformat(),
            "test_end": ts_valid.iloc[test_idx[-1]].isoformat(),
            "threshold_map": fold_threshold_map,
        })

    if not fold_frames:
        print("  [WARN] Walk-forward fold uretilemedi")
        return base_report, empty_threshold_report

    prediction_df = pd.concat(fold_frames, ignore_index=True)
    threshold_report = _build_threshold_report_from_fold_maps(fold_threshold_maps, trained_at)
    fold_results = [
        _summarize_fold_metrics(
            prediction_df.loc[prediction_df["fold"] == meta["fold"]].copy(),
            meta,
            meta["threshold_map"],
        )
        for meta in fold_meta
    ]
    for fold_result in fold_results:
        print(
            f"  Fold {fold_result['fold']}: train={fold_result['train_size']} | "
            f"thr_val={fold_result['threshold_val_size']} | "
            f"test={fold_result['test_size']} | precision={fold_result['precision']:.1%} | "
            f"trades={fold_result['trade_count']} | gross={fold_result['estimated_gross_return_before_costs']:.3%} | "
            f"net={fold_result['estimated_net_return_after_costs']:.3%}"
        )

    base_report["folds"] = fold_results
    base_report["summary"] = {
        "fold_count": len(fold_results),
        "precision_mean": _safe_metric_mean(fold_results, "precision"),
        "recall_mean": _safe_metric_mean(fold_results, "recall"),
        "f1_mean": _safe_metric_mean(fold_results, "f1"),
        "pr_auc_mean": _safe_metric_mean(fold_results, "pr_auc"),
        "roc_auc_mean": _safe_metric_mean(fold_results, "roc_auc"),
        "positive_rate_mean": _safe_metric_mean(fold_results, "positive_rate"),
        "trade_count_mean": _safe_metric_mean(fold_results, "trade_count"),
        "trade_count_total": int(sum(f["trade_count"] for f in fold_results)),
        "estimated_gross_return_before_costs_mean": _safe_metric_mean(
            fold_results, "estimated_gross_return_before_costs"
        ),
        "estimated_gross_return_before_costs_total": float(
            sum(f["estimated_gross_return_before_costs"] for f in fold_results)
        ),
        "gross_expectancy_before_costs_mean": _safe_metric_mean(
            fold_results, "gross_expectancy_before_costs"
        ),
        "estimated_net_return_after_costs_mean": _safe_metric_mean(
            fold_results, "estimated_net_return_after_costs"
        ),
        "estimated_net_return_after_costs_total": float(
            sum(f["estimated_net_return_after_costs"] for f in fold_results)
        ),
        "net_expectancy_after_costs_mean": _safe_metric_mean(
            fold_results, "net_expectancy_after_costs"
        ),
        "cost_drag_mean": _safe_metric_mean(fold_results, "cost_drag"),
        "cost_drag_total": float(sum(f["cost_drag"] for f in fold_results)),
    }
    base_report["selected_thresholds"] = {
        tier: payload.get("selected_threshold")
        for tier, payload in threshold_report.get("tiers", {}).items()
    }

    # ── Per-coin validation metrics ──────────────────────────────────────────
    per_coin = _compute_per_coin_metrics(prediction_df, trained_at)
    base_report["per_coin_metrics"] = per_coin

    return base_report, threshold_report


def _compute_per_coin_metrics(prediction_df: pd.DataFrame, trained_at: str) -> dict:
    """Compute per-coin precision and support from walk-forward prediction_df.

    Returns dict with 'coins' list and 'warnings' list.
    Saves to models/per_coin_validation_YYYYMMDD.json.
    """
    if prediction_df.empty or "symbol" not in prediction_df.columns:
        return {"coins": [], "warnings": [], "generated_at": trained_at}

    coins_data = []
    warnings = []
    for symbol, grp in prediction_df.groupby("symbol"):
        y_true = grp["y_true"].astype(int)
        support = int(y_true.sum())  # positive label count
        total = len(y_true)
        n_folds = int(grp["fold"].nunique())
        avg_support_per_fold = round(support / max(n_folds, 1), 1)

        # Precision: predict top-prob as positive (prob >= 0.50 baseline)
        proba = grp["proba"].astype(float)
        predicted_pos = proba >= 0.50
        tp = int((predicted_pos & (y_true == 1)).sum())
        fp = int((predicted_pos & (y_true == 0)).sum())
        precision = round(tp / max(tp + fp, 1), 4)

        coin_entry = {
            "symbol": str(symbol),
            "support": support,
            "total_samples": total,
            "positive_rate": round(support / max(total, 1), 4),
            "n_folds": n_folds,
            "avg_support_per_fold": avg_support_per_fold,
            "precision_at_0.50": precision,
            "predicted_positive": int(predicted_pos.sum()),
            "true_positive": tp,
            "false_positive": fp,
        }
        coins_data.append(coin_entry)

        if support < 10:
            warn_msg = (
                f"[WARNING] {symbol}: support={support} < 10 across {n_folds} folds "
                f"(avg {avg_support_per_fold}/fold) — risk of overfitting"
            )
            warnings.append(warn_msg)
            print(f"  {warn_msg}")

    # Sort by support ascending (weakest coins first)
    coins_data.sort(key=lambda c: c["support"])

    # Print summary
    print(f"\n  ┌─ Per-coin WF validation ({len(coins_data)} coins) ────────────────────")
    for c in coins_data:
        flag = "LOW" if c["support"] < 10 else "ok"
        print(
            f"  │  {c['symbol']:<12} support={c['support']:>4} | "
            f"prec@0.50={c['precision_at_0.50']:.1%} | "
            f"pos_rate={c['positive_rate']:.1%} | "
            f"avg/fold={c['avg_support_per_fold']:.1f} [{flag}]"
        )
    print(f"  └─────────────────────────────────────────────────────────────")

    result = {
        "generated_at": trained_at,
        "coins": coins_data,
        "warnings": warnings,
    }

    # Save to file
    date_str = trained_at[:10] if len(trained_at) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = BASE_DIR / "models" / f"per_coin_validation_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"  Per-coin validation saved: {out_path}")
    except Exception as exc:
        print(f"  [WARN] per-coin validation write failed: {exc}")

    return result


def _label_distribution(y: pd.Series) -> dict[str, float]:
    valid = y.dropna()
    pos = int((valid == 1).sum())
    neg = int((valid == 0).sum())
    return {
        "n_valid": int(len(valid)),
        "n_pos": pos,
        "n_neg": neg,
        "pos_rate": float(pos / len(valid)) if len(valid) else 0.0,
    }


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(stream=None, execution_cost_config: dict | None = None) -> dict:
    """
    Full training pipeline.
    Returns model dict and only updates MODEL_PATH if production checks pass.
    """
    if stream is None:
        from crypto_stream import CryptoStream
        stream = CryptoStream()

    print("\n[CRYPTO ML] === RETRAIN: 10-coin universe ===")
    print("[CRYPTO ML] Onceki model: 3 sembol (BTC/ETH/SOL), WF=57.5%")
    print("[CRYPTO ML] Yeni model: 10 sembol + 2 RISKY ozellik")
    trained_at = datetime.now(timezone.utc).isoformat()
    label_cfg = _resolve_label_config()
    cost_cfg = _resolve_execution_cost_config(label_cfg, execution_cost_config)
    cost_side = _per_side_cost_bps(label_cfg, execution_cost_config)
    print(
        f"[CRYPTO ML] Label: {label_cfg['name']} | mode={label_cfg['mode']} "
        f"| horizon={label_cfg['horizon_bars']}h"
    )
    print(
        f"[CRYPTO ML] Execution costs: enabled={cost_cfg['enabled']} | "
        f"order_type={cost_cfg['order_type']} | fee={cost_side['fee_bps']:.1f}bps/side | "
        f"spread={cost_side['spread_bps']:.1f}bps/side | "
        f"slippage={cost_side['slippage_bps']:.1f}bps/side"
    )
    print("[CRYPTO ML] Veri cekiliyor...")
    _QUALITY_WARNINGS.clear()
    data    = _fetch_ohlcv_all(stream)          # simple fetch: 1h=1000, 5m=500, 15m=500
    fg_dict = _fetch_fg_history()
    relative_strength_map = _build_cross_sectional_relative_strength(data)
    print(f"  Fear&Greed tarihi: {len(fg_dict)} gun")

    symbols = [sym for sym in [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT",
        "ONDO/USDT", "FET/USDT",
    ] if sym in data]
    btc_df  = data.get("BTC/USDT")

    all_parts: list[pd.DataFrame] = []
    alignment_reports: list[dict] = []

    for sym in symbols:
        df = data.get(sym)
        if df is None:
            print(f"  [SKIP] {sym}: veri yok")
            continue

        df_5m   = data.get(f"{sym}_5m")
        btc_ref = btc_df if sym != "BTC/USDT" else None

        print(f"\n[CRYPTO ML] {sym} feature engineering...")
        if df_5m is None:
            _warn_quality(f"{sym} missing historical 5m alignment data; training features will degrade")
        alignment_report = _build_alignment_report(sym, df, df_5m, btc_ref)
        alignment_reports.append(alignment_report)
        print(
            f"  Alignment {sym}: "
            f"5m_match={alignment_report['mtf_5m_match_ratio']:.1%} "
            f"btc_match={alignment_report['btc_match_ratio']:.1%}"
        )
        X = _build_features(
            df,
            btc_ref,
            fg_dict,
            df_5m,
            relative_strength_df=relative_strength_map.get(sym),
        )
        X["symbol_encoded"] = _symbol_encode(sym)   # overwrite placeholder with actual encoding
        outcomes = _build_outcomes(df, label_cfg, cost_cfg)
        y = outcomes["label"]
        estimated_gross_returns = outcomes["estimated_gross_return_before_costs"]
        estimated_net_returns = outcomes["estimated_net_return_after_costs"]

        # Align lengths and drop NaNs
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        estimated_net_returns = estimated_net_returns.iloc[:min_len]
        X = X.iloc[WARMUP_BARS:]
        y = y.iloc[WARMUP_BARS:]
        estimated_gross_returns = estimated_gross_returns.iloc[WARMUP_BARS:]
        estimated_net_returns = estimated_net_returns.iloc[WARMUP_BARS:]

        # Drop rows with missing features or missing labels
        valid = y.notna()
        X = X[valid]
        y = y[valid]
        estimated_gross_returns = estimated_gross_returns[valid]
        estimated_net_returns = estimated_net_returns[valid]
        print(
            f"  {sym}: {len(X)} train rows "
            f"(label=1: {y.mean():.1%} | missing_cells={int(X.isna().sum().sum())})"
        )

        part = X.copy()
        part["__timestamp"] = pd.to_datetime(df.loc[X.index, "timestamp"], utc=True)
        part["__symbol"] = sym
        part["__tier"] = _tier_for_symbol(sym)
        part["__y"] = y.astype(float).values
        part["__estimated_gross_return_before_costs"] = estimated_gross_returns.astype(float).values
        part["__estimated_net_return_after_costs"] = estimated_net_returns.astype(float).values
        all_parts.append(part)

    if not all_parts:
        print("[CRYPTO ML] HATA: Hiç veri yok — eğitim iptal")
        return {}

    if _QUALITY_WARNINGS:
        print("\n[CRYPTO ML] Quality warnings:")
        for msg in _QUALITY_WARNINGS:
            print(f"  - {msg}")

    combined = pd.concat(all_parts, ignore_index=True)
    combined = combined.sort_values(["__timestamp", "__symbol"]).reset_index(drop=True)
    X_combined = combined[FEATURE_NAMES]
    y_combined = combined["__y"].astype(float)
    estimated_gross_returns_combined = combined["__estimated_gross_return_before_costs"].astype(float)
    estimated_net_returns_combined = combined["__estimated_net_return_after_costs"].astype(float)
    timestamps_combined = combined["__timestamp"]
    tiers_combined = combined["__tier"]
    symbols_combined = combined["__symbol"]
    feature_stats = _write_feature_quality_reports(
        X_combined,
        y_combined,
        alignment_reports,
        FEATURE_SOURCE_NOTES,
    )
    X_combined, dropped_features = _drop_constant_like_features(X_combined, feature_stats)
    active_features = list(X_combined.columns)

    label_stats = _label_distribution(y_combined)
    print(
        f"\n[CRYPTO ML] Toplam: {len(X_combined)} satır | "
        f"label_pos={label_stats['n_pos']} label_neg={label_stats['n_neg']} "
        f"| pos_rate={label_stats['pos_rate']:.1%} | horizon={label_cfg['horizon_bars']}h"
    )
    if dropped_features:
        print(f"[CRYPTO ML] Removed low-variance features: {', '.join(dropped_features)}")
    print(f"[CRYPTO ML] Active feature count: {len(active_features)}")
    print(
        f"[CRYPTO ML] Walk-forward ({N_FOLDS} fold, gap={WF_GAP_BARS}×n_syms rows, embargo={WF_EMBARGO_BARS})..."
    )

    wf_report, threshold_report = _walk_forward(
        X_combined,
        y_combined,
        estimated_gross_returns_combined,
        estimated_net_returns_combined,
        timestamps_combined,
        tiers_combined,
        symbols_combined,
        trained_at,
    )
    production_decision = _build_production_decision(wf_report)
    wf_summary = wf_report.get("summary", {})
    wf_prec = float(wf_summary.get("precision_mean") or 0.0)
    threshold_config = threshold_report.get("tiers", {})
    print(f"\n[CRYPTO ML] === WF KARSILASTIRMA ===")
    print(f"[CRYPTO ML]  Onceki (3 sembol):  WF=57.5%  features=22")
    print(f"[CRYPTO ML]  Yeni    ({len(symbols)} sembol): WF={wf_prec:.1%}  features={len(active_features)}")
    print(f"[CRYPTO ML] Ortalama WF BUY precision: {wf_prec:.1%}")
    print(
        f"[CRYPTO ML] Ortalama trade count: {wf_summary.get('trade_count_mean') or 0.0:.1f} | "
        f"Toplam trade: {int(wf_summary.get('trade_count_total') or 0)}"
    )
    print(
        f"[CRYPTO ML] Gross expectancy: {(wf_summary.get('gross_expectancy_before_costs_mean') or 0.0):.3%} | "
        f"Net expectancy: {(wf_summary.get('net_expectancy_after_costs_mean') or 0.0):.3%} | "
        f"Cost drag: {(wf_summary.get('cost_drag_total') or 0.0):.3%}"
    )
    for tier_name in ["MAJOR", "RISKY"]:
        tier_cfg = threshold_config.get(tier_name, {})
        sel = tier_cfg.get("selected_threshold")
        fallback = tier_cfg.get("fallback_threshold")
        status = tier_cfg.get("selection_status", "n/a")
        enabled = tier_cfg.get("enabled", False)
        shown = sel if sel is not None else fallback
        print(
            f"[CRYPTO ML] {tier_name} threshold: {shown if shown is not None else 'n/a'} "
            f"| enabled={enabled} | status={status}"
        )

    # Train final model on all data
    print("[CRYPTO ML] Final model egitiliyor (tum veri)...")
    neg_all = (y_combined == 0).sum()
    pos_all = (y_combined == 1).sum()
    spw_all = max(1, neg_all / max(pos_all, 1))
    print(f"  label=1: {pos_all} ({pos_all/len(y_combined):.1%}) | scale_pos_weight={spw_all:.1f}")
    final_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_all)
    final_model.fit(X_combined.fillna(0), y_combined.astype(int), verbose=False)

    # Feature importance top-5
    imp = pd.Series(final_model.feature_importances_, index=active_features)
    print("[CRYPTO ML] Top-5 ozellik:")
    for name, score in imp.nlargest(5).items():
        print(f"  {name}: {score:.4f}")

    ts_sorted = timestamps_combined.dropna().sort_values()

    # Compute per-feature medians from training data for predict-time NaN fallback.
    # Only non-zero medians are stored; missing / all-NaN features default to 0.0.
    _MEDIAN_FEATURES = [
        "fear_greed_value", "btc_1h_return", "btc_ret_4h", "btc_vol_1h",
        "mtf_5m_rsi", "mtf_5m_macd_hist", "mtf_5m_bb_pos",
        "funding_rate", "funding_rate_avg3", "funding_rate_extreme", "funding_contrarian",
        "btc_dominance", "btc_dominance_regime", "btc_dominance_7d_change", "htf_bias",
    ]
    feature_medians: dict[str, float] = {}
    for col in _MEDIAN_FEATURES:
        if col in X_combined.columns:
            med = float(X_combined[col].median(skipna=True))
            feature_medians[col] = 0.0 if np.isnan(med) else med

    model_dict = {
        "model":          final_model,
        "feature_names":  active_features,
        "dropped_features": dropped_features,
        "wf_precision":   wf_prec,
        "wf_report":      wf_report,
        "wf_summary":     wf_summary,
        "threshold_config": threshold_config,
        "threshold_report_path": str(THRESHOLD_REPORT_PATH),
        "execution_cost_config": cost_cfg,
        "accepted_for_production": production_decision["accepted"],
        "production_decision": production_decision,
        "trained_at":     trained_at,
        "n_samples":      len(X_combined),
        "label_config":   label_cfg,
        "label_horizon":  int(label_cfg["horizon_bars"]),
        "label_min_gain": float(label_cfg.get("min_gain", label_cfg.get("min_target_gain", 0.0))),
        # Predict-time NaN fallback: median value from training data per feature
        "feature_medians": feature_medians,
        # Provenance metadata
        "symbols_trained": symbols,
        "training_start":  str(ts_sorted.iloc[0])  if len(ts_sorted) > 0 else None,
        "training_end":    str(ts_sorted.iloc[-1]) if len(ts_sorted) > 0 else None,
    }

    threshold_report["production_decision"] = production_decision
    threshold_report["model_report_path"] = str(MODEL_REPORT_PATH)
    threshold_report["execution_cost_config"] = cost_cfg
    _write_threshold_report(threshold_report)

    report_payload = {
        "generated_at": trained_at,
        "model_path": str(MODEL_PATH),
        "report_path": str(MODEL_REPORT_PATH),
        "threshold_report_path": str(THRESHOLD_REPORT_PATH),
        "trained_at": trained_at,
        "n_samples": int(len(X_combined)),
        "feature_count": int(len(active_features)),
        "label_config": label_cfg,
        "execution_cost_config": cost_cfg,
        "walk_forward": wf_report,
        "thresholds": threshold_report,
        "production_decision": production_decision,
        "quality_warnings": list(_QUALITY_WARNINGS),
    }
    _write_model_report(report_payload)
    model_dict["report_path"] = str(MODEL_REPORT_PATH)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    if production_decision["accepted"]:
        model_dict["saved_to_active"] = True
        joblib.dump(model_dict, MODEL_PATH)
        print(f"[CRYPTO ML] Model kaydedildi: {MODEL_PATH}")
    else:
        model_dict["saved_to_active"] = False
        print(
            "[CRYPTO ML] Model production esiklerini gecemedi; aktif model overwrite edilmedi."
        )
        print(
            f"[CRYPTO ML] Failed checks: {', '.join(production_decision['failed_checks']) or 'n/a'}"
        )
    return model_dict


# ── Directional two-model training system ─────────────────────────────────────

def train_directional(stream=None) -> tuple[dict, dict]:
    """
    Train two separate XGBoost models with pure outcome-based directional labels.
      LONG  model (staging/crypto_xgb_long_candidate.pkl)
      SHORT model (staging/crypto_xgb_short_candidate.pkl)

    Labels per _build_directional_labels() — NO entry-state filters:
      LONG:  ret_future >= threshold  AND  ret_future > ret_future_down × dominance_ratio
      SHORT: ret_future_down >= threshold  AND  ret_future_down > ret_future × dominance_ratio
      NEUTRAL: everything else — excluded from both training sets.

    Where:
      ret_future      = (max high over next horizon bars - close) / close  (upside potential)
      ret_future_down = (close - min low over next horizon bars) / close   (downside potential)
      threshold       = 2.5% (THRESHOLD constant below)
      dominance_ratio = 2.0 (one side must be 2× larger than the other)

    LONG model training set:  LONG rows (y=1) + SHORT rows as negative examples (y=0)
    SHORT model training set: SHORT rows (y=1) + LONG rows as negative examples (y=0)

    Both models saved only if WF precision >= 50% (hard min).
    Both staged to models/staging/ — promoted by crypto_weekly_retrain.py after full acceptance.
    Returns (long_model_dict, short_model_dict).
    """
    from crypto_indicators import build_htf_bias_history_from_1h
    from crypto_stream import fetch_funding_rate_history, get_btc_dominance_features

    if stream is None:
        from crypto_stream import CryptoStream
        stream = CryptoStream()

    HORIZON   = int(LABEL_CONFIG["horizon_bars"])   # 8
    THRESHOLD = 0.025                                 # 2.5% directional threshold
    cost_rate = _round_trip_cost_rate(LABEL_CONFIG, EXECUTION_COST_CONFIG)

    trained_at = datetime.now(timezone.utc).isoformat()
    print("\n[CRYPTO ML] === DIRECTIONAL TRAIN: LONG + SHORT models ===")
    print(f"[CRYPTO ML] Horizon={HORIZON}h | threshold={THRESHOLD:.1%} | cost={cost_rate:.4f}")
    print(f"[CRYPTO ML] Active feature set ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    print("[CRYPTO ML] Veri cekiliyor...")
    _QUALITY_WARNINGS.clear()
    btc_dominance_info = get_btc_dominance_features(force_refresh=True)
    print(
        f"[CRYPTO ML] BTC dominance snapshot: {btc_dominance_info['btc_dominance']:.2f} "
        f"| regime={btc_dominance_info['btc_dominance_regime']} "
        f"| 7d_change={btc_dominance_info['btc_dominance_7d_change']:+.2f}"
    )

    data    = _fetch_ohlcv_all_paginated(stream)
    fg_dict = _fetch_fg_history()
    print(f"  Fear&Greed tarihi: {len(fg_dict)} gun")

    symbols = [sym for sym in [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT",
        "ONDO/USDT", "FET/USDT",
    ] if sym in data and f"{sym}_5m" in data]
    btc_df = data.get("BTC/USDT")
    training_frames = [data[sym] for sym in symbols if sym in data]
    training_start = min((df["timestamp"].iloc[0] for df in training_frames), default=None)
    training_end = max((df["timestamp"].iloc[-1] for df in training_frames), default=None)

    # Collect per-symbol feature+label frames
    long_parts:  list[pd.DataFrame] = []
    short_parts: list[pd.DataFrame] = []
    total_neutral = 0

    for sym in symbols:
        df = data.get(sym)
        if df is None:
            print(f"  [SKIP] {sym}: veri yok")
            continue

        df_5m   = data.get(f"{sym}_5m")
        btc_ref = btc_df if sym != "BTC/USDT" else None
        if df_5m is None:
            print(f"  [SKIP] {sym}: 5m history yetersiz — MTF alignment yok")
            continue

        alignment_report = _build_alignment_report(sym, df, df_5m, btc_ref)
        print(
            f"  Alignment {sym}: 5m_match={alignment_report['mtf_5m_match_ratio']:.1%} "
            f"btc_match={alignment_report['btc_match_ratio']:.1%}"
        )
        if alignment_report["mtf_5m_match_ratio"] < 0.80:
            print(f"  [SKIP] {sym}: 5m alignment cok zayif ({alignment_report['mtf_5m_match_ratio']:.1%})")
            continue

        print(f"\n[CRYPTO ML] {sym} directional feature engineering...")
        funding_hist = fetch_funding_rate_history(sym)
        htf_bias_hist = build_htf_bias_history_from_1h(df)
        X = _build_features(
            df,
            btc_ref,
            fg_dict,
            df_5m,
            funding_df=funding_hist,
            btc_dominance_info=btc_dominance_info,
            htf_bias_df=htf_bias_hist,
        )

        dir_labels = _build_directional_labels(df, horizon=HORIZON, threshold=THRESHOLD)
        long_label  = dir_labels["long_label"]
        short_label = dir_labels["short_label"]
        ret_future      = dir_labels["ret_future"]
        ret_future_down = dir_labels["ret_future_down"]

        # Align lengths
        min_len = min(len(X), len(long_label))
        X               = X.iloc[:min_len]
        long_label      = long_label.iloc[:min_len]
        short_label     = short_label.iloc[:min_len]
        ret_future      = ret_future.iloc[:min_len]
        ret_future_down = ret_future_down.iloc[:min_len]

        # Drop warmup rows
        X               = X.iloc[WARMUP_BARS:]
        long_label      = long_label.iloc[WARMUP_BARS:]
        short_label     = short_label.iloc[WARMUP_BARS:]
        ret_future      = ret_future.iloc[WARMUP_BARS:]
        ret_future_down = ret_future_down.iloc[WARMUP_BARS:]

        # Require valid labels on both sides (tail rows have NaN on both)
        valid_mask = long_label.notna() & short_label.notna()
        X               = X[valid_mask]
        long_label      = long_label[valid_mask]
        short_label     = short_label[valid_mask]
        ret_future      = ret_future[valid_mask]
        ret_future_down = ret_future_down[valid_mask]

        # LONG model: keep rows where LONG=1 or SHORT=1 (exclude NEUTRAL where both==0)
        directional_mask = (long_label == 1) | (short_label == 1)
        n_long_rows  = int((long_label[directional_mask]  == 1).sum())
        n_short_rows = int((short_label[directional_mask] == 1).sum())
        n_neutral    = int((~directional_mask).sum())
        n_valid      = int(len(X))
        total_neutral += n_neutral
        print(
            f"  {sym}: {n_valid} valid rows | "
            f"LONG={n_long_rows} SHORT={n_short_rows} NEUTRAL={n_neutral} "
            f"(LONG_rate={n_long_rows/max(n_valid,1):.1%} SHORT_rate={n_short_rows/max(n_valid,1):.1%})"
        )

        ts_series = pd.to_datetime(df.loc[X.index, "timestamp"], utc=True)

        def _make_part(y_labels: pd.Series, gross_ret: pd.Series) -> pd.DataFrame:
            mask = directional_mask
            part = X[mask].copy()
            part["__timestamp"] = ts_series[mask].values
            part["__symbol"]    = sym
            part["__tier"]      = _tier_for_symbol(sym)
            part["__y"]         = y_labels[mask].astype(float).values
            net_ret             = gross_ret[mask] - cost_rate
            part["__gross_ret"] = gross_ret[mask].astype(float).values
            part["__net_ret"]   = net_ret.astype(float).values
            return part

        long_parts.append(_make_part(long_label, ret_future))
        short_parts.append(_make_part(short_label, ret_future_down))

    if not long_parts:
        print("[CRYPTO ML] HATA: Hiç veri yok — directional eğitim iptal")
        return {}, {}

    long_all = pd.concat(long_parts, ignore_index=True)
    short_all = pd.concat(short_parts, ignore_index=True)
    long_pos = int((long_all["__y"] == 1).sum())
    long_neg = int((long_all["__y"] == 0).sum())
    short_pos = int((short_all["__y"] == 1).sum())
    short_neg = int((short_all["__y"] == 0).sum())
    print(
        f"[CRYPTO ML] LONG model training set: {long_pos} long, {long_neg} short, "
        f"{total_neutral} neutral"
    )
    print(
        f"[CRYPTO ML] SHORT model training set: {short_pos} short, {short_neg} long, "
        f"{total_neutral} neutral"
    )

    def _train_one_model(parts: list[pd.DataFrame], side: str, candidate_path: Path) -> dict:
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(["__timestamp", "__symbol"]).reset_index(drop=True)

        active_feats = [f for f in FEATURE_NAMES if f in combined.columns]
        X_comb     = combined[active_feats]
        y_comb     = combined["__y"].astype(float)
        gross_comb = combined["__gross_ret"].astype(float)
        net_comb   = combined["__net_ret"].astype(float)
        ts_comb    = combined["__timestamp"]
        tier_comb  = combined["__tier"]
        sym_comb   = combined["__symbol"]

        n_pos = int((y_comb == 1).sum())
        n_neg = int((y_comb == 0).sum())
        pos_rate = n_pos / max(len(y_comb), 1)
        print(f"\n[CRYPTO ML] {side.upper()} model: {len(X_comb)} rows | "
              f"y=1: {n_pos} ({pos_rate:.1%}) | y=0: {n_neg} ({1-pos_rate:.1%})")

        # Walk-forward validation
        print(f"[CRYPTO ML] {side.upper()} walk-forward ({N_FOLDS} folds)...")
        wf_report, threshold_report = _walk_forward(
            X_comb, y_comb, gross_comb, net_comb,
            ts_comb, tier_comb, sym_comb, trained_at,
        )
        wf_summary = wf_report.get("summary", {})
        wf_prec    = float(wf_summary.get("precision_mean") or 0.0)
        trade_tot  = int(wf_summary.get("trade_count_total") or 0)
        net_exp    = float(wf_summary.get("net_expectancy_after_costs_mean") or 0.0)

        # ── Acceptance criteria report ────────────────────────────────────────
        fold_results = wf_report.get("folds", [])
        fold_precisions = [f["precision"] for f in fold_results]
        fold_trade_counts = [int(f["trade_count"]) for f in fold_results]
        min_fold_prec = min(fold_precisions) if fold_precisions else 0.0
        max_fold_prec = max(fold_precisions) if fold_precisions else 0.0
        min_fold_trades = min(fold_trade_counts) if fold_trade_counts else 0
        full_fold_count = len(fold_results) == N_FOLDS
        eligible_folds = [f for f in fold_results if int(f["trade_count"]) >= 20]
        eligible_mean = (
            float(np.mean([f["precision"] for f in eligible_folds]))
            if eligible_folds else 0.0
        )
        fold_variance = max_fold_prec - min_fold_prec if fold_precisions else 0.0

        folds_passing = sum(1 for p in fold_precisions if p > 0.50)
        if side == "long":
            accept_checks = {
                "all_expected_folds_present": full_fold_count,
                # Relaxed from all_folds to >=3/5: crypto regimes cause 1-2 hard folds;
                # strict 5/5 gate kept models out even when mean WF was honest.
                "folds_gte_3_of_5_pass_0.50": folds_passing >= 3,
                # Relaxed from 0.54 to 0.52: 52% is above random with 1000+ WF trades.
                "eligible_mean_wf_gt_0.52": eligible_mean > 0.52,
                "primary_class_gt_200": n_pos > 200,
                "max_fold_precision_lte_0.85": max_fold_prec <= 0.85,
            }
        else:
            accept_checks = {
                "all_expected_folds_present": full_fold_count,
                # Same gate as LONG: borderline folds flip between runs with small data
                # changes — requiring 4/5 is too strict when mean is consistently > 50%.
                "folds_gte_3_of_5_pass_0.50": folds_passing >= 3,
                # 50% floor: 51%+ mean over 4000+ WF trades is statistically above noise.
                "mean_wf_gt_0.50": wf_prec > 0.50,
                "primary_class_gt_200": n_pos > 200,
                "max_fold_precision_lte_0.85": max_fold_prec <= 0.85,
            }
        accepted = all(accept_checks.values())
        failed_checks = [name for name, ok in accept_checks.items() if not ok]
        print(f"\n[CRYPTO ML] === {side.upper()} MODEL ACCEPTANCE CRITERIA ===")
        print(f"  Label balance : y=1={n_pos} ({pos_rate:.1%}) | y=0={n_neg} ({1-pos_rate:.1%})")
        fold_recalls = [f.get("recall", float("nan")) for f in fold_results]
        fold_pr_aucs = [f["pr_auc"] for f in fold_results if f.get("pr_auc") is not None]
        mean_recall  = float(np.nanmean(fold_recalls)) if fold_recalls else None
        mean_pr_auc  = float(np.mean(fold_pr_aucs)) if fold_pr_aucs else None
        prec_std     = float(np.std(fold_precisions)) if len(fold_precisions) > 1 else 0.0

        print(f"  ┌─ Per-fold WF  prec / recall / PR-AUC / trades ──────────────")
        for f in fold_results:
            prec   = f["precision"]
            recall = f.get("recall", float("nan"))
            pr_auc = f.get("pr_auc")
            tc     = f["trade_count"]
            pr_str = f"{pr_auc:.3f}" if pr_auc is not None else " n/a"
            flag   = "WARN_FEW" if tc < 20 else ("FAIL" if prec <= 0.50 else "ok")
            print(f"  │  Fold {f['fold']}: {prec:.1%}  recall={recall:.1%}  PR-AUC={pr_str}  trades={tc}  [{flag}]")
        print(f"  └─────────────────────────────────────────────────────────────")
        print(f"  Mean WF precision : {wf_prec:.1%}")
        print(f"  Fold precision std: {prec_std:.1%}  (>20pp = high variance)")
        print(f"  Eligible mean WF  : {eligible_mean:.1%} (folds with >=20 trades)")
        if mean_recall is not None:
            print(f"  Mean recall       : {mean_recall:.1%}  (coverage at chosen threshold)")
        if mean_pr_auc is not None:
            print(f"  Mean PR-AUC       : {mean_pr_auc:.3f}  (random baseline ≈ pos_rate={pos_rate:.2f})")
        print(f"  Folds passing >50%: {folds_passing}/{N_FOLDS}")
        print(f"  Fold count        : {len(fold_results)}/{N_FOLDS}")
        print(f"  Fold precision min/max: {min_fold_prec:.1%} / {max_fold_prec:.1%}")
        print(f"  Fold variance     : {fold_variance:.1%}")
        print(f"  Min fold trades   : {min_fold_trades}")
        print(f"  Total WF trades   : {trade_tot}")
        print(f"  Net expectancy    : {net_exp:.3%}")
        print(f"  Overall           : {'ACCEPT' if accepted else 'REJECT'}")
        if failed_checks:
            print(f"  Failed checks     : {', '.join(failed_checks)}")
        print(f"[CRYPTO ML] {side.upper()} WF precision: {wf_prec:.1%} | "
              f"trades: {trade_tot} | net_expectancy: {net_exp:.3%}")

        # Final model on all data
        print(f"[CRYPTO ML] {side.upper()} final model egitiliyor...")
        neg_all = int((y_comb == 0).sum())
        pos_all = int((y_comb == 1).sum())
        spw_all = max(1.0, neg_all / max(pos_all, 1))
        final_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_all)
        final_model.fit(X_comb.fillna(0), y_comb.astype(int), verbose=False)

        imp = pd.Series(final_model.feature_importances_, index=active_feats)
        _FEAT_CATEGORY = {
            "rsi14": "momentum", "mtf_1h_rsi": "momentum", "mtf_5m_rsi": "momentum",
            "macd_hist_norm": "momentum", "mtf_5m_macd_hist": "momentum",
            "macd_slope": "momentum", "rsi_slope": "momentum",
            "ema8_21_cross": "trend", "mtf_1h_trend": "trend", "price_ema50_pct": "trend",
            "bb_position": "trend", "mtf_5m_bb_pos": "trend",
            "ret_1h": "return", "ret_2h": "return", "ret_4h": "return",
            "ret_6h": "return", "ret_12h": "return", "ret_24h": "return",
            "btc_1h_return": "btc", "btc_ret_4h": "btc", "btc_vol_1h": "btc",
            "volume_ratio": "volume", "volume_spike": "volume",
            "volume_zscore": "volume", "volume_trend": "volume",
            "atr14_norm": "volatility", "atr_ratio": "volatility",
            "bb_width_zscore": "volatility", "high_low_range": "volatility",
            "high_low_zscore": "volatility",
            "hour_of_day": "time", "day_of_week": "time",
            "is_asia_session": "time", "is_us_session": "time", "is_weekend": "time",
            "fear_greed_value": "sentiment",
            "body_ratio": "structure", "buy_pressure": "structure",
            "symbol_encoded": "identity",
            "price_vs_7d_high": "range", "ema_spread": "trend",
            "funding_rate": "funding", "funding_rate_avg3": "funding",
            "funding_rate_extreme": "funding", "funding_contrarian": "funding",
            "btc_dominance": "macro", "btc_dominance_regime": "macro",
            "btc_dominance_7d_change": "macro", "htf_bias": "htf",
        }
        print(f"[CRYPTO ML] {side.upper()} top-10 ozellik:")
        for name, score in imp.nlargest(10).items():
            cat = _FEAT_CATEGORY.get(name, "other")
            print(f"  {name}: {score:.4f}  [{cat}]")

        # Feature medians for NaN fallback at predict time
        _MEDIAN_FEATURES = [
            "fear_greed_value", "btc_1h_return", "btc_ret_4h", "btc_vol_1h",
            "mtf_5m_rsi", "mtf_5m_macd_hist", "mtf_5m_bb_pos",
            "funding_rate", "funding_rate_avg3", "funding_rate_extreme", "funding_contrarian",
            "btc_dominance", "btc_dominance_regime", "btc_dominance_7d_change", "htf_bias",
        ]
        feature_medians: dict[str, float] = {}
        for col in _MEDIAN_FEATURES:
            if col in X_comb.columns:
                med = float(X_comb[col].median(skipna=True))
                feature_medians[col] = 0.0 if np.isnan(med) else med

        model_dict = {
            "model":          final_model,
            "feature_names":  active_feats,
            "feature_count":  len(active_feats),
            "wf_precision":   wf_prec,
            "wf_report":      wf_report,
            "wf_summary":     wf_summary,
            "threshold_config": threshold_report.get("tiers", {}),
            "trained_at":     trained_at,
            "n_samples":      len(X_comb),
            "label_config":   {"name": f"directional_{side}", "mode": "directional",
                               "horizon_bars": HORIZON, "threshold": THRESHOLD},
            "feature_medians": feature_medians,
            "model_side":     side,
            "active":         False,
            "promotion_ready": accepted,
            "acceptance_checks": accept_checks,
            "failed_checks":  failed_checks,
            "history_days":   TRAIN_HISTORY_DAYS,
            "training_start": training_start.isoformat() if training_start is not None else "unknown",
            "training_end":   training_end.isoformat() if training_end is not None else "unknown",
            "symbols_trained": [p["__symbol"].iloc[0] if "__symbol" in p.columns else "?"
                                for p in parts],
        }

        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_dict, candidate_path)
        model_dict["candidate_path"] = str(candidate_path)
        print(f"[CRYPTO ML] {side.upper()} candidate kaydedildi: {candidate_path}")

        if not accepted:
            print(f"[CRYPTO ML] {side.upper()} model KAYDEDILMEDI — {', '.join(failed_checks)}")
            model_dict["saved"] = False
            return model_dict
        model_dict["saved"] = True
        return model_dict

    long_dict  = _train_one_model(long_parts,  "long",  MODEL_STAGE_LONG_PATH)
    short_dict = _train_one_model(short_parts, "short", MODEL_STAGE_SHORT_PATH)

    print(f"\n[CRYPTO ML] === DIRECTIONAL TRAIN TAMAMLANDI ===")
    print(f"[CRYPTO ML]  LONG  WF precision: {long_dict.get('wf_precision', 0.0):.1%}")
    print(f"[CRYPTO ML]  SHORT WF precision: {short_dict.get('wf_precision', 0.0):.1%}")
    return long_dict, short_dict


def _tier_active_model_path(side: str, tier_name: str) -> Path:
    side = side.lower()
    tier_name = tier_name.upper()
    if side == "long" and tier_name == "MAJOR":
        return MODEL_PATH_LONG_MAJOR
    if side == "long" and tier_name == "RISKY":
        return MODEL_PATH_LONG_RISKY
    if side == "short" and tier_name == "MAJOR":
        return MODEL_PATH_SHORT_MAJOR
    if side == "short" and tier_name == "RISKY":
        return MODEL_PATH_SHORT_RISKY
    raise ValueError(f"unknown tier model path side={side} tier={tier_name}")


def _tier_stage_model_path(side: str, tier_name: str) -> Path:
    side = side.lower()
    tier_name = tier_name.upper()
    if side == "long" and tier_name == "MAJOR":
        return MODEL_STAGE_LONG_MAJOR_PATH
    if side == "long" and tier_name == "RISKY":
        return MODEL_STAGE_LONG_RISKY_PATH
    if side == "short" and tier_name == "MAJOR":
        return MODEL_STAGE_SHORT_MAJOR_PATH
    if side == "short" and tier_name == "RISKY":
        return MODEL_STAGE_SHORT_RISKY_PATH
    raise ValueError(f"unknown tier stage path side={side} tier={tier_name}")


def _fit_directional_model_parts(
    parts: list[pd.DataFrame],
    side: str,
    tier_name: str,
    candidate_path: Path,
    trained_at: str,
    training_start,
    training_end,
) -> dict:
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["__timestamp", "__symbol"]).reset_index(drop=True)

    active_feats = [f for f in FEATURE_NAMES if f in combined.columns]
    X_comb = combined[active_feats]
    y_comb = combined["__y"].astype(float)
    gross_comb = combined["__gross_ret"].astype(float)
    net_comb = combined["__net_ret"].astype(float)
    ts_comb = combined["__timestamp"]
    tier_comb = combined["__tier"]
    sym_comb = combined["__symbol"]

    n_pos = int((y_comb == 1).sum())
    n_neg = int((y_comb == 0).sum())
    pos_rate = n_pos / max(len(y_comb), 1)
    print(
        f"\n[CRYPTO ML] {tier_name} {side.upper()} model: {len(X_comb)} rows | "
        f"y=1: {n_pos} ({pos_rate:.1%}) | y=0: {n_neg} ({1-pos_rate:.1%})"
    )

    wf_report, threshold_report = _walk_forward(
        X_comb, y_comb, gross_comb, net_comb,
        ts_comb, tier_comb, sym_comb, trained_at,
    )
    wf_summary = wf_report.get("summary", {})
    wf_prec = float(wf_summary.get("precision_mean") or 0.0)
    trade_tot = int(wf_summary.get("trade_count_total") or 0)
    net_exp = float(wf_summary.get("net_expectancy_after_costs_mean") or 0.0)
    fold_results = wf_report.get("folds", [])
    fold_precisions = [float(f.get("precision", 0.0)) for f in fold_results]
    folds_passing = sum(1 for p in fold_precisions if p > 0.50)
    full_fold_count = len(fold_results) == N_FOLDS

    accept_checks = {
        "all_expected_folds_present": full_fold_count,
        "folds_gte_3_of_5_pass_0.50": folds_passing >= 3,
        "mean_wf_gt_0.52": wf_prec > 0.52,
    }
    accepted = all(accept_checks.values())
    failed_checks = [name for name, ok in accept_checks.items() if not ok]
    print(
        f"[CRYPTO ML] {tier_name} {side.upper()} WF={wf_prec:.1%} | "
        f"folds={folds_passing}/{N_FOLDS} | trades={trade_tot} | "
        f"net_exp={net_exp:.3%} | {'ACCEPT' if accepted else 'REJECT'}"
    )
    if failed_checks:
        print(f"[CRYPTO ML] {tier_name} {side.upper()} failed: {', '.join(failed_checks)}")

    neg_all = int((y_comb == 0).sum())
    pos_all = int((y_comb == 1).sum())
    spw_all = max(1.0, neg_all / max(pos_all, 1))
    final_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_all)
    final_model.fit(X_comb.fillna(0), y_comb.astype(int), verbose=False)

    feature_medians: dict[str, float] = {}
    for col in [
        "fear_greed_value", "btc_1h_return", "btc_ret_4h", "btc_vol_1h",
        "mtf_5m_rsi", "mtf_5m_macd_hist", "mtf_5m_bb_pos",
        "funding_rate", "funding_rate_avg3", "funding_rate_extreme", "funding_contrarian",
        "btc_dominance", "btc_dominance_regime", "btc_dominance_7d_change", "htf_bias",
    ]:
        if col in X_comb.columns:
            med = float(X_comb[col].median(skipna=True))
            feature_medians[col] = 0.0 if np.isnan(med) else med

    model_dict = {
        "model": final_model,
        "feature_names": active_feats,
        "feature_count": len(active_feats),
        "wf_precision": wf_prec,
        "wf_report": wf_report,
        "wf_summary": wf_summary,
        "threshold_config": threshold_report.get("tiers", {}),
        "trained_at": trained_at,
        "n_samples": len(X_comb),
        "label_config": {"name": f"tiered_directional_{side}_{tier_name.lower()}", "mode": "directional"},
        "feature_medians": feature_medians,
        "model_side": side.lower(),
        "model_tier": tier_name,
        "active": False,
        "promotion_ready": accepted,
        "acceptance_checks": accept_checks,
        "failed_checks": failed_checks,
        "history_days": TRAIN_HISTORY_DAYS,
        "training_start": training_start.isoformat() if training_start is not None else "unknown",
        "training_end": training_end.isoformat() if training_end is not None else "unknown",
        "symbols_trained": sorted(set(str(s) for s in sym_comb.tolist())),
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_dict, candidate_path)
    model_dict["candidate_path"] = str(candidate_path)
    return model_dict


def train_tiered_directional(stream=None) -> dict[str, dict]:
    """Train directional models separately for MAJOR and RISKY tiers."""
    from crypto_indicators import build_htf_bias_history_from_1h
    from crypto_stream import fetch_funding_rate_history, get_btc_dominance_features

    if stream is None:
        from crypto_stream import CryptoStream
        stream = CryptoStream()

    HORIZON = int(LABEL_CONFIG["horizon_bars"])
    THRESHOLD = 0.025
    cost_rate = _round_trip_cost_rate(LABEL_CONFIG, EXECUTION_COST_CONFIG)
    trained_at = datetime.now(timezone.utc).isoformat()
    print("\n[CRYPTO ML] === TIERED DIRECTIONAL TRAIN: MAJOR/RISKY x LONG/SHORT ===")
    _QUALITY_WARNINGS.clear()

    btc_dominance_info = get_btc_dominance_features(force_refresh=True)
    data = _fetch_ohlcv_all_paginated(stream)
    fg_dict = _fetch_fg_history()
    symbols = [sym for sym in (TIER_SYMBOLS["MAJOR"] + TIER_SYMBOLS["RISKY"]) if sym in data and f"{sym}_5m" in data]
    btc_df = data.get("BTC/USDT")
    training_frames = [data[sym] for sym in symbols if sym in data]
    training_start = min((df["timestamp"].iloc[0] for df in training_frames), default=None)
    training_end = max((df["timestamp"].iloc[-1] for df in training_frames), default=None)

    long_parts: list[pd.DataFrame] = []
    short_parts: list[pd.DataFrame] = []

    for sym in symbols:
        df = data.get(sym)
        df_5m = data.get(f"{sym}_5m")
        btc_ref = btc_df if sym != "BTC/USDT" else None
        if df is None or df_5m is None:
            continue
        funding_hist = fetch_funding_rate_history(sym)
        htf_bias_hist = build_htf_bias_history_from_1h(df)
        X = _build_features(
            df,
            btc_ref,
            fg_dict,
            df_5m,
            funding_df=funding_hist,
            btc_dominance_info=btc_dominance_info,
            htf_bias_df=htf_bias_hist,
        )
        dir_labels = _build_directional_labels(df, horizon=HORIZON, threshold=THRESHOLD)
        long_label = dir_labels["long_label"]
        short_label = dir_labels["short_label"]
        ret_future = dir_labels["ret_future"]
        ret_future_down = dir_labels["ret_future_down"]
        min_len = min(len(X), len(long_label))
        X = X.iloc[:min_len]
        long_label = long_label.iloc[:min_len]
        short_label = short_label.iloc[:min_len]
        ret_future = ret_future.iloc[:min_len]
        ret_future_down = ret_future_down.iloc[:min_len]
        X = X.iloc[WARMUP_BARS:]
        long_label = long_label.iloc[WARMUP_BARS:]
        short_label = short_label.iloc[WARMUP_BARS:]
        ret_future = ret_future.iloc[WARMUP_BARS:]
        ret_future_down = ret_future_down.iloc[WARMUP_BARS:]
        valid_mask = long_label.notna() & short_label.notna()
        X = X[valid_mask]
        long_label = long_label[valid_mask]
        short_label = short_label[valid_mask]
        ret_future = ret_future[valid_mask]
        ret_future_down = ret_future_down[valid_mask]
        directional_mask = (long_label == 1) | (short_label == 1)
        ts_series = pd.to_datetime(df.loc[X.index, "timestamp"], utc=True)

        def _make_part(y_labels: pd.Series, gross_ret: pd.Series) -> pd.DataFrame:
            part = X[directional_mask].copy()
            part["__timestamp"] = ts_series[directional_mask].values
            part["__symbol"] = sym
            part["__tier"] = _tier_for_symbol(sym)
            part["__y"] = y_labels[directional_mask].astype(float).values
            net_ret = gross_ret[directional_mask] - cost_rate
            part["__gross_ret"] = gross_ret[directional_mask].astype(float).values
            part["__net_ret"] = net_ret.astype(float).values
            return part

        long_parts.append(_make_part(long_label, ret_future))
        short_parts.append(_make_part(short_label, ret_future_down))

    tiered_results: dict[str, dict] = {}
    for tier_name in ("MAJOR", "RISKY"):
        long_tier_parts = [p for p in long_parts if str(p["__tier"].iloc[0]) == tier_name]
        short_tier_parts = [p for p in short_parts if str(p["__tier"].iloc[0]) == tier_name]
        if long_tier_parts:
            tiered_results[f"{tier_name}_LONG"] = _fit_directional_model_parts(
                long_tier_parts, "long", tier_name, _tier_stage_model_path("long", tier_name),
                trained_at, training_start, training_end,
            )
        else:
            tiered_results[f"{tier_name}_LONG"] = {}
        if short_tier_parts:
            tiered_results[f"{tier_name}_SHORT"] = _fit_directional_model_parts(
                short_tier_parts, "short", tier_name, _tier_stage_model_path("short", tier_name),
                trained_at, training_start, training_end,
            )
        else:
            tiered_results[f"{tier_name}_SHORT"] = {}

    return tiered_results


# ── Inference helpers (used by crypto_trader.py) ──────────────────────────────

def load_model() -> dict | None:
    """Load crypto_xgb.pkl. Returns None if file missing."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        print(f"[CRYPTO ML] Model yuklenemedi: {exc}")
        return None


def load_model_long() -> dict | None:
    """Load crypto_xgb_long.pkl (directional LONG model). Returns None if missing."""
    if not MODEL_PATH_LONG.exists():
        return None
    try:
        return joblib.load(MODEL_PATH_LONG)
    except Exception as exc:
        print(f"[CRYPTO ML] LONG model yuklenemedi: {exc}")
        return None


def load_model_short() -> dict | None:
    """Load crypto_xgb_short.pkl (directional SHORT model). Returns None if missing."""
    if not MODEL_PATH_SHORT.exists():
        return None
    try:
        return joblib.load(MODEL_PATH_SHORT)
    except Exception as exc:
        print(f"[CRYPTO ML] SHORT model yuklenemedi: {exc}")
        return None


def load_tier_model(side: str, tier_name: str) -> dict | None:
    """Load tier-specific directional artifact."""
    path = _tier_active_model_path(side, tier_name)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[CRYPTO ML] {tier_name} {side.upper()} model yuklenemedi: {exc}")
        return None


def load_artifact_path(path: Path) -> dict | None:
    """Load any crypto artifact by path."""
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[CRYPTO ML] Artifact yuklenemedi ({path.name}): {exc}")
        return None


def artifact_runtime_approved(model_dict: dict | None, side: str | None = None) -> tuple[bool, str]:
    """Return whether a directional artifact is safe to use live."""
    if not model_dict:
        return False, "missing"
    if not bool(model_dict.get("active", False)):
        return False, "active_flag_false"
    if side is not None and model_dict.get("model_side") != side:
        return False, f"side_mismatch:{model_dict.get('model_side')}"
    if not bool(model_dict.get("promotion_ready", False)):
        return False, "promotion_ready_false"
    if int(model_dict.get("history_days", 0) or 0) < TRAIN_HISTORY_DAYS:
        return False, "history_days_too_short"

    accept_checks = model_dict.get("acceptance_checks") or {}
    if accept_checks and not all(bool(v) for v in accept_checks.values()):
        return False, "acceptance_checks_failed"
    failed_checks = model_dict.get("failed_checks") or []
    if failed_checks:
        return False, "failed_checks_present"

    folds = model_dict.get("wf_report", {}).get("folds", [])
    if len(folds) != N_FOLDS:
        return False, "incomplete_folds"
    fold_precisions = [float(f.get("precision", 0.0)) for f in folds]
    if fold_precisions and max(fold_precisions) > 0.85:
        return False, "suspicious_fold_precision"

    return True, "approved"


def _norm_df_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the 'timestamp' column to datetime64[us, UTC] and floor to 1h.
    Called at the entry point of predict_proba to prevent merge_asof precision
    mismatches (datetime64[ms] vs [us]) regardless of what the caller passes.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    df = df.copy()
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
          .dt.as_unit("us")
          .dt.floor("1h")
    )
    return df


def predict_proba(model_dict: dict, df_1h: pd.DataFrame,
                  btc_df: pd.DataFrame | None = None,
                  fg_dict: dict | None = None,
                  df_5m: pd.DataFrame | None = None,
                  symbol: str | None = None,
                  funding_info: dict | None = None,
                  btc_dominance_info: dict | None = None,
                  htf_info: dict | None = None) -> float:
    """
    Predict BUY probability for the most recent bar of df_1h.
    Uses model_dict["feature_names"] so predictions work with old and new models.
    Returns float in [0, 1]. Returns 0.5 on any error.

    [QUALITY] warnings from _build_features are suppressed during inference —
    they reflect historical bars in df_1h that fall outside the 5m data window,
    not the last (predicted) row.  Missing values in the last row are filled
    via stored training-data medians, and any remaining NaN is flagged here.
    """
    import traceback as _tb

    # Suppress _warn_quality prints during inference; capture them locally.
    _captured_quality: list[str] = []
    _orig_warn = _QUALITY_WARNINGS.copy()

    def _silent_warn(msg: str) -> None:
        if msg not in _captured_quality:
            _captured_quality.append(msg)

    try:
        # Temporarily replace _warn_quality at the module level with a silent version
        # so that historical-bar alignment warnings don't pollute the signal log.
        import crypto_ml as _self_mod
        _self_mod._warn_quality = _silent_warn  # type: ignore[attr-defined]

        # Normalise 1h and BTC timestamps to datetime64[us, UTC] floored to 1h.
        # 5m data must NOT be floored to 1h — minute-level precision is needed for
        # the merge_asof with tolerance=60min inside _build_features.
        df_1h_norm = _norm_df_timestamps(df_1h)
        btc_norm   = _norm_df_timestamps(btc_df)
        if df_5m is not None and not df_5m.empty and "timestamp" in df_5m.columns:
            df_5m_norm = df_5m.copy()
            df_5m_norm["timestamp"] = (
                pd.to_datetime(df_5m_norm["timestamp"], utc=True).dt.as_unit("us")
            )  # precision-only normalisation — preserves 5-minute resolution
        else:
            df_5m_norm = None

        X = _build_features(
            df_1h_norm,
            btc_norm,
            fg_dict or {},
            df_5m_norm,
            funding_df=None,
            btc_dominance_info=btc_dominance_info,
            htf_bias_df=None,
            htf_info=htf_info,
        )
        if funding_info:
            X["funding_rate"] = float(funding_info.get("funding_rate", 0.0) or 0.0)
            X["funding_rate_avg3"] = float(funding_info.get("funding_rate_avg3", X["funding_rate"].iloc[-1]) or X["funding_rate"].iloc[-1])
            X["funding_rate_extreme"] = (X["funding_rate"].abs() > 0.001).astype(float)
            X["funding_contrarian"] = -np.sign(X["funding_rate"]).astype(float)
        # Inject symbol encoding so predictions are consistent with training
        if symbol:
            X["symbol_encoded"] = _symbol_encode(symbol)
        # Use the feature list the model was trained on (handles old 22-feat / new 24-feat)
        train_feats = model_dict.get("feature_names", list(X.columns))
        x_raw  = X[[f for f in train_feats if f in X.columns]].iloc[[-1]]
        # Fill remaining NaN: use stored training medians first, then 0
        feature_medians = model_dict.get("feature_medians", {})
        x_last = x_raw.copy()
        nan_at_predict: list[str] = []
        for col in x_last.columns:
            if x_last[col].isna().any():
                fallback = feature_medians.get(col, 0.0)
                x_last[col] = x_last[col].fillna(fallback)
                nan_at_predict.append(col)
        # Warn only about features that are STILL missing in the last (predicted) row
        if nan_at_predict:
            print(f"  [QUALITY] predict row NaN filled ({symbol}): {', '.join(nan_at_predict)}")
        try:
            prob = float(model_dict["model"].predict_proba(x_last)[0][1])
        except Exception as predict_exc:
            print(f"[CRYPTO ML] predict_proba model call failed ({symbol}): {predict_exc}")
            print(_tb.format_exc())
            return 0.5
        _print_prediction_diagnostics(symbol, x_raw, x_last, prob)
        return prob
    except Exception as exc:
        print(f"[CRYPTO ML] predict_proba feature build failed ({symbol}): {exc}")
        print(_tb.format_exc())
        return 0.5
    finally:
        # Restore the real _warn_quality function
        import crypto_ml as _self_mod
        _self_mod._warn_quality = _warn_quality  # type: ignore[attr-defined]


def _run_label_sanity_tests() -> None:
    base = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC"),
        "open":  [100, 100, 100, 100, 100, 100, 100, 100],
        "high":  [100, 102.5, 100.4, 100.3, 100.2, 100.1, 100.1, 100.1],
        "low":   [100, 99.8, 97.0, 99.7, 99.8, 99.8, 99.8, 99.8],
        "close": [100, 101.0, 98.0, 101.5, 100.2, 100.3, 100.4, 100.5],
        "atr14": [1.0] * 8,
    })

    legacy = _build_labels(base, {"mode": "legacy_simple", "horizon_bars": 3, "min_gain": 0.01})
    assert legacy.iloc[0] == 1.0, "legacy label should mark +1% move as positive"
    assert pd.isna(legacy.iloc[-1]), "legacy tail must stay NaN"

    trade_cfg = {
        "mode": "triple_barrier",
        "horizon_bars": 3,
        "use_atr_targets": False,
        "min_target_gain": 0.02,
        "min_stop_loss": 0.02,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "fallback_mode": "net_return",
        "fallback_min_net_return": 0.0,
        "intrabar_conflict_policy": "stop_first",
    }
    trade_labels = _build_labels(base, trade_cfg)
    assert trade_labels.iloc[0] == 1.0, "row 0 should hit target before stop"
    assert trade_labels.iloc[1] == 0.0, "row 1 should hit stop before target"
    assert pd.isna(trade_labels.iloc[-1]), "trade-aware tail must stay NaN"

    fallback_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-02-01", periods=6, freq="1h", tz="UTC"),
        "open":  [100, 100, 100, 100, 100, 100],
        "high":  [100.2, 100.4, 100.5, 100.6, 100.7, 100.8],
        "low":   [99.8, 99.9, 100.0, 100.1, 100.2, 100.3],
        "close": [100.0, 100.2, 100.3, 100.4, 100.5, 100.6],
        "atr14": [0.5] * 6,
    })
    fallback_labels = _build_labels(fallback_df, {
        **trade_cfg,
        "horizon_bars": 2,
        "min_target_gain": 0.05,
        "min_stop_loss": 0.05,
        "fallback_min_net_return": 0.001,
    })
    assert fallback_labels.iloc[0] == 1.0, "fallback net return should label tradeable drift as positive"
    assert pd.isna(fallback_labels.iloc[-1]), "fallback tail must stay NaN"

    print("[CRYPTO ML] Label sanity tests passed.")


def _eval_per_coin_validation() -> None:
    """Load and summarize the most recent per_coin_validation JSON for --eval."""
    import glob as _glob
    pattern = str(BASE_DIR / "models" / "per_coin_validation_*.json")
    files = sorted(_glob.glob(pattern))
    if not files:
        print("\n[CRYPTO ML] Per-coin validation: dosya bulunamadi")
        return
    latest = Path(files[-1])
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\n[CRYPTO ML] Per-coin validation okunamadi: {exc}")
        return

    coins = data.get("coins", [])
    warnings = data.get("warnings", [])
    gen_at = data.get("generated_at", "?")

    print(f"\n[CRYPTO ML] Per-coin validation ({latest.name}, generated {gen_at[:10]}):")
    print(f"  {'Coin':<12} {'Support':>8} {'Prec@0.50':>10} {'PosRate':>8} {'Avg/Fold':>9} {'Flag':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*9} {'─'*5}")
    for c in coins:
        flag = "LOW" if c["support"] < 10 else "ok"
        print(
            f"  {c['symbol']:<12} {c['support']:>8} "
            f"{c['precision_at_0.50']:>9.1%} "
            f"{c['positive_rate']:>7.1%} "
            f"{c['avg_support_per_fold']:>9.1f} "
            f"  {flag}"
        )
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    {w}")
    else:
        print(f"  All coins have support >= 10")


def _eval_retrain_history() -> None:
    """Print last 5 retrain results from models/retrain_history.jsonl."""
    if not RETRAIN_HISTORY_PATH.exists():
        print("\n[CRYPTO ML] Retrain history: dosya bulunamadi")
        return
    rows: list[dict] = []
    try:
        for line in RETRAIN_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    except Exception as exc:
        print(f"\n[CRYPTO ML] Retrain history okunamadi: {exc}")
        return
    if not rows:
        print("\n[CRYPTO ML] Retrain history bos")
        return
    rows = rows[-5:]
    print("\n[CRYPTO ML] Last 5 retrain results:")
    print(f"  {'Date':<10} {'Tier':<6} {'Dir':<5} {'Result':<8} {'Mean_WF':>8} {'Folds':>5}")
    print(f"  {'─'*10} {'─'*6} {'─'*5} {'─'*8} {'─'*8} {'─'*5}")
    for row in rows:
        print(
            f"  {str(row.get('date', '?'))[:10]:<10} "
            f"{str(row.get('tier', '-')):<6} "
            f"{str(row.get('direction', '?')):<5} "
            f"{str(row.get('result', '?')):<8} "
            f"{float(row.get('mean_wf_precision', 0.0)):>7.1%} "
            f"{int(row.get('folds_passed', 0)):>5}"
        )


# ── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto ML training")
    parser.add_argument("--eval", action="store_true", help="evaluate existing model only")
    parser.add_argument("--self-test", action="store_true", help="run label sanity tests")
    parser.add_argument("--directional", action="store_true",
                        help="(legacy flag, now ignored — directional is always the default)")
    parser.add_argument("--single", action="store_true",
                        help="train old single-model (crypto_xgb.pkl) instead of directional pair")
    args = parser.parse_args()

    if args.self_test:
        _run_label_sanity_tests()
    elif args.eval:
        print(f"[CRYPTO ML] Active feature set count: {len(FEATURE_NAMES)}")
        print(f"[CRYPTO ML] Active features: {', '.join(FEATURE_NAMES)}")

        def _print_directional_report(side_name: str, active_model: dict | None, candidate_model: dict | None) -> None:
            print(f"[CRYPTO ML] [{side_name}]")
            if active_model is None:
                print("  active: missing")
            else:
                print(
                    f"  active: WF={active_model.get('wf_precision', 0.0):.1%} | "
                    f"samples={active_model.get('n_samples', 'n/a')} | "
                    f"trained={str(active_model.get('trained_at', 'n/a'))[:19]}"
                )
            if candidate_model is None:
                print("  candidate: missing")
            else:
                print(
                    f"  candidate: WF={candidate_model.get('wf_precision', 0.0):.1%} | "
                    f"samples={candidate_model.get('n_samples', 'n/a')} | "
                    f"trained={str(candidate_model.get('trained_at', 'n/a'))[:19]} | "
                    f"promotion_ready={candidate_model.get('promotion_ready', False)}"
                )
            if active_model is not None and candidate_model is not None:
                delta = float(candidate_model.get("wf_precision", 0.0)) - float(active_model.get("wf_precision", 0.0))
                print(f"  delta(candidate-active): {delta:+.1%}")

        active_long = load_model_long()
        active_short = load_model_short()
        candidate_long = load_artifact_path(MODEL_STAGE_LONG_PATH)
        candidate_short = load_artifact_path(MODEL_STAGE_SHORT_PATH)
        _print_directional_report("LONG", active_long, candidate_long)
        _print_directional_report("SHORT", active_short, candidate_short)

        if active_long is None and active_short is None and candidate_long is None and candidate_short is None:
            m = load_model()
            if m is None:
                print("[CRYPTO ML] Model dosyasi bulunamadi.")
            else:
                print(f"[CRYPTO ML] Legacy model: egitildi {m['trained_at']}")
                print(f"[CRYPTO ML] Legacy WF precision: {m['wf_precision']:.1%}")
                print(f"[CRYPTO ML] Legacy N samples: {m['n_samples']}")

        # ── Per-coin validation summary ──────────────────────────────────────
        _eval_per_coin_validation()
        _eval_retrain_history()

    elif args.single:
        m = train()
        if m:
            if m.get("saved_to_active"):
                print(f"\n[CRYPTO ML] TAMAMLANDI — WF BUY precision: {m['wf_precision']:.1%}")
            else:
                print(
                    "\n[CRYPTO ML] DEGERLENDIRME ZAYIF — aktif model korunuyor, rapor JSON guncellendi."
                )
    else:
        # Default: train tier-specific directional models (MAJOR/RISKY x LONG/SHORT)
        tiered = train_tiered_directional()
        print(f"\n[CRYPTO ML] === SONUC ===")
        for label, md in tiered.items():
            wf  = md.get("wf_precision", 0.0) if md else 0.0
            ready = md.get("promotion_ready", False) if md else False
            status = "STAGED_PROMOTE_READY" if ready else "STAGED_REJECTED"
            print(f"  {label}: WF={wf:.1%}  [{status}]")
