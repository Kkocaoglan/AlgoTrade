"""
regime_hmm.py -- 3-state Hidden Markov Model regime classifier for BIST proxy.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / "trade_data.db"
MODELS_DIR = Path(__file__).parent / "models"
HMM_MODEL_PATH = MODELS_DIR / "hmm_model.pkl"
PROXY_SYMBOLS = ["GARAN", "THYAO", "EREGL"]
HMM_LOOKBACK_DAYS = 252
HMM_CONFIRMATION_DAYS = 2


def _load_proxy_ohlcv(db_path):
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" * len(PROXY_SYMBOLS))
        df = pd.read_sql(
            f"""
            SELECT symbol, date, high, low, close
            FROM ohlcv
            WHERE symbol IN ({placeholders})
            ORDER BY date
            """,
            conn,
            params=PROXY_SYMBOLS,
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    for col in ("high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_macro_frame(db_path):
    conn = sqlite3.connect(db_path)
    try:
        try:
            mac = pd.read_sql("SELECT * FROM macro_data ORDER BY date", conn)
        except Exception:
            return pd.DataFrame()
    finally:
        conn.close()

    if mac.empty:
        return pd.DataFrame()

    mac["date"] = pd.to_datetime(mac["date"])
    mac = mac.set_index("date").sort_index()
    for col in mac.columns:
        mac[col] = pd.to_numeric(mac[col], errors="coerce")
    return mac


def build_raw_feature_matrix(db_path):
    """Load unstandardized market-regime features from SQLite."""
    proxy_df = _load_proxy_ohlcv(db_path)
    if proxy_df.empty:
        return pd.DataFrame(
            columns=[
                "xu100_proxy_ret",
                "xu100_proxy_vol20",
                "hl_range_pct_avg",
                "usdtry_ret_5d",
                "vix_level_z",
            ]
        )

    close_pivot = proxy_df.pivot(index="date", columns="symbol", values="close").sort_index()
    high_pivot = proxy_df.pivot(index="date", columns="symbol", values="high").sort_index()
    low_pivot = proxy_df.pivot(index="date", columns="symbol", values="low").sort_index()

    proxy_ret = close_pivot.pct_change().mean(axis=1)
    proxy_vol20 = proxy_ret.rolling(20, min_periods=20).std()
    hl_range_pct_avg = ((high_pivot - low_pivot) / close_pivot.replace(0, np.nan)).mean(axis=1)

    raw = pd.DataFrame(
        {
            "xu100_proxy_ret": proxy_ret,
            "xu100_proxy_vol20": proxy_vol20,
            "hl_range_pct_avg": hl_range_pct_avg,
        }
    )

    macro_df = _load_macro_frame(db_path)
    if not macro_df.empty:
        macro_aligned = macro_df.reindex(raw.index).ffill()
        if "usdtry_5d_ret" in macro_aligned.columns:
            raw["usdtry_ret_5d"] = macro_aligned["usdtry_5d_ret"].fillna(0.0)
        else:
            raw["usdtry_ret_5d"] = 0.0

        if "vix_level" in macro_aligned.columns:
            vix = macro_aligned["vix_level"].astype(float)
            vix_mean = vix.rolling(60, min_periods=20).mean()
            vix_std = vix.rolling(60, min_periods=20).std()
            raw["vix_level_z"] = ((vix - vix_mean) / (vix_std + 1e-9)).fillna(0.0)
        else:
            raw["vix_level_z"] = 0.0
    else:
        raw["usdtry_ret_5d"] = 0.0
        raw["vix_level_z"] = 0.0

    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    return raw


def build_feature_matrix(db_path):
    """Load daily BIST regime features from trade_data.db."""
    raw = build_raw_feature_matrix(db_path)
    if raw.empty:
        return raw
    scaler = StandardScaler()
    values = scaler.fit_transform(raw.values)
    return pd.DataFrame(values, index=raw.index, columns=raw.columns)


def train_hmm(feature_matrix, n_states=3, n_seeds=10):
    """Train GaussianHMM, pick best seed by log-likelihood."""
    best_model, best_score = None, -np.inf
    if feature_matrix.empty:
        return None

    X = feature_matrix.values
    for seed in range(n_seeds):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=seed,
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_model, best_score = model, score
        except Exception:
            continue
    return best_model


def label_states(model, feature_matrix):
    """Assign semantic labels to HMM states based on mean returns."""
    states = model.predict(feature_matrix.values)
    state_df = feature_matrix.copy()
    state_df["state"] = states
    means = state_df.groupby("state")["xu100_proxy_ret"].mean()
    all_states = list(range(model.n_components))
    if means.empty:
        means = pd.Series(0.0, index=all_states, dtype=float)
    else:
        missing_states = [state for state in all_states if state not in means.index]
        if missing_states:
            floor = float(means.min()) - 1e-6
            for offset, state in enumerate(missing_states, start=1):
                means.loc[state] = floor - (offset * 1e-6)
    sorted_states = means.sort_values(ascending=False).index.tolist()
    label_map = {
        sorted_states[0]: "BULL",
        sorted_states[1]: "RANGE",
        sorted_states[2]: "BEAR",
    }
    state_df["regime"] = state_df["state"].map(label_map)
    return state_df, label_map


def _prob_for_regime(probs, label_map, regime_name):
    for state_id, label in label_map.items():
        if label == regime_name:
            return float(probs[state_id])
    return 0.0


def get_current_regime(model, feature_matrix, label_map, confirmation_days=2):
    """Get regime for today with confirmation filter (avoid noise flips)."""
    states = model.predict(feature_matrix.values)
    last_n = states[-confirmation_days:] if len(states) >= confirmation_days else states
    if len(last_n) and len(set(last_n)) == 1:
        current_state = last_n[-1]
    else:
        current_state = states[-1]
    probs = model.predict_proba(feature_matrix.values)[-1]
    return {
        "regime": label_map[current_state],
        "prob_bull": _prob_for_regime(probs, label_map, "BULL"),
        "prob_bear": _prob_for_regime(probs, label_map, "BEAR"),
        "prob_range": _prob_for_regime(probs, label_map, "RANGE"),
        "confirmed": len(last_n) > 0 and len(set(last_n)) == 1,
    }


def compute_historical_regime_features(
    db_path,
    lookback_days=HMM_LOOKBACK_DAYS,
    n_states=3,
    n_seeds=10,
):
    """Replay historical rolling-window HMM predictions for ML features."""
    raw = build_raw_feature_matrix(db_path)
    out = pd.DataFrame(
        index=raw.index,
        data={
            "hmm_prob_bull": 1.0 / 3.0,
            "hmm_prob_bear": 1.0 / 3.0,
            "hmm_prob_range": 1.0 / 3.0,
            "hmm_days_in_state": 0.0,
            "hmm_regime": "RANGE",
            "hmm_confirmed": False,
        },
    )
    if len(raw) <= lookback_days:
        return out

    prev_regime = None
    prev_days = 0
    prev_confirmed = False

    for end_idx in range(lookback_days, len(raw)):
        train_raw = raw.iloc[end_idx - lookback_days:end_idx]
        pred_raw = raw.iloc[[end_idx]]
        scaler = StandardScaler()
        train_std_vals = scaler.fit_transform(train_raw.values)
        train_std = pd.DataFrame(train_std_vals, index=train_raw.index, columns=train_raw.columns)
        model = train_hmm(train_std, n_states=n_states, n_seeds=n_seeds)
        if model is None:
            continue

        state_df, label_map = label_states(model, train_std)
        _ = state_df
        pred_std = pd.DataFrame(
            scaler.transform(pred_raw.values),
            index=pred_raw.index,
            columns=pred_raw.columns,
        )
        state = int(model.predict(pred_std.values)[0])
        probs = model.predict_proba(pred_std.values)[0]
        regime = label_map[state]
        was_same_regime = regime == prev_regime
        days = prev_days + 1 if was_same_regime else 1
        confirmed = was_same_regime and prev_confirmed
        prev_regime = regime
        prev_days = days
        prev_confirmed = confirmed

        out.loc[pred_raw.index[0], "hmm_prob_bull"] = _prob_for_regime(probs, label_map, "BULL")
        out.loc[pred_raw.index[0], "hmm_prob_bear"] = _prob_for_regime(probs, label_map, "BEAR")
        out.loc[pred_raw.index[0], "hmm_prob_range"] = _prob_for_regime(probs, label_map, "RANGE")
        out.loc[pred_raw.index[0], "hmm_days_in_state"] = float(days)
        out.loc[pred_raw.index[0], "hmm_regime"] = regime
        out.loc[pred_raw.index[0], "hmm_confirmed"] = bool(days >= HMM_CONFIRMATION_DAYS)

    return out.ffill().fillna(
        {
            "hmm_prob_bull": 1.0 / 3.0,
            "hmm_prob_bear": 1.0 / 3.0,
            "hmm_prob_range": 1.0 / 3.0,
            "hmm_days_in_state": 0.0,
            "hmm_regime": "RANGE",
            "hmm_confirmed": False,
        }
    )


def build_live_regime_feature_frame(index_like, regime_info):
    """Build a one-row regime feature frame for current live inference."""
    idx = pd.DatetimeIndex(index_like)
    if len(idx) == 0:
        return pd.DataFrame()

    last_idx = idx[-1]
    if not regime_info:
        payload = {
            "hmm_prob_bull": 1.0 / 3.0,
            "hmm_prob_bear": 1.0 / 3.0,
            "hmm_prob_range": 1.0 / 3.0,
            "hmm_days_in_state": 0.0,
            "hmm_regime": "RANGE",
            "hmm_confirmed": False,
        }
    else:
        payload = {
            "hmm_prob_bull": float(regime_info.get("prob_bull", 1.0 / 3.0)),
            "hmm_prob_bear": float(regime_info.get("prob_bear", 1.0 / 3.0)),
            "hmm_prob_range": float(regime_info.get("prob_range", 1.0 / 3.0)),
            "hmm_days_in_state": float(regime_info.get("days_in_state", 0.0)),
            "hmm_regime": str(regime_info.get("regime", "RANGE")),
            "hmm_confirmed": bool(regime_info.get("confirmed", False)),
        }
    return pd.DataFrame([payload], index=[last_idx])


def train_and_save_hmm_model(db_path=DB_PATH, model_path=HMM_MODEL_PATH, n_states=3, n_seeds=10):
    """Train the full-sample HMM and save model + scaler + labels."""
    raw = build_raw_feature_matrix(db_path)
    if raw.empty:
        return None

    scaler = StandardScaler()
    std_values = scaler.fit_transform(raw.values)
    feature_matrix = pd.DataFrame(std_values, index=raw.index, columns=raw.columns)
    model = train_hmm(feature_matrix, n_states=n_states, n_seeds=n_seeds)
    if model is None:
        return None

    state_df, label_map = label_states(model, feature_matrix)
    state_stats = (
        state_df["regime"].value_counts(normalize=True).mul(100.0).round(2).to_dict()
    )
    payload = {
        "model": model,
        "scaler": scaler,
        "label_map": label_map,
        "feature_columns": list(raw.columns),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "state_stats": state_stats,
    }
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(payload, model_path)
    return payload


def load_hmm_model(model_path=HMM_MODEL_PATH):
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def get_current_regime_from_artifact(db_path=DB_PATH, artifact=None, confirmation_days=HMM_CONFIRMATION_DAYS):
    """Load current regime using a saved HMM artifact."""
    artifact = artifact or load_hmm_model()
    if artifact is None:
        return None

    raw = build_raw_feature_matrix(db_path)
    if raw.empty:
        return None

    cols = artifact.get("feature_columns", list(raw.columns))
    raw = raw[cols]
    std_values = artifact["scaler"].transform(raw.values)
    feature_matrix = pd.DataFrame(std_values, index=raw.index, columns=raw.columns)
    info = get_current_regime(
        artifact["model"],
        feature_matrix,
        artifact["label_map"],
        confirmation_days=confirmation_days,
    )
    states = artifact["model"].predict(feature_matrix.values)
    current_state = states[-1]
    days_in_state = 1
    for idx in range(len(states) - 2, -1, -1):
        if states[idx] == current_state:
            days_in_state += 1
        else:
            break
    info["days_in_state"] = days_in_state
    info["as_of"] = str(feature_matrix.index[-1].date())
    return info


def get_paper_trade_regime_stats(db_path=DB_PATH, historical_regime_df=None):
    """Compute paper-trade win rates by HMM regime at trade entry date."""
    if historical_regime_df is None:
        historical_regime_df = compute_historical_regime_features(db_path)
    regime_map = historical_regime_df[["hmm_regime"]].copy()
    regime_map = regime_map.rename(columns={"hmm_regime": "regime"})
    regime_map["date"] = regime_map.index.normalize()

    conn = sqlite3.connect(db_path)
    try:
        trades = pd.read_sql(
            """
            SELECT entry_date, pnl, pct_return
            FROM paper_trades
            WHERE exit_date IS NOT NULL
            ORDER BY entry_date
            """,
            conn,
        )
    finally:
        conn.close()

    if trades.empty:
        return pd.DataFrame(columns=["regime", "n_trades", "win_rate_pct", "avg_pct_return"])

    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    trades["is_win"] = (
        pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0) > 0
    ).astype(int)
    trades["pct_return"] = pd.to_numeric(trades["pct_return"], errors="coerce")
    merged = trades.merge(regime_map, left_on="entry_date", right_on="date", how="left")
    merged["regime"] = merged["regime"].fillna("UNKNOWN")

    stats = (
        merged.groupby("regime")
        .agg(
            n_trades=("is_win", "size"),
            win_rate_pct=("is_win", lambda s: float(np.mean(s) * 100.0)),
            avg_pct_return=("pct_return", "mean"),
        )
        .reset_index()
        .sort_values("regime")
    )
    return stats


if __name__ == "__main__":
    features = build_feature_matrix(DB_PATH)
    model = train_hmm(features)
    if model is None or features.empty:
        print("HMM training failed or no data available.")
    else:
        state_df, label_map = label_states(model, features)
        regime_info = get_current_regime(model, features, label_map, confirmation_days=HMM_CONFIRMATION_DAYS)
        state_stats = state_df["regime"].value_counts(normalize=True).mul(100.0).round(2)

        print("HMM STATE STATISTICS")
        for regime, pct in state_stats.items():
            print(f"  {regime:<6} {pct:>6.2f}%")

        print("\nCURRENT REGIME")
        print(
            f"  {regime_info['regime']}  "
            f"P(bull)={regime_info['prob_bull']:.2f}  "
            f"P(bear)={regime_info['prob_bear']:.2f}  "
            f"P(range)={regime_info['prob_range']:.2f}  "
            f"Confirmed={regime_info['confirmed']}"
        )

        print("\nLAST 30 DAYS")
        print(state_df[["state", "regime"]].tail(30).to_string())
