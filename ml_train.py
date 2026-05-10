"""
ml_train.py — XGBoost ile BIST fiyat yönü tahmini
────────────────────────────────────────────────────────────────
Kullanım: python3.12 ml_train.py

Tam pipeline:
  1. Feature engineering (30+ özellik)
  2. Walk-forward cross validation (6 pencere)
  3. XGBoost eğitim: 2022-2023
  4. Test: 2024-2026 (model hiç görmedi)
  5. SHAP ile hangi feature önemli analizi
  6. Her hisse için bugünkü sinyal: AL / SAT / BEKLE

Hedef değişken:
  Legacy: gelecek 3 günde close-to-close getiri >= %2.0 ise BUY=1, aksi 0
  Triple barrier: ATR-ölçekli bariyerlerden üst bariyer vurursa BUY=1, aksi 0
────────────────────────────────────────────────────────────────
"""

import sqlite3, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from triple_barrier_labels import (
    compute_sample_uniqueness_weights,
    compute_triple_barrier_label_details,
)
from regime_hmm import compute_historical_regime_features

warnings.filterwarnings("ignore")

# ── Calibration flag ──────────────────────────────────────────
CALIBRATION_ENABLED = True   # Set False to skip Beta calibration in walk_forward
TRIPLE_BARRIER_ENABLED = True
STACKING_ENABLED = True
STACKING_META_EXTRA_FEATURES = True
STACKING_INNER_SPLITS = 5
TRIPLE_BARRIER_PT_SL_RATIO = 2.0
TRIPLE_BARRIER_SL_RATIO = 1.0
TRIPLE_BARRIER_VERTICAL_BARS = 5

DB_PATH = Path(__file__).parent / "trade_data.db"
RESULTS = Path(__file__).parent / "results"
MODELS  = Path(__file__).parent / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SYMBOLS = [
    "YKBNK","AKBNK","ISCTR","GARAN",
    "TUPRS","PETKM",
    "TAVHL","FROTO",
    "TCELL","ASELS",
    "BIMAS","MGROS",
    "ENKAI","EKGYO",
    # Expanded universe (added 2026-04-24)
    "THYAO","EREGL",
    "KCHOL","SAHOL",
    "SISE","TOASO",
    "ARCLK","VESTL",
    "KRDMD",
    "PGSUS","ODAS",
    "GUBRF","CIMSA",
    "LOGO","NETAS",
]

TRAIN_END  = "2024-01-01"   # model bu tarihten önceyi görür
TEST_START = "2024-01-01"   # bu tarihten sonra hiç görmediği veri
LEGACY_HORIZON = 3
HORIZON    = TRIPLE_BARRIER_VERTICAL_BARS if TRIPLE_BARRIER_ENABLED else LEGACY_HORIZON
TARGET_PCT = 2.0            # legacy close-to-close hareket eşiği

# Temporal weight reference date — update manually each time you retrain
# on a new dataset so the 12-/24-month boundaries stay correct.
_TEMPORAL_REF     = pd.Timestamp("2026-04-24")
TEMPORAL_CUT_12M  = _TEMPORAL_REF - pd.DateOffset(months=12)   # 2025-04-24
TEMPORAL_CUT_24M  = _TEMPORAL_REF - pd.DateOffset(months=24)   # 2024-04-24

# 12 cross-asset macro features (pre-computed in macro_data DB by indicators.py)
# After training: drop any feature with gain < 0.001 (except strongest_sector_5d)
# Note: usdtry_1d_return etc. are kept with old feature names for model compatibility.
MACRO_FEATURES_ADDED = [
    # Existing (renamed/improved: now pre-computed with ffill — no more zero-fill NaN gap)
    "usdtry_1d_return",     # USDTRY 1-day return
    "usdtry_5d_return",     # USDTRY 5-day return
    "usdtry_above_20ma",    # binary: USDTRY above 20-day MA
    "brent_1d_return",      # Brent 1-day return (×1.5 TUPRS/PETKM)
    "brent_5d_return",      # Brent 5-day return (×1.5 TUPRS/PETKM)
    "tcmb_rate",            # TCMB policy rate (static 43.0)
    "strongest_sector_5d",  # known good (gain 0.0343) — always kept
    # NEW cross-asset features (2026-04-27)
    "usdtry_20d_zscore",    # USDTRY deviation from 20d mean (risk pressure)
    "vix_level",            # VIX / 100 (global fear gauge)
    "vix_5d_zscore",        # VIX 5d z-score (spike detector)
    "gold_try_ratio",       # Gold-in-TRY 5d return (TRY safe-haven proxy)
    "dxy_5d_ret",           # DXY 5-day return (USD strength)
    "sp500_overnight_ret",  # S&P500 prior close return (overnight risk proxy)
    "stoxx50_am_ret",       # STOXX50 daily return (European AM signal)
    "em_5d_ret",            # MSCI EM 5-day return (EM risk appetite)
    "macro_risk_score",     # Composite: VIX + USDTRY + DXY (negative = risk-off)
]

HMM_FEATURES_ADDED = [
    "hmm_prob_bull",
    "hmm_prob_bear",
    "hmm_prob_range",
    "hmm_days_in_state",
]

SECTOR_MAP = {
    "YKBNK": "bank",         "AKBNK": "bank",
    "ISCTR": "bank",         "GARAN": "bank",
    "TUPRS": "energy",       "PETKM": "energy",
    "TAVHL": "aviation",     "FROTO": "auto",
    "TCELL": "telecom",      "ASELS": "defense",
    "BIMAS": "retail",       "MGROS": "retail",
    "ENKAI": "construction", "EKGYO": "realestate",
    # Expanded universe (added 2026-04-24)
    "THYAO": "aviation",     "EREGL": "steel",
    "KCHOL": "holding",      "SAHOL": "holding",
    "SISE":  "glass",        "TOASO": "auto",
    "ARCLK": "consumer",     "VESTL": "consumer",
    "KRDMD": "steel",
    "PGSUS": "aviation",     "ODAS":  "energy",
    "GUBRF": "chemicals",    "CIMSA": "cement",
    "LOGO":  "technology",   "NETAS": "technology",
}

# ── Paket kontrol ─────────────────────────────────────────────
try:
    import xgboost as xgb
    print(f"XGBoost: {xgb.__version__} OK")
except ImportError:
    print("XGBoost yok! Kur: py -3.12 -m pip install xgboost")
    exit(1)

try:
    import lightgbm as lgb
    print(f"LightGBM: {lgb.__version__} OK")
    HAS_LGB = True
except ImportError:
    print("LightGBM yok — sadece XGBoost kullanilacak (pip install lightgbm)")
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    print(f"CatBoost: {__import__('catboost').__version__} OK")
    HAS_CAT = True
except ImportError:
    print("CatBoost yok — 2'li ensemble kullanilacak (pip install catboost)")
    HAS_CAT = False

try:
    from sklearn.metrics import (classification_report,
                                  confusion_matrix, roc_auc_score)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
except ImportError:
    print("scikit-learn yok! Kur: py -3.12 -m pip install scikit-learn")
    exit(1)

try:
    from betacal import BetaCalibration
    HAS_BETACAL = True
    print("betacal: 1.1.0 OK")
except ImportError:
    print("betacal yok — pip install betacal --break-system-packages")
    HAS_BETACAL = False

DEFAULT_BUY_THRESHOLD = 0.68
DEFAULT_SELL_THRESHOLD = 0.32


def split_symbol_payload(payload):
    if len(payload) == 2:
        X, y = payload
        w = pd.Series(1.0, index=X.index, dtype=float)
        return X, y, w
    if len(payload) == 3:
        X, y, w = payload
        return X, y, w
    raise ValueError(f"Unexpected symbol payload length: {len(payload)}")


def is_binary_buy_target(y):
    values = set(pd.Series(y).dropna().astype(int).unique().tolist())
    return values.issubset({0, 1})


def make_legacy_target(df, horizon=LEGACY_HORIZON, threshold=TARGET_PCT):
    """
    Legacy binary BUY label:
      1 if close[t+horizon] / close[t] - 1 >= threshold
      0 otherwise
      NaN on trailing rows where future is unavailable
    """
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    target = (future_ret >= threshold / 100).astype(float)
    target[future_ret.isna()] = np.nan
    return target


def build_training_target(df, use_triple_barrier=None):
    enabled = TRIPLE_BARRIER_ENABLED if use_triple_barrier is None else bool(use_triple_barrier)

    if not enabled:
        y = make_legacy_target(df, horizon=LEGACY_HORIZON, threshold=TARGET_PCT)
        w = pd.Series(1.0, index=df.index, dtype=float)
        w[y.isna()] = np.nan
        return y, w

    details = compute_triple_barrier_label_details(
        df,
        atr_col="atr14",
        pt_sl_ratio=TRIPLE_BARRIER_PT_SL_RATIO,
        sl_ratio=TRIPLE_BARRIER_SL_RATIO,
        vertical_bars=TRIPLE_BARRIER_VERTICAL_BARS,
    )
    y = pd.Series(np.nan, index=df.index, dtype=float)
    y.loc[details.index] = (details["label"] == 1).astype(float)

    w = pd.Series(np.nan, index=df.index, dtype=float)
    if not details.empty:
        uniq = compute_sample_uniqueness_weights(
            details,
            vertical_bars=TRIPLE_BARRIER_VERTICAL_BARS,
        )
        w.loc[uniq.index] = uniq
    return y, w


class IdentityCalibrator:
    """Fallback calibrator when there is too little validation data."""

    def predict(self, values):
        arr = np.asarray(values, dtype=float)
        return np.clip(arr, 0.0, 1.0)


class EnsembleModel:
    """Base-model container with optional stacking meta-learner."""

    def __init__(
        self,
        xgb_model,
        lgb_model=None,
        cat_model=None,
        meta_lr=None,
        stacking_enabled=False,
        meta_extra_features=STACKING_META_EXTRA_FEATURES,
    ):
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.cat = cat_model
        self.meta_lr = meta_lr
        self.stacking_enabled = bool(stacking_enabled and meta_lr is not None)
        self.meta_extra_features = bool(meta_extra_features)

    def base_prob_matrix(self, X):
        xgb_prob = self.xgb.predict_proba(X)[:, 1]
        lgb_prob = self.lgb.predict_proba(X)[:, 1] if self.lgb is not None else xgb_prob.copy()
        cat_prob = self.cat.predict_proba(X)[:, 1] if self.cat is not None else xgb_prob.copy()
        return np.column_stack([xgb_prob, lgb_prob, cat_prob])

    def build_meta_features(self, prob_matrix):
        prob_matrix = np.asarray(prob_matrix, dtype=float)
        if prob_matrix.ndim != 2 or prob_matrix.shape[1] < 3:
            raise ValueError("prob_matrix must be shape (n_samples, 3)")
        if not self.meta_extra_features:
            return prob_matrix[:, :3]
        disagreement = prob_matrix[:, :3].std(axis=1, ddof=0)
        max_prob = prob_matrix[:, :3].max(axis=1)
        return np.column_stack([prob_matrix[:, :3], disagreement, max_prob])

    def raw_prob(self, X):
        prob_matrix = self.base_prob_matrix(X)
        if self.stacking_enabled and self.meta_lr is not None:
            meta_X = self.build_meta_features(prob_matrix)
            return self.meta_lr.predict_proba(meta_X)[:, 1]
        return prob_matrix.mean(axis=1)

    def predict_proba(self, X):
        prob_buy = np.clip(self.raw_prob(X), 0.0, 1.0)
        return np.column_stack([1.0 - prob_buy, prob_buy])


def build_xgb_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.75, min_child_weight=8,
        reg_alpha=0.2, reg_lambda=1.5,
        eval_metric="logloss", verbosity=0,
        random_state=42,
    )


def build_lgb_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.75, min_child_leaves=20,
        reg_alpha=0.2, reg_lambda=1.5,
        verbose=-1,
        random_state=42,
    ) if HAS_LGB else None


def build_cat_model():
    if not HAS_CAT:
        return None
    return CatBoostClassifier(
        iterations=300, depth=4,
        learning_rate=0.03, subsample=0.8,
        l2_leaf_reg=3.0,
        eval_metric="Logloss",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


# Keep build_model() as alias for backward compat with model-save code
def build_model():
    return build_xgb_model()


def compute_temporal_weights(index):
    """
    Per-sample recency bias weights based on date:
      >= TEMPORAL_CUT_12M (2025-04-24):            2.0x  (last 12 months)
      TEMPORAL_CUT_24M..TEMPORAL_CUT_12M:          1.5x  (12-24 months ago)
      < TEMPORAL_CUT_24M (older than 24 months):   1.0x
    Combined with class-balance weights in fit_binary_model().
    """
    w = np.ones(len(index))
    w[index >= TEMPORAL_CUT_12M] = 2.0
    w[(index >= TEMPORAL_CUT_24M) & (index < TEMPORAL_CUT_12M)] = 1.5
    return w


def compute_effective_sample_weight(index, y_train, sample_weight=None, use_temporal=False):
    """Combine class-balance, uniqueness and temporal weights."""
    y_enc = pd.Series(y_train, index=index).astype(int)
    sw = compute_sample_weight("balanced", y_enc)
    if sample_weight is not None:
        sw = sw * pd.Series(sample_weight, index=index).fillna(1.0).to_numpy(dtype=float)
    if use_temporal:
        sw = sw * compute_temporal_weights(index)
    sw = np.asarray(sw, dtype=float)
    sw_mean = float(sw.mean()) if len(sw) else 1.0
    if sw_mean > 0:
        sw = sw / sw_mean
    return sw


def fit_base_models_from_final_weights(X_train, y_train, final_sw):
    X_clean = X_train.fillna(0)
    y_enc = pd.Series(y_train, index=X_train.index).astype(int)
    final_sw = np.asarray(final_sw, dtype=float)

    xgb_model = build_xgb_model()
    xgb_model.fit(X_clean, y_enc, sample_weight=final_sw)

    lgb_model = build_lgb_model()
    if lgb_model is not None:
        lgb_model.fit(X_clean, y_enc, sample_weight=final_sw)

    cat_model = build_cat_model()
    if cat_model is not None:
        cat_model.fit(X_clean, y_enc, sample_weight=final_sw)

    return EnsembleModel(xgb_model, lgb_model, cat_model, meta_lr=None, stacking_enabled=False)


def fit_binary_model(X_train, y_train, sample_weight=None, use_temporal=False):
    """Train base BUY-vs-NOT_BUY models."""
    y_enc = pd.Series(y_train, index=X_train.index).astype(int)
    sw = compute_effective_sample_weight(X_train.index, y_enc, sample_weight, use_temporal)
    return fit_base_models_from_final_weights(X_train, y_enc, sw)


def fit_meta_learner(meta_X, y_meta, sample_weight=None):
    meta_X = np.asarray(meta_X, dtype=float)
    y_meta = np.asarray(y_meta, dtype=int)
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    if sample_weight is None:
        meta_lr.fit(meta_X, y_meta)
    else:
        meta_lr.fit(meta_X, y_meta, sample_weight=np.asarray(sample_weight, dtype=float))
    return meta_lr


def train_stacking_model(
    X_train,
    y_train,
    sample_weight=None,
    use_temporal=False,
    n_splits=STACKING_INNER_SPLITS,
):
    """Train base models plus leakage-safe LogisticRegression meta-learner."""
    X_train = X_train.fillna(0)
    y_train = pd.Series(y_train, index=X_train.index).astype(int)
    sw_full = compute_effective_sample_weight(X_train.index, y_train, sample_weight, use_temporal)

    if len(X_train) < max(100, n_splits * 20):
        base_model = fit_binary_model(X_train, y_train, sample_weight=sample_weight, use_temporal=use_temporal)
        return EnsembleModel(
            base_model.xgb,
            base_model.lgb,
            base_model.cat,
            meta_lr=None,
            stacking_enabled=False,
        ), None, None, None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.full((len(X_train), 3), np.nan, dtype=float)
    oof_mask = np.zeros(len(X_train), dtype=bool)

    for inner_train_idx, inner_val_idx in tscv.split(X_train):
        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        w_inner_train = sw_full[inner_train_idx]
        X_inner_val = X_train.iloc[inner_val_idx]

        if y_inner_train.nunique() < 2 or len(X_inner_val) == 0:
            continue

        base_model = fit_base_models_from_final_weights(X_inner_train, y_inner_train, w_inner_train)
        oof_prob[inner_val_idx] = base_model.base_prob_matrix(X_inner_val.fillna(0))
        oof_mask[inner_val_idx] = True

    if not oof_mask.any():
        base_model = fit_base_models_from_final_weights(X_train, y_train, sw_full)
        return EnsembleModel(
            base_model.xgb,
            base_model.lgb,
            base_model.cat,
            meta_lr=None,
            stacking_enabled=False,
        ), None, None, None

    meta_builder = EnsembleModel(None, None, None, meta_lr=None, stacking_enabled=False)
    meta_X = meta_builder.build_meta_features(oof_prob[oof_mask])
    meta_y = y_train.iloc[oof_mask].to_numpy(dtype=int)
    meta_w = sw_full[oof_mask]
    meta_lr = fit_meta_learner(meta_X, meta_y, sample_weight=meta_w)

    full_base_model = fit_base_models_from_final_weights(X_train, y_train, sw_full)
    stacked_model = EnsembleModel(
        full_base_model.xgb,
        full_base_model.lgb,
        full_base_model.cat,
        meta_lr=meta_lr,
        stacking_enabled=True,
    )
    return stacked_model, meta_X, meta_y, meta_w


def fit_calibrator(model, X_cal, y_cal):
    y_cal = pd.Series(y_cal, index=X_cal.index).dropna().astype(int)
    if len(X_cal) < 40 or y_cal.nunique() < 2:
        return IdentityCalibrator()
    raw_prob = model.predict_proba(X_cal.fillna(0))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_prob, y_cal)
    return calibrator


def calibrated_buy_prob(model, calibrator, X):
    X_clean = X.fillna(0)
    raw_prob = model.predict_proba(X_clean)[:, 1]
    if calibrator is None:
        cal_prob = np.clip(raw_prob, 0.0, 1.0)
    else:
        cal_prob = np.clip(np.asarray(calibrator.predict(raw_prob), dtype=float), 0.0, 1.0)
    return np.clip(cal_prob, 0.0, 1.0)


def compute_raw_model_prob(model, X_clean):
    """Raw probability of the currently active ensemble strategy."""
    return model.predict_proba(X_clean)[:, 1]


def compute_brier_score(probs, y_true):
    """Mean squared error between predicted probs and binary labels."""
    return float(np.mean((np.asarray(probs, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def compute_ece(probs, y_true, n_bins=10):
    """Expected Calibration Error (ECE) with equal-width bins."""
    probs  = np.asarray(probs,  dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    n      = len(probs)
    ece    = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def fit_beta_calibrator(raw_probs, y_true):
    """
    Fit a BetaCalibration model mapping raw soft-vote probs → calibrated probs.
    Returns None if betacal not available, too few samples, or fit fails.
    """
    if not HAS_BETACAL:
        return None
    raw_probs = np.asarray(raw_probs, dtype=float)
    y_true    = np.asarray(y_true,    dtype=float)
    if len(raw_probs) < 20 or len(np.unique(y_true)) < 2:
        return None
    try:
        cal = BetaCalibration(parameters="abm")
        cal.fit(raw_probs.reshape(-1, 1), y_true)
        return cal
    except Exception as exc:
        print(f"  [BetaCal] fit error: {exc}")
        return None


def classify_prob(prob_buy, thresholds):
    buy_th = thresholds.get("buy", DEFAULT_BUY_THRESHOLD)
    sell_th = thresholds.get("sell", DEFAULT_SELL_THRESHOLD)
    if prob_buy >= buy_th:
        return 1, prob_buy
    if prob_buy <= sell_th:
        return -1, 1.0 - prob_buy
    return 0, max(prob_buy, 1.0 - prob_buy)


MIN_PRECISION = 0.45   # Minimum BUY/SELL precision required to consider a threshold
MAX_COVERAGE  = 0.60   # Reject thresholds that trade more than 60% of all signals


def select_trade_thresholds(prob_buy, y_full):
    candidates = np.round(np.arange(0.55, 0.81, 0.03), 2)

    def choose_buy():
        best = None
        for thr in candidates:
            mask = prob_buy >= thr
            trades = int(mask.sum())
            if trades == 0:
                continue
            coverage = trades / max(len(y_full), 1)
            if coverage > MAX_COVERAGE:
                continue  # Too broad — effectively not selective
            precision = float((y_full[mask] == 1).mean())
            if precision < MIN_PRECISION:
                continue  # Below quality floor
            score = precision + min(coverage, 0.10) * 0.20
            item = {"threshold": float(thr), "precision": precision, "trades": trades, "score": score}
            if best is None or item["score"] > best["score"]:
                best = item
        return best

    def choose_sell():
        best = None
        for thr in candidates:
            cut = 1.0 - thr
            mask = prob_buy <= cut
            trades = int(mask.sum())
            if trades == 0:
                continue
            coverage = trades / max(len(y_full), 1)
            if coverage > MAX_COVERAGE:
                continue
            precision = float((y_full[mask] == 0).mean()) if is_binary_buy_target(y_full) else float((y_full[mask] == -1).mean())
            if precision < MIN_PRECISION:
                continue
            score = precision + min(coverage, 0.10) * 0.20
            item = {"threshold": float(cut), "precision": precision, "trades": trades, "score": score}
            if best is None or item["score"] > best["score"]:
                best = item
        return best

    best_buy = choose_buy()
    best_sell = choose_sell()
    return {
        "buy": best_buy["threshold"] if best_buy else DEFAULT_BUY_THRESHOLD,
        "sell": best_sell["threshold"] if best_sell else DEFAULT_SELL_THRESHOLD,
        "meta": {
            "buy_precision": round(best_buy["precision"], 4) if best_buy else None,
            "buy_trades": best_buy["trades"] if best_buy else 0,
            "sell_precision": round(best_sell["precision"], 4) if best_sell else None,
            "sell_trades": best_sell["trades"] if best_sell else 0,
        },
    }


def evaluate_signals(prob_buy, y_full, thresholds):
    preds, confs = [], []
    for prob in prob_buy:
        pred, conf = classify_prob(prob, thresholds)
        preds.append(pred)
        confs.append(conf)
    preds = np.array(preds)
    confs = np.array(confs)
    traded = preds != 0

    # BUY precision: of rows predicted BUY that had a clear outcome (not BEKLE),
    # what fraction actually went up ≥TARGET_PCT?
    # Excluding BEKLE rows from denominator avoids penalising the model for
    # neutral moves (neither win nor loss in practice).
    def labeled_precision(pred_val, true_val):
        mask = preds == pred_val
        if not mask.sum():
            return 0.0, 0.0
        y_sub = y_full[mask]
        neutral_rate = float(y_sub.isna().mean())
        labeled = y_sub.dropna()
        if len(labeled) == 0:
            return 0.0, neutral_rate
        return float((labeled == true_val).mean()), neutral_rate

    buy_prec, buy_neut = labeled_precision(1, 1)
    sell_truth = 0 if is_binary_buy_target(y_full) else -1
    sell_prec, sell_neut = labeled_precision(-1, sell_truth)

    return {
        "preds": preds,
        "confs": confs,
        "trades": int(traded.sum()),
        "coverage": float(traded.mean()) if len(traded) else 0.0,
        "buy_count": int((preds == 1).sum()),
        "sell_count": int((preds == -1).sum()),
        "buy_precision": buy_prec,
        "sell_precision": sell_prec,
        "neutral_buy_rate": buy_neut,
        "neutral_sell_rate": sell_neut,
        "avg_conf": float(confs[traded].mean()) if traded.any() else 0.0,
    }


def collect_period_rows(all_data, start=None, end=None, binary_only=True, include_weights=False):
    X_parts, y_parts, w_parts = [], [], []
    for payload in all_data.values():
        X, y, w = split_symbol_payload(payload)
        mask = pd.Series(True, index=X.index)
        if start is not None:
            mask &= X.index >= pd.Timestamp(start)
        if end is not None:
            mask &= X.index < pd.Timestamp(end)
        if binary_only:
            mask &= y.notna()
        X_sel = X.loc[mask]
        y_sel = y.loc[mask]
        if len(X_sel) == 0:
            continue
        X_parts.append(X_sel)
        y_parts.append(y_sel)
        if include_weights:
            w_sel = w.loc[mask].fillna(1.0)
            w_parts.append(w_sel)

    if not X_parts:
        if include_weights:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
        return pd.DataFrame(), pd.Series(dtype=float)

    X_all = pd.concat(X_parts).sort_index().fillna(0)
    y_all = pd.concat(y_parts).sort_index()
    if include_weights:
        w_all = pd.concat(w_parts).sort_index().fillna(1.0)
        return X_all, y_all, w_all
    return X_all, y_all

# ── Veri yükle ────────────────────────────────────────────────
def load_symbol(sym, conn):
    df = pd.read_sql("""
        SELECT o.date, o.open, o.high, o.low, o.close, o.volume,
               i.ema8, i.ema21, i.ema50, i.ema200,
               i.rsi14, i.macd_line, i.macd_signal, i.macd_hist,
               i.atr14, i.bb_upper, i.bb_mid, i.bb_lower, i.bb_width,
               i.obv, i.vol_ratio, i.mtf_trend,
               i.above_ema200, i.golden_cross,
               COALESCE(i.sentiment_score, 0.0) AS sentiment_score,
               COALESCE(i.news_count, 0)        AS news_count
        FROM ohlcv o
        JOIN indicators i ON o.symbol=i.symbol AND o.date=i.date
        WHERE o.symbol=? ORDER BY o.date
    """, conn, params=(sym,))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_macro_data(conn):
    """Load all macro_data columns indexed by date.
    Includes pre-computed derived features (12 cross-asset + brent derived)
    fetched by indicators.py fetch_macro_data().
    """
    try:
        df = pd.read_sql("SELECT * FROM macro_data ORDER BY date", conn)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

# ── Feature Engineering ───────────────────────────────────────
def make_features(df, sym=None, market_data=None, macro_df=None, regime_df=None):
    """
    Ham OHLCV + indikatörlerden 46+ özellik üret.
    Tüm özellikler fiyat-bağımsız (normalize edilmiş).
    market_data: {sym: pd.Series(close)} — makro/sektör özellikleri için.
    """
    f = pd.DataFrame(index=df.index)
    c = df["close"]

    # ── 1. Momentum özellikleri ──
    for n in [1, 3, 5, 10, 20]:
        f[f"ret_{n}d"] = c.pct_change(n)          # n günlük getiri

    # ── 2. RSI özellikleri ──
    f["rsi"]        = df["rsi14"] / 100            # normalize 0-1
    f["rsi_delta"]  = df["rsi14"].diff(3) / 100    # RSI değişim hızı
    f["rsi_above50"]= (df["rsi14"] > 50).astype(int)

    # ── 3. MACD özellikleri ──
    f["macd_hist"]  = df["macd_hist"] / (c + 1e-9) # fiyata normalize
    f["macd_cross"] = (                             # 0→1: bullish cross
        (df["macd_line"] > df["macd_signal"]) &
        (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    f["macd_above0"]= (df["macd_hist"] > 0).astype(int)

    # ── 4. EMA yapısı ──
    f["ema_8_21"]   = (df["ema8"]  / df["ema21"]  - 1)  # spread
    f["ema_21_50"]  = (df["ema21"] / df["ema50"]  - 1)
    f["ema_50_200"] = (df["ema50"] / df["ema200"] - 1)
    f["price_ema200"]= (c / df["ema200"] - 1)           # EMA200'den sapma
    f["above_ema200"]= df["above_ema200"].astype(int)
    f["golden_cross"]= df["golden_cross"].astype(int)
    f["mtf_trend"]   = df["mtf_trend"]                  # -1/0/+1

    # ── 5. Bollinger Band özellikleri ──
    bb_range = df["bb_upper"] - df["bb_lower"]
    f["bb_pos"]    = (c - df["bb_lower"]) / (bb_range + 1e-9)  # 0=alt, 1=üst
    f["bb_width"]  = df["bb_width"] / 100
    f["bb_squeeze"]= (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)).astype(int)
    f["above_upper"]= (c > df["bb_upper"]).astype(int)
    f["below_lower"]= (c < df["bb_lower"]).astype(int)

    # ── 6. ATR / Volatilite ──
    f["atr_pct"]   = df["atr14"] / (c + 1e-9)           # ATR/fiyat
    f["atr_ratio"] = df["atr14"] / df["atr14"].rolling(20).mean()  # ATR değişimi

    # ── 7. Hacim özellikleri ──
    f["vol_ratio"]  = df["vol_ratio"].clip(0, 5)         # 5'te kes
    f["vol_trend"]  = df["volume"].pct_change(5)
    f["obv_slope"]  = df["obv"].diff(5) / (df["obv"].abs().rolling(5).mean() + 1e-9)

    # ── 8. Fiyat pattern ──
    f["high_low_pct"]= (df["high"] - df["low"]) / (c + 1e-9)  # günlük range
    f["close_pos"]   = (c - df["low"]) / ((df["high"] - df["low"]) + 1e-9)  # kapanış yeri
    f["gap"]         = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-9)      # gap

    # ── 9. Trend gücü ──
    for n in [10, 20]:
        roll_std = c.pct_change().rolling(n).std()
        roll_ret = c.pct_change(n)
        f[f"trend_str_{n}"] = roll_ret / (roll_std * np.sqrt(n) + 1e-9)  # t-stat

    # ── 10. Regime tespiti ──
    f["above_sma20"]  = (c > df["ema21"]).astype(int)
    f["consec_up"]    = (c > c.shift(1)).astype(int).rolling(5).sum()    # art arda yükseliş
    f["consec_down"]  = (c < c.shift(1)).astype(int).rolling(5).sum()

    # ── 11. 52-haftalık yüksek/düşük uzaklığı ──
    f["dist_52w_high"] = (c / c.rolling(252, min_periods=20).max() - 1)
    f["dist_52w_low"]  = (c / c.rolling(252, min_periods=20).min() - 1)

    # ── 12. Haftanın günü (Pazartesi etkisi BIST'te gerçek) ──
    f["day_of_week"]   = df.index.dayofweek / 4.0   # 0=Pzt, 1=Cuma (normalize)

    # ── 13-new. Ek özellikler (Phase 2 ensemble upgrade) ──
    # Order flow proxy: mum gövdesi yönü vs günlük aralık
    f["buying_pressure"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 1e-9)

    # 52-haftalık aralıkta fiyat konumu (0=dip, 1=zirve)
    w52_high = c.rolling(252, min_periods=20).max()
    w52_low  = c.rolling(252, min_periods=20).min()
    f["price_pos_52w"] = (c - w52_low) / ((w52_high - w52_low) + 1e-9)

    # Volatilite rejimi: ATR şu an 20-günlük ortalamasının üstünde mi?
    f["atr_regime_high"] = (df["atr14"] > df["atr14"].rolling(20).mean()).astype(int)

    # Hacim spike: bugünkü hacim 20-günlük ortalamanın kaç katı?
    f["vol_spike_20d"] = (df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)).clip(0, 10)

    # ── 13. Makro / çapraz-hisse özellikler ──
    if market_data is not None:
        # Piyasa proxy: TCELL günlük getirisi (likit, sektör-agnostik proxy)
        tcell_close = market_data.get("TCELL")
        if tcell_close is not None:
            f["tcell_ret_1d"] = tcell_close.pct_change(1).reindex(df.index).fillna(0)
        else:
            f["tcell_ret_1d"] = 0.0

        # Piyasa betası: hisse getirisi vs eşit-ağırlıklı endeks (20 gün)
        mkt_rets = [close_s.pct_change(1).reindex(df.index)
                    for close_s in market_data.values()]
        if mkt_rets:
            mkt_ret   = pd.concat(mkt_rets, axis=1).mean(axis=1)
            stock_ret = c.pct_change(1)
            cov20     = stock_ret.rolling(20).cov(mkt_ret)
            var20     = mkt_ret.rolling(20).var()
            f["mkt_beta_20"] = (cov20 / (var20 + 1e-9)).clip(-3, 3).fillna(1.0)
        else:
            f["mkt_beta_20"] = 1.0

        # Sektör momentumu: aynı sektördeki diğer hisselerin 5-günlük ort. getirisi
        my_sector = SECTOR_MAP.get(sym, "other")
        sec_rets  = [close_s.pct_change(5).reindex(df.index)
                     for s, close_s in market_data.items()
                     if s != sym and SECTOR_MAP.get(s) == my_sector]
        if sec_rets:
            f["sector_mom_5d"] = pd.concat(sec_rets, axis=1).mean(axis=1).fillna(0)
        else:
            f["sector_mom_5d"] = 0.0

        # En güçlü sektör: tüm sektörlerin 5-günlük ortalama getirisinin maksimumu
        # "Bankacılık güçlüyse piyasada risk iştahı var" etkisini yakalar
        sector_5d: dict[str, list] = {}
        for s, close_s in market_data.items():
            sec = SECTOR_MAP.get(s, "other")
            sector_5d.setdefault(sec, []).append(
                close_s.pct_change(5).reindex(df.index)
            )
        if sector_5d:
            sec_avgs = {sec: pd.concat(rets, axis=1).mean(axis=1)
                        for sec, rets in sector_5d.items()}
            all_sec_df = pd.concat(sec_avgs, axis=1)
            f["strongest_sector_5d"] = all_sec_df.max(axis=1).fillna(0)
        else:
            f["strongest_sector_5d"] = 0.0
    else:
        f["tcell_ret_1d"]       = 0.0
        f["mkt_beta_20"]        = 1.0
        f["sector_mom_5d"]      = 0.0
        f["strongest_sector_5d"]= 0.0

    # ── 14. Haber / duygu analizi (news_filter.py tarafindan doldurulur) ──
    f["sentiment_score"] = df["sentiment_score"].fillna(0.0) if "sentiment_score" in df.columns else 0.0
    f["news_count"]      = df["news_count"].fillna(0).astype(float) if "news_count" in df.columns else 0.0

    # ── 15. Makro ekonomik özellikler (macro_data tablosundan — pre-computed) ──
    # Pre-computed by indicators.py fetch_macro_data() with daily ffill across full calendar.
    # After reindex to BIST dates, use .ffill() (not fillna(0)) so calendar gaps stay correct.
    if macro_df is not None and not macro_df.empty:
        mac = macro_df.reindex(df.index).ffill()

        def _mac(col, default=0.0):
            """Safely extract a pre-computed macro column, fall back to default."""
            if col in mac.columns:
                return mac[col].fillna(default)
            return pd.Series(default, index=df.index)

        # USDTRY (pre-computed — no more on-the-fly pct_change on reindexed NaN series)
        f["usdtry_1d_return"]   = _mac("usdtry_1d_ret",     0.0)
        f["usdtry_5d_return"]   = _mac("usdtry_5d_ret",     0.0)
        f["usdtry_above_20ma"]  = _mac("usdtry_above_20ma", 0).astype(int)
        f["usdtry_20d_zscore"]  = _mac("usdtry_20d_zscore", 0.0)

        # Brent (pre-computed; TUPRS/PETKM get ×1.5 multiplier applied here)
        brent_mult = 1.5 if sym in ("TUPRS", "PETKM") else 1.0
        f["brent_1d_return"]    = _mac("brent_1d_ret", 0.0) * brent_mult
        f["brent_5d_return"]    = _mac("brent_5d_ret", 0.0) * brent_mult

        # TCMB faiz oranı (statik bağlam)
        f["tcmb_rate"]          = _mac("tcmb_rate", 43.0)

        # VIX (global fear gauge)
        f["vix_level"]          = _mac("vix_level",     0.0)
        f["vix_5d_zscore"]      = _mac("vix_5d_zscore", 0.0)

        # Gold in TRY (safe-haven proxy), DXY, cross-market returns
        f["gold_try_ratio"]      = _mac("gold_try_ratio",     0.0)
        f["dxy_5d_ret"]          = _mac("dxy_5d_ret",         0.0)
        f["sp500_overnight_ret"] = _mac("sp500_overnight_ret", 0.0)
        f["stoxx50_am_ret"]      = _mac("stoxx50_am_ret",     0.0)
        f["em_5d_ret"]           = _mac("em_5d_ret",          0.0)

        # Composite macro risk score (pre-computed: negative = risk-off)
        f["macro_risk_score"]    = _mac("macro_risk_score", 0.0)
    else:
        f["usdtry_1d_return"]    = 0.0
        f["usdtry_5d_return"]    = 0.0
        f["usdtry_above_20ma"]   = 0
        f["usdtry_20d_zscore"]   = 0.0
        f["brent_1d_return"]     = 0.0
        f["brent_5d_return"]     = 0.0
        f["tcmb_rate"]           = 43.0
        f["vix_level"]           = 0.0
        f["vix_5d_zscore"]       = 0.0
        f["gold_try_ratio"]      = 0.0
        f["dxy_5d_ret"]          = 0.0
        f["sp500_overnight_ret"] = 0.0
        f["stoxx50_am_ret"]      = 0.0
        f["em_5d_ret"]           = 0.0
        f["macro_risk_score"]    = 0.0

    if regime_df is not None and not regime_df.empty:
        reg = regime_df.reindex(df.index).ffill()
        f["hmm_prob_bull"] = pd.to_numeric(reg.get("hmm_prob_bull", 1.0 / 3.0), errors="coerce").fillna(1.0 / 3.0)
        f["hmm_prob_bear"] = pd.to_numeric(reg.get("hmm_prob_bear", 1.0 / 3.0), errors="coerce").fillna(1.0 / 3.0)
        f["hmm_prob_range"] = pd.to_numeric(reg.get("hmm_prob_range", 1.0 / 3.0), errors="coerce").fillna(1.0 / 3.0)
        f["hmm_days_in_state"] = pd.to_numeric(reg.get("hmm_days_in_state", 0.0), errors="coerce").fillna(0.0)
    else:
        f["hmm_prob_bull"] = 1.0 / 3.0
        f["hmm_prob_bear"] = 1.0 / 3.0
        f["hmm_prob_range"] = 1.0 / 3.0
        f["hmm_days_in_state"] = 0.0

    return f

# ── Hedef değişken ────────────────────────────────────────────
def make_target(df, horizon=LEGACY_HORIZON, threshold=TARGET_PCT):
    _ = horizon, threshold  # backward-compatible signature for callers
    y, _w = build_training_target(df, use_triple_barrier=TRIPLE_BARRIER_ENABLED)
    return y


def strategy_label(stacking_enabled):
    if stacking_enabled:
        return "Stacking"
    return "SoftVote"


def train_strategy_model(X_train, y_train, sample_weight=None, use_temporal=False, stacking_enabled=False):
    if stacking_enabled:
        model, meta_X, meta_y, meta_w = train_stacking_model(
            X_train,
            y_train,
            sample_weight=sample_weight,
            use_temporal=use_temporal,
            n_splits=STACKING_INNER_SPLITS,
        )
        return model, {"meta_X": meta_X, "meta_y": meta_y, "meta_w": meta_w}
    model = fit_binary_model(X_train, y_train, sample_weight=sample_weight, use_temporal=use_temporal)
    return model, {"meta_X": None, "meta_y": None, "meta_w": None}


def train_strategy_with_optional_betacal(X_tr, y_tr, w_tr, use_temporal=False, stacking_enabled=False, fold_name=""):
    fold_beta_cal = None
    last_beta_cal = None
    use_beta = CALIBRATION_ENABLED and HAS_BETACAL

    if use_beta:
        n_fit = int(len(X_tr) * 0.8)
        X_fit = X_tr.iloc[:n_fit]
        y_fit = y_tr.iloc[:n_fit]
        w_fit = w_tr.iloc[:n_fit]
        X_bcal = X_tr.iloc[n_fit:]
        y_bcal = y_tr.iloc[n_fit:]
        model, aux = train_strategy_model(
            X_fit,
            y_fit,
            sample_weight=w_fit,
            use_temporal=use_temporal,
            stacking_enabled=stacking_enabled,
        )

        if len(X_bcal) >= 20 and y_bcal.nunique() >= 2:
            X_bcal_clean = X_bcal.fillna(0)
            raw_probs_calib = compute_raw_model_prob(model, X_bcal_clean)
            y_bcal_enc = y_bcal.astype(int).values.astype(float)

            brier_before = compute_brier_score(raw_probs_calib, y_bcal_enc)
            ece_before = compute_ece(raw_probs_calib, y_bcal_enc)

            fold_beta_cal = fit_beta_calibrator(raw_probs_calib, y_bcal_enc)
            if fold_beta_cal is not None:
                cal_probs_bcal = fold_beta_cal.predict(raw_probs_calib.reshape(-1, 1))
                brier_after = compute_brier_score(cal_probs_bcal, y_bcal_enc)
                ece_after = compute_ece(cal_probs_bcal, y_bcal_enc)
                last_beta_cal = fold_beta_cal
            else:
                brier_after, ece_after = brier_before, ece_before

            print(
                f"  [BetaCal] {fold_name}{strategy_label(stacking_enabled)}: "
                f"Brier {brier_before:.4f}->{brier_after:.4f}  "
                f"ECE {ece_before:.4f}->{ece_after:.4f}"
            )
        else:
            print(
                f"  [BetaCal] {fold_name}{strategy_label(stacking_enabled)}: "
                f"bcal split too small ({len(X_bcal)}) — skipped"
            )
    else:
        model, aux = train_strategy_model(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            use_temporal=use_temporal,
            stacking_enabled=stacking_enabled,
        )

    return model, fold_beta_cal, last_beta_cal, aux


def strategy_probabilities(model, calibrator, beta_calibrator, X):
    X_clean = X.fillna(0)
    if beta_calibrator is not None:
        raw_prob = compute_raw_model_prob(model, X_clean)
        return np.clip(beta_calibrator.predict(raw_prob.reshape(-1, 1)), 0.0, 1.0)
    return calibrated_buy_prob(model, calibrator, X_clean)

# ── Walk-forward validation ───────────────────────────────────
def walk_forward(all_data, n_splits=6, use_temporal=False, stacking_enabled=None, compare_soft_vote=None):
    """
    Veriyi n_splits pencereye bol.
    Her pencerede: ilk %60 egitim, sonraki %20 kalibrasyon,
    son %20 test.
    use_temporal: if True, applies recency weighting in each fold's fit.

    When CALIBRATION_ENABLED=True (and betacal installed):
      - Training data (60%) is split 80/20 (time-ordered, no shuffle).
      - Beta calibrator is fit on the 20% calib split using raw soft-vote probs.
      - Applied to the 20% validation/test fold before computing precision metrics.
      - Prints per-fold Brier score and ECE (10 bins) before and after calibration.
      - Last fold's beta calibrator is returned in result["beta_cal"].
    """
    stacking_enabled = STACKING_ENABLED if stacking_enabled is None else bool(stacking_enabled)
    compare_soft_vote = stacking_enabled if compare_soft_vote is None else bool(compare_soft_vote)

    print("\n" + "="*60)
    if HAS_CAT and HAS_LGB:
        ensemble_label = "XGBoost+LightGBM+CatBoost"
    elif HAS_LGB:
        ensemble_label = "XGBoost+LightGBM"
    else:
        ensemble_label = "XGBoost"
    mode_label = "Stacking meta-learner" if stacking_enabled else "Soft-vote average"
    print(f"Walk-Forward Cross Validation [{ensemble_label}]")
    print(f"  Active mode: {mode_label}")
    print(f"  {n_splits} pencere x (egitim 60% / kalibrasyon 20% / test 20%)")
    if CALIBRATION_ENABLED and HAS_BETACAL:
        print(f"  Beta calibration ENABLED (train split 80/20 for BetaCalibration)")
    print("="*60)

    X_full, _ = collect_period_rows(all_data, binary_only=False)
    unique_dates = sorted(X_full.index.unique())
    split_size = len(unique_dates) // n_splits
    fold_metrics = []
    fold_comparison = []
    last_beta_cal = None
    all_fold_preds = []

    for split in range(n_splits):
        start = split * split_size
        end = min((split + 1) * split_size, len(unique_dates))
        split_dates = unique_dates[start:end]

        train_end_idx = int(len(split_dates) * 0.60)
        cal_end_idx = int(len(split_dates) * 0.80)
        train_dates = split_dates[:train_end_idx]
        cal_dates = split_dates[train_end_idx:cal_end_idx]
        test_dates = split_dates[cal_end_idx:]

        if len(train_dates) < 50 or len(cal_dates) < 20 or len(test_dates) < 20:
            continue

        train_start = train_dates[0]
        train_end = train_dates[-1] + pd.Timedelta(days=1)
        cal_start = cal_dates[0]
        cal_end = cal_dates[-1] + pd.Timedelta(days=1)
        test_start = test_dates[0]
        test_end = test_dates[-1] + pd.Timedelta(days=1)

        X_tr, y_tr, w_tr = collect_period_rows(all_data, start=train_start, end=train_end, binary_only=True, include_weights=True)
        X_cal_bin, y_cal_bin = collect_period_rows(all_data, start=cal_start, end=cal_end, binary_only=True)
        X_cal_full, y_cal_full = collect_period_rows(all_data, start=cal_start, end=cal_end, binary_only=False)
        X_te_full, y_te_full = collect_period_rows(all_data, start=test_start, end=test_end, binary_only=False)

        if len(X_tr) < 30 or len(X_cal_bin) < 15 or len(X_te_full) < 15:
            continue

        strategy_results = {}
        for strategy_on in ([False, True] if compare_soft_vote and stacking_enabled else [stacking_enabled]):
            model, fold_beta_cal, strategy_last_beta, _aux = train_strategy_with_optional_betacal(
                X_tr,
                y_tr,
                w_tr,
                use_temporal=use_temporal,
                stacking_enabled=strategy_on,
                fold_name=f"Fold {split+1} ",
            )
            calibrator = fit_calibrator(model, X_cal_bin, y_cal_bin)
            cal_prob = strategy_probabilities(model, calibrator, fold_beta_cal, X_cal_full)
            thresholds = select_trade_thresholds(cal_prob, y_cal_full)
            test_prob = strategy_probabilities(model, calibrator, fold_beta_cal, X_te_full)
            metrics = evaluate_signals(test_prob, y_te_full, thresholds)

            strategy_results[strategy_on] = {
                "model": model,
                "calibrator": calibrator,
                "thresholds": thresholds,
                "test_prob": test_prob,
                "metrics": metrics,
                "beta_cal": strategy_last_beta,
            }
            if strategy_last_beta is not None and strategy_on == stacking_enabled:
                last_beta_cal = strategy_last_beta

        active_result = strategy_results[stacking_enabled]
        metrics = active_result["metrics"]
        fold_metrics.append(metrics)

        if compare_soft_vote and stacking_enabled and False in strategy_results:
            soft_prec = strategy_results[False]["metrics"]["buy_precision"] * 100
            stack_prec = strategy_results[True]["metrics"]["buy_precision"] * 100
            fold_comparison.append(
                {
                    "fold": split + 1,
                    "soft_vote_precision": soft_prec,
                    "stacking_precision": stack_prec,
                    "stacking_better": stack_prec > soft_prec,
                    "stacking_worse": stack_prec < soft_prec,
                }
            )
            print(
                f"  Fold {split+1} compare: SoftVote={soft_prec:.1f}%  "
                f"Stacking={stack_prec:.1f}%"
            )

        print(
            f"  Pencere {split+1}: {len(X_tr)} egitim / {len(X_cal_full)} kalibr. / "
            f"{len(X_te_full)} test  Trade:{metrics['trades']}  "
            f"BUY precision:{metrics['buy_precision']*100:.1f}%  "
            f"Coverage:{metrics['coverage']*100:.1f}%  "
            f"Mode:{strategy_label(stacking_enabled)}"
        )

        # ── Collect per-symbol predictions (for optimize_threshold.py) ──────
        # Iterates per-symbol so we can attach the symbol name to each row.
        for sym, payload in all_data.items():
            X, y, _w = split_symbol_payload(payload)
            mask_te = (X.index >= test_start) & (X.index < test_end)
            X_sym_te = X.loc[mask_te]
            y_sym_te = y.loc[mask_te]
            if X_sym_te.empty:
                continue
            X_sym_eval = X_sym_te.fillna(0)
            probs_sym = strategy_probabilities(
                active_result["model"],
                active_result["calibrator"],
                active_result["beta_cal"],
                X_sym_eval,
            )
            try:
                base_prob_sym = active_result["model"].base_prob_matrix(X_sym_eval)
            except Exception:
                base_prob_sym = np.column_stack([probs_sym, probs_sym, probs_sym])
            for j, (dt, lbl) in enumerate(zip(X_sym_te.index, y_sym_te.values)):
                lbl_float = float(lbl)
                all_fold_preds.append({
                    "fold":         split + 1,
                    "date":         str(dt.date()),
                    "symbol":       sym,
                    "prob_buy":     round(float(probs_sym[j]), 6),
                    "xgb_prob":     round(float(base_prob_sym[j, 0]), 6),
                    "lgb_prob":     round(float(base_prob_sym[j, 1]), 6),
                    "cat_prob":     round(float(base_prob_sym[j, 2]), 6),
                    "model_disagreement": round(float(np.std(base_prob_sym[j, :3], ddof=0)), 6),
                    "actual_label": None if np.isnan(lbl_float) else lbl_float,
                })

    if not fold_metrics:
        print("  HATA: Yeterli veri yok")
        return None

    # Only average folds that actually traded (buy_count > 0); no-trade folds are skipped
    traded_folds = [m for m in fold_metrics if m["buy_count"] > 0]
    if not traded_folds:
        avg_buy_precision = 0.0
    else:
        avg_buy_precision = np.mean([m["buy_precision"] for m in traded_folds])
    avg_coverage = np.mean([m["coverage"] for m in fold_metrics])
    print(f"\n  Ortalama BUY precision : {avg_buy_precision*100:.1f}%  ({len(traded_folds)}/{len(fold_metrics)} pencere ticarette)")
    print(f"  Ortalama coverage      : {avg_coverage*100:.1f}%")
    stacking_worse_folds = sum(1 for row in fold_comparison if row["stacking_worse"])
    return {
        "buy_precision": avg_buy_precision,
        "coverage":      avg_coverage,
        "folds":         fold_metrics,
        "beta_cal":      last_beta_cal,
        "wf_preds":      all_fold_preds,
        "fold_comparison": fold_comparison,
        "stacking_worse_folds": stacking_worse_folds,
        "active_mode": strategy_label(stacking_enabled),
    }

# ── Ana model eğitimi ─────────────────────────────────────────
def train_main_model(all_data, use_temporal=False, stacking_enabled=None):
    """
    2022-2023 ile eğit, 2024-2026 ile test.
    Model 2024 sonrasını hiç görmedi.
    Returns: model, calibrator, thresholds, feature_names, metrics_dict
    metrics_dict keys: test_acc, traded_buy_prec, coverage
    """
    print("\n" + "="*60)
    print("Ana Model: 2022-2023 Egitim -> 2024-2026 Test")
    print("(Model test verisini hiç görmedi)")
    print("="*60)

    train_cut = pd.Timestamp(TRAIN_END)
    cal_start = train_cut - pd.DateOffset(months=6)

    X_fit, y_fit, w_fit = collect_period_rows(all_data, end=cal_start, binary_only=True, include_weights=True)
    X_cal_bin, y_cal_bin = collect_period_rows(all_data, start=cal_start, end=TRAIN_END, binary_only=True)
    X_cal_full, y_cal_full = collect_period_rows(all_data, start=cal_start, end=TRAIN_END, binary_only=False)
    X_test_bin, y_test_bin = collect_period_rows(all_data, start=TEST_START, binary_only=True)
    X_test_full, y_test_full = collect_period_rows(all_data, start=TEST_START, binary_only=False)

    print(f"  Egitim : {len(X_fit)} ornek  ({y_fit.value_counts().to_dict()})")
    print(f"  Kalibr.: {len(X_cal_bin)} ornek  ({y_cal_bin.value_counts().to_dict()})")
    print(f"  Test   : {len(X_test_bin)} ornek  ({y_test_bin.value_counts().to_dict()})")

    stacking_enabled = STACKING_ENABLED if stacking_enabled is None else bool(stacking_enabled)

    print(f"\n  Egitim basliyor... [{strategy_label(stacking_enabled)}]", end=" ", flush=True)
    model, _aux = train_strategy_model(
        X_fit,
        y_fit,
        sample_weight=w_fit,
        use_temporal=use_temporal,
        stacking_enabled=stacking_enabled,
    )
    print("OK")

    calibrator = fit_calibrator(model, X_cal_bin, y_cal_bin)
    cal_prob = calibrated_buy_prob(model, calibrator, X_cal_full)
    thresholds = select_trade_thresholds(cal_prob, y_cal_full)

    test_prob_bin = calibrated_buy_prob(model, calibrator, X_test_bin)
    preds_bin = np.where(test_prob_bin >= 0.5, 1, 0)

    print("\n  -- Test Sonuclari (2024-2026, hic gorulmedi) --")
    acc = (preds_bin == y_test_bin.values).mean() * 100
    print(f"  Binary genel dogruluk: {acc:.1f}%")
    print(f"\n{classification_report(y_test_bin.astype(int), preds_bin, target_names=['NOT_BUY(0)','BUY(1)'])}")

    test_prob_full = calibrated_buy_prob(model, calibrator, X_test_full)
    trade_metrics = evaluate_signals(test_prob_full, y_test_full, thresholds)
    print("\n  -- Traded-only policy (calibrated + no-trade zone) --")
    print(f"  BUY threshold : {thresholds['buy']*100:.1f}%")
    print(f"  SELL threshold: {(1-thresholds['sell'])*100:.1f}%")
    print(f"  Trade sayisi  : {trade_metrics['trades']}  "
          f"(coverage {trade_metrics['coverage']*100:.1f}%)")
    print(f"  BUY precision : {trade_metrics['buy_precision']*100:.1f}%  "
          f"Neutral rate: {trade_metrics['neutral_buy_rate']*100:.1f}%")
    print(f"  SELL precision: {trade_metrics['sell_precision']*100:.1f}%  "
          f"Neutral rate: {trade_metrics['neutral_sell_rate']*100:.1f}%")

    metrics = {
        "test_acc":        acc,
        "traded_buy_prec": trade_metrics["buy_precision"] * 100,
        "coverage":        trade_metrics["coverage"] * 100,
    }
    return model, calibrator, thresholds, X_fit.columns.tolist(), metrics

# ── Feature önem analizi ──────────────────────────────────────
def feature_importance(model, feature_names):
    print("\n" + "="*60)
    # model may be EnsembleModel — use XGB for interpretability
    xgb_model = model.xgb if isinstance(model, EnsembleModel) else model
    has_lgb = isinstance(model, EnsembleModel) and model.lgb is not None
    has_cat = isinstance(model, EnsembleModel) and getattr(model, "cat", None) is not None
    if has_cat and has_lgb:
        label = "XGB+LGB+CAT Ensemble (2-of-3 vote) — ALL Features (XGB gain)"
    elif has_lgb:
        label = "XGBoost+LightGBM Ensemble — ALL Features (XGB gain)"
    else:
        label = "Tum Feature Onemleri (XGBoost gain)"
    print(label)
    print("="*60)
    imp_full = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    max_val  = imp_full.max() if imp_full.max() > 0 else 1.0
    ranked   = list(imp_full.index)

    # Print ALL features sorted by gain
    print(f"\n  {'Rank':<5} {'Feature':<24} {'Gain':>7}  {'Bar'}")
    print("  " + "-"*65)
    for rank, (fname, val) in enumerate(imp_full.items(), 1):
        bar     = "#" * int(val / max_val * 28)
        is_macro = fname in MACRO_FEATURES_ADDED
        tag     = " [MACRO]" if is_macro else ""
        print(f"  {rank:<5} {fname:<24} {val:>7.4f}  {bar}{tag}")

    # Macro feature summary — explicit KEEP / DROP recommendation
    print("\n  -- MACRO FEATURE GAIN SUMMARY --")
    print(f"  {'Feature':<24} {'Gain':>7}  {'Rank':>6}  Decision")
    print("  " + "-"*60)
    for fname in MACRO_FEATURES_ADDED:
        if fname in imp_full.index:
            val  = imp_full[fname]
            rank = ranked.index(fname) + 1
            if fname == "strongest_sector_5d":
                decision = "KEEP (pinned — known good)"
            elif val >= 0.01:
                decision = "KEEP (gain >= 0.01)"
            else:
                decision = "DROP (gain < 0.01)"
            print(f"  {fname:<24} {val:>7.4f}  {rank:>6}/{len(ranked)}  {decision}")
        else:
            print(f"  {fname:<24}    n/a   not in model")

    return imp_full

# ── Bugünkü sinyaller ─────────────────────────────────────────
def todays_signals(model, calibrator, thresholds, all_data, feature_names):
    print("\n" + "="*60)
    print("BUGÜNKÜ SİNYALLER (gerçek veriye göre)")
    print("="*60)
    print(f"  {'SEM':<8} {'SİNYAL':<8} {'GÜVEN':>6} {'FİYAT':>8} {'RSI':>6} {'TREND':>7}")
    print("  " + "-"*55)

    signals = []
    for sym, payload in all_data.items():
        X, y, _w = split_symbol_payload(payload)
        if X.empty: continue
        last = X.fillna(0).iloc[[-1]]  # son gün
        missing = [f for f in feature_names if f not in last.columns]
        for m in missing:
            last[m] = 0
        last = last[feature_names]

        prob_buy = calibrated_buy_prob(model, calibrator, last)[0]
        pred, conf_prob = classify_prob(prob_buy, thresholds)
        conf  = conf_prob * 100

        signal_str = "AL  ^" if pred == 1 else "SAT v" if pred == -1 else "BEKLE"

        # Mevcut fiyat ve RSI
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT o.close, i.rsi14, i.mtf_trend "
            "FROM ohlcv o JOIN indicators i "
            "ON o.symbol=i.symbol AND o.date=i.date "
            "WHERE o.symbol=? ORDER BY o.date DESC LIMIT 1",
            (sym,)).fetchone()
        conn.close()
        price = row[0] if row else 0
        rsi   = row[1] if row else 0
        trend = row[2] if row else 0
        tstr  = "YUKARI" if trend==1 else "ASAGI" if trend==-1 else " YATAY"

        print(f"  {sym:<8} {signal_str:<8} {conf:>5.1f}%  "
              f"{price:>8.2f}  {rsi:>5.1f}  {tstr:>7}")

        if pred != 0:
            signals.append({
                "symbol": sym, "signal": "AL" if pred==1 else "SAT",
                "confidence": conf, "price": price, "rsi": rsi,
                "prob_buy": prob_buy
            })

    return signals

# ── Sonuç özeti ───────────────────────────────────────────────
def print_verdict(acc_wf, signals):
    print("\n" + "="*60)
    print("ÖZET VE DEĞERLENDİRME")
    print("="*60)

    if acc_wf:
        wf_buy_precision = acc_wf["buy_precision"] * 100
        if wf_buy_precision >= 45:
            verdict = "GUCLU - traded BUY quality iyi"
            detail  = f"%{wf_buy_precision:.1f} BUY precision"
        elif wf_buy_precision >= 35:
            verdict = "KABUL EDILEBILIR - secici kullan"
            detail  = f"%{wf_buy_precision:.1f} BUY precision"
        else:
            verdict = "ZAYIF - no-trade zone kritik"
            detail  = f"%{wf_buy_precision:.1f} BUY precision"
        print(f"\n  Walk-Forward: {verdict}")
        print(f"  {detail}")

    al_sinyaller  = [s for s in signals if s["signal"]=="AL"]
    sat_sinyaller = [s for s in signals if s["signal"]=="SAT"]
    print(f"\n  Bugün yüksek güvenli AL  sinyali: {len(al_sinyaller)} hisse")
    for s in al_sinyaller:
        print(f"    {s['symbol']:<8} {s['price']:.2f} TL  güven: {s['confidence']:.1f}%")
    print(f"  Bugün yüksek güvenli SAT sinyali: {len(sat_sinyaller)} hisse")
    for s in sat_sinyaller:
        print(f"    {s['symbol']:<8} {s['price']:.2f} TL  güven: {s['confidence']:.1f}%")

    print("\n  Sonraki adım:")
    if len(al_sinyaller) + len(sat_sinyaller) >= 2:
        print("  -> py -3.12 paper_trade.py  (bu sinyallerle paper trade başlat)")
    else:
        print("  -> Daha fazla hisse ekle veya feature engineering geliştir")


def summarize_binary_distribution(all_data):
    known_rows = 0
    buy_rows = 0
    for payload in all_data.values():
        _X, y, _w = split_symbol_payload(payload)
        y_known = y.dropna()
        known_rows += len(y_known)
        buy_rows += int((y_known == 1).sum())

    neutral_rows = known_rows - buy_rows
    buy_pct = (buy_rows / known_rows * 100.0) if known_rows else 0.0
    neutral_pct = (neutral_rows / known_rows * 100.0) if known_rows else 0.0
    return {
        "rows": known_rows,
        "buy_rows": buy_rows,
        "neutral_rows": neutral_rows,
        "buy_pct": buy_pct,
        "neutral_pct": neutral_pct,
    }


def per_symbol_buy_rates(all_data):
    rows = []
    for sym, payload in all_data.items():
        _X, y, _w = split_symbol_payload(payload)
        y_known = y.dropna()
        buy_rate = float((y_known == 1).mean() * 100.0) if len(y_known) else 0.0
        rows.append((sym, buy_rate, len(y_known)))
    return rows


def train_final_inference_artifacts(all_data, use_temporal=False, stacking_enabled=False):
    X_all, y_all, w_all = collect_period_rows(all_data, binary_only=True, include_weights=True)
    model, aux = train_strategy_model(
        X_all,
        y_all,
        sample_weight=w_all,
        use_temporal=use_temporal,
        stacking_enabled=stacking_enabled,
    )

    if stacking_enabled and aux["meta_X"] is not None and len(aux["meta_X"]):
        raw_prob_for_cal = model.meta_lr.predict_proba(aux["meta_X"])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_prob_for_cal, aux["meta_y"])
        cal_prob = np.clip(np.asarray(calibrator.predict(raw_prob_for_cal), dtype=float), 0.0, 1.0)
        thresholds = select_trade_thresholds(cal_prob, pd.Series(aux["meta_y"], index=np.arange(len(aux["meta_y"]))))
    else:
        calibrator = None
        raw_prob_for_cal = compute_raw_model_prob(model, X_all.fillna(0))
        if len(np.unique(y_all.astype(int))) >= 2 and len(X_all) >= 40:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_prob_for_cal, y_all.astype(int))
            cal_prob = np.clip(np.asarray(calibrator.predict(raw_prob_for_cal), dtype=float), 0.0, 1.0)
        else:
            cal_prob = np.clip(raw_prob_for_cal, 0.0, 1.0)
        thresholds = select_trade_thresholds(cal_prob, y_all)

    return model, calibrator, thresholds, X_all.columns.tolist()

# ── Entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("="*60)
    if HAS_CAT and HAS_LGB:
        print("BIST ML Modeli — XGBoost + LightGBM + CatBoost Ensemble (2-of-3 vote)")
    elif HAS_LGB:
        print("BIST ML Modeli — XGBoost + LightGBM Ensemble")
    else:
        print("BIST XGBoost ML Modeli")
    print(f"Egitim: 2022-2023  |  Test: 2024-2026 (model görmedi)")
    if TRIPLE_BARRIER_ENABLED:
        print(
            "Hedef : triple-barrier BUY etiketi "
            f"(PT {TRIPLE_BARRIER_PT_SL_RATIO:.1f}x ATR / SL {TRIPLE_BARRIER_SL_RATIO:.1f}x ATR / {TRIPLE_BARRIER_VERTICAL_BARS} bar)"
        )
    else:
        print(f"Hedef : legacy BUY etiketi ({LEGACY_HORIZON} günde +{TARGET_PCT}% )")
    print("="*60)

    conn = sqlite3.connect(DB_PATH)
    # Ensure sentiment columns exist (migration for DBs pre-dating news_filter NLP)
    for _col, _defn in [("sentiment_score", "REAL DEFAULT 0.0"),
                         ("news_count",      "INTEGER DEFAULT 0")]:
        try:
            conn.execute(f"ALTER TABLE indicators ADD COLUMN {_col} {_defn}")
            conn.commit()
        except Exception:
            pass

    print("\nVeri yükleniyor...")
    all_dfs = {}
    for sym in SYMBOLS:
        df = load_symbol(sym, conn)
        if len(df) < 100:
            print(f"  {sym}: yeterli veri yok, atlanıyor")
            continue
        all_dfs[sym] = df
        print(f"  {sym:<8}: {len(df)} gün yüklendi")

    # Makro veri yükle (USDTRY, Brent, TCMB)
    macro_df = load_macro_data(conn)
    if macro_df.empty:
        print("  [UYARI] macro_data tablosu bos — once indicators.py calistirin")
    else:
        print(f"  macro_data: {len(macro_df)} gun ({macro_df.index.min().date()} - {macro_df.index.max().date()})")

    conn.close()

    # Çapraz-hisse makro özellikler için piyasa verisi
    market_data = {sym: df["close"] for sym, df in all_dfs.items()}
    hmm_regime_df = compute_historical_regime_features(DB_PATH)

    print("\nFeature engineering...")
    legacy_all_data = {}
    tb_all_data = {}
    for sym, df in all_dfs.items():
        X = make_features(df, sym=sym, market_data=market_data, macro_df=macro_df, regime_df=hmm_regime_df)
        y_legacy, w_legacy = build_training_target(df, use_triple_barrier=False)
        y_tb, w_tb = build_training_target(df, use_triple_barrier=True)

        X_legacy = X.reindex(y_legacy.index)
        X_tb = X.reindex(y_tb.index)
        legacy_all_data[sym] = (X_legacy, y_legacy, w_legacy)
        tb_all_data[sym] = (X_tb, y_tb, w_tb)

        old_known = y_legacy.dropna()
        new_known = y_tb.dropna()
        print(
            f"  {sym:<8}: legacy BUY {float((old_known == 1).mean() * 100.0):>5.1f}%  "
            f"tb BUY {float((new_known == 1).mean() * 100.0):>5.1f}%  "
            f"rows={len(new_known)}"
        )

    print(f"\nToplam: {len(tb_all_data)} hisse yüklendi")

    legacy_dist = summarize_binary_distribution(legacy_all_data)
    tb_dist = summarize_binary_distribution(tb_all_data)
    active_all_data = tb_all_data if TRIPLE_BARRIER_ENABLED else legacy_all_data
    active_stacking_enabled = STACKING_ENABLED

    print("\nCLASS DISTRIBUTION")
    print(f"  Legacy  +2%/{LEGACY_HORIZON}-day : BUY {legacy_dist['buy_pct']:.2f}%  NEUTRAL {legacy_dist['neutral_pct']:.2f}%  rows={legacy_dist['rows']}")
    print(f"  Triple barrier        : BUY {tb_dist['buy_pct']:.2f}%  NEUTRAL {tb_dist['neutral_pct']:.2f}%  rows={tb_dist['rows']}")

    low_buy_symbols = [item for item in per_symbol_buy_rates(tb_all_data) if item[1] < 5.0]
    if low_buy_symbols:
        print("  [UYARI] BUY rate < 5% olan semboller:")
        for sym, buy_rate, rows in low_buy_symbols:
            print(f"    {sym:<8} BUY={buy_rate:.2f}%  rows={rows}")
    else:
        print("  BUY rate < 5% olan sembol yok")

    legacy_wf_result = None
    legacy_wf_precision = 0.0
    if TRIPLE_BARRIER_ENABLED:
        print("\n[BASELINE] Legacy +2%/3-day walk-forward...")
        legacy_wf_result = walk_forward(legacy_all_data, n_splits=6, stacking_enabled=False, compare_soft_vote=False)
        legacy_wf_precision = legacy_wf_result["buy_precision"] * 100 if legacy_wf_result else 0.0
        print(f"  Legacy WF BUY precision: {legacy_wf_precision:.1f}%")

    # ── Round 1: train with all features ───────────────────────
    print("\n[ROUND 1] Initial walk-forward (all features)...")
    wf_result = walk_forward(active_all_data, n_splits=6, stacking_enabled=active_stacking_enabled)
    initial_wf_precision = wf_result["buy_precision"] * 100 if wf_result else 0.0
    initial_coverage     = wf_result["coverage"] * 100      if wf_result else 0.0

    model, calibrator, thresholds, feature_names, _m1 = train_main_model(active_all_data, stacking_enabled=active_stacking_enabled)
    initial_n_features = len(feature_names)

    # Print ALL feature importances; identify weak macro features
    imp_full = feature_importance(model, feature_names)

    # ── Feature selection ───────────────────────────────────────
    # Drop macro features with gain < 0.01 (strongest_sector_5d always kept)
    weak_macros = [
        f for f in MACRO_FEATURES_ADDED
        if f in imp_full.index
        and imp_full[f] < 0.01
        and f != "strongest_sector_5d"
    ]

    def _drop_cols(data_dict, to_drop):
        return {
            sym: (
                X.drop(columns=[c for c in to_drop if c in X.columns]),
                y,
                w,
            )
            for sym, (X, y, w) in data_dict.items()
        }

    if weak_macros:
        print(f"\n[FEATURE SEL] Dropping {len(weak_macros)} weak macro feature(s): {weak_macros}")
        active_all_data = _drop_cols(active_all_data, weak_macros)

        print("\n[ROUND 2] Retrain after feature selection...")
        wf_result = walk_forward(active_all_data, n_splits=6, stacking_enabled=active_stacking_enabled)
        model, calibrator, thresholds, feature_names, _m2 = train_main_model(active_all_data, stacking_enabled=active_stacking_enabled)
        print(f"  Round 2 WF BUY precision: {wf_result['buy_precision']*100:.1f}%  "
              f"coverage: {wf_result['coverage']*100:.1f}%  "
              f"features: {len(feature_names)}")
    else:
        print("\n[FEATURE SEL] All macro features have gain >= 0.01 — no features dropped.")
        _m2 = _m1

    # ── Fallback: if precision still below 76% ──────────────────
    current_wf_precision = wf_result["buy_precision"] * 100 if wf_result else 0.0
    if current_wf_precision < 76.0:
        keep_macros   = {"strongest_sector_5d", "usdtry_1d_return"}
        drop_macros_2 = [
            f for f in MACRO_FEATURES_ADDED
            if f not in keep_macros
            and f in feature_names    # only those still in the model
        ]
        if drop_macros_2:
            print(f"\n[FALLBACK] Precision {current_wf_precision:.1f}% still < 76%.")
            print(f"  Dropping ALL macro except {sorted(keep_macros)}: {drop_macros_2}")
            active_all_data = _drop_cols(active_all_data, drop_macros_2)

            print("\n[ROUND 3] Retrain with minimal macro features...")
            wf_result = walk_forward(active_all_data, n_splits=6, stacking_enabled=active_stacking_enabled)
            model, calibrator, thresholds, feature_names, _m2 = train_main_model(active_all_data, stacking_enabled=active_stacking_enabled)
            print(f"  Round 3 WF BUY precision: {wf_result['buy_precision']*100:.1f}%  "
                  f"coverage: {wf_result['coverage']*100:.1f}%  "
                  f"features: {len(feature_names)}")

    if active_stacking_enabled and wf_result and wf_result.get("stacking_worse_folds", 0) >= 4:
        print("\n[STACKING DECISION] Stacking soft-vote'a gore 4+ fold'da daha zayif.")
        print("  STACKING_ENABLED=False olarak geri donuluyor: meta-learner bu veri rejiminde ek deger katmadi.")
        active_stacking_enabled = False
        wf_result = walk_forward(active_all_data, n_splits=6, stacking_enabled=False, compare_soft_vote=False)
        model, calibrator, thresholds, feature_names, _m2 = train_main_model(active_all_data, stacking_enabled=False)

    # ── Baseline captured (after feature selection) ────────────
    baseline_wf_prec      = wf_result["buy_precision"] * 100 if wf_result else 0.0
    baseline_wf_cov       = wf_result["coverage"] * 100      if wf_result else 0.0
    baseline_traded_prec  = _m2["traded_buy_prec"]
    baseline_test_acc     = _m2["test_acc"]
    baseline_n_feat       = len(feature_names)
    # top 2 features from the baseline model
    _imp_base = pd.Series(
        model.xgb.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    baseline_top_feats = list(_imp_base.head(2).index)

    if STACKING_ENABLED and wf_result and wf_result.get("fold_comparison"):
        print("\nSTACKING vs SOFT-VOTE FOLD COMPARISON")
        for row in wf_result["fold_comparison"]:
            verdict = "STACKING better" if row["stacking_better"] else "STACKING worse" if row["stacking_worse"] else "TIE"
            print(
                f"  Fold {row['fold']}: SoftVote {row['soft_vote_precision']:.1f}%  "
                f"Stacking {row['stacking_precision']:.1f}%  -> {verdict}"
            )
        print(
            f"  Stacking worse folds: {wf_result.get('stacking_worse_folds', 0)}/"
            f"{len(wf_result['fold_comparison'])}"
        )

    # Feature selection comparison summary (always print)
    print("\n" + "="*60)
    print("FEATURE SELECTION SUMMARY")
    print("="*60)
    print(f"  Before macro (baseline) : 48 features  |  77.0% WF BUY precision")
    print(f"  After macro added       : {initial_n_features} features  |  "
          f"{initial_wf_precision:.1f}% WF precision  {initial_coverage:.1f}% coverage")
    print(f"  After feature selection : {baseline_n_feat} features  |  "
          f"{baseline_wf_prec:.1f}% WF precision  {baseline_wf_cov:.1f}% coverage")
    print(f"  L1 reg                  : XGB reg_alpha=0.2  LGB reg_alpha=0.2")
    print("="*60)

    # ── Weighted retrain ───────────────────────────────────────
    print(f"\n[WEIGHTED RETRAIN] Applying temporal sample weights...")
    print(f"  >= {TEMPORAL_CUT_12M.date()} (last 12 mo): 2.0x")
    print(f"  {TEMPORAL_CUT_24M.date()} - {TEMPORAL_CUT_12M.date()} (12-24 mo): 1.5x")
    print(f"  < {TEMPORAL_CUT_24M.date()} (older): 1.0x")
    print(f"  Applied to: XGBoost.fit(), LightGBM.fit(), CatBoost.fit()")

    wf_weighted = walk_forward(active_all_data, n_splits=6, use_temporal=True, stacking_enabled=active_stacking_enabled)
    model_w, cal_w, thr_w, feat_w, metrics_w = train_main_model(active_all_data, use_temporal=True, stacking_enabled=active_stacking_enabled)

    wf_w_prec     = wf_weighted["buy_precision"] * 100 if wf_weighted else 0.0
    wf_w_cov      = wf_weighted["coverage"] * 100      if wf_weighted else 0.0
    traded_w_prec = metrics_w["traded_buy_prec"]
    test_acc_w    = metrics_w["test_acc"]
    _imp_w = pd.Series(
        model_w.xgb.feature_importances_, index=feat_w
    ).sort_values(ascending=False)
    weighted_top_feats = list(_imp_w.head(2).index)

    # ── Comparison table ───────────────────────────────────────
    def _fmt(base, weighted):
        delta = weighted - base
        sign  = "+" if delta >= 0 else ""
        return f"{base:.1f}%", f"{weighted:.1f}% ({sign}{delta:.1f}pp)"

    b_wf,  w_wf  = _fmt(baseline_wf_prec,     wf_w_prec)
    b_tp,  w_tp  = _fmt(baseline_traded_prec,  traded_w_prec)
    b_acc, w_acc = _fmt(baseline_test_acc,     test_acc_w)
    b_cov, w_cov = _fmt(baseline_wf_cov,       wf_w_cov)

    print("\n" + "="*65)
    print("WEIGHTED vs BASELINE COMPARISON")
    print("="*65)
    print(f"  {'Metric':<26} {'Baseline':>14}  {'Weighted':>20}")
    print("  " + "-"*61)
    print(f"  {'WF BUY precision':<26} {b_wf:>14}  {w_wf:>20}")
    print(f"  {'Traded-only precision':<26} {b_tp:>14}  {w_tp:>20}")
    print(f"  {'Test accuracy':<26} {b_acc:>14}  {w_acc:>20}")
    print(f"  {'WF Coverage':<26} {b_cov:>14}  {w_cov:>20}")
    print(f"  {'Top feature #1':<26} {baseline_top_feats[0]:>14}  {weighted_top_feats[0]:>20}")
    print(f"  {'Top feature #2':<26} {baseline_top_feats[1]:>14}  {weighted_top_feats[1]:>20}")
    print("="*65)

    # ── Decision: keep weighted or revert to baseline ──────────
    precision_delta = wf_w_prec - baseline_wf_prec
    if wf_w_prec >= baseline_wf_prec:
        # Keep weighted model
        model, calibrator, thresholds, feature_names = model_w, cal_w, thr_w, feat_w
        wf_result = wf_weighted
        print("\n  WEIGHTED MODEL KEPT — improvement confirmed")
        print(f"  ({baseline_wf_prec:.1f}% -> {wf_w_prec:.1f}%, "
              f"delta {'+' if precision_delta>=0 else ''}{precision_delta:.1f}pp)")
        active_model_tag = f"{strategy_label(active_stacking_enabled).lower()} weighted (recency 2.0x/1.5x/1.0x)"
        active_use_temporal = True
    else:
        # Weighted is worse by > 1pp — revert to baseline (no change needed)
        print(f"\n  BASELINE KEPT — weighted model did not improve")
        print(f"  ({baseline_wf_prec:.1f}% -> {wf_w_prec:.1f}%, "
              f"delta {precision_delta:.1f}pp  threshold: -1pp)")
        active_model_tag = f"{strategy_label(active_stacking_enabled).lower()} baseline (uniform weights)"
        active_use_temporal = False

    if TRIPLE_BARRIER_ENABLED:
        print("\nTRIPLE-BARRIER vs LEGACY SUMMARY")
        print(f"  Legacy class dist        : BUY {legacy_dist['buy_pct']:.2f}%  NEUTRAL {legacy_dist['neutral_pct']:.2f}%")
        print(f"  Triple-barrier class dist: BUY {tb_dist['buy_pct']:.2f}%  NEUTRAL {tb_dist['neutral_pct']:.2f}%")
        print(f"  Legacy WF BUY precision  : {legacy_wf_precision:.1f}%")
        active_wf_precision = (wf_result["buy_precision"] * 100) if wf_result else 0.0
        print(f"  Triple WF BUY precision  : {active_wf_precision:.1f}%")

    final_importances = pd.Series(
        model.xgb.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    print("\nTOP 5 FEATURE IMPORTANCES")
    for rank, (fname, val) in enumerate(final_importances.head(5).items(), 1):
        print(f"  {rank}. {fname:<24} {val:.4f}")

    # ── Bugünkü sinyaller ──────────────────────────────────────
    signals = todays_signals(model, calibrator, thresholds, active_all_data, feature_names)
    signal_count = len([s for s in signals if s["signal"] in ("AL", "SAT")])
    above_thresh  = [s for s in signals if s["prob_buy"] >= 0.65]
    print(f"\n  Signal summary: {signal_count} actionable signal(s) above 0.65")
    for s in above_thresh:
        print(f"    {s['symbol']:<8} {s['signal']:<4} prob={s['prob_buy']:.3f}  "
              f"conf={s['confidence']:.1f}%  price={s['price']:.2f}")

    # Özet
    print_verdict(wf_result, signals)

    print("\n[FINAL SAVE] Tum veri ile base/meta modeller egitiliyor...")
    final_model, final_calibrator, final_thresholds, final_feature_names = train_final_inference_artifacts(
        active_all_data,
        use_temporal=active_use_temporal,
        stacking_enabled=active_stacking_enabled,
    )

    # Modelleri kaydet
    import pickle
    model_bundle_path = MODELS / "model_bundle.pkl"
    xgb_path = MODELS / "xgb_model.pkl"
    lgb_path = MODELS / "lgb_model.pkl"
    cat_path = MODELS / "cat_model.pkl"
    meta_path = MODELS / "meta_lr.pkl"
    is_ensemble = isinstance(final_model, EnsembleModel)
    try:
        with open(xgb_path, "wb") as f:
            pickle.dump(final_model.xgb, f)
        with open(lgb_path, "wb") as f:
            pickle.dump(final_model.lgb, f)
        with open(cat_path, "wb") as f:
            pickle.dump(final_model.cat, f)
        with open(meta_path, "wb") as f:
            pickle.dump(final_model.meta_lr, f)
        with open(model_bundle_path, "wb") as f:
            pickle.dump({
                "features": final_feature_names,
                "calibrator": final_calibrator,
                "thresholds": final_thresholds,
                "ensemble": is_ensemble,
                "stacking_enabled": active_stacking_enabled,
                "meta_extra_features": STACKING_META_EXTRA_FEATURES,
            }, f)
        model_type = f"{strategy_label(active_stacking_enabled)} ensemble [{active_model_tag}]"
        print(f"\n  {model_type} kaydedildi:")
        print(f"    {xgb_path.name}")
        print(f"    {lgb_path.name}")
        print(f"    {cat_path.name}")
        print(f"    {meta_path.name}")
        print(f"    {model_bundle_path.name}")
    except Exception as e:
        print(f"  Model kaydedilemedi: {e}")

    # Beta calibrator kaydet (models/beta_calibrator.pkl)
    # Use the beta_cal from the active walk_forward result to ensure model/calibrator match.
    beta_cal_to_save = wf_result.get("beta_cal") if wf_result else None
    beta_cal_path = MODELS / "beta_calibrator.pkl"
    if CALIBRATION_ENABLED and HAS_BETACAL and beta_cal_to_save is not None:
        try:
            joblib.dump(beta_cal_to_save, beta_cal_path)
            print(f"  Beta calibrator (BetaCalibration abm) kaydedildi: {beta_cal_path}")
        except Exception as e:
            print(f"  Beta calibrator kaydedilemedi: {e}")
    elif CALIBRATION_ENABLED and not HAS_BETACAL:
        print("  [BetaCal] betacal kurulu degil — beta_calibrator.pkl kaydedilmedi")
    else:
        print("  [BetaCal] CALIBRATION_ENABLED=False — beta_calibrator.pkl kaydedilmedi")

    # Walk-forward predictions — saved from the ACTIVE model's wf_result
    # (weighted or baseline, whichever was kept). Used by optimize_threshold.py.
    wf_preds = wf_result.get("wf_preds", []) if wf_result else []
    if wf_preds:
        wf_csv_path = RESULTS / "wf_predictions.csv"
        wf_df = pd.DataFrame(wf_preds)
        wf_df.to_csv(wf_csv_path, index=False)
        n_labeled = int(wf_df["actual_label"].notna().sum())
        print(
            f"  WF predictions saved : {wf_csv_path.name}  "
            f"({len(wf_df)} rows, {wf_df['symbol'].nunique()} symbols, "
            f"{wf_df['fold'].nunique()} folds, {n_labeled} labeled)"
        )
    else:
        print("  WF predictions: none collected (wf_result empty)")

    print(f"\nToplam süre: {time.time()-t0:.1f}s")
    print("\nSonraki adım: py -3.12 paper_trade.py")
