"""
crypto_config.py -- Shared Crypto module configuration.

This module is the single source for crypto runtime, status, and tier policy
constants. Values mirror crypto_trader.py active defaults.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

CRYPTO_CAPITAL_USDT = 10000.0
CRYPTO_MAX_POSITIONS = 6
CRYPTO_MAX_MAJOR_POSITIONS = 5
CRYPTO_MAX_RISKY_POSITIONS = 1
CRYPTO_MAX_EXPOSURE_PER_COIN_MAJOR_PCT = 0.20
CRYPTO_MAX_EXPOSURE_PER_COIN_RISKY_PCT = 0.10
CRYPTO_MAX_EXPOSURE_PER_TIER_MAJOR_PCT = 0.80
CRYPTO_MAX_EXPOSURE_PER_TIER_RISKY_PCT = 0.10
CRYPTO_MAX_DAILY_TRADES = 8
CRYPTO_STOP_LOSS_COOLDOWN_MIN = 60
CRYPTO_STOP_TRADING_FILE = BASE_DIR / "STOP_TRADING"
CRYPTO_STALE_PRICE_GUARD_SEC = 15
CRYPTO_REQUIRE_WS_FOR_ENTRIES = True

CRYPTO_MAJOR = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT",
]
CRYPTO_RISKY = ["ONDO/USDT", "FET/USDT"]
CRYPTO_SYMBOLS = CRYPTO_MAJOR + CRYPTO_RISKY

THRESHOLD_REPORT_PATH = BASE_DIR / "models" / "threshold_report.json"

MAJOR_MAX_SIZE_PCT = 0.20
RISKY_MAX_SIZE_PCT = 0.10
RISKY_MTF_THRESHOLD = 5
CRYPTO_DAILY_LOSS_LIMIT = 0.05
CRYPTO_DAILY_MAJOR_TARGET_PCT = 0.015
CRYPTO_SHADOW_HALT_PCT = 0.04
CRYPTO_BTC_CRASH_THRESHOLD = -0.03
CRYPTO_MTF_THRESHOLD_ML = 4
CRYPTO_MTF_THRESHOLD_RB = 4
CRYPTO_CORR_THRESHOLD = 0.85
CRYPTO_CORR_WINDOW = 24
CRYPTO_RISK_PCT = 0.03
CRYPTO_ATR_STOP_MULT = 1.5
CRYPTO_SIZE_FEAR_MAX = 1.00
CRYPTO_FG_EXTREME_FEAR = 20
CRYPTO_FG_EXTREME_GREED = 80

ENABLE_MARKET_REGIME_FILTER = True
REGIME_SYMBOLS_PROXY = ("BTC/USDT", "ETH/USDT")
REGIME_1H_LOOKBACK = 72
REGIME_RET_LOOKBACK_HOURS = 24
REGIME_VOL_LOOKBACK_HOURS = 24
REGIME_BREADTH_EMA_COL = "ema50"
REGIME_CORR_WINDOW = 20
REGIME_CORR_BASELINE_WINDOW = 20
REGIME_CRASH_BLOCK_LONGS = True
REGIME_CRASH_SIZE_MULT = 0.40
REGIME_CHOP_MTF_ADD = 1
REGIME_CHOP_ML_ADD = 0.05
REGIME_CRASH_BTC_RET = -0.045
REGIME_CRASH_ETH_RET = -0.055
REGIME_CRASH_BTC_RVOL = 0.028
REGIME_CRASH_BREADTH_MAX = 0.35
REGIME_CRASH_CORR_SPIKE = 0.15
REGIME_RISK_ON_BTC_RET = 0.020
REGIME_RISK_ON_ETH_RET = 0.025
REGIME_RISK_ON_BREADTH_MIN = 0.60
REGIME_CHOP_RET_ABS_MAX = 0.015
REGIME_CHOP_ETH_RET_ABS_MAX = 0.020
REGIME_CHOP_BREADTH_MIN = 0.35
REGIME_CHOP_BREADTH_MAX = 0.65
REGIME_CHOP_CORR_MIN = 0.50

ENABLE_RELATIVE_STRENGTH_MTF = True
RS_BONUS_LOOKBACK_HOURS = 24
RS_BONUS_TOP_PCT = 0.75
RS_BONUS_BOTTOM_PCT = 0.25
RS_MTF_BONUS_POINTS = 1

FUNDING_SENTIMENT_THRESHOLD = 0.0001
FUNDING_SIZE_BOOST = 1.20

COIN_GROUP_POLICIES = {
    "CORE_MAJOR": {
        "symbols": {"BTC/USDT", "ETH/USDT", "BNB/USDT"},
        "max_coin_exposure_pct": 0.20,
        "stop_pct": 0.022,
        "target_pct": 0.024,
        "trail_trigger": 0.015,
        "trail_pct": 0.012,
        "partial_take_profit_pct": 0.018,
        "partial_close_fraction": 0.50,
        "stop_loss_cooldown_min": 60,
    },
    "VOL_MAJOR": {
        "symbols": {"SOL/USDT", "AVAX/USDT", "SUI/USDT", "DOT/USDT", "HYPE/USDT"},
        "max_coin_exposure_pct": 0.20,
        "stop_pct": 0.025,
        "target_pct": 0.030,
        "trail_trigger": 0.020,
        "trail_pct": 0.015,
        "partial_take_profit_pct": 0.022,
        "partial_close_fraction": 0.50,
        "stop_loss_cooldown_min": 75,
    },
    "RISKY": {
        "symbols": {"ONDO/USDT", "FET/USDT"},
        "max_coin_exposure_pct": 0.05,
        "stop_pct": 0.030,
        "target_pct": 0.050,
        "trail_trigger": 0.030,
        "trail_pct": 0.020,
        "partial_take_profit_pct": 0.030,
        "partial_close_fraction": 0.50,
        "stop_loss_cooldown_min": 90,
    },
}

LEGACY_THRESHOLD_CONFIG = {
    "MAJOR": {
        "enabled": True,
        "selected_threshold": 0.63,
        "fallback_threshold": 0.63,
        "no_trade_zone": {
            "short_below_or_equal": 0.37,
            "long_above_or_equal": 0.63,
        },
        "selection_status": "legacy_fallback",
    },
    "RISKY": {
        "enabled": True,
        "selected_threshold": 0.72,
        "fallback_threshold": 0.72,
        "no_trade_zone": {
            "short_below_or_equal": 0.28,
            "long_above_or_equal": 0.72,
        },
        "selection_status": "legacy_fallback",
    },
}
