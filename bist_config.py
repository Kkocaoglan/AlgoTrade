"""
bist_config.py -- Shared BIST trading configuration.

This module is the single source for BIST risk/execution constants that are
used by multiple runtime helpers. Values mirror loop_trader.py active defaults.
"""

# Capital allocation
TOTAL_CAPITAL = 100_000
MIN_CASH_RESERVE = 20_000
TRADEABLE_CAPITAL = 80_000
MAX_POSITION_SIZE = 10_000
MAX_SHORT_SIZE = 5_000
RISK_PER_TRADE = 0.01
MAX_POS = 8

# Signal thresholds
BUY_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.65
DISPLAY_THRESHOLD = 0.50
MODEL_FLIP_THRESHOLD = 0.55

# Exits and sizing
SL_PCT = 0.015
SHORT_SL_PCT = 0.015
ATR_TARGET_MULT = 1.5
TRAIL_BE_PCT = 0.015
TRAIL_START_PCT = 0.020
TRAIL_DIST_PCT = 0.010

# Portfolio and risk gates
MAX_PER_SECTOR = 2
MAX_PORTFOLIO_HEAT = 0.06
CORR_THRESHOLD = 0.85
CORR_WINDOW_DAYS = 60

# Circuit breakers
DRAWDOWN_WARN_PCT = 0.04
DRAWDOWN_HALT_PCT = 0.08
DRAWDOWN_KILL_PCT = 0.12
DAILY_STOP_PCT = 0.08

# Kill switch checks
DAILY_LOSS_LIMIT = -2_000
FAT_FINGER_PCT = 0.05
MAX_ORDERS_PER_MIN = 5
