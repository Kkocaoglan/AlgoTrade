"""
crypto_trader.py — Main loop for Crypto Module v3 (7/24).

Architecture (3 daemon threads + main keep-alive):
  Thread 1 — Signal scanner   : every 5 minutes, computes MTF+ML+F&G
  Thread 2 — Exit monitor     : every 15 seconds, checks stops/targets via WS price
  Thread 3 — Health monitor   : every 60 seconds, prints live status line

WebSocket:
  CryptoWebSocket receives live BTC/ETH/SOL prices in background.
  Exit monitor uses WS price first, falls back to REST ticker if WS is stale.

MTF timeframes: 5m / 15m / 1h  (updated from 15m/1h/4h)

Usage:
  python3.12 crypto_trader.py          # production loop (7/24)
  python3.12 crypto_trader.py --once   # WS connect + 1 signal scan + 1 exit scan, then exit

Completely isolated from loop_trader.py, viop_trader.py, and all spot/VIOP files.
"""

import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from crypto_stream     import (
    CryptoStream,
    CryptoWebSocket,
    check_symbol_on_exchange,
    fetch_funding_rate,
    get_btc_dominance_features,
)
from crypto_indicators import compute_all, compute_mtf, compute_htf_bias
from crypto_oms        import CryptoOMS
from crypto_journal    import (
    estimate_trade_costs,
    make_signal_id,
    mark_signal_entry,
    mark_signal_exit,
    record_signal,
)
from logger            import algo_log
from crypto_logger     import crypto_log
from crypto_trade_journal import journal_open, journal_close, journal_partial_close

try:
    from crypto_sentiment import sentiment as _fg_sentiment
    from crypto_sentiment import fetch_funding_rates as _fetch_funding_rates
    from crypto_sentiment import get_avg_funding_rate as _get_avg_funding_rate
except Exception:
    class _DummySentiment:
        def get_fear_greed(self): return {"value": 50, "label": "Neutral", "timestamp": ""}
        def get_signal_modifier(self, v): return 0.0, 0.0
    def _fetch_funding_rates(): return {}
    def _get_avg_funding_rate(): return 0.0
    _fg_sentiment = _DummySentiment()

try:
    from telegram_bot import send_telegram_alert
except Exception:
    def send_telegram_alert(msg, **kw): pass

# ── Config ────────────────────────────────────────────────────────────────────
CRYPTO_CAPITAL_USDT      = 10000.0
CRYPTO_MAX_POSITIONS     = 6        # major max 5, risky max 1
CRYPTO_MAX_MAJOR_POSITIONS = 5
CRYPTO_MAX_RISKY_POSITIONS = 1
CRYPTO_MAX_EXPOSURE_PER_COIN_MAJOR_PCT = 0.20
CRYPTO_MAX_EXPOSURE_PER_COIN_RISKY_PCT = 0.10
CRYPTO_MAX_EXPOSURE_PER_TIER_MAJOR_PCT = 0.80
CRYPTO_MAX_EXPOSURE_PER_TIER_RISKY_PCT = 0.10
CRYPTO_MAX_DAILY_TRADES   = 8
CRYPTO_STOP_LOSS_COOLDOWN_MIN = 60
CRYPTO_STOP_TRADING_FILE  = Path(__file__).parent / "STOP_TRADING"
CRYPTO_STALE_PRICE_GUARD_SEC = 15
CRYPTO_REQUIRE_WS_FOR_ENTRIES = True

CRYPTO_MAJOR = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
                'AVAX/USDT', 'SUI/USDT', 'DOT/USDT', 'HYPE/USDT']
CRYPTO_RISKY = ['ONDO/USDT', 'FET/USDT']
CRYPTO_SYMBOLS = CRYPTO_MAJOR + CRYPTO_RISKY
THRESHOLD_REPORT_PATH     = Path(__file__).parent / "models" / "threshold_report.json"

MAJOR_MAX_SIZE_PCT       = 0.20     # 20% of capital per major position
RISKY_MAX_SIZE_PCT       = 0.10     # 10% of capital per risky position (1000 USDT)
RISKY_MTF_THRESHOLD      = 5        # higher MTF bar for risky coins
CRYPTO_DAILY_LOSS_LIMIT  = 0.05     # 5% daily loss → halt entries
CRYPTO_DAILY_MAJOR_TARGET_PCT = 0.015  # +1.5% daily equity target -> lock gains
CRYPTO_SHADOW_HALT_PCT   = 0.04     # post-lock shadow pnl >= 4% -> stop shadow entries
CRYPTO_BTC_CRASH_THRESHOLD = -0.03  # BTC 1h drop < -3% → block all entries
CRYPTO_MTF_THRESHOLD_ML  = 4        # MTF adj ≥ 4 when ML active (raised from 3)
CRYPTO_MTF_THRESHOLD_RB  = 4        # MTF adj ≥ 4 rule-based only
CRYPTO_CORR_THRESHOLD    = 0.85     # fallback BTC correlation gate
CRYPTO_CORR_WINDOW       = 24       # last 24 x 1h bars for correlation gate
CRYPTO_RISK_PCT          = 0.03     # 3% capital risk per trade
CRYPTO_ATR_STOP_MULT     = 1.5
CRYPTO_SIZE_FEAR_MAX     = 1.00     # 100% of per-symbol cap hard cap
CRYPTO_FG_EXTREME_FEAR   = 20       # below → size × 1.3
CRYPTO_FG_EXTREME_GREED  = 80       # above → size × 0.7
ENABLE_MARKET_REGIME_FILTER = True
REGIME_SYMBOLS_PROXY        = ("BTC/USDT", "ETH/USDT")
REGIME_1H_LOOKBACK          = 72
REGIME_RET_LOOKBACK_HOURS   = 24
REGIME_VOL_LOOKBACK_HOURS   = 24
REGIME_BREADTH_EMA_COL      = "ema50"
REGIME_CORR_WINDOW          = 20
REGIME_CORR_BASELINE_WINDOW = 20
REGIME_CRASH_BLOCK_LONGS    = True
REGIME_CRASH_SIZE_MULT      = 0.40
REGIME_CHOP_MTF_ADD         = 1
REGIME_CHOP_ML_ADD          = 0.05
REGIME_CRASH_BTC_RET        = -0.045
REGIME_CRASH_ETH_RET        = -0.055
REGIME_CRASH_BTC_RVOL       = 0.028
REGIME_CRASH_BREADTH_MAX    = 0.35
REGIME_CRASH_CORR_SPIKE     = 0.15
REGIME_RISK_ON_BTC_RET      = 0.020
REGIME_RISK_ON_ETH_RET      = 0.025
REGIME_RISK_ON_BREADTH_MIN  = 0.60
REGIME_CHOP_RET_ABS_MAX     = 0.015
REGIME_CHOP_ETH_RET_ABS_MAX = 0.020
REGIME_CHOP_BREADTH_MIN     = 0.35
REGIME_CHOP_BREADTH_MAX     = 0.65
REGIME_CHOP_CORR_MIN        = 0.50
ENABLE_RELATIVE_STRENGTH_MTF = True
RS_BONUS_LOOKBACK_HOURS      = 24
RS_BONUS_TOP_PCT             = 0.75
RS_BONUS_BOTTOM_PCT          = 0.25
RS_MTF_BONUS_POINTS          = 1
FUNDING_SENTIMENT_THRESHOLD  = 0.0001   # 0.01%
FUNDING_SIZE_BOOST           = 1.20

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

# ── Tier helpers ──────────────────────────────────────────────────────────────

def get_tier(symbol: str) -> str:
    return 'RISKY' if symbol in CRYPTO_RISKY else 'MAJOR'


def get_coin_group(symbol: str) -> str:
    for group_name, policy in COIN_GROUP_POLICIES.items():
        if symbol in policy["symbols"]:
            return group_name
    return "RISKY" if get_tier(symbol) == "RISKY" else "VOL_MAJOR"


def get_trade_policy(symbol: str) -> dict:
    return COIN_GROUP_POLICIES[get_coin_group(symbol)]

def _tier_threshold_config(symbol: str) -> dict:
    tier = get_tier(symbol)
    return (_threshold_config or LEGACY_THRESHOLD_CONFIG).get(tier, LEGACY_THRESHOLD_CONFIG[tier])

def get_ml_threshold(symbol: str) -> float:
    cfg = _tier_threshold_config(symbol)
    threshold = cfg.get("selected_threshold")
    if threshold is None:
        threshold = cfg.get("fallback_threshold")
    if threshold is None:
        return 1.01
    if not cfg.get("enabled", True):
        return 1.01
    return float(threshold)

def get_short_ml_threshold(symbol: str) -> float:
    """Return ML threshold for SHORT entries from SHORT model's own threshold_config."""
    tier = get_tier(symbol)
    cfg = (_short_threshold_config or LEGACY_THRESHOLD_CONFIG).get(tier, LEGACY_THRESHOLD_CONFIG[tier])
    threshold = cfg.get("selected_threshold")
    if threshold is None:
        threshold = cfg.get("fallback_threshold")
    if threshold is None:
        return 1.01
    if not cfg.get("enabled", True):
        return 1.01
    return float(threshold)

def get_no_trade_zone(symbol: str) -> tuple[float, float]:
    cfg = _tier_threshold_config(symbol)
    zone = cfg.get("no_trade_zone", {})
    short_edge = float(zone.get("short_below_or_equal", 0.0))
    long_edge = float(zone.get("long_above_or_equal", 1.0))
    return short_edge, long_edge

def get_mtf_threshold(symbol: str, ml_on: bool) -> int:
    if get_tier(symbol) == 'RISKY':
        return RISKY_MTF_THRESHOLD
    return CRYPTO_MTF_THRESHOLD_ML if ml_on else CRYPTO_MTF_THRESHOLD_RB

def get_max_size(symbol: str, capital: float) -> float:
    return get_max_coin_exposure(symbol, capital)


def get_max_coin_exposure(symbol: str, capital: float) -> float:
    policy = get_trade_policy(symbol)
    pct = float(policy.get("max_coin_exposure_pct", (
        CRYPTO_MAX_EXPOSURE_PER_COIN_RISKY_PCT if get_tier(symbol) == 'RISKY'
        else CRYPTO_MAX_EXPOSURE_PER_COIN_MAJOR_PCT
    )))
    return capital * pct


def get_max_tier_exposure(tier: str, capital: float) -> float:
    pct = (
        CRYPTO_MAX_EXPOSURE_PER_TIER_RISKY_PCT
        if tier == 'RISKY'
        else CRYPTO_MAX_EXPOSURE_PER_TIER_MAJOR_PCT
    )
    return capital * pct


SIGNAL_INTERVAL = 300   # 5 minutes between signal scans
EXIT_INTERVAL   = 15    # 15 seconds between exit checks
HEALTH_INTERVAL = 60    # 60 seconds between health prints
WS_ALIVE_SEC    = 10    # price considered live if < 10s old

# ── Module globals ────────────────────────────────────────────────────────────
stream = CryptoStream()
oms    = CryptoOMS()
ws: CryptoWebSocket | None = None
RESULTS_DIR = Path(__file__).parent / "results"
MAJORAFTER_STATE_PATH = RESULTS_DIR / "majorafter_state.json"

_pos_hwm: dict[int, float] = {}    # { position_id: hwm_price }
# symbol → [(pos_id, direction), ...] — rebuilt each exit cycle; read by WS callback
_symbol_pos_index: dict[str, list[tuple[int, str]]] = {}
_hwm_lock = threading.Lock()
_majorafter_lock = threading.RLock()

# ── WS price callback: updates HWM/LWM on every tick ─────────────────────────

def _ws_price_callback(symbol: str, price: float) -> None:
    """Called from the WS thread on every ticker message.

    Updates _pos_hwm for any open position whose symbol matches the incoming
    tick, so that the exit-check thread reads an always-current HWM/LWM rather
    than one that may be 15 s stale during a rapid pump or dump.

    _symbol_pos_index is rebuilt by _run_exits() every 15 s from a real DB
    query, so this callback never touches the DB — it only does an in-memory
    dict lookup under _hwm_lock.

    Must be fast, non-blocking, and must never raise (called in WS thread).
    """
    with _hwm_lock:
        entries = _symbol_pos_index.get(symbol)
        if not entries:
            return
        for pos_id, direction in entries:
            prev = _pos_hwm.get(pos_id, price)
            _pos_hwm[pos_id] = max(prev, price) if direction == "long" else min(prev, price)


def _warn_once(key: str, message: str) -> None:
    if key in _legacy_model_warnings:
        return
    _legacy_model_warnings.add(key)
    print(message)
    algo_log.system(message)


def _get_runtime_model(symbol: str, side: str) -> dict | None:
    tier = get_tier(symbol)
    if side == "long":
        model = _tier_models_long.get(tier)
        if model is not None:
            return model
        if _crypto_model_long is not None:
            _warn_once(
                f"legacy_long_{tier}",
                f"[CRYPTO] WARN: {tier} LONG tier modeli yok — legacy LONG artifact fallback kullaniliyor",
            )
            return _crypto_model_long
        if _crypto_model is not None:
            _warn_once(
                f"single_long_{tier}",
                f"[CRYPTO] WARN: {tier} LONG legacy single model fallback kullaniliyor",
            )
            return _crypto_model
        return None

    model = _tier_models_short.get(tier)
    if model is not None:
        return model
    if _crypto_model_short is not None:
        _warn_once(
            f"legacy_short_{tier}",
            f"[CRYPTO] WARN: {tier} SHORT tier modeli yok — legacy SHORT artifact fallback kullaniliyor",
        )
        return _crypto_model_short
    return None


def _has_runtime_model(symbol: str, side: str) -> bool:
    return _get_runtime_model(symbol, side) is not None

_crypto_model: dict | None = None        # single model fallback (crypto_xgb.pkl)
_crypto_model_long:  dict | None = None  # legacy side-level LONG fallback
_crypto_model_short: dict | None = None  # legacy side-level SHORT fallback
_tier_models_long: dict[str, dict | None] = {"MAJOR": None, "RISKY": None}
_tier_models_short: dict[str, dict | None] = {"MAJOR": None, "RISKY": None}
_threshold_config: dict | None = None        # LONG model threshold config
_short_threshold_config: dict | None = None  # SHORT model threshold config (separate calibration)
_model_load_time: float = 0.0            # epoch when _load_crypto_model() last ran
_MODEL_RELOAD_INTERVAL: int = 600        # check model files every 10 min (health ticks × 60s)
_market_regime_snapshot: dict | None = None
_relative_strength_snapshot: dict[str, dict] = {}
_corr_gate_snapshot: dict[str, float | str | int] = {
    "avg_corr": 0.0,
    "threshold": CRYPTO_CORR_THRESHOLD,
    "sample_count": 0,
    "market_state": "normal",
}
_funding_snapshot: dict[str, object] = {
    "rates": {},
    "avg_rate": 0.0,
    "bias": "neutral",
    "long_block": False,
    "short_block": False,
    "long_size_mult": 1.0,
    "short_size_mult": 1.0,
}
_btc_regime: str = "NEUTRAL"       # BTC combined regime: BULL / BEAR / CAUTION / NEUTRAL
_btc_regime_detail: dict = {}      # full two-layer info (structural, momentum, size_mult)
_BTC_REGIME_SIZE_MULT: float = 1.0 # size multiplier from BTC regime (0.7 for CAUTION)
_btc_dominance_snapshot: dict | None = None
PAPER_LONG_ONLY = False            # both LONG and SHORT entries enabled when approved artifacts exist
SHORT_MODEL_ACTIVE = True          # SHORT model active — requires promoted artifact with active=True
_long_model_active: bool = False
_short_model_active: bool = False
_short_gate_snapshot: dict[str, str] = {}

_btc_1h_return: float = 0.0        # updated each signal scan
_last_scan_ts:  float = 0.0        # epoch of last signal scan
_next_scan_ts:  float = 0.0        # epoch of next planned signal scan
_stop_event   = threading.Event()  # set to terminate all threads cleanly
_DRY_RUN_ONCE: bool = False        # --once diagnostics: never open/close positions
_last_hourly_status_slot: str = ""
TZ_IST = ZoneInfo("Europe/Istanbul")
_daily_risk_mode_snapshot: dict[str, object] = {"mode": "NORMAL", "threshold_add": 0.0, "size_mult": 1.0, "major_only": False}
_signal_telegram_dedupe: set[tuple[str, str, str, str]] = set()
_legacy_model_warnings: set[str] = set()


def _today_utc_date() -> str:
    return datetime.now(TZ_IST).strftime("%Y-%m-%d")


def _iso_to_local_date(date_str: str | None, tz: ZoneInfo = TZ_IST) -> str:
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def _fmt_hold_short(entry_date_str: str) -> str:
    try:
        entry_dt = datetime.fromisoformat(entry_date_str).replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - entry_dt).total_seconds())
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h{m:02d}m" if h > 0 else f"{m}m"
    except Exception:
        return "?"


def _calc_position_pnl(position: dict, current_price: float) -> float:
    entry = float(position["entry_price"])
    coins = float(position["amount_coin"])
    if position["direction"] == "long":
        return round((current_price - entry) * coins, 4)
    return round((entry - current_price) * coins, 4)


def _current_equity(open_positions: list[dict] | None = None) -> float:
    positions = oms.get_open_positions() if open_positions is None else open_positions
    closed_pnl = oms.get_cumulative_pnl()
    open_pnl = 0.0
    for pos in positions:
        cur = _get_price(pos["symbol"]) or float(pos["entry_price"])
        open_pnl += oms.calc_pnl(pos, cur)
    return round(CRYPTO_CAPITAL_USDT + closed_pnl + open_pnl, 4)


def _majorafter_report_path(date_str: str) -> Path:
    return RESULTS_DIR / f"majorafter_{date_str}.txt"


def _new_majorafter_state(date_str: str, start_equity: float) -> dict:
    target_equity = round(start_equity * (1.0 + CRYPTO_DAILY_MAJOR_TARGET_PCT), 4)
    return {
        "date": date_str,
        "start_equity": round(start_equity, 4),
        "target_equity": target_equity,
        "locked": False,
        "locked_at": "",
        "locked_equity": None,
        "lock_reason": "",
        "shadow_open_positions": [],
        "shadow_closed_positions": [],
    }


def _write_majorafter_report(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = _majorafter_report_path(state["date"])
    locked_equity = float(state.get("locked_equity") or 0.0)
    start_equity = float(state.get("start_equity") or CRYPTO_CAPITAL_USDT)
    actual_gain = locked_equity - start_equity if state.get("locked") else 0.0
    actual_gain_pct = (actual_gain / start_equity * 100.0) if start_equity else 0.0
    shadow_closed = state.get("shadow_closed_positions", [])
    shadow_open = state.get("shadow_open_positions", [])
    shadow_total = sum(float(pos.get("pnl_usdt") or 0.0) for pos in shadow_closed)
    shadow_pct = (shadow_total / locked_equity * 100.0) if locked_equity else 0.0
    lines = [
        f"MAJORAFTER REPORT {state['date']}",
        "=" * 60,
        f"Start equity      : {start_equity:.2f} USDT",
        f"Daily target      : {state.get('target_equity', 0.0):.2f} USDT (+{CRYPTO_DAILY_MAJOR_TARGET_PCT*100:.1f}%)",
        f"Locked            : {'YES' if state.get('locked') else 'NO'}",
    ]
    if state.get("locked"):
        lines.extend([
            f"Locked at         : {state.get('locked_at', '')}",
            f"Lock reason       : {state.get('lock_reason', '')}",
            f"Locked equity     : {locked_equity:.2f} USDT",
            f"Actual major gain : {actual_gain:+.2f} USDT ({actual_gain_pct:+.2f}%)",
            "",
            "Post-lock shadow book (capital frozen at locked equity):",
            f"Closed trades     : {len(shadow_closed)}",
            f"Open trades       : {len(shadow_open)}",
            f"Shadow total P&L  : {shadow_total:+.2f} USDT ({shadow_pct:+.2f}%)",
            f"Hypothetical equity: {locked_equity + shadow_total:.2f} USDT",
        ])
    else:
        lines.extend([
            "Locked equity     : N/A",
            "Actual major gain : N/A",
            "",
            "Post-lock shadow book: waiting for daily lock",
        ])

    lines.append("")
    lines.append("Closed shadow trades:")
    if shadow_closed:
        for pos in shadow_closed:
            lines.append(
                f"- {pos['symbol']} {pos['direction'].upper()} | "
                f"entry={pos['entry_price']:.4f} exit={pos['exit_price']:.4f} | "
                f"pnl={float(pos['pnl_usdt']):+.2f} USDT ({float(pos['pnl_pct']):+.2f}%) | "
                f"reason={pos['exit_reason']}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Open shadow trades:")
    if shadow_open:
        for pos in shadow_open:
            lines.append(
                f"- {pos['symbol']} {pos['direction'].upper()} | "
                f"entry={pos['entry_price']:.4f} size={float(pos['amount_usdt']):.2f} USDT | "
                f"stop={pos['stop_price']:.4f} target={pos['target_price']:.4f}"
            )
    else:
        lines.append("- none")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_majorafter_state(state: dict) -> None:
    with _majorafter_lock:
        RESULTS_DIR.mkdir(exist_ok=True)
        MAJORAFTER_STATE_PATH.write_text(
            json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_majorafter_report(state)


def _load_majorafter_state(current_equity: float | None = None) -> dict:
    with _majorafter_lock:
        date_str = _today_utc_date()
        state = {}
        if MAJORAFTER_STATE_PATH.exists():
            try:
                state = json.loads(MAJORAFTER_STATE_PATH.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        if state.get("date") != date_str:
            if current_equity is None:
                current_equity = _current_equity()
            state = _new_majorafter_state(date_str, current_equity)
            _save_majorafter_state(state)
            return state
        state.setdefault("shadow_open_positions", [])
        state.setdefault("shadow_closed_positions", [])
        state.setdefault("locked", False)
        state.setdefault("locked_at", "")
        state.setdefault("locked_equity", None)
        state.setdefault("lock_reason", "")
        if "target_equity" not in state:
            start_equity = float(state.get("start_equity") or CRYPTO_CAPITAL_USDT)
            state["target_equity"] = round(start_equity * (1.0 + CRYPTO_DAILY_MAJOR_TARGET_PCT), 4)
            _save_majorafter_state(state)
        return state


def _shadow_open_symbols(state: dict) -> set[str]:
    return {pos["symbol"] for pos in state.get("shadow_open_positions", [])}


def _shadow_trade_count(state: dict) -> int:
    return len(state.get("shadow_open_positions", [])) + len(state.get("shadow_closed_positions", []))


def _shadow_entry_gate(symbol: str, tier: str, size: float, cur: float | None, state: dict) -> tuple[bool, str]:
    if cur is None or cur <= 0:
        return False, "shadow_price_unavailable"
    locked_equity = float(state.get("locked_equity") or 0.0)
    shadow_total = sum(float(pos.get("pnl_usdt") or 0.0) for pos in state.get("shadow_closed_positions", []))
    if locked_equity > 0 and shadow_total / locked_equity >= CRYPTO_SHADOW_HALT_PCT:
        return False, "shadow_daily_halt"
    open_positions = state.get("shadow_open_positions", [])
    if len(open_positions) >= CRYPTO_MAX_POSITIONS:
        return False, "shadow_max_open_positions"
    if symbol in _shadow_open_symbols(state):
        return False, "shadow_already_open_position"
    tier_open = [p for p in open_positions if get_tier(p["symbol"]) == tier]
    if tier == "RISKY" and len(tier_open) >= CRYPTO_MAX_RISKY_POSITIONS:
        return False, "shadow_max_open_positions_tier_risky"
    if tier == "MAJOR" and len(tier_open) >= CRYPTO_MAX_MAJOR_POSITIONS:
        return False, "shadow_max_open_positions_tier_major"
    base_equity = float(state.get("locked_equity") or CRYPTO_CAPITAL_USDT)
    max_coin = get_max_coin_exposure(symbol, base_equity)
    coin_exposure = sum(float(p.get("amount_usdt", 0.0) or 0.0) for p in open_positions if p["symbol"] == symbol)
    if coin_exposure + size > max_coin:
        return False, "shadow_max_exposure_per_coin"
    max_tier = get_max_tier_exposure(tier, base_equity)
    tier_exposure = sum(float(p.get("amount_usdt", 0.0) or 0.0) for p in tier_open)
    if tier_exposure + size > max_tier:
        return False, "shadow_max_exposure_per_tier"
    return True, "shadow_risk_checks_passed"


def _open_shadow_position(state: dict, symbol: str, tier: str, direction: str,
                          price: float, size: float, signal_id: str,
                          entry_reason: str, entry_mtf: int, entry_ml: float) -> None:
    policy = get_trade_policy(symbol)
    stop_pct = float(policy["stop_pct"])
    target_pct = float(policy["target_pct"])
    if direction == "long":
        stop_price = round(price * (1 - stop_pct), 8)
        target_price = round(price * (1 + target_pct), 8)
    else:
        stop_price = round(price * (1 + stop_pct), 8)
        target_price = round(price * (1 - target_pct), 8)
    now_iso = datetime.now(timezone.utc).isoformat()
    shadow_pos = {
        "id": f"shadow_{uuid.uuid4().hex[:12]}",
        "symbol": symbol,
        "tier": tier,
        "direction": direction,
        "entry_date": now_iso,
        "entry_price": float(price),
        "amount_usdt": float(round(size, 2)),
        "amount_coin": float(round(size / price, 8)),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "best_price": float(price),
        "signal_id": signal_id,
        "entry_reason": entry_reason,
        "entry_mtf": int(entry_mtf),
        "entry_ml": float(entry_ml),
        "partial_exit_done": 0,
    }
    state["shadow_open_positions"].append(shadow_pos)
    _save_majorafter_state(state)
    msg = (
        f"[MAJORAFTER] OPEN {symbol} {direction.upper()} | "
        f"entry={price:.4f} size={size:.2f} stop={stop_price:.4f} target={target_price:.4f}"
    )
    print(msg)
    crypto_log.system(msg)


def _close_shadow_position(state: dict, shadow_pos: dict, current_price: float, reason: str) -> None:
    pnl = _calc_position_pnl(shadow_pos, current_price)
    pnl_pct = (pnl / float(shadow_pos["amount_usdt"]) * 100.0) if float(shadow_pos["amount_usdt"]) else 0.0
    closed = dict(shadow_pos)
    closed["exit_date"] = datetime.now(timezone.utc).isoformat()
    closed["exit_price"] = float(current_price)
    closed["pnl_usdt"] = round(pnl, 4)
    closed["pnl_pct"] = round(pnl_pct, 4)
    closed["exit_reason"] = reason
    state["shadow_open_positions"] = [p for p in state.get("shadow_open_positions", []) if p["id"] != shadow_pos["id"]]
    state["shadow_closed_positions"].append(closed)
    _save_majorafter_state(state)
    msg = (
        f"[MAJORAFTER] CLOSE {shadow_pos['symbol']} {shadow_pos['direction'].upper()} | "
        f"entry={float(shadow_pos['entry_price']):.4f} exit={current_price:.4f} | "
        f"pnl={pnl:+.2f} USDT ({pnl_pct:+.2f}%) | reason={reason}"
    )
    print(msg)
    crypto_log.system(msg)


def _run_shadow_exits() -> None:
    state = _load_majorafter_state()
    if not state.get("locked"):
        return
    updated = False
    for shadow_pos in list(state.get("shadow_open_positions", [])):
        cur = _get_price(shadow_pos["symbol"])
        if cur is None:
            continue
        policy = get_trade_policy(shadow_pos["symbol"])
        trail_trigger = float(policy["trail_trigger"])
        trail_pct = float(policy["trail_pct"])
        best_price = float(shadow_pos.get("best_price", shadow_pos["entry_price"]))
        if shadow_pos["direction"] == "long":
            best_price = max(best_price, cur)
        else:
            best_price = min(best_price, cur)
        shadow_pos["best_price"] = float(best_price)
        unrealised = _calc_position_pnl(shadow_pos, cur)
        profit_pct = unrealised / float(shadow_pos["amount_usdt"]) if float(shadow_pos["amount_usdt"]) else 0.0
        reason = None
        if shadow_pos["direction"] == "long" and cur <= float(shadow_pos["stop_price"]):
            reason = "stop_loss"
        elif shadow_pos["direction"] == "short" and cur >= float(shadow_pos["stop_price"]):
            reason = "stop_loss"
        elif shadow_pos["direction"] == "long" and cur >= float(shadow_pos["target_price"]):
            reason = "target_hit"
        elif shadow_pos["direction"] == "short" and cur <= float(shadow_pos["target_price"]):
            reason = "target_hit"
        elif shadow_pos["direction"] == "long" and profit_pct >= trail_trigger:
            if cur <= best_price * (1 - trail_pct):
                reason = "trailing_stop"
        elif shadow_pos["direction"] == "short" and profit_pct >= trail_trigger:
            if cur >= best_price * (1 + trail_pct):
                reason = "trailing_stop"
        if reason:
            _close_shadow_position(state, shadow_pos, cur, reason)
            updated = True
    if updated:
        _save_majorafter_state(state)


def _build_hourly_status_lines() -> list[str]:
    positions = oms.get_open_positions()
    ws_prices = ws.get_all_prices() if (ws and ws.is_alive()) else {}
    cum_pnl = oms.get_cumulative_pnl()
    today_closed = oms.get_today_closed("Europe/Istanbul")
    today_pnl = sum(float(p.get("pnl_usdt") or 0.0) for p in today_closed)
    today_wins = sum(1 for p in today_closed if float(p.get("pnl_usdt") or 0.0) > 0)
    today_losses = sum(1 for p in today_closed if float(p.get("pnl_usdt") or 0.0) < 0)
    open_rows = []
    open_total = 0.0
    open_wins = 0
    open_losses = 0
    for pos in positions:
        cur = ws_prices.get(pos["symbol"]) or _get_price(pos["symbol"]) or float(pos["entry_price"])
        pnl = oms.calc_pnl(pos, cur)
        pct = pnl / float(pos["amount_usdt"]) * 100.0 if float(pos["amount_usdt"]) else 0.0
        if pnl > 0:
            open_wins += 1
        elif pnl < 0:
            open_losses += 1
        open_total += pnl
        open_rows.append({
            "symbol": pos["symbol"],
            "direction": pos["direction"],
            "entry_price": float(pos["entry_price"]),
            "live_price": float(cur),
            "pnl": float(pnl),
            "pct": float(pct),
            "hold": _fmt_hold_short(pos["entry_date"]),
        })
    open_rows.sort(key=lambda item: item["pnl"], reverse=True)
    eq_now = CRYPTO_CAPITAL_USDT + cum_pnl + open_total
    eq_pct = (cum_pnl / CRYPTO_CAPITAL_USDT * 100.0) if CRYPTO_CAPITAL_USDT else 0.0
    now_ist = datetime.now(TZ_IST).strftime("%Y-%m-%d %H:%M")
    lines = [
        f"🪙 <b>CRYPTO SAATLIK DURUM</b>",
        f"Saat: {now_ist} TSİ",
        f"Gercek sermaye: {eq_now:.2f} USDT",
        f"Kapali toplam P&amp;L: {cum_pnl:+.2f} USDT ({eq_pct:+.2f}%)",
        f"Bugun kapanan: {len(today_closed)} | Kar: {today_wins} | Zarar: {today_losses} | Net: {today_pnl:+.2f} USDT",
        f"Acik pozisyon: {len(positions)} | Karda: {open_wins} | Zararda: {open_losses} | Unrealized: {open_total:+.2f} USDT",
    ]
    if open_rows:
        lines.append("Acik pozisyonlar:")
        for row in open_rows[:5]:
            side = "L" if row["direction"] == "long" else "S"
            coin = row["symbol"].split("/")[0]
            lines.append(
                f"{coin} {side} {row['pnl']:+.2f} USDT ({row['pct']:+.1f}%) | sure {row['hold']}"
            )
    else:
        lines.append("Acik pozisyonlar: yok")

    majorafter_state = _load_majorafter_state()
    if majorafter_state.get("locked"):
        locked_eq = float(majorafter_state.get("locked_equity") or 0.0)
        shadow_closed = majorafter_state.get("shadow_closed_positions", [])
        shadow_open = majorafter_state.get("shadow_open_positions", [])
        shadow_total = sum(float(p.get("pnl_usdt") or 0.0) for p in shadow_closed)
        lines.append(
            f"MAJORAFTER: AKTIF | lock_eq={locked_eq:.2f} | shadow_closed={len(shadow_closed)} | "
            f"shadow_open={len(shadow_open)} | shadow_pnl={shadow_total:+.2f} USDT"
        )
    else:
        target_eq = float(majorafter_state.get("target_equity") or 0.0)
        start_eq = float(majorafter_state.get("start_equity") or CRYPTO_CAPITAL_USDT)
        progress = ((eq_now - start_eq) / max(target_eq - start_eq, 1e-9) * 100.0) if target_eq > start_eq else 0.0
        lines.append(f"Gunluk hedef: {target_eq:.2f} USDT (+%{CRYPTO_DAILY_MAJOR_TARGET_PCT*100:.1f}) | Ilerleme: {progress:.1f}%")
    return lines


def _maybe_send_hourly_status() -> None:
    global _last_hourly_status_slot
    now_ist = datetime.now(TZ_IST)
    slot = now_ist.strftime("%Y-%m-%d %H")
    if _last_hourly_status_slot == slot:
        return
    if now_ist.minute != 0:
        return
    lines = _build_hourly_status_lines()
    print("[CRYPTO HOURLY] " + " | ".join(lines[1:4]))
    send_telegram_alert("\n".join(lines))
    _last_hourly_status_slot = slot


def _close_real_position(pos: dict, cur: float, reason: str, close_fraction: float = 1.0) -> None:
    from datetime import datetime, timezone as _tz

    symbol = pos["symbol"]
    pos_id = pos["id"]
    direction = pos["direction"]
    tier = get_tier(symbol)
    entry_price = pos["entry_price"]
    result = oms.create_sell(symbol, pos_id, cur, reason, signal_id=pos.get("signal_id"), close_fraction=close_fraction)
    if result.get("status") != "FILLED":
        msg = f"[CRYPTO EXIT] close rejected {symbol} reason={reason} detail={result.get('reason', 'unknown')}"
        print(msg)
        crypto_log.system(msg)
        return
    pnl = result.get("pnl_usdt", 0)
    is_partial = bool(result.get("is_partial"))
    closed_amount_usdt = float(result.get("closed_amount_usdt") or pos["amount_usdt"])
    side_str = "LONG" if direction == "long" else "SHORT"
    sign = "+" if pnl >= 0 else ""
    ws_tag = "(WS live)" if (ws and ws.is_alive()) else "(REST)"
    pct = pnl / closed_amount_usdt * 100 if closed_amount_usdt > 0 else 0
    try:
        entry_dt = datetime.fromisoformat(pos["entry_date"]).replace(tzinfo=_tz.utc)
        hold_secs = (datetime.now(_tz.utc) - entry_dt).total_seconds()
    except Exception:
        hold_secs = 0.0

    close_label = "PARTIAL" if is_partial else "EXIT"
    print(f"[CRYPTO {close_label}] [{tier}] {symbol} {side_str} | {reason} | "
          f"entry={entry_price:.4f} -> exit={cur:.4f} {ws_tag} | "
          f"P&L: {sign}{pnl:.2f} USDT ({sign}{pct:.2f}%)")
    algo_log.trade(
        f"[CRYPTO][{tier}] SELL {symbol} {side_str} | reason={reason} | "
        f"entry={entry_price:.4f} exit={cur:.4f} | "
        f"P&L={sign}{pnl:.2f} USDT ({sign}{pct:.2f}%) | size={closed_amount_usdt:.2f} USDT"
    )
    signal_id = result.get("signal_id") or pos.get("signal_id")
    if signal_id and not is_partial:
        cost_est = estimate_trade_costs(closed_amount_usdt)
        mark_signal_exit(
            signal_id=signal_id,
            exit_reason=reason,
            exit_order_id=result.get("order_id", ""),
            exit_price=float(cur),
            gross_pnl=float(pnl),
            net_pnl_estimate=float(pnl) - float(cost_est["total_cost_usdt"]),
            slippage_estimate=float(cost_est["slippage_usdt"]),
        )
    crypto_log.log_sell(symbol, tier, direction, entry_price, cur, pnl, pct, hold_secs, reason)
    if is_partial:
        journal_partial_close(symbol, direction, entry_price, cur, pnl, pct, hold_secs, reason, float(result.get("closed_fraction") or close_fraction))
        send_telegram_alert(
            f"🪙 [CRYPTO][{tier}] {symbol} {side_str} PARCA KAPATILDI\n"
            f"💰 P&L: {sign}${pnl:.2f} ({sign}{pct:.1f}%) | Kapanan: %{float(result.get('closed_fraction') or close_fraction)*100:.0f}\n"
            f"📌 Sebep: {reason}"
        )
    else:
        journal_close(symbol, direction, entry_price, cur, pnl, pct, hold_secs, reason)
        _tg_close(symbol, direction, pnl, pct, reason, cur, hold_secs)
        with _hwm_lock:
            _pos_hwm.pop(pos_id, None)
            # Remove from WS-callback index so stale pos_id is not updated
            entries = _symbol_pos_index.get(symbol, [])
            _symbol_pos_index[symbol] = [(pid, d) for pid, d in entries if pid != pos_id]


def _ensure_daily_major_lock() -> None:
    if _DRY_RUN_ONCE:
        return
    open_positions = oms.get_open_positions()
    current_equity = _current_equity(open_positions)
    state = _load_majorafter_state(current_equity)
    if state.get("locked"):
        return
    target_equity = float(state.get("target_equity") or 0.0)
    if current_equity < target_equity:
        return
    for pos in list(open_positions):
        cur = _get_price(pos["symbol"]) or float(pos["entry_price"])
        _close_real_position(pos, cur, "daily_major_target")
    locked_equity = _current_equity([])
    state["locked"] = True
    state["locked_at"] = datetime.now(timezone.utc).isoformat()
    state["locked_equity"] = round(locked_equity, 4)
    state["lock_reason"] = "daily_major_target_hit"
    _save_majorafter_state(state)
    msg = (
        f"[MAJOR TARGET] Daily target hit | equity={locked_equity:.2f} "
        f"target={target_equity:.2f} -> real trading locked, shadow book active"
    )
    print(msg)
    crypto_log.system(msg)


# ── ML helpers ────────────────────────────────────────────────────────────────

def _load_crypto_model() -> None:
    global _crypto_model, _crypto_model_long, _crypto_model_short
    global _tier_models_long, _tier_models_short
    global _threshold_config, _short_threshold_config
    global _long_model_active, _short_model_active, _model_load_time
    _model_load_time = time.time()
    try:
        from crypto_ml import (
            load_model,
            load_model_long,
            load_model_short,
            load_tier_model,
            artifact_runtime_approved,
        )

        _legacy_model_warnings.clear()
        _crypto_model = load_model()
        raw_long = load_model_long()
        raw_short = load_model_short()
        raw_long_major = load_tier_model("long", "MAJOR")
        raw_long_risky = load_tier_model("long", "RISKY")
        raw_short_major = load_tier_model("short", "MAJOR")
        raw_short_risky = load_tier_model("short", "RISKY")
        long_runtime_ok, long_reason = artifact_runtime_approved(raw_long, side="long")
        short_runtime_ok, short_reason = artifact_runtime_approved(raw_short, side="short")

        def _pick_tier(raw_model: dict | None, side: str, tier: str) -> dict | None:
            ok, _reason = artifact_runtime_approved(raw_model, side=side)
            if ok:
                return raw_model
            return None

        _tier_models_long = {
            "MAJOR": _pick_tier(raw_long_major, "long", "MAJOR"),
            "RISKY": _pick_tier(raw_long_risky, "long", "RISKY"),
        }
        _tier_models_short = {
            "MAJOR": _pick_tier(raw_short_major, "short", "MAJOR"),
            "RISKY": _pick_tier(raw_short_risky, "short", "RISKY"),
        }
        _crypto_model_long = raw_long if long_runtime_ok else None
        _crypto_model_short = raw_short if SHORT_MODEL_ACTIVE and short_runtime_ok else None
        _long_model_active = any(model is not None for model in _tier_models_long.values()) or (_crypto_model_long is not None) or (_crypto_model is not None)
        _short_model_active = any(model is not None for model in _tier_models_short.values()) or (_crypto_model_short is not None)

        _threshold_config = dict(LEGACY_THRESHOLD_CONFIG)
        for tier_name in ("MAJOR", "RISKY"):
            tier_model = _tier_models_long.get(tier_name)
            if tier_model and tier_model.get("threshold_config", {}).get(tier_name):
                _threshold_config[tier_name] = tier_model["threshold_config"][tier_name]
            elif _crypto_model_long and _crypto_model_long.get("threshold_config", {}).get(tier_name):
                _threshold_config[tier_name] = _crypto_model_long["threshold_config"][tier_name]
            elif _crypto_model and _crypto_model.get("threshold_config", {}).get(tier_name):
                _threshold_config[tier_name] = _crypto_model["threshold_config"][tier_name]

        _short_threshold_config = dict(LEGACY_THRESHOLD_CONFIG)
        for tier_name in ("MAJOR", "RISKY"):
            tier_model = _tier_models_short.get(tier_name)
            if tier_model and tier_model.get("threshold_config", {}).get(tier_name):
                _short_threshold_config[tier_name] = tier_model["threshold_config"][tier_name]
            elif _crypto_model_short and _crypto_model_short.get("threshold_config", {}).get(tier_name):
                _short_threshold_config[tier_name] = _crypto_model_short["threshold_config"][tier_name]

        for tier_name, model in _tier_models_long.items():
            if model is not None:
                print(f"[CRYPTO] {tier_name} LONG active | WF={model.get('wf_precision', 0.0):.1%} | trained={str(model.get('trained_at', 'n/a'))[:10]}")
            elif _crypto_model_long is not None or _crypto_model is not None:
                fallback_name = "crypto_xgb_long.pkl" if _crypto_model_long is not None else "crypto_xgb.pkl"
                print(f"[CRYPTO] WARN: {tier_name} LONG tier artifact yok/rejected — fallback={fallback_name}")
            else:
                print(f"[CRYPTO] WARN: {tier_name} LONG modeli yok — {tier_name} LONG entry bloklu")

        for tier_name, model in _tier_models_short.items():
            if model is not None:
                print(f"[CRYPTO] {tier_name} SHORT active | WF={model.get('wf_precision', 0.0):.1%} | trained={str(model.get('trained_at', 'n/a'))[:10]}")
            elif _crypto_model_short is not None:
                print(f"[CRYPTO] WARN: {tier_name} SHORT tier artifact yok/rejected — fallback=crypto_xgb_short.pkl")
            else:
                print(f"[CRYPTO] WARN: {tier_name} SHORT modeli yok — {tier_name} SHORT entry bloklu")

        for tier_name in ["MAJOR", "RISKY"]:
            tier_cfg = (_threshold_config or LEGACY_THRESHOLD_CONFIG).get(tier_name, {})
            shown = tier_cfg.get("selected_threshold")
            if shown is None:
                shown = tier_cfg.get("fallback_threshold")
            print(
                f"[CRYPTO] {tier_name} LONG threshold: {shown} | "
                f"enabled={tier_cfg.get('enabled', True)} | "
                f"status={tier_cfg.get('selection_status', 'n/a')}"
            )
    except Exception as exc:
        print(f"[CRYPTO] WARN: ML model yuklenemedi: {exc}")


def _load_threshold_config_from_report(model_dict: dict | None) -> dict | None:
    try:
        if not THRESHOLD_REPORT_PATH.exists():
            return None
        report = json.loads(THRESHOLD_REPORT_PATH.read_text(encoding="utf-8"))
        report_trained_at = report.get("trained_at")
        model_trained_at = None if model_dict is None else model_dict.get("trained_at")
        if model_trained_at and report_trained_at != model_trained_at:
            return None
        return report.get("tiers")
    except Exception as exc:
        print(f"[CRYPTO] threshold_report okunamadi: {exc}")
        return None


def _prepare_ml_inputs(symbol: str, df_1h, df_5m, btc_df):
    """
    Shared input preparation for ML inference.
    Returns (fg_dict_ml, btc_df_ml, df_5m_ml, funding_info, btc_dom_info, htf_info).
    """
    from crypto_indicators import add_rsi, add_macd, add_bollinger

    # F&G dict: cover last 5 days so all 72 1h bars can match
    try:
        fg_data = _fg_sentiment.get_fear_greed()
        fg_val  = fg_data["value"]
    except Exception as _e:
        print(f"[CRYPTO] F&G predict fetch hatasi: {_e} — neutral(50) kullaniliyor")
        fg_val = 50
    _now = datetime.now(timezone.utc)
    fg_dict_ml: dict = {
        (_now - timedelta(days=d)).strftime("%Y-%m-%d"): fg_val
        for d in range(5)
    }

    # BTC 1h df — fetch fresh if missing
    btc_df_ml = btc_df
    if btc_df_ml is None or len(btc_df_ml) < 2:
        try:
            btc_df_ml = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=10)
            if btc_df_ml is not None:
                btc_df_ml = compute_all(btc_df_ml)
        except Exception as _e:
            print(f"[CRYPTO] BTC 1h fetch predict hatasi: {_e} — btc_df=None")
            btc_df_ml = None

    # 5m MTF — fetch and compute indicators if missing
    df_5m_ml = df_5m
    if df_5m_ml is None or len(df_5m_ml) < 20:
        try:
            df_5m_ml = stream.get_ohlcv(symbol, timeframe="5m", limit=60)
        except Exception as _e:
            print(f"[CRYPTO] 5m fetch predict hatasi ({symbol}): {_e}")
            df_5m_ml = None
    if df_5m_ml is not None and "rsi14" not in df_5m_ml.columns:
        try:
            df_5m_ml = add_rsi(df_5m_ml.copy())
            df_5m_ml = add_macd(df_5m_ml)
            df_5m_ml = add_bollinger(df_5m_ml)
        except Exception as _e:
            print(f"[CRYPTO] 5m indicator compute hatasi ({symbol}): {_e}")
            df_5m_ml = None

    try:
        funding_info = fetch_funding_rate(symbol)
    except Exception as _e:
        print(f"[CRYPTO] funding fetch hatasi ({symbol}): {_e}")
        funding_info = {"funding_rate": 0.0, "funding_rate_avg3": 0.0}

    try:
        btc_dom_info = get_btc_dominance_features()
    except Exception as _e:
        print(f"[CRYPTO] btc dominance fetch hatasi: {_e}")
        btc_dom_info = {
            "btc_dominance": 0.0,
            "btc_dominance_regime": "neutral",
            "btc_dominance_regime_code": 1.0,
            "btc_dominance_7d_change": 0.0,
        }

    try:
        htf_info = compute_htf_bias(stream, symbol)
    except Exception as _e:
        print(f"[CRYPTO] HTF bias compute hatasi ({symbol}): {_e}")
        htf_info = {"htf_bias": "BEAR", "htf_close": 0.0, "htf_ema50": 0.0}

    return fg_dict_ml, btc_df_ml, df_5m_ml, funding_info, btc_dom_info, htf_info


def _predict_proba(symbol: str, df_1h, df_5m=None, btc_df=None, side: str = "long", ml_inputs=None) -> float:
    """
    Run ML prediction for the last bar of df_1h.

    side="long"  → uses _crypto_model_long  (prob_long)
    side="short" → uses _crypto_model_short (prob_short) only when short ML is active

    Fallback:
      side="long"  → 0.5 if no approved active LONG artifact exists
      side="short" → 0.5 (no inverted-long fallback)

    Returns float in [0, 1]. Returns 0.5 if all models are missing.
    """
    from crypto_ml import predict_proba

    # Choose model
    model_dict = _get_runtime_model(symbol, side)
    if model_dict is None:
        return 0.5

    try:
        if ml_inputs is None:
            ml_inputs = _prepare_ml_inputs(symbol, df_1h, df_5m, btc_df)
        (
            fg_dict_ml,
            btc_df_ml,
            df_5m_ml,
            funding_info,
            btc_dom_info,
            htf_info,
        ) = ml_inputs
        prob = predict_proba(
            model_dict,
            df_1h,
            btc_df_ml,
            fg_dict_ml,
            df_5m_ml,
            symbol=symbol,
            funding_info=funding_info,
            btc_dominance_info=btc_dom_info,
            htf_info=htf_info,
        )

        return prob
    except Exception as exc:
        print(f"[CRYPTO] predict_proba({symbol}, side={side}) hatasi: {exc}")
        return 0.5


# ── Market regime filter ─────────────────────────────────────────────────────

def _fetch_symbol_1h_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in CRYPTO_SYMBOLS:
        df_1h = stream.get_ohlcv(symbol, timeframe="1h", limit=REGIME_1H_LOOKBACK)
        if df_1h is None or len(df_1h) < 51:
            continue
        try:
            frames[symbol] = compute_all(df_1h)
        except Exception as exc:
            print(f"[CRYPTO] {symbol} 1h feature compute hatasi: {exc}")
    return frames


def _hour_return(df: pd.DataFrame, bars: int) -> float:
    if df is None or len(df) <= bars:
        return 0.0
    close_now = float(df["close"].iloc[-1])
    close_prev = float(df["close"].iloc[-(bars + 1)])
    if close_prev <= 0:
        return 0.0
    return (close_now / close_prev) - 1.0


def _realized_vol(df: pd.DataFrame, window: int) -> float:
    if df is None or len(df) < window + 1:
        return 0.0
    returns = df["close"].pct_change().dropna()
    if len(returns) < window:
        return 0.0
    return float(returns.tail(window).std())


def _ema_trend_state(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"bullish": False, "bearish": False, "label": "unknown"}
    close = float(df["close"].iloc[-1])
    ema21 = float(df["ema21"].iloc[-1]) if "ema21" in df.columns and pd.notna(df["ema21"].iloc[-1]) else close
    ema50 = float(df["ema50"].iloc[-1]) if "ema50" in df.columns and pd.notna(df["ema50"].iloc[-1]) else close
    bullish = close > ema50 and ema21 >= ema50
    bearish = close < ema50 and ema21 <= ema50
    label = "bullish" if bullish else "bearish" if bearish else "mixed"
    return {
        "bullish": bullish,
        "bearish": bearish,
        "label": label,
        "close": close,
        "ema21": ema21,
        "ema50": ema50,
    }


def _market_breadth_ratio(frames: dict[str, pd.DataFrame]) -> float:
    samples = 0
    above = 0
    for df in frames.values():
        if df.empty or REGIME_BREADTH_EMA_COL not in df.columns:
            continue
        ema_val = df[REGIME_BREADTH_EMA_COL].iloc[-1]
        if pd.isna(ema_val):
            continue
        samples += 1
        above += int(float(df["close"].iloc[-1]) > float(ema_val))
    return float(above / samples) if samples else 0.0


def _avg_correlation_snapshot(frames: dict[str, pd.DataFrame]) -> dict:
    btc_df = frames.get("BTC/USDT")
    eth_df = frames.get("ETH/USDT")
    if btc_df is None or eth_df is None:
        return {"recent": 0.0, "baseline": 0.0, "spike": 0.0, "sample_count": 0}

    proxy = pd.DataFrame({
        "btc": btc_df["close"].pct_change(),
        "eth": eth_df["close"].pct_change(),
    }).dropna()
    if len(proxy) < REGIME_CORR_WINDOW + REGIME_CORR_BASELINE_WINDOW:
        return {"recent": 0.0, "baseline": 0.0, "spike": 0.0, "sample_count": 0}
    proxy_ret = proxy.mean(axis=1)

    recent_vals: list[float] = []
    baseline_vals: list[float] = []
    for symbol, df in frames.items():
        if symbol in REGIME_SYMBOLS_PROXY:
            continue
        series = df["close"].pct_change().dropna()
        joined = pd.concat([proxy_ret.rename("proxy"), series.rename("sym")], axis=1).dropna()
        need = REGIME_CORR_WINDOW + REGIME_CORR_BASELINE_WINDOW
        if len(joined) < need:
            continue
        recent = joined.iloc[-REGIME_CORR_WINDOW:]
        baseline = joined.iloc[-need:-REGIME_CORR_WINDOW]
        recent_corr = recent["proxy"].corr(recent["sym"])
        baseline_corr = baseline["proxy"].corr(baseline["sym"])
        if pd.notna(recent_corr):
            recent_vals.append(float(recent_corr))
        if pd.notna(baseline_corr):
            baseline_vals.append(float(baseline_corr))

    recent_avg = float(np.mean(recent_vals)) if recent_vals else 0.0
    baseline_avg = float(np.mean(baseline_vals)) if baseline_vals else 0.0
    return {
        "recent": recent_avg,
        "baseline": baseline_avg,
        "spike": recent_avg - baseline_avg,
        "sample_count": len(recent_vals),
    }


def _compute_dynamic_correlation_gate(frames: dict[str, pd.DataFrame]) -> dict[str, float | str | int]:
    """Compute BTC-vs-market average 24h rolling correlation and dynamic threshold."""
    btc_df = frames.get("BTC/USDT")
    if btc_df is None or len(btc_df) < CRYPTO_CORR_WINDOW + 1:
        return {
            "avg_corr": 0.0,
            "threshold": CRYPTO_CORR_THRESHOLD,
            "sample_count": 0,
            "market_state": "normal",
        }

    btc_ret = btc_df["close"].pct_change().dropna().tail(CRYPTO_CORR_WINDOW)
    corr_vals: list[float] = []
    for symbol in CRYPTO_SYMBOLS:
        if symbol == "BTC/USDT":
            continue
        df = frames.get(symbol)
        if df is None or len(df) < CRYPTO_CORR_WINDOW + 1:
            continue
        sym_ret = df["close"].pct_change().dropna().tail(CRYPTO_CORR_WINDOW)
        joined = pd.concat([btc_ret.rename("btc"), sym_ret.rename("sym")], axis=1).dropna()
        if len(joined) < 12:
            continue
        corr = joined["btc"].corr(joined["sym"])
        if pd.notna(corr):
            corr_vals.append(float(corr))

    avg_corr = float(np.mean(corr_vals)) if corr_vals else 0.0
    if avg_corr > 0.90:
        threshold = 0.95
        market_state = "panic"
    elif avg_corr < 0.70:
        threshold = 0.75
        market_state = "calm"
    else:
        threshold = 0.85
        market_state = "normal"
    return {
        "avg_corr": avg_corr,
        "threshold": threshold,
        "sample_count": len(corr_vals),
        "market_state": market_state,
    }


def _funding_gate_snapshot() -> dict[str, object]:
    rates = _fetch_funding_rates()
    avg_rate = float(sum(rates.values()) / len(rates)) if rates else float(_get_avg_funding_rate() or 0.0)
    if avg_rate > FUNDING_SENTIMENT_THRESHOLD:
        bias = "overleveraged_long"
        return {
            "rates": rates,
            "avg_rate": avg_rate,
            "bias": bias,
            "long_block": True,
            "short_block": False,
            "long_size_mult": 1.0,
            "short_size_mult": FUNDING_SIZE_BOOST,
        }
    if avg_rate < -FUNDING_SENTIMENT_THRESHOLD:
        bias = "overleveraged_short"
        return {
            "rates": rates,
            "avg_rate": avg_rate,
            "bias": bias,
            "long_block": False,
            "short_block": True,
            "long_size_mult": FUNDING_SIZE_BOOST,
            "short_size_mult": 1.0,
        }
    return {
        "rates": rates,
        "avg_rate": avg_rate,
        "bias": "neutral",
        "long_block": False,
        "short_block": False,
        "long_size_mult": 1.0,
        "short_size_mult": 1.0,
    }


def compute_btc_regime() -> tuple[str, dict]:
    """
    Two-layer BTC regime detection.

    Layer 1 — Structural trend (slow):
      BTC 1h close > EMA200 → STRUCTURAL_BULL
      BTC 1h close < EMA200 → STRUCTURAL_BEAR

    Layer 2 — Short-term momentum (fast):
      1h return > +1% → MOMENTUM_UP
      1h return < -1% → MOMENTUM_DOWN
      else            → MOMENTUM_NEUTRAL

    Combined regime matrix:
      STRUCTURAL_BULL + MOMENTUM_UP       → BULL    (LONG only)
      STRUCTURAL_BULL + MOMENTUM_DOWN     → CAUTION (LONG allowed, size x0.7)
      STRUCTURAL_BEAR + MOMENTUM_DOWN     → BEAR    (SHORT only)
      STRUCTURAL_BEAR + MOMENTUM_UP       → CAUTION (SHORT allowed, size x0.7)
      any + MOMENTUM_NEUTRAL              → NEUTRAL (both, ML thr +5%)

    Returns (regime_str, detail_dict).
    Fails-open to ("NEUTRAL", {...}) on any data / fetch error.
    """
    default_detail = {
        "structural": "UNKNOWN",
        "momentum": "UNKNOWN",
        "combined": "NEUTRAL",
        "size_mult": 1.0,
        "btc_close": 0.0,
        "ema200": 0.0,
        "ret_1h": 0.0,
        "caution_direction": None,  # 'long' or 'short' — which side is allowed in CAUTION
    }
    try:
        df = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=210)
        if df is None or len(df) < 201:
            return "NEUTRAL", default_detail

        close = df["close"].astype(float)
        ema200 = close.ewm(span=200, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema200 = float(ema200.iloc[-1])
        prev_close = float(close.iloc[-2])

        # Layer 1: Structural
        structural = "STRUCTURAL_BULL" if last_close > last_ema200 else "STRUCTURAL_BEAR"

        # Layer 2: Momentum (1h return)
        ret_1h = (last_close - prev_close) / prev_close if prev_close > 0 else 0.0
        if ret_1h > 0.01:
            momentum = "MOMENTUM_UP"
        elif ret_1h < -0.01:
            momentum = "MOMENTUM_DOWN"
        else:
            momentum = "MOMENTUM_NEUTRAL"

        # Combined matrix
        if momentum == "MOMENTUM_NEUTRAL":
            combined = "NEUTRAL"
            size_mult = 1.0
            caution_dir = None
        elif structural == "STRUCTURAL_BULL" and momentum == "MOMENTUM_UP":
            combined = "BULL"
            size_mult = 1.0
            caution_dir = None
        elif structural == "STRUCTURAL_BULL" and momentum == "MOMENTUM_DOWN":
            combined = "CAUTION"
            size_mult = 0.7
            caution_dir = "long"   # LONG allowed at reduced size
        elif structural == "STRUCTURAL_BEAR" and momentum == "MOMENTUM_DOWN":
            combined = "BEAR"
            size_mult = 1.0
            caution_dir = None
        elif structural == "STRUCTURAL_BEAR" and momentum == "MOMENTUM_UP":
            combined = "CAUTION"
            size_mult = 0.7
            caution_dir = "short"  # SHORT allowed at reduced size
        else:
            combined = "NEUTRAL"
            size_mult = 1.0
            caution_dir = None

        detail = {
            "structural": structural,
            "momentum": momentum,
            "combined": combined,
            "size_mult": size_mult,
            "btc_close": last_close,
            "ema200": last_ema200,
            "ret_1h": ret_1h,
            "caution_direction": caution_dir,
        }
        return combined, detail

    except Exception:
        return "NEUTRAL", default_detail


def _compute_market_regime(frames: dict[str, pd.DataFrame]) -> dict:
    if not ENABLE_MARKET_REGIME_FILTER:
        return {
            "enabled": False,
            "regime": "NEUTRAL",
            "reason": "market regime filter disabled by config",
            "signals": [],
            "block_new_longs": False,
            "size_mult": 1.0,
            "trend_mtf_threshold_add": 0,
            "trend_ml_threshold_add": 0.0,
        }

    btc_df = frames.get("BTC/USDT")
    eth_df = frames.get("ETH/USDT")
    if btc_df is None or eth_df is None:
        return {
            "enabled": True,
            "regime": "NEUTRAL",
            "reason": "missing BTC/ETH proxy data, fail-open neutral",
            "signals": ["proxy_data_missing"],
            "block_new_longs": False,
            "size_mult": 1.0,
            "trend_mtf_threshold_add": 0,
            "trend_ml_threshold_add": 0.0,
        }

    btc_ret_24h = _hour_return(btc_df, REGIME_RET_LOOKBACK_HOURS)
    eth_ret_24h = _hour_return(eth_df, REGIME_RET_LOOKBACK_HOURS)
    btc_rvol = _realized_vol(btc_df, REGIME_VOL_LOOKBACK_HOURS)
    btc_trend = _ema_trend_state(btc_df)
    eth_trend = _ema_trend_state(eth_df)
    breadth = _market_breadth_ratio(frames)
    corr = _avg_correlation_snapshot(frames)

    signals: list[str] = []
    regime = "NEUTRAL"
    block_new_longs = False
    size_mult = 1.0
    mtf_add = 0
    ml_add = 0.0

    crash = (
        btc_ret_24h <= REGIME_CRASH_BTC_RET
        or eth_ret_24h <= REGIME_CRASH_ETH_RET
        or (
            btc_trend["bearish"]
            and breadth <= REGIME_CRASH_BREADTH_MAX
            and (btc_rvol >= REGIME_CRASH_BTC_RVOL or corr["spike"] >= REGIME_CRASH_CORR_SPIKE)
        )
    )
    risk_on = (
        btc_ret_24h >= REGIME_RISK_ON_BTC_RET
        and eth_ret_24h >= REGIME_RISK_ON_ETH_RET
        and btc_trend["bullish"]
        and eth_trend["bullish"]
        and breadth >= REGIME_RISK_ON_BREADTH_MIN
        and corr["spike"] < REGIME_CRASH_CORR_SPIKE
    )
    chop = (
        abs(btc_ret_24h) <= REGIME_CHOP_RET_ABS_MAX
        and abs(eth_ret_24h) <= REGIME_CHOP_ETH_RET_ABS_MAX
        and REGIME_CHOP_BREADTH_MIN <= breadth <= REGIME_CHOP_BREADTH_MAX
        and corr["recent"] >= REGIME_CHOP_CORR_MIN
    )

    if crash:
        regime = "CRASH_RISK"
        block_new_longs = REGIME_CRASH_BLOCK_LONGS
        size_mult = REGIME_CRASH_SIZE_MULT if not REGIME_CRASH_BLOCK_LONGS else 1.0
        if btc_ret_24h <= REGIME_CRASH_BTC_RET:
            signals.append(f"btc_24h_return={btc_ret_24h:+.2%}")
        if eth_ret_24h <= REGIME_CRASH_ETH_RET:
            signals.append(f"eth_24h_return={eth_ret_24h:+.2%}")
        if btc_trend["bearish"]:
            signals.append("btc_ema_trend=bearish")
        if breadth <= REGIME_CRASH_BREADTH_MAX:
            signals.append(f"breadth={breadth:.0%}")
        if btc_rvol >= REGIME_CRASH_BTC_RVOL:
            signals.append(f"btc_realized_vol={btc_rvol:.2%}")
        if corr["spike"] >= REGIME_CRASH_CORR_SPIKE:
            signals.append(f"corr_spike={corr['spike']:+.2f}")
    elif risk_on:
        regime = "RISK_ON"
        signals.extend([
            f"btc_24h_return={btc_ret_24h:+.2%}",
            f"eth_24h_return={eth_ret_24h:+.2%}",
            f"breadth={breadth:.0%}",
            "btc_ema_trend=bullish",
            "eth_ema_trend=bullish",
        ])
    elif chop:
        regime = "CHOP"
        mtf_add = REGIME_CHOP_MTF_ADD
        ml_add = REGIME_CHOP_ML_ADD
        signals.extend([
            f"btc_24h_return={btc_ret_24h:+.2%}",
            f"eth_24h_return={eth_ret_24h:+.2%}",
            f"breadth={breadth:.0%}",
            f"avg_corr_recent={corr['recent']:.2f}",
        ])
    else:
        signals.extend([
            f"btc_24h_return={btc_ret_24h:+.2%}",
            f"btc_realized_vol={btc_rvol:.2%}",
            f"breadth={breadth:.0%}",
            f"avg_corr_spike={corr['spike']:+.2f}",
        ])

    reason = (
        f"{regime} | btc24h={btc_ret_24h:+.2%} eth24h={eth_ret_24h:+.2%} "
        f"btc_rvol={btc_rvol:.2%} breadth={breadth:.0%} "
        f"btc_trend={btc_trend['label']} eth_trend={eth_trend['label']} "
        f"corr_recent={corr['recent']:.2f} corr_spike={corr['spike']:+.2f}"
    )
    return {
        "enabled": True,
        "regime": regime,
        "reason": reason,
        "signals": signals,
        "btc_24h_return": btc_ret_24h,
        "eth_24h_return": eth_ret_24h,
        "btc_realized_vol": btc_rvol,
        "btc_ema_trend": btc_trend["label"],
        "eth_ema_trend": eth_trend["label"],
        "market_breadth": breadth,
        "avg_corr_recent": corr["recent"],
        "avg_corr_baseline": corr["baseline"],
        "avg_corr_spike": corr["spike"],
        "correlation_samples": corr["sample_count"],
        "block_new_longs": block_new_longs,
        "size_mult": size_mult,
        "trend_mtf_threshold_add": mtf_add,
        "trend_ml_threshold_add": ml_add,
    }


def _log_market_regime(snapshot: dict) -> None:
    msg = (
        f"[CRYPTO REGIME] {snapshot.get('regime')} | "
        f"{snapshot.get('reason')} | signals={', '.join(snapshot.get('signals', [])) or 'none'}"
    )
    print(msg)
    crypto_log.system(msg)


def _descending_pct_snapshot(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return out
    ranks = valid.rank(method="average", ascending=False)
    if len(valid) == 1:
        out.loc[valid.index] = 0.5
    else:
        out.loc[valid.index] = 1.0 - ((ranks - 1.0) / (len(valid) - 1.0))
    return out


def _build_relative_strength_snapshot(frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    rows = []
    for symbol, df in frames.items():
        if df is None or len(df) <= RS_BONUS_LOOKBACK_HOURS:
            continue
        close = df["close"].astype(float)
        ret_1h = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else np.nan
        ret_4h = float(close.pct_change(4).iloc[-1]) if len(close) > 4 else np.nan
        ret_24h = float(close.pct_change(RS_BONUS_LOOKBACK_HOURS).iloc[-1]) if len(close) > RS_BONUS_LOOKBACK_HOURS else np.nan
        vol_ratio = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns and pd.notna(df["vol_ratio"].iloc[-1]) else np.nan
        rows.append({
            "symbol": symbol,
            "ret_1h": ret_1h,
            "ret_4h": ret_4h,
            "ret_24h": ret_24h,
            "vol_ratio": vol_ratio,
        })

    if not rows:
        return {}

    snap = pd.DataFrame(rows).set_index("symbol")
    snap["ret_1h_pct"] = _descending_pct_snapshot(snap["ret_1h"])
    snap["ret_4h_pct"] = _descending_pct_snapshot(snap["ret_4h"])
    snap["ret_24h_pct"] = _descending_pct_snapshot(snap["ret_24h"])
    snap["vol_ratio_pct"] = _descending_pct_snapshot(snap["vol_ratio"])
    btc_ret_24h = float(snap.loc["BTC/USDT", "ret_24h"]) if "BTC/USDT" in snap.index and pd.notna(snap.loc["BTC/USDT", "ret_24h"]) else 0.0
    snap["rel_btc_ret_24h"] = snap["ret_24h"] - btc_ret_24h
    if "BTC/USDT" in snap.index:
        snap.loc["BTC/USDT", "rel_btc_ret_24h"] = 0.0
    return snap.to_dict(orient="index")


def _relative_strength_mtf_adjustment(symbol: str) -> tuple[int, int, str]:
    if not ENABLE_RELATIVE_STRENGTH_MTF:
        return 0, 0, "disabled"
    snap = _relative_strength_snapshot.get(symbol)
    if not snap:
        return 0, 0, "missing"
    pct_24h = snap.get("ret_24h_pct")
    rel_btc = snap.get("rel_btc_ret_24h")
    if pct_24h is None or rel_btc is None or pd.isna(pct_24h) or pd.isna(rel_btc):
        return 0, 0, "insufficient"
    if pct_24h >= RS_BONUS_TOP_PCT and rel_btc > 0:
        return RS_MTF_BONUS_POINTS, 0, f"rs_long_bonus pct={pct_24h:.2f} rel_btc={rel_btc:+.2%}"
    if pct_24h <= RS_BONUS_BOTTOM_PCT and rel_btc < 0:
        return 0, RS_MTF_BONUS_POINTS, f"rs_short_bonus pct={pct_24h:.2f} rel_btc={rel_btc:+.2%}"
    return 0, 0, f"neutral pct={pct_24h:.2f} rel_btc={rel_btc:+.2%}"


# ── Price helper ──────────────────────────────────────────────────────────────

def _get_price(symbol: str) -> float | None:
    """WS price first, fall back to REST ticker if WS stale or not connected."""
    if ws is not None and ws.is_alive():
        p = ws.get_price(symbol)
        if p and p > 0:
            return p
    ticker = stream.get_ticker(symbol)
    return ticker["last"] if ticker and ticker.get("last") else None


# ── BTC gates ─────────────────────────────────────────────────────────────────

def _check_btc_gate() -> tuple[bool, float]:
    """
    Compute BTC 1h return. Block all entries if crash < -3%.
    Returns (gate_passed, btc_1h_return). Fails open on fetch error.
    """
    global _btc_1h_return
    df = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=3)
    if df is None or len(df) < 2:
        return True, 0.0

    prev = df["close"].iloc[-2]
    last = df["close"].iloc[-1]
    ret  = (last - prev) / prev if prev > 0 else 0.0
    _btc_1h_return = ret

    if ret < CRYPTO_BTC_CRASH_THRESHOLD:
        pct = ret * 100
        msg = f"[CRYPTO GATE] BTC crash {pct:.2f}% → entries blocked"
        print(msg)
        algo_log.risk(msg)
        send_telegram_alert(f"⚠️ [CRYPTO] BTC ani dusu {pct:.2f}% — yeni giris yok")
        return False, ret
    return True, ret


_BTC_CORR_SYMBOLS = {"ETH/USDT", "SOL/USDT", "BNB/USDT"}   # gate applies only to these alts


def _get_btc_4h_trend() -> str:
    """
    Return 'BULLISH' or 'BEARISH' based on BTC 4h close vs EMA21.
    Returns 'UNKNOWN' on fetch failure (gate is skipped when unknown).
    """
    df_4h = stream.get_ohlcv("BTC/USDT", timeframe="4h", limit=25)
    if df_4h is None or len(df_4h) < 22:
        return "UNKNOWN"
    ema21 = df_4h["close"].ewm(span=21, adjust=False).mean()
    last_close = float(df_4h["close"].iloc[-1])
    last_ema21 = float(ema21.iloc[-1])
    return "BULLISH" if last_close > last_ema21 else "BEARISH"


def _check_btc_correlation(symbol: str, df_alt, direction: str = "long") -> tuple[bool, str]:
    """
    Symmetric BTC-correlation gate for high-corr alts (ETH/SOL/BNB).

    Blocks:
      LONG  when corr > threshold AND BTC 4h trend is BEARISH
      SHORT when corr > threshold AND BTC 4h trend is BULLISH

    Returns (allowed: bool, status_tag: str) where status_tag is used by
    _print_gate_summary() — "PASS", "BLOCK_LONG", "BLOCK_SHORT", or "N/A".
    """
    if symbol not in _BTC_CORR_SYMBOLS:
        return True, "N/A"

    df_btc = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=CRYPTO_CORR_WINDOW + 2)
    if df_btc is None or len(df_btc) < CRYPTO_CORR_WINDOW:
        return True, "PASS"   # fail-open: data unavailable

    btc_ret = df_btc["close"].pct_change().dropna().tail(CRYPTO_CORR_WINDOW).values
    alt_ret = df_alt["close"].pct_change().dropna().tail(CRYPTO_CORR_WINDOW).values
    n = min(len(btc_ret), len(alt_ret))
    if n < 10:
        return True, "PASS"

    try:
        corr = float(np.corrcoef(btc_ret[-n:], alt_ret[-n:])[0, 1])
    except Exception:
        return True, "PASS"

    dyn_threshold = float(_corr_gate_snapshot.get("threshold", CRYPTO_CORR_THRESHOLD) or CRYPTO_CORR_THRESHOLD)
    if corr <= dyn_threshold:
        return True, f"PASS(corr={corr:.2f})"

    # High correlation — check BTC trend direction
    btc_trend = _get_btc_4h_trend()
    if btc_trend == "UNKNOWN":
        return True, "PASS"   # fail-open: 4h data unavailable

    if direction == "long" and btc_trend == "BEARISH":
        msg = (f"[CRYPTO GATE] CORR_GATE blocked LONG {symbol}: "
               f"BTC bearish (4h EMA21), corr={corr:.2f}, dyn_thr={dyn_threshold:.2f}")
        print(msg)
        algo_log.risk(msg)
        return False, f"BLOCK_LONG(corr={corr:.2f},thr={dyn_threshold:.2f},btc=BEAR)"

    if direction == "short" and btc_trend == "BULLISH":
        msg = (f"[CRYPTO GATE] CORR_GATE blocked SHORT {symbol}: "
               f"BTC bullish (4h EMA21), corr={corr:.2f}, dyn_thr={dyn_threshold:.2f}")
        print(msg)
        algo_log.risk(msg)
        return False, f"BLOCK_SHORT(corr={corr:.2f},thr={dyn_threshold:.2f},btc=BULL)"

    return True, f"PASS(corr={corr:.2f},thr={dyn_threshold:.2f},btc={btc_trend[:4]})"


# ── ATR sizing ────────────────────────────────────────────────────────────────

def _atr_position_size(df, price: float, fg_val: int,
                       max_size_usdt: float | None = None,
                       capital_base: float = CRYPTO_CAPITAL_USDT) -> float:
    """Risk 1% of capital per trade. Apply F&G size modifier."""
    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns else 0.0
    hard_cap = ((max_size_usdt if max_size_usdt is not None
                 else capital_base / CRYPTO_MAX_POSITIONS)
                * CRYPTO_SIZE_FEAR_MAX)
    if atr <= 0 or price <= 0:
        return round(hard_cap, 2)
    stop_dist  = atr * CRYPTO_ATR_STOP_MULT
    risk_usdt  = capital_base * CRYPTO_RISK_PCT
    atr_size   = risk_usdt / (stop_dist / price)
    base       = min(atr_size, hard_cap)
    if fg_val < CRYPTO_FG_EXTREME_FEAR:
        base *= 1.3
    elif fg_val > CRYPTO_FG_EXTREME_GREED:
        base *= 0.7
    return max(round(min(base, hard_cap), 2), 10.0)


# ── Daily P&L ─────────────────────────────────────────────────────────────────

def _today_pnl() -> float:
    return sum(
        p["pnl_usdt"] for p in oms.get_today_closed("Europe/Istanbul")
        if p["pnl_usdt"] is not None
    )


def _build_entry_reason(symbol: str, tier: str, direction: str, mtf_score: float,
                        ml_probability: float, threshold: float, regime: str | None) -> str:
    regime_txt = regime or "N/A"
    return (
        f"{direction}_entry|symbol={symbol}|tier={tier}|mtf={mtf_score:.1f}|"
        f"ml={ml_probability:.3f}|threshold={threshold:.3f}|regime={regime_txt}"
    )


def _get_today_entry_count(open_positions: list[dict] | None = None) -> int:
    today = _today_utc_date()
    current_open = oms.get_open_positions() if open_positions is None else open_positions
    open_today = sum(1 for pos in current_open if _iso_to_local_date(pos.get("entry_date")) == today)
    closed_today = sum(
        1 for pos in oms.get_today_closed("Europe/Istanbul")
        if _iso_to_local_date(pos.get("entry_date")) == today
    )
    return open_today + closed_today


def _build_open_risk_snapshot(open_positions: list[dict]) -> dict:
    per_symbol: dict[str, float] = {}
    per_tier: dict[str, float] = {"MAJOR": 0.0, "RISKY": 0.0}
    tier_counts: dict[str, int] = {"MAJOR": 0, "RISKY": 0}
    for pos in open_positions:
        symbol = pos["symbol"]
        tier = get_tier(symbol)
        amount = float(pos.get("amount_usdt", 0.0) or 0.0)
        per_symbol[symbol] = per_symbol.get(symbol, 0.0) + amount
        per_tier[tier] = per_tier.get(tier, 0.0) + amount
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    return {
        "open_count": len(open_positions),
        "per_symbol": per_symbol,
        "per_tier": per_tier,
        "tier_counts": tier_counts,
    }


def _recent_stop_loss_reject(symbol: str) -> tuple[bool, str]:
    cooldown_min = max(0, int(get_trade_policy(symbol).get("stop_loss_cooldown_min", CRYPTO_STOP_LOSS_COOLDOWN_MIN)))
    if cooldown_min <= 0:
        return False, ""
    since_dt = datetime.now(timezone.utc) - timedelta(minutes=cooldown_min)
    recent_closed = oms.get_closed_positions_since(since_dt.isoformat())
    for pos in recent_closed:
        if pos.get("symbol") != symbol:
            continue
        reason = str(pos.get("exit_reason") or "")
        if not reason.startswith("stop"):
            continue
        exit_date = str(pos.get("exit_date") or "")[:19]
        return True, f"cooldown_after_stop_loss:{reason}:{exit_date}"
    return False, ""


def _format_risk_context(open_snapshot: dict, today_trade_count: int, ws_age: float, cur: float | None) -> str:
    per_tier = open_snapshot.get("per_tier", {})
    price_txt = f"{cur:.4f}" if cur is not None else "NA"
    return (
        f"open={open_snapshot.get('open_count', 0)}/{CRYPTO_MAX_POSITIONS}"
        f" daily_trades={today_trade_count}/{CRYPTO_MAX_DAILY_TRADES}"
        f" tier_exp_major={per_tier.get('MAJOR', 0.0):.0f}/{get_max_tier_exposure('MAJOR', CRYPTO_CAPITAL_USDT):.0f}"
        f" tier_exp_risky={per_tier.get('RISKY', 0.0):.0f}/{get_max_tier_exposure('RISKY', CRYPTO_CAPITAL_USDT):.0f}"
        f" ws_age={ws_age if np.isfinite(ws_age) else -1:.1f}s"
        f" price={price_txt}"
    )


def _build_score_components(symbol: str, direction: str, tier: str,
                            mtf_score: float, ml_probability: float,
                            threshold: float, regime_name: str | None,
                            btc_regime: str, funding_rate: float,
                            btc_dominance_regime: str,
                            htf_bias: str) -> dict:
    tier_bonus = 0.0 if tier == "MAJOR" else -0.05
    threshold_edge = max(0.0, ml_probability - threshold)
    regime_bonus_map = {
        ("long", "RISK_ON"): 0.05,
        ("short", "CRASH_RISK"): 0.05,
        ("short", "CHOP"): 0.02,
    }
    regime_bonus = regime_bonus_map.get((direction, regime_name or ""), 0.0)
    btc_penalty = 0.0
    if direction == "long" and btc_regime == "BEAR":
        btc_penalty = -0.08
    elif direction == "short" and btc_regime == "BULL":
        btc_penalty = -0.08
    funding_penalty = 0.0
    if direction == "long" and funding_rate > 0.01:
        funding_penalty = -0.03
    elif direction == "short" and funding_rate < -0.01:
        funding_penalty = -0.03
    htf_bonus = 0.03 if ((direction == "long" and htf_bias == "BULL") or (direction == "short" and htf_bias == "BEAR")) else 0.0
    dom_penalty = -0.03 if (tier == "RISKY" and direction == "long" and btc_dominance_regime == "rising") else 0.0
    score = (
        ml_probability * 0.55 +
        min(float(mtf_score) / 6.0, 1.0) * 0.25 +
        min(threshold_edge / 0.15, 1.0) * 0.10 +
        regime_bonus + htf_bonus + tier_bonus + btc_penalty + funding_penalty + dom_penalty
    )
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "ml_probability": round(float(ml_probability), 4),
        "mtf_score": round(float(mtf_score), 2),
        "threshold": round(float(threshold), 4),
        "regime": regime_name or "N/A",
        "btc_regime": btc_regime,
        "funding_rate": round(float(funding_rate), 6),
        "btc_dominance_regime": btc_dominance_regime,
        "htf_bias": htf_bias,
        "threshold_edge": round(float(threshold_edge), 4),
        "tier_adjustment": round(tier_bonus, 4),
        "btc_penalty": round(btc_penalty, 4),
        "funding_penalty": round(funding_penalty, 4),
        "htf_bonus": round(htf_bonus, 4),
        "dom_penalty": round(dom_penalty, 4),
        "regime_bonus": round(regime_bonus, 4),
    }


def _get_daily_risk_mode(daily_realized_pnl: float, capital_base: float = CRYPTO_CAPITAL_USDT) -> dict[str, object]:
    draw_pct = (daily_realized_pnl / capital_base) if capital_base else 0.0
    if draw_pct <= -0.04:
        return {"mode": "HALT", "threshold_add": 0.08, "size_mult": 0.0, "major_only": True}
    if draw_pct <= -0.03:
        return {"mode": "DEFENSE", "threshold_add": 0.05, "size_mult": 0.60, "major_only": True}
    if draw_pct <= -0.02:
        return {"mode": "WARN", "threshold_add": 0.03, "size_mult": 0.80, "major_only": False}
    return {"mode": "NORMAL", "threshold_add": 0.0, "size_mult": 1.0, "major_only": False}


def _entry_risk_gate(symbol: str, tier: str, direction: str, size: float, cur: float | None,
                     open_snapshot: dict, daily_loss: float, daily_limit: float) -> tuple[bool, str, str]:
    today_trade_count = _get_today_entry_count()
    ws_alive = ws is not None and ws.is_alive()
    ws_age = ws.last_update_age() if ws is not None else float("inf")
    symbol_ws_age = ws.get_price_age(symbol) if ws is not None else float("inf")
    risk_context = _format_risk_context(open_snapshot, today_trade_count, ws_age, cur)

    if CRYPTO_STOP_TRADING_FILE.exists():
        return False, "kill_switch_file", risk_context
    if daily_loss <= daily_limit:
        return False, f"daily_loss_limit:{daily_loss:.2f}<={daily_limit:.2f}", risk_context
    if CRYPTO_REQUIRE_WS_FOR_ENTRIES and not ws_alive:
        return False, "websocket_disconnected", risk_context
    if ws_alive and symbol_ws_age > CRYPTO_STALE_PRICE_GUARD_SEC:
        return False, f"stale_price_guard:{symbol_ws_age:.1f}s", risk_context
    if cur is None or cur <= 0:
        return False, "price_unavailable", risk_context
    if open_snapshot["open_count"] >= CRYPTO_MAX_POSITIONS:
        return False, "max_open_positions", risk_context
    if tier == 'RISKY' and open_snapshot["tier_counts"].get("RISKY", 0) >= CRYPTO_MAX_RISKY_POSITIONS:
        return False, "max_open_positions_tier_risky", risk_context
    if tier == 'MAJOR' and open_snapshot["tier_counts"].get("MAJOR", 0) >= CRYPTO_MAX_MAJOR_POSITIONS:
        return False, "max_open_positions_tier_major", risk_context
    if today_trade_count >= CRYPTO_MAX_DAILY_TRADES:
        return False, "max_daily_trades", risk_context
    cooldown_hit, cooldown_reason = _recent_stop_loss_reject(symbol)
    if cooldown_hit:
        return False, cooldown_reason, risk_context

    projected_coin_exposure = open_snapshot["per_symbol"].get(symbol, 0.0) + size
    max_coin_exposure = get_max_coin_exposure(symbol, CRYPTO_CAPITAL_USDT)
    if projected_coin_exposure > max_coin_exposure:
        return False, f"max_exposure_per_coin:{projected_coin_exposure:.2f}>{max_coin_exposure:.2f}", risk_context

    projected_tier_exposure = open_snapshot["per_tier"].get(tier, 0.0) + size
    max_tier_exposure = get_max_tier_exposure(tier, CRYPTO_CAPITAL_USDT)
    if projected_tier_exposure > max_tier_exposure:
        return False, f"max_exposure_per_tier_{tier.lower()}:{projected_tier_exposure:.2f}>{max_tier_exposure:.2f}", risk_context

    return True, "risk_checks_passed", risk_context


# ── Telegram helpers ──────────────────────────────────────────────────────────

def _tg_open(symbol: str, direction: str, price: float, stop: float,
             target: float, ml_prob: float, mtf: int, size: float) -> None:
    tier = get_tier(symbol)
    side = "LONG" if direction == "long" else "SHORT"
    fg_val = _fg_sentiment.get_fear_greed()["value"]
    stop_pct  = abs(price - stop)   / price * 100
    tgt_pct   = abs(target - price) / price * 100

    send_telegram_alert(
        f"🪙 [CRYPTO][{tier}] {symbol} {side}\n"
        f"📈 Giris: ${price:,.4f} | Size: ${size:.0f}\n"
        f"🛑 Stop: ${stop:,.4f} ({stop_pct:.1f}%) | 🎯 Hedef: ${target:,.4f} (+{tgt_pct:.1f}%)\n"
        f"🤖 ML: {ml_prob:.0%} | 📊 MTF: {mtf}/6 | 😱 F&G: {fg_val}"
    )
    if tier == "RISKY":
        send_telegram_alert(
            f"⚠️ [CRYPTO][RISKI] {symbol} {side}\n"
            f"🎲 Riskli tier — kucuk lot: ${size:.0f}\n"
            f"ML: {ml_prob:.0%} | MTF: {mtf}/6 (esik gecildi)"
        )


def _tg_close(symbol: str, direction: str, pnl: float, pnl_pct: float,
              reason: str, price: float, hold_secs: float) -> None:
    tier    = get_tier(symbol)
    sign    = "+" if pnl >= 0 else ""
    side    = "LONG" if direction == "long" else "SHORT"
    h       = int(hold_secs) // 3600
    m       = (int(hold_secs) % 3600) // 60
    hold_str = f"{h}h{m:02d}m" if h > 0 else f"{m}m"
    total_pnl = oms.get_cumulative_pnl()
    send_telegram_alert(
        f"🪙 [CRYPTO][{tier}] {symbol} {side} KAPATILDI\n"
        f"💰 P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%) | Sure: {hold_str}\n"
        f"📌 Sebep: {reason}\n"
        f"📊 Toplam P&L: {total_pnl:+.2f} USDT"
    )


def _tg_signal_candidate(symbol: str, direction: str, tier: str, price: float,
                         stop: float, target: float, ml_prob: float, mtf: int,
                         size: float, status: str, note: str = "") -> None:
    slot = datetime.now(timezone.utc)
    slot_key = f"{slot.strftime('%Y-%m-%dT%H')}-{slot.minute // 30}"
    dedupe_key = (symbol, direction, status, slot_key)
    if dedupe_key in _signal_telegram_dedupe:
        return
    _signal_telegram_dedupe.add(dedupe_key)
    side = "LONG" if direction == "long" else "SHORT"
    stop_pct = abs(price - stop) / price * 100 if price > 0 else 0.0
    tgt_pct = abs(target - price) / price * 100 if price > 0 else 0.0
    note_line = f"\nNot: {note}" if note else ""
    send_telegram_alert(
        f"📡 <b>[CRYPTO SIGNAL][{tier}] {symbol} {side}</b>\n"
        f"Durum: {status}\n"
        f"Giris: ${price:,.4f} | Size: ${size:.0f}\n"
        f"Stop: ${stop:,.4f} ({stop_pct:.1f}%) | Hedef: ${target:,.4f} ({tgt_pct:.1f}%)\n"
        f"ML: {ml_prob:.0%} | MTF: {mtf}/6{note_line}"
    )


# ── Thread 2: Exit monitor (every 15 seconds) ─────────────────────────────────

def _exit_loop() -> None:
    """Check stops/targets every 15 seconds using WS live price."""
    while not _stop_event.is_set():
        try:
            _run_exits()
        except Exception as exc:
            print(f"[CRYPTO ERROR] exit_loop: {exc}")
            algo_log.system(f"[CRYPTO] exit_loop error: {exc}")
        _stop_event.wait(EXIT_INTERVAL)


def _run_exits() -> None:
    """Single pass: check every open position for exit conditions."""
    if _DRY_RUN_ONCE:
        print("[CRYPTO] --once dry-run: exit checks skipped; open positions untouched")
        return
    positions   = oms.get_open_positions()

    # Rebuild WS-callback index so _ws_price_callback never needs a DB call.
    new_index: dict[str, list[tuple[int, str]]] = {}
    for pos in positions:
        new_index.setdefault(pos["symbol"], []).append((pos["id"], pos["direction"]))
    with _hwm_lock:
        _symbol_pos_index.clear()
        _symbol_pos_index.update(new_index)
        for pos in positions:
            _pos_hwm.setdefault(int(pos["id"]), float(pos["entry_price"]))

    for pos in positions:
        symbol    = pos["symbol"]
        pos_id    = pos["id"]
        direction = pos["direction"]
        stop      = pos["stop_price"]
        tgt       = pos["target_price"]
        tier      = get_tier(symbol)
        policy    = get_trade_policy(symbol)
        trail_trigger = float(policy["trail_trigger"])
        trail_pct = float(policy["trail_pct"])
        partial_tp_pct = float(policy["partial_take_profit_pct"])
        partial_close_fraction = float(policy["partial_close_fraction"])

        cur = _get_price(symbol)
        if cur is None:
            continue

        unrealised   = oms.calc_pnl(pos, cur)
        profit_pct   = unrealised / pos["amount_usdt"] if pos["amount_usdt"] else 0.0
        trailing_on  = profit_pct >= trail_trigger
        reason = None

        partial_done = int(pos.get("partial_exit_done", 0) or 0) == 1
        if not partial_done and partial_close_fraction > 0 and profit_pct >= partial_tp_pct:
            _close_real_position(pos, cur, "partial_take_profit", close_fraction=partial_close_fraction)
            continue

        # 1. Stop loss
        if reason is None:
            if direction == "long"  and cur <= stop:
                reason = "stop_loss"
            elif direction == "short" and cur >= stop:
                reason = "stop_loss"

        # 2. Target hit
        if reason is None:
            if direction == "long"  and cur >= tgt:
                reason = "target_hit"
            elif direction == "short" and cur <= tgt:
                reason = "target_hit"

        # 3. Trailing stop (activated after CRYPTO_TRAIL_TRIGGER profit)
        if reason is None:
            with _hwm_lock:
                hwm_now = _pos_hwm.get(pos_id, float(pos["entry_price"]))
            if direction == "long" and profit_pct >= trail_trigger:
                if cur <= hwm_now * (1 - trail_pct):
                    reason = "trailing_stop"
            elif direction == "short" and profit_pct >= trail_trigger:
                if cur >= hwm_now * (1 + trail_pct):
                    reason = "trailing_stop"

        # 4. Fear & Greed exit (cached hourly sentiment, after trailing checks)
        mtf_long, mtf_short = compute_mtf(stream, symbol)
        if reason is None:
            try:
                fg_now = _fg_sentiment.get_fear_greed()
                fg_val = int(fg_now.get("value", 50) or 50)
            except Exception:
                fg_val = 50
            if direction == "long" and fg_val >= 85 and mtf_long <= 3:
                reason = "fg_extreme_greed"
            elif direction == "short" and fg_val <= 15 and mtf_short <= 3:
                reason = "fg_extreme_fear"

        # 5. Katman 1 — MTF score flip (piyasa yonu tam tersine dondu)
        # SHORT acikken MTF net LONG sinyali veriyorsa kapat (long>=4, short<=1)
        # LONG acikken MTF net SHORT sinyali veriyorsa kapat (short>=4, long<=1)
        if reason is None:
            if direction == "short" and mtf_long >= 4 and mtf_short <= 1:
                reason = "mtf_flip_long"
            elif direction == "long" and mtf_short >= 4 and mtf_long <= 1:
                reason = "mtf_flip_short"

        # 6. Katman 2 — HTF regime flip (4h trend yapısal olarak tersine dondu)
        # HTF BULL iken SHORT aciksa ve MTF de desteklemiyorsa (short<=2) kapat
        # HTF BEAR iken LONG aciksa ve MTF de desteklemiyorsa (long<=2) kapat
        if reason is None:
            htf = _btc_regime_detail.get("htf_bias", "")
            if direction == "short" and htf == "BULLISH" and mtf_short <= 2:
                reason = "htf_flip_bull"
            elif direction == "long" and htf == "BEARISH" and mtf_long <= 2:
                reason = "htf_flip_bear"

        # Exit check log (every cycle, for every open position)
        crypto_log.log_exit_check(
            symbol, tier, direction,
            pos["entry_price"], cur, unrealised,
            stop, tgt, trailing_on,
        )

        if reason:
            _close_real_position(pos, cur, reason)

    _ensure_daily_major_lock()
    _run_shadow_exits()


# ── Thread 1: Signal scanner (every 5 minutes) ────────────────────────────────

def _signal_loop() -> None:
    """Compute MTF + ML + F&G signals; open positions if criteria met."""
    global _last_scan_ts, _next_scan_ts
    while not _stop_event.is_set():
        try:
            _run_signal_scan()
        except Exception as exc:
            print(f"[CRYPTO ERROR] signal_loop: {exc}")
            algo_log.system(f"[CRYPTO] signal_loop error: {exc}")
        _last_scan_ts = time.time()
        _next_scan_ts = _last_scan_ts + SIGNAL_INTERVAL
        # Write state file so crypto_status.py can show timing
        try:
            state_path = Path(__file__).parent / "logs" / "crypto_scan_state.json"
            state_path.parent.mkdir(exist_ok=True)
            regime = _market_regime_snapshot or {}
            state_path.write_text(json.dumps({
                "last_scan_ts": _last_scan_ts,
                "next_scan_ts": _next_scan_ts,
                "signal_interval": SIGNAL_INTERVAL,
                "market_regime": regime.get("regime"),
                "market_regime_reason": regime.get("reason"),
                "avg_corr": _corr_gate_snapshot.get("avg_corr"),
                "corr_threshold": _corr_gate_snapshot.get("threshold"),
                "corr_market_state": _corr_gate_snapshot.get("market_state"),
                "avg_funding": _funding_snapshot.get("avg_rate"),
                "funding_bias": _funding_snapshot.get("bias"),
            }))
        except Exception:
            pass
        _stop_event.wait(SIGNAL_INTERVAL)


def _run_signal_scan() -> None:
    """Single signal scan pass across all symbols."""
    global _market_regime_snapshot, _relative_strength_snapshot, _btc_regime, _short_gate_snapshot
    global _btc_dominance_snapshot, _daily_risk_mode_snapshot, _corr_gate_snapshot, _funding_snapshot
    _ensure_daily_major_lock()
    open_positions = oms.get_open_positions()
    open_symbols   = {p["symbol"] for p in open_positions}
    open_snapshot  = _build_open_risk_snapshot(open_positions)
    majorafter_state = _load_majorafter_state(_current_equity(open_positions))
    major_lock_active = bool(majorafter_state.get("locked"))
    if major_lock_active:
        open_symbols |= _shadow_open_symbols(majorafter_state)

    daily_loss  = _today_pnl()
    daily_limit = -CRYPTO_DAILY_LOSS_LIMIT * CRYPTO_CAPITAL_USDT
    _daily_risk_mode_snapshot = _get_daily_risk_mode(daily_loss, CRYPTO_CAPITAL_USDT)
    daily_mode = str(_daily_risk_mode_snapshot["mode"])
    if daily_loss <= daily_limit:
        msg = (f"[CRYPTO RISK] Gunluk zarar limiti aktif — yeni entry bloklu | "
               f"pnl={daily_loss:.2f} limit={daily_limit:.2f}")
        print(msg)
        algo_log.risk(msg)
    if daily_mode != "NORMAL":
        msg = (
            f"[CRYPTO RISK MODE] {daily_mode} | pnl={daily_loss:.2f} USDT | "
            f"thr_add={float(_daily_risk_mode_snapshot['threshold_add']):.2f} "
            f"size_mult={float(_daily_risk_mode_snapshot['size_mult']):.2f} "
            f"major_only={bool(_daily_risk_mode_snapshot['major_only'])}"
        )
        print(msg)
        algo_log.risk(msg)

    if open_snapshot["open_count"] >= CRYPTO_MAX_POSITIONS and not _DRY_RUN_ONCE:
        msg = f"[CRYPTO RISK] Maks open position dolu | open={open_snapshot['open_count']}/{CRYPTO_MAX_POSITIONS}"
        print(msg)
        algo_log.risk(msg)

    # BTC crash gate
    btc_ok, _ = _check_btc_gate()

    # Fear & Greed
    fg           = _fg_sentiment.get_fear_greed()
    fg_val       = fg["value"]
    fg_label     = fg["label"]
    # fg_lmod / fg_smod no longer used — F&G now adjusts the integer threshold,
    # not the float score.  Keep fg_val for threshold logic and logging.

    # Preload 1h frames once so market regime and per-symbol scans share the same data snapshot.
    market_frames_1h = _fetch_symbol_1h_frames()
    _market_regime_snapshot = _compute_market_regime(market_frames_1h)
    _relative_strength_snapshot = _build_relative_strength_snapshot(market_frames_1h)
    _corr_gate_snapshot = _compute_dynamic_correlation_gate(market_frames_1h)
    _funding_snapshot = _funding_gate_snapshot()
    _log_market_regime(_market_regime_snapshot)
    crypto_log.scan(
        f"[CORR_GATE] avg_corr={float(_corr_gate_snapshot['avg_corr']):.2f} "
        f"thr={float(_corr_gate_snapshot['threshold']):.2f} "
        f"state={_corr_gate_snapshot['market_state']} samples={int(_corr_gate_snapshot['sample_count'])}"
    )
    crypto_log.scan(
        f"[FUNDING] avg={float(_funding_snapshot['avg_rate']):+.4%} "
        f"bias={_funding_snapshot['bias']} "
        f"long_block={bool(_funding_snapshot['long_block'])} "
        f"short_block={bool(_funding_snapshot['short_block'])}"
    )

    # BTC two-layer regime (structural EMA200 + momentum 1h return)
    btc_regime, btc_detail = compute_btc_regime()
    global _BTC_REGIME_SIZE_MULT, _btc_regime_detail
    prev_regime = _btc_regime
    _btc_regime = btc_regime
    _btc_regime_detail = btc_detail
    _BTC_REGIME_SIZE_MULT = btc_detail.get("size_mult", 1.0)
    # Log regime transitions
    if prev_regime != btc_regime and prev_regime != "NEUTRAL":
        transition_msg = (
            f"[CRYPTO] BTC regime transition: {prev_regime} -> {btc_regime} | "
            f"{btc_detail['structural']} + {btc_detail['momentum']} | "
            f"BTC={btc_detail['btc_close']:.2f} EMA200={btc_detail['ema200']:.2f} "
            f"ret_1h={btc_detail['ret_1h']*100:+.2f}%"
        )
        crypto_log.system(transition_msg)
    caution_note = f" ({btc_detail.get('caution_direction', '?')} allowed, size x0.7)" if btc_regime == "CAUTION" else ""
    print(
        f"[BTC REGIME] {btc_regime}{caution_note} | "
        f"{btc_detail['structural']} + {btc_detail['momentum']} | "
        f"BTC={btc_detail['btc_close']:.2f} EMA200={btc_detail['ema200']:.2f} "
        f"ret_1h={btc_detail['ret_1h']*100:+.2f}%"
    )

    # BTC df for correlation checks / ML context
    df_btc_1h = market_frames_1h.get("BTC/USDT")
    if df_btc_1h is None:
        df_btc_1h = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=CRYPTO_CORR_WINDOW + 5)

    for symbol in CRYPTO_SYMBOLS:
        tier = get_tier(symbol)
        signal_ts = datetime.now(timezone.utc).isoformat()
        if symbol in open_symbols and not _DRY_RUN_ONCE:
            signal_id = make_signal_id(symbol, "none", signal_ts)
            risk_context = _format_risk_context(
                open_snapshot,
                _get_today_entry_count(open_positions),
                ws.last_update_age() if ws is not None else float("inf"),
                None,
            )
            record_signal(
                signal_id, signal_ts, symbol, None, tier, None, None, None,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "already_open_position",
            )
            crypto_log.log_signal_decision(symbol, tier, None, False, "already_open_position",
                                           risk_context=risk_context, signal_id=signal_id)
            continue

        # MTF scores (4h / 1h / 15m)
        mtf_long, mtf_short = compute_mtf(stream, symbol)
        rs_long_bonus, rs_short_bonus, rs_note = _relative_strength_mtf_adjustment(symbol)
        mtf_long += rs_long_bonus
        mtf_short += rs_short_bonus

        # 1h candles for ATR sizing + ML base
        df_1h = market_frames_1h.get(symbol)
        if df_1h is None or len(df_1h) < 51:
            print(f"[CRYPTO] {symbol}: yetersiz 1h veri, atlanıyor")
            signal_id = make_signal_id(symbol, "none", signal_ts)
            record_signal(
                signal_id, signal_ts, symbol, None, tier, None, None, None,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "insufficient_1h_data",
            )
            crypto_log.log_signal_decision(symbol, tier, None, False, "insufficient_1h_data", signal_id=signal_id)
            continue

        # 5m candles for ML features
        df_5m = stream.get_ohlcv(symbol, timeframe="5m", limit=60)

        # Live price from WS or REST
        cur  = _get_price(symbol) or float(df_1h["close"].iloc[-1])
        rsi  = float(df_1h["rsi14"].iloc[-1]) if "rsi14" in df_1h.columns else 50.0
        close_1h = float(df_1h["close"].iloc[-1])
        ema21_1h = float(df_1h["ema21"].iloc[-1]) if "ema21" in df_1h.columns and pd.notna(df_1h["ema21"].iloc[-1]) else close_1h

        # ML predictions — use separate directional models when available
        long_ml_on = _has_runtime_model(symbol, "long")
        short_ml_on = _has_runtime_model(symbol, "short")
        ml_on = long_ml_on or short_ml_on
        ml_inputs = _prepare_ml_inputs(symbol, df_1h, df_5m, df_btc_1h)
        _, _, _, funding_info, btc_dom_info, htf_info = ml_inputs
        _btc_dominance_snapshot = btc_dom_info
        prob_long  = _predict_proba(symbol, df_1h, df_5m=df_5m, btc_df=df_btc_1h, side="long", ml_inputs=ml_inputs)
        prob_short = _predict_proba(symbol, df_1h, df_5m=df_5m, btc_df=df_btc_1h, side="short", ml_inputs=ml_inputs)
        ml_prob = prob_long

        regime_mtf_add = int(_market_regime_snapshot.get("trend_mtf_threshold_add", 0)) if _market_regime_snapshot else 0
        regime_ml_add = float(_market_regime_snapshot.get("trend_ml_threshold_add", 0.0)) if _market_regime_snapshot else 0.0
        thresh    = min(6, get_mtf_threshold(symbol, long_ml_on) + regime_mtf_add)
        ml_thresh = min(0.99, get_ml_threshold(symbol) + regime_ml_add + float(_daily_risk_mode_snapshot["threshold_add"]))
        # LONG: BTC NEUTRAL regime requires 5% higher ML confidence
        if btc_regime == "NEUTRAL" and long_ml_on:
            ml_thresh = min(0.99, ml_thresh * 1.05)
        # SHORT gets its own calibrated threshold from SHORT model's threshold_config
        short_ml_thresh = min(0.99, get_short_ml_threshold(symbol) + regime_ml_add + float(_daily_risk_mode_snapshot["threshold_add"]))
        if btc_regime == "NEUTRAL" and short_ml_on:
            short_ml_thresh = min(0.99, short_ml_thresh * 1.05)
        using_directional = _long_model_active and _short_model_active
        no_trade = False

        # F&G threshold adjustment — lowers/raises the INTEGER MTF bar so the
        # gate actually changes.  A float score bump (e.g. +0.5) vs an int
        # threshold never moved the needle; this does.
        if fg_val <= 24:        # Extreme Fear  → easier long, harder short
            long_thresh  = max(1, thresh - 1)
            short_thresh = min(6, thresh + 1)
        elif fg_val >= 75:      # Extreme Greed → harder long, easier short
            long_thresh  = min(6, thresh + 1)
            short_thresh = max(1, thresh - 1)
        else:                   # Fear / Neutral / Greed → no change
            long_thresh  = thresh
            short_thresh = thresh

        short_rule_based = False
        short_rule_regime_ok = btc_regime == "BEAR" or (
            btc_regime == "CAUTION" and btc_detail.get("caution_direction") == "short"
        )
        short_rule_rsi_ok = rsi > 65
        short_rule_ema_ok = close_1h > ema21_1h
        short_rule_fg_ok = fg_val >= 60
        # Rule-based MTF threshold is +1 higher (more conservative without ML).
        # ML mode uses the same MTF threshold as LONG.
        short_rule_mtf_thresh = max(short_thresh, CRYPTO_MTF_THRESHOLD_RB + 1) if short_rule_based else short_thresh
        # BTC regime short permission: block in BULL, block in CAUTION(long-only)
        _btc_allows_short = btc_regime not in ("BULL",) and not (
            btc_regime == "CAUTION" and btc_detail.get("caution_direction") == "long"
        )
        if short_ml_on:
            _short_gate_snapshot = {
                "symbol": symbol,
                "model": "ML",
                "thr": f"{short_ml_thresh:.2f}",
                "MTF": "PASS" if mtf_short >= short_rule_mtf_thresh else "BLOCK",
                "BTC!BULL": "PASS" if _btc_allows_short else "BLOCK",
                "F&G<=74": "PASS" if fg_val <= 74 else "BLOCK",
            }
        else:
            _short_gate_snapshot = {
                "symbol": symbol,
                "model": "RULE-BASED",
                "rsi": "PASS" if short_rule_rsi_ok else "BLOCK",
                "ema21": "PASS" if short_rule_ema_ok else "BLOCK",
                "fg": "PASS" if short_rule_fg_ok else "BLOCK",
                "regime": "PASS" if short_rule_regime_ok else "BLOCK",
            }

        funding_long_block = bool(_funding_snapshot.get("long_block"))
        funding_short_block = bool(_funding_snapshot.get("short_block"))
        long_signal_ready = mtf_long >= long_thresh and (not long_ml_on or prob_long >= ml_thresh)
        long_blocked_no_model = not long_ml_on and long_signal_ready
        long_ok = long_ml_on and long_signal_ready

        # SHORT signal gate — two modes:
        #
        # ML mode (short_ml_on=True):
        #   MTF threshold + ML prob >= short_ml_thresh + BTC not BULL
        #   RSI/EMA/F&G filters are dropped — the ML model already learned these
        #   as features (rsi14, ema8_21_cross, bb_position, ret_24h, etc.).
        #   Requiring RSI>65 on top of ML would double-filter and miss setups
        #   where the model sees a short opportunity the rule doesn't.
        #
        # Rule-based mode (short_ml_on=False):
        #   All strict conditions required (BTC BEAR + RSI>65 + EMA21 + F&G>=60).
        if short_ml_on:
            short_signal_ready = (
                mtf_short >= short_rule_mtf_thresh and
                _btc_allows_short and              # never short in BULL or CAUTION(long)
                prob_short >= short_ml_thresh and  # SHORT model's own calibrated threshold
                fg_val <= 74                       # skip extreme greed (> 74) — unclear regime
            )
        else:
            short_signal_ready = False
        short_disabled_by_policy = PAPER_LONG_ONLY
        # short_ok: True only if signal passes all gates AND policy permits
        # (PAPER_LONG_ONLY=True forces this False; when SHORT is eventually activated,
        #  this correctly reflects the gate state before regime filtering below)
        short_ok = short_signal_ready and not short_disabled_by_policy

        if funding_long_block and long_ok:
            long_ok = False
        if funding_short_block and short_ok:
            short_ok = False

        # BTC two-layer regime direction filter (applied after ML/MTF gates)
        btc_regime_block_long  = False
        btc_regime_block_short = False
        if btc_regime == "BULL":
            if short_ok:
                btc_regime_block_short = True
                short_ok = False   # never short in BULL regime
        elif btc_regime == "BEAR":
            if long_ok:
                btc_regime_block_long = True
                long_ok = False    # never long in BEAR regime
        elif btc_regime == "CAUTION":
            caution_dir = btc_detail.get("caution_direction")
            if caution_dir == "long" and short_ok:
                btc_regime_block_short = True
                short_ok = False   # CAUTION(long): only long allowed
            elif caution_dir == "short" and long_ok:
                btc_regime_block_long = True
                long_ok = False    # CAUTION(short): only short allowed

        blocked_by_crash_regime = False
        if _market_regime_snapshot and _market_regime_snapshot.get("regime") == "CRASH_RISK":
            if _market_regime_snapshot.get("block_new_longs"):
                blocked_by_crash_regime = bool(
                    mtf_long >= long_thresh and (not ml_on or prob_long >= ml_thresh)
                )
                long_ok = False
                long_blocked_no_model = False

        ws_tag = f"WS={cur:.2f}" if (ws and ws.is_alive()) else f"REST={cur:.2f}"
        model_tag = (
            "DIR" if using_directional else
            "LONG_ONLY" if _long_model_active else
            "SCAN_ONLY"
        )
        if short_ml_on:
            short_gate_note = (
                f"SHORT_GATE ML thr={short_ml_thresh:.2f} "
                f"MTF={'Y' if mtf_short>=short_rule_mtf_thresh else 'N'} "
                f"BTC!BULL={'Y' if _btc_allows_short else 'N'} "
                f"F&G<=74={'Y' if fg_val<=74 else 'N'}"
            )
        else:
            short_gate_note = (
                f"SHORT_GATE model={'DISABLED' if short_disabled_by_policy else 'RB'} "
                f"RSI>65={'Y' if short_rule_rsi_ok else 'N'} "
                f"EMA21={'Y' if short_rule_ema_ok else 'N'} "
                f"F&G>=60={'Y' if short_rule_fg_ok else 'N'} "
                f"REGIME={'Y' if short_rule_regime_ok else 'N'}"
        )
        print(f"[CRYPTO SIGNAL] [{tier}] {symbol} | {ws_tag} | RSI={rsi:.1f} | "
              f"MTF L/S={mtf_long}/{mtf_short} | "
              f"ML({model_tag}) L={prob_long:.2f} S={prob_short:.2f} thr={ml_thresh:.2f} | "
              f"MTF thr L={long_thresh} S={short_rule_mtf_thresh} | BTC={btc_regime} | RS={rs_note} | "
              f"Regime={_market_regime_snapshot.get('regime') if _market_regime_snapshot else 'N/A'} | "
              f"F&G={fg_val}({fg_label[:4]}) | "
              f"fund={funding_info.get('funding_rate', 0.0):+.4f} | "
              f"BTC.D={btc_dom_info.get('btc_dominance', 0.0):.1f}({btc_dom_info.get('btc_dominance_regime', 'neutral')}) | "
              f"HTF={htf_info.get('htf_bias', 'BEAR')} | {short_gate_note}")

        # Scan log (every symbol, every scan cycle)
        verdict = (
            "LONG blocked by CRASH_RISK" if blocked_by_crash_regime else
            "LONG blocked by funding sentiment" if funding_long_block and long_signal_ready else
            "LONG blocked by no active model" if long_blocked_no_model else
            "SHORT blocked by funding sentiment" if funding_short_block and short_signal_ready else
            "SHORT disabled by paper-long-only" if short_signal_ready and short_disabled_by_policy else
            "SHORT blocked by BTC BULL regime" if btc_regime_block_short else
            "LONG blocked by BTC BEAR regime" if btc_regime_block_long else
            "LONG candidate" if long_ok else
            "SHORT candidate" if short_ok else
            "NO_TRADE_ZONE" if no_trade else "WAIT"
        )
        signal_side = "long" if long_ok else "short" if short_ok else None
        signal_prob = prob_long if signal_side != "short" else prob_short
        signal_score = mtf_long if signal_side != "short" else mtf_short
        signal_threshold = ml_thresh if signal_side != "short" else short_ml_thresh
        signal_score_payload = None
        if signal_side:
            signal_score_payload = _build_score_components(
                symbol=symbol,
                direction=signal_side,
                tier=tier,
                mtf_score=signal_score,
                ml_probability=signal_prob,
                threshold=signal_threshold,
                regime_name=(_market_regime_snapshot or {}).get("regime"),
                btc_regime=btc_regime,
                funding_rate=float(funding_info.get("funding_rate", 0.0) or 0.0),
                btc_dominance_regime=str(btc_dom_info.get("btc_dominance_regime", "neutral")),
                htf_bias=str(htf_info.get("htf_bias", "BEAR")).upper(),
            )
        signal_id = make_signal_id(symbol, signal_side or "none", signal_ts)
        crypto_log.log_scan(symbol, tier, ml_prob, mtf_long, mtf_short,
                            long_thresh, short_thresh, fg_val, fg_label[:4],
                            ml_thresh, thresh, verdict)
        if _DRY_RUN_ONCE:
            dry_reason = (
                "dry_run_signal_long" if long_ok else
                "dry_run_signal_short" if short_ok else
                "dry_run_crash_risk_block" if blocked_by_crash_regime else
                "dry_run_no_active_long_model" if long_blocked_no_model else
                "dry_run_short_disabled_policy" if short_signal_ready and short_disabled_by_policy else
                "dry_run_btc_bull_block_short" if btc_regime_block_short else
                "dry_run_btc_bear_block_long" if btc_regime_block_long else
                "dry_run_no_trade_zone" if no_trade else
                "dry_run_wait"
            )
            record_signal(
                signal_id, signal_ts, symbol, signal_side, tier, signal_score, signal_prob,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", dry_reason,
            )
            crypto_log.log_signal_decision(symbol, tier, signal_side, False, dry_reason, signal_id=signal_id)
            continue

        direction = None
        entry_mtf = 0
        entry_ml = 0.0
        if btc_regime_block_short:
            record_signal(
                signal_id, signal_ts, symbol, "short", tier, mtf_short, prob_short,
                None, None, short_ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "btc_bull_regime_block_short",
            )
            crypto_log.log_signal_decision(symbol, tier, "short", False, "btc_bull_regime_block_short", signal_id=signal_id)
            continue
        if btc_regime_block_long:
            record_signal(
                signal_id, signal_ts, symbol, "long", tier, mtf_long, prob_long,
                None, None, ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "btc_bear_regime_block_long",
            )
            crypto_log.log_signal_decision(symbol, tier, "long", False, "btc_bear_regime_block_long", signal_id=signal_id)
            continue
        if blocked_by_crash_regime:
            record_signal(
                signal_id, signal_ts, symbol, "long", tier, mtf_long, prob_long,
                None, None, ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "crash_risk_block_long",
            )
            crypto_log.log_signal_decision(symbol, tier, "long", False, "crash_risk_block_long", signal_id=signal_id)
            continue
        if funding_long_block and long_signal_ready:
            record_signal(
                signal_id, signal_ts, symbol, "long", tier, mtf_long, prob_long,
                None, None, ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "funding_overleveraged_long_block",
            )
            crypto_log.log_signal_decision(symbol, tier, "long", False, "funding_overleveraged_long_block", signal_id=signal_id)
            continue
        if long_blocked_no_model:
            record_signal(
                signal_id, signal_ts, symbol, "long", tier, mtf_long, prob_long,
                None, None, ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "no_active_long_model",
            )
            crypto_log.log_signal_decision(symbol, tier, "long", False, "no_active_long_model", signal_id=signal_id)
            continue
        if short_signal_ready and short_disabled_by_policy:
            record_signal(
                signal_id, signal_ts, symbol, "short", tier, mtf_short, prob_short,
                None, None, short_ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "paper_long_only_short_disabled",
            )
            crypto_log.log_signal_decision(symbol, tier, "short", False, "paper_long_only_short_disabled", signal_id=signal_id)
            continue
        if funding_short_block and short_signal_ready:
            record_signal(
                signal_id, signal_ts, symbol, "short", tier, mtf_short, prob_short,
                None, None, short_ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "funding_overleveraged_short_block",
            )
            crypto_log.log_signal_decision(symbol, tier, "short", False, "funding_overleveraged_short_block", signal_id=signal_id)
            continue
        if no_trade:
            record_signal(
                signal_id, signal_ts, symbol, None, tier, None, ml_prob,
                None, None, ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "no_trade_zone",
            )
            crypto_log.log_signal_decision(symbol, tier, None, False, "no_trade_zone", signal_id=signal_id)
            continue
        if long_ok:
            direction = "long"
            entry_mtf = mtf_long
            entry_ml = prob_long
        elif short_ok:
            direction = "short"
            entry_mtf = mtf_short
            entry_ml = prob_short
        else:
            record_signal(
                signal_id, signal_ts, symbol, None, tier, None, ml_prob,
                None, None, ml_thresh,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "signal_not_ready",
            )
            crypto_log.log_signal_decision(symbol, tier, None, False, "signal_not_ready", signal_id=signal_id)
            continue
        if daily_mode == "HALT":
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "daily_risk_halt",
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "daily_risk_halt", signal_id=signal_id)
            continue
        if bool(_daily_risk_mode_snapshot["major_only"]) and tier == "RISKY":
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "daily_risk_major_only",
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "daily_risk_major_only", signal_id=signal_id)
            continue

        htf_bias = str(htf_info.get("htf_bias", "BEAR")).upper()
        if direction == "long" and htf_bias == "BEAR":
            print(f"[CRYPTO] HTF_BLOCK LONG {symbol}: 4h price below EMA50")
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "htf_block_long",
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "htf_block_long", signal_id=signal_id)
            continue
        if direction == "short" and htf_bias == "BULL":
            print(f"[CRYPTO] HTF_BLOCK SHORT {symbol}: 4h price above EMA50")
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "htf_block_short",
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "htf_block_short", signal_id=signal_id)
            continue
        if (
            tier == "RISKY" and
            direction == "long" and
            float(btc_dom_info.get("btc_dominance_7d_change", 0.0) or 0.0) > 2.0
        ):
            reject_reason = (
                f"btc_dominance_risky_long_block:"
                f"{float(btc_dom_info.get('btc_dominance_7d_change', 0.0)):.2f}"
            )
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", reject_reason,
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, reject_reason, signal_id=signal_id)
            algo_log.system(
                f"[CRYPTO] BTC_DOM_BLOCK {symbol} LONG | "
                f"btc_d_7d_change={float(btc_dom_info.get('btc_dominance_7d_change', 0.0)):.2f}"
            )
            continue

        if not btc_ok:
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "btc_crash_gate",
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "btc_crash_gate", signal_id=signal_id)
            algo_log.risk(f"[CRYPTO RISK] {symbol} {direction} rejected | reason=btc_crash_gate")
            continue
        corr_ok, _corr_tag = _check_btc_correlation(symbol, df_1h, direction)
        if not corr_ok:
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                (_market_regime_snapshot or {}).get("regime"),
                "rejected", "btc_correlation_gate",
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "btc_correlation_gate", signal_id=signal_id)
            continue

        # ── EMA21 trend filter ────────────────────────────────────────────────
        if "ema21" in df_1h.columns and pd.notna(df_1h["ema21"].iloc[-1]):
            trend_ok = (
                (direction == "long"  and close_1h > ema21_1h) or
                (direction == "short" and (
                    (short_ml_on and close_1h < ema21_1h) or
                    (not short_ml_on and close_1h > ema21_1h)
                ))
            )
            if not trend_ok:
                print(f"[CRYPTO] TREND_FILTER blocked {symbol} {direction} | "
                      f"close={close_1h:.4f} ema21={ema21_1h:.4f}")
                record_signal(
                    signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                    (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                    (_market_regime_snapshot or {}).get("regime"),
                    "rejected", "trend_filter_ema21",
                )
                crypto_log.log_signal_decision(symbol, tier, direction, False, "trend_filter_ema21", signal_id=signal_id)
                continue

        capital_base = float(majorafter_state.get("locked_equity") or CRYPTO_CAPITAL_USDT) if major_lock_active else CRYPTO_CAPITAL_USDT
        max_sz = get_max_size(symbol, capital_base)
        size = _atr_position_size(df_1h, cur, fg_val, max_sz, capital_base=capital_base)
        size = round(max(10.0, size * float(_daily_risk_mode_snapshot["size_mult"])), 2)
        if direction == "long":
            size = round(max(10.0, size * float(_funding_snapshot.get("long_size_mult", 1.0))), 2)
        else:
            size = round(max(10.0, size * float(_funding_snapshot.get("short_size_mult", 1.0))), 2)
        if direction == "long" and _market_regime_snapshot and _market_regime_snapshot.get("regime") == "CRASH_RISK":
            size = round(max(10.0, size * float(_market_regime_snapshot.get("size_mult", 1.0))), 2)
        # BTC regime CAUTION size reduction (x0.7)
        if _BTC_REGIME_SIZE_MULT < 1.0:
            size = round(max(10.0, size * _BTC_REGIME_SIZE_MULT), 2)

        regime_name = (_market_regime_snapshot or {}).get("regime")
        entry_reason = _build_entry_reason(symbol, tier, direction, entry_mtf, entry_ml, ml_thresh, regime_name)

        if major_lock_active:
            shadow_ok, shadow_reason = _shadow_entry_gate(symbol, tier, size, cur, majorafter_state)
            stop_shadow = float(cur * (1 - float(get_trade_policy(symbol)["stop_pct"]))) if direction == "long" else float(cur * (1 + float(get_trade_policy(symbol)["stop_pct"])))
            target_shadow = float(cur * (1 + float(get_trade_policy(symbol)["target_pct"]))) if direction == "long" else float(cur * (1 - float(get_trade_policy(symbol)["target_pct"])))
            if not shadow_ok:
                record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                regime_name, "rejected", shadow_reason, entry_reason=entry_reason,
            )
                crypto_log.log_signal_decision(symbol, tier, direction, False, shadow_reason, size=size,
                                               signal_id=signal_id)
                _tg_signal_candidate(symbol, direction, tier, cur, stop_shadow, target_shadow, entry_ml, entry_mtf, size, "BLOKE", shadow_reason)
                continue
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                regime_name, "rejected", "daily_major_lock_active", entry_reason=entry_reason,
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, "daily_major_lock_active_shadow", size=size,
                                           signal_id=signal_id)
            _tg_signal_candidate(symbol, direction, tier, cur, stop_shadow, target_shadow, entry_ml, entry_mtf, size, "SHADOW", "gunluk hedef sonrasi arastirma modu")
            _open_shadow_position(
                majorafter_state, symbol, tier, direction, cur, size,
                signal_id, entry_reason, entry_mtf, entry_ml,
            )
            open_symbols.add(symbol)
            continue

        risk_ok, risk_reason, risk_context = _entry_risk_gate(
            symbol, tier, direction, size, cur, open_snapshot, daily_loss, daily_limit
        )
        if not risk_ok:
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                regime_name, "rejected", risk_reason, entry_reason=entry_reason,
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, risk_reason, size=size,
                                           risk_context=risk_context, signal_id=signal_id)
            algo_log.risk(f"[CRYPTO RISK] {symbol} {direction} rejected | reason={risk_reason} | {risk_context}")
            stop_preview = float(cur * (1 - float(get_trade_policy(symbol)["stop_pct"]))) if direction == "long" else float(cur * (1 + float(get_trade_policy(symbol)["stop_pct"])))
            target_preview = float(cur * (1 + float(get_trade_policy(symbol)["target_pct"]))) if direction == "long" else float(cur * (1 - float(get_trade_policy(symbol)["target_pct"])))
            _tg_signal_candidate(symbol, direction, tier, cur, stop_preview, target_preview, entry_ml, entry_mtf, size, "BLOKE", risk_reason)
            continue

        stop_pct_cfg = float(get_trade_policy(symbol)["stop_pct"])
        target_pct_cfg = float(get_trade_policy(symbol)["target_pct"])
        result = oms.create_buy(
            symbol, size, cur, stop_pct_cfg, target_pct_cfg, direction,
            signal_id=signal_id, entry_reason=entry_reason,
        )
        if result["status"] != "FILLED":
            reject_reason = f"oms_rejected:{result.get('reason', 'unknown')}"
            record_signal(
                signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
                (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
                regime_name, "rejected", reject_reason, entry_reason=entry_reason,
            )
            crypto_log.log_signal_decision(symbol, tier, direction, False, reject_reason, size=size,
                                           risk_context=risk_context, signal_id=signal_id)
            algo_log.risk(f"[CRYPTO RISK] {symbol} {direction} rejected | reason={reject_reason} | {risk_context}")
            stop_preview = float(cur * (1 - float(get_trade_policy(symbol)["stop_pct"]))) if direction == "long" else float(cur * (1 + float(get_trade_policy(symbol)["stop_pct"])))
            target_preview = float(cur * (1 + float(get_trade_policy(symbol)["target_pct"]))) if direction == "long" else float(cur * (1 - float(get_trade_policy(symbol)["target_pct"])))
            _tg_signal_candidate(symbol, direction, tier, cur, stop_preview, target_preview, entry_ml, entry_mtf, size, "BLOKE", reject_reason)
            continue

        record_signal(
            signal_id, signal_ts, symbol, direction, tier, entry_mtf, entry_ml,
            (signal_score_payload or {}).get("score"), signal_score_payload, signal_threshold,
            regime_name, "accepted", "entry_opened", entry_reason=entry_reason,
        )
        mark_signal_entry(
            signal_id=signal_id,
            position_id=int(result["position_id"]),
            entry_order_id=str(result["order_id"]),
            entry_price=float(cur),
            position_size_usdt=float(size),
            entry_reason=entry_reason,
        )
        crypto_log.log_signal_decision(symbol, tier, direction, True, "entry_opened", size=size,
                                       risk_context=risk_context, signal_id=signal_id)
        with _hwm_lock:
            _pos_hwm[int(result["position_id"])] = float(cur)

        open_symbols.add(symbol)
        open_snapshot["open_count"] += 1
        open_snapshot["tier_counts"][tier] = open_snapshot["tier_counts"].get(tier, 0) + 1
        open_snapshot["per_symbol"][symbol] = open_snapshot["per_symbol"].get(symbol, 0.0) + size
        open_snapshot["per_tier"][tier] = open_snapshot["per_tier"].get(tier, 0.0) + size

        stop_p  = result["stop_price"]
        tgt_p   = result["target_price"]
        atr_val = float(df_1h["atr14"].iloc[-1]) if "atr14" in df_1h.columns else 0.0
        vol_r   = float(df_1h["vol_ratio"].iloc[-1]) if "vol_ratio" in df_1h.columns else 1.0
        if direction == "long":
            stop_pct = (cur - stop_p) / cur * 100
            tgt_pct  = (tgt_p - cur) / cur * 100
            print(f"[CRYPTO ENTRY] [{tier}] {symbol} LONG @ {cur:.4f} | "
                  f"Stop: {stop_p:.4f} | Target: {tgt_p:.4f} | size={size:.0f} USDT")
            algo_log.trade(
                f"[CRYPTO][{tier}] BUY {symbol} @ {cur:.4f} | size={size:.2f} USDT | "
                f"stop={stop_p:.4f} | ml={entry_ml:.2f} | mtf={entry_mtf}/6 | fg={fg_val}"
            )
            crypto_log.log_buy(symbol, tier, "long", cur, size,
                               stop_p, -stop_pct, tgt_p, tgt_pct,
                               entry_ml, entry_mtf, fg_val, fg_label[:4],
                               atr_val, vol_r)
            journal_open(symbol, "long", cur, size, stop_p, tgt_p)
            _tg_open(symbol, "long", cur, stop_p, tgt_p, entry_ml, entry_mtf, size)
            continue

        stop_pct = (stop_p - cur) / cur * 100
        tgt_pct  = (cur - tgt_p) / cur * 100
        print(f"[CRYPTO ENTRY] [{tier}] {symbol} SHORT @ {cur:.4f} | "
              f"Stop: {stop_p:.4f} | Target: {tgt_p:.4f} | size={size:.0f} USDT")
        algo_log.trade(
            f"[CRYPTO][{tier}] SHORT {symbol} @ {cur:.4f} | size={size:.2f} USDT | "
            f"stop={stop_p:.4f} | ml_sell={entry_ml:.2f} | mtf={entry_mtf}/6 | fg={fg_val}"
        )
        crypto_log.log_buy(symbol, tier, "short", cur, size,
                           stop_p, stop_pct, tgt_p, tgt_pct,
                           entry_ml, entry_mtf, fg_val, fg_label[:4],
                           atr_val, vol_r)
        journal_open(symbol, "short", cur, size, stop_p, tgt_p)
        _tg_open(symbol, "short", cur, stop_p, tgt_p, entry_ml, entry_mtf, size)

    _print_gate_summary()


def _print_gate_summary() -> None:
    """Print a one-line summary of all active entry gates after each scan cycle."""
    major_cfg = (_threshold_config or LEGACY_THRESHOLD_CONFIG).get("MAJOR", {})
    risky_cfg = (_threshold_config or LEGACY_THRESHOLD_CONFIG).get("RISKY", {})
    major_thr = major_cfg.get("selected_threshold") or major_cfg.get("fallback_threshold", 0.63)
    risky_thr = risky_cfg.get("selected_threshold") or risky_cfg.get("fallback_threshold", 0.72)
    dyn_thr = float(_corr_gate_snapshot.get("threshold", CRYPTO_CORR_THRESHOLD) or CRYPTO_CORR_THRESHOLD)
    avg_corr = float(_corr_gate_snapshot.get("avg_corr", 0.0) or 0.0)
    avg_funding = float(_funding_snapshot.get("avg_rate", 0.0) or 0.0)

    btc_label = "CRASH" if _btc_1h_return < CRYPTO_BTC_CRASH_THRESHOLD else "PASS"

    # BTC correlation gate status for high-corr alts — probe with LONG direction
    # (shows whether the gate is currently active; direction shown as L/S on block)
    btc_trend = _get_btc_4h_trend()
    corr_parts = []
    for sym in ["ETH/USDT", "SOL/USDT", "BNB/USDT"]:
        tag = sym.split("/")[0]
        # Only fetch 1h OHLCV to compute correlation; reuse what we already know about trend
        df_1h_chk = stream.get_ohlcv(sym, timeframe="1h", limit=CRYPTO_CORR_WINDOW + 2)
        if df_1h_chk is None or len(df_1h_chk) < CRYPTO_CORR_WINDOW:
            corr_parts.append(f"{tag}=PASS(no-data)")
            continue
        df_btc_chk = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=CRYPTO_CORR_WINDOW + 2)
        if df_btc_chk is None or len(df_btc_chk) < CRYPTO_CORR_WINDOW:
            corr_parts.append(f"{tag}=PASS(no-btc)")
            continue
        btc_ret = df_btc_chk["close"].pct_change().dropna().tail(CRYPTO_CORR_WINDOW).values
        alt_ret = df_1h_chk["close"].pct_change().dropna().tail(CRYPTO_CORR_WINDOW).values
        n = min(len(btc_ret), len(alt_ret))
        if n < 10:
            corr_parts.append(f"{tag}=PASS(short)")
            continue
        try:
            corr = float(np.corrcoef(btc_ret[-n:], alt_ret[-n:])[0, 1])
        except Exception:
            corr_parts.append(f"{tag}=PASS(err)")
            continue
        if corr <= dyn_thr:
            corr_parts.append(f"{tag}=PASS(c={corr:.2f})")
        elif btc_trend == "BEARISH":
            corr_parts.append(f"{tag}=BLOCK_L(c={corr:.2f},BEAR)")
        elif btc_trend == "BULLISH":
            corr_parts.append(f"{tag}=BLOCK_S(c={corr:.2f},BULL)")
        else:
            corr_parts.append(f"{tag}=PASS(unk)")

    print(
        f"[CRYPTO GATES] ML MAJOR={major_thr:.2f} | RISKY={risky_thr:.2f} | "
        f"MTF_ML={CRYPTO_MTF_THRESHOLD_ML} | MTF_RB={CRYPTO_MTF_THRESHOLD_RB} | "
        f"RISKY_MTF={RISKY_MTF_THRESHOLD} | TREND=EMA21 | BTC_CRASH={btc_label} | "
        f"CORR avg={avg_corr:.2f}/thr={dyn_thr:.2f} | "
        f"FUND avg={avg_funding:+.4%}/{_funding_snapshot.get('bias', 'neutral')} | "
        f"REGIME={_btc_regime} | "
        f"CORR({btc_trend[:4]}): {' '.join(corr_parts)}"
    )


# ── Thread 3: Health monitor (every 60 seconds) ───────────────────────────────

def _check_model_reload() -> None:
    """
    Reload directional model files if they were updated since last load.
    Called from health loop every _MODEL_RELOAD_INTERVAL seconds.
    Handles the case where crypto_weekly_retrain.py promotes a new model
    while the trader is running — without this, _long_model_active stays stale.
    """
    from pathlib import Path as _Path
    model_paths = [
        _Path(__file__).parent / "models" / "crypto_xgb_long.pkl",
        _Path(__file__).parent / "models" / "crypto_xgb_short.pkl",
        _Path(__file__).parent / "models" / "crypto_xgb_long_major.pkl",
        _Path(__file__).parent / "models" / "crypto_xgb_long_risky.pkl",
        _Path(__file__).parent / "models" / "crypto_xgb_short_major.pkl",
        _Path(__file__).parent / "models" / "crypto_xgb_short_risky.pkl",
        _Path(__file__).parent / "models" / "crypto_xgb.pkl",
    ]
    try:
        newest_mtime = max(p.stat().st_mtime if p.exists() else 0.0 for p in model_paths)
    except Exception:
        return
    if newest_mtime > _model_load_time:
        msg = "[CRYPTO MODEL] Model dosyasi degisti — yeniden yukleniyor..."
        print(msg)
        crypto_log.system(msg)
        _load_crypto_model()


def _health_loop() -> None:
    _last_model_check: float = 0.0
    while not _stop_event.is_set():
        try:
            _print_health_line()
            _maybe_send_hourly_status()
        except Exception as _health_exc:
            try:
                crypto_log.system(f"[CRYPTO HEALTH] health_loop exception: {_health_exc}")
                print(f"[CRYPTO HEALTH] HATA: {_health_exc}")
            except Exception:
                pass
        # Periodic model reload check (every _MODEL_RELOAD_INTERVAL seconds)
        _now = time.time()
        if _now - _last_model_check >= _MODEL_RELOAD_INTERVAL:
            try:
                _check_model_reload()
            except Exception as _reload_exc:
                print(f"[CRYPTO MODEL] Reload check hatasi: {_reload_exc}")
            _last_model_check = _now
        _stop_event.wait(HEALTH_INTERVAL)


def _print_health_line() -> None:
    """Compact one-line health status."""
    positions  = oms.get_open_positions()
    today_pnl  = _today_pnl()
    fg         = _fg_sentiment.get_fear_greed()
    ws_status  = "CANLI" if (ws and ws.is_alive()) else "KOPUK"
    regime     = (_market_regime_snapshot or {}).get("regime", "N/A")
    btc_dom    = _btc_dominance_snapshot or {}

    prices = {}
    if ws and ws.is_alive():
        prices = ws.get_all_prices()

    def _fmt(sym): return f"{sym.split('/')[0]}={prices.get(sym, 0):.0f}"
    price_str = "  ".join(_fmt(s) for s in CRYPTO_SYMBOLS) if prices else "fiyat yok"

    now_ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"[CRYPTO WS]  {ws_status} | {price_str} | "
          f"Pos: {len(positions)}/{CRYPTO_MAX_POSITIONS} | Regime: {regime} | "
          f"BTC: {_btc_regime} | BTC.D: {btc_dom.get('btc_dominance', 0.0):.1f} "
          f"({btc_dom.get('btc_dominance_regime', 'neutral')}) | F&G: {fg['value']} | "
          f"P&L: {today_pnl:+.2f} USDT | Mode: {_daily_risk_mode_snapshot.get('mode', 'NORMAL')} | {now_ts}")
    majorafter_state = _load_majorafter_state()
    if majorafter_state.get("locked"):
        locked_eq = float(majorafter_state.get("locked_equity") or 0.0)
        shadow_closed = majorafter_state.get("shadow_closed_positions", [])
        shadow_total = sum(float(pos.get("pnl_usdt") or 0.0) for pos in shadow_closed)
        print(
            f"[MAJOR TARGET] LOCKED | locked_eq={locked_eq:.2f} | "
            f"shadow_open={len(majorafter_state.get('shadow_open_positions', []))} | "
            f"shadow_closed={len(shadow_closed)} | shadow_pnl={shadow_total:+.2f} USDT"
        )
    short_mode = "DISABLED" if PAPER_LONG_ONLY else ("ACTIVE" if _short_model_active else "RULE-BASED")
    if _short_model_active and _short_gate_snapshot.get("model") == "ML":
        short_gate_str = (
            f"thr={_short_gate_snapshot.get('thr', 'N/A')} | "
            f"MTF={_short_gate_snapshot.get('MTF', 'N/A')} | "
            f"BTC!BULL={_short_gate_snapshot.get('BTC!BULL', 'N/A')} | "
            f"F&G<=74={_short_gate_snapshot.get('F&G<=74', 'N/A')}"
        )
    else:
        short_gate_str = (
            f"RSI>65={_short_gate_snapshot.get('rsi', 'N/A')} | "
            f"EMA21={_short_gate_snapshot.get('ema21', 'N/A')} | "
            f"F&G>=60={_short_gate_snapshot.get('fg', 'N/A')} | "
            f"REGIME={_short_gate_snapshot.get('regime', 'N/A')}"
        )
    print(f"[SHORT] Model: {short_mode}")
    print(f"[SHORT] Gate: {short_gate_str}")


# ── Health check at startup ───────────────────────────────────────────────────

def _print_health_check() -> None:
    global _btc_dominance_snapshot
    fg = _fg_sentiment.get_fear_greed()
    print()
    print(f"[CRYPTO HEALTH] Mod: PAPER | Sermaye: {CRYPTO_CAPITAL_USDT:.0f} USDT")

    # WS status
    ws_alive = ws and ws.is_alive()
    if ws_alive:
        prices = ws.get_all_prices()
        btc_p  = prices.get("BTC/USDT", 0)
        age    = ws.last_update_age()
        print(f"[CRYPTO HEALTH] WebSocket: CANLI | BTC: ${btc_p:,.2f} | "
              f"Son guncelleme: {age:.1f}s önce")
    else:
        ticker = stream.get_ticker("BTC/USDT")
        btc_p  = ticker["last"] if ticker and ticker.get("last") else 0
        print(f"[CRYPTO HEALTH] WebSocket: KOPUK | BTC (REST): ${btc_p:,.2f}")

    print(f"[CRYPTO HEALTH] Fear & Greed: {fg['value']} ({fg['label']})")
    try:
        _btc_dominance_snapshot = get_btc_dominance_features()
    except Exception:
        _btc_dominance_snapshot = _btc_dominance_snapshot or {
            "btc_dominance": 0.0,
            "btc_dominance_regime": "neutral",
            "btc_dominance_7d_change": 0.0,
        }
    print(
        f"[CRYPTO HEALTH] BTC dominance: {_btc_dominance_snapshot.get('btc_dominance', 0.0):.2f} | "
        f"regime={_btc_dominance_snapshot.get('btc_dominance_regime', 'neutral')} | "
        f"7d_change={_btc_dominance_snapshot.get('btc_dominance_7d_change', 0.0):+.2f}"
    )
    major_long = _tier_models_long.get("MAJOR")
    risky_long = _tier_models_long.get("RISKY")
    major_short = _tier_models_short.get("MAJOR")
    risky_short = _tier_models_short.get("RISKY")
    ml_str = (
        f"MAJOR L={major_long.get('wf_precision', 0.0):.1%} S={major_short.get('wf_precision', 0.0):.1%} | "
        f"RISKY L={risky_long.get('wf_precision', 0.0):.1%} S={risky_short.get('wf_precision', 0.0):.1%}"
        if any([major_long, risky_long, major_short, risky_short])
        else "SCAN_ONLY — active tier artifact yok"
    )
    print(f"[CRYPTO HEALTH] ML model: {ml_str}")
    print(f"[SHORT] Model: {'DISABLED' if PAPER_LONG_ONLY else ('ACTIVE' if _short_model_active else 'BLOCKED')}")
    print(
        f"[CRYPTO HEALTH] Funding sentiment: avg={float(_funding_snapshot.get('avg_rate', 0.0) or 0.0):+.4%} | "
        f"bias={_funding_snapshot.get('bias', 'neutral')}"
    )
    print(
        f"[CRYPTO HEALTH] Corr gate: avg={float(_corr_gate_snapshot.get('avg_corr', 0.0) or 0.0):.2f} | "
        f"thr={float(_corr_gate_snapshot.get('threshold', CRYPTO_CORR_THRESHOLD) or CRYPTO_CORR_THRESHOLD):.2f} | "
        f"state={_corr_gate_snapshot.get('market_state', 'normal')}"
    )
    btc_regime_now, btc_detail_now = compute_btc_regime()
    caution_note = f" ({btc_detail_now.get('caution_direction', '?')} allowed, x0.7)" if btc_regime_now == "CAUTION" else ""
    print(f"[BTC REGIME] {btc_regime_now}{caution_note} | "
          f"{btc_detail_now['structural']} + {btc_detail_now['momentum']} | "
          f"BULL=long only | BEAR=short only | CAUTION=one side x0.7 | NEUTRAL=both (ml_thr +5%)")
    if _market_regime_snapshot:
        print(f"[CRYPTO HEALTH] Market regime: {_market_regime_snapshot.get('regime')}")
        print(f"[CRYPTO HEALTH] Regime note : {_market_regime_snapshot.get('reason')}")
    if _threshold_config:
        for tier_name in ["MAJOR", "RISKY"]:
            tier_cfg = (_threshold_config or LEGACY_THRESHOLD_CONFIG).get(tier_name, {})
            shown = tier_cfg.get("selected_threshold")
            if shown is None:
                shown = tier_cfg.get("fallback_threshold")
            short_cfg = (_short_threshold_config or LEGACY_THRESHOLD_CONFIG).get(tier_name, {})
            short_shown = short_cfg.get("selected_threshold") or short_cfg.get("fallback_threshold")
            print(
                f"[CRYPTO HEALTH] {tier_name} LONG thr: {shown} | "
                f"SHORT thr: {short_shown} | "
                f"enabled={tier_cfg.get('enabled', True)}"
            )
    print(f"[CRYPTO HEALTH] Pairs: {', '.join(CRYPTO_SYMBOLS)}")
    print(f"[CRYPTO HEALTH] Sinyal: 5m/15m/1h MTF | Tarama: her 5 dk | Cikis: her 15 sn")
    print("[CRYPTO HEALTH] Spot sistem: BAGIMSIZ | VIOP sistem: BAGIMSIZ")
    open_positions = oms.get_open_positions()
    if open_positions:
        print("[CRYPTO HEALTH] Open position funding:")
        for pos in open_positions:
            funding_info = fetch_funding_rate(pos["symbol"])
            print(
                f"  {pos['symbol']} {pos['direction'].upper()} | "
                f"funding={funding_info.get('funding_rate', 0.0):+.6f} | "
                f"avg3={funding_info.get('funding_rate_avg3', 0.0):+.6f}"
            )
    else:
        print("[CRYPTO HEALTH] Open position funding: none")
    print()

    if fg["value"] <= 24:
        send_telegram_alert(f"😱 [CRYPTO] Extreme Fear={fg['value']} — LONG bias aktif")
    elif fg["value"] >= 75:
        send_telegram_alert(f"😱 [CRYPTO] Extreme Greed={fg['value']} — SHORT bias aktif")


# ── HYPE availability check ───────────────────────────────────────────────────

def _check_hype_availability() -> None:
    global CRYPTO_MAJOR, CRYPTO_SYMBOLS
    if 'HYPE/USDT' not in CRYPTO_MAJOR:
        return
    try:
        if not check_symbol_on_exchange('HYPE/USDT'):
            msg = "[CRYPTO] HYPE/USDT mevcut degil — atlanıyor"
            print(msg)
            algo_log.system(msg)
            CRYPTO_MAJOR = [s for s in CRYPTO_MAJOR if s != 'HYPE/USDT']
            CRYPTO_SYMBOLS = CRYPTO_MAJOR + CRYPTO_RISKY
    except Exception as exc:
        print(f"[CRYPTO] HYPE check hatası: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(once: bool = False) -> None:
    global ws, _DRY_RUN_ONCE
    _DRY_RUN_ONCE = once
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 60)
    print(f"CRYPTO TRADER v3  |  python3.12 crypto_trader.py")
    print(f"Baslangic: {now}")
    print(f"Mod: PAPER | Sermaye: {CRYPTO_CAPITAL_USDT:.0f} USDT")
    print(f"Universe: {len(CRYPTO_MAJOR)} major + {len(CRYPTO_RISKY)} risky = {len(CRYPTO_SYMBOLS)} pairs")
    print(f"Mimari: WebSocket + 3 thread (signal/exit/health)")
    print("=" * 60)

    algo_log.system("[CRYPTO] crypto_trader.py v3 basladi")
    crypto_log.system(f"[CRYPTO] crypto_trader.py v3 basladi | {now} | {CRYPTO_CAPITAL_USDT:.0f} USDT")

    # Load ML model
    _load_crypto_model()

    # Verify HYPE/USDT is listed; remove from universe if not
    print("[CRYPTO] HYPE/USDT varlik kontrolu yapılıyor...")
    _check_hype_availability()
    crypto_log.system(f"[CRYPTO] Universe: {len(CRYPTO_MAJOR)} major + {len(CRYPTO_RISKY)} risky = {len(CRYPTO_SYMBOLS)} pairs")

    # Start WebSocket
    ws = CryptoWebSocket()
    ws.set_price_callback(_ws_price_callback)
    ws.start()
    print("[CRYPTO] WebSocket basladi, baglanti bekleniyor (3s)...")
    time.sleep(3)

    _print_health_check()

    # Startup reconciliation
    try:
        from crypto_reconciliation import crypto_recon
        crypto_recon.run()
    except Exception as _recon_exc:
        print(f"[CRYPTO] Reconciliation hatasi (atlanıyor): {_recon_exc}")

    _load_majorafter_state(_current_equity())

    if once:
        # --once mode: one signal scan + one exit check, then exit
        print("[CRYPTO] --once modu: tek tarama çalıştırılıyor...")
        _run_signal_scan()
        _run_exits()
        _print_health_line()
        ws.stop()
        print("[CRYPTO] --once modu — cikiliyor.")
        return

    # Production: launch 3 daemon threads
    threads = [
        threading.Thread(target=_signal_loop, daemon=True, name="CryptoSignal"),
        threading.Thread(target=_exit_loop,   daemon=True, name="CryptoExit"),
        threading.Thread(target=_health_loop, daemon=True, name="CryptoHealth"),
    ]
    for t in threads:
        t.start()

    print(f"[CRYPTO] 3 thread aktif: Signal(5dk), Exit(15sn), Health(60sn)")
    print("[CRYPTO] Ctrl+C ile durdurun\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[CRYPTO] KeyboardInterrupt — durduruluyor...")
        _stop_event.set()
        ws.stop()
        algo_log.system("[CRYPTO] crypto_trader.py kapatildi")
        print("[CRYPTO] Temiz kapanis tamamlandi.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Trader v3")
    parser.add_argument("--once", action="store_true",
                        help="WS connect + 1 signal scan + 1 exit check, then exit")
    args = parser.parse_args()
    main(once=args.once)
