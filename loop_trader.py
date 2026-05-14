"""
loop_trader.py -- BIST Loop Trader v2
---------------------------------------
Run : py -3.12 loop_trader.py
Test: py -3.12 loop_trader.py --once   (single iteration, no sleep)

Core loop (every 60 seconds, 10:00-18:00 Istanbul):
  - Skip signal generation 10:00-10:30 (opening volatility)
  - Run ensemble model -> BUY signals (prob > 0.65, max 3 open positions)
  - Position size: 20% of current equity
  - Every loop: check stop/target on open positions via live prices
  - 18:30: append daily_summary.csv + Telegram

trade_log.csv    -> one row per CLOSED trade
daily_summary.csv -> one row per day at 18:30
"""

import os, sys, csv, json, time, sqlite3, subprocess
from datetime import datetime, date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pytz
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# -- Project imports ----------------------------------------------------------
from paper_trade import (
    get_db, get_open_positions, get_cash, get_latest,
    load_model, get_ml_signals, calc_position_pnl,
    open_position, close_position, apply_fill_price, fetch_live_prices,
    make_features_single, calibrated_buy_prob, classify_prob,
    BUY_THRESHOLD_DEFAULT, SELL_THRESHOLD_DEFAULT,
    SYMBOLS,
)
from telegram_bot import send_telegram_alert
from sentiment import score_news, read_recent_news
from kill_switch import KillSwitch, get_equity_peak, get_current_equity_ks
from algolab_stream import stream as algolab
from oms import oms, OrderStatus
from reconciliation import recon
from logger import algo_log
from portfolio_risk import PortfolioRisk, MAX_PORTFOLIO_HEAT
from bist_data_quality import check_entry_price, format_data_quality_block
from volatility_regime import (
    VolatilityRegime,
    HIGH_VOL_THRESHOLD,
    EXTREME_VOL_THRESHOLD,
)
from regime_hmm import (
    DB_PATH as HMM_DB_PATH,
    HMM_MODEL_PATH,
    HMM_CONFIRMATION_DAYS,
    build_live_regime_feature_frame,
    get_current_regime_from_artifact,
    load_hmm_model,
    train_and_save_hmm_model,
)
from meta_labeler import (
    META_MIN_CLOSED_TRADES,
    META_MODEL_PATH,
    META_THRESHOLD,
    build_live_meta_feature_vector,
    compute_microstructure_features,
    load_meta_model,
    predict_meta_probability,
)
from bist_config import (
    ATR_TARGET_MULT,
    BUY_THRESHOLD,
    DAILY_STOP_PCT,
    DISPLAY_THRESHOLD,
    DRAWDOWN_HALT_PCT,
    DRAWDOWN_KILL_PCT,
    DRAWDOWN_WARN_PCT,
    MAX_PER_SECTOR,
    MAX_POS,
    MAX_POSITION_SIZE,
    MAX_SHORT_SIZE,
    MIN_CASH_RESERVE,
    MODEL_FLIP_THRESHOLD,
    RISK_PER_TRADE,
    SHORT_SL_PCT,
    SHORT_THRESHOLD,
    SL_PCT,
    TOTAL_CAPITAL,
    TRADEABLE_CAPITAL,
    TRAIL_BE_PCT,
    TRAIL_DIST_PCT,
    TRAIL_START_PCT,
)

# Global instances — shared across all calls in this process
_portfolio_risk  = PortfolioRisk()
_vol_regime      = VolatilityRegime()
_beta_cal        = None   # BetaCalibration loaded at startup; None = pass-through
_hmm_artifact    = None
_hmm_regime_info = None
_meta_labeler_artifact = None
_meta_labeling_active = False
_meta_closed_trades = 0

# -- Constants ----------------------------------------------------------------
TZ             = pytz.timezone("Europe/Istanbul")

# Capital allocation and core risk constants are imported from bist_config.py.
CAPITAL          = TOTAL_CAPITAL  # alias used in daily summary % calc

# SHORT selling parameters
RSI_SHORT_MIN    = 60     # RSI must be > 60 to open SHORT (overbought)
DEDUP_HOURS          = 2       # same-symbol cooldown after close/open (hours)
HMM_RETRAIN_DAY      = 1
META_LABELING_ENABLED = False

# Sector diversification — local map with Turkish names
SECTOR_MAP = {
    "YKBNK": "banka",       "AKBNK": "banka",
    "ISCTR": "banka",       "GARAN": "banka",
    "TUPRS": "enerji",      "PETKM": "enerji",
    "TAVHL": "havacilik",   "FROTO": "otomotiv",
    "TCELL": "telekom",     "ASELS": "savunma",
    "BIMAS": "perakende",   "MGROS": "perakende",
    "ENKAI": "insaat",      "EKGYO": "insaat",
    # Expanded universe (added 2026-04-24)
    "THYAO": "havacilik",   "EREGL": "celik",
    "KCHOL": "holding",     "SAHOL": "holding",
    "SISE":  "cam",         "TOASO": "otomotiv",
    "ARCLK": "tuketim",     "VESTL": "tuketim",
    "KRDMD": "celik",
    "PGSUS": "havacilik",   "ODAS":  "enerji",
    "GUBRF": "kimya",       "CIMSA": "cimento",
    "LOGO":  "teknoloji",   "NETAS": "teknoloji",
}

# Friday EOD warning (no forced close — user decides)
FRIDAY_WARN_TIME = (17, 30)  # send weekend warning on Fridays at 17:30

MARKET_START   = (10,  0)      # (hour, minute) Istanbul
MARKET_END     = (18,  0)
SKIP_UNTIL_MIN = 10 * 60 + 30  # no signals until 10:30 (opening volatility)
SUMMARY_HOUR   = 18
SUMMARY_MIN    = 30

TICK_SEC           = 60        # loop interval in seconds
EXIT_EVERY_N       = 5         # run exit check every N ticks (5 min)
TG_SIGNAL_INTERVAL = 5 * 60   # seconds between regular Telegram signal batches
EOD_REFRESH_MIN    = 18 * 60 + 5   # run fetch+indicators after 18:05

# Volatility override — set True to trade at x0.4 size even in EXTREME vol
# instead of full stop. Enabled when EXTREME vol persists > 2 trading days.
OVERRIDE_VOL_BLOCK = True

# P3.2 Tail risk protection
TAIL_RISK_DROP_PCT = 0.03   # GARAN -3% in last 30min → EXTREME override this tick
TAIL_RISK_PROXY    = "GARAN"
BASE_DIR           = Path(__file__).parent

RESULTS        = Path(__file__).parent / "results"
TRADE_LOG      = RESULTS / "trade_log.csv"
DAILY_SUMMARY  = RESULTS / "daily_summary.csv"
RESULTS.mkdir(exist_ok=True)

TRADE_LOG_COLS = [
    "tarih", "saat", "sembol", "giris_fiyat", "cikis_fiyat",
    "giris_saati", "cikis_saati", "boyut_tl", "kar_zarar_tl", "kar_zarar_pct",
    "sebep_al", "sebep_sat", "rsi_giris", "model_prob_giris",
    "model_prob_cikis", "tutulma_dk", "kumulatif_pnl",
    "risk_tl", "available_cash_before", "cash_reserve_pct",
    "net_pnl_tl",
]

DAILY_SUMMARY_COLS = [
    "tarih", "toplam_islem", "kazanan", "kaybeden", "win_rate",
    "gunluk_pnl_tl", "gunluk_pnl_pct", "kumulatif_pnl",
    "en_iyi_islem_tl", "en_kotu_islem_tl",
]

# -- Runtime state ------------------------------------------------------------
_dedup: dict[tuple, datetime] = {}   # (sym, direction) -> timezone-aware datetime
_summary_sent: str | None = None     # ISO date of last sent summary
_last_tg_signal: datetime | None = None  # timestamp of last regular signal Telegram
_eod_done: str | None = None         # ISO date of last post-market EOD refresh
_friday_warn_sent: str | None = None  # ISO date of last Friday weekend warning

# Trailing stop state (keyed by symbol; cleared on position close)
_pos_hwm: dict[str, float] = {}       # LONG sym -> session high-water-mark price
_pos_trailing: dict[str, float] = {}  # LONG sym -> current effective trailing stop
_pos_lwm: dict[str, float] = {}       # SHORT sym -> session low-water-mark price
ONE_SHOT = "--once" in sys.argv

# -- Kill switch equity peak (updated every loop) ----------------------------
_equity_peak: float = 100_000.0   # initialised in main(), updated each tick

# Circuit breaker state
_halt_notified: bool = False   # True once HALT telegram sent; reset on recovery

# P3.2 Daily stop state — reset each new calendar day
_today_start_equity: float | None = None   # equity at start of today
_today_date_equity:  str   | None = None   # ISO date matching above

# Reconciliation -- run once at market open each day
_recon_done_today: str | None = None


def send_telegram_if_critical(reason: str):
    """Send Telegram only for serious kill-switch events (drawdown / daily loss / kill)."""
    critical_keywords = ["KILL", "drawdown", "Drawdown", "Gunluk kayip"]
    if any(k in reason for k in critical_keywords):
        send_telegram_alert(f"[!] <b>KILL SWITCH</b>: {reason}")


# =============================================================================
# KILL SWITCH AUTO-EXPIRY
# =============================================================================

def _check_kill_switch_expiry() -> bool:
    """
    Auto-expire KILL_SWITCH.txt at session startup if ALL of these are true:
      1. The file is older than 24 hours.
      2. The drawdown that triggered it is no longer >= DRAWDOWN_KILL_PCT.

    Returns True if the file was deleted, False otherwise.
    """
    kill_path = Path(__file__).parent / "KILL_SWITCH.txt"
    if not kill_path.exists():
        return False

    age_h = (
        datetime.now() - datetime.fromtimestamp(kill_path.stat().st_mtime)
    ).total_seconds() / 3600

    if age_h < 24:
        print(f"[KS] KILL_SWITCH.txt mevcut (yas: {age_h:.1f}h < 24h) — hala aktif")
        return False

    # Check whether the drawdown condition still applies
    try:
        _, _, dd_pct = _compute_drawdown()
    except Exception:
        return False  # safe default: keep the file

    if dd_pct >= DRAWDOWN_KILL_PCT:
        print(
            f"[KS] KILL_SWITCH.txt yas {age_h:.1f}h > 24h ancak "
            f"DD={dd_pct:.1%} hala >= {DRAWDOWN_KILL_PCT:.0%} — korunuyor"
        )
        return False

    kill_path.unlink()
    msg = (
        f"KILL_SWITCH.txt otomatik suresi doldu: "
        f"yas={age_h:.1f}h, DD={dd_pct:.1%} < {DRAWDOWN_KILL_PCT:.0%}. "
        f"Islem yeniden basladi."
    )
    print(f"[KS] {msg}")
    algo_log.system(f"KILL_SWITCH auto-expired age={age_h:.1f}h dd={dd_pct:.1%}")
    send_telegram_alert(f"[KS] {msg}")
    return True


# =============================================================================
# STARTUP HEALTH CHECK
# =============================================================================

def _print_health_check(conn, model) -> None:
    """
    Printed once at startup. Shows the state of every major system component
    before trading begins, so invisible blockers surface immediately.
    """
    now_str   = ist_now().strftime("%Y-%m-%d %H:%M:%S")
    kill_path = Path(__file__).parent / "KILL_SWITCH.txt"

    print(f"\n{'='*60}")
    print(f"[HEALTH CHECK] {now_str}")
    print(f"{'='*60}")

    # 1. Kill switch file
    if kill_path.exists():
        age_h = (
            datetime.now() - datetime.fromtimestamp(kill_path.stat().st_mtime)
        ).total_seconds() / 3600
        try:
            content = kill_path.read_text().strip()[:80]
        except Exception:
            content = "(unreadable)"
        print(f"  KILL_SWITCH  : ACTIVE (yas:{age_h:.1f}h)  \"{content}\"")
    else:
        print(f"  KILL_SWITCH  : clear")

    # 2. Volatility regime
    try:
        vr = _vol_regime.get_regime_report()
        print(
            f"  VOL REGIME   : {vr['regime']} ({vr['vol_pct']:.2f}%)  "
            f"boyut x{vr['multiplier']:.1f}  "
            f"[HIGH>={HIGH_VOL_THRESHOLD*100:.1f}%  EXTREME>={EXTREME_VOL_THRESHOLD*100:.1f}%]"
            + ("  OVERRIDE_VOL_BLOCK=True" if OVERRIDE_VOL_BLOCK else "")
        )
    except Exception as e:
        print(f"  VOL REGIME   : HATA ({e})")

    # 2b. HMM regime
    try:
        if _hmm_regime_info is not None:
            print(
                f"  [REGIME] {_hmm_regime_info['regime']} | "
                f"P(bull)={_hmm_regime_info['prob_bull']:.2f} | "
                f"P(bear)={_hmm_regime_info['prob_bear']:.2f} | "
                f"Confirmed={_hmm_regime_info['confirmed']}"
            )
        else:
            print("  [REGIME] unavailable")
    except Exception as e:
        print(f"  [REGIME] HATA ({e})")

    # 2c. Meta-labeler status
    try:
        if _meta_labeling_active:
            print(
                f"  META LABEL   : ACTIVE "
                f"({ _meta_closed_trades } closed trades, thr={META_THRESHOLD:.2f})"
            )
        else:
            reason = (
                f"cold-start ({_meta_closed_trades}/{META_MIN_CLOSED_TRADES} closed trades)"
                if _meta_closed_trades < META_MIN_CLOSED_TRADES
                else ("model missing" if not META_MODEL_PATH.exists() else "disabled")
            )
            print(f"  META LABEL   : OFF ({reason})")
    except Exception as e:
        print(f"  META LABEL   : HATA ({e})")

    # 3. Model file timestamp
    if model is not None:
        model_files = sorted(Path("models").glob("*.pkl"), reverse=True)
        if model_files:
            mtime = datetime.fromtimestamp(model_files[0].stat().st_mtime)
            print(f"  MODEL        : {model_files[0].name}  (mtime: {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"  MODEL        : yuklendi (models/*.pkl bulunamadi)")
    else:
        print(f"  MODEL        : YUKLENEMEDI")

    # 4. Positions and cash
    open_pos  = get_open_positions(conn)
    cash      = get_cash(conn)
    tradeable = max(0.0, cash - MIN_CASH_RESERVE)
    print(f"  POZISYONLAR  : {len(open_pos)} acik / {MAX_POS} max")
    print(
        f"  NAKIT        : {cash:,.0f} TL  |  "
        f"islem:{tradeable:,.0f} TL  |  rezerv:{MIN_CASH_RESERVE:,} TL"
    )

    # 5. Drawdown
    try:
        cur_eq, peak_eq, dd = _compute_drawdown()
        tier = (
            "KILL" if dd >= DRAWDOWN_KILL_PCT else
            "HALT" if dd >= DRAWDOWN_HALT_PCT else
            "WARN" if dd >= DRAWDOWN_WARN_PCT else "OK"
        )
        print(
            f"  DRAWDOWN     : {dd:.1%}  "
            f"(equity:{cur_eq:,.0f}  peak:{peak_eq:,.0f})  CB:{tier}"
        )
    except Exception as e:
        print(f"  DRAWDOWN     : HATA ({e})")

    print(f"{'='*60}\n")


def _refresh_hmm_artifact(force_retrain: bool = False):
    """Ensure monthly HMM artifact exists and refresh current regime snapshot."""
    global _hmm_artifact, _hmm_regime_info
    now = ist_now()

    needs_retrain = force_retrain or not HMM_MODEL_PATH.exists()
    if not needs_retrain:
        payload = load_hmm_model(HMM_MODEL_PATH)
        trained_at = payload.get("trained_at") if payload else None
        if trained_at:
            try:
                trained_dt = datetime.fromisoformat(trained_at)
                if (
                    trained_dt.year != now.year
                    or trained_dt.month != now.month
                ) and now.day >= HMM_RETRAIN_DAY:
                    needs_retrain = True
            except Exception:
                needs_retrain = True
        else:
            needs_retrain = True

    if needs_retrain:
        try:
            _hmm_artifact = train_and_save_hmm_model(HMM_DB_PATH, HMM_MODEL_PATH)
            if _hmm_artifact is not None:
                print(f"  [HMM] retrained and saved: {HMM_MODEL_PATH.name}")
        except Exception as e:
            print(f"  [HMM] retrain failed: {e}")

    if _hmm_artifact is None:
        _hmm_artifact = load_hmm_model(HMM_MODEL_PATH)

    try:
        _hmm_regime_info = get_current_regime_from_artifact(
            HMM_DB_PATH,
            artifact=_hmm_artifact,
            confirmation_days=HMM_CONFIRMATION_DAYS,
        )
    except Exception as e:
        print(f"  [HMM] regime load failed: {e}")
        _hmm_regime_info = None

    return _hmm_artifact, _hmm_regime_info


def _refresh_meta_labeler(conn):
    """Enable meta-labeling only after enough closed paper trades exist."""
    global META_LABELING_ENABLED, _meta_labeler_artifact, _meta_labeling_active, _meta_closed_trades
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE exit_date IS NOT NULL"
        ).fetchone()
        _meta_closed_trades = int((row or [0])[0] or 0)
    except Exception:
        _meta_closed_trades = 0

    META_LABELING_ENABLED = _meta_closed_trades >= META_MIN_CLOSED_TRADES
    if not META_LABELING_ENABLED:
        _meta_labeler_artifact = None
        _meta_labeling_active = False
        return None, False

    if _meta_labeler_artifact is None:
        try:
            _meta_labeler_artifact = load_meta_model(META_MODEL_PATH)
        except Exception as e:
            print(f"  [META] model load failed: {e}")
            _meta_labeler_artifact = None

    _meta_labeling_active = _meta_labeler_artifact is not None
    return _meta_labeler_artifact, _meta_labeling_active


def _effective_buy_threshold(base_threshold: float, regime_info: dict | None) -> float:
    if not regime_info or not regime_info.get("confirmed", False):
        return float(base_threshold)
    regime = regime_info.get("regime")
    if regime == "BEAR":
        # Slight uplift for LONG caution in BEAR; SHORT scan uses its own threshold.
        # 1.5x (=0.975) was effectively a LONG block — 1.1x (=0.715) is cautious but workable.
        return float(min(0.999, base_threshold * 1.1))
    if regime == "RANGE":
        return float(min(0.999, base_threshold * 1.2))
    if regime == "BULL":
        return float(base_threshold * 0.85)
    return float(base_threshold)


# =============================================================================
# PER-TICK GATE SUMMARY
# =============================================================================

def _print_gate_summary(
    cb_status: str,
    dd_pct: float,
    vr_regime: str,
    vr_vol_pct: float,
    vr_mult: float,
    cash: float,
    n_open: int,
    n_tradeable: int,
    n_tradeable_long: int = 0,
    n_tradeable_short: int = 0,
) -> None:
    """
    Printed after every signal scan while market is open.
    Shows every system-level gate's current value vs threshold so that
    invisible blocking is surfaced in the console output.
    Format: [GATE] name: value -> PASS / BLOCK
    """
    now_str = ist_now().strftime("%H:%M")
    print(f"\n  [{now_str}] ---- GATE SUMMARY ----")

    # 1. Kill switch file
    ks_active = (Path(__file__).parent / "KILL_SWITCH.txt").exists()
    print(
        f"  [GATE] KillFile     : {'ACTIVE' if ks_active else 'clear'}"
        f" -> {'BLOCK' if ks_active else 'PASS'}"
    )

    # 2. Circuit breaker
    cb_ok = cb_status in ("OK", "WARN")
    cb_detail = f" [size 50%]" if cb_status == "WARN" else ""
    print(
        f"  [GATE] CircuitBreak : {cb_status} (DD:{dd_pct:.1%})"
        f" -> {'PASS' + cb_detail if cb_ok else 'BLOCK'}"
    )

    # 3. Volatility regime
    if vr_regime == "EXTREME":
        vr_label = "PASS [OVERRIDE x0.4]" if OVERRIDE_VOL_BLOCK else "BLOCK"
    elif vr_regime == "HIGH_VOL":
        vr_label = f"PASS [size x{vr_mult:.1f}]"
    else:
        vr_label = "PASS"
    print(
        f"  [GATE] VolRegime    : {vr_regime} ({vr_vol_pct:.2f}%)"
        f" -> {vr_label}"
    )

    # 4. Cash reserve
    tradeable = cash - MIN_CASH_RESERVE
    cash_ok   = tradeable >= 1_000
    print(
        f"  [GATE] Cash         : {cash:,.0f} TL (islem:{tradeable:,.0f})"
        f" -> {'PASS' if cash_ok else 'BLOCK (<1k islem edilebilir)'}"
    )

    # 5. Max positions
    pos_ok = n_open < MAX_POS
    print(
        f"  [GATE] MaxPositions : {n_open}/{MAX_POS}"
        f" -> {'PASS' if pos_ok else 'BLOCK'}"
    )

    # 6. Portfolio heat
    try:
        ph = _portfolio_risk.get_risk_report()
        ph_ok  = ph["allowed_new_position"]
        heat_p = ph["total_heat_pct"]
        print(
            f"  [GATE] PortfolioHeat: {heat_p:.2f}% vs {MAX_PORTFOLIO_HEAT*100:.0f}%"
            f" -> {'PASS' if ph_ok else 'BLOCK'}"
        )
    except Exception:
        print(f"  [GATE] PortfolioHeat: HATA (PASS varsayildi)")

    # 7. Tradeable signal count (LONG + SHORT, before per-signal gates)
    _lbl = "PASS" if n_tradeable else "BLOCK"
    print(
        f"  [GATE] Tradeable    : {n_tradeable} sinyal "
        f"(L:{n_tradeable_long} >= {BUY_THRESHOLD} | S:{n_tradeable_short} >= {SHORT_THRESHOLD})"
        f" -> {_lbl}"
    )

    print(f"  [{now_str}] -----------------------\n")

    # ── debug.log: structured gate summary ───────────────────────────────────
    ks_str   = "BLOCK" if ks_active else "PASS"
    cb_str   = "PASS" if cb_ok else "BLOCK"
    vr_str   = vr_label
    cash_str = "PASS" if cash_ok else "BLOCK"
    pos_str  = "PASS" if pos_ok else "BLOCK"
    try:
        _ph2 = _portfolio_risk.get_risk_report()
        heat_str = "PASS" if _ph2["allowed_new_position"] else "BLOCK"
    except Exception:
        heat_str = "PASS"
    sig_str = "PASS" if n_tradeable else "BLOCK"
    algo_log.debug(
        f"GATES "
        f"KILL={ks_str} CB={cb_str} VOL={vr_str} "
        f"CASH={cash_str} POS={pos_str} HEAT={heat_str} "
        f"SIGNALS={sig_str}({n_tradeable})"
    )


# =============================================================================
# P3.2A — INTRADAY TAIL RISK CHECK
# =============================================================================

def _check_intraday_tail_risk() -> tuple[bool, float]:
    """
    Detect a sharp intraday drop in the vol proxy (GARAN).

    Method: download GARAN.IS 5-min bars (period="2d"), compare the last
    available bar's close to the bar closest to 30 minutes ago.
    If the drop >= TAIL_RISK_DROP_PCT (3%), return (True, drop_pct).

    Returns (is_tail_risk: bool, drop_pct: float).
    drop_pct is a positive fraction when price fell (e.g. 0.035 = -3.5%).
    Called once per tick, inside the signal-scan block, before scan_and_trade.
    On error always returns (False, 0.0) — fail-safe.
    """
    if not HAS_YF:
        return False, 0.0
    try:
        raw = yf.download(
            f"{TAIL_RISK_PROXY}.IS", period="2d", interval="5m",
            progress=False, auto_adjust=True,
        )
        if raw is None or raw.empty:
            return False, 0.0
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw[raw["close"].notna() & (raw["close"] > 0)]
        if len(raw) < 7:
            return False, 0.0

        # Normalise to UTC-aware index
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=30)
        past   = raw[raw.index <= cutoff]
        if past.empty:
            return False, 0.0

        price_30m_ago = float(past["close"].iloc[-1])
        price_now     = float(raw["close"].iloc[-1])

        if price_30m_ago <= 0:
            return False, 0.0

        drop_pct = (price_30m_ago - price_now) / price_30m_ago  # positive = fell
        return drop_pct >= TAIL_RISK_DROP_PCT, drop_pct
    except Exception:
        return False, 0.0


# =============================================================================
# TIME HELPERS
# =============================================================================

def ist_now() -> datetime:
    return datetime.now(TZ)


def _mins(h: int, m: int) -> int:
    return h * 60 + m


def in_market() -> bool:
    t   = ist_now()
    cur = _mins(t.hour, t.minute)
    return _mins(*MARKET_START) <= cur < _mins(*MARKET_END)


def past_skip_window() -> bool:
    t = ist_now()
    return _mins(t.hour, t.minute) >= SKIP_UNTIL_MIN


def get_scan_interval_seconds(now_istanbul) -> int | None:
    """
    Returns the minimum scan interval (seconds) for the current Istanbul time,
    based on BIST intraday L-shaped volatility pattern (high vol at open & close).
    Returns None to skip signal scan entirely (before 10:30 OR after 17:30).

    Schedule:
      before 10:30  → None  (opening volatility skip window)
      10:30-12:30   → 900s  (15 min — morning high vol)
      12:30-14:30   → 1800s (30 min — low vol lunch dip)
      14:30-17:00   → 900s  (15 min — afternoon ramp)
      17:00-17:30   → 900s  (15 min — pre-close vol spike)
      after 17:30   → None  (no new entries; exits still active)
    """
    h, m = now_istanbul.hour, now_istanbul.minute
    t = h * 60 + m
    if t < 630:        # before 10:30
        return None
    elif t < 750:      # 10:30-12:30
        return 900
    elif t < 870:      # 12:30-14:30 (lunch)
        return 1800
    elif t < 1020:     # 14:30-17:00
        return 900
    elif t < 1050:     # 17:00-17:30
        return 900
    else:              # after 17:30
        return None


def is_friday_eod_warning() -> bool:
    """True on Friday at or after 17:30 (weekend warning window)."""
    t = ist_now()
    return t.weekday() == 4 and _mins(t.hour, t.minute) >= _mins(*FRIDAY_WARN_TIME)


def is_summary_window() -> bool:
    t = ist_now()
    return t.hour == SUMMARY_HOUR and t.minute >= SUMMARY_MIN


def should_eod_refresh() -> bool:
    t = ist_now()
    return _mins(t.hour, t.minute) >= EOD_REFRESH_MIN and not in_market()


# =============================================================================
# INDICATOR HELPERS  (replicated from indicators.py — pure numpy/pandas)
# =============================================================================

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _rsi(s, p=14):
    d    = s.diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(s, fast=12, slow=26, sig=9):
    line   = _ema(s, fast) - _ema(s, slow)
    signal = _ema(line, sig)
    return line, signal, line - signal

def _atr(h, l, c, p=14):
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()

def _bollinger(s, p=20, k=2.0):
    mid   = s.rolling(p).mean()
    sigma = s.rolling(p).std()
    upper = mid + k * sigma
    lower = mid - k * sigma
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    return upper, mid, lower, width

def _obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()

def _vol_ratio(v, p=20):
    return v / v.rolling(p).mean().replace(0, np.nan)

def _mtf(row):
    score = 0
    for a, b in [("ema8", "ema21"), ("ema21", "ema50"), ("ema50", "ema200")]:
        if pd.notna(row.get(a)) and pd.notna(row.get(b)):
            score += 1 if row[a] > row[b] else -1
    return 1 if score >= 2 else (-1 if score <= -2 else 0)


def recompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on an OHLCV dataframe (same logic as indicators.py)."""
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ema8"]   = _ema(c, 8)
    df["ema21"]  = _ema(c, 21)
    df["ema50"]  = _ema(c, 50)
    df["ema200"] = _ema(c, 200)
    df["rsi14"]  = _rsi(c, 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = _macd(c)
    df["atr14"]  = _atr(h, l, c, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_width"] = _bollinger(c)
    df["obv"]       = _obv(c, v)
    df["vol_ratio"] = _vol_ratio(v, 20)
    df["mtf_trend"] = df.apply(_mtf, axis=1)
    df["above_ema200"] = (c > df["ema200"]).astype(int)
    df["golden_cross"] = (df["ema50"] > df["ema200"]).astype(int)
    return df


# =============================================================================
# LIVE INTRADAY PRICE FETCH  (yfinance, 15-min delayed)
# =============================================================================

_OHLCV_SQL = (
    "SELECT date, open, high, low, close, volume "
    "FROM ohlcv WHERE symbol=? ORDER BY date"
)

# Cache live bars for the current tick so we don't re-download per symbol
_live_cache: dict = {}   # {"bars": {sym: dict}, "ts": datetime}
LIVE_CACHE_TTL = 55      # seconds — refresh every tick (slightly under TICK_SEC)


def fetch_all_live_bars() -> tuple[dict, datetime | None]:
    """
    Batch-fetch today's intraday OHLCV bars for all 14 symbols.

    Routes through algolab:
      MOCK mode (no API key): delegates to yfinance batch download (identical
                               to previous behavior -- zero functional change).
      LIVE mode (API key set): calls Algolab per-symbol endpoint.

    Returns ({sym: {open,high,low,close,volume}}, fetch_timestamp).
    """
    global _live_cache

    now = ist_now()
    cached = _live_cache.get("ts")
    if cached and (now - cached).total_seconds() < LIVE_CACHE_TTL:
        return _live_cache["bars"], cached

    src_tag = "ALGOLAB" if algolab.mode == "LIVE" else "yfinance(MOCK)"
    try:
        bars = algolab.get_all_bars(SYMBOLS)
    except Exception as e:
        print(f"  [Live] {src_tag} error: {e}")
        return {}, None

    if not bars:
        return {}, None

    _live_cache = {"bars": bars, "ts": now}
    return bars, now


# =============================================================================
# LIVE SIGNAL ENGINE  (replaces get_ml_signals each tick)
# =============================================================================

def get_live_signals(conn, model, feature_names, calibrator, thresholds) -> list:
    """
    Like get_ml_signals() but uses live intraday prices from yfinance.

    Per symbol:
      1. Load full OHLCV history from DB
      2. Patch today's row with live bar (open/high/low/close/volume)
      3. Recompute all indicators from scratch
      4. Build ML features and run model.predict_proba()

    Each signal dict carries extra keys:
      live_close, db_close, data_ts, is_live
    """
    today_str  = date.today().isoformat()
    live_bars, fetch_ts = fetch_all_live_bars()
    regime_info = _hmm_regime_info
    effective_buy_threshold = _effective_buy_threshold(BUY_THRESHOLD, regime_info)
    local_thresholds = dict(thresholds or {"buy": BUY_THRESHOLD_DEFAULT, "sell": SELL_THRESHOLD_DEFAULT})
    local_thresholds["buy"] = effective_buy_threshold

    # -- Load raw OHLCV histories from DB ------------------------------------
    all_dfs: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = pd.read_sql(_OHLCV_SQL, conn, params=(sym,))
        if df.empty or len(df) < 50:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        all_dfs[sym] = df

    # -- Patch with live bars + recompute indicators -------------------------
    ready_dfs: dict[str, pd.DataFrame] = {}
    for sym, df_hist in all_dfs.items():
        df = df_hist.copy()

        if sym in live_bars:
            bar       = live_bars[sym]
            today_ts  = pd.Timestamp(today_str)
            if today_ts in df.index:
                # Update existing today row
                for col in ("open", "high", "low", "close", "volume"):
                    df.loc[today_ts, col] = bar[col]
            else:
                # Append new row
                new_row = pd.DataFrame(
                    [{col: bar[col] for col in ("open","high","low","close","volume")}],
                    index=[today_ts],
                )
                df = pd.concat([df, new_row])

            # Persist live close to ohlcv so kill_switch fat-finger uses current price
            try:
                conn.execute(
                    """INSERT INTO ohlcv (symbol, date, open, high, low, close, volume, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'live')
                       ON CONFLICT(symbol, date) DO UPDATE SET
                         open=excluded.open, high=excluded.high, low=excluded.low,
                         close=excluded.close, volume=excluded.volume, source='live'""",
                    (sym, today_str, float(bar["open"]), float(bar["high"]),
                     float(bar["low"]), float(bar["close"]), float(bar["volume"])),
                )
                conn.commit()
            except Exception:
                pass   # fail-safe — stale price is better than crashing

        ready_dfs[sym] = recompute_indicators(df)

    # -- Market-wide close series (for cross-stock features) ----------------
    market_data = {sym: df["close"] for sym, df in ready_dfs.items()}

    # -- Load pre-computed macro features from DB ----------------------------
    try:
        _macro_live = pd.read_sql(
            "SELECT * FROM macro_data ORDER BY date", conn
        )
        if not _macro_live.empty:
            _macro_live["date"] = pd.to_datetime(_macro_live["date"])
            _macro_live = _macro_live.set_index("date")
            for _c in _macro_live.columns:
                _macro_live[_c] = pd.to_numeric(_macro_live[_c], errors="coerce")
        else:
            _macro_live = None
    except Exception:
        _macro_live = None

    # -- Run model per symbol ------------------------------------------------
    signals = []
    for sym, df in ready_dfs.items():
        try:
            regime_df = build_live_regime_feature_frame(df.index, regime_info)
            micro_df = compute_microstructure_features(df)
            X    = make_features_single(df, sym=sym, market_data=market_data, macro_df=_macro_live, regime_df=regime_df)
            last = X.fillna(0).iloc[[-1]]
            for feat in feature_names:
                if feat not in last.columns:
                    last[feat] = 0
            last = last[feature_names]
            try:
                base_prob = model.base_prob_matrix(last.fillna(0))[0]
            except Exception:
                base_prob = np.array([0.5, 0.5, 0.5], dtype=float)
            disagreement = float(np.std(base_prob[:3], ddof=0))

            # Shared model wrapper from paper_trade.py handles:
            #   - soft-vote average when stacking is disabled
            #   - meta-LR stacking on [xgb, lgb, cat, disagreement, max_prob] when enabled
            prob_buy          = calibrated_buy_prob(model, calibrator, last)[0]
            # Apply Beta calibration if available (post-calibration layer)
            if _beta_cal is not None:
                try:
                    prob_buy = float(np.clip(_beta_cal.predict([[prob_buy]])[0], 0.0, 1.0))
                except Exception:
                    pass   # fallback: use isotonic-calibrated prob as-is
            pred, conf_prob   = classify_prob(
                prob_buy,
                local_thresholds,
            )
            conf  = conf_prob * 100
            last_row = df.iloc[-1]
            last_micro = micro_df.iloc[-1] if not micro_df.empty else pd.Series(dtype=float)
            amihud_20 = float(last_micro.get("amihud_20", 0.0)) if pd.notna(last_micro.get("amihud_20", np.nan)) else 0.0
            mfi_14 = float(last_micro.get("mfi_14", 50.0)) if pd.notna(last_micro.get("mfi_14", np.nan)) else 50.0
            volume_surge = float(last_micro.get("volume_surge", 1.0)) if pd.notna(last_micro.get("volume_surge", np.nan)) else 1.0
            vol_pressure_z = float(last_micro.get("vol_pressure_z", 0.0)) if pd.notna(last_micro.get("vol_pressure_z", np.nan)) else 0.0
            sentiment_score = float(last_row.get("sentiment_score")) if pd.notna(last_row.get("sentiment_score", np.nan)) else 0.0

            # db_close = last close in DB (yesterday EOD)
            hist_df  = all_dfs.get(sym)
            db_close = float(hist_df["close"].iloc[-1]) if hist_df is not None else None

            signals.append({
                "symbol":     sym,
                "signal":     pred,
                "confidence": conf,
                "prob_buy":   float(prob_buy),
                "primary_prob_xgb": float(base_prob[0]),
                "primary_prob_lgb": float(base_prob[1]),
                "primary_prob_cat": float(base_prob[2]),
                "model_disagreement": disagreement,
                "price":      float(last_row["close"]),
                "atr":        float(last_row.get("atr14") or 0),
                "rsi":        float(last_row.get("rsi14") or 50),
                "mtf_trend":  int(last_row.get("mtf_trend") or 0),
                "data_date":  str(df.index[-1])[:10],
                "effective_threshold": float(effective_buy_threshold),
                "hmm_regime": regime_info["regime"] if regime_info else "UNKNOWN",
                "hmm_confirmed": bool(regime_info["confirmed"]) if regime_info else False,
                "hmm_prob_bull": float(regime_info["prob_bull"]) if regime_info else (1.0 / 3.0),
                "hmm_prob_bear": float(regime_info["prob_bear"]) if regime_info else (1.0 / 3.0),
                "hmm_days_in_state": float(regime_info.get("days_in_state", 0.0)) if regime_info else 0.0,
                "amihud_20": amihud_20,
                "mfi_14": mfi_14,
                "volume_surge": volume_surge,
                "vol_pressure_z": vol_pressure_z,
                "sentiment_score": sentiment_score,
                # --- debug extras ---
                "live_close": float(live_bars[sym]["close"]) if sym in live_bars else None,
                "db_close":   db_close,
                "data_ts":    fetch_ts,
                "is_live":    sym in live_bars,
            })
        except Exception as e:
            print(f"  [LiveSig] {sym} HATA: {e}")

    return signals


# =============================================================================
# DEBUG LINES  (printed each tick for key symbols)
# =============================================================================

DEBUG_SYMS = ["GARAN", "AKBNK", "TUPRS"]

def print_debug_lines(signals: list):
    """Show per-symbol data freshness and probability for key symbols."""
    sig_map  = {s["symbol"]: s for s in signals}
    src_tag  = "ALGOLAB" if algolab.mode == "LIVE" else "MOCK"
    for sym in DEBUG_SYMS:
        s = sig_map.get(sym)
        if s is None:
            continue
        ts_str = (
            s["data_ts"].strftime("%Y-%m-%d %H:%M")
            if s.get("data_ts") else "DB only"
        )
        live_tag = f"{src_tag}:LIVE" if s.get("is_live") else f"{src_tag}:DB"
        close    = s["live_close"] if s.get("is_live") else s["db_close"]
        print(
            f"  DBG {sym:<6} close:{close:.2f}  "
            f"[{live_tag}] ts:{ts_str}  "
            f"prob:{s['prob_buy']:.4f}"
        )


# =============================================================================
# FRIDAY WEEKEND WARNING  (17:30 on Fridays, once per day — no forced close)
# =============================================================================

def send_friday_weekend_warnings(conn):
    """
    On Fridays at 17:30, send one Telegram warning per open position.
    Positions are NOT closed — the user decides whether to hold over the weekend.
    Fires only once per day (guarded by _friday_warn_sent).
    """
    global _friday_warn_sent
    today_str = date.today().isoformat()
    if _friday_warn_sent == today_str:
        return

    open_pos = get_open_positions(conn)
    if not open_pos:
        _friday_warn_sent = today_str
        return

    print(f"\n[CUMA] Hafta sonu uyarisi gonderiliyor ({len(open_pos)} pozisyon)...")
    for pos in open_pos:
        sym   = pos["symbol"]
        entry = pos["entry_price"]
        stop  = _pos_trailing.get(sym, pos["stop_price"])
        size  = pos["size_tl"]
        lat   = get_latest(conn, sym)
        cur   = float(lat["close"]) if lat else entry
        pnl_pct = (cur - entry) / entry * 100

        send_telegram_alert(
            f"⚠ <b>HAFTA SONU UYARISI: {sym}</b>\n"
            f"Pozisyon acik kalacak. Hafta sonu kapatamazsiniz.\n"
            f"Pazartesi fiyat bogluklarla acilabilir.\n\n"
            f"Giris     : {entry:.2f} TL\n"
            f"Su an     : {cur:.2f} TL  ({pnl_pct:+.2f}%)\n"
            f"Eff. Stop : {stop:.2f} TL\n"
            f"Boyut     : {size:,.0f} TL\n\n"
            f"Kapatmak icin: KILL_SWITCH.txt olustur, loop_trader'i --once ile calistir."
        )
        print(f"  [CUMA] {sym}  giris:{entry:.2f}  simdi:{cur:.2f}  "
              f"pnl:{pnl_pct:+.2f}%  stop:{stop:.2f}")

    _friday_warn_sent = today_str


# =============================================================================
# EOD AUTO-REFRESH  (runs fetch_data.py + indicators.py after 18:05)
# =============================================================================

def run_eod_refresh():
    global _eod_done
    today_str = date.today().isoformat()
    if _eod_done == today_str:
        return

    print(f"\n[EOD] Post-market refresh basladi ({today_str})...")
    py = sys.executable
    for script in ["fetch_data.py", "indicators.py"]:
        print(f"[EOD] Calistiriliyor: {script} ...", end=" ", flush=True)
        try:
            r = subprocess.run(
                [py, str(BASE_DIR / script)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                print("OK")
            else:
                err = (r.stderr or "").strip()[:120]
                print(f"HATA  {err}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT (300s)")
        except Exception as e:
            print(f"HATA: {e}")

    _eod_done = today_str
    send_telegram_alert(
        f"🔄 <b>EOD Veri Guncelleme Tamamlandi</b>\n"
        f"Tarih  : {today_str}\n"
        f"Scripts: fetch_data.py + indicators.py"
    )
    print(f"[EOD] Tamamlandi.")


# =============================================================================
# CSV HELPERS
# =============================================================================

def _migrate_csv(fpath: Path, cols: list):
    """Create CSV if missing; rewrite if columns differ (adds or removes)."""
    if not fpath.exists():
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
        return
    with open(fpath, "r", encoding="utf-8") as f:
        existing_cols = next(csv.reader(f), [])
    added   = [c for c in cols if c not in existing_cols]
    removed = [c for c in existing_cols if c not in cols]
    if not added and not removed:
        return
    with open(fpath, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            for col in added:
                row.setdefault(col, "")
            # removed columns are simply not written (DictWriter ignores extras)
            w.writerow({c: row.get(c, "") for c in cols})
    parts = []
    if added:
        parts.append(f"added {added}")
    if removed:
        parts.append(f"removed {removed}")
    print(f"  [CSV] Migrated {fpath.name}: {', '.join(parts)}")


def ensure_csvs():
    _migrate_csv(TRADE_LOG,    TRADE_LOG_COLS)
    _migrate_csv(DAILY_SUMMARY, DAILY_SUMMARY_COLS)


def append_trade_log(row: dict):
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRADE_LOG_COLS).writerow(row)


def append_daily_summary(row: dict):
    with open(DAILY_SUMMARY, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=DAILY_SUMMARY_COLS).writerow(row)


# =============================================================================
# POSITION HELPERS
# =============================================================================

def get_current_equity(conn) -> float:
    """cash + mark-to-market value of open positions (using last DB close)."""
    open_pos = get_open_positions(conn)
    cash     = get_cash(conn)
    pos_val  = 0.0
    for pos in open_pos:
        lat = get_latest(conn, pos["symbol"])
        if lat:
            cur = float(lat["close"])
            if pos["is_long"]:
                pos_val += pos["size_tl"] * (cur / pos["entry_price"])
            else:
                pct = (pos["entry_price"] - cur) / pos["entry_price"]
                pos_val += pos["size_tl"] * (1 + pct)
        else:
            pos_val += pos["size_tl"]
    return cash + pos_val


def get_cumulative_pnl(conn) -> float:
    row = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM paper_trades").fetchone()
    return float(row[0])


def is_dedup(sym: str, direction: str) -> bool:
    key = (sym, direction)
    ts  = _dedup.get(key)
    if ts is None:
        return False
    return (ist_now() - ts) < timedelta(hours=DEDUP_HOURS)


def mark_dedup(sym: str, direction: str):
    _dedup[(sym, direction)] = ist_now()


def calculate_position_size(price: float, atr: float, available_cash: float) -> float:
    """
    ATR-based position sizing with hard caps.

    Logic:
      1. risk_tl      = TOTAL_CAPITAL * RISK_PER_TRADE  (1,000 TL)
      2. stop_dist    = atr * 1.5  (same multiplier as ATR_TARGET_MULT)
      3. size_by_risk = (risk_tl / stop_dist) * price   (shares * price)
      4. Capped at MAX_POSITION_SIZE (10,000 TL)
      5. Further capped so remaining cash stays above MIN_CASH_RESERVE
      6. Returns 0 if result < 1,000 TL (too small to be worth it)
    """
    if price <= 0 or atr <= 0:
        return 0.0
    risk_tl      = TOTAL_CAPITAL * RISK_PER_TRADE           # 1,000 TL
    stop_dist    = atr * ATR_TARGET_MULT                     # reuse same multiplier
    size_by_risk = (risk_tl / stop_dist) * price
    size         = min(size_by_risk, MAX_POSITION_SIZE)      # hard cap
    size         = min(size, available_cash - MIN_CASH_RESERVE)  # never breach reserve
    if size < 1_000:
        return 0.0
    return round(size, 0)


# =============================================================================
# SIGNAL DISPLAY  (every tick, informational only)
# =============================================================================

def show_signal_report(signals: list, now_str: str):
    """
    Print every symbol whose directional probability exceeds DISPLAY_THRESHOLD.

    BUY  direction: prob_buy  > 0.50  (displayed as prob_buy)
    SELL direction: prob_buy  < 0.50  (displayed as 1 - prob_buy)

    If nothing crosses 0.50 at all, dump raw prob_buy for all symbols so we
    can diagnose model drift or data staleness.
    """
    rows = []
    for sig in signals:
        pb  = sig.get("prob_buy", 0.5)
        sym = sig["symbol"]
        rsi = sig.get("rsi", 0.0)
        if pb > DISPLAY_THRESHOLD:
            rows.append((sym, "BUY ", pb,        rsi))
        elif pb < (1.0 - DISPLAY_THRESHOLD):
            rows.append((sym, "SELL", 1.0 - pb,  rsi))

    rows.sort(key=lambda x: x[2], reverse=True)

    if not rows:
        # Nothing above 0.50 in either direction — debug dump
        print(f"  [{now_str}] Signals above {DISPLAY_THRESHOLD}: NONE "
              f"(raw prob_buy for all {len(signals)} symbols):")
        for s in sorted(signals, key=lambda x: x.get("prob_buy", 0.5), reverse=True):
            pb = s.get("prob_buy", 0.5)
            print(f"    {s['symbol']:<6}  prob_buy={pb:.4f}  "
                  f"RSI={s.get('rsi', 0):.0f}  price={s.get('price', 0):.2f}")
        return

    print(f"  [{now_str}] Signals above {DISPLAY_THRESHOLD}:")
    n_tradeable = 0
    for sym, direction, disp_prob, rsi in rows:
        sig_obj   = next((s for s in signals if s["symbol"] == sym), None)
        raw_pb    = sig_obj.get("prob_buy", 0.5) if sig_obj else 0.5
        raw_ps    = 1.0 - raw_pb
        regime    = sig_obj.get("hmm_regime", "?") if sig_obj else "?"
        # Tier: LONG uses effective_threshold; SELL uses SHORT_THRESHOLD
        if direction == "BUY ":
            eff_thr = sig_obj.get("effective_threshold", BUY_THRESHOLD) if sig_obj else BUY_THRESHOLD
            tradeable = disp_prob >= eff_thr
        else:
            eff_thr   = SHORT_THRESHOLD
            tradeable = disp_prob >= eff_thr and rsi > RSI_SHORT_MIN
        if tradeable:
            marker = "<= TRADEABLE"
            tier   = "TRADEABLE"
            n_tradeable += 1
        else:
            marker = f"<- below threshold ({eff_thr:.3f})"
            tier   = "WEAK"
        print(f"    {sym:<6}  {direction}  {disp_prob:.2f}  RSI:{rsi:.0f}  {marker}")
        algo_log.debug(
            f"SIGNAL sym={sym} prob_buy={raw_pb:.4f} "
            f"prob_sell={raw_ps:.4f} rsi={rsi:.1f} "
            f"regime={regime} tier={tier}"
        )

    ref_thr = signals[0].get("effective_threshold", BUY_THRESHOLD) if signals else BUY_THRESHOLD
    print(f"  Above threshold ({ref_thr:.3f}): {n_tradeable} signals")


# =============================================================================
# TELEGRAM SIGNAL REPORT
# =============================================================================

def _categorise(signals: list) -> tuple[list, list, list]:
    """
    Split signals into (strong, medium, weak) lists by directional probability.

    BUY  direction: prob_buy  >= 0.50
    SELL direction: prob_buy  <= 0.50  (sell_prob = 1 - prob_buy)

    Bands:
      strong >= 0.65  |  medium 0.60-0.64  |  weak 0.50-0.59
    """
    strong, medium, weak = [], [], []
    for sig in signals:
        pb  = sig.get("prob_buy", 0.5)
        sym = sig["symbol"]
        rsi = sig.get("rsi", 0.0)

        if pb >= 0.65:
            strong.append((sym, "BUY",  pb,        rsi))
        elif pb >= 0.60:
            medium.append((sym, "BUY",  pb,        rsi))
        elif pb > 0.50:
            weak.append(  (sym, "BUY",  pb,        rsi))
        elif pb < 0.35:                          # sell_prob >= 0.65
            strong.append((sym, "SELL", 1.0 - pb, rsi))
        elif pb <= 0.40:                         # sell_prob 0.60-0.64
            medium.append((sym, "SELL", 1.0 - pb, rsi))
        elif pb < 0.50:                          # sell_prob 0.50-0.59
            weak.append(  (sym, "SELL", 1.0 - pb, rsi))

    for lst in (strong, medium, weak):
        lst.sort(key=lambda x: x[2], reverse=True)

    return strong, medium, weak


def send_signal_telegram(signals: list, now: datetime):
    """
    Send grouped Telegram signal message.

    Logic:
      - Nothing above 0.50 in either direction  -> send nothing
      - STRONG (>=0.65) detected                -> send full grouped message immediately
      - No STRONG                               -> send WEAK+MEDIUM every TG_SIGNAL_INTERVAL
    """
    global _last_tg_signal

    strong, medium, weak = _categorise(signals)

    if not (strong or medium or weak):
        return  # nothing above 0.50, stay silent

    has_strong = bool(strong)

    # Decide whether to send the regular batch now
    send_regular = has_strong or (
        _last_tg_signal is None or
        (now - _last_tg_signal).total_seconds() >= TG_SIGNAL_INTERVAL
    )
    if not send_regular:
        return

    # Build grouped message
    lines = [f"📊 <b>BIST Sinyal Raporu  {now.strftime('%H:%M')}</b>"]

    if weak:
        lines.append("\n🟡 <b>WEAK (0.50-0.59):</b>")
        for sym, direction, prob, _ in weak:
            lines.append(f"  {sym} {direction} {prob:.2f}")

    if medium:
        lines.append("\n🟠 <b>MEDIUM (0.60-0.64):</b>")
        for sym, direction, prob, _ in medium:
            lines.append(f"  {sym} {direction} {prob:.2f}")

    lines.append("\n🟢 <b>STRONG (0.65+) — TRADEABLE:</b>")
    if strong:
        for sym, direction, prob, rsi in strong:
            lines.append(f"  {sym} {direction} {prob:.2f}  RSI:{rsi:.0f}")
    else:
        lines.append("  (none yet)")

    ok = send_telegram_alert("\n".join(lines))
    if ok:
        _last_tg_signal = now
        kind = "STRONG alert" if has_strong else "regular update"
        n = len(strong) + len(medium) + len(weak)
        print(f"  [TG] Signal report sent ({kind}): "
              f"{len(strong)} strong / {len(medium)} medium / {len(weak)} weak")


# =============================================================================
# SIGNAL SCAN + TRADE EXECUTION
# =============================================================================

def _log_signal(conn, symbol: str, prob_buy: float, prob_sell: float,
                regime: str, gate_result: str, reason_blocked,
                trade_opened: bool) -> None:
    """
    Persist one scan decision to signals_log AND write a debug.log line.
    Never raises — any error is silently swallowed so the main loop is safe.
    gate_result : "PASS" | "BLOCK"
    reason_blocked: str code or None (e.g. "RSI_HIGH", "SECTOR_LIMIT", ...)
    """
    # debug.log per-symbol gate decision
    algo_log.debug(
        f"SIGNAL_GATE sym={symbol} result={gate_result} "
        f"reason={reason_blocked} prob={prob_buy:.4f}"
    )
    try:
        conn.execute(
            """INSERT INTO signals_log
               (ts, signal_date, symbol, prob_buy, prob_sell, regime,
                gate_result, reason_blocked,
                threshold_passed, trade_opened)
               VALUES (datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol,
                round(float(prob_buy),  6),
                round(float(prob_sell), 6),
                regime,
                gate_result,
                reason_blocked,
                1 if (prob_buy >= BUY_THRESHOLD or prob_sell >= SHORT_THRESHOLD) else 0,
                1 if trade_opened else 0,
            ),
        )
        conn.commit()
    except Exception as _e:
        algo_log.debug(f"signals_log write error: {_e}")


def scan_and_trade(conn, model, feature_names, calibrator, thresholds,
                   _signals=None, max_pos_override=None,
                   vol_multiplier: float = 1.0) -> int:
    """
    Scan ML signals; open BUY positions for qualifying signals.
    max_pos_override: when set (WARN mode), cap each position at this TL value.
    vol_multiplier  : 0.6 in HIGH_VOL regime; 1.0 normally. Applied to size_tl.
    Returns number of new positions opened.
    """
    today_str = date.today().isoformat()
    now       = ist_now()

    open_pos  = get_open_positions(conn)
    n_open    = len(open_pos)
    if n_open >= MAX_POS:
        return 0

    open_syms   = {p["symbol"] for p in open_pos}
    signals     = _signals if _signals is not None else \
                  get_ml_signals(conn, model, feature_names, calibrator, thresholds)
    opened      = 0
    recent_news = read_recent_news(60)   # news items from last 60 minutes

    # Sort best confidence first
    for sig in sorted(signals, key=lambda x: x["confidence"], reverse=True):
        sym      = sig["symbol"]
        pred     = sig["signal"]
        prob_buy = sig.get("prob_buy", 0.5)
        eff_buy_threshold = float(sig.get("effective_threshold", BUY_THRESHOLD))
        conf     = sig["confidence"]
        price    = sig["price"]
        atr      = sig["atr"]
        rsi      = sig["rsi"]

        # Only BUY signals above threshold
        if pred != 1:
            _log_signal(conn, sym, prob_buy, 1-prob_buy,
                        sig.get("hmm_regime","?"), "BLOCK", "LOW_PROB", False)
            continue
        if prob_buy < eff_buy_threshold:
            _log_signal(conn, sym, prob_buy, 1-prob_buy,
                        sig.get("hmm_regime","?"), "BLOCK", "LOW_PROB", False)
            continue
        meta_prob = None
        if _meta_labeling_active:
            try:
                meta_X = build_live_meta_feature_vector(conn, sig, as_of_date=today_str)
                meta_prob = float(predict_meta_probability(_meta_labeler_artifact, meta_X)[0])
                if meta_prob < META_THRESHOLD:
                    print(
                        f"  [META ] {sym} primary={prob_buy:.3f} meta={meta_prob:.3f} "
                        f"-> BLOCK (< {META_THRESHOLD:.2f})"
                    )
                    algo_log.system(f"META_BLOCK {sym} primary={prob_buy:.3f} meta={meta_prob:.3f}")
                    continue
            except Exception as e:
                print(f"  [META ] {sym} gate hata ({e}) -- fail-open")

        # --- From here: tradeable signal (prob >= 0.65, pred == BUY) ---
        # Every skip path below is now logged so it appears in console output.
        _sig_regime = sig.get("hmm_regime", "?")
        if sym in open_syms:
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (zaten acik pozisyon var)")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "OPEN_POS", False)
            continue
        if n_open + opened >= MAX_POS:
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (MAX_POS={MAX_POS} dolu: {n_open+opened})")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "MAX_POS", False)
            break
        if atr <= 0:
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (ATR={atr:.6f} <= 0, boyutlandirma imkansiz)")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "ATR_ZERO", False)
            continue
        if is_dedup(sym, "AL"):
            dedup_ts  = _dedup.get((sym, "AL"))
            elapsed_h = round((ist_now() - dedup_ts).total_seconds() / 3600, 1) if dedup_ts else 0
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (dedup {elapsed_h}h / {DEDUP_HOURS}h bekleniyor)")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "DEDUP", False)
            continue
        # RSI extreme filter
        if rsi > 72:
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (RSI={rsi:.0f} > 72 asiri alim)")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "RSI_HIGH", False)
            continue

        # Sector correlation limit
        sym_sector   = SECTOR_MAP.get(sym, "other")
        sector_count = sum(
            1 for p in open_pos
            if SECTOR_MAP.get(p["symbol"], "other") == sym_sector
        )
        if sector_count >= MAX_PER_SECTOR:
            print(f"  [SKIP ] {sym} — sektor limiti ({sym_sector}: {sector_count}/{MAX_PER_SECTOR})")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "SECTOR_LIMIT", False)
            continue

        # News filter — skip entry if any NEGATIVE news in last 60 min
        sym_news = [n for n in recent_news if n.get("symbol") == sym]
        if sym_news and any(score_news(n["title"], sym)[0] == "NEGATIVE"
                            for n in sym_news):
            print(f"  [SKIP ] {sym} -- negatif haber var ({len(sym_news)} haber)")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "NEWS_NEG", False)
            continue

        # Fill = exact market price (no slippage markup)
        entry_fill = price
        dq = check_entry_price(sym, entry_fill)
        if not dq.ok:
            reason = format_data_quality_block(dq)
            print(f"  [DATA ] {sym} -> SKIP ({reason})")
            algo_log.risk(f"DATA QUALITY BLOCK {sym}: {reason}")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", dq.code, False)
            continue
        stop   = round(entry_fill * (1 - SL_PCT), 2)
        target = round(entry_fill + atr * ATR_TARGET_MULT, 2)  # ATR x 1.5

        # ATR-based position sizing
        cash    = get_cash(conn)
        size_tl = calculate_position_size(entry_fill, atr, cash)
        if size_tl == 0:
            risk_tl_needed = TOTAL_CAPITAL * RISK_PER_TRADE
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (size=0: "
                  f"risk={risk_tl_needed:.0f}TL ATR={atr:.4f} cash={cash:,.0f}TL)")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "SIZE_ZERO", False)
            continue

        # Cash reserve gate (defensive double-check)
        if cash - size_tl < MIN_CASH_RESERVE:
            print(f"  [GATE ] {sym} prob={prob_buy:.3f} -> SKIP (nakit rezervi ihlali: "
                  f"{cash:,.0f} - {size_tl:,.0f} < rezerv {MIN_CASH_RESERVE:,})")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "CASH_RESERVE", False)
            continue

        # Portfolio heat + correlation gate
        stop_dist_tl = size_tl * (entry_fill - stop) / entry_fill if entry_fill > 0 else 0
        ph_allowed, ph_reason = _portfolio_risk.check_new_position_allowed(sym, stop_dist_tl)
        if not ph_allowed:
            print(f"  [SKIP ] {sym} — portfolio risk gate: {ph_reason}")
            algo_log.risk(f"PORTFOLIO GATE BLOCK {sym}: {ph_reason}")
            _bl = "CORR_GATE" if "corr" in ph_reason.lower() else "HEAT_CAP"
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", _bl, False)
            continue

        # Circuit breaker WARN cap: reduce position size when DD >= 4%
        if max_pos_override is not None and size_tl > max_pos_override:
            size_tl = max_pos_override
            if size_tl < 1_000:
                print(f"  [SKIP ] {sym} — WARN cap: {size_tl:.0f} TL < 1000 TL min")
                continue

        # Volatility regime size scaling (HIGH_VOL = 0.6x, applied after all other caps)
        if vol_multiplier < 1.0:
            size_tl = round(size_tl * vol_multiplier, 0)
            if size_tl < 1_000:
                print(f"  [SKIP ] {sym} — HIGH_VOL size kucultme sonrasi {size_tl:.0f} TL < 1000 TL min")
                continue

        # Compute regime and entry_reason before OMS check
        mtf    = sig.get("mtf_trend", 0)
        regime = "YUKARI" if mtf == 1 else "ASAGI" if mtf == -1 else "YATAY"
        entry_reason = (
            f"ML BUY prob={prob_buy:.3f} conf={conf:.1f}% "
            f"RSI={rsi:.1f} {regime}"
        )

        # OMS: kill switch check + order lifecycle log (replaces KillSwitch.check_all)
        order = oms.create_order(sym, "BUY", entry_fill, size_tl, reason=entry_reason)
        if order.status == OrderStatus.REJECTED:
            print(f"  [KILL ] {sym} BUY engellendi: {order.rejected_reason}")
            send_telegram_if_critical(order.rejected_reason or "")
            _log_signal(conn, sym, prob_buy, 1-prob_buy, _sig_regime, "BLOCK", "KILL_SWITCH", False)
            continue

        # macd_hist from DB
        latest = get_latest(conn, sym)
        macd_h = float(latest["macd_hist"]) if (latest and latest["macd_hist"] is not None) else 0.0

        qty    = round(size_tl / entry_fill, 4)

        # Risk metrics
        risk_tl_entry    = round(size_tl * (entry_fill - stop) / entry_fill, 2)
        avail_cash_entry = round(cash, 2)
        reserve_pct      = round(avail_cash_entry / TOTAL_CAPITAL * 100, 1)

        # Encode entry metadata in entry_note for exit log reconstruction
        entry_note = json.dumps({
            "reason":               entry_reason,
            "rsi":                  round(rsi, 2),
            "macd_hist":            round(macd_h, 6),
            "prob":                 round(prob_buy, 4),
            "meta_prob":            round(meta_prob, 4) if meta_prob is not None else None,
            "entry_time":           now.strftime("%Y-%m-%d %H:%M:%S"),
            "size_tl":              round(size_tl, 2),
            "quantity":             qty,
            "atr":                  round(atr, 4),
            "risk_tl":              risk_tl_entry,
            "available_cash_before": avail_cash_entry,
            "cash_reserve_pct":     reserve_pct,
        })

        open_position(
            conn, sym, "ML_AL", entry_fill, True,
            size_tl, stop, target, conf, today_str,
            signal_prob=prob_buy, regime=regime, entry_note=entry_note,
        )

        # OMS: fill order (also logs to kill switch rate limiter)
        oms.fill_order(order, entry_fill)
        algo_log.log_buy(sym, entry_fill, size_tl, prob_buy,
                         stop=stop, target=target)

        # Initialise trailing-stop memory for this position
        _pos_hwm[sym]      = entry_fill
        _pos_trailing[sym] = stop

        _log_signal(conn, sym, prob_buy, 1-prob_buy, regime, "PASS", None, True)
        mark_dedup(sym, "AL")
        open_syms.add(sym)
        opened += 1

        tp_pct = (target - entry_fill) / entry_fill * 100
        print(f"  [BUY ] {sym} @ {entry_fill:.2f} | "
              f"size={size_tl:,.0f} TL  ({qty:.4f} adet) | "
              f"sektor={sym_sector}  risk={risk_tl_entry:.0f} TL | "
              f"stop={stop}  hedef={target} (+{tp_pct:.1f}%)")

        send_telegram_alert(
            f"🟢 <b>ALINDI: {sym} @ {entry_fill:.2f}</b>\n"
            f"Boyut          : {size_tl:,.0f} TL  ({qty:.4f} adet)\n"
            f"Risk           : {risk_tl_entry:.0f} TL\n"
            f"Nakit Kalan    : {avail_cash_entry:,.0f} TL  (Rezerv:{reserve_pct:.0f}%)\n"
            f"Sektor         : {sym_sector}\n"
            f"Model Prob     : {prob_buy:.2f}  |  RSI:{rsi:.1f}  Trend:{regime}\n"
            f"ATR            : {atr:.4f}\n"
            f"Stop           : {stop} TL  (-{SL_PCT*100:.1f}%)\n"
            f"Hedef          : {target} TL  (+{tp_pct:.1f}%  ATR*{ATR_TARGET_MULT})"
        )

    # ── SHORT SCAN ────────────────────────────────────────────────────────────
    # Open SHORT when prob_sell = 1-prob_buy >= SHORT_THRESHOLD and RSI > RSI_SHORT_MIN
    for sig in sorted(signals, key=lambda x: 1 - x.get("prob_buy", 0.5), reverse=True):
        sym      = sig["symbol"]
        prob_buy = sig.get("prob_buy", 0.5)
        prob_sell = 1.0 - prob_buy
        price    = sig["price"]
        atr      = sig["atr"]
        rsi      = sig["rsi"]

        _s_regime = sig.get("hmm_regime", "?")
        if prob_sell < SHORT_THRESHOLD:
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "LOW_PROB", False)
            continue
        if rsi <= RSI_SHORT_MIN:
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "RSI_LOW", False)
            continue

        if sym in open_syms:
            print(f"  [GATE ] [SHORT] {sym} prob_sell={prob_sell:.3f} -> SKIP (zaten acik pozisyon)")
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "OPEN_POS", False)
            continue
        if n_open + opened >= MAX_POS:
            print(f"  [GATE ] [SHORT] {sym} -> SKIP (MAX_POS dolu)")
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "MAX_POS", False)
            break
        if atr <= 0:
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "ATR_ZERO", False)
            continue
        if is_dedup(sym, "SAT"):
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "DEDUP", False)
            continue

        sym_sector   = SECTOR_MAP.get(sym, "other")
        sector_count = sum(
            1 for p in open_pos
            if SECTOR_MAP.get(p["symbol"], "other") == sym_sector
        )
        if sector_count >= MAX_PER_SECTOR:
            print(f"  [SKIP ] [SHORT] {sym} — sektor limiti ({sym_sector})")
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "SECTOR_LIMIT", False)
            continue

        # News filter — skip SHORT if very positive news
        sym_news = [n for n in recent_news if n.get("symbol") == sym]
        if sym_news and any(score_news(n["title"], sym)[1] >= 2 for n in sym_news):
            print(f"  [SKIP ] [SHORT] {sym} -- pozitif haber engeli")
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "NEWS_POS", False)
            continue

        entry_fill = price
        dq = check_entry_price(sym, entry_fill)
        if not dq.ok:
            reason = format_data_quality_block(dq)
            print(f"  [DATA ] [SHORT] {sym} -> SKIP ({reason})")
            algo_log.risk(f"DATA QUALITY BLOCK SHORT {sym}: {reason}")
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", dq.code, False)
            continue
        stop       = round(entry_fill * (1 + SHORT_SL_PCT), 2)   # above entry
        target     = round(entry_fill - atr * ATR_TARGET_MULT, 2) # below entry

        cash    = get_cash(conn)
        # Short size: ATR-based but capped at MAX_SHORT_SIZE
        size_tl = min(calculate_position_size(entry_fill, atr, cash), MAX_SHORT_SIZE)
        if size_tl < 1_000:
            print(f"  [GATE ] [SHORT] {sym} -> SKIP (size={size_tl:.0f} TL < 1000)")
            continue
        if cash - size_tl < MIN_CASH_RESERVE:
            print(f"  [GATE ] [SHORT] {sym} -> SKIP (nakit rezerv ihlali)")
            continue

        # Vol multiplier
        if vol_multiplier < 1.0:
            size_tl = round(size_tl * vol_multiplier, 0)
            if size_tl < 1_000:
                continue

        # WARN cap
        if max_pos_override is not None and size_tl > max_pos_override:
            size_tl = max_pos_override
            if size_tl < 1_000:
                continue

        # Portfolio heat
        stop_dist_tl = size_tl * (stop - entry_fill) / entry_fill if entry_fill > 0 else 0
        ph_allowed, ph_reason = _portfolio_risk.check_new_position_allowed(sym, stop_dist_tl)
        if not ph_allowed:
            print(f"  [SKIP ] [SHORT] {sym} — portfolio risk gate: {ph_reason}")
            continue

        conf = round(prob_sell * 100, 1)
        entry_reason = (
            f"ML SHORT prob_sell={prob_sell:.3f} conf={conf:.1f}% RSI={rsi:.1f}"
        )

        order = oms.create_order(sym, "SHORT", entry_fill, size_tl, reason=entry_reason)
        if order.status == OrderStatus.REJECTED:
            print(f"  [KILL ] [SHORT] {sym} engellendi: {order.rejected_reason}")
            _log_signal(conn, sym, prob_buy, prob_sell, _s_regime, "BLOCK", "KILL_SWITCH", False)
            continue

        qty = round(size_tl / entry_fill, 4)
        sl_pct_show = SHORT_SL_PCT * 100
        tp_pct_show = (entry_fill - target) / entry_fill * 100

        open_position(
            conn, sym, "ML_SAT", entry_fill, False,
            size_tl, stop, target, conf, today_str,
            signal_prob=prob_sell, regime="ASAGI",
            entry_note=json.dumps({
                "reason":    entry_reason,
                "rsi":       round(rsi, 2),
                "prob":      round(prob_sell, 4),
                "entry_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "size_tl":   round(size_tl, 2),
                "quantity":  qty,
                "atr":       round(atr, 4),
            }),
        )
        oms.fill_order(order, entry_fill)
        algo_log.log_buy(sym, entry_fill, size_tl, prob_sell,
                         stop=stop, target=target)

        _log_signal(conn, sym, prob_buy, prob_sell, "ASAGI", "PASS", None, True)
        _pos_lwm[sym] = entry_fill
        mark_dedup(sym, "SAT")
        open_syms.add(sym)
        opened += 1

        print(f"  [SHORT] {sym} @ {entry_fill:.2f} | "
              f"size={size_tl:,.0f} TL  ({qty:.4f} adet) | "
              f"stop={stop} (+{sl_pct_show:.1f}%)  hedef={target} (-{tp_pct_show:.1f}%)")

        send_telegram_alert(
            f"🔴 <b>[SHORT] ACILDI: {sym} @ {entry_fill:.2f}</b>\n"
            f"Boyut          : {size_tl:,.0f} TL  ({qty:.4f} adet)\n"
            f"Sektor         : {sym_sector}\n"
            f"Model prob_sell: {prob_sell:.2f}  RSI:{rsi:.1f}\n"
            f"Stop           : {stop} TL  (+{sl_pct_show:.1f}%)\n"
            f"Hedef          : {target} TL  (-{tp_pct_show:.1f}%  ATR*{ATR_TARGET_MULT})"
        )

    return opened


# =============================================================================
# DETAILED EXIT LOG
# =============================================================================

_EXIT_TR_DETAIL = {
    "stop_loss":     "Stop Kesildi",
    "trailing_stop": "Trailing Stop",
    "target_hit":    "Hedef Tuttu",
    "model_flip":    "Model Sinyal Dustu",
    "news_exit":     "Haber Cikisi",
    "drawdown_kill": "Drawdown Kill",
}

_DETAIL_LOG = Path("results") / "trade_log_detailed.txt"


def _write_detailed_exit_log(
    sym: str, entry: float, giris_saati: str, exit_fill: float,
    cikis_saati: str, reason_tr: str, pnl: float, pct: float,
    hold_minutes: float, prob_e: float, prob_cikis: float, now,
) -> None:
    """Append one line to results/trade_log_detailed.txt on every position close."""
    h = int(hold_minutes // 60)
    m = int(hold_minutes % 60)
    tutulma_str = f"{h}s {m}dk" if h > 0 else f"{m}dk"

    reason_label = _EXIT_TR_DETAIL.get(reason_tr, reason_tr)

    line = (
        f"[{now.strftime('%Y-%m-%d %H:%M')}] SATIS | {sym} | "
        f"Giris: {entry:.2f} @ {giris_saati} | "
        f"Cikis: {exit_fill:.2f} @ {cikis_saati} | "
        f"Sebep: {reason_label} | "
        f"P&L: {pnl:+.0f} TL ({pct:+.2f}%) | "
        f"Tutulma: {tutulma_str} | "
        f"Model prob giris: {prob_e:.2f} | "
        f"Model prob cikis: {prob_cikis:.2f}"
    )
    os.makedirs("results", exist_ok=True)
    with open(_DETAIL_LOG, "a", encoding="utf-8") as _f:
        _f.write(line + "\n")


# =============================================================================
# EXIT MONITORING
# =============================================================================

def check_exits(conn, live_signals: list | None = None) -> int:
    """
    Fetch live prices; close positions that hit any exit condition.

    Exit priority:
      1. News exit   (score <= -2, very negative, last 30 min)
      2. Stop hit    — original OR trailing stop (whichever is higher)
      3. Target hit  (ATR-based)
      4. Model flip  (prob_buy < MODEL_FLIP_THRESHOLD)
      Note: No EOD forced close (Mon-Thu positions stay open overnight).
            No time exit. Friday 17:30 sends a warning only — no forced close.

    Trailing stop rules (in-memory, keyed by symbol):
      - profit >= TRAIL_BE_PCT  (+1.5%): stop moves up to breakeven
      - profit >= TRAIL_START_PCT (+2.0%): stop trails HWM at -TRAIL_DIST_PCT (-1%)
      - Effective stop = max(original stop, trailing stop) — never moves down.

    live_signals: current tick's signal list (for model-flip check).
    Returns number of positions closed.
    """
    try:
        if algolab.mode == "LIVE":
            live_prices = algolab.get_all_prices(SYMBOLS)
        else:
            live_prices, _ = fetch_live_prices(SYMBOLS)
    except Exception as e:
        print(f"  [Exits] Fiyat alinamadi: {e}")
        return 0

    if not live_prices:
        src_tag = "ALGOLAB" if algolab.mode == "LIVE" else "yfinance"
        print(f"  [Exits] Live fiyat alinamadi ({src_tag} bos dondurdu)")
        return 0

    # Build prob lookup from live signals for model-flip check
    prob_map: dict[str, float] = {}
    if live_signals:
        for s in live_signals:
            prob_map[s["symbol"]] = s.get("prob_buy", 0.5)

    open_pos    = get_open_positions(conn)
    today_str   = date.today().isoformat()
    now         = ist_now()
    closed      = 0
    recent_news = read_recent_news(30)   # news items from last 30 minutes

    for pos in open_pos:
        sym = pos["symbol"]
        cur = live_prices.get(sym)
        if cur is None:
            continue

        entry  = pos["entry_price"]
        stop   = pos["stop_price"]
        target = pos["target_price"]

        # -- Parse entry metadata (needed for time-based exit) ----------------
        meta = {}
        try:
            raw = pos.get("entry_note", "")
            if raw and raw.strip().startswith("{"):
                meta = json.loads(raw)
        except Exception:
            pass

        entry_time_str = meta.get("entry_time", f"{pos.get('entry_date', today_str)} 10:00:00")
        try:
            entry_dt   = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            exit_dt    = now.replace(tzinfo=None)
            hold_hours = (exit_dt - entry_dt).total_seconds() / 3600
            tutulma_dk = round(hold_hours * 60, 1)
        except Exception:
            hold_hours = 0.0
            tutulma_dk = 0.0

        is_long = bool(pos.get("is_long", 1))

        if is_long:
            # ── LONG trailing & exit ──────────────────────────────────────────
            profit_pct = (cur - entry) / entry

            hwm = _pos_hwm.get(sym, entry)
            if cur > hwm:
                hwm = cur
                _pos_hwm[sym] = hwm

            if profit_pct >= TRAIL_START_PCT:
                trail_stop = round(hwm * (1 - TRAIL_DIST_PCT), 2)
            elif profit_pct >= TRAIL_BE_PCT:
                trail_stop = entry
            else:
                trail_stop = stop

            eff_stop = max(stop, trail_stop)
            _pos_trailing[sym] = eff_stop

            # News exit: very negative score
            _news_title = ""
            for ni in recent_news:
                if ni.get("symbol") != sym:
                    continue
                _, nscore = score_news(ni["title"], sym)
                if nscore <= -2:
                    _news_title = ni["title"][:50]
                    break

            exit_p = None
            reason = None
            if _news_title:
                exit_p, reason = cur, "news_exit"
            elif cur <= eff_stop:
                reason = "trailing_stop" if eff_stop > stop else "stop_loss"
                exit_p = cur
            elif cur >= target:
                exit_p, reason = cur, "target_hit"
            elif sym in prob_map and prob_map[sym] < MODEL_FLIP_THRESHOLD:
                exit_p, reason = cur, "model_flip"

            if reason is None:
                if eff_stop > stop:
                    trail_tag = "BE" if eff_stop == entry else f"TRAIL HWM:{hwm:.2f}"
                    print(f"  [Trail] {sym}  cur:{cur:.2f}  eff_stop:{eff_stop:.2f}  "
                          f"profit:{profit_pct*100:+.2f}%  [{trail_tag}]")
                continue

        else:
            # ── SHORT trailing & exit ─────────────────────────────────────────
            profit_pct = (entry - cur) / entry  # positive when price falls

            lwm = _pos_lwm.get(sym, entry)
            if cur < lwm:
                lwm = cur
                _pos_lwm[sym] = lwm

            if profit_pct >= TRAIL_START_PCT:
                # Trail: follow LWM up by TRAIL_DIST_PCT
                trail_stop = round(lwm * (1 + TRAIL_DIST_PCT), 2)
            elif profit_pct >= TRAIL_BE_PCT:
                trail_stop = entry  # breakeven stop
            else:
                trail_stop = stop  # original stop (above entry)

            # For SHORT: effective stop = min(orig_stop, trail_stop)
            # Both are above current price; lower = tighter protection
            eff_stop = min(stop, trail_stop)
            _pos_trailing[sym] = eff_stop

            # News exit: very positive score (bad for short)
            _news_title = ""
            for ni in recent_news:
                if ni.get("symbol") != sym:
                    continue
                _, nscore = score_news(ni["title"], sym)
                if nscore >= 2:
                    _news_title = ni["title"][:50]
                    break

            exit_p = None
            reason = None
            if _news_title:
                exit_p, reason = cur, "news_exit"
            elif cur >= eff_stop:
                reason = "trailing_stop" if eff_stop < stop else "stop_loss"
                exit_p = cur
            elif cur <= target:
                exit_p, reason = cur, "target_hit"
            elif sym in prob_map and prob_map[sym] > (1 - MODEL_FLIP_THRESHOLD):
                # prob_buy rising above 0.45 → model no longer bearish
                exit_p, reason = cur, "model_flip"

            if reason is None:
                if eff_stop < stop:
                    trail_tag = "BE" if eff_stop == entry else f"TRAIL LWM:{lwm:.2f}"
                    print(f"  [Trail][SHORT] {sym}  cur:{cur:.2f}  eff_stop:{eff_stop:.2f}  "
                          f"profit:{profit_pct*100:+.2f}%  [{trail_tag}]")
                continue

        # -- Turkish reason labels -------------------------------------------
        reason_tr = {
            "stop_loss":     "Stop-Loss",
            "trailing_stop": "Trailing Stop",
            "target_hit":    "Hedef",
            "model_flip":    "Model Flip",
            "news_exit":     "Haber Cikisi",
        }.get(reason, reason)

        # Exit fill = exact market price (no slippage markdown)
        exit_fill = exit_p

        # OMS: log order lifecycle (exits always pass — no kill switch block)
        oms_dir    = "COVER" if not is_long else "SELL"
        sell_order = oms.create_order(sym, oms_dir, exit_fill, pos["size_tl"], reason=reason_tr)
        oms.fill_order(sell_order, exit_fill)

        pnl, pct = calc_position_pnl(entry, exit_fill, is_long, pos["size_tl"])
        algo_log.log_sell(sym, exit_fill, pos["size_tl"], pnl, pct, reason)
        close_position(
            conn, pos, exit_fill, reason, today_str,
            exit_note=f"loop_trader {reason} @ {exit_fill:.2f}",
        )
        closed += 1

        # Clean up trailing state for this symbol
        _pos_hwm.pop(sym, None)
        _pos_trailing.pop(sym, None)
        _pos_lwm.pop(sym, None)

        giris_saati = entry_time_str[11:16] if len(entry_time_str) >= 16 else "?"
        size_tl     = pos["size_tl"]
        cum_pnl     = get_cumulative_pnl(conn)
        rsi_e       = meta.get("rsi", 0.0)
        prob_e      = meta.get("prob", 0.0)
        prob_cikis  = prob_map.get(sym, 0.0)
        e_reason    = meta.get("reason", "ML BUY")

        # -- Detailed exit log -------------------------------------------------
        try:
            _write_detailed_exit_log(
                sym=sym,
                entry=entry,
                giris_saati=giris_saati,
                exit_fill=exit_fill,
                cikis_saati=now.strftime("%H:%M"),
                reason_tr=reason_tr,
                pnl=pnl,
                pct=pct,
                hold_minutes=tutulma_dk,
                prob_e=prob_e,
                prob_cikis=prob_cikis,
                now=now,
            )
        except Exception as _detail_err:
            pass  # never let logging crash the main loop

        append_trade_log({
            "tarih":                 today_str,
            "saat":                  now.strftime("%H:%M:%S"),
            "sembol":                sym,
            "giris_fiyat":           round(entry, 2),
            "cikis_fiyat":           round(exit_fill, 2),
            "giris_saati":           giris_saati,
            "cikis_saati":           now.strftime("%H:%M"),
            "boyut_tl":              round(size_tl, 2),
            "kar_zarar_tl":          round(pnl, 2),
            "kar_zarar_pct":         round(pct, 2),
            "sebep_al":              e_reason,
            "sebep_sat":             reason_tr,
            "rsi_giris":             rsi_e,
            "model_prob_giris":      prob_e,
            "model_prob_cikis":      prob_cikis,
            "tutulma_dk":            tutulma_dk,
            "kumulatif_pnl":         round(cum_pnl, 2),
            "risk_tl":               meta.get("risk_tl", ""),
            "available_cash_before": meta.get("available_cash_before", ""),
            "cash_reserve_pct":      meta.get("cash_reserve_pct", ""),
            "net_pnl_tl":            round(pnl, 2),
        })

        reason_upper = reason.upper()

        # Extra Telegram alert for news-driven exits
        if reason == "news_exit":
            send_telegram_alert(
                f"[!] <b>{sym} haber nedeniyle kapatildi</b>\n"
                f"Haber: {_news_title}"
            )

        dir_label  = "[SHORT] KAPATILDI" if not is_long else "SATILDI"
        dir_emoji  = "🔴" if not is_long else ("🟩" if pnl >= 0 else "🟥")
        send_telegram_alert(
            f"{dir_emoji} <b>{dir_label}: {sym} @ {exit_fill:.2f}</b>\n"
            f"Giris          : {entry:.2f} TL  ({giris_saati})\n"
            f"Boyut          : {size_tl:,.0f} TL\n"
            f"K/Z            : {pnl:+,.2f} TL  ({pct:+.2f}%)\n"
            f"Tutulma        : {tutulma_dk:.0f} dk\n"
            f"Kumulatif PnL  : {cum_pnl:+,.2f} TL\n"
            f"Sebep          : {reason_tr}"
        )

        pos_tag = "[SHORT]" if not is_long else "[SELL ]"
        print(f"  {pos_tag} {sym} @ {exit_fill:.2f} | "
              f"pnl={pnl:+,.2f} TL ({pct:+.2f}%) | "
              f"reason={reason_upper}  {tutulma_dk:.0f} dk")

    return closed


# =============================================================================
# DAILY SUMMARY  (18:30, once per day)
# =============================================================================

def run_daily_summary(conn):
    global _summary_sent
    today_str = date.today().isoformat()
    if _summary_sent == today_str:
        return

    print(f"\n[Summary] Gunluk ozet olusturuluyor ({today_str})...")

    # Read today's closed trades
    today_trades = []
    if TRADE_LOG.exists():
        with open(TRADE_LOG, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("tarih") == today_str:
                    today_trades.append(row)

    total    = len(today_trades)
    pnl_list = [float(t.get("kar_zarar_tl", 0)) for t in today_trades]
    winners  = sum(1 for p in pnl_list if p > 0)
    losers   = sum(1 for p in pnl_list if p <= 0)
    win_rate = round(winners / total * 100, 1) if total > 0 else 0.0
    day_pnl  = round(sum(pnl_list), 2)
    best     = round(max(pnl_list), 2) if pnl_list else 0.0
    worst    = round(min(pnl_list), 2) if pnl_list else 0.0
    cum_pnl  = round(get_cumulative_pnl(conn), 2)

    equity       = get_current_equity(conn)
    day_pnl_pct  = round(day_pnl / CAPITAL * 100, 2)

    summary_row = {
        "tarih":           today_str,
        "toplam_islem":    total,
        "kazanan":         winners,
        "kaybeden":        losers,
        "win_rate":        win_rate,
        "gunluk_pnl_tl":  day_pnl,
        "gunluk_pnl_pct": day_pnl_pct,
        "kumulatif_pnl":  cum_pnl,
        "en_iyi_islem_tl": best,
        "en_kotu_islem_tl": worst,
    }
    append_daily_summary(summary_row)
    print(f"[Summary] daily_summary.csv'ye eklendi")

    open_pos = get_open_positions(conn)
    cash     = get_cash(conn)
    d_emoji  = "📈" if day_pnl >= 0 else "📉"

    tg = (
        f"{d_emoji} <b>Loop Trader Gunluk Ozet</b>\n"
        f"Tarih          : {today_str}\n"
        f"\n"
        f"Islemler       : {total}  (Kazan:{winners}  Kaybet:{losers}  Oran:{win_rate}%)\n"
        f"Gunluk P&amp;L : {day_pnl:+,.2f} TL  ({day_pnl_pct:+.2f}%)\n"
        f"Kumulatif PnL  : {cum_pnl:+,.2f} TL\n"
        f"En Iyi Islem   : {best:+,.2f} TL\n"
        f"En Kotu Islem  : {worst:+,.2f} TL\n"
        f"\n"
        f"Equity         : {equity:,.0f} TL\n"
        f"Nakit          : {cash:,.0f} TL\n"
        f"Acik Pozisyon  : {len(open_pos)}/{MAX_POS}"
    )
    ok = send_telegram_alert(tg)
    print(f"[Summary] Telegram: {'OK' if ok else 'HATA'}")
    _summary_sent = today_str


# =============================================================================
# STATUS PRINT
# =============================================================================

def print_status(conn, tick: int, n_opened: int, n_closed: int):
    now_str  = ist_now().strftime("%H:%M:%S")
    open_pos = get_open_positions(conn)
    cash     = get_cash(conn)
    equity   = get_current_equity(conn)
    cum_pnl  = get_cumulative_pnl(conn)

    print(f"\n{'='*60}")
    print(f"[{now_str}] Tick #{tick}  |  Acik:{len(open_pos)}/{MAX_POS}")
    print(f"  Nakit   : {cash:>10,.0f} TL")
    print(f"  Equity  : {equity:>10,.0f} TL")
    print(f"  Kum PnL : {cum_pnl:>+10,.2f} TL")
    if open_pos:
        print(f"  Acik Pozisyonlar:")
        for pos in open_pos:
            sym = pos["symbol"]
            lat = get_latest(conn, sym)
            cur = float(lat["close"]) if lat else pos["entry_price"]
            unreal    = (cur - pos["entry_price"]) / pos["entry_price"] * 100
            eff_stop  = _pos_trailing.get(sym, pos["stop_price"])
            stop_prox = (cur - pos["entry_price"]) / max(pos["entry_price"] - eff_stop, 1e-9) * 100
            be_tag    = " [BE]"    if eff_stop == pos["entry_price"] else ""
            trail_tag = " [TRAIL]" if eff_stop > pos["stop_price"] and eff_stop != pos["entry_price"] else ""
            warn      = " *** STOP YAKIN!" if eff_stop >= cur * 0.995 else ""
            print(f"    {sym:<6}  Giris:{pos['entry_price']:.2f}  "
                  f"Simdi:{cur:.2f}  P&L:{unreal:+.2f}%  "
                  f"Stop:{eff_stop:.2f}{be_tag}{trail_tag}  "
                  f"Hedef:{pos['target_price']:.2f}{warn}")
    print(f"  Bu tick: {n_opened} acildi / {n_closed} kapandi")
    print(f"{'='*60}")


# =============================================================================
# DRAWDOWN CIRCUIT BREAKER
# =============================================================================

def _compute_drawdown() -> tuple[float, float, float]:
    """
    Compute current equity, equity peak, and drawdown fraction.
    Opens its own DB connection (safe to call every tick).
    Returns (current_equity, equity_peak, drawdown_pct).
    """
    try:
        ks_conn = sqlite3.connect("trade_data.db")

        # Need at least 3 equity data points before drawdown is meaningful
        eq_rows = ks_conn.execute(
            "SELECT COUNT(*) FROM paper_equity WHERE total_equity > 0"
        ).fetchone()[0]
        if eq_rows < 3:
            ks_conn.close()
            return TOTAL_CAPITAL, TOTAL_CAPITAL, 0.0

        current_eq = get_current_equity_ks(ks_conn)
        peak       = get_equity_peak(ks_conn)

        if current_eq <= 0 or peak <= 0:
            ks_conn.close()
            return TOTAL_CAPITAL, TOTAL_CAPITAL, 0.0
        if current_eq > peak:
            ks_conn.execute(
                "INSERT INTO equity_tracker (equity_peak, updated_at) VALUES (?, ?)",
                (current_eq, datetime.now().isoformat()),
            )
            ks_conn.commit()
            peak = current_eq
        ks_conn.close()
        dd_pct = (peak - current_eq) / peak if peak > 0 else 0.0
        return current_eq, peak, dd_pct
    except Exception:
        return TOTAL_CAPITAL, TOTAL_CAPITAL, 0.0


def check_circuit_breaker(drawdown_pct: float) -> str:
    """
    Map drawdown fraction to circuit-breaker tier.
    Returns one of: "OK" / "WARN" / "HALT" / "KILL"
    """
    if drawdown_pct >= DRAWDOWN_KILL_PCT:
        return "KILL"
    if drawdown_pct >= DRAWDOWN_HALT_PCT:
        return "HALT"
    if drawdown_pct >= DRAWDOWN_WARN_PCT:
        return "WARN"
    return "OK"


def _handle_halt_notify(dd_pct: float):
    """Send HALT Telegram once per drawdown episode; silent on repeats."""
    global _halt_notified
    if not _halt_notified:
        print(f"[HALT] Trading durduruldu -- drawdown {dd_pct:.1%} "
              f"(limit %{DRAWDOWN_HALT_PCT*100:.0f})")
        send_telegram_alert(
            f"TRADING DURDURULDU: Drawdown {dd_pct:.1%} "
            f"(limit %{DRAWDOWN_HALT_PCT*100:.0f})\n"
            f"Sadece cikis yapilacak -- yeni giris yok."
        )
        _halt_notified = True
    else:
        print(f"[HALT] Yeni giris yok -- drawdown {dd_pct:.1%}")


def _handle_kill(conn, dd_pct: float):
    """
    Force-close ALL open positions, write KILL_SWITCH.txt, alert Telegram.
    Called once before the main loop breaks on KILL tier.
    """
    print(f"[KILL] {dd_pct:.1%} drawdown asildi! Tum pozisyonlar kapatiliyor...")
    send_telegram_alert(
        f"KILL SWITCH: Drawdown {dd_pct:.1%} asildi\n"
        f"TUM POZISYONLAR KAPATILIYOR"
    )

    open_pos  = get_open_positions(conn)
    today_str = date.today().isoformat()
    closed    = 0
    cum_pnl   = 0.0

    for pos in open_pos:
        sym = pos["symbol"]
        # Try live price first, fall back to last DB close
        cur = algolab.get_price(sym)
        if cur <= 0:
            lat = get_latest(conn, sym)
            cur = float(lat["close"]) if lat else pos["entry_price"]

        pnl, pct = calc_position_pnl(pos["entry_price"], cur, True, pos["size_tl"])
        close_position(
            conn, pos, cur, "drawdown_kill", today_str,
            exit_note=f"DRAWDOWN_KILL {dd_pct:.1%}",
        )
        cum_pnl += pnl
        closed  += 1
        print(f"  [KILL] {sym} @ {cur:.2f}  pnl={pnl:+.2f} TL")

        # Clean up trailing state
        _pos_hwm.pop(sym, None)
        _pos_trailing.pop(sym, None)

    # Write kill file so kill_switch.py also sees it
    kill_path = Path(__file__).parent / "KILL_SWITCH.txt"
    with open(kill_path, "w") as f:
        f.write(
            f"DRAWDOWN_KILL: {dd_pct:.1%} at {datetime.now().isoformat()}\n"
            f"Auto-created by loop_trader.py circuit breaker."
        )

    print(f"[KILL] {closed} pozisyon kapatildi. Toplam PnL: {cum_pnl:+.2f} TL")
    print("[KILL] KILL_SWITCH.txt olusturuldu. Sistemi yeniden baslatmak icin sil.")
    send_telegram_alert(
        f"KILL COMPLETE: {closed} pozisyon kapatildi\n"
        f"Toplam PnL  : {cum_pnl:+.2f} TL\n"
        f"Drawdown    : {dd_pct:.1%} (limit %{DRAWDOWN_KILL_PCT*100:.0f})\n"
        f"Sistem durdu. KILL_SWITCH.txt silerek yeniden baslat."
    )


# =============================================================================
# P3.2B — PORTFOLIO DAILY STOP
# =============================================================================

def _handle_daily_stop(conn, daily_loss_pct: float) -> None:
    """
    Portfolio daily stop: triggered when today's equity has fallen >= DAILY_STOP_PCT
    (8%) from day-start equity.

    Actions (mirrors _handle_kill):
      1. Close all open positions at current price.
      2. Write KILL_SWITCH.txt with reason "DAILY_STOP_8PCT".
      3. Send Telegram: TAIL RISK EVENT alert.
      4. Log to risk.log.
    """
    msg = (
        f"TAIL RISK EVENT: Daily loss {daily_loss_pct:.1%} >= "
        f"{DAILY_STOP_PCT:.0%} -- tum pozisyonlar kapatiliyor"
    )
    print(f"\n[TAIL] {msg}")
    algo_log.risk(
        f"DAILY_STOP triggered: daily_loss={daily_loss_pct:.1%} "
        f">= {DAILY_STOP_PCT:.0%}"
    )

    open_pos  = get_open_positions(conn)
    today_str = date.today().isoformat()
    closed    = 0
    cum_pnl   = 0.0

    for pos in open_pos:
        sym = pos["symbol"]
        cur = algolab.get_price(sym)
        if cur <= 0:
            lat = get_latest(conn, sym)
            cur = float(lat["close"]) if lat else pos["entry_price"]

        pnl, pct = calc_position_pnl(pos["entry_price"], cur, True, pos["size_tl"])
        close_position(
            conn, pos, cur, "drawdown_kill", today_str,
            exit_note=f"DAILY_STOP {daily_loss_pct:.1%}",
        )
        cum_pnl += pnl
        closed  += 1
        _pos_hwm.pop(sym, None)
        _pos_trailing.pop(sym, None)
        print(f"  [TAIL] {sym} @ {cur:.2f}  pnl={pnl:+.2f} TL")

    kill_path = Path(__file__).parent / "KILL_SWITCH.txt"
    with open(kill_path, "w") as f:
        f.write(
            f"DAILY_STOP_8PCT: daily loss {daily_loss_pct:.1%} "
            f"at {datetime.now().isoformat()}\n"
            f"Auto-created by loop_trader.py P3.2 tail risk protection."
        )

    print(f"[TAIL] {closed} pozisyon kapatildi. PnL: {cum_pnl:+.2f} TL")
    print("[TAIL] KILL_SWITCH.txt olusturuldu. Silerek yeniden baslat.")
    send_telegram_alert(
        f"TAIL RISK EVENT: Daily loss -{daily_loss_pct*100:.1f}% — all positions closed\n"
        f"Kapatilan : {closed} pozisyon  |  PnL: {cum_pnl:+.2f} TL\n"
        f"Sebep     : Gunluk kayip limiti {DAILY_STOP_PCT:.0%} asildi\n"
        f"KILL_SWITCH.txt olusturuldu. Silerek yeniden baslat."
    )


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    global BUY_THRESHOLD   # may be overridden at startup from optimal_threshold.json

    # Load cost-optimised BUY threshold FIRST so the banner shows the live value.
    # Reads results/optimal_threshold.json produced by optimize_threshold.py.
    _opt_thr_path_early = Path(__file__).parent / "results" / "optimal_threshold.json"
    if _opt_thr_path_early.exists():
        try:
            with open(_opt_thr_path_early) as _f:
                _opt_early = json.load(_f)
            BUY_THRESHOLD = float(_opt_early["buy_threshold"])
        except Exception:
            pass  # keep default; detailed error logged below after model load

    print("=" * 60)
    print("BIST LOOP TRADER v2  |  py -3.12 loop_trader.py")
    print(f"Baslangic  : {ist_now().strftime('%Y-%m-%d %H:%M:%S')} Istanbul")
    print(f"Piyasa     : 10:00-18:00  |  Sinyal: 10:30-18:00")
    print(f"Sermaye    : {TOTAL_CAPITAL:,} TL  |  Rezerv:{MIN_CASH_RESERVE:,} TL  "
          f"|  Islem:{TRADEABLE_CAPITAL:,} TL  |  MaxPos:{MAX_POSITION_SIZE:,} TL")
    print(f"Risk/Trade : {RISK_PER_TRADE*100:.1f}% = {TOTAL_CAPITAL*RISK_PER_TRADE:,.0f} TL  "
          f"|  Max Pozisyon: {MAX_POS}  |  BUY prob > {BUY_THRESHOLD}")
    print(f"Sektor     : max {MAX_PER_SECTOR} pozisyon/sektor  "
          f"|  Sektorler: {len(set(SECTOR_MAP.values()))} farkli sektor")
    print(f"Hedef      : ATR*{ATR_TARGET_MULT}  |  SL:{SL_PCT*100:.1f}%  "
          f"|  Trailing: BE@+{TRAIL_BE_PCT*100:.1f}%  Trail@+{TRAIL_START_PCT*100:.1f}% (-{TRAIL_DIST_PCT*100:.1f}%)")
    print(f"Cuma Uyari : {FRIDAY_WARN_TIME[0]:02d}:{FRIDAY_WARN_TIME[1]:02d} hafta sonu uyarisi (kapatma yok)")
    print(f"Dedup      : {DEDUP_HOURS} saat  |  Tick: {TICK_SEC}s")
    if ONE_SHOT:
        print("MODE       : --once (tek iterasyon)")
    print("=" * 60)
    print("EMERGENCY STOP : echo. > KILL_SWITCH.txt")
    print("RESUME         : del KILL_SWITCH.txt")
    print("=" * 60)

    ensure_csvs()

    print("Model yukleniyor...", end=" ", flush=True)
    model, feature_names, calibrator, thresholds = load_model()
    if model is None:
        print("HATA: Model dosyasi bulunamadi. Once ml_train.py calistir.")
        sys.exit(1)
    print("OK")

    # Load Beta calibrator (models/beta_calibrator.pkl) — optional post-calibration layer
    global _beta_cal
    _beta_cal_path = Path(__file__).parent / "models" / "beta_calibrator.pkl"
    if _beta_cal_path.exists():
        try:
            import joblib as _joblib
            _beta_cal = _joblib.load(_beta_cal_path)
            print(f"  [BetaCal] Loaded: {_beta_cal_path.name}")
        except Exception as _e:
            print(f"  [BetaCal] Load failed ({_e}) — running without beta calibration")
            _beta_cal = None
    else:
        print(f"  [BetaCal] {_beta_cal_path.name} not found — running without beta calibration")

    # Log the active BUY threshold (loaded silently before banner, or default 0.65).
    _opt_thr_path = Path(__file__).parent / "results" / "optimal_threshold.json"
    if _opt_thr_path.exists():
        try:
            with open(_opt_thr_path) as _f:
                _opt_data = json.load(_f)
            print(
                f"  [OptThr] Active: {BUY_THRESHOLD:.2f}  "
                f"(precision={_opt_data.get('precision_at_threshold', 0):.3f}, "
                f"trades={_opt_data.get('n_trades', 0)}, "
                f"computed={_opt_data.get('computed_at', 'n/a')})"
            )
        except Exception as _e:
            print(f"  [OptThr] Read failed ({_e}) — using BUY_THRESHOLD={BUY_THRESHOLD}")
    else:
        print(f"  [OptThr] optimal_threshold.json not found — using default BUY_THRESHOLD={BUY_THRESHOLD}")

    # Algolab data source init (MOCK: no-op; LIVE: SMS 2FA)
    algolab.login()
    print(f"[DATA] Veri kaynagi: {algolab.mode}")

    algo_log.system(
        f"Loop Trader v2 started | "
        f"capital={TOTAL_CAPITAL} TL reserve={MIN_CASH_RESERVE} TL "
        f"max_pos={MAX_POS} buy_thr={BUY_THRESHOLD} "
        f"data_src={algolab.mode}"
    )

    conn = get_db()

    # Startup reconciliation (DB sanity checks before trading begins)
    print("[RECON] Baslangic kontrol...")
    _initial_pos = get_open_positions(conn)
    _recon_ok, _recon_issues = recon.run(_initial_pos)
    if not _recon_ok:
        _high = [i for i in _recon_issues if i["severity"] == "HIGH"]
        if _high:
            print(f"[RECON] KRITIK tutarsizlik ({len(_high)} sorun) -- "
                  "devam etmeden once kontrol et!")

    # Kill switch auto-expiry: delete KILL_SWITCH.txt if >24h old and DD recovered
    _check_kill_switch_expiry()

    # HMM regime artifact: refresh once at startup and retrain monthly if stale.
    _refresh_hmm_artifact()
    _refresh_meta_labeler(conn)

    # Startup health check: surfaces every blocking condition before first tick
    _print_health_check(conn, model)

    # Initialise equity peak and circuit-breaker state
    global _equity_peak, _halt_notified, _recon_done_today, _friday_warn_sent
    global _today_start_equity, _today_date_equity
    try:
        _ks_conn     = sqlite3.connect("trade_data.db")
        _equity_peak = get_equity_peak(_ks_conn)
        _ks_conn.close()
    except Exception:
        pass

    # Portfolio heat snapshot for startup message
    try:
        _ph_report = _portfolio_risk.get_risk_report()
        _ph_str    = (
            f"{_ph_report['total_heat_pct']:.2f}%  "
            f"({_ph_report['heat_budget_used_pct']:.1f}% of {MAX_PORTFOLIO_HEAT*100:.0f}% cap)"
        )
        _ph_warn   = "  *** CAP REACHED" if not _ph_report["allowed_new_position"] else ""
        _corr_warn_count = len(_ph_report["corr_warnings"])
        _corr_str  = f"{_corr_warn_count} yuksek korelasyon cifti" if _corr_warn_count > 0 else "OK"
    except Exception:
        _ph_str, _ph_warn, _corr_str = "N/A", "", "N/A"

    # Volatility regime snapshot for startup message
    try:
        _vr_startup = _vol_regime.get_regime_report()
        _regime_icons = {"NORMAL": "", "HIGH_VOL": " ⚠", "EXTREME": " 🚨 TRADING PAUSED"}
        _vr_str = (
            f"{_vr_startup['regime']}{_regime_icons.get(_vr_startup['regime'], '')}  "
            f"(gunluk vol: {_vr_startup['vol_pct']:.2f}%  |  "
            f"boyut carpani: x{_vr_startup['multiplier']:.1f})"
        )
    except Exception:
        _vr_str = "N/A"

    send_telegram_alert(
        f"🤖 <b>Loop Trader v2 Basladi</b>\n"
        f"Tarih          : {date.today().isoformat()}\n"
        f"Saat           : {ist_now().strftime('%H:%M')} Istanbul\n"
        f"Sermaye        : {TOTAL_CAPITAL:,} TL  |  Rezerv: {MIN_CASH_RESERVE:,} TL\n"
        f"Max pos boyutu : {MAX_POSITION_SIZE:,} TL  |  Max pozisyon: {MAX_POS}\n"
        f"Risk/Trade     : {RISK_PER_TRADE*100:.1f}% = {TOTAL_CAPITAL*RISK_PER_TRADE:,.0f} TL\n"
        f"BUY threshold  : {BUY_THRESHOLD}  |  ATR*{ATR_TARGET_MULT}  SL:{SL_PCT*100:.1f}%\n"
        f"Trailing       : BE@+{TRAIL_BE_PCT*100:.1f}%  Trail@+{TRAIL_START_PCT*100:.1f}% (-{TRAIL_DIST_PCT*100:.1f}%)\n"
        f"Cikis          : Stop / Hedef / ModelFlip / Haber  (EOD ve zaman limiti YOK)\n"
        f"Cuma Uyari     : {FRIDAY_WARN_TIME[0]:02d}:{FRIDAY_WARN_TIME[1]:02d} hafta sonu uyarisi (pozisyon kapatilmaz)\n"
        f"Portfolio Heat : {_ph_str}{_ph_warn}\n"
        f"Korelasyon     : {_corr_str}\n"
        f"Vol Rejimi     : {_vr_str}"
    )

    tick = 0
    _last_scan_time: datetime | None = None   # asymmetric scan schedule state
    while True:
        try:
            tick += 1
            n_opened = 0
            n_closed = 0

            # -- Drawdown & circuit breaker (every tick) ----------------------
            current_eq, peak, dd_pct = _compute_drawdown()
            _equity_peak = peak
            cb_status    = check_circuit_breaker(dd_pct)

            # -- P3.2B: Daily start equity tracking (reset each new calendar day)
            _today_key = date.today().isoformat()
            if _today_date_equity != _today_key or _today_start_equity is None:
                _today_start_equity = current_eq
                _today_date_equity  = _today_key

            if in_market():
                sigs: list = []

                # -- Morning reconciliation (once per day at market open) ------
                today_str_r = date.today().isoformat()
                if _recon_done_today != today_str_r:
                    t = ist_now()
                    if _mins(t.hour, t.minute) < _mins(10, 5):
                        print("[RECON] Sabah kontrolu...")
                        _morning_pos = get_open_positions(conn)
                        recon.run(_morning_pos)
                        _recon_done_today = today_str_r

                # -- P3.2B: Portfolio daily stop (8% daily loss) --------------
                if _today_start_equity and _today_start_equity > 0:
                    _daily_loss_pct = (
                        (_today_start_equity - current_eq) / _today_start_equity
                    )
                    if _daily_loss_pct >= DAILY_STOP_PCT:
                        algo_log.risk(
                            f"DAILY_STOP: daily_loss={_daily_loss_pct:.1%} "
                            f">= {DAILY_STOP_PCT:.0%} -- triggering tail risk stop"
                        )
                        _handle_daily_stop(conn, _daily_loss_pct)
                        break   # exits while True

                # -- KILL: %12 drawdown — force-close all, halt loop ----------
                if cb_status == "KILL":
                    algo_log.log_risk_event(
                        "KILL",
                        f"DD {dd_pct:.1%} >= {DRAWDOWN_KILL_PCT:.0%} -- "
                        "force-closing all positions",
                    )
                    _handle_kill(conn, dd_pct)
                    break   # exits while True

                # -- Asymmetric scan schedule (L-shaped BIST intraday vol) ------
                now_ist        = ist_now()
                _scan_interval = get_scan_interval_seconds(now_ist)
                _t_mins        = _mins(now_ist.hour, now_ist.minute)
                _elapsed_sec   = (
                    (now_ist - _last_scan_time).total_seconds()
                    if _last_scan_time is not None else float("inf")
                )
                _due_for_scan  = (
                    _scan_interval is not None and
                    _elapsed_sec >= _scan_interval
                )

                if _due_for_scan:
                    _refresh_hmm_artifact()
                    _refresh_meta_labeler(conn)
                    algo_log.system(
                        f"SCAN interval={_scan_interval}s at {now_ist.strftime('%H:%M')}"
                    )
                    _last_scan_time = now_ist
                    # Fetch live bars + recompute indicators; share with trading
                    sigs = get_live_signals(
                        conn, model, feature_names, calibrator, thresholds
                    )
                    print_debug_lines(sigs)
                    show_signal_report(sigs, now_ist.strftime("%H:%M"))
                    send_signal_telegram(sigs, now_ist)

                    # ── Signal logging to signals_log DB ────────────────────
                    _sig_date_str = now_ist.strftime("%Y-%m-%d %H:%M:%S")
                    _open_before  = {p["symbol"] for p in get_open_positions(conn)}
                    for _sig in sigs:
                        _pb = _sig.get("prob_buy", 0.5)
                        if _pb < DISPLAY_THRESHOLD and (1.0 - _pb) < DISPLAY_THRESHOLD:
                            continue  # below display threshold on both directions
                        _eff_thr = float(_sig.get("effective_threshold", BUY_THRESHOLD))
                        _thr_pass = 1 if (_pb >= _eff_thr or (1.0 - _pb) >= SHORT_THRESHOLD) else 0
                        algo_log.log_signal_db(
                            db_path          = "trade_data.db",
                            symbol           = _sig["symbol"],
                            signal_date      = _sig_date_str,
                            prob_buy         = _pb,
                            prob_sell        = 1.0 - _pb,
                            rsi              = _sig.get("rsi"),
                            atr              = _sig.get("atr"),
                            regime           = _sig.get("hmm_regime"),
                            threshold_passed = _thr_pass,
                        )
                    # ── end signal logging ───────────────────────────────────

                    # Volatility regime — computed here so gate summary has it before
                    # the trade decision, and so OVERRIDE_VOL_BLOCK can adjust _vr_mult.
                    _vr_report  = _vol_regime.get_regime_report()
                    _vr_regime  = _vr_report["regime"]
                    _vr_mult    = _vr_report["multiplier"]

                    # Gate summary — printed every scan tick with live values
                    _n_tradeable_long = sum(
                        1 for s in sigs
                        if s.get("prob_buy", 0) >= s.get("effective_threshold", BUY_THRESHOLD)
                    )
                    _n_tradeable_short = sum(
                        1 for s in sigs
                        if (1.0 - s.get("prob_buy", 0.5)) >= SHORT_THRESHOLD
                        and s.get("rsi", 50) > RSI_SHORT_MIN
                    )
                    _n_tradeable = _n_tradeable_long + _n_tradeable_short
                    _n_open_now = len(get_open_positions(conn))
                    _cash_now   = get_cash(conn)
                    _print_gate_summary(
                        cb_status, dd_pct,
                        _vr_regime, _vr_report["vol_pct"], _vr_mult,
                        _cash_now, _n_open_now, _n_tradeable,
                        n_tradeable_long=_n_tradeable_long,
                        n_tradeable_short=_n_tradeable_short,
                    )

                    # -- P3.2A: Intraday tail risk — override vol regime to EXTREME --
                    _tail_risk, _tail_drop = _check_intraday_tail_risk()
                    if _tail_risk:
                        _tail_msg = (
                            f"[TAIL] Intraday drop -{_tail_drop*100:.1f}% detected "
                            f"({TAIL_RISK_PROXY} last 30min) -> EXTREME override (this tick)"
                        )
                        print(_tail_msg)
                        algo_log.risk(
                            f"TAIL_RISK intraday drop {_tail_drop:.2%} "
                            f">= {TAIL_RISK_DROP_PCT:.0%} -- EXTREME override this tick"
                        )
                        send_telegram_alert(
                            f"TAIL RISK: {TAIL_RISK_PROXY} -{_tail_drop*100:.1f}% "
                            f"last 30min -- entries paused this tick"
                        )
                        _vr_regime = "EXTREME"
                        _vr_mult   = 0.0

                    if cb_status == "HALT":
                        # HALT: exits run below; no new entries
                        _handle_halt_notify(dd_pct)
                        algo_log.log_risk_event(
                            "HALT",
                            f"DD {dd_pct:.1%} >= {DRAWDOWN_HALT_PCT:.0%} -- no new entries",
                        )
                    else:
                        if _vr_regime == "EXTREME" and not OVERRIDE_VOL_BLOCK:
                            print(f"  [VOL  ] EXTREME VOL ({_vr_report['vol_pct']:.2f}%) "
                                  f"-- trading paused (OVERRIDE_VOL_BLOCK=True ile x0.4 boyutta islem yap)")
                            algo_log.log_risk_event(
                                "HALT",
                                f"EXTREME VOL {_vr_report['vol_pct']:.2f}% "
                                f">= {EXTREME_VOL_THRESHOLD*100:.1f}% -- no new entries",
                            )
                        else:
                            # EXTREME + override=True, or HIGH_VOL, or NORMAL
                            if _vr_regime == "EXTREME":
                                # OVERRIDE_VOL_BLOCK is True — trade at reduced size
                                _vr_mult = 0.4
                                print(f"  [VOL  ] EXTREME VOL ({_vr_report['vol_pct']:.2f}%) "
                                      f"-- OVERRIDE_VOL_BLOCK=True, x{_vr_mult:.1f} boyut")
                                algo_log.log_risk_event(
                                    "WARN",
                                    f"EXTREME VOL {_vr_report['vol_pct']:.2f}% OVERRIDE "
                                    f"-- size x{_vr_mult:.1f}",
                                )

                            # OK or WARN: allow new entries (WARN = 50% size cap)
                            max_override = (
                                MAX_POSITION_SIZE * 0.5 if cb_status == "WARN" else None
                            )
                            if cb_status == "WARN":
                                print(f"  [WARN] Pozisyon limiti {max_override:.0f} TL "
                                      f"-- DD:{dd_pct:.1%}")
                                algo_log.log_risk_event(
                                    "WARN",
                                    f"DD {dd_pct:.1%} >= {DRAWDOWN_WARN_PCT:.0%} -- "
                                    f"position cap {max_override:.0f} TL",
                                )
                            if _vr_regime == "HIGH_VOL":
                                print(f"  [VOL  ] HIGH_VOL ({_vr_report['vol_pct']:.2f}%) "
                                      f"-- pozisyon boyutu x{_vr_mult:.1f}")
                                algo_log.log_risk_event(
                                    "WARN",
                                    f"HIGH_VOL {_vr_report['vol_pct']:.2f}% "
                                    f">= {HIGH_VOL_THRESHOLD*100:.1f}% -- "
                                    f"size multiplier x{_vr_mult:.1f}",
                                )
                            n_opened = scan_and_trade(
                                conn, model, feature_names, calibrator, thresholds,
                                _signals=sigs, max_pos_override=max_override,
                                vol_multiplier=_vr_mult,
                            )
                            # Mark trade_opened=1 for newly opened positions
                            if n_opened > 0:
                                try:
                                    _open_after  = {p["symbol"] for p in get_open_positions(conn)}
                                    _new_symbols = _open_after - _open_before
                                    if _new_symbols:
                                        _upd_conn = sqlite3.connect("trade_data.db", timeout=10)
                                        for _new_sym in _new_symbols:
                                            _upd_conn.execute(
                                                """UPDATE signals_log
                                                   SET trade_opened=1
                                                   WHERE symbol=? AND date(signal_date)=date(?)
                                                     AND trade_opened=0
                                                   ORDER BY id DESC LIMIT 1""",
                                                (_new_sym, _sig_date_str),
                                            )
                                        _upd_conn.commit()
                                        _upd_conn.close()
                                except Exception as _ue:
                                    algo_log.debug(f"signal trade_opened update WARN: {_ue}")
                        # Reset HALT flag once we're fully back to OK/WARN
                        if _halt_notified and cb_status in ("OK", "WARN"):
                            _halt_notified = False
                            print("[CB] Drawdown normale dondu -- ticaret yeniden basliyor")
                elif _scan_interval is None and _t_mins >= SKIP_UNTIL_MIN:
                    # After 17:30 — no new entries; exits still active (handled below)
                    print(f"  [{now_ist.strftime('%H:%M')}] Giris penceresi kapali "
                          f"(17:30+) -- cikislar aktif")
                elif _scan_interval is None:
                    # Before 10:30 — opening volatility window
                    print(f"  Acilis penceresi ({now_ist.strftime('%H:%M')}) -- "
                          f"sinyal taramasi 10:30'da baslayacak")
                else:
                    # Interval not yet elapsed — show countdown
                    wait = max(0, int(_scan_interval - _elapsed_sec))
                    print(f"  [{now_ist.strftime('%H:%M')}] Sonraki tarama: ~{wait}s "
                          f"(aralik={_scan_interval}s)")

                # -- Exit check (always runs, including HALT mode) ------------
                if ONE_SHOT or tick % EXIT_EVERY_N == 0:
                    n_closed = check_exits(conn, live_signals=sigs if sigs else None)

                # -- Friday 17:30 weekend warning (once per day, no forced close)
                if is_friday_eod_warning():
                    send_friday_weekend_warnings(conn)

            else:
                # -- Outside market hours ------------------------------------
                if should_eod_refresh():
                    run_eod_refresh()

                if is_summary_window():
                    run_daily_summary(conn)

                t = ist_now()
                print(f"[{t.strftime('%H:%M:%S')}] Piyasa kapali "
                      f"({t.strftime('%H:%M')}) -- bekleniyor...")

            # -- Risk display (always shown, every tick) ----------------------
            dd_n   = min(int(dd_pct * 100), 12)
            dd_bar = "#" * dd_n + "." * (12 - dd_n)
            print(f"[RISK] Equity:{current_eq:.0f}  Peak:{peak:.0f}  "
                  f"DD:{dd_pct:.1%}  [{dd_bar}]  Status:{cb_status}")
            algo_log.debug(
                f"LOOP tick={tick} equity={current_eq:.0f} "
                f"peak={peak:.0f} dd={dd_pct:.2%} cb={cb_status}"
            )

            print_status(conn, tick, n_opened, n_closed)
            oms.summary()

        except KeyboardInterrupt:
            print("\n[Loop] Ctrl+C — durduruluyor...")
            send_telegram_alert("🛑 <b>Loop Trader durduruldu</b> (Ctrl+C)")
            break
        except Exception as e:
            import traceback
            print(f"[HATA] {e}")
            traceback.print_exc()

        if ONE_SHOT:
            print("\n[--once] Tamamlandi. Cikiliyor.")
            break

        time.sleep(TICK_SEC)

    conn.close()
    print("Cikis tamam.")


if __name__ == "__main__":
    main()
