"""
crypto_status.py — Live dashboard for Crypto Module (10-coin tiered universe).

Usage:
  python3.12 crypto_status.py

Completely isolated from spot and VIOP systems.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from crypto_stream     import CryptoStream, CryptoWebSocket, check_symbol_on_exchange
from crypto_indicators import compute_all, compute_mtf
from crypto_oms        import CryptoOMS
from crypto_config import (
    COIN_GROUP_POLICIES,
    CRYPTO_BTC_CRASH_THRESHOLD,
    CRYPTO_CAPITAL_USDT,
    CRYPTO_MAJOR,
    CRYPTO_MAX_POSITIONS,
    CRYPTO_MTF_THRESHOLD_ML,
    CRYPTO_MTF_THRESHOLD_RB,
    CRYPTO_RISKY,
    CRYPTO_SYMBOLS,
    LEGACY_THRESHOLD_CONFIG,
    RISKY_MTF_THRESHOLD,
)

try:
    from crypto_sentiment import sentiment as _fg_sentiment
    from crypto_sentiment import get_avg_funding_rate
except Exception:
    class _DummySentiment:
        def get_fear_greed(self): return {"value": 50, "label": "Neutral", "timestamp": ""}
        def get_signal_modifier(self, v): return 0.0, 0.0
    def get_avg_funding_rate(): return 0.0
    _fg_sentiment = _DummySentiment()

try:
    from crypto_ml import load_model, load_model_long, load_tier_model, predict_proba
    # Prefer new directional LONG artifact; fall back to legacy single model
    _ml_dict = load_model_long() or load_model()
except Exception:
    _ml_dict = None

# ── Universe & tier config ───────────────────────────────────────────────────
# Imported from crypto_config.py so the dashboard mirrors crypto_trader.py.
CRYPTO_ML_THRESHOLD = LEGACY_THRESHOLD_CONFIG["MAJOR"]["fallback_threshold"]
RISKY_ML_THRESHOLD = LEGACY_THRESHOLD_CONFIG["RISKY"]["fallback_threshold"]
CRYPTO_MTF_THRESH_ML = CRYPTO_MTF_THRESHOLD_ML
CRYPTO_MTF_THRESH_RB = CRYPTO_MTF_THRESHOLD_RB
CRYPTO_BTC_CRASH = CRYPTO_BTC_CRASH_THRESHOLD

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
MAJORAFTER_STATE_PATH = RESULTS_DIR / "majorafter_state.json"
TZ_IST = ZoneInfo("Europe/Istanbul")


def get_tier(symbol: str) -> str:
    return "RISKY" if symbol in CRYPTO_RISKY else "MAJOR"

def get_coin_group(symbol: str) -> str:
    for group_name, policy in COIN_GROUP_POLICIES.items():
        if symbol in policy["symbols"]:
            return group_name
    return "RISKY" if get_tier(symbol) == "RISKY" else "VOL_MAJOR"

def get_trade_policy(symbol: str) -> dict:
    return COIN_GROUP_POLICIES[get_coin_group(symbol)]

def _model_threshold_for_tier(tier: str) -> float | None:
    try:
        tier_model = load_tier_model("long", tier)
        for payload in (tier_model, _ml_dict):
            if not payload:
                continue
            cfg = payload.get("threshold_config", {}).get(tier, {})
            threshold = cfg.get("selected_threshold") or cfg.get("fallback_threshold")
            if threshold is not None:
                return float(threshold)
    except Exception:
        return None
    return None

def get_ml_threshold(symbol: str) -> float:
    tier = get_tier(symbol)
    model_threshold = _model_threshold_for_tier(tier)
    if model_threshold is not None:
        return model_threshold
    return RISKY_ML_THRESHOLD if tier == "RISKY" else CRYPTO_ML_THRESHOLD

def get_mtf_threshold(symbol: str, ml_on: bool) -> int:
    if get_tier(symbol) == "RISKY":
        return RISKY_MTF_THRESHOLD
    return CRYPTO_MTF_THRESH_ML if ml_on else CRYPTO_MTF_THRESH_RB


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_pnl(pnl: float) -> str:
    return f"{pnl:+.2f}"

def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"

def _fmt_countdown(seconds: float) -> str:
    if seconds <= 0:
        return "simdi"
    if seconds < 60:
        return f"{seconds:.0f}s sonra"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s sonra"

def _fmt_hold(entry_date_str: str) -> str:
    try:
        entry_dt = datetime.fromisoformat(entry_date_str).replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - entry_dt).total_seconds())
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h{m:02d}m" if h > 0 else f"{m}m"
    except Exception:
        return "?"


def _parse_iso_dt(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fmt_local_ts(date_str: str | None) -> str:
    dt = _parse_iso_dt(date_str)
    if dt is None:
        return "?"
    return dt.astimezone(TZ_IST).strftime("%Y-%m-%d %H:%M")


def _fmt_trade_span(entry_date_str: str | None, exit_date_str: str | None) -> str:
    entry_dt = _parse_iso_dt(entry_date_str)
    exit_dt = _parse_iso_dt(exit_date_str)
    if entry_dt is None:
        return "?"
    if exit_dt is None:
        return _fmt_local_ts(entry_date_str)
    hold_secs = max((exit_dt - entry_dt).total_seconds(), 0.0)
    hold_h = int(hold_secs // 3600)
    hold_m = int((hold_secs % 3600) // 60)
    hold_txt = f"{hold_h}h{hold_m:02d}m" if hold_h > 0 else f"{hold_m}m"
    return f"{_fmt_local_ts(entry_date_str)} -> {_fmt_local_ts(exit_date_str)} | sure={hold_txt}"


def _is_same_local_day(date_str: str | None, day_str: str) -> bool:
    dt = _parse_iso_dt(date_str)
    if dt is None:
        return False
    return dt.astimezone(TZ_IST).strftime("%Y-%m-%d") == day_str


def _load_majorafter_state(day_str: str) -> dict:
    if not MAJORAFTER_STATE_PATH.exists():
        return {}
    try:
        state = json.loads(MAJORAFTER_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if state.get("date") != day_str:
        return {}
    return state


# ── Verdict helper (tier-aware) ───────────────────────────────────────────────

def _verdict(symbol: str, adj_long: float, adj_short: float,
             ml_prob: float, ml_on: bool) -> str:
    ml_thr  = get_ml_threshold(symbol)
    mtf_thr = get_mtf_threshold(symbol, ml_on)
    long_ok  = adj_long  >= mtf_thr and (not ml_on or ml_prob >= ml_thr)
    short_ok = adj_short >= mtf_thr and (not ml_on or (1 - ml_prob) >= ml_thr)
    risky    = get_tier(symbol) == "RISKY"
    thr_note = f" (thr>={ml_thr:.2f}+mtf>={mtf_thr})" if risky else ""
    if long_ok:
        return f"-> LONG candidate{thr_note} [OK]"
    if short_ok:
        return f"-> SHORT candidate{thr_note} [OK]"
    return f"-> WAIT{thr_note}"


# ── WebSocket snapshot (4-second connect) ────────────────────────────────────

def _get_ws_prices() -> tuple[dict[str, float], float, bool]:
    """Connect WS briefly, collect prices for all coins. Returns (prices, age_s, alive)."""
    ws = CryptoWebSocket()
    ws.start()
    time.sleep(4)
    alive  = ws.is_alive()
    prices = ws.get_all_prices() if alive else {}
    age    = ws.last_update_age() if alive else float("inf")
    ws.stop()
    return prices, age, alive


# ── Scan timing ───────────────────────────────────────────────────────────────

def _scan_timing() -> str:
    state_path = BASE_DIR / "logs" / "crypto_scan_state.json"
    try:
        data = json.loads(state_path.read_text())
        now  = time.time()
        age  = now - data.get("last_scan_ts", 0)
        rem  = data.get("next_scan_ts", 0) - now
        return f"Son tarama: {_fmt_age(age)} once | Sonraki: {_fmt_countdown(rem)}"
    except Exception:
        return "Son tarama: bilinmiyor (crypto_trader.py calismiyor)"


def _scan_state() -> dict:
    state_path = BASE_DIR / "logs" / "crypto_scan_state.json"
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def _position_exit_preview(pos: dict, cur: float, pnl: float) -> tuple[float, str]:
    entry = float(pos["entry_price"])
    stop = float(pos["stop_price"])
    target = float(pos["target_price"])
    amount_usdt = float(pos["amount_usdt"] or 0.0)
    direction = str(pos["direction"])
    policy = get_trade_policy(str(pos["symbol"]))
    profit_pct = (pnl / amount_usdt) if amount_usdt else 0.0
    trail_trigger = float(policy["trail_trigger"])
    trail_pct = float(policy["trail_pct"])
    partial_tp_pct = float(policy["partial_take_profit_pct"])
    partial_done = int(pos.get("partial_exit_done", 0) or 0) == 1

    if direction == "long":
        partial_price = entry * (1 + partial_tp_pct)
        trail_arm_price = entry * (1 + trail_trigger)
        trail_now = cur * (1 - trail_pct)
        if profit_pct >= trail_trigger:
            return trail_now, "trail"
        if profit_pct < 0:
            return stop, "stop"
        if not partial_done:
            return partial_price, "partial"
        return target, "target"

    else:
        partial_price = entry * (1 - partial_tp_pct)
        trail_arm_price = entry * (1 - trail_trigger)
        trail_now = cur * (1 + trail_pct)
        if profit_pct >= trail_trigger:
            return trail_now, "trail"
        if profit_pct < 0:
            return stop, "stop"
        if not partial_done:
            return partial_price, "partial"
        if cur <= trail_arm_price:
            return trail_now, "trail"
        return target, "target"


def _tier_model_status_lines() -> list[str]:
    lines: list[str] = []
    for tier in ("MAJOR", "RISKY"):
        long_model = load_tier_model("long", tier)
        short_model = load_tier_model("short", tier)
        long_txt = f"L={float(long_model.get('wf_precision', 0.0)):.1%}" if long_model else "L=missing"
        short_txt = f"S={float(short_model.get('wf_precision', 0.0)):.1%}" if short_model else "S=missing"
        lines.append(f"[MODEL] {tier}: {long_txt} | {short_txt}")
    return lines


# ── BTC regime display ────────────────────────────────────────────────────────

def _compute_btc_regime_display(stream) -> str:
    """Compute and format BTC two-layer regime for dashboard."""
    try:
        df = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=210)
        if df is None or len(df) < 201:
            return "UNKNOWN (yetersiz veri)"
        close = df["close"].astype(float)
        ema200 = close.ewm(span=200, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema200 = float(ema200.iloc[-1])
        prev_close = float(close.iloc[-2])

        structural = "BULL" if last_close > last_ema200 else "BEAR"
        ret_1h = (last_close - prev_close) / prev_close if prev_close > 0 else 0.0
        if ret_1h > 0.01:
            momentum = "UP"
        elif ret_1h < -0.01:
            momentum = "DOWN"
        else:
            momentum = "NEUTRAL"

        # Combined
        if momentum == "NEUTRAL":
            combined = "NEUTRAL"
        elif structural == "BULL" and momentum == "UP":
            combined = "BULL"
        elif structural == "BULL" and momentum == "DOWN":
            combined = "CAUTION(long x0.7)"
        elif structural == "BEAR" and momentum == "DOWN":
            combined = "BEAR"
        elif structural == "BEAR" and momentum == "UP":
            combined = "CAUTION(short x0.7)"
        else:
            combined = "NEUTRAL"

        ema_dist = (last_close - last_ema200) / last_ema200 * 100
        return (
            f"{combined} | struct={structural} mom={momentum} | "
            f"BTC={last_close:,.0f} EMA200={last_ema200:,.0f} ({ema_dist:+.1f}%) | "
            f"ret_1h={ret_1h*100:+.2f}%"
        )
    except Exception as exc:
        return f"ERROR ({exc})"


# ── Bar freshness check ───────────────────────────────────────────────────────

# Thresholds in seconds (2× bar duration)
_FRESHNESS_THRESHOLDS = {
    "5m":  600,   # 10 minutes
    "15m": 1200,  # 20 minutes
    "1h":  5400,  # 90 minutes
}

def _bar_freshness(stream, symbol: str) -> str:
    """Return compact freshness string for 5m/15m/1h bars.

    Format: '5m:OK 15m:OK 1h:OK' — STALE shown with age if too old.
    Fetches only 3 bars per timeframe (minimal REST cost).
    """
    parts = []
    for tf, limit in [("5m", 3), ("15m", 3), ("1h", 3)]:
        stale_sec = _FRESHNESS_THRESHOLDS[tf]
        try:
            df = stream.get_ohlcv(symbol, timeframe=tf, limit=limit)
            if df is None or df.empty:
                parts.append(f"{tf}:NO_DATA")
                continue
            ts = df["timestamp"].iloc[-1]
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
            age_m = age_s / 60.0
            if age_s > stale_sec:
                parts.append(f"{tf}:STALE({age_m:.0f}m)")
            else:
                parts.append(f"{tf}:OK")
        except Exception:
            parts.append(f"{tf}:ERR")
    return " ".join(parts)


# ── Main dashboard ────────────────────────────────────────────────────────────

def main() -> None:
    stream = CryptoStream()
    oms    = CryptoOMS()
    now_ist = datetime.now(TZ_IST)
    today_ist = now_ist.strftime("%Y-%m-%d")
    now_ts = now_ist.strftime("%Y-%m-%d %H:%M:%S TSİ")
    ml_on  = _ml_dict is not None

    # WS prices (brief connect)
    ws_prices, ws_age, ws_alive = _get_ws_prices()

    # Positions
    positions   = oms.get_open_positions()
    major_pos   = [p for p in positions if get_tier(p["symbol"]) == "MAJOR"]
    risky_pos   = [p for p in positions if get_tier(p["symbol"]) == "RISKY"]
    cum_pnl     = oms.get_cumulative_pnl()
    today_closed = oms.get_today_closed("Europe/Istanbul")
    today_pnl   = sum(p["pnl_usdt"] for p in today_closed if p["pnl_usdt"] is not None)
    open_total_pnl = 0.0
    today_opened = []
    for pos in positions:
        cur = ws_prices.get(pos["symbol"]) or (stream.get_ticker(pos["symbol"]) or {}).get("last") or pos["entry_price"]
        pnl = oms.calc_pnl(pos, cur)
        open_total_pnl += pnl
        if _is_same_local_day(pos.get("entry_date"), today_ist):
            today_opened.append((pos, cur, pnl))
    eq_now = CRYPTO_CAPITAL_USDT + cum_pnl + open_total_pnl
    majorafter = _load_majorafter_state(today_ist)

    # BTC gate
    df_btc = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=3)
    btc_ret = 0.0
    btc_gate = "PASS"
    if df_btc is not None and len(df_btc) >= 2:
        prev = df_btc["close"].iloc[-2]
        last = df_btc["close"].iloc[-1]
        btc_ret = (last - prev) / prev if prev > 0 else 0.0
        if btc_ret < CRYPTO_BTC_CRASH:
            btc_gate = f"BLOCK (crash {btc_ret*100:.2f}%)"

    # Fear & Greed
    fg       = _fg_sentiment.get_fear_greed()
    fg_val   = fg["value"]
    fg_label = fg["label"]
    fg_lmod, fg_smod = _fg_sentiment.get_signal_modifier(fg_val)

    # ── Print dashboard ───────────────────────────────────────────────────────
    SEP = "=" * 52

    print()
    print(SEP)
    print(f"CRYPTO TRADER STATUS — {now_ts}")
    print(SEP)

    # Capital
    eq_pct = cum_pnl / CRYPTO_CAPITAL_USDT * 100 if CRYPTO_CAPITAL_USDT else 0
    print(f"[CAPITAL]   {CRYPTO_CAPITAL_USDT:.0f} USDT | "
          f"Cumulative P&L: {_fmt_pnl(cum_pnl)} USDT ({eq_pct:+.1f}%) | "
          f"Today: {_fmt_pnl(today_pnl)} USDT")
    print(f"[EQUITY]    Simdi: {eq_now:.2f} USDT | Acik P&L: {_fmt_pnl(open_total_pnl)} USDT")

    if majorafter:
        start_eq = float(majorafter.get("start_equity") or CRYPTO_CAPITAL_USDT)
        target_eq = float(majorafter.get("target_equity") or start_eq)
        if majorafter.get("locked"):
            locked_eq = float(majorafter.get("locked_equity") or 0.0)
            shadow_closed = majorafter.get("shadow_closed_positions", [])
            shadow_open = majorafter.get("shadow_open_positions", [])
            shadow_total = sum(float(pos.get("pnl_usdt") or 0.0) for pos in shadow_closed)
            print(f"[MAJORAFTER] AKTIF | lock_eq={locked_eq:.2f} | shadow_closed={len(shadow_closed)} | "
                  f"shadow_open={len(shadow_open)} | shadow_pnl={shadow_total:+.2f} USDT")
        else:
            target_gap = max(target_eq - start_eq, 1e-9)
            progress = (eq_now - start_eq) / target_gap * 100.0
            print(f"[MAJORAFTER] Pasif | Start={start_eq:.2f} | Target={target_eq:.2f} (+1.5%) | "
                  f"Ilerleme={progress:.1f}%")

    # Position counts
    print(f"[POSITIONS] Open: {len(positions)}/{CRYPTO_MAX_POSITIONS} "
          f"(Major: {len(major_pos)}/4, Risky: {len(risky_pos)}/1)")

    # Major positions
    print()
    print("MAJOR POZISYONLAR:")
    if not major_pos:
        print("  (bos)")
    for pos in major_pos:
        sym  = pos["symbol"]
        cur  = ws_prices.get(sym) or (stream.get_ticker(sym) or {}).get("last") or pos["entry_price"]
        pnl  = oms.calc_pnl(pos, cur)
        pct  = pnl / pos["amount_usdt"] * 100 if pos["amount_usdt"] else 0
        side = "LONG " if pos["direction"] == "long" else "SHORT"
        hold = _fmt_hold(pos["entry_date"])
        sign = "+" if pnl >= 0 else ""
        ml_s = ""
        if ml_on:
            try:
                df1h = stream.get_ohlcv(sym, "1h", 60)
                if df1h is not None and len(df1h) >= 51:
                    df1h = compute_all(df1h)
                    pr = predict_proba(_ml_dict, df1h)
                    ml_s = f" | ml={pr:.2f}"
            except Exception:
                pass
        print(f"  {sym} {side} @ {pos['entry_price']:,.4f} | "
              f"Poz: {float(pos['amount_usdt']):.0f} USDT | "
              f"Live: {cur:,.4f} | {sign}{pnl:.2f} USDT ({sign}{pct:.1f}%) | "
              f"Hold: {hold}{ml_s}")

    # Risky positions
    print()
    print("RISKI POZISYONLAR:")
    if not risky_pos:
        print("  (bos)")
    for pos in risky_pos:
        sym  = pos["symbol"]
        cur  = ws_prices.get(sym) or (stream.get_ticker(sym) or {}).get("last") or pos["entry_price"]
        pnl  = oms.calc_pnl(pos, cur)
        pct  = pnl / pos["amount_usdt"] * 100 if pos["amount_usdt"] else 0
        side = "LONG " if pos["direction"] == "long" else "SHORT"
        hold = _fmt_hold(pos["entry_date"])
        sign = "+" if pnl >= 0 else ""
        print(f"  {sym} {side} @ {pos['entry_price']:,.4f} | "
              f"Poz: {float(pos['amount_usdt']):.0f} USDT | "
              f"Live: {cur:,.4f} | {sign}{pnl:.2f} USDT ({sign}{pct:.1f}%) | Hold: {hold}")

    print()
    print(f"GUNLUK ISLEM OZETI ({today_ist} TSİ):")
    if not today_closed and not today_opened:
        print("  bugun acilan/kapanan islem yok")
    else:
        for pos in sorted(today_closed, key=lambda item: item.get("exit_date") or ""):
            pnl = float(pos.get("pnl_usdt") or 0.0)
            pct = pnl / float(pos["amount_usdt"]) * 100 if float(pos["amount_usdt"]) else 0.0
            side = "LONG " if pos["direction"] == "long" else "SHORT"
            sign = "+" if pnl >= 0 else ""
            print(f"  KAPANDI | {pos['symbol']} {side} | {sign}{pnl:.2f} USDT ({sign}{pct:.1f}%) | "
                  f"{_fmt_trade_span(pos.get('entry_date'), pos.get('exit_date'))} | reason={pos.get('exit_reason') or '?'}")
        for pos, cur, pnl in sorted(today_opened, key=lambda item: item[0].get("entry_date") or ""):
            pct = pnl / float(pos["amount_usdt"]) * 100 if float(pos["amount_usdt"]) else 0.0
            side = "LONG " if pos["direction"] == "long" else "SHORT"
            sign = "+" if pnl >= 0 else ""
            print(f"  ACIK    | {pos['symbol']} {side} | {sign}{pnl:.2f} USDT ({sign}{pct:.1f}%) | "
                  f"poz={float(pos['amount_usdt']):.0f} USDT | "
                  f"alis={_fmt_local_ts(pos.get('entry_date'))} | live={cur:,.4f}")

    # Signal scan (fresh — filter symbols not listed on exchange, then scan all available)
    available = [s for s in CRYPTO_SYMBOLS if check_symbol_on_exchange(s)]
    unavailable = [s for s in CRYPTO_SYMBOLS if s not in available]

    print()
    print("SON TARAMA SINYALLERI:")
    for symbol in available:
        tier     = get_tier(symbol)

        mtf_long, mtf_short = compute_mtf(stream, symbol)
        adj_long  = mtf_long  + fg_lmod
        adj_short = mtf_short + fg_smod

        ml_prob = 0.5
        if ml_on:
            try:
                df1h = stream.get_ohlcv(symbol, "1h", 60)
                if df1h is not None and len(df1h) >= 51:
                    df1h = compute_all(df1h)
                    ml_prob = predict_proba(_ml_dict, df1h)
            except Exception:
                pass

        v = _verdict(symbol, adj_long, adj_short, ml_prob, ml_on)
        sym_short = symbol.split("/")[0].ljust(5)
        freshness = _bar_freshness(stream, symbol)
        print(f"  [{tier[:5]:5s}] {sym_short} ml={ml_prob:.2f} mtf={mtf_long}/{mtf_short} | fresh={freshness} | {v}")

    for symbol in unavailable:
        tier = get_tier(symbol)
        sym_short = symbol.split("/")[0].ljust(5)
        print(f"  [{tier[:5]:5s}] {sym_short} [borsa listesinde yok — atlanıyor]")

    # WS status
    print()
    ws_str = f"CANLI | {ws_age:.1f}s once" if ws_alive else "KOPUK"
    price_parts = []
    for sym in ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]:
        p = ws_prices.get(sym, 0)
        if p > 0:
            price_parts.append(f"{sym.split('/')[0]}={p:,.0f}")
    print(f"[WS]     {ws_str} | {' '.join(price_parts)}")

    # F&G, BTC gate, scan timing
    fg_bias = ""
    if fg_val <= 24:
        fg_bias = " | LONG bias aktif"
    elif fg_val >= 75:
        fg_bias = " | SHORT bias aktif"
    print(f"[F&G]    {fg_val} — {fg_label}{fg_bias}")
    print(f"[BTC GATE] {btc_gate} | BTC 1h: {btc_ret*100:+.2f}%")
    state = _scan_state()
    if state:
        print(
            f"[CORR]   avg={float(state.get('avg_corr', 0.0) or 0.0):.2f} | "
            f"thr={float(state.get('corr_threshold', 0.85) or 0.85):.2f} | "
            f"state={state.get('corr_market_state', 'normal')}"
        )
        print(
            f"[FUNDING] avg={float(state.get('avg_funding', 0.0) or 0.0):+.4%} | "
            f"bias={state.get('funding_bias', 'neutral')}"
        )
    else:
        print(f"[FUNDING] avg={float(get_avg_funding_rate() or 0.0):+.4%}")

    # BTC two-layer regime
    btc_regime_str = _compute_btc_regime_display(stream)
    print(f"[BTC REGIME] {btc_regime_str}")

    print(f"[SIGNALS] {_scan_timing()}")
    for line in _tier_model_status_lines():
        print(line)

    # ── Açık Pozisyonlar Özet ─────────────────────────────────────────────────
    print()
    print("ACIK POZISYONLAR:")
    if not positions:
        print("  pozisyon yok")
    else:
        for pos in positions[:10]:
            sym  = pos["symbol"]
            cur  = ws_prices.get(sym) or (stream.get_ticker(sym) or {}).get("last") or pos["entry_price"]
            pnl  = oms.calc_pnl(pos, cur)
            pct  = pnl / pos["amount_usdt"] * 100 if pos["amount_usdt"] else 0
            side = "L" if pos["direction"] == "long" else "S"
            sign = "+" if pnl >= 0 else ""
            exit_price, exit_reason = _position_exit_preview(pos, float(cur), float(pnl))
            print(
                f"  {sym.split('/')[0]:<6} {side}  poz={float(pos['amount_usdt']):.0f} USDT  "
                f"pnl={sign}{pnl:.2f} USDT  {sign}{pct:.1f}% | "
                f"entry={float(pos['entry_price']):,.4f} | "
                f"live={float(cur):,.4f} | tahmini_kapanis={exit_price:,.4f} ({exit_reason})"
            )

    print(SEP)


if __name__ == "__main__":
    main()
