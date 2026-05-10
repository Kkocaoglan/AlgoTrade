"""
crypto_indicators.py — In-memory indicator computation for Crypto Module.

All single-timeframe functions accept a DataFrame and return an enriched DataFrame.
No DB reads or writes — computed fresh each tick.

Multi-timeframe support via compute_mtf(stream, symbol):
  Fetches 5m / 15m / 1h candles and scores them into (long_score, short_score).
  (Updated 2026-04-24: was 15m/1h/4h → now 5m/15m/1h for faster signal response)

Completely isolated from spot (indicators.py) and VIOP (viop_indicators.py).
"""

import pandas as pd
import numpy as np


# ── EMA ───────────────────────────────────────────────────────────────────────

def add_ema(df: pd.DataFrame, periods: list[int] = [8, 21, 50]) -> pd.DataFrame:
    """Add ema{p} columns for each period in periods."""
    df = df.copy()
    for p in periods:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add rsi14 column (Wilder smoothing via ewm)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ──────────────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Add macd_line, macd_signal, macd_hist, macd_bull_cross, macd_bear_cross.
    Bull/bear cross = macd_line crossed signal on the last completed bar.
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_line"]   = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]
    prev_line   = df["macd_line"].shift(1)
    prev_signal = df["macd_signal"].shift(1)
    df["macd_bull_cross"] = (df["macd_line"] > df["macd_signal"]) & (prev_line <= prev_signal)
    df["macd_bear_cross"] = (df["macd_line"] < df["macd_signal"]) & (prev_line >= prev_signal)
    return df


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def add_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Add bb_upper, bb_mid, bb_lower, bb_width columns."""
    df = df.copy()
    df["bb_mid"]   = df["close"].rolling(period).mean()
    rolling_std    = df["close"].rolling(period).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + std_dev * rolling_std
    df["bb_lower"] = df["bb_mid"] - std_dev * rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    return df


# ── ATR ───────────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add atr14 column (Wilder ATR)."""
    df = df.copy()
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


# ── Volume Ratio ──────────────────────────────────────────────────────────────

def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add vol_ratio column: current volume / rolling 20-period average.
    Values > 1.5 indicate above-average volume surge.
    """
    df = df.copy()
    avg_vol = df["volume"].rolling(window).mean()
    df["vol_ratio"] = df["volume"] / avg_vol.replace(0, np.nan)
    return df


def add_funding_features(df: pd.DataFrame, funding_info: dict | None = None) -> pd.DataFrame:
    """Add funding-rate-derived columns as scalar features."""
    df = df.copy()
    funding_info = funding_info or {}
    funding_rate = float(funding_info.get("funding_rate", 0.0) or 0.0)
    funding_rate_avg3 = float(funding_info.get("funding_rate_avg3", funding_rate) or funding_rate)
    df["funding_rate"] = funding_rate
    df["funding_rate_avg3"] = funding_rate_avg3
    df["funding_rate_extreme"] = float(abs(funding_rate) > 0.001)
    df["funding_contrarian"] = float(-np.sign(funding_rate))
    return df


def compute_htf_bias(stream, symbol: str) -> dict:
    """4h timeframe trend bias — gates all entries."""
    try:
        exch = getattr(stream, "market_exchange", None) or stream.exchange
        ohlcv_4h = exch.fetch_ohlcv(symbol, "4h", limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=["ts", "open", "high", "low", "close", "volume"])
        if df_4h.empty:
            return {"htf_bias": "BEAR", "htf_close": 0.0, "htf_ema50": 0.0}
        for col in ["open", "high", "low", "close", "volume"]:
            df_4h[col] = pd.to_numeric(df_4h[col], errors="coerce")
        df_4h["ema50_4h"] = df_4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
        latest_close = float(df_4h["close"].iloc[-1])
        ema50 = float(df_4h["ema50_4h"].iloc[-1]) if pd.notna(df_4h["ema50_4h"].iloc[-1]) else latest_close
        bias = "BULL" if latest_close > ema50 else "BEAR"
        return {"htf_bias": bias, "htf_close": latest_close, "htf_ema50": ema50}
    except Exception:
        return {"htf_bias": "BEAR", "htf_close": 0.0, "htf_ema50": 0.0}


def build_htf_bias_history_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Approximate historical 4h bias from 1h candles for training alignment."""
    if df_1h is None or df_1h.empty:
        return pd.DataFrame(columns=["timestamp", "htf_bias"])
    tmp = df_1h[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    tmp = tmp.sort_values("timestamp").set_index("timestamp")
    ohlc_4h = tmp.resample("4h", label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    if ohlc_4h.empty:
        return pd.DataFrame(columns=["timestamp", "htf_bias"])
    ohlc_4h["ema50_4h"] = ohlc_4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
    ohlc_4h["htf_bias"] = (ohlc_4h["close"] > ohlc_4h["ema50_4h"]).astype(float)
    return ohlc_4h.reset_index()[["timestamp", "htf_bias"]]


# ── All-in-one (1h baseline) ──────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators in sequence. Returns enriched DataFrame."""
    df = add_ema(df, periods=[8, 21, 50])
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_ratio(df)
    return df


# ── Multi-Timeframe Signal Fusion ─────────────────────────────────────────────

def compute_mtf(stream, symbol: str) -> tuple[int, int]:
    """
    Fetch 5m / 15m / 1h candles and compute a fused MTF signal score.

    long_score and short_score are computed INDEPENDENTLY — both can be non-zero
    at the same time.  short_score is NOT (6 - long_score).

    MTF_LONG_SCORE (max 6):
      +2  1h  close > EMA21  (intermediate trend)
      +2  15m close > EMA21  (momentum confirmation)
      +1  5m  RSI < 42       (oversold dip — long entry timing)
      +1  5m  MACD bull_cross (bullish momentum cross)

    MTF_SHORT_SCORE (max 6):
      +2  1h  close < EMA21  (intermediate downtrend)
      +2  15m close < EMA21  (momentum confirmation)
      +1  5m  RSI > 58       (overbought — short entry timing)
      +1  5m  MACD bear_cross (bearish momentum cross)

    Returns (long_score, short_score), each capped at 6.

    Bar freshness: if the last bar for a timeframe is too old (after WS reconnect
    or stale feed), that timeframe contributes 0 and logs a STALE warning.
      5m  → STALE if last bar older than 10 minutes
      15m → STALE if last bar older than 20 minutes
      1h  → STALE if last bar older than 90 minutes
    """
    from datetime import datetime, timezone as _tz
    try:
        from crypto_logger import crypto_log as _clog
    except Exception:
        _clog = None

    def _bar_age_sec(df: pd.DataFrame) -> float:
        """Seconds since last bar's timestamp (bar open time)."""
        try:
            ts = df["timestamp"].iloc[-1]
            if ts is None:
                return float("inf")
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=_tz.utc)
            return (datetime.now(_tz.utc) - ts).total_seconds()
        except Exception:
            return float("inf")

    long_score  = 0
    short_score = 0

    # ── 1h timeframe ── intermediate trend (weight 2 each, mutually exclusive) ─
    df_1h = stream.get_ohlcv(symbol, timeframe="1h", limit=30)
    if df_1h is not None and len(df_1h) >= 22:
        age_1h = _bar_age_sec(df_1h)
        if age_1h > 5400:  # 90 minutes
            msg = f"[STALE] {symbol} 1h bar age={age_1h/60:.1f}m > 90m — 1h MTF contributes 0"
            print(msg)
            if _clog:
                _clog.scan(msg)
        else:
            df_1h = add_ema(df_1h, periods=[21])
            last_1h = df_1h.iloc[-1]
            if last_1h["close"] > last_1h["ema21"]:
                long_score  += 2
            elif last_1h["close"] < last_1h["ema21"]:
                short_score += 2

    # ── 15m timeframe ── momentum (weight 2 each, mutually exclusive) ─────────
    df_15m = stream.get_ohlcv(symbol, timeframe="15m", limit=50)
    if df_15m is not None and len(df_15m) >= 22:
        age_15m = _bar_age_sec(df_15m)
        if age_15m > 1200:  # 20 minutes
            msg = f"[STALE] {symbol} 15m bar age={age_15m/60:.1f}m > 20m — 15m MTF contributes 0"
            print(msg)
            if _clog:
                _clog.scan(msg)
        else:
            df_15m = add_ema(df_15m, periods=[21])
            last_15m = df_15m.iloc[-1]
            if last_15m["close"] > last_15m["ema21"]:
                long_score  += 2
            elif last_15m["close"] < last_15m["ema21"]:
                short_score += 2

    # ── 5m timeframe ── entry timing (weight 1 each, INDEPENDENT) ─────────────
    df_5m = stream.get_ohlcv(symbol, timeframe="5m", limit=60)
    if df_5m is not None and len(df_5m) >= 30:
        age_5m = _bar_age_sec(df_5m)
        if age_5m > 600:  # 10 minutes
            msg = f"[STALE] {symbol} 5m bar age={age_5m/60:.1f}m > 10m — 5m MTF contributes 0"
            print(msg)
            if _clog:
                _clog.scan(msg)
        else:
            df_5m = add_rsi(df_5m)
            df_5m = add_macd(df_5m)
            last_5m = df_5m.iloc[-1]

            rsi_5m = float(last_5m.get("rsi14", 50))
            if rsi_5m < 42:
                long_score  += 1   # oversold → long opportunity
            if rsi_5m > 58:
                short_score += 1   # overbought → short opportunity

            bull_cross = bool(last_5m.get("macd_bull_cross", False))
            bear_cross = bool(last_5m.get("macd_bear_cross", False))
            if bull_cross:
                long_score  += 1   # bullish momentum cross
            if bear_cross:
                short_score += 1   # bearish momentum cross

    return min(long_score, 6), min(short_score, 6)


def get_mtf_dataframes(stream, symbol: str) -> dict:
    """
    Fetch 5m / 15m / 1h candles and return as a dict keyed by timeframe.
    Values are DataFrames with all indicators computed, or None on failure.
    Used by crypto_ml.py for feature extraction.
    """
    result = {}
    for tf, limit in [("5m", 60), ("15m", 30), ("1h", 60)]:
        df = stream.get_ohlcv(symbol, timeframe=tf, limit=limit)
        if df is not None and len(df) >= 22:
            df = compute_all(df)
        result[tf] = df
    return result


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[CRYPTO] crypto_indicators.py standalone test baslıyor...")

    from crypto_stream import CryptoStream
    stream = CryptoStream()
    df_raw = stream.get_ohlcv("BTC/USDT", limit=60)
    if df_raw is None:
        print("[CRYPTO ERROR] Veri alinamadi — test iptal")
    else:
        df = compute_all(df_raw)
        last = df.iloc[-1]
        print("[CRYPTO] BTC/USDT 1h son bar indikatörleri:")
        print(f"  Close:      {last['close']:.2f}")
        print(f"  EMA8/21/50: {last['ema8']:.2f} / {last['ema21']:.2f} / {last['ema50']:.2f}")
        print(f"  RSI14:      {last['rsi14']:.1f}")
        print(f"  MACD line:  {last['macd_line']:.4f}  hist: {last['macd_hist']:.4f}")
        print(f"  BB:         {last['bb_lower']:.2f} — {last['bb_upper']:.2f}")
        print(f"  ATR14:      {last['atr14']:.2f}")
        print(f"  Vol ratio:  {last['vol_ratio']:.2f}x")

        print("\n[CRYPTO] MTF skor hesaplanıyor (BTC/USDT) — 15m/1h/4h...")
        ls, ss = compute_mtf(stream, "BTC/USDT")
        print(f"  MTF LONG score:  {ls}/6")
        print(f"  MTF SHORT score: {ss}/6")
        print("[CRYPTO] Indikatör testi tamamlandi.")
