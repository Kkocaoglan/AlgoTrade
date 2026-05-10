"""
crypto_stream.py — Bybit connection and data fetching for Crypto Module.

Completely isolated from spot (loop_trader) and VIOP systems.
Uses ccxt unified API. PAPER_MODE=True → Bybit Testnet, no real orders.

CryptoWebSocket: real-time price feed via Bybit v5 public linear stream.
  - Uses public production WS endpoint (market data, no auth required)
  - Subscribes to tickers.<SYMBOL> topics after connection
  - Auto-reconnect up to 5 times on disconnect
  - Thread-safe shared dict _live_prices updated in background thread
"""

import json
import threading
import time
from datetime import datetime, timezone

import os

import ccxt
import numpy as np
import pandas as pd
import requests
import websocket  # websocket-client library
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PAPER_MODE = os.getenv("PAPER_MODE", "True").lower() != "false"

RATE_LIMIT_SLEEP = 2   # seconds between calls when throttling manually
RETRY_SLEEP      = 60  # seconds to wait after 429 before retry
PAGINATION_PAUSE = 0.05
FUNDING_SUPPORTED_SYMBOLS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "AVAX/USDT", "DOT/USDT"}
_BTC_DOMINANCE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
    "history": [],  # list[(date_str, value)]
}

_TIMEFRAME_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


def fetch_funding_rate(symbol: str) -> dict:
    """Fetch 8h perpetual funding rate from Bybit v5 API."""
    if symbol not in FUNDING_SUPPORTED_SYMBOLS:
        return {"funding_rate": 0.0, "funding_rate_avg3": 0.0}
    try:
        bybit_symbol = symbol.replace("/", "")
        url = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={bybit_symbol}&limit=3"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        result = response.json()
        rates = [float(r["fundingRate"]) for r in result["result"]["list"]]
        if not rates:
            return {"funding_rate": 0.0, "funding_rate_avg3": 0.0}
        return {
            "funding_rate": float(rates[0]),  # Bybit returns newest first
            "funding_rate_avg3": float(np.mean(rates)),
        }
    except Exception as exc:
        print(f"[CRYPTO ERROR] fetch_funding_rate({symbol}): {exc}")
        return {"funding_rate": 0.0, "funding_rate_avg3": 0.0}


def fetch_funding_rate_history(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch recent 8h Bybit perpetual funding history for training alignment."""
    if symbol not in FUNDING_SUPPORTED_SYMBOLS:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_rate_avg3", "mark_price"])
    try:
        bybit_symbol = symbol.replace("/", "")
        url = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={bybit_symbol}&limit={int(limit)}"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        result = response.json()
        data = result["result"]["list"]
        if not data:
            return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_rate_avg3", "mark_price"])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["fundingRateTimestamp"], errors="coerce"), unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["mark_price"] = 0.0  # not included in Bybit funding history endpoint
        df = df.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp").reset_index(drop=True)
        df["funding_rate_avg3"] = df["funding_rate"].rolling(3, min_periods=1).mean()
        return df[["timestamp", "funding_rate", "funding_rate_avg3", "mark_price"]]
    except Exception as exc:
        print(f"[CRYPTO ERROR] fetch_funding_rate_history({symbol}): {exc}")
        return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_rate_avg3", "mark_price"])


def fetch_btc_dominance(force_refresh: bool = False) -> float:
    """Fetch BTC market cap dominance from CoinGecko (cached 30m)."""
    now = time.time()
    cached = _BTC_DOMINANCE_CACHE.get("value")
    if not force_refresh and cached is not None and (now - float(_BTC_DOMINANCE_CACHE.get("fetched_at", 0.0))) < 1800:
        return float(cached)
    try:
        url = "https://api.coingecko.com/api/v3/global"
        data = requests.get(url, timeout=10).json()
        btc_d = float(data["data"]["market_cap_percentage"]["btc"])
        _BTC_DOMINANCE_CACHE["value"] = btc_d
        _BTC_DOMINANCE_CACHE["fetched_at"] = now
        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        history = list(_BTC_DOMINANCE_CACHE.get("history", []))
        if not history or history[-1][0] != day_key:
            history.append((day_key, btc_d))
        else:
            history[-1] = (day_key, btc_d)
        _BTC_DOMINANCE_CACHE["history"] = history[-14:]
        return btc_d
    except Exception as exc:
        print(f"[CRYPTO ERROR] fetch_btc_dominance: {exc}")
        return float(cached or 0.0)


def get_btc_dominance_features(force_refresh: bool = False) -> dict:
    """Return cached BTC dominance level, regime, and simple 7d change."""
    value = float(fetch_btc_dominance(force_refresh=force_refresh))
    if value < 45.0:
        regime = "altseason"
        regime_code = 0.0
    elif value < 55.0:
        regime = "neutral"
        regime_code = 1.0
    elif value < 65.0:
        regime = "btc_lead"
        regime_code = 2.0
    else:
        regime = "btc_dominant"
        regime_code = 3.0

    history = list(_BTC_DOMINANCE_CACHE.get("history", []))
    change_7d = 0.0
    if len(history) >= 2:
        latest_day = history[-1][0]
        anchor_idx = max(0, len(history) - 7)
        anchor_value = float(history[anchor_idx][1])
        change_7d = value - anchor_value
        _ = latest_day

    return {
        "btc_dominance": value,
        "btc_dominance_regime": regime,
        "btc_dominance_regime_code": regime_code,
        "btc_dominance_7d_change": float(change_7d),
        "history_len": len(history),
    }


class CryptoStream:
    """
    Wraps ccxt.bybit for OHLCV, ticker, order placement, and balance queries.
    All public methods log errors with [CRYPTO] prefix and return None on failure.
    """

    def __init__(self):
        self.exchange = ccxt.bybit({
            "apiKey": os.getenv("BYBIT_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET"),
            "enableRateLimit": True,        # ccxt auto-throttles between calls
            "options": {"defaultType": "spot"},
        })
        self.market_exchange = ccxt.bybit({
            "apiKey": os.getenv("BYBIT_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET"),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if PAPER_MODE:
            self.exchange.set_sandbox_mode(True)

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame | None:
        """
        Fetch OHLCV candles from Bybit.

        Returns DataFrame with columns: timestamp, open, high, low, close, volume.
        Returns None on any error.
        """
        try:
            raw = self._call(self.market_exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
            if raw is None:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as exc:
            print(f"[CRYPTO ERROR] get_ohlcv({symbol}): {exc}")
            return None

    def get_ohlcv_paginated(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int | None = None,
        limit_per_call: int = 1000,
        max_batches: int = 500,
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV history in batches using exchange pagination.

        Returns a deduplicated DataFrame with columns:
        timestamp, open, high, low, close, volume
        """
        tf_ms = _TIMEFRAME_MS.get(timeframe)
        if tf_ms is None:
            raise ValueError(f"Unsupported timeframe for paginated fetch: {timeframe}")

        all_rows: list[list] = []
        cursor = int(since_ms)
        batch = 0
        expected_batches = None
        if until_ms is not None and until_ms > since_ms:
            expected_candles = max(1, ((int(until_ms) - int(since_ms)) // tf_ms) + 1)
            expected_batches = max(1, (expected_candles + limit_per_call - 1) // limit_per_call)

        while batch < max_batches:
            raw = self._call(
                self.market_exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since=cursor,
                limit=limit_per_call,
            )
            if not raw:
                break

            all_rows.extend(raw)
            batch += 1
            last_ts = int(raw[-1][0])
            next_cursor = last_ts + tf_ms

            if expected_batches is not None and (batch == 1 or batch % 10 == 0):
                print(
                    f"    {symbol} {timeframe}: batch {batch}/{expected_batches} "
                    f"| last={datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}",
                    flush=True,
                )

            if until_ms is not None and last_ts >= until_ms:
                break
            if next_cursor <= cursor:
                break
            if len(raw) < limit_per_call:
                break

            cursor = next_cursor
            time.sleep(PAGINATION_PAUSE)

        if not all_rows:
            return None

        df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if until_ms is not None:
            until_dt = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)
            df = df[df["timestamp"] <= until_dt]
        return df.reset_index(drop=True)

    def get_ticker(self, symbol: str) -> dict | None:
        """
        Fetch current price, bid, ask for symbol.

        Returns dict with keys: symbol, last, bid, ask.
        Returns None on error.
        """
        try:
            t = self._call(self.market_exchange.fetch_ticker, symbol)
            if t is None:
                return None
            return {
                "symbol": symbol,
                "last": t.get("last"),
                "bid":  t.get("bid"),
                "ask":  t.get("ask"),
            }
        except Exception as exc:
            print(f"[CRYPTO ERROR] get_ticker({symbol}): {exc}")
            return None

    def place_order(self, symbol: str, side: str, amount_usdt: float, price: float | None = None) -> dict:
        """
        Place a market order (or simulate in paper mode).

        side: 'buy' or 'sell'
        amount_usdt: nominal size in USDT; converted to coin amount using price.
        price: required for paper mode fill simulation; optional for live (ignored, market order).

        Returns order dict with at minimum:
            order_id, symbol, side, amount_usdt, amount_coin, price, status, paper_mode
        """
        if price is None or price <= 0:
            # attempt to get live price for sizing
            ticker = self.get_ticker(symbol)
            price = ticker["last"] if ticker else None
        if price is None:
            return {"status": "REJECTED", "reason": "price unavailable"}

        amount_coin = round(amount_usdt / price, 8)

        if PAPER_MODE:
            order = {
                "order_id":    f"PAPER-{int(time.time()*1000)}",
                "symbol":      symbol,
                "side":        side,
                "amount_usdt": amount_usdt,
                "amount_coin": amount_coin,
                "price":       price,
                "status":      "FILLED",
                "paper_mode":  True,
            }
            print(f"[PAPER] {side.upper()} {symbol} | {amount_coin:.6f} @ {price} | {amount_usdt:.2f} USDT")
            return order

        # Live order
        try:
            raw = self._call(self.exchange.create_order, symbol, "market", side, amount_coin)
            if raw is None:
                return {"status": "REJECTED", "reason": "ccxt returned None"}
            return {
                "order_id":      raw.get("id"),
                "symbol":        symbol,
                "side":          side,
                "amount_usdt":   amount_usdt,
                "amount_coin":   amount_coin,
                "price":         raw.get("average") or price,
                "status":        "FILLED",
                "paper_mode":    False,
                "ccxt_order_id": raw.get("id"),
            }
        except Exception as exc:
            print(f"[CRYPTO ERROR] place_order({symbol}, {side}): {exc}")
            return {"status": "REJECTED", "reason": str(exc)}

    def get_balance(self) -> dict:
        """
        Return balance dict.
        In PAPER_MODE: queries crypto_positions table via CryptoOMS helper.
        In live mode: calls ccxt fetch_balance.
        """
        if PAPER_MODE:
            # Deferred import to avoid circular; crypto_oms handles DB
            try:
                from crypto_oms import CryptoOMS
                oms = CryptoOMS()
                invested = sum(p["amount_usdt"] for p in oms.get_open_positions())
                return {"USDT": {"free": max(0, 1000 - invested), "total": 1000}}
            except Exception as exc:
                print(f"[CRYPTO ERROR] get_balance (paper): {exc}")
                return {"USDT": {"free": 1000, "total": 1000}}

        try:
            raw = self._call(self.exchange.fetch_balance)
            if raw is None:
                return {}
            return raw.get("total", {})
        except Exception as exc:
            print(f"[CRYPTO ERROR] get_balance: {exc}")
            return {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, fn, *args, **kwargs):
        """
        Wrapper: call ccxt function, handle 429 with one retry, log errors.
        Returns function result or None on failure.
        """
        try:
            return fn(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            print(f"[CRYPTO] 429 rate limit — sleeping {RETRY_SLEEP}s then retrying")
            time.sleep(RETRY_SLEEP)
            try:
                return fn(*args, **kwargs)
            except Exception as exc2:
                print(f"[CRYPTO ERROR] retry failed: {exc2}")
                return None
        except ccxt.NetworkError as exc:
            print(f"[CRYPTO ERROR] NetworkError: {exc}")
            return None
        except ccxt.ExchangeError as exc:
            print(f"[CRYPTO ERROR] ExchangeError: {exc}")
            return None


# ── WebSocket real-time price feed ───────────────────────────────────────────

class CryptoWebSocket:
    """
    Bybit v5 public linear WebSocket for 10-coin real-time prices.

    Uses public production endpoint (market data, no API key required).
    Subscribes to tickers.<SYMBOL> topics after connection opens.
    Runs in a background daemon thread; auto-reconnects up to MAX_RETRIES times.

    Usage:
        ws = CryptoWebSocket()
        ws.start()                # launch background thread
        price = ws.get_price("BTC/USDT")  # None if not connected yet
        alive  = ws.is_alive()    # True if updated within last 10 seconds
        ws.stop()                 # clean shutdown
    """

    WS_URL = "wss://stream.bybit.com/v5/public/linear"

    # Topics to subscribe after connection
    SUBSCRIBE_TOPICS = [
        "tickers.BTCUSDT",
        "tickers.ETHUSDT",
        "tickers.BNBUSDT",
        "tickers.SOLUSDT",
        "tickers.AVAXUSDT",
        "tickers.SUIUSDT",
        "tickers.DOTUSDT",
        "tickers.HYPEUSDT",
        "tickers.ONDOUSDT",
        "tickers.FETUSDT",
    ]
    # Map Bybit topic names → module symbol format
    STREAM_MAP = {
        "tickers.BTCUSDT":  "BTC/USDT",
        "tickers.ETHUSDT":  "ETH/USDT",
        "tickers.BNBUSDT":  "BNB/USDT",
        "tickers.SOLUSDT":  "SOL/USDT",
        "tickers.AVAXUSDT": "AVAX/USDT",
        "tickers.SUIUSDT":  "SUI/USDT",
        "tickers.DOTUSDT":  "DOT/USDT",
        "tickers.HYPEUSDT": "HYPE/USDT",
        "tickers.ONDOUSDT": "ONDO/USDT",
        "tickers.FETUSDT":  "FET/USDT",
    }
    MAX_RETRIES   = 5
    RECONNECT_SEC = 5
    ALIVE_SEC     = 10   # price is "live" if updated within this many seconds

    def __init__(self) -> None:
        self._live_prices: dict[str, dict] = {
            sym: {"price": 0.0, "change_pct": 0.0, "updated_at": None}
            for sym in self.STREAM_MAP.values()
        }
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._ws          = None
        self._thread: threading.Thread | None = None
        self._connected   = False
        self._price_callback = None  # callable(symbol: str, price: float) | None

    def set_price_callback(self, fn) -> None:
        """Register a callback invoked on every WS price tick.

        The callback is called as fn(symbol, price) from the WS thread.
        It must be fast, non-blocking, and must never raise.
        """
        self._price_callback = fn

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch WebSocket in a background daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CryptoWS")
        self._thread.start()

    def stop(self) -> None:
        """Signal the WS thread to stop and close the connection."""
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_price(self, symbol: str) -> float | None:
        """Return latest live price for symbol. None if not yet received."""
        with self._lock:
            data = self._live_prices.get(symbol, {})
        price = data.get("price", 0.0)
        return price if price > 0 else None

    def get_price_age(self, symbol: str) -> float:
        """Seconds since the last update for a symbol. inf if never updated."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        with self._lock:
            data = self._live_prices.get(symbol, {})
        updated_at = data.get("updated_at")
        if updated_at is None:
            return float("inf")
        return (now - updated_at).total_seconds()

    def get_all_prices(self) -> dict[str, float]:
        """Return {symbol: price} for all 3 pairs (0.0 if not yet received)."""
        with self._lock:
            return {sym: d["price"] for sym, d in self._live_prices.items()}

    def is_alive(self) -> bool:
        """True if at least one price was updated within ALIVE_SEC seconds."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        with self._lock:
            for data in self._live_prices.values():
                upd = data.get("updated_at")
                if upd and (now - upd).total_seconds() < self.ALIVE_SEC:
                    return True
        return False

    def last_update_age(self) -> float:
        """Seconds since last price update (across any symbol). inf if never."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        best = float("inf")
        with self._lock:
            for data in self._live_prices.values():
                upd = data.get("updated_at")
                if upd:
                    best = min(best, (now - upd).total_seconds())
        return best

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main reconnect loop — runs in the background thread."""
        retry = 0
        while retry <= self.MAX_RETRIES and not self._stop_event.is_set():
            if retry > 0:
                print(f"[CRYPTO WS ERROR] Baglanti koptu, yeniden baglanıyor... ({retry}/{self.MAX_RETRIES})")
                time.sleep(self.RECONNECT_SEC)

            try:
                self._ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # run_forever blocks until disconnect
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                print(f"[CRYPTO WS ERROR] run_forever exception: {exc}")

            if self._stop_event.is_set():
                break
            retry += 1

        if retry > self.MAX_RETRIES:
            print("[CRYPTO WS ERROR] Maksimum yeniden baglanti denemesi asıldı — REST fallback")

    def _on_open(self, ws) -> None:
        self._connected = True
        sub_msg = json.dumps({"op": "subscribe", "args": self.SUBSCRIBE_TOPICS})
        ws.send(sub_msg)
        print("[CRYPTO WS] Baglandi — Bybit 10-coin stream aktif (BTC/ETH/BNB/SOL/AVAX/SUI/DOT/HYPE/ONDO/FET)")

    def _on_close(self, ws, status_code, close_msg) -> None:
        self._connected = False

    def _on_error(self, ws, error) -> None:
        print(f"[CRYPTO WS ERROR] {error}")

    def _on_message(self, ws, raw: str) -> None:
        """Parse Bybit v5 ticker message and update _live_prices."""
        try:
            from datetime import datetime, timezone
            msg = json.loads(raw)

            # Ignore subscription confirmations and pong frames
            if "op" in msg or "ret_msg" in msg:
                return

            topic  = msg.get("topic", "")
            symbol = self.STREAM_MAP.get(topic)
            if symbol is None:
                return

            data       = msg.get("data", {})
            price      = float(data.get("lastPrice", 0))
            # Bybit returns price24hPcnt as decimal (0.0123 = 1.23%)
            change_pct = float(data.get("price24hPcnt", 0)) * 100
            now        = datetime.now(timezone.utc)

            with self._lock:
                self._live_prices[symbol] = {
                    "price":      price,
                    "change_pct": change_pct,
                    "updated_at": now,
                }
            cb = self._price_callback
            if cb is not None:
                try:
                    cb(symbol, price)
                except Exception:
                    pass  # never crash the WS thread on callback error
        except Exception:
            pass  # never crash the WS thread on bad message


# ── Symbol availability check ────────────────────────────────────────────────

def check_symbol_on_exchange(symbol: str) -> bool:
    """Check if symbol is listed on Bybit spot. Returns True on error (fail-open)."""
    try:
        exch = ccxt.bybit()
        exch.load_markets()
        return symbol in exch.markets
    except Exception as exc:
        print(f"[CRYPTO] {symbol} varlik kontrolu hatası: {exc} — universte bırakıldı")
        return True


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[CRYPTO] crypto_stream.py standalone test başlıyor...")
    stream = CryptoStream()

    # REST: BTC ticker
    ticker = stream.get_ticker("BTC/USDT")
    if ticker:
        print(f"[CRYPTO] REST baglantisi OK | BTC: ${ticker['last']:,.2f}")
    else:
        print("[CRYPTO ERROR] BTC ticker alinamadi")

    # REST: 5 candles
    df = stream.get_ohlcv("BTC/USDT", timeframe="1h", limit=5)
    if df is not None:
        print(f"[CRYPTO] BTC/USDT son 5 1h mum:")
        print(df[["timestamp", "close", "volume"]].to_string(index=False))

    # WebSocket test
    print("\n[CRYPTO] WebSocket baglantisi test ediliyor (5 saniye)...")
    ws = CryptoWebSocket()
    ws.start()
    time.sleep(5)

    if ws.is_alive():
        prices = ws.get_all_prices()
        print(f"[CRYPTO WS] Fiyat: BTC={prices.get('BTC/USDT', 0):,.2f}  "
              f"ETH={prices.get('ETH/USDT', 0):,.2f}  "
              f"SOL={prices.get('SOL/USDT', 0):.2f}")
        print(f"[CRYPTO WS] Son guncelleme: {ws.last_update_age():.1f}s önce")
    else:
        print("[CRYPTO WS] Canli veri alınamadi (testnet veya ag sorunu)")

    ws.stop()
    print("[CRYPTO] Test tamamlandi.")
