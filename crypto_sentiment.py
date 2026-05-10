"""
crypto_sentiment.py — Fear & Greed Index for Crypto Module.

Fetches from api.alternative.me/fng (free, no API key).
Caches result for 1 hour — safe to call every tick.
Provides contrarian signal modifier: buy fear, sell greed.

Completely isolated from spot and VIOP systems.
"""

import json
import time
import urllib.request
from datetime import datetime, timezone

import requests

try:
    from logger import algo_log
except Exception:
    algo_log = None

try:
    from telegram_bot import send_telegram_alert
except Exception:
    def send_telegram_alert(msg, **kw):  # type: ignore[no-redef]
        pass

FNG_URL          = "https://api.alternative.me/fng/?limit=1"
FNG_HISTORY_URL  = "https://api.alternative.me/fng/?limit=365&format=json"
CACHE_TTL        = 3600   # seconds (1 hour)
FETCH_TIMEOUT    = 10     # seconds per HTTP request
_NEUTRAL_DEFAULT = {"value": 50, "label": "Neutral", "timestamp": ""}
FUNDING_URL      = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_SYMBOLS  = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


class CryptoSentiment:
    """
    Fear & Greed Index wrapper with 1-hour cache.

    Usage:
        from crypto_sentiment import sentiment  # module singleton
        fg = sentiment.get_fear_greed()         # {value, label, timestamp}
        long_mod, short_mod = sentiment.get_signal_modifier(fg['value'])
    """

    def __init__(self) -> None:
        self._cache: dict | None = None
        self._cache_time: float   = 0.0
        self._funding_cache: dict[str, float] | None = None
        self._funding_cache_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_fear_greed(self) -> dict:
        """
        Return current Fear & Greed index.
        Cached for 1 hour. Returns Neutral(50) on any error.

        Return dict:
          value (int)      : 0 (Extreme Fear) … 100 (Extreme Greed)
          label (str)      : 'Extreme Fear' | 'Fear' | 'Neutral' | 'Greed' | 'Extreme Greed'
          timestamp (str)  : ISO-like epoch string from API
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < CACHE_TTL:
            return self._cache

        try:
            with urllib.request.urlopen(FNG_URL, timeout=FETCH_TIMEOUT) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            item = raw["data"][0]
            result = {
                "value":     int(item["value"]),
                "label":     item["value_classification"],
                "timestamp": item.get("timestamp", ""),
            }
            self._cache      = result
            self._cache_time = now
            print(f"[CRYPTO] Fear&Greed guncellendi: {result['value']} ({result['label']})")
            return result

        except Exception as exc:
            print(f"[CRYPTO] Fear&Greed fetch hatasi: {exc} — Neutral(50) kullaniliyor")
            return _NEUTRAL_DEFAULT.copy()

    def get_signal_modifier(self, value: int) -> tuple[float, float]:
        """
        Contrarian signal modifier based on Fear & Greed value.

        Returns (long_modifier, short_modifier) — floats, added to MTF scores.

        Extreme Fear  (0-24)  : buy the panic  → LONG +1, SHORT -1
        Fear          (25-49) : slight long bias → LONG +0.5, SHORT 0
        Neutral       (50-74) : no modifier    → 0, 0
        Extreme Greed (75-100): fade euphoria  → SHORT +1, LONG -1
        """
        if value <= 24:
            return  1.0, -1.0    # Extreme Fear → long bias
        elif value <= 49:
            return  0.5,  0.0    # Fear         → slight long
        elif value <= 74:
            return  0.0,  0.0    # Greed        → neutral
        else:
            return -1.0,  1.0    # Extreme Greed → short bias

    def get_historical(self, days: int = 365) -> list[dict]:
        """
        Fetch historical F&G data (up to 365 days).
        Returns list of {date_str: 'YYYY-MM-DD', value: int} sorted ascending.
        Returns [] on error.

        Used by crypto_ml.py for feature alignment during training.
        """
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        try:
            with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            result = []
            for item in raw.get("data", []):
                ts  = int(item.get("timestamp", 0))
                val = int(item.get("value", 50))
                dt  = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                result.append({"date_str": dt, "value": val})
            return sorted(result, key=lambda x: x["date_str"])
        except Exception as exc:
            print(f"[CRYPTO] F&G history fetch hatasi: {exc}")
            return []

    def fetch_funding_rates(self) -> dict[str, float]:
        """Fetch latest Binance perp funding for BTC/ETH/BNB/SOL with 1h cache."""
        now = time.time()
        if self._funding_cache is not None and (now - self._funding_cache_time) < CACHE_TTL:
            return dict(self._funding_cache)

        rates: dict[str, float] = {}
        try:
            for symbol in FUNDING_SYMBOLS:
                resp = requests.get(
                    FUNDING_URL,
                    params={"symbol": symbol, "limit": 1},
                    timeout=FETCH_TIMEOUT,
                )
                resp.raise_for_status()
                payload = resp.json()
                if not payload:
                    continue
                rates[symbol] = float(payload[-1].get("fundingRate", 0.0) or 0.0)
            if rates:
                self._funding_cache = rates
                self._funding_cache_time = now
                msg = (
                    "[CRYPTO] Funding rates guncellendi: " +
                    " ".join(f"{sym}={rate:+.4%}" for sym, rate in sorted(rates.items()))
                )
                print(msg)
                if algo_log:
                    algo_log.system(msg)
                return dict(rates)
        except Exception as exc:
            warn = f"[CRYPTO] Funding rate fetch hatasi: {exc}"
            print(warn)
            if algo_log:
                algo_log.system(warn)
            send_telegram_alert(f"Crypto funding fetch warning: {exc}")

        return dict(self._funding_cache or {})

    def get_avg_funding_rate(self) -> float:
        rates = self.fetch_funding_rates()
        if not rates:
            return 0.0
        return float(sum(rates.values()) / len(rates))


# ── Module singleton ──────────────────────────────────────────────────────────
sentiment = CryptoSentiment()


def fetch_funding_rates() -> dict[str, float]:
    """Module-level wrapper for crypto_trader.py / crypto_status.py."""
    return sentiment.fetch_funding_rates()


def get_avg_funding_rate() -> float:
    """Module-level wrapper for dashboard / signal scans."""
    return sentiment.get_avg_funding_rate()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[CRYPTO] crypto_sentiment.py test baslıyor...")
    fg = sentiment.get_fear_greed()
    print(f"[CRYPTO] Fear & Greed: {fg['value']} ({fg['label']})")
    lm, sm = sentiment.get_signal_modifier(fg["value"])
    print(f"[CRYPTO] Sinyal modifier → LONG: {lm:+.1f} | SHORT: {sm:+.1f}")

    print("\n[CRYPTO] 7 gunluk gecmis F&G:")
    hist = sentiment.get_historical(days=7)
    for row in hist:
        print(f"  {row['date_str']}: {row['value']}")

    print("[CRYPTO] Sentiment testi tamamlandi.")
