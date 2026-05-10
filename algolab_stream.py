"""
algolab_stream.py -- Algolab API + yfinance fallback
--------------------------------------------------------------
Usage:
  py -3.12 algolab_stream.py --test

MOCK mode (default — API key empty):
  All price/bar calls go through yfinance.
  Identical to current loop_trader behavior. Nothing breaks.

LIVE mode (ALGOLAB_API_KEY set):
  Uses Algolab REST API. Login requires SMS 2FA (one-time per session).
  Falls back to yfinance on any API error.

To switch LIVE — only change these 3 lines:
  ALGOLAB_API_KEY  = "buraya_api_key"
  ALGOLAB_USERNAME = "TC_kimlik_no"
  ALGOLAB_PASSWORD = "denizbank_sifre"
  Nothing else changes. System auto-switches to LIVE.
--------------------------------------------------------------
"""

import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ==============================================================================
# CREDENTIALS -- fill in to go LIVE; leave empty for MOCK (yfinance)
# ==============================================================================
ALGOLAB_API_KEY  = ""   # Deniz Yatirim'dan gelecek -- bos birak
ALGOLAB_USERNAME = ""   # TC Kimlik No
ALGOLAB_PASSWORD = ""   # DenizBank sifresi

ALGOLAB_HOSTNAME = "www.algolab.com.tr"
ALGOLAB_API_URL  = f"https://{ALGOLAB_HOSTNAME}/api"
ALGOLAB_WS_URL   = f"wss://{ALGOLAB_HOSTNAME}/api/ws"

MODE = "MOCK" if not ALGOLAB_API_KEY else "LIVE"

# ==============================================================================
# AES encryption (required by Algolab API)
# ==============================================================================
try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
    import base64
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


def encrypt(text, api_key):
    """AES-CBC encrypt text with base64-encoded api_key."""
    if not HAS_CRYPTO:
        raise RuntimeError(
            "pycryptodome not installed -- "
            "pip install pycryptodome --break-system-packages"
        )
    iv     = b'\0' * 16
    key    = base64.b64decode(api_key.encode("utf-8"))
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return base64.b64encode(cipher.encrypt(pad(text.encode(), 16))).decode()


# ==============================================================================
# Optional deps (graceful degradation)
# ==============================================================================
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

# ==============================================================================
# Universe
# ==============================================================================
SYMBOLS = [
    "YKBNK", "AKBNK", "ISCTR", "GARAN",
    "TUPRS", "PETKM",
    "TAVHL", "FROTO",
    "TCELL", "ASELS",
    "BIMAS", "MGROS",
    "ENKAI", "EKGYO",
    # Expanded universe (added 2026-04-24)
    "THYAO", "EREGL",
    "KCHOL", "SAHOL",
    "SISE",  "TOASO",
    "ARCLK", "VESTL",
    "KRDMD",
    "PGSUS", "ODAS",
    "GUBRF", "CIMSA",
    "LOGO",  "NETAS",
]

_EMPTY_BAR = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}


# ==============================================================================
# AlgolabStream
# ==============================================================================
class AlgolabStream:
    """
    Algolab REST API wrapper with transparent yfinance fallback.

    Architecture:
      - MOCK mode: all calls route to yfinance (same as existing loop_trader).
      - LIVE mode: calls Algolab REST API; falls back to yfinance on error.
      - get_all_bars() in MOCK mode does the original batch yfinance download
        (efficient — single HTTP call for all 14 symbols).
    """

    def __init__(self):
        self.token     = ""
        self.hash      = ""
        self.prices    = {}   # {symbol: float}  -- last-seen price cache
        self.connected = False
        self.mode      = MODE

    # --------------------------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------------------------

    def login(self):
        """
        Authenticate against Algolab.
        MOCK mode: prints info, returns True immediately (no network call).
        LIVE mode: two-step SMS 2FA.  Falls back to MOCK on error.
        """
        if self.mode == "MOCK":
            print("[ALGOLAB] MOCK mode -- API key yok, yfinance kullaniliyor")
            return True

        if not HAS_REQUESTS:
            print("[ALGOLAB] requests kutuphane yok -- MOCK moda geciliyor")
            self.mode = "MOCK"
            return False

        try:
            # Step 1: LoginUser -- SMS gonder
            payload = {
                "username": encrypt(ALGOLAB_USERNAME, ALGOLAB_API_KEY),
                "password": encrypt(ALGOLAB_PASSWORD, ALGOLAB_API_KEY),
            }
            headers = {"APIKEY": ALGOLAB_API_KEY}
            r = requests.post(
                f"{ALGOLAB_API_URL}/LoginUser",
                json=payload, headers=headers, timeout=10,
            )
            data       = r.json()
            self.token = data.get("token", "")
            print(f"[ALGOLAB] SMS gonderildi. Token: {self.token[:20]}...")

            # Step 2: LoginUserControl -- kullanicidan SMS kodu al
            sms_code = input("[ALGOLAB] SMS kodunu gir: ").strip()
            payload2 = {
                "token":    self.token,
                "password": encrypt(sms_code, ALGOLAB_API_KEY),
            }
            r2 = requests.post(
                f"{ALGOLAB_API_URL}/LoginUserControl",
                json=payload2, headers=headers, timeout=10,
            )
            data2      = r2.json()
            self.hash  = data2.get("hash", "")
            self.connected = True
            print(f"[ALGOLAB] Giris basarili. Hash: {self.hash[:20]}...")
            return True

        except Exception as e:
            print(f"[ALGOLAB] Login hatasi: {e} -- MOCK moda geciliyor")
            self.mode = "MOCK"
            return False

    # --------------------------------------------------------------------------
    # SINGLE PRICE
    # --------------------------------------------------------------------------

    def get_price(self, symbol) -> float:
        """Get latest price for one symbol. LIVE: Algolab API. MOCK: yfinance."""
        if self.mode == "LIVE":
            return self._get_price_live(symbol)
        return self._get_price_mock(symbol)

    def _get_price_live(self, symbol) -> float:
        try:
            headers = {
                "APIKEY":        ALGOLAB_API_KEY,
                "Authorization": self.hash,
            }
            r     = requests.get(
                f"{ALGOLAB_API_URL}/GetEquityInfo",
                params={"symbol": symbol},
                headers=headers, timeout=5,
            )
            price = float(r.json().get("lastPrice", 0))
            self.prices[symbol] = price
            return price
        except Exception as e:
            print(f"[ALGOLAB] Fiyat hatasi {symbol}: {e} -- yfinance fallback")
            return self._get_price_mock(symbol)

    def _get_price_mock(self, symbol) -> float:
        if not HAS_YF:
            return self.prices.get(symbol, 0.0)
        try:
            # fast_info.last_price: reliable, returns today's live price even when
            # history(period="1d") fails for BIST in yfinance 1.x
            price = float(yf.Ticker(f"{symbol}.IS").fast_info.last_price or 0)
            if price > 0:
                self.prices[symbol] = price
                return price
        except Exception:
            pass
        return self.prices.get(symbol, 0.0)

    # --------------------------------------------------------------------------
    # SINGLE BAR (5-minute OHLCV)
    # --------------------------------------------------------------------------

    def get_last_bar(self, symbol) -> dict:
        """Get last 5-minute OHLCV bar. LIVE: Algolab API. MOCK: yfinance."""
        if self.mode == "LIVE":
            return self._get_bar_live(symbol)
        return self._get_bar_mock(symbol)

    def _get_bar_live(self, symbol) -> dict:
        try:
            headers = {
                "APIKEY":        ALGOLAB_API_KEY,
                "Authorization": self.hash,
            }
            r    = requests.get(
                f"{ALGOLAB_API_URL}/GetEquityInfo",
                params={"symbol": symbol, "period": "5"},
                headers=headers, timeout=5,
            )
            data = r.json()
            return {
                "open":   float(data.get("open",      0)),
                "high":   float(data.get("high",      0)),
                "low":    float(data.get("low",       0)),
                "close":  float(data.get("lastPrice", data.get("close", 0))),
                "volume": float(data.get("volume",    0)),
            }
        except Exception:
            return self._get_bar_mock(symbol)

    def _get_bar_mock(self, symbol) -> dict:
        if not HAS_YF:
            return dict(_EMPTY_BAR)
        try:
            ticker = yf.Ticker(f"{symbol}.IS")
            # period="2d" is reliable; period="1d" fails for BIST in yfinance 1.x
            hist   = ticker.history(period="2d", interval="5m")
            if not hist.empty:
                last = hist.iloc[-1]
                # Update close with today's live price (fast_info works even when
                # history(period="1d") does not)
                try:
                    live = float(ticker.fast_info.last_price or 0)
                except Exception:
                    live = 0.0
                close = live if live > 0 else float(last["Close"])
                return {
                    "open":   float(last["Open"]),
                    "high":   float(last["High"]),
                    "low":    float(last["Low"]),
                    "close":  close,
                    "volume": float(last["Volume"]),
                }
        except Exception:
            pass
        return dict(_EMPTY_BAR)

    # --------------------------------------------------------------------------
    # BATCH: prices (all symbols)
    # --------------------------------------------------------------------------

    def get_all_prices(self, symbols) -> dict:
        """Get current prices for a list of symbols. Returns {symbol: price}."""
        return {sym: self.get_price(sym) for sym in symbols}

    # --------------------------------------------------------------------------
    # BATCH: bars (all symbols) -- used by loop_trader fetch_all_live_bars
    # --------------------------------------------------------------------------

    def get_all_bars(self, symbols) -> dict:
        """
        Batch-fetch last 5-minute OHLCV bar for all symbols.

        LIVE mode  : per-symbol API calls (parallel behavior expected by Algolab).
        MOCK mode  : single batch yfinance download (efficient -- identical to
                     the original fetch_all_live_bars() in loop_trader.py).

        Returns {symbol: {open, high, low, close, volume}}.
        """
        if self.mode == "LIVE":
            return self._get_all_bars_live(symbols)
        return self._get_all_bars_mock(symbols)

    def _get_all_bars_live(self, symbols) -> dict:
        bars = {}
        for sym in symbols:
            b = self._get_bar_live(sym)
            if b["close"] > 0:
                bars[sym] = b
        return bars

    def _get_all_bars_mock(self, symbols) -> dict:
        """
        Batch yfinance OHLCV for MOCK mode.

        fix 2026-04-24: fast_info.last_price (batch yf.Tickers) BIST hisseleri
        icin acilis fiyatini donduruyor — gercek intraday kapanis degil.
        Tamamen kaldirildi.

        Yeni strateji (fast_info yok):
          1. Birincil : period="1d"/interval="1m" — en son 1dk bar kapanis.
          2. Fallback : period="2d"/interval="5m" — eksik semboller icin.
          Her iki kaynak da bugunun intraday hareketini dogru yansitiyor.
        """
        import datetime as _dt

        if not HAS_YF or not HAS_PD:
            return {}

        today = _dt.date.today()

        def _extract_bars(raw, sym_list) -> dict:
            """yf.download sonucundan {sym: OHLCV} cikar."""
            result = {}
            if raw is None or raw.empty:
                return result
            for sym in sym_list:
                sym_is = sym + ".IS"
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df_s = raw[sym_is].dropna(how="all")
                    else:
                        # Tek sembol indirme: raw direkt o sembolun verisi
                        df_s = raw.dropna(how="all")
                    if df_s.empty:
                        continue
                    df_s.columns = [c.lower() for c in df_s.columns]
                    # Bugunun satirlarini sec; yoksa son mevcut seansi kullan
                    today_rows = df_s[[i.date() == today for i in df_s.index]]
                    session = today_rows if not today_rows.empty else df_s
                    if session.empty:
                        continue
                    close = float(session["close"].iloc[-1])
                    result[sym] = {
                        "open":   float(session["open"].iloc[0]),
                        "high":   float(session["high"].max()),
                        "low":    float(session["low"].min()),
                        "close":  close,
                        "volume": float(session["volume"].sum()),
                    }
                except Exception:
                    pass
            return result

        bars = {}

        # 1. Birincil: 1dk barlar (daha guncel kapanis)
        tickers_is = [s + ".IS" for s in symbols]
        try:
            raw1m = yf.download(
                tickers_is, period="1d", interval="1m",
                progress=False, auto_adjust=True, group_by="ticker",
            )
            bars = _extract_bars(raw1m, symbols)
        except Exception as e:
            print(f"  [ALGOLAB MOCK] 1m download error: {e}")

        # 2. Fallback: 5dk barlar — 1m'de gozukmeyen semboller icin
        missing = [s for s in symbols if s not in bars]
        if missing:
            missing_is = [s + ".IS" for s in missing]
            try:
                raw5m = yf.download(
                    missing_is, period="2d", interval="5m",
                    progress=False, auto_adjust=True, group_by="ticker",
                )
                bars5 = _extract_bars(raw5m, missing)
                bars.update(bars5)
            except Exception as e:
                print(f"  [ALGOLAB MOCK] 5m fallback error: {e}")

        return bars

    # --------------------------------------------------------------------------
    # STATUS / DISCONNECT
    # --------------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "mode":           self.mode,
            "connected":      self.connected,
            "cached_symbols": list(self.prices.keys()),
        }

    def disconnect(self):
        self.connected = False
        print("[ALGOLAB] Baglanti kapatildi")


# ==============================================================================
# Global instance -- imported by loop_trader.py
# ==============================================================================
stream = AlgolabStream()


# ==============================================================================
# CLI test
# ==============================================================================
if __name__ == "__main__":
    if "--test" not in sys.argv:
        print("Kullanim: py -3.12 algolab_stream.py --test")
        sys.exit(0)

    print(f"[TEST] Mode: {stream.mode}")
    stream.login()

    src = "ALGOLAB" if stream.mode == "LIVE" else "yfinance(MOCK)"
    print(f"\n[TEST] 14 hisse fiyat testi (kaynak: {src}):")
    for sym in SYMBOLS:
        price = stream.get_price(sym)
        bar   = stream.get_last_bar(sym)
        print(f"  {sym:6} {price:8.2f} TL | "
              f"bar_close={bar['close']:.2f} | kaynak={src}")

    print(f"\n[TEST] Status: {stream.status()}")
    stream.disconnect()
