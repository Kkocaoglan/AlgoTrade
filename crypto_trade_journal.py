"""
crypto_trade_journal.py — Sade işlem günlüğü (results/crypto_trades.txt)

Her pozisyon açılışında ve kapanışında tek satır yazar.
Crypto modülüne özel, tamamen izole.

Kullanım (crypto_trader.py'den):
    from crypto_trade_journal import journal_open, journal_close
    journal_open(symbol, direction, price, size, stop, target)
    journal_close(symbol, direction, entry_price, exit_price, pnl, pnl_pct, hold_secs, reason)
    journal_partial_close(symbol, direction, entry_price, exit_price, pnl, pnl_pct, hold_secs, reason, close_fraction)

Dosya: results/crypto_trades.txt
"""

from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).parent / "results" / "crypto_trades.txt"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_hold(secs: float) -> str:
    if secs < 60:
        return f"{int(secs)}s"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    if h > 0:
        return f"{h}s {m:02d}d"
    return f"{m}d"


def _write(line: str) -> None:
    _LOG_PATH.parent.mkdir(exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def journal_open(symbol: str, direction: str, price: float,
                 size: float, stop: float, target: float) -> None:
    """Pozisyon açılışını yazar."""
    coin = symbol.split("/")[0]
    side = "LONG " if direction == "long" else "SHORT"
    stop_pct  = abs(price - stop)   / price * 100
    tgt_pct   = abs(price - target) / price * 100
    line = (
        f"[ACILDI ] {_now()} | {coin:<6} {side} | "
        f"Giris: ${price:,.4f} | Boyut: {size:.0f} USDT | "
        f"Stop: ${stop:,.4f} (-%{stop_pct:.1f}) | Hedef: ${target:,.4f} (+%{tgt_pct:.1f})"
    )
    _write(line)


def journal_close(symbol: str, direction: str,
                  entry_price: float, exit_price: float,
                  pnl: float, pnl_pct: float,
                  hold_secs: float, reason: str) -> None:
    """Pozisyon kapanışını yazar."""
    coin   = symbol.split("/")[0]
    side   = "LONG " if direction == "long" else "SHORT"
    sign   = "+" if pnl >= 0 else ""
    sonuc  = "KAR" if pnl >= 0 else "ZARAR"
    reason_map = {
        "target_hit":    "hedef",
        "stop_loss":     "stop",
        "trailing_stop": "trailing",
        "score_flip":    "sinyal dondu",
        "daily_loss":    "gunluk limit",
        "manual":        "manuel",
    }
    reason_tr = reason_map.get(reason, reason)
    line = (
        f"[KAPANDI] {_now()} | {coin:<6} {side} | "
        f"Giris: ${entry_price:,.4f} -> Cikis: ${exit_price:,.4f} | "
        f"{sonuc}: {sign}{pnl:.2f} USDT ({sign}{pnl_pct:.1f}%) | "
        f"Sure: {_fmt_hold(hold_secs)} | Neden: {reason_tr}"
    )
    _write(line)


def journal_partial_close(symbol: str, direction: str,
                          entry_price: float, exit_price: float,
                          pnl: float, pnl_pct: float,
                          hold_secs: float, reason: str,
                          close_fraction: float) -> None:
    coin   = symbol.split("/")[0]
    side   = "LONG " if direction == "long" else "SHORT"
    sign   = "+" if pnl >= 0 else ""
    reason_map = {
        "partial_take_profit": "kismi kar alma",
        "target_hit_partial": "kismi hedef",
        "manual_partial": "kismi manuel",
    }
    reason_tr = reason_map.get(reason, reason)
    line = (
        f"[PARCA  ] {_now()} | {coin:<6} {side} | "
        f"Giris: ${entry_price:,.4f} -> Cikis: ${exit_price:,.4f} | "
        f"P&L: {sign}{pnl:.2f} USDT ({sign}{pnl_pct:.1f}%) | "
        f"Kapanan: %{close_fraction*100:.0f} | Sure: {_fmt_hold(hold_secs)} | Neden: {reason_tr}"
    )
    _write(line)
