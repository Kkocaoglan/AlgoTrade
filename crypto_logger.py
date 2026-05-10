"""
crypto_logger.py — 4-file rotating logger for Crypto Module.

4 separate log files (RotatingFileHandler 50MB x 20 each):
  crypto_trade.log   — BUY/SELL executions
  crypto_scan.log    — SCAN lines every 5 min (per symbol)
  crypto_system.log  — startup, health, errors
  crypto_exit.log    — EXIT CHECK every 15 sec (per open position)

Usage:
  from crypto_logger import crypto_log

  crypto_log.log_buy(...)
  crypto_log.log_sell(...)
  crypto_log.log_scan(...)
  crypto_log.log_exit_check(...)
  crypto_log.system("startup event")

Completely isolated from logger.py (spot system).
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"

# ── Duration formatter ────────────────────────────────────────────────────────

def _fmt_dur(seconds: float) -> str:
    seconds = int(abs(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m"


# ── CryptoLogger ──────────────────────────────────────────────────────────────

class CryptoLogger:
    """4-file rotating logger for the Crypto module."""

    def __init__(self):
        LOGS_DIR.mkdir(exist_ok=True)
        formatter = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")

        self._trade  = self._make("crypto.trade",  "crypto_trade.log",  formatter)
        self._scan   = self._make("crypto.scan",   "crypto_scan.log",   formatter)
        self._system = self._make("crypto.system", "crypto_system.log", formatter)
        self._exit   = self._make("crypto.exit",   "crypto_exit.log",   formatter)

    @staticmethod
    def _make(name: str, filename: str, formatter) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            h = RotatingFileHandler(
                LOGS_DIR / filename,
                maxBytes=50 * 1024 * 1024,
                backupCount=20,
                encoding="utf-8",
            )
            h.setFormatter(formatter)
            logger.addHandler(h)
        return logger

    # ── Core methods ──────────────────────────────────────────────────────────

    def trade(self, msg: str):  self._trade.info(msg)
    def scan(self, msg: str):   self._scan.info(msg)
    def system(self, msg: str): self._system.info(msg)
    def exit_check(self, msg: str): self._exit.info(msg)

    # ── Structured helpers ────────────────────────────────────────────────────

    def log_buy(self, symbol: str, tier: str, direction: str,
                price: float, size: float,
                stop: float, stop_pct: float,
                target: float, tgt_pct: float,
                ml: float, mtf: int,
                fg: int, fg_label: str,
                atr: float = 0.0, vol_ratio: float = 1.0):
        """Structured BUY record → crypto_trade.log"""
        dir_str = "LONG" if direction == "long" else "SHORT"
        self._trade.info(
            f"[CRYPTO][{tier}] BUY {symbol} {dir_str} @ {price:,.4f}\n"
            f"| size={size:.0f} USDT | stop={stop:,.4f} ({stop_pct:+.1f}%) "
            f"| target={target:,.4f} ({tgt_pct:+.1f}%)\n"
            f"| ml={ml:.2f} | mtf={mtf}/6 | fg={fg}({fg_label})\n"
            f"| atr={atr:.4f} | vol_ratio={vol_ratio:.1f} | tier={tier}"
        )

    def log_sell(self, symbol: str, tier: str, direction: str,
                 entry: float, exit_price: float,
                 pnl: float, pnl_pct: float,
                 hold_secs: float, reason: str):
        """Structured SELL record → crypto_trade.log"""
        sign    = "+" if pnl >= 0 else ""
        dir_str = "LONG" if direction == "long" else "SHORT"
        self._trade.info(
            f"[CRYPTO][{tier}] SELL {symbol} {dir_str} CLOSED\n"
            f"| entry={entry:,.4f} exit={exit_price:,.4f} "
            f"| P&L={sign}{pnl:.2f} USDT ({sign}{pnl_pct:.1f}%)\n"
            f"| hold_time={_fmt_dur(hold_secs)} | reason={reason} | tier={tier}"
        )

    def log_scan(self, symbol: str, tier: str,
                 ml: float, mtf_long: int, mtf_short: int,
                 adj_long: float, adj_short: float,
                 fg: int, fg_label: str,
                 ml_thresh: float, mtf_thresh: int,
                 verdict: str):
        """Per-symbol scan record → crypto_scan.log"""
        self._scan.info(
            f"[CRYPTO SCAN][{tier}] {symbol}\n"
            f"| ml={ml:.2f} mtf={mtf_long}/{mtf_short} "
            f"(adj {adj_long:.1f}/{adj_short:.1f}) fg={fg}({fg_label})\n"
            f"| threshold=ml>={ml_thresh:.2f}+mtf>={mtf_thresh} -> {verdict}"
        )

    def log_signal_decision(self, symbol: str, tier: str, direction: str | None,
                            accepted: bool, reason: str,
                            size: float | None = None,
                            risk_context: str = "",
                            signal_id: str = ""):
        """Per-symbol entry decision record → crypto_scan.log"""
        direction_txt = direction.upper() if direction else "NONE"
        decision = "ACCEPTED" if accepted else "REJECTED"
        size_txt = f" | size={size:.2f} USDT" if size is not None else ""
        context_txt = f" | {risk_context}" if risk_context else ""
        signal_txt = f" | signal_id={signal_id}" if signal_id else ""
        self._scan.info(
            f"[CRYPTO DECISION][{tier}] {symbol} {direction_txt}\n"
            f"| {decision} | reason={reason}{signal_txt}{size_txt}{context_txt}"
        )

    def log_exit_check(self, symbol: str, tier: str, direction: str,
                       entry: float, live: float, pnl: float,
                       stop: float, target: float, trailing_active: bool):
        """Per-position exit check record → crypto_exit.log"""
        sign     = "+" if pnl >= 0 else ""
        dir_str  = "LONG" if direction == "long" else "SHORT"
        trail    = "ACTIVE" if trailing_active else "inactive"
        self._exit.info(
            f"[CRYPTO EXIT][{tier}] {symbol} {dir_str}\n"
            f"| entry={entry:,.4f} live={live:,.4f} | pnl={sign}{pnl:.2f} USDT\n"
            f"| stop={stop:,.4f} target={target:,.4f} | trailing={trail}"
        )

    def log_daily(self, date_str: str, opened: int, closed: int,
                  pnl: float, win_rate: float,
                  major_pnl: float, risky_pnl: float):
        """Daily summary record → crypto_system.log"""
        sign = "+" if pnl >= 0 else ""
        self._system.info(
            f"[CRYPTO DAILY] {date_str}\n"
            f"| opened={opened} closed={closed} | pnl={sign}{pnl:.2f} USDT\n"
            f"| win_rate={win_rate:.0%} | major_pnl={major_pnl:+.2f} risky_pnl={risky_pnl:+.2f}"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

crypto_log = CryptoLogger()
