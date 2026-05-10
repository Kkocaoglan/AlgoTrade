"""
logger.py -- Structured, levelled logging for BIST Algo Trader
---------------------------------------------------------------
4 custom log levels:
  DEBUG  (10) : loop tick, signal scan internals
  TRADE  (25) : every BUY/SELL execution
  RISK   (35) : kill-switch, circuit breaker, reconciliation HIGH issues
  SYSTEM (40) : startup, shutdown, EOD refresh, reconciliation system events
  ERROR  (50) : exceptions

4 rotating log files (all in logs/):
  trade.log  -- RotatingFileHandler  50 MB x 20 files
  risk.log   -- TimedRotatingHandler 730 backups (2 years)
  system.log -- TimedRotatingHandler 365 backups (1 year)
  debug.log  -- TimedRotatingHandler 30 backups  (1 month)

All loggers also write >= SYSTEM level to stderr console.

Usage:
  from logger import algo_log

  algo_log.trade("GARAN BUY @ 140.00")
  algo_log.risk("HALT: DD 9.3%")
  algo_log.system("Loop Trader v2 started")
  algo_log.debug("Tick #5 -- 14 signals computed")
  algo_log.error("Unhandled exception", exc_info=True)

  algo_log.log_buy("GARAN",  139.74, 5000, 0.72, stop=137.0, target=143.0)
  algo_log.log_sell("GARAN", 143.23, 5000, 459.0, 2.5, "target_hit")
  algo_log.log_risk_event("HALT", "DD 9.3% >= 8.0%")
  algo_log.log_signal("GARAN", "BUY", 0.73, 62.1)

Test:
  python3.12 logger.py --test
"""

import sys
import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Custom log levels
# ---------------------------------------------------------------------------
TRADE_LEVEL  = 25   # between DEBUG(10) and WARNING(30)
RISK_LEVEL   = 35   # between WARNING(30) and ERROR(40)
SYSTEM_LEVEL = 40   # same numeric as ERROR in stdlib; we rename the label

logging.addLevelName(TRADE_LEVEL,  "TRADE")
logging.addLevelName(RISK_LEVEL,   "RISK")
logging.addLevelName(SYSTEM_LEVEL, "SYSTEM")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"


# ---------------------------------------------------------------------------
# AlgoLogger
# ---------------------------------------------------------------------------

class AlgoLogger:
    """
    Central logger for the BIST algo trading system.

    Methods
    -------
    trade(msg, **kw)    -- log at TRADE level  -> trade.log
    risk(msg, **kw)     -- log at RISK level   -> risk.log
    system(msg, **kw)   -- log at SYSTEM level -> system.log
    debug(msg, **kw)    -- log at DEBUG level  -> debug.log
    error(msg, **kw)    -- log at ERROR level  -> all files

    High-level helpers
    ------------------
    log_buy(sym, price, size, prob, stop, target)
    log_sell(sym, price, size, pnl, pct, reason)
    log_risk_event(event, detail)
    log_signal(sym, direction, prob, rsi)
    """

    def __init__(self):
        LOGS_DIR.mkdir(exist_ok=True)
        fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)

        # -- Console handler (SYSTEM level and above) -------------------------
        _console = logging.StreamHandler(sys.stderr)
        _console.setLevel(SYSTEM_LEVEL)
        _console.setFormatter(formatter)

        # -- trade.log  (RotatingFileHandler, 50 MB x 20) --------------------
        self._trade = logging.getLogger("algo.trade")
        self._trade.setLevel(TRADE_LEVEL)
        self._trade.propagate = False
        if not self._trade.handlers:
            h = RotatingFileHandler(
                LOGS_DIR / "trade.log",
                maxBytes=50 * 1024 * 1024,
                backupCount=20,
                encoding="utf-8",
            )
            h.setLevel(TRADE_LEVEL)
            h.setFormatter(formatter)
            self._trade.addHandler(h)
            self._trade.addHandler(_console)

        # -- risk.log  (daily rotation, 730 backups) --------------------------
        self._risk = logging.getLogger("algo.risk")
        self._risk.setLevel(RISK_LEVEL)
        self._risk.propagate = False
        if not self._risk.handlers:
            h = TimedRotatingFileHandler(
                LOGS_DIR / "risk.log",
                when="midnight",
                backupCount=730,
                encoding="utf-8",
            )
            h.setLevel(RISK_LEVEL)
            h.setFormatter(formatter)
            self._risk.addHandler(h)
            self._risk.addHandler(_console)

        # -- system.log  (daily rotation, 365 backups) ------------------------
        self._system = logging.getLogger("algo.system")
        self._system.setLevel(SYSTEM_LEVEL)
        self._system.propagate = False
        if not self._system.handlers:
            h = TimedRotatingFileHandler(
                LOGS_DIR / "system.log",
                when="midnight",
                backupCount=365,
                encoding="utf-8",
            )
            h.setLevel(SYSTEM_LEVEL)
            h.setFormatter(formatter)
            self._system.addHandler(h)
            self._system.addHandler(_console)

        # -- debug.log  (daily rotation, 30 backups) --------------------------
        self._debug = logging.getLogger("algo.debug")
        self._debug.setLevel(logging.DEBUG)
        self._debug.propagate = False
        if not self._debug.handlers:
            h = TimedRotatingFileHandler(
                LOGS_DIR / "debug.log",
                when="midnight",
                backupCount=30,
                encoding="utf-8",
            )
            h.setLevel(logging.DEBUG)
            h.setFormatter(formatter)
            self._debug.addHandler(h)
            # debug console handler: only SYSTEM+ to avoid spamming stderr
            self._debug.addHandler(_console)

    # -------------------------------------------------------------------------
    # Core level methods
    # -------------------------------------------------------------------------

    def trade(self, msg: str, **kw):
        self._trade.log(TRADE_LEVEL, msg, **kw)

    def risk(self, msg: str, **kw):
        self._risk.log(RISK_LEVEL, msg, **kw)

    def system(self, msg: str, **kw):
        self._system.log(SYSTEM_LEVEL, msg, **kw)

    def debug(self, msg: str, **kw):
        self._debug.debug(msg, **kw)

    def error(self, msg: str, **kw):
        """Log to all four files at ERROR level."""
        for logger in (self._trade, self._risk, self._system, self._debug):
            logger.error(msg, **kw)

    # -------------------------------------------------------------------------
    # High-level structured helpers (ASCII-only messages)
    # -------------------------------------------------------------------------

    def log_buy(
        self,
        symbol: str,
        price: float,
        size_tl: float,
        prob: float,
        stop: float  = 0.0,
        target: float = 0.0,
    ):
        """
        Structured BUY record.
        Example: BUY GARAN @ 139.74 | size=5000 TL prob=0.720 stop=137.00 target=143.00
        """
        msg = (
            f"BUY {symbol} @ {price:.2f} | "
            f"size={size_tl:.0f} TL "
            f"prob={prob:.3f} "
            f"stop={stop:.2f} "
            f"target={target:.2f}"
        )
        self.trade(msg)

    def log_sell(
        self,
        symbol: str,
        price: float,
        size_tl: float,
        pnl: float,
        pct: float,
        reason: str,
    ):
        """
        Structured SELL record.
        Example: SELL GARAN @ 143.23 | pnl=+459 TL (+2.50%) reason=target_hit
        """
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"SELL {symbol} @ {price:.2f} | "
            f"pnl={sign}{pnl:.0f} TL "
            f"({sign}{pct:.2f}%) "
            f"reason={reason}"
        )
        self.trade(msg)

    def log_risk_event(self, event: str, detail: str):
        """
        Structured RISK record.
        Example: [HALT] DD 9.3% >= 8.0% -- no new entries
        """
        self.risk(f"[{event}] {detail}")

    def log_signal(
        self,
        symbol: str,
        direction: str,
        prob: float,
        rsi: float,
    ):
        """
        Signal scan record (DEBUG level).
        Example: SIGNAL GARAN BUY prob=0.730 RSI=62.1
        """
        self.debug(
            f"SIGNAL {symbol} {direction} "
            f"prob={prob:.3f} RSI={rsi:.1f}"
        )

    def log_signal_db(
        self,
        db_path: str,
        symbol: str,
        signal_date: str,
        prob_buy: float,
        prob_sell: float,
        rsi: float,
        atr: float,
        bb_pos: float       = None,
        above_upper: int    = None,
        regime: str         = None,
        threshold_passed: int = 0,
        trade_opened: int   = 0,
        trade_id: int       = None,
    ):
        """
        Persist one signal row to signals_log table.
        Never raises — any DB error is swallowed so the main loop cannot crash.
        """
        try:
            import sqlite3 as _sqlite3
            con = _sqlite3.connect(db_path, timeout=10)
            con.execute(
                """
                INSERT INTO signals_log
                    (ts, signal_date, symbol, prob_buy, prob_sell, rsi, atr,
                     bb_pos, above_upper, regime,
                     threshold_passed, trade_opened, trade_id)
                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_date, symbol,
                    round(float(prob_buy),   6) if prob_buy   is not None else None,
                    round(float(prob_sell),  6) if prob_sell  is not None else None,
                    round(float(rsi),        4) if rsi        is not None else None,
                    round(float(atr),        6) if atr        is not None else None,
                    round(float(bb_pos),     6) if bb_pos     is not None else None,
                    int(above_upper)             if above_upper is not None else None,
                    str(regime)                  if regime     is not None else None,
                    int(threshold_passed),
                    int(trade_opened),
                    int(trade_id) if trade_id is not None else None,
                ),
            )
            con.commit()
            con.close()
        except Exception as _e:
            # Never let DB errors bubble up into the main loop
            try:
                self.debug(f"log_signal_db WARN: {_e}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Global singleton instance
# ---------------------------------------------------------------------------
algo_log = AlgoLogger()


# ---------------------------------------------------------------------------
# --test MODE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" not in sys.argv:
        print("Usage: python3.12 logger.py --test")
        sys.exit(0)

    import os as _os

    print("=" * 55)
    print("AlgoLogger Test Suite")
    print("=" * 55)

    passed = 0
    failed = 0

    def chk(name, cond, detail=""):
        global passed, failed
        if cond:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}  {detail}")
            failed += 1

    # -- Test 1: log files created after first write -------------------------
    print("\n[1] Log file creation:")
    algo_log.trade("TEST trade message")
    algo_log.risk("TEST risk message")
    algo_log.system("TEST system message")
    algo_log.debug("TEST debug message")

    chk("trade.log  created", (LOGS_DIR / "trade.log").exists())
    chk("risk.log   created", (LOGS_DIR / "risk.log").exists())
    chk("system.log created", (LOGS_DIR / "system.log").exists())
    chk("debug.log  created", (LOGS_DIR / "debug.log").exists())

    # -- Test 2: message appears in correct file ------------------------------
    print("\n[2] Message routing:")
    sentinel_trade  = "TEST_SENTINEL_TRADE_12345"
    sentinel_risk   = "TEST_SENTINEL_RISK_12345"
    sentinel_system = "TEST_SENTINEL_SYS_12345"
    sentinel_debug  = "TEST_SENTINEL_DBG_12345"

    algo_log.trade(sentinel_trade)
    algo_log.risk(sentinel_risk)
    algo_log.system(sentinel_system)
    algo_log.debug(sentinel_debug)

    def file_has(path, token):
        try:
            return token in (LOGS_DIR / path).read_text(encoding="utf-8")
        except Exception:
            return False

    chk("TRADE sentinel -> trade.log",  file_has("trade.log",  sentinel_trade))
    chk("RISK sentinel  -> risk.log",   file_has("risk.log",   sentinel_risk))
    chk("SYSTEM sentinel-> system.log", file_has("system.log", sentinel_system))
    chk("DEBUG sentinel -> debug.log",  file_has("debug.log",  sentinel_debug))

    # -- Test 3: high-level helpers -------------------------------------------
    print("\n[3] Structured helpers:")
    try:
        algo_log.log_buy("GARAN", 139.74, 5000, 0.720, stop=137.00, target=143.00)
        chk("log_buy() no exception", True)
    except Exception as e:
        chk("log_buy() no exception", False, str(e))

    try:
        algo_log.log_sell("GARAN", 143.23, 5000, 459.0, 2.50, "target_hit")
        chk("log_sell() no exception", True)
    except Exception as e:
        chk("log_sell() no exception", False, str(e))

    try:
        algo_log.log_risk_event("HALT", "DD 9.3% >= 8.0% -- no new entries")
        chk("log_risk_event() no exception", True)
    except Exception as e:
        chk("log_risk_event() no exception", False, str(e))

    try:
        algo_log.log_signal("GARAN", "BUY", 0.730, 62.1)
        chk("log_signal() no exception", True)
    except Exception as e:
        chk("log_signal() no exception", False, str(e))

    # -- Test 4: error() writes to all files ---------------------------------
    print("\n[4] error() writes to all files:")
    sentinel_error = "TEST_SENTINEL_ERROR_12345"
    algo_log.error(sentinel_error)

    chk("error -> trade.log",  file_has("trade.log",  sentinel_error))
    chk("error -> risk.log",   file_has("risk.log",   sentinel_error))
    chk("error -> system.log", file_has("system.log", sentinel_error))
    chk("error -> debug.log",  file_has("debug.log",  sentinel_error))

    # -- Test 5: log level values --------------------------------------------
    print("\n[5] Custom level values:")
    chk("TRADE_LEVEL  == 25", TRADE_LEVEL  == 25)
    chk("RISK_LEVEL   == 35", RISK_LEVEL   == 35)
    chk("SYSTEM_LEVEL == 40", SYSTEM_LEVEL == 40)
    chk("TRADE label correct",
        logging.getLevelName(TRADE_LEVEL)  == "TRADE")
    chk("RISK label correct",
        logging.getLevelName(RISK_LEVEL)   == "RISK")
    chk("SYSTEM label correct",
        logging.getLevelName(SYSTEM_LEVEL) == "SYSTEM")

    # -- Test 6: log file content format check --------------------------------
    print("\n[6] Log format check (timestamp + level + message):")
    trade_content = (LOGS_DIR / "trade.log").read_text(encoding="utf-8")
    lines = [l for l in trade_content.splitlines() if sentinel_trade in l]
    chk("trade.log line exists", bool(lines))
    if lines:
        line = lines[0]
        has_ts    = len(line) > 19 and line[4] == "-" and line[7] == "-"
        has_level = "TRADE" in line
        chk("line has timestamp", has_ts)
        chk("line has TRADE level", has_level)

    print(f"\n{'='*55}")
    print(f"Result: {passed} PASS / {failed} FAIL")
    print(f"{'='*55}")
    print(f"Log directory: {LOGS_DIR}")
    for name in ("trade.log", "risk.log", "system.log", "debug.log"):
        p = LOGS_DIR / name
        size = p.stat().st_size if p.exists() else 0
        print(f"  {name:<12} {size:>8} bytes")

    sys.exit(0 if failed == 0 else 1)
