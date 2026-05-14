# BIST Algo Trading — System Reference

## Commands
```
# Mac always python3.12 | Windows always py -3.12
python3.12 loop_trader.py --once       # single tick (safe test)
python3.12 loop_trader.py              # MAIN: production 60s loop
python3.12 paper_trade.py --status     # show open positions + P&L
python3.12 fetch_data.py && python3.12 indicators.py   # update DB
python3.12 ml_train.py                 # retrain ensemble model
python3.12 portfolio_risk.py --report  # heat + correlation check
python3.12 volatility_regime.py --report  # vol regime + 5-day history
```

## What This Is
- BIST paper trading system: 100,000 TL virtual capital, 29 stocks (LONG + SHORT), XGBoost+LGB+CatBoost ensemble.
- No real money. Goal: paper validate ML signals → live via Algolab API (P4).
- Phase: paper trading since 2026-04-08. Eval deadline: 2026-06-08 (extended from 2026-05-08).
- Required: win_rate ≥ 58%, Sharpe ≥ 1.2, ≥ 30 closed trades.
- Platform: Python 3.12, macOS/Windows, SQLite (`trade_data.db`).

## File Map
```
trade_data.db           SQLite: OHLCV, indicators, positions, orders, equity
fetch_data.py           Fetches OHLCV from isyatirimhisse (primary) + yfinance (fallback)
indicators.py           Computes EMA8/21/50/200, RSI14, MACD, ATR14, BB, OBV, MTF into DB
backtest.py             Rule-based strategies: S1=MTF Momentum, S2=RSI MeanRev, S3=BB Breakout
cost_aware_backtest.py  Gross vs net P&L with realistic BIST slippage + commission
bist_live_wf_sim.py     Read-only replay of signals_log with loop_trader-style sizing/exits
ml_train.py             Trains XGBoost+LightGBM+CatBoost ensemble; saves to models/
validate_model.py       4-test model validation (lookahead, net precision, features, bias)
paper_trade.py          Paper engine (--run/--status/--watch); max 6 positions
loop_trader.py          MAIN: 60s loop, live signals, OMS, circuit breaker, reconciliation
bist_config.py          Shared BIST runtime risk/config source for loop_trader, kill_switch, portfolio_risk
algolab_stream.py       Algolab REST API + yfinance MOCK fallback; global: stream
kill_switch.py          Pre-BUY safety gate (6 checks); reads/writes KILL_SWITCH.txt
oms.py                  Order lifecycle: NEW→SENT→FILLED/REJECTED/CANCELLED; global: oms
reconciliation.py       DB consistency checks (7 checks); reports only, never auto-fixes
logger.py               Structured logging: 4 levels, 4 rotating files in logs/; global: algo_log
daily_report.py         Daily P&L report + results/pnl_chart.png
daily_journal.py        EOD journal at 18:30 → performance_tracker.csv
news_filter.py          News scraper: KAP (Google RSS fallback), Investing.com, TCMB
sentiment.py            BERTurk+keyword hybrid; SentimentAnalyzer; score_news(); lazy BERT
portfolio_risk.py       Portfolio heat cap (6%) + correlation gate (0.85); global: _portfolio_risk
volatility_regime.py    Realized vol regime filter NORMAL/HIGH_VOL/EXTREME; global: _vol_regime
telegram_bot.py         Telegram alerts (TELEGRAM_TOKEN + TELEGRAM_CHAT_ID from .env)
```

## System Flow (loop_trader.py every 60s)
1. Startup: load model, algolab.login(), reconciliation, get equity peak, send Telegram; `_check_kill_switch_expiry()` (auto-delete if >24h old & DD<12%); `_print_health_check()` (KILL_SWITCH/VOL/MODEL/positions/cash/DD)
2. Each tick: `_compute_drawdown()` → `check_circuit_breaker()` → cb_status
3. If `in_market()` (10:00-18:00 Istanbul):
4. Morning recon once (before 10:05) if not done today
5. If `cb_status == KILL` (DD ≥ 12%): force-close all → write KILL_SWITCH.txt → break
6. If `past_skip_window()` (after 10:30): `get_live_signals()` → fetch live bars, patch DB, recompute indicators, run model per symbol
7. `print_debug_lines()` + `show_signal_report()` + `send_signal_telegram()` + `_print_gate_summary()` (7 gate PASS/BLOCK her tick)
8. If `HALT`: no new entries, exits still run
9. If `EXTREME vol` AND `OVERRIDE_VOL_BLOCK=False`: skip `scan_and_trade()`, log HALT. If `OVERRIDE_VOL_BLOCK=True`: trade at x0.4 size.
10. Else (OK/WARN/HIGH_VOL): `scan_and_trade(max_pos_override, vol_multiplier)`
11. Every 5th tick (or --once): `check_exits()`
12. Friday 17:30: `send_friday_weekend_warnings()` — no forced close
13. Outside market: `run_eod_refresh()` after 18:05; `run_daily_summary()` at 18:30
14. Always print risk bar: `[RISK] Equity:X Peak:Y DD:Z% Status:CB`
15. Always: `print_status()` + `oms.summary()`

## Universe (29 Stocks — expanded 2026-04-24)
| Symbol | Sector     | Notes                           |
|--------|------------|---------------------------------|
| YKBNK  | banka      | Yapı Kredi; max 2/sector        |
| AKBNK  | banka      | Akbank                          |
| ISCTR  | banka      | İş Bankası                      |
| GARAN  | banka      | Garanti BBVA; vol proxy         |
| TUPRS  | enerji     | Tüpraş; most liquid refiner     |
| PETKM  | enerji     | Petkim                          |
| TAVHL  | havacilik  | TAV Airports; tourism beta      |
| FROTO  | otomotiv   | Ford Otosan; export proxy       |
| TCELL  | telekom    | Turkcell; cross-stock feature   |
| ASELS  | savunma    | Aselsan; defence budget beta    |
| BIMAS  | perakende  | BİM                             |
| MGROS  | perakende  | Migros (NOT MIGROS)             |
| ENKAI  | insaat     | ENKA (NOT ENKA)                 |
| EKGYO  | insaat     | Emlak Konut (NOT EMLAK)         |
| THYAO  | havacilik  | Turkish Airlines; also vol proxy |
| EREGL  | celik      | Ereğli Demir Çelik; also vol proxy |
| KCHOL  | holding    | Koç Holding                     |
| SAHOL  | holding    | Sabancı Holding                 |
| SISE   | cam        | Şişecam                         |
| TOASO  | otomotiv   | Tofaş                           |
| ARCLK  | tuketim    | Arçelik                         |
| VESTL  | tuketim    | Vestel                          |
| KRDMD  | celik      | Kardemir                        |
| PGSUS  | havacilik  | Pegasus Airlines                |
| ODAS   | enerji     | Odaş Elektrik                   |
| GUBRF  | kimya      | Gübre Fabrikaları               |
| CIMSA  | cimento    | Çimsa                           |
| LOGO   | teknoloji  | Logo Yazılım                    |
| NETAS  | teknoloji  | Netaş                           |

**Ticker corrections**: MIGROS→MGROS | ENKA→ENKAI | EMLAK→EKGYO
**KOZAL delisted** (HTTP 404 on yfinance + ISY). Not in universe.

## ML Model
- **Ensemble**: XGBoost + LightGBM + CatBoost, soft-vote averaged probability.
- **Retrained**: 2026-04-27 on 29-stock universe + 12 cross-asset macro features. Saved in `models/xgb_model.pkl`.
- **HORIZON**: 3 days (label = +2.0% within 3 days); walk-forward 6 folds; 49 features.
- **Metrics**: WF BUY precision 81.1%, traded-only BUY precision 73.4%, test accuracy 62.4%.
- **Active model**: BASELINE (uniform weights) — temporal weighting tested but reverted (WF dropped 81.1%→79.8%, -1.3pp).
- **Retrain trigger**: 30-trade live win rate < 58% at 2026-06-08.
- **Top features**: ret_20d, trend_str_20, above_upper, price_ema200, strongest_sector_5d, bb_pos, atr_ratio.
- **Macro features (2026-04-27)**: 12 cross-asset features added (^VIX, EEM, GC=F, DX-Y.NYB, ^GSPC, ^STOXX50E). ALL 12 new features DROPPED (gain=0.0). strongest_sector_5d KEPT (gain 0.0323, rank 5/64). Root cause: BIST-specific returns have no linear correlation with global macro in walk-forward test window. Final feature count: 49 (unchanged from baseline — macro dropped; sector momentum kept). indicators.py now fetches 8 tickers with full ffill; derived features pre-computed and stored in macro_data DB (20 new columns). No NaN-fills masking signals.
- **Temporal weighting (2026-04-27, reverted)**: WF precision 79.8% < baseline 81.1% → baseline kept. Re-evaluate at next retrain cycle.

| Tier      | prob range | Action                        |
|-----------|------------|-------------------------------|
| WEAK      | 0.50-0.64  | Displayed only, NOT traded    |
| TRADEABLE | ≥ 0.65     | BUY_THRESHOLD — opens trade   |
| MEDIUM    | 0.65-0.69  | Traded                        |
| STRONG    | ≥ 0.70     | Immediate Telegram alert      |

Thresholds: BUY_THRESHOLD=0.65 | SHORT_THRESHOLD=0.65 | MODEL_FLIP_THRESHOLD=0.55 | DISPLAY_THRESHOLD=0.50

## SHORT Capability (added 2026-04-24)
- **Open SHORT**: prob_sell = 1-prob_buy ≥ 0.65 AND RSI > 60 (overbought filter)
- **Max short size**: 5,000 TL per position (half of LONG max)
- **Short stop**: entry × 1.015 (1.5% above entry)
- **Short target**: entry − ATR × 1.5 (below entry)
- **Short trailing**: LWM-based (low-water-mark). profit ≥ +2% → trail at LWM+1%; profit ≥ +1.5% → breakeven
- **Short exits**: stop if cur ≥ eff_stop | target if cur ≤ target | model flip if prob_buy > 0.45 | news if score ≥ +2
- **Telegram**: 🔴 [SHORT] ACILDI / 🔴 [SHORT] KAPATILDI
- **OMS direction**: SHORT to open, COVER to close (both go through kill switch)
- Same sector/heat/corr/dedup gates as LONG

## Exit Logic (LONG)
| Priority | Type          | Condition                              | Reason Code    |
|----------|---------------|----------------------------------------|----------------|
| 1        | News Exit     | news score ≤ -2 in last 30 min        | news_exit      |
| 2        | Stop Loss     | cur ≤ original stop (-1.5% from entry) | stop_loss      |
| 2        | Trailing Stop | cur ≤ HWM-1% (after +2.0% profit)    | trailing_stop  |
| 3        | Target Hit    | cur ≥ entry + ATR × 1.5               | target_hit     |
| 4        | Model Flip    | prob_buy < 0.55                        | model_flip     |

## Exit Logic (SHORT)
| Priority | Type          | Condition                              | Reason Code    |
|----------|---------------|----------------------------------------|----------------|
| 1        | News Exit     | news score ≥ +2 in last 30 min        | news_exit      |
| 2        | Stop Loss     | cur ≥ original stop (+1.5% above entry)| stop_loss     |
| 2        | Trailing Stop | cur ≥ LWM+1% (after +2.0% profit)    | trailing_stop  |
| 3        | Target Hit    | cur ≤ entry − ATR × 1.5               | target_hit     |
| 4        | Model Flip    | prob_buy > 0.45 (model now bullish)    | model_flip     |

- No EOD forced close (hold overnight Mon-Thu). No time exit.
- Friday 17:30: warning Telegram only — positions NOT closed.
- SELL/COVER orders always bypass kill switch — exits must always execute.

## Capital Allocation
- Total capital: 100,000 TL | Min cash reserve: 20,000 TL | Tradeable: 80,000 TL
- Max per position: 10,000 TL | Risk per trade: 1% = 1,000 TL max loss
- Size formula: `size = min((1000 / (ATR×1.5)) × price, 10000, cash - 20000)`; skip if < 1,000 TL
- Max simultaneous: 8 (loop_trader) | Sector limit: MAX_PER_SECTOR = 2
- WARN mode (DD ≥ 4%): additional cap at 5,000 TL
- HIGH_VOL regime: size × 0.6 after all other caps

## Risk Layers
| Layer | File              | Threshold            | Action                          |
|-------|-------------------|----------------------|---------------------------------|
| 1     | kill_switch.py    | KILL_SWITCH.txt, daily loss -2k, fat finger 5%, rate 5/min, DD 8%/12% | BLOCK BUY |
| 2     | loop_trader.py    | DD 4%/8%/12%         | WARN cap / HALT entries / KILL all |
| 3     | loop_trader.py    | sector count ≥ 2     | Skip BUY                        |
| 4     | news_filter.py    | score ≤ -1 entry / ≤ -2 exit | Skip entry / force close  |
| 5     | loop_trader.py    | RSI > 72 at entry    | Skip BUY                        |
| 6     | loop_trader.py    | same symbol < 2h     | Dedup cooldown                  |
| 7     | portfolio_risk.py | heat > 6% or corr ≥ 0.85 | Block BUY                  |
| 8     | volatility_regime.py | EXTREME (vol ≥ 4%) | Skip scan_and_trade             |

## Portfolio Heat (portfolio_risk.py)
- `MAX_PORTFOLIO_HEAT = 0.06` (6%) | `CORR_THRESHOLD = 0.85` | `CORR_WINDOW_DAYS = 60`
- Heat formula: `risk_tl = (entry - stop) / entry × size_tl` per position; `total_heat = Σrisk_tl / 100000`
- Correlation: pct_change returns over 60 days; block if |corr| ≥ 0.85 to any open position
- CLI: `python3.12 portfolio_risk.py --report` | `--check GARAN 1000`

## Volatility Regime (volatility_regime.py)
| Regime   | Daily Vol  | Size Multiplier | Effect                      |
|----------|------------|-----------------|-----------------------------|
| NORMAL   | < 2.5%     | 1.0             | Full size, normal trading   |
| HIGH_VOL | 2.5%-4.0%  | 0.6             | 40% size reduction          |
| EXTREME  | ≥ 4.0%     | 0.0             | scan_and_trade skipped      |

- Proxy: GARAN+THYAO+EREGL equal-weighted daily returns (XU100 not in DB).
- Realized vol: `std(last 20 daily returns)` — daily, not annualized.
- **Current state (2026-04-24)**: HIGH_VOL at 3.00% (3-symbol proxy: GARAN+THYAO+EREGL) — size x0.6, trading allowed.
- CLI: `python3.12 volatility_regime.py --report`

## OMS (oms.py)
`NEW → SENT → FILLED | PARTIAL | REJECTED | CANCELLED`
- BUY / SHORT: `oms.create_order()` runs `KillSwitch.check_all()` internally; returns REJECTED if blocked.
- SELL / COVER: kill switch skipped — exits always execute. `oms.fill_order()` auto-logs to kill switch rate limiter.

## Reconciliation (reconciliation.py)
- 7 checks: GHOST_POSITION, MISSING_POSITION, SIZE_MISMATCH (HIGH/MEDIUM), BALANCE_MISMATCH, NEGATIVE_CASH (HIGH), ORPHAN_ORDER, PNL_MISMATCH.
- Runs: startup + 10:05 morning (loop_trader); 18:30 DB-only (daily_journal).
- Reports only — **never auto-fixes**. HIGH severity → Telegram alert. Tolerances: BALANCE 100 TL, PNL 50 TL.

## Logging (logger.py)
| File       | Level  | Rotation               | Content                          |
|------------|--------|------------------------|----------------------------------|
| trade.log  | TRADE  | 50MB×20 rotating       | Every BUY/SELL fill              |
| risk.log   | RISK   | daily, 730 backups     | Kill switch, CB events, HIGH recon |
| system.log | SYSTEM | daily, 365 backups     | Startup, EOD, recon system events |
| debug.log  | DEBUG  | daily, 30 backups      | Loop tick, signal scan           |

`from logger import algo_log` → `algo_log.log_buy/log_sell/log_risk_event/log_signal/debug/risk/system`
Console stderr: SYSTEM level and above only. ASCII-only messages (no Turkish chars in log files).

## News & Sentiment
- **Sources**: KAP JSON API (HTTP 666 bot block) → Google News RSS (working fallback); Investing.com TR RSS (working); TCMB HTML (working).
- Output: `results/news_log.jsonl` (one JSON per line, includes `sentiment_score` + `sentiment_label`).
- **BERTurk hybrid** (`savasy/bert-base-turkish-sentiment-cased`): keywords primary (domain-specific, sector-aware); BERT fills when keyword=0, clamped [-2,+2] to prevent false forced-exits from model's domain mismatch on financial text.
- **Mac segfault fix**: BERT loads lazily on first `score()` call (not at import). `_load_bert()` sets `TOKENIZERS_PARALLELISM=false` on Darwin. Any load exception → keyword fallback, no crash.
- `from sentiment import analyzer` — module singleton. Never instantiate `SentimentAnalyzer()` twice.

## Paper Trade Criteria (30-Day Eval)
| Metric       | Target   | Status (2026-04-24) |
|--------------|----------|---------------------|
| Win rate     | ≥ 58%    | hesaplanacak (5 stop-loss eklendi) |
| Sharpe       | ≥ 1.2    | N/A (yetersiz veri) |
| Closed trades| ≥ 30     | 6/30 (20%) — 5 stop-loss kapandı  |
| Max drawdown | < 15%    | 0.0% (equity_tracker quirk çözüldü) |

Eval window: 2026-04-08 → 2026-06-08 (extended). Performance tracker: `results/performance_tracker.csv` (22 cols, updated by daily_journal.py at 18:30).

## Current Open Positions (2026-04-24)
**0 açık pozisyon.** Tüm Nisan pozisyonları kapatıldı (stop-loss / April correction).
Cash: ~100,886 TL | Equity: ~100,886 TL | DD: 0.0% | Vol regime: HIGH_VOL (3.46%, size x0.6)
**equity_tracker quirk**: ÇÖZÜLDÜ — peak artık 100,000 TL olarak okunuyor, DD=0.0%.

## Performance So Far
| Date       | Symbol | Entry  | Exit   | P&L    | P&L%  | Reason     |
|------------|--------|--------|--------|--------|-------|------------|
| 2026-04-14 | GARAN  | 139.74 | 143.23 | +459 TL| +2.5% | target_hit |
| 2026-04-20 | AKBNK  | 77.83  | —      | —      | —     | stop_loss  |
| 2026-04-20 | MGROS  | 650.15 | —      | —      | —     | stop_loss  |
| 2026-04-20 | GARAN  | 139.40 | —      | —      | —     | stop_loss  |
| 2026-04-20 | YKBNK  | 37.50  | —      | —      | —     | stop_loss  |
| 2026-04-20 | EKGYO  | 21.22  | —      | —      | —     | stop_loss  |

Cumulative P&L: ~+886 TL | Win rate: hesaplanacak (5 stop-loss kapatması dahil edilmeli)
Market in correction since April 14. Tüm pozisyonlar ~2026-04-20 stop-loss ile kapandı.

## VIOP Module (completely independent from spot system)
```
python3.12 viop_data.py          # fetch XU030F OHLCV -> viop_ohlcv
python3.12 viop_indicators.py    # compute indicators -> viop_indicators
python3.12 viop_trader.py        # production 60s loop
python3.12 viop_trader.py --once # safe single-tick test
python3.12 viop_journal.py       # EOD summary (run at 18:30)
```

**File map:**
```
viop_data.py        Fetches XU030.IS (yfinance); GARAN.IS proxy if all fail
viop_indicators.py  EMA8/21, RSI14, ATR14, MACD(12/26/9), BB(20) for viop_ohlcv
viop_oms.py         VIOP order lifecycle: LONG_OPEN/SHORT_OPEN/LONG_CLOSE/SHORT_CLOSE
viop_trader.py      60s loop; rule-based signals; no ML model
viop_journal.py     EOD P&L summary + Telegram
```

**Config (viop_trader.py top):**
```
VIOP_CAPITAL         = 20,000 TL
VIOP_LEVERAGE        = 2              # 1:2 max
VIOP_MAX_POSITIONS   = 2
VIOP_MARGIN_PCT      = 0.15           # 15% margin requirement
VIOP_STOP_PCT        = 0.02           # 2% stop loss
VIOP_TARGET_PCT      = 0.03           # 3% target
VIOP_KILL_DAILY_LOSS = 0.05           # 5% daily loss -> halt
```

**DB tables (all new — spot tables untouched):**
```
viop_ohlcv      (symbol, date PK, open, high, low, close, volume, source)
viop_indicators (symbol, date PK, close, ema8, ema21, rsi14, atr14,
                 macd_line, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower, bb_width)
viop_orders     (order_id PK, symbol, direction, size_tl, contracts,
                 entry_price, exit_price, status, created_at, filled_at, reason, rejected_reason)
viop_positions  (id PK, symbol, direction, entry_date, entry_price,
                 contracts, size_tl, stop_price, target_price,
                 status, exit_date, exit_price, pnl, exit_reason)
```

**Signal logic (rule-based, no ML):**
- LONG:  RSI < 40 AND price < BB lower AND MACD bullish cross
- SHORT: RSI > 65 AND price > BB upper AND MACD bearish cross
- Size: `floor(20000 × 2 × 0.30 / price)` contracts (30% of leveraged capital)

**Isolation guarantee:** viop_* files do NOT import from loop_trader, paper_trade, oms, kill_switch, portfolio_risk, or volatility_regime. They only import telegram_bot (read-only) and logger (read-only). KILL_SWITCH.txt stops the spot system only.

## P3 Status
- **P3.1 Weekly retrain**: DONE. `weekly_retrain.py` — 5-step pipeline: fetch_data → indicators → backup → ml_train → validate. PASS keeps new model + Telegram success; FAIL restores backup + Telegram alert. Backup dir: `models/backup/xgb_model_YYYYMMDD.pkl`. Cron: `0 8 * * 0 cd ~/Desktop/CS/Trade && python3.12 weekly_retrain.py >> logs/weekly_retrain.log 2>&1`.
- **P3.2 Tail risk**: DONE. A) `_check_intraday_tail_risk()`: GARAN 5-min bars, -3% in last 30min → EXTREME override this tick (no new entries). B) `_handle_daily_stop()`: today's equity loss >= 8% of day-start → close all + KILL_SWITCH.txt "DAILY_STOP_8PCT". Constants: `TAIL_RISK_DROP_PCT=0.03`, `DAILY_STOP_PCT=0.08`.
- **P3.3 Universe expansion**: DONE. 14 → 29 stocks (added THYAO/EREGL/KCHOL/SAHOL/SISE/TOASO/ARCLK/VESTL/KRDMD/PGSUS/ODAS/GUBRF/CIMSA/LOGO/NETAS; KOZAL delisted). Model retrained on 29-stock universe (WF 77.0%). THYAO/EREGL moved from vol-proxy-only to full trading universe.
- **P3.4 SHORT capability**: DONE. scan_and_trade() adds SHORT scan (prob_sell≥0.65, RSI>60); check_exits() handles LONG/SHORT separately; _pos_lwm dict for SHORT trailing; OMS uses SHORT/COVER directions (both go through kill switch). MAX_SHORT_SIZE=5,000 TL.
- **P3.5 VPS migration**: Linux VPS (Hetzner €4-6/mo); systemd services for loop_trader + news_filter; `requirements.txt` first.
- **P3.6→P4 Algolab live**: fill 3 lines in `algolab_stream.py`; first phase 10,000 TL real, max 3 positions; add LIMIT order support to `oms.py`.

## Algolab Integration (algolab_stream.py)
- **Current**: MOCK mode (API key empty) — all data from yfinance (15-min delayed).
- **Go live**: set `ALGOLAB_API_KEY`, `ALGOLAB_USERNAME`, `ALGOLAB_PASSWORD` in `algolab_stream.py`. Nothing else changes.
- Login requires pycryptodome (`pip install pycryptodome --break-system-packages`) + SMS 2FA one-time.

## Hard Rules
- **NEVER read trade_data.db with Read tool** — use `sqlite3 trade_data.db "SELECT..."` or Python.
- **NEVER read *.db *.pkl *.csv *.png** with Read tool — use Python subprocess or Bash.
- **Always use python3.12** (Mac) / py -3.12 (Windows). Never python or python3.
- **After every code change**: run the file to verify before reporting done.
- **Never break loop_trader main loop** — test all changes with `--once` first.
- **paper_positions.status is lowercase** — always `WHERE status='open'` not `'OPEN'`.
- **SELL orders never hit kill switch** — exits must always execute.
- **Don't create SentimentAnalyzer() twice** — reuse `from sentiment import analyzer`.

## DB Schema (Key Tables)
```sql
paper_positions (id, symbol, strategy, entry_date, entry_price, size_tl, is_long,
                 stop_price, target_price, confidence, status, partial_exit_done,
                 signal_prob, regime, entry_note)
                 -- status: 'open' / 'closed' (LOWERCASE)

paper_trades    (id, symbol, strategy, entry_date, exit_date, entry_price, exit_price,
                 size_tl, is_long, pnl, pct_return, exit_reason, signal_prob, regime,
                 entry_note, exit_note)

ohlcv           (symbol, date, open, high, low, close, volume, source)
                 -- 21 symbols (14 active + 7 legacy); latest date: 2026-12-03 (yfinance)

indicators      (symbol, date, close, ema8, ema21, ema50, ema200, sma20, rsi14,
                 macd_line, macd_signal, macd_hist, atr14, bb_upper, bb_mid, bb_lower,
                 bb_width, obv, vol_ratio, mtf_trend, rsi_zone, bb_zone, above_ema200,
                 golden_cross, sentiment_score, news_count)

macro_data      (date PK, usdtry, brent, tcmb_rate)
                 -- Fetched by indicators.py at run time: TRY=X (usdtry) + BZ=F (brent) via yfinance
                 -- tcmb_rate hardcoded at 43.0 — UPDATE MANUALLY after each TCMB meeting
                 -- Used as ML features: usdtry_1d/5d_return, usdtry_above_20ma,
                 --   brent_1d/5d_return (×1.5 for TUPRS/PETKM), tcmb_rate (static context)

paper_equity    (date PK, cash, positions_value, total_equity, daily_pnl, open_positions)
                 -- WARNING: only updated by paper_trade.py --run, NOT loop_trader

order_log       (order_id PK, symbol, direction, requested_price, requested_shares,
                 size_tl, filled_price, filled_shares, status, reason, rejected_reason,
                 created_at, filled_at)

equity_tracker  (id, equity_peak, updated_at)
                 -- peak=100,000 TL (çözüldü); eski quirk: 169,553 TL artifact → DD ~43% → KILL tier
```

## Crypto Module (added 2026-04-24, upgraded 2026-04-25)
```
python3.12 crypto_trader.py           # production loop 7/24 (3-thread daemon)
python3.12 crypto_trader.py --once    # WS connect + 1 signal scan + 1 exit check, exit
python3.12 crypto_status.py           # live dashboard: tiers, positions, all 10 coins
python3.12 crypto_journal.py          # daily P&L summary at 00:00 UTC
python3.12 crypto_ml.py               # train/retrain ML model
python3.12 crypto_ml.py --eval        # evaluate existing model
python3.12 crypto_weekly_retrain.py   # weekly retrain pipeline (cron Sun 07:00 UTC)
python3.12 crypto_sentiment.py        # test Fear & Greed fetch
python3.12 crypto_stream.py           # REST + WebSocket connection test
python3.12 crypto_indicators.py       # indicator + MTF test on BTC data
```

**Architecture (v3 — 3-thread daemon):**
```
Thread 1 — Signal scanner : every 5 min  | MTF+ML+F&G → open positions (tier gates)
Thread 2 — Exit monitor   : every 15 sec | checks stops/targets via WS price (REST fallback)
Thread 3 — Health monitor : every 60 sec | prints WS status, P&L, positions
Main thread               : sleep(1) keep-alive | KeyboardInterrupt → clean shutdown
WebSocket                 : CryptoWebSocket (Binance combined stream, public, production URL)
                            10-coin ticker stream; auto-reconnect ×5; daemon thread
State file                : logs/crypto_scan_state.json  (last/next scan ts; read by crypto_status.py)
```

**Universe (10 coins — upgraded 2026-04-25):**
```
CRYPTO_MAJOR = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT',
                'AVAX/USDT','SUI/USDT','DOT/USDT','HYPE/USDT']  # HYPE: checked at startup
CRYPTO_RISKY = ['ONDO/USDT','FET/USDT']
CRYPTO_SYMBOLS = CRYPTO_MAJOR + CRYPTO_RISKY                    # 10 total
```
**Note:** HYPE/USDT currently not listed on Binance — auto-removed at startup via check_symbol_on_exchange().

**Config:**
```
CRYPTO_CAPITAL_USDT        = 10000 USDT (paper)
CRYPTO_MAX_POSITIONS       = 6          # major max 5, risky max 1
MAJOR_MAX_SIZE_PCT         = 0.20       # 20% of capital per major position (2000 USDT)
RISKY_MAX_SIZE_PCT         = 0.10       # 10% of capital per risky position (1000 USDT)
CRYPTO_ML_THRESHOLD        = 0.63       # XGBoost BUY prob threshold (MAJOR)
RISKY_ML_THRESHOLD         = 0.72       # higher ML bar for RISKY tier
CRYPTO_MTF_THRESHOLD_ML    = 4          # MTF adj ≥ 4 when ML active (MAJOR)
CRYPTO_MTF_THRESHOLD_RB    = 4          # MTF adj ≥ 4 rule-based only (MAJOR)
RISKY_MTF_THRESHOLD        = 5          # MTF adj ≥ 5 required for RISKY tier
COIN_GROUP_POLICIES        = CORE_MAJOR stop 2.2% target 2.4%;
                             VOL_MAJOR stop 2.5% target 3.0%;
                             RISKY stop 3.0% target 5.0%
CRYPTO_DAILY_LOSS_LIMIT    = 0.05       # 5% of capital → halt
CRYPTO_BTC_CRASH_THRESHOLD = -0.03      # BTC 1h drop < -3% → block all entries
CRYPTO_CORR_THRESHOLD      = 0.85       # BTC-altcoin correlation gate
CRYPTO_RISK_PCT            = 0.03       # 3% capital risk per trade (ATR sizing)
SIGNAL_INTERVAL            = 300        # 5 min between signal scans
EXIT_INTERVAL              = 15         # 15 sec between exit checks
HEALTH_INTERVAL            = 60         # 60 sec between health prints
```

**Tier system (2026-04-27):**
| Tier  | Symbols            | LONG ML thr | SHORT ML thr | MTF thr (ML) | MTF thr (RB) | Max size |
|-------|--------------------|-------------|--------------|--------------|--------------|----------|
| MAJOR | BTC/ETH/BNB/SOL/AVAX/SUI/DOT | 0.70 (fold-median) | 0.65 (fold-median) | 4 | 4 | 2000 USDT (20%) |
| RISKY | ONDO/FET           | 0.70        | 0.75         | 5            | 5            | 1000 USDT (10%) |
- Thresholds are fold-median calibrated from trained artifact; NEUTRAL regime adds +5% to threshold
- major_open ≥ 4 → block new MAJOR entries
- risky_open ≥ 1 → block new RISKY entries

**Exchange:** Binance Testnet REST (PAPER_MODE=True) + Production WS (public market data, no key)
**Library:** ccxt 4.5.50, websocket-client 1.9.0
**Timeframes:** 5m / 15m / 1h (updated from 15m/1h/4h in v3 for faster signal response)
**Fear & Greed API:** api.alternative.me/fng (free, no key, cached 1h)

**File map:**
```
crypto_stream.py          ccxt.binance REST wrapper + CryptoWebSocket class (10-coin stream)
                          WS: wss://stream.binance.com:9443; auto-reconnect ×5
                          check_symbol_on_exchange(symbol) → bool (Binance spot availability)
crypto_config.py          Shared crypto runtime/status config source; mirrors active crypto_trader defaults
crypto_indicators.py      In-memory indicators: EMA/RSI/MACD/BB/ATR/vol_ratio
                          + compute_mtf(stream, symbol) → (long_score, short_score)
                            1h:  price>EMA21 (+2 trend)
                            15m: price>EMA21 (+2 momentum)
                            5m:  RSI<42 (+1), MACD bull_cross (+1)  [max 6 total]
crypto_oms.py             Paper OMS: crypto_positions + crypto_orders; CryptoOMS class
crypto_trader.py          Main v3: 3-thread daemon; WS-first price; MTF+ML+F&G+BTC gate
                          Tier gates: major_open<5, risky_open<1; per-tier thresholds+sizing
crypto_logger.py          4-file rotating logger (50MB×20 each):
                          crypto_trade.log | crypto_scan.log | crypto_system.log | crypto_exit.log
crypto_journal.py         Daily P&L summary + Telegram at 00:00 UTC
crypto_gate_risk_analysis.py Read-only gate blocker + risk_pct scaling analysis from crypto_signal_journal
crypto_ml.py              Directional XGBoost trainer (LONG + SHORT) for crypto module
                          Active features: 20 single-coin OHLCV-computable features (reverted from 28; expansion hurt WF)
                          Training window: 180 days via paginated Binance history (1h + aligned 5m)
                          dominance_ratio=2.0 for sharper directional labels (raised from 1.5)
                          Label: pure outcome-based — no RSI/EMA entry-state filters (leakage fixed 2026-04-27)
                          New directional artifacts are staged with active=False; weekly retrain promotes only accepted sides
                          LONG gate: folds_gte_3_of_5_pass_0.50 AND eligible_mean_wf_gt_0.52
                          SHORT gate: folds_gte_3_of_5_pass_0.50 AND mean_wf_gt_0.50
crypto_sentiment.py       Fear & Greed Index; 1h cache; contrarian modifier
crypto_status.py          Live dashboard: tiers, all 10 coins, tier thresholds, positions+PnL
crypto_weekly_retrain.py  Weekly directional promotion pipeline (cron Sun 07:00 UTC)
                          LONG and SHORT evaluated independently from staged candidates
                          Accepted side → promoted active=True | rejected side → models/rejected/ | active artifact unchanged
```

**DB tables (new — spot/VIOP tables untouched):**
```
crypto_positions (id, symbol, direction, entry_date, entry_price, amount_usdt,
                  amount_coin, stop_price, target_price, status, exit_date,
                  exit_price, pnl_usdt, exit_reason)
                  -- status: 'open' / 'closed' (lowercase)

crypto_orders    (order_id, symbol, side, amount_usdt, amount_coin, price,
                  status, paper_mode, created_at, filled_at, ccxt_order_id)
```

**Signal pipeline (entry conditions):**
1. BTC crash gate: BTC 1h return < -3% → block ALL entries
2. MTF score: 5m/15m/1h fusion (max 6) with independent LONG/SHORT scoring
3. BTC regime master switch: BULL=LONG only | BEAR=SHORT only | NEUTRAL=both with tighter ML threshold (+5%)
4. LONG ML gate: active LONG artifact (active=True) required; MTF≥4 + prob≥long_ml_thresh + F&G<=74
5. SHORT ML gate: active SHORT artifact (active=True) required; MTF≥4 + prob≥short_ml_thresh + BTC!=BULL + F&G<=74
   Policy: PAPER_LONG_ONLY=False, SHORT_MODEL_ACTIVE=True — both LONG and SHORT entries enabled
6. Rule-based fallback (no active artifact): MTF≥4 + RSI filter + BTC regime filter + F&G filter
7. BTC correlation gate (ETH/SOL/BNB): high corr + opposing BTC trend → block
8. WS price used for entries; REST ticker fallback if WS stale (> 10s)

**Fear & Greed handling:**
| F&G range | LONG threshold | SHORT threshold | Size modifier |
|-----------|----------------|-----------------|---------------|
| 0-24 Extreme Fear   | easier (-1 MTF) | harder (+1 MTF) | ×1.3 |
| 25-74 Fear/Greed    | unchanged       | unchanged       | ×1.0 |
| 75-100 Extreme Greed| harder (+1 MTF) | easier (-1 MTF) | ×0.7 |

**Position sizing (ATR-based):**
`size = min(capital×0.03 / (ATR×1.5/price), per-coin cap)` × F&G modifier
Per-coin cap: CORE/VOL_MAJOR 20% of capital, RISKY 5% of capital; tier cap MAJOR 80%, RISKY 10%.

**Exit logic:**
1. Stop loss: LONG cur ≤ entry×0.975 | SHORT cur ≥ entry×1.025
2. Target: LONG cur ≥ entry×1.05 | SHORT cur ≤ entry×0.95
3. Trailing: profit ≥ 3% → trail at HWM×0.98 (LONG) / LWM×1.02 (SHORT)
4. Score flip: MTF long_score < 2 for LONG | MTF short_score < 2 for SHORT
5. Daily loss: today_pnl < -50 USDT (5% of 1000) → halt all entries

**Crypto ML status (2026-04-27):**
- LONG model (`crypto_xgb_long.pkl`): ACTIVE (active=True)
  WF precision: 52.4% | 3/5 folds passing >50% | threshold MAJOR=0.70, RISKY=0.70 (fold-median)
  Training: 180d paginated history, 5-fold validation, 20 features, dominance_ratio=2.0
  Label: pure outcome — ret_future >= 2.5% in 8h AND dominance ≥ 2.0 (no RSI/EMA entry filters)
  Gate: folds_gte_3_of_5_pass_0.50 AND eligible_mean_wf_gt_0.52
- SHORT model (`crypto_xgb_short.pkl`): ACTIVE (active=True)
  WF precision: 51.2% | 3/5 folds passing >50% | threshold MAJOR=0.65, RISKY=0.75 (fold-median)
  Training: same design as LONG, inverted label direction
  Gate: folds_gte_3_of_5_pass_0.50 AND mean_wf_gt_0.50
  Each side promoted independently by weekly retrain; rejected side → models/rejected/

**Isolation guarantee:** crypto_* files import ONLY telegram_bot + logger (read-only).
No imports from loop_trader, paper_trade, oms, kill_switch, portfolio_risk,
volatility_regime, weekly_retrain, viop_*.

**Go live:** Set BINANCE_API_KEY + BINANCE_SECRET in `.env` (loaded via python-dotenv in crypto_stream.py), set PAPER_MODE=False in `.env`.

**Config files:**
- `.env` — BINANCE_API_KEY, BINANCE_SECRET, PAPER_MODE (loaded by crypto_stream.py)
- `.env.example` — şablon, key boş, PAPER_MODE=True

**requirements.txt (2026-04-26):** ccxt>=4.5, websocket-client>=1.8, python-dotenv>=1.0, joblib>=1.4, lightgbm>=4.0, catboost>=1.2, requests>=2.30 eklendi.

**Known issues (crypto):**
- 180-day 5m pagination is materially heavier than the old 41-day fetch; cron/runtime should be monitored for duration and rate limits.
- Feature expansion risk: going 20→28 features hurt LONG WF (51.6%→46.3%). Any FEATURE_NAMES changes must be tested with a dry-run retrain before committing.

## On Compact
Preserve: 29-stock universe (KOZAL delisted), SHORT capability (SHORT_THRESHOLD=0.65, MAX_SHORT_SIZE=5k, RSI>60, _pos_lwm trailing), model metrics (WF 77.1%, traded-only 75.0%, test 62.5%), macro features (USDTRY+Brent+TCMB+strongest_sector_5d added 2026-04-24), 0 open BIST positions (tüm Nisan pozisyonları stop-loss ile kapandı), ticker corrections (MGROS/ENKAI/EKGYO), eval window (2026-04-08→2026-06-08, 58%/1.2/30), equity_tracker peak=100k, P2 risk module constants now centralized in bist_config.py (heat 6%, corr 0.85, vol 2.5%/4.0%), BERTurk hybrid (keywords primary, BERT fills at 0, clamped [-2,+2], lazy Mac load), P3 status all done, paper_positions status lowercase, OMS SHORT/COVER directions both go through kill switch. Crypto module v3: 11 files (config/stream/indicators/oms/trader/journal/ml/sentiment/status/weekly_retrain/logger), 2 DB tables, ccxt Binance testnet REST + production WS, 10000 USDT paper, 10 coins (9 active, HYPE unlisted), 3-thread daemon (signal 5m/exit 15s/health 60s), 5m/15m/1h MTF, 20 features (reverted from 28; expansion hurt WF), dominance_ratio=2.0, pure outcome labels (no RSI/EMA entry filters — leakage fixed). LONG model active WF=52.4% (thr MAJOR=0.70, RISKY=0.70), SHORT model active WF=51.2% (thr MAJOR=0.65, RISKY=0.75), both fold-median calibrated. Acceptance gates: LONG folds_gte_3_of_5 + mean>52%; SHORT folds_gte_3_of_5 + mean>50%. PAPER_LONG_ONLY=False, SHORT_MODEL_ACTIVE=True. crypto_config.py centralizes runtime/status values: max positions 6 (major 5/risky 1), CRYPTO_RISK_PCT=0.03, major exposure 20%, risky max position 5% with risky tier cap 10%. logs/crypto_scan_state.json for status timing. Weekly retrain: REJECTED→models/rejected/; backup→models/backup/. Config: .env (BINANCE_API_KEY/SECRET/PAPER_MODE), .env.example şablon. Feature expansion risk: 20→28 features hurt LONG WF — always dry-run test before adding to FEATURE_NAMES.

Last updated: 2026-04-27 — Both LONG (WF=52.4%) and SHORT (WF=51.2%) models promoted to active. Pure outcome labels (RSI/EMA entry-state leakage removed). Feature set reverted to 20 (from 28). Acceptance gates relaxed to folds_gte_3_of_5. PAPER_LONG_ONLY=False, SHORT_MODEL_ACTIVE=True. 5 open SHORT positions running in paper mode. Mac: python3.12, Windows: py -3.12

BIST model retrained 2026-04-27 with 12 cross-asset macro features (^VIX/EEM/GC=F/DXY/^GSPC/^STOXX50E). All 12 new macro features dropped (gain=0.0); strongest_sector_5d kept (0.0323). WF BUY precision: 77.1%→81.1% (+4pp) — improvement from beta calibration + ffill fix. Feature count stays 49. indicators.py now fetches 8 tickers; macro_data has 20 new columns pre-computed with ffill. loop_trader.py passes macro_df to make_features_single(). paper_trade.py make_features_single() has macro_df=None parameter + macro section.
