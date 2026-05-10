# AUDIT.md

## Repository Architecture

This repository is a SQLite-centered BIST research and paper-trading system.

- `fetch_data.py` fetches OHLCV data into `trade_data.db`
- `indicators.py` computes technical indicators into the same DB
- `backtest.py` runs rule-based strategy simulations on `ohlcv + indicators`
- `optimize.py` performs parameter search for rule-based strategies
- `ml_train.py` builds features, labels targets, trains the XGBoost model, and emits signals
- `paper_trade.py` loads the trained model, opens/closes paper positions, and tracks portfolio state
- `daily_report.py` summarizes paper-trade state and generates reporting output
- `news_filter.py` monitors RSS/news risk signals
- `run_daily.bat` orchestrates the daily workflow

Primary state is stored in:

- `ohlcv`
- `indicators`
- `paper_positions`
- `paper_trades`
- `paper_equity`

## Data Leakage Risks

- `ml_train.py::make_target()` labels each row with future closes via `shift(-horizon)`, but split logic in `train_main_model()` and `walk_forward()` does not purge the last `HORIZON` rows before each boundary.
- This means late training rows can use labels whose outcomes occur inside the test period.
- `ml_train.py::walk_forward()` is not a proper purged forward-chaining validation; it uses block-internal 70/30 splits without embargo.
- Evaluation excludes neutral rows (`NaN` targets), so reported metrics are on a filtered binary subset rather than the full decision universe.
- `indicators.py::bb_zone()` uses full-series quantiles, which is non-causal. It is not the main ML feature path today, but it is still a future-information risk.
- Feature logic exists in both `ml_train.py` and `paper_trade.py`, creating feature-parity drift risk between training and live inference.

## Execution Realism Risks

- `backtest.py` enters trades on the same bar close used to generate the signal.
- `paper_trade.py::run_daily()` also opens positions using the latest known close, implying same-bar execution.
- Stop/target logic in both `backtest.py` and `paper_trade.py` is close/snapshot-based rather than open-high-low-close barrier-aware.
- Overnight gaps are not modeled correctly; stop-through and target-through fills are optimistic.
- `paper_trade.py::watch_mode()` auto-closes positions using delayed `yfinance` intraday data labeled as roughly 15 minutes delayed.
- Slippage and costs are fixed constants, not volatility-, spread-, liquidity-, or urgency-aware.
- Partial exits and trailing stops assume frictionless perfect fills at the sampled price.
- Daily and intraday execution assumptions are mixed, which makes paper-trade behavior internally inconsistent.

## Code Quality Issues

- Core trading logic is duplicated across files instead of centralized.
- Feature engineering is duplicated in `ml_train.py` and `paper_trade.py`.
- Risk, sizing, valuation, and exit logic are partially repeated between backtest and paper-trade paths.
- Encoding issues affect readability in multiple files.
- Repository is not a git repository, reducing traceability and change safety.
- Operational/config hygiene is weak; there is a likely typo file under `.claude` (`settsings.json`).
- Some declared controls appear incomplete or unused, which increases audit difficulty.

## Operational Risks

- `run_daily.bat` chains fetch -> indicators -> paper trade immediately, which encourages trading on freshly completed bars without explicit next-bar execution rules.
- `news_filter.py` produces warnings but is not a hard execution gate for new trades.
- There is no formal migration layer for DB schema evolution.
- There is no visible test suite for feature causality, split correctness, or execution logic.
- Portfolio controls are limited; there is no robust daily loss cap, portfolio stop-out, gross/net exposure control, or correlated factor control.
- The system mixes research outputs, artifacts, and live-like operational state in one workspace.

## Top Priority Fixes

1. Remove temporal leakage from training and validation by purging `HORIZON` rows at split boundaries.
2. Replace current walk-forward logic with purged rolling or expanding-window validation.
3. Enforce feature parity between training and live inference from a single shared implementation.
4. Move entries to next-bar execution and make exits gap-aware using OHLC logic.
5. Stop using delayed intraday quotes as executable auto-close prices.
6. Upgrade transaction cost and slippage modeling to per-side, volatility-aware assumptions.
7. Add portfolio-level risk limits: daily max loss, exposure caps, concentration limits, and cash buffer rules.
8. Turn news/risk signals into enforceable trade-gating controls.
9. Clean repository hygiene: encoding, config ambiguity, and version-control baseline.

## Suggested Next Audit Order

1. Data leakage and evaluation validity
2. Execution realism and fill assumptions
3. Portfolio risk engine design
4. ML signal quality under traded-only thresholds
5. Longer paper-trade validation protocol
