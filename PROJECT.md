# PROJECT.md

## Project
BIST algorithmic trading research and paper-trading system.

## Goal
Build a realistic paper-trading system for BIST equities that can later be considered for live deployment only after sufficient validation.

## Main components
- `fetch_data.py` -> fetches BIST market data
- `indicators.py` -> computes technical indicators
- `backtest.py` -> rule-based strategy backtests
- `optimize.py` -> parameter search
- `ml_train.py` -> trains ML model
- `paper_trade.py` -> paper trading engine and position management
- `daily_report.py` -> daily reporting and charts
- `news_filter.py` -> RSS/news risk filter
- `run_daily.bat` -> batch runner

## Database
- `trade_data.db` stores market data and paper-trading state

## Current direction
- Focus on realistic paper trading first
- Improve execution realism
- Improve risk engine
- Audit for data leakage
- Improve ML signal usefulness, not just accuracy

## Principles
- No claim of guaranteed profitability
- Prefer realism over optimistic backtests
- Prefer robust research over fast feature additions