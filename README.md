# Algo-Trade

A Python-based BIST algorithmic trading research and paper-trading project for data collection, indicator generation, model training, paper trading, reporting, intraday scans, and trading journal workflows.

## Project Overview

The repository contains scripts for:

- Fetching market data.
- Calculating indicators and model features.
- Training machine learning models.
- Running paper-trading workflows.
- Generating daily reports.
- Running intraday scans.
- Maintaining a trading journal.
- Sending optional Telegram notifications.

## Setup: Mac / Linux

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make the shell scripts executable:

```bash
chmod +x run_daily.sh run_intraday.sh run_journal.sh run_morning.sh
```

## Setup: Windows

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Manually

Daily pipeline:

```bash
python3 fetch_data.py
python3 indicators.py
python3 ml_train.py
python3 paper_trade.py --run
python3 daily_report.py
```

Intraday scan:

```bash
python3 intraday_scan.py
```

Journal:

```bash
python3 daily_journal.py
```

Morning workflow:

```bash
python3 fetch_data.py
python3 indicators.py
python3 intraday_scan.py
python3 news_filter.py
```

On Windows, use `py -3.12` instead of `python3` when running the same scripts manually.

## Convenience Scripts

Mac / Linux:

```bash
./run_daily.sh
./run_intraday.sh
./run_journal.sh
./run_morning.sh
```

Windows:

Use the existing `.bat` files.

## Crypto Module

The crypto module runs completely independently from the BIST spot and VIOP systems.
It trades 10 coins on Binance (paper mode by default — no real money, no API key needed).

### First-time setup

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

`PAPER_MODE=True` (the default) uses Binance Testnet REST for order placement and the
public production WebSocket for market data — both work without API keys.

### Run the crypto trader (7/24 daemon)

```bash
# Mac
python3.12 crypto_trader.py          # production loop (3-thread daemon)
python3.12 crypto_trader.py --once   # single signal scan + exit check, then exit

# Windows
py -3.12 crypto_trader.py
py -3.12 crypto_trader.py --once
```

### Live dashboard

```bash
python3.12 crypto_status.py          # positions, tiers, all 10 coins, next scan time
```

### ML model

```bash
python3.12 crypto_ml.py              # train + save to models/crypto_xgb.pkl
python3.12 crypto_ml.py --eval       # evaluate existing model

python3.12 crypto_weekly_retrain.py  # full retrain pipeline with acceptance gate
```

Weekly retrain acceptance gate (all four must pass):

| Check | Threshold |
|---|---|
| WF BUY precision | >= 55% |
| Total WF trades | >= 12 |
| Net expectancy after costs | > 0.0 |
| Degradation vs active model | <= 5pp |

Rejected models are saved to `models/rejected/` for forensics.
Backups are saved to `models/backup/` before every retrain.

### Connection tests

```bash
python3.12 crypto_stream.py          # REST + WebSocket connection test
python3.12 crypto_sentiment.py       # Fear & Greed fetch test
python3.12 crypto_indicators.py      # indicator + MTF test on BTC data
```

### Daily journal (run at 00:00 UTC)

```bash
python3.12 crypto_journal.py
```

### Go live (real money)

Set `PAPER_MODE=False` in `.env` and add your Binance API credentials.
Nothing else in the codebase changes.

## Notes

- `.bat` files are Windows-only.
- `.sh` files are for Mac/Linux.
- Generated files such as `results/`, `logs/`, `trade_data.db`, `__pycache__/`, `.claude/`, and `.codex/` are ignored in Git.
- `results/xgb_model.pkl` may need to be regenerated if package versions differ across machines. Retrain the model with `ml_train.py` if loading the model fails.
