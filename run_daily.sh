#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python3 fetch_data.py
python3 indicators.py
python3 ml_train.py
python3 paper_trade.py --run
python3 daily_report.py
