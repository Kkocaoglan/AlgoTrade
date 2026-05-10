#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python3 fetch_data.py
python3 indicators.py
python3 intraday_scan.py
python3 news_filter.py
