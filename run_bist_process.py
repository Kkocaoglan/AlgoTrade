"""
run_bist_process.py -- One-command BIST daily process.

Default flow:
  1. Data-quality precheck
  2. Fetch BIST data
  3. Recompute indicators/macro
  4. Data-quality postcheck
  5. Run loop_trader continuously

One-shot research flow:
  add --once to run loop_trader.py --once and then run the simulator.

Usage:
  python3.12 run_bist_process.py
  python3.12 run_bist_process.py --once
  python3.12 run_bist_process.py --train --optimize-threshold
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).parent
DEFAULT_THRESHOLDS = "0.60,0.65,0.70,0.73"


def _run(label: str, args: list[str], keep_going: bool = False) -> bool:
    print()
    print("=" * 88)
    print(f"[BIST] {label}")
    print("=" * 88)
    print(" ".join(args))
    started = time.time()
    proc = subprocess.run(args, cwd=BASE_DIR)
    elapsed = time.time() - started
    if proc.returncode == 0:
        print(f"[OK] {label} ({elapsed:.1f}s)")
        return True

    print(f"[FAIL] {label} exit={proc.returncode} ({elapsed:.1f}s)")
    if keep_going:
        print("[WARN] keep-going aktif, sonraki adima geciliyor.")
        return False
    raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BIST daily process from one command.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch_data.py")
    parser.add_argument("--skip-indicators", action="store_true", help="Skip indicators.py")
    parser.add_argument("--skip-quality", action="store_true", help="Skip bist_data_quality.py checks")
    parser.add_argument("--skip-loop", action="store_true", help="Skip loop_trader.py")
    parser.add_argument("--skip-sim", action="store_true", help="Skip bist_live_wf_sim.py")
    parser.add_argument("--once", action="store_true", help="Run loop_trader.py --once and then run simulator")
    parser.add_argument("--train", action="store_true", help="Run ml_train.py after indicators")
    parser.add_argument("--optimize-threshold", action="store_true", help="Run optimize_threshold.py after training")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated simulator thresholds")
    parser.add_argument("--keep-going", action="store_true", help="Continue after non-critical failures")
    args = parser.parse_args()

    py = sys.executable
    print("=" * 88)
    print("BIST DAILY PROCESS")
    print("=" * 88)
    print(f"Python     : {py}")
    print(f"Workspace  : {BASE_DIR}")
    print(f"Thresholds : {args.thresholds}")

    if not args.skip_quality:
        _run("Data-quality precheck", [py, "bist_data_quality.py"], keep_going=True)

    if not args.skip_fetch:
        _run("Fetch BIST OHLCV", [py, "fetch_data.py"], keep_going=args.keep_going)

    if not args.skip_indicators:
        _run("Compute indicators and macro", [py, "indicators.py"], keep_going=args.keep_going)

    if not args.skip_quality:
        _run("Data-quality postcheck", [py, "bist_data_quality.py"], keep_going=True)

    if args.train:
        _run("Train BIST model", [py, "ml_train.py"], keep_going=args.keep_going)
        if args.optimize_threshold:
            _run("Optimize model threshold", [py, "optimize_threshold.py"], keep_going=args.keep_going)
    elif args.optimize_threshold:
        _run("Optimize model threshold", [py, "optimize_threshold.py"], keep_going=args.keep_going)

    if not args.skip_loop:
        if args.once:
            _run("Run BIST loop once", [py, "loop_trader.py", "--once"], keep_going=args.keep_going)
        else:
            print()
            print("[INFO] Continuous mode: simulator is skipped until loop_trader.py exits.")
            _run("Run BIST loop continuously", [py, "loop_trader.py"], keep_going=args.keep_going)
            args.skip_sim = True

    if not args.skip_sim:
        _run(
            "Run live-rule simulator",
            [py, "bist_live_wf_sim.py", "--thresholds", args.thresholds],
            keep_going=args.keep_going,
        )

    print()
    print("=" * 88)
    print("[DONE] BIST daily process completed")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
