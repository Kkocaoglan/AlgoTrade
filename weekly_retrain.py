"""
weekly_retrain.py -- P3.1 Weekly Auto-Retrain Pipeline
-------------------------------------------------------
Run : python3.12 weekly_retrain.py
Cron: 0 8 * * 0  cd ~/Desktop/CS/Trade && python3.12 weekly_retrain.py >> logs/weekly_retrain.log 2>&1

Pipeline:
  1. fetch_data.py       -- update OHLCV
  2. indicators.py       -- recompute all technical features
  3. backup model        -- models/xgb_model.pkl -> models/backup/xgb_model_YYYYMMDD.pkl
  4. ml_train.py         -- retrain XGB+LGB+CatBoost ensemble; capture WF BUY precision
  5. validate_model.py   -- 4-test validation (lookahead, net prec, features, survivorship)
  6. meta_labeler.py --train  -- retrain secondary BUY filter on wf_predictions.csv
  7a. PASS -> keep new model; Telegram success summary with new WF precision
  7b. FAIL -> restore backup model; Telegram alert "Weekly retrain FAILED"
  8. All steps logged to system.log via algo_log

Validation PASS criteria (validate_model.py):
  T1 Lookahead  : "TEMIZ" (diff <= 5pp)
  T2 Net prec   : "GECERLI" (net precision >= 58%)
  T3 Features   : "TEMIZ" (no RED flags)
  T4 Survivor   : "TEMIZ" (all symbols >= 800 rows)
  -> Overall "DEVAM ET" in output = PASS
"""

import sys, shutil, subprocess, re
from datetime import datetime, date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
BACKUP_DIR = BASE_DIR / "models" / "backup"
LOG_DIR    = BASE_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# -- Logger and Telegram (imported after path setup) -------------------------
sys.path.insert(0, str(BASE_DIR))
from logger import algo_log
from telegram_bot import send_telegram_alert

PYTHON = sys.executable   # always python3.12 in the correct venv


# =============================================================================
# HELPERS
# =============================================================================

def _run(label: str, script: str | list[str], timeout: int = 600) -> tuple[bool, str]:
    """
    Run BASE_DIR/script as a subprocess.
    Returns (success: bool, stdout: str).
    Prints progress; logs to system.log.
    """
    print(f"\n[STEP] {label} ...", flush=True)
    algo_log.system(f"weekly_retrain: START {label}")
    try:
        cmd = [PYTHON]
        if isinstance(script, (list, tuple)):
            cmd.extend(str(part) for part in script)
        else:
            cmd.append(str(BASE_DIR / script))
        r = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(BASE_DIR),
        )
        if r.returncode != 0:
            stderr_tail = (r.stderr or "").strip()[-400:]
            print(f"  [FAIL] {label} exit={r.returncode}")
            if stderr_tail:
                print(f"  stderr: {stderr_tail}")
            algo_log.system(
                f"weekly_retrain: FAIL {label} exit={r.returncode} "
                f"stderr={stderr_tail[:200]}"
            )
            return False, r.stdout or ""
        out_tail = (r.stdout or "").strip()[-200:]
        print(f"  [OK]   {label}")
        algo_log.system(f"weekly_retrain: OK {label}")
        return True, r.stdout or ""

    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {label} — TIMEOUT ({timeout}s)")
        algo_log.system(f"weekly_retrain: TIMEOUT {label} ({timeout}s)")
        return False, ""

    except Exception as e:
        print(f"  [FAIL] {label} — EXCEPTION: {e}")
        algo_log.system(f"weekly_retrain: EXCEPTION {label}: {e}")
        return False, ""


def _parse_wf_precision(stdout: str) -> str:
    """Extract 'Ortalama BUY precision : X.X%' from ml_train.py output."""
    m = re.search(r"Ortalama BUY precision\s*:\s*([\d.]+)%", stdout)
    return (m.group(1) + "%") if m else "?"


def _parse_traded_precision(stdout: str) -> str:
    """Extract 'BUY precision : X.X%' (traded-only) from ml_train.py output."""
    # ml_train prints: "BUY precision : 90.1%  Neutral rate: ..."
    m = re.search(r"BUY precision\s*:\s*([\d.]+)%\s+Neutral", stdout)
    return (m.group(1) + "%") if m else "?"


def _validation_passed(stdout: str) -> bool:
    """
    True if validate_model.py overall verdict is PASS.
    Checks for 'DEVAM ET' (in report) or 'EVET' (in 'kullanmaya devam et' line).
    """
    return "DEVAM ET" in stdout or "kullanmaya devam et : EVET" in stdout


def _extract_verdicts(stdout: str) -> list[str]:
    """Pull the 4 'Sonuc:' lines from validate_model.py output."""
    return re.findall(r"Sonuc:\s*(.+)", stdout)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> int:
    today_str = date.today().isoformat()
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"WEEKLY RETRAIN PIPELINE  {now_str}")
    print(f"{'='*60}")
    algo_log.system(f"weekly_retrain: pipeline started {now_str}")

    backup_path: Path | None = None

    # ── STEP 1: fetch_data.py ────────────────────────────────────
    ok, _ = _run("fetch_data.py (OHLCV guncelleme)", "fetch_data.py", timeout=300)
    if not ok:
        msg = f"Weekly retrain FAILED at fetch_data.py — {today_str}"
        algo_log.system(f"weekly_retrain: ABORT {msg}")
        send_telegram_alert(f"Weekly Retrain HATA\n{msg}")
        return 1

    # ── STEP 2: indicators.py ────────────────────────────────────
    ok, _ = _run("indicators.py (indikatör yeniden hesaplama)", "indicators.py", timeout=120)
    if not ok:
        msg = f"Weekly retrain FAILED at indicators.py — {today_str}"
        algo_log.system(f"weekly_retrain: ABORT {msg}")
        send_telegram_alert(f"Weekly Retrain HATA\n{msg}")
        return 1

    # ── STEP 3: Backup current model ─────────────────────────────
    print(f"\n[BACKUP] Mevcut model yedekleniyor...")
    if MODEL_PATH.exists():
        stamp       = datetime.now().strftime("%Y%m%d")
        backup_path = BACKUP_DIR / f"xgb_model_{stamp}.pkl"
        shutil.copy2(MODEL_PATH, backup_path)
        size_mb = backup_path.stat().st_size / 1024 / 1024
        print(f"  {MODEL_PATH.name} -> {backup_path}  ({size_mb:.1f} MB)")
        algo_log.system(f"weekly_retrain: backup created {backup_path.name} {size_mb:.1f}MB")
    else:
        print(f"  [WARN] {MODEL_PATH} bulunamadi — yedek alinamadi")
        algo_log.system("weekly_retrain: no existing model found — skipping backup")

    # ── STEP 4: ml_train.py ──────────────────────────────────────
    ok, train_out = _run("ml_train.py (ensemble retrain)", "ml_train.py", timeout=600)
    if not ok:
        _restore(backup_path)
        msg = f"Weekly retrain FAILED at ml_train.py — backup restored — {today_str}"
        algo_log.system(f"weekly_retrain: ABORT {msg}")
        send_telegram_alert(f"Weekly Retrain HATA\n{msg}")
        return 1

    wf_prec      = _parse_wf_precision(train_out)
    traded_prec  = _parse_traded_precision(train_out)
    print(f"  WF BUY precision    : {wf_prec}")
    print(f"  Traded-only prec    : {traded_prec}")

    # ── STEP 5: validate_model.py ────────────────────────────────
    ok, val_out = _run(
        "validate_model.py (4-test dogrulama)",
        "validate_model.py", timeout=600,
    )
    if not ok:
        # Script crash (not validation failure) — restore for safety
        _restore(backup_path)
        msg = (
            f"Weekly retrain FAILED: validate_model.py crash — "
            f"backup restored — {today_str}"
        )
        algo_log.system(f"weekly_retrain: ABORT {msg}")
        send_telegram_alert(f"Weekly Retrain HATA\nvalidate_model.py coktu\n{msg}")
        return 1

    # ── STEP 6: Keep or restore ──────────────────────────────────
    passed   = _validation_passed(val_out)
    verdicts = _extract_verdicts(val_out)   # ["TEMIZ: ...", "GECERLI: ...", ...]

    if passed:
        meta_ok, meta_out = _run(
            "meta_labeler.py --train (secondary BUY filter)",
            [str(BASE_DIR / "meta_labeler.py"), "--train"], timeout=600,
        )
        if not meta_ok:
            print("  [WARN] meta_labeler train basarisiz -- primary model korunuyor")
            algo_log.system("weekly_retrain: WARN meta_labeler train failed")

        print(f"\n[RESULT] ✓ Dogrulama GECTI — yeni model aktif")
        algo_log.system(
            f"weekly_retrain: PASSED wf={wf_prec} traded={traded_prec} "
            f"verdicts={verdicts}"
        )

        verdict_lines = "\n".join(
            f"  T{i+1}: {v[:50]}" for i, v in enumerate(verdicts)
        ) if verdicts else "  (parcalanamadi)"

        send_telegram_alert(
            f"Weekly Retrain BASARILI\n"
            f"Tarih         : {today_str}\n"
            f"WF precision  : {wf_prec}\n"
            f"Traded prec   : {traded_prec}\n"
            f"Dogrulama     :\n{verdict_lines}\n"
            f"Model aktif   : models/xgb_model.pkl"
        )
        return 0

    else:
        print(f"\n[RESULT] ✗ Dogrulama BASARISIZ — yedek geri yukleniyor...")
        algo_log.system(
            f"weekly_retrain: FAILED validation wf={wf_prec} "
            f"verdicts={verdicts} — restoring backup"
        )
        _restore(backup_path)

        verdict_lines = "\n".join(
            f"  T{i+1}: {v[:50]}" for i, v in enumerate(verdicts)
        ) if verdicts else "  (parcalanamadi)"

        send_telegram_alert(
            f"Weekly retrain FAILED — backup restored\n"
            f"Tarih          : {today_str}\n"
            f"WF precision   : {wf_prec}\n"
            f"Dogrulama      :\n{verdict_lines}\n"
            f"Geri yuklendi  : {backup_path.name if backup_path else 'N/A'}"
        )
        return 1


def _restore(backup_path: "Path | None") -> None:
    """Restore backup model to MODEL_PATH, or warn if no backup exists."""
    if backup_path and backup_path.exists():
        shutil.copy2(backup_path, MODEL_PATH)
        print(f"  [RESTORE] {backup_path.name} -> {MODEL_PATH.name}")
        algo_log.system(f"weekly_retrain: restored {backup_path.name}")
    else:
        print(f"  [WARN] Yedek dosyasi yok — eski model geri yuklenemedi!")
        algo_log.system("weekly_retrain: WARN no backup to restore")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    rc = main()
    print(f"\n{'='*60}")
    print(f"weekly_retrain: cikis kodu {rc}  ({'BASARILI' if rc == 0 else 'HATA'})")
    print(f"{'='*60}\n")
    sys.exit(rc)
