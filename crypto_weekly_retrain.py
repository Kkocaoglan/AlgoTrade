"""
crypto_weekly_retrain.py — Weekly auto-retrain for tiered Crypto ML artifacts.

Trains and evaluates four staged artifacts independently:
  - LONG  MAJOR
  - LONG  RISKY
  - SHORT MAJOR
  - SHORT RISKY

Accepted artifacts are promoted independently.
Rejected artifacts are archived with JSON metadata and appended to a shared
retrain history log.
"""

import json
import sys
import shutil
import joblib
from datetime import datetime, date
from pathlib import Path

BASE_DIR = Path(__file__).parent
BACKUP_DIR = BASE_DIR / "models" / "backup"
REJECTED_DIR = BASE_DIR / "models" / "rejected"
RETRAIN_HISTORY_PATH = BASE_DIR / "models" / "retrain_history.jsonl"

for _d in [BACKUP_DIR, REJECTED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from logger import algo_log
from telegram_bot import send_telegram_alert
from crypto_ml import (
    load_tier_model,
    train_tiered_directional,
    _tier_active_model_path,
)


ARTIFACT_ORDER = [
    ("LONG", "MAJOR"),
    ("LONG", "RISKY"),
    ("SHORT", "MAJOR"),
    ("SHORT", "RISKY"),
]


def _backup_if_exists(path: Path, label: str, stamp: str) -> Path | None:
    if not path.exists():
        print(f"[{label}] Backup atlandi — aktif artifact yok")
        return None
    backup_path = BACKUP_DIR / f"{path.stem}_{stamp}.pkl"
    shutil.copy2(path, backup_path)
    print(f"[{label}] Backup: {path.name} -> {backup_path.name}")
    return backup_path


def _promote_candidate(model_dict: dict, dest_path: Path, label: str) -> None:
    promoted = dict(model_dict)
    promoted["active"] = True
    promoted["promoted_at"] = datetime.now().isoformat()
    joblib.dump(promoted, dest_path)
    print(f"[{label}] PROMOTED -> {dest_path.name}")
    algo_log.system(f"crypto_weekly_retrain: promoted {label} artifact {dest_path.name}")


def _archive_rejected(model_dict: dict, direction: str, tier: str, stamp: str) -> Path:
    rejected_path = REJECTED_DIR / f"{tier.lower()}_{direction.lower()}_rejected_{stamp}.pkl"
    joblib.dump(model_dict, rejected_path)
    print(f"[{tier} {direction}] REJECTED archive: {rejected_path.name}")
    return rejected_path


def _rejection_reason(model_dict: dict) -> str:
    failed = list(model_dict.get("failed_checks") or [])
    if not failed:
        return "candidate_rejected"
    if "mean_wf_gt_0.52" in failed or "eligible_mean_wf_gt_0.52" in failed:
        return "mean_wf < 0.52"
    if "folds_gte_3_of_5_pass_0.50" in failed:
        return "folds_passed < 3"
    return ", ".join(failed)


def _write_rejected_json(model_dict: dict, direction: str, tier: str, today_str: str) -> Path:
    folds = model_dict.get("wf_report", {}).get("folds", [])
    payload = {
        "date": today_str,
        "tier": tier,
        "direction": direction,
        "mean_wf_precision": float(model_dict.get("wf_precision", 0.0) or 0.0),
        "folds_passed": int(sum(1 for f in folds if float(f.get("precision", 0.0)) > 0.50)),
        "fold_results": [float(f.get("precision", 0.0) or 0.0) for f in folds],
        "rejection_reason": _rejection_reason(model_dict),
        "threshold_used": float(
            (
                (model_dict.get("threshold_config") or {})
                .get(tier, {})
                .get("selected_threshold")
            ) or (
                (model_dict.get("threshold_config") or {})
                .get(tier, {})
                .get("fallback_threshold", 0.0)
            ) or 0.0
        ),
        "feature_count": int(model_dict.get("feature_count", 0) or 0),
    }
    json_path = REJECTED_DIR / f"rejected_{direction}_{today_str.replace('-', '')}_{tier.lower()}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def _append_retrain_history(model_dict: dict, direction: str, tier: str, result: str, today_str: str) -> None:
    folds = model_dict.get("wf_report", {}).get("folds", [])
    row = {
        "date": today_str,
        "tier": tier,
        "direction": direction,
        "result": result,
        "mean_wf_precision": float(model_dict.get("wf_precision", 0.0) or 0.0),
        "folds_passed": int(sum(1 for f in folds if float(f.get("precision", 0.0)) > 0.50)),
        "feature_count": int(model_dict.get("feature_count", 0) or 0),
        "rejection_reason": None if result == "accepted" else _rejection_reason(model_dict),
    }
    with RETRAIN_HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _load_active_metrics(direction: str, tier: str) -> dict:
    model = load_tier_model(direction.lower(), tier)
    if not model:
        return {"direction": direction, "tier": tier, "active": False}
    return {
        "direction": direction,
        "tier": tier,
        "active": bool(model.get("active", False)),
        "wf_precision": float(model.get("wf_precision", 0.0) or 0.0),
        "trained_at": model.get("trained_at", "unknown"),
    }


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    today_str = date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"CRYPTO WEEKLY RETRAIN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    for direction, tier in ARTIFACT_ORDER:
        active_path = _tier_active_model_path(direction.lower(), tier)
        label = f"{tier} {direction}"
        _backup_if_exists(active_path, label, stamp)

    try:
        staged = train_tiered_directional()
    except Exception as exc:
        print(f"[FAIL] tiered train exception: {exc}")
        algo_log.system(f"crypto_weekly_retrain: FAIL tiered train exception {exc}")
        send_telegram_alert(f"Crypto Weekly Retrain error: {exc}")
        return 1

    final_rows: list[str] = []
    for direction, tier in ARTIFACT_ORDER:
        key = f"{tier}_{direction}"
        model_dict = staged.get(key) or {}
        label = f"{tier} {direction}"
        if not model_dict:
            print(f"[{label}] candidate missing")
            continue

        dest_path = _tier_active_model_path(direction.lower(), tier)
        if bool(model_dict.get("promotion_ready", False)):
            _promote_candidate(model_dict, dest_path, label)
            _append_retrain_history(model_dict, direction, tier, "accepted", today_str)
            result_txt = "ACCEPT"
        else:
            _archive_rejected(model_dict, direction, tier, stamp)
            rejected_json = _write_rejected_json(model_dict, direction, tier, today_str)
            print(f"[{label}] rejection meta: {rejected_json.name}")
            _append_retrain_history(model_dict, direction, tier, "rejected", today_str)
            result_txt = f"REJECT ({_rejection_reason(model_dict)})"

        final_rows.append(
            f"{tier} {direction}: {result_txt} | WF={float(model_dict.get('wf_precision', 0.0)):.1%}"
        )

    print("\n[FINAL RUNTIME SUMMARY]")
    for row in final_rows:
        print(f"  {row}")

    send_telegram_alert("Crypto Weekly Retrain:\n" + "\n".join(final_rows))
    return 0


if __name__ == "__main__":
    rc = main()
    print(f"\n{'='*60}")
    print(f"crypto_weekly_retrain: cikis kodu {rc}")
    print(f"{'='*60}\n")
    sys.exit(rc)
