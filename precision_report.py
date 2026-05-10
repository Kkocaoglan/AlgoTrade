"""
precision_report.py — Model precision / recall report from signals_log
-----------------------------------------------------------------------
Queries signals_log where outcome_3d IS NOT NULL and calculates:

  Precision = TP / (TP + FP)
            = (threshold_passed=1 AND outcome_3d=1) / (threshold_passed=1)

  Recall    = TP / (TP + FN)
            = (threshold_passed=1 AND outcome_3d=1) / (outcome_3d=1)

  F1        = 2 * precision * recall / (precision + recall)

Breakdown by:
  - Overall
  - Symbol
  - Regime (NORMAL / HIGH_VOL / EXTREME / UNKNOWN)

Baseline comparison: random classifier precision = base_rate
  base_rate = fraction of all resolved signals where outcome_3d=1

Usage:
  python3.12 precision_report.py
  python3.12 precision_report.py --min-samples 10   # hide rows with < N samples
"""

import sqlite3
import sys

DB_PATH = "trade_data.db"


# ── helpers ──────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> tuple:
    """Return (precision, recall, f1) — all as floats 0..1."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _row(label: str, tp: int, fp: int, fn: int, base_rate: float, width: int = 20) -> str:
    prec, rec, f1 = _prf(tp, fp, fn)
    total         = tp + fp
    lift          = prec / base_rate if base_rate > 0 else 0.0
    return (
        f"  {label:<{width}}  "
        f"n={total:>4}  "
        f"prec={prec:.3f}  "
        f"rec={rec:.3f}  "
        f"f1={f1:.3f}  "
        f"lift={lift:.2f}x"
    )


# ── main ─────────────────────────────────────────────────────────────────────

def run(min_samples: int = 5) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # All resolved rows
    cur.execute(
        """SELECT symbol, regime, threshold_passed, trade_opened, outcome_3d
           FROM signals_log
           WHERE outcome_3d IS NOT NULL"""
    )
    rows = cur.fetchall()

    # Pending rows (for context)
    cur.execute("SELECT COUNT(*) FROM signals_log WHERE outcome_3d IS NULL")
    pending = cur.fetchone()[0]

    conn.close()

    if not rows:
        print("No resolved signals yet (outcome_3d IS NOT NULL = 0 rows).")
        print(f"Pending (outcome_3d IS NULL): {pending}")
        return

    total_resolved = len(rows)
    total_pos      = sum(1 for r in rows if r["outcome_3d"] == 1)
    base_rate      = total_pos / total_resolved if total_resolved > 0 else 0.0

    print("=" * 70)
    print("SIGNAL PRECISION / RECALL REPORT")
    print("=" * 70)
    print(f"Resolved signals   : {total_resolved}")
    print(f"Pending signals    : {pending}")
    print(f"Positive rate (base rate): {base_rate:.3f}  ({total_pos}/{total_resolved})")
    print(f"Min samples shown  : {min_samples}")
    print()

    # ── 1. Overall ────────────────────────────────────────────────────────────
    print("── OVERALL ─────────────────────────────────────────────────────────")
    tp = sum(1 for r in rows if r["threshold_passed"] == 1 and r["outcome_3d"] == 1)
    fp = sum(1 for r in rows if r["threshold_passed"] == 1 and r["outcome_3d"] == 0)
    fn = sum(1 for r in rows if r["threshold_passed"] == 0 and r["outcome_3d"] == 1)

    print(_row("threshold>=0.65", tp, fp, fn, base_rate))

    all_as_tp = total_pos
    all_as_fp = total_resolved - total_pos
    print(_row("random baseline", all_as_tp, all_as_fp, 0, base_rate))

    # Traded-only precision (trade_opened=1)
    tp_t = sum(1 for r in rows if r["trade_opened"] == 1 and r["outcome_3d"] == 1)
    fp_t = sum(1 for r in rows if r["trade_opened"] == 1 and r["outcome_3d"] == 0)
    fn_t = sum(1 for r in rows if r["trade_opened"] == 0 and r["outcome_3d"] == 1)
    if (tp_t + fp_t) >= min_samples:
        print(_row("traded only", tp_t, fp_t, fn_t, base_rate))

    print()

    # ── 2. By Symbol ─────────────────────────────────────────────────────────
    print("── BY SYMBOL ───────────────────────────────────────────────────────")
    symbols = sorted({r["symbol"] for r in rows})
    for sym in symbols:
        sym_rows = [r for r in rows if r["symbol"] == sym]
        if len(sym_rows) < min_samples:
            continue
        tp = sum(1 for r in sym_rows if r["threshold_passed"] == 1 and r["outcome_3d"] == 1)
        fp = sum(1 for r in sym_rows if r["threshold_passed"] == 1 and r["outcome_3d"] == 0)
        fn = sum(1 for r in sym_rows if r["threshold_passed"] == 0 and r["outcome_3d"] == 1)
        sym_pos = sum(1 for r in sym_rows if r["outcome_3d"] == 1)
        sym_br  = sym_pos / len(sym_rows)
        print(_row(sym, tp, fp, fn, sym_br))

    print()

    # ── 3. By Regime ─────────────────────────────────────────────────────────
    print("── BY REGIME ───────────────────────────────────────────────────────")
    regimes = sorted({(r["regime"] or "UNKNOWN") for r in rows})
    for reg in regimes:
        reg_rows = [r for r in rows if (r["regime"] or "UNKNOWN") == reg]
        if len(reg_rows) < min_samples:
            continue
        tp = sum(1 for r in reg_rows if r["threshold_passed"] == 1 and r["outcome_3d"] == 1)
        fp = sum(1 for r in reg_rows if r["threshold_passed"] == 1 and r["outcome_3d"] == 0)
        fn = sum(1 for r in reg_rows if r["threshold_passed"] == 0 and r["outcome_3d"] == 1)
        reg_pos = sum(1 for r in reg_rows if r["outcome_3d"] == 1)
        reg_br  = reg_pos / len(reg_rows)
        print(_row(reg, tp, fp, fn, reg_br))

    print()

    # ── 4. Calibration check — prob_buy buckets ───────────────────────────────
    print("── PROB_BUY CALIBRATION (threshold_passed=1 rows only) ─────────────")

    import sqlite3 as _s3
    c2 = _s3.connect(DB_PATH)
    c2.row_factory = _s3.Row
    cur2 = c2.cursor()
    cur2.execute(
        """SELECT prob_buy, outcome_3d FROM signals_log
           WHERE outcome_3d IS NOT NULL AND threshold_passed=1
           ORDER BY prob_buy"""
    )
    cal_rows = cur2.fetchall()
    c2.close()

    buckets = {}
    for r in cal_rows:
        pb  = r["prob_buy"] or 0.0
        key = round(int(pb * 10) / 10, 1)   # 0.6, 0.7, 0.8, 0.9 …
        buckets.setdefault(key, []).append(r["outcome_3d"])

    if buckets:
        print(f"  {'prob bucket':<14}  {'n':>5}  {'hit%':>7}  {'expected%':>10}")
        for k in sorted(buckets):
            n    = len(buckets[k])
            hits = sum(buckets[k])
            if n >= 2:
                print(f"  {k:.1f}-{k+0.1:.1f}         "
                      f"  {n:>5}  {hits/n*100:>6.1f}%  {k*100:>9.1f}%")
    else:
        print("  (no threshold_passed=1 rows with resolved outcomes yet)")

    print()
    print("=" * 70)
    print("Interpretation:")
    print("  lift > 1.0 → model beats random; lift < 1.0 → model worse than random")
    print(f"  Base rate (random): {base_rate:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    min_s = 5
    for arg in sys.argv[1:]:
        if arg.startswith("--min-samples"):
            try:
                min_s = int(arg.split("=")[-1]) if "=" in arg else int(sys.argv[sys.argv.index(arg) + 1])
            except (ValueError, IndexError):
                pass
    run(min_samples=min_s)
