"""
daily_journal.py -- BIST Gun Sonu Ozet
---------------------------------------
Task Scheduler tarafindan 18:30'da calistirilir.

Gonderilen Telegram mesaji:
  🌙 GUN SONU ozeti — bugun kapanan tradeler, acik pozisyonlar, P&L
"""

import sys, csv, math, sqlite3, struct
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    from scipy.stats import proportion_confint
    from scipy.stats import beta as _beta_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "trade_data.db"
RESULTS  = BASE_DIR / "results"
CAPITAL  = 100_000

# Performance tracker
TRADE_LOG_PATH = RESULTS / "trade_log.csv"
PERF_TRACKER   = RESULTS / "performance_tracker.csv"
PERF_COLS = [
    "date", "balance", "daily_pnl_tl", "daily_pnl_pct", "cumulative_pnl_tl",
    "total_closed_trades", "wins", "losses", "win_rate",
    "sharpe", "max_drawdown_pct", "avg_hold_min",
    "best_trade_tl", "best_trade_symbol",
    "worst_trade_tl", "worst_trade_symbol",
    "open_positions", "cash_tl", "cash_pct",
    "criteria_winrate_ok", "criteria_sharpe_ok", "criteria_trades_ok",
]

# 61-day paper trade evaluation window (extended: 2026-04-08 → 2026-06-08)
EVAL_START = date(2026, 4, 8)   # portfolio reset date
EVAL_DAYS  = 61

from telegram_bot import send_telegram_alert

try:
    from oms import oms as _oms
    HAS_OMS = True
except Exception:
    HAS_OMS = False

try:
    from reconciliation import recon as _recon
    HAS_RECON = True
except Exception:
    HAS_RECON = False


# =====================================================================
# DB QUERIES
# =====================================================================

def _db():
    return sqlite3.connect(DB_PATH)


def _load_equity() -> dict:
    conn = _db()
    row = conn.execute(
        "SELECT total_equity, daily_pnl, cash, open_positions "
        "FROM paper_equity ORDER BY date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return {
            "equity":     float(row[0]),
            "daily_pnl":  float(row[1]),
            "cash":       float(row[2]),
            "n_open":     int(row[3]),
        }
    return {"equity": float(CAPITAL), "daily_pnl": 0.0, "cash": float(CAPITAL), "n_open": 0}


def _load_todays_trades(today: str) -> list[dict]:
    conn = _db()
    rows = conn.execute("""
        SELECT symbol, is_long, entry_price, exit_price,
               size_tl, pnl, pct_return, exit_reason
        FROM paper_trades
        WHERE exit_date = ?
        ORDER BY pnl DESC
    """, (today,)).fetchall()
    conn.close()
    cols = ["symbol","is_long","entry_price","exit_price",
            "size_tl","pnl","pct_return","exit_reason"]
    return [dict(zip(cols, r)) for r in rows]


def _load_open_positions() -> list[dict]:
    conn = _db()
    rows = conn.execute("""
        SELECT symbol, is_long, entry_price, stop_price, target_price,
               size_tl, COALESCE(confidence, 0)
        FROM paper_positions WHERE status='open'
        ORDER BY size_tl DESC
    """).fetchall()
    cols = ["symbol","is_long","entry_price","stop_price","target_price","size_tl","confidence"]
    positions = []
    for r in rows:
        pos = dict(zip(cols, r))
        c = pos["confidence"]
        if isinstance(c, (bytes, bytearray)):
            c = struct.unpack("<f", c)[0]
        pos["confidence"] = float(c)
        # Get latest close for unrealized P&L
        row = conn.execute(
            "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (pos["symbol"],)
        ).fetchone()
        pos["current_price"] = float(row[0]) if row else pos["entry_price"]
        positions.append(pos)
    conn.close()
    return positions


def _load_all_time_stats() -> dict:
    conn = _db()
    row = conn.execute("""
        SELECT COUNT(*) as n,
               SUM(pnl) as total_pnl,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               AVG(pct_return) as avg_pct
        FROM paper_trades
    """).fetchone()
    conn.close()
    if row and row[0]:
        n    = int(row[0])
        wins = int(row[2] or 0)
        return {
            "n":         n,
            "wins":      wins,
            "total_pnl": float(row[1] or 0),
            "win_rate":  wins / n * 100 if n else 0,
            "avg_pct":   float(row[3] or 0),
        }
    return {"n": 0, "wins": 0, "total_pnl": 0.0, "win_rate": 0.0, "avg_pct": 0.0}


def _wilson_ci(wins: int, total: int) -> dict:
    """
    Compute Wilson 95% CI and Jeffreys posterior mode for a win rate.
    Falls back to normal approximation if scipy is not installed.
    Returns dict with keys: lo, hi, mode, summary.
    """
    if total == 0:
        return {"lo": 0.0, "hi": 1.0, "mode": 0.0,
                "summary": "Wilson CI: N/A (no trades yet)"}

    p_hat = wins / total

    if HAS_SCIPY:
        lo, hi = proportion_confint(wins, total, alpha=0.05, method="wilson")
        # Jeffreys posterior mode: Beta(wins+0.5, losses+0.5)
        a = wins + 0.5
        b = (total - wins) + 0.5
        mode = (a - 1) / (a + b - 2) if (a > 1 and b > 1) else p_hat
    else:
        # Normal approximation fallback
        import math as _math
        z   = 1.96
        n   = total
        lo  = max(0.0, p_hat - z * _math.sqrt(p_hat * (1 - p_hat) / n))
        hi  = min(1.0, p_hat + z * _math.sqrt(p_hat * (1 - p_hat) / n))
        mode = p_hat   # point estimate only when scipy missing

    need = max(0, 30 - total)
    summary = (
        f"Win rate: {wins}/{total} = {p_hat:.1%} | "
        f"95% CI: [{lo:.1%}, {hi:.1%}] | Jeffreys mode: {mode:.1%}"
    )
    note = f"Statistical note: need {need} more trades for baseline validity"
    return {"lo": lo, "hi": hi, "mode": mode, "summary": summary, "note": note,
            "p_hat": p_hat, "wins": wins, "total": total}


# =====================================================================
# PERFORMANCE TRACKER
# =====================================================================

def _read_trade_log() -> list[dict]:
    if not TRADE_LOG_PATH.exists():
        return []
    with open(TRADE_LOG_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(d: dict, key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key, "")
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _migrate_perf_tracker():
    """Rewrite performance_tracker.csv if its columns differ from PERF_COLS."""
    if not PERF_TRACKER.exists():
        with open(PERF_TRACKER, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PERF_COLS).writeheader()
        return
    with open(PERF_TRACKER, "r", encoding="utf-8") as f:
        existing_cols = next(csv.reader(f), [])
    added   = [c for c in PERF_COLS if c not in existing_cols]
    removed = [c for c in existing_cols if c not in PERF_COLS]
    if not added and not removed:
        return
    with open(PERF_TRACKER, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(PERF_TRACKER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PERF_COLS)
        w.writeheader()
        for row in rows:
            for col in added:
                row.setdefault(col, "")
            w.writerow({c: row.get(c, "") for c in PERF_COLS})
    parts = []
    if added:
        parts.append(f"added {added}")
    if removed:
        parts.append(f"removed {removed}")
    print(f"  [CSV] Migrated {PERF_TRACKER.name}: {', '.join(parts)}")


def _count_open_positions() -> int:
    try:
        conn = _db()
        row = conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE status='open'"
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def update_performance_tracker(today: str) -> dict:
    """
    Compute daily performance metrics, append to performance_tracker.csv,
    print 30-day paper trade criteria, and send Telegram summary.
    Returns the metrics dict.
    """
    RESULTS.mkdir(exist_ok=True)
    _migrate_perf_tracker()
    all_trades = _read_trade_log()

    # ── Trade-level aggregation ────────────────────────────────────────
    def _pnl(t: dict) -> float:
        v = _safe_float(t, "net_pnl_tl") or _safe_float(t, "kar_zarar_tl")
        return v

    pnl_list  = [_pnl(t) for t in all_trades]
    hold_list = [_safe_float(t, "tutulma_dk") for t in all_trades]
    n_closed  = len(all_trades)

    wins     = sum(1 for p in pnl_list if p > 0)
    losses   = sum(1 for p in pnl_list if p <= 0)
    win_rate = round(wins / n_closed, 4) if n_closed > 0 else 0.0
    avg_hold = round(sum(hold_list) / len(hold_list), 1) if hold_list else 0.0
    cum_pnl  = round(sum(pnl_list), 2)

    best_tl  = round(max(pnl_list), 2) if pnl_list else 0.0
    worst_tl = round(min(pnl_list), 2) if pnl_list else 0.0

    # best/worst trade symbol
    best_sym  = ""
    worst_sym = ""
    if all_trades:
        idx_best  = pnl_list.index(max(pnl_list))
        idx_worst = pnl_list.index(min(pnl_list))
        best_sym  = all_trades[idx_best].get("sembol", "")
        worst_sym = all_trades[idx_worst].get("sembol", "")

    # ── Daily P&L (aggregate by date) ─────────────────────────────────
    daily_pnl_map: dict[str, float] = {}
    for t in all_trades:
        d = t.get("tarih", "")
        if not d:
            continue
        daily_pnl_map[d] = daily_pnl_map.get(d, 0.0) + _pnl(t)
    today_pnl   = round(daily_pnl_map.get(today, 0.0), 2)
    daily_dates = sorted(daily_pnl_map.keys())
    n_days      = len(daily_dates)

    # ── Daily P&L % ───────────────────────────────────────────────────
    # Use CAPITAL as base (starting portfolio value)
    today_pnl_pct = round(today_pnl / CAPITAL * 100, 4) if CAPITAL > 0 else 0.0

    # ── Sharpe (annualised) — need >= 5 trading days ───────────────────
    sharpe_val = None
    if n_days >= 5:
        balance_est = CAPITAL
        rets = []
        for d in daily_dates:
            r = daily_pnl_map[d] / balance_est if balance_est > 0 else 0.0
            rets.append(r)
            balance_est += daily_pnl_map[d]
        mean_r = sum(rets) / len(rets)
        var_r  = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        std_r  = math.sqrt(var_r) if var_r > 0 else 0.0
        if std_r > 0:
            sharpe_val = round((mean_r / std_r) * math.sqrt(252), 2)

    # ── Max drawdown ──────────────────────────────────────────────────
    max_dd = 0.0
    if daily_dates:
        equity = CAPITAL
        peak   = CAPITAL
        for d in daily_dates:
            equity += daily_pnl_map[d]
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (equity - peak) / peak
                if dd < max_dd:
                    max_dd = dd
    max_dd_pct = round(max_dd * 100, 2)  # negative percentage

    # ── Balance, cash, open positions ────────────────────────────────
    eq           = _load_equity()
    balance      = round(eq["equity"], 2)
    cash_tl      = round(eq["cash"], 2)
    cash_pct     = round(cash_tl / balance * 100, 1) if balance > 0 else 100.0
    open_pos_cnt = _count_open_positions()

    # ── 30-day criteria check ─────────────────────────────────────────
    today_dt       = date.fromisoformat(today)
    days_elapsed   = (today_dt - EVAL_START).days
    days_remaining = max(0, EVAL_DAYS - days_elapsed)

    wr_ok = win_rate >= 0.58
    sh_ok = sharpe_val is not None and sharpe_val >= 1.2
    tr_ok = n_closed >= 30

    # ── Append to tracker CSV ─────────────────────────────────────────
    row = {
        "date":                today,
        "balance":             balance,
        "daily_pnl_tl":        today_pnl,
        "daily_pnl_pct":       today_pnl_pct,
        "cumulative_pnl_tl":   cum_pnl,
        "total_closed_trades": n_closed,
        "wins":                wins,
        "losses":              losses,
        "win_rate":            win_rate,
        "sharpe":              sharpe_val if sharpe_val is not None else "",
        "max_drawdown_pct":    max_dd_pct,
        "avg_hold_min":        avg_hold,
        "best_trade_tl":       best_tl,
        "best_trade_symbol":   best_sym,
        "worst_trade_tl":      worst_tl,
        "worst_trade_symbol":  worst_sym,
        "open_positions":      open_pos_cnt,
        "cash_tl":             cash_tl,
        "cash_pct":            cash_pct,
        "criteria_winrate_ok": int(wr_ok),
        "criteria_sharpe_ok":  int(sh_ok),
        "criteria_trades_ok":  int(tr_ok),
    }
    with open(PERF_TRACKER, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=PERF_COLS).writerow(row)

    # ── Print ─────────────────────────────────────────────────────────
    sharpe_str = f"{sharpe_val:.2f}" if sharpe_val is not None else "N/A (<5 gun)"
    print("=" * 44)
    print("=== 30-GUN PAPER TRADE KRITERLERI ===")
    print(f"Win Rate : {win_rate:.1%}  hedef>=58%  {'OK' if wr_ok else 'EKSIK'}")
    print(f"Sharpe   : {sharpe_str:<12}  hedef>=1.2  {'OK' if sh_ok else 'EKSIK'}")
    print(f"Islem    : {n_closed:<12}  hedef>=30   {'OK' if tr_ok else 'EKSIK'}")
    print(f"Max DD   : {max_dd_pct:.2f}%")
    print(f"Win/Loss : {wins}W / {losses}L")
    print(f"Cum PnL  : {cum_pnl:+,.0f} TL  ({today_pnl:+,.0f} TL bugun)")
    print(f"Balance  : {balance:,.0f} TL  |  Nakit: {cash_tl:,.0f} TL ({cash_pct:.0f}%)")
    print(f"Acik Poz : {open_pos_cnt}")
    if best_sym:
        print(f"Best     : {best_sym} {best_tl:+,.0f} TL  |  Worst: {worst_sym} {worst_tl:+,.0f} TL")
    print(f"Kalan gun: {days_remaining}")
    print("=" * 44)

    # ── Wilson CI stats ───────────────────────────────────────────────
    ci = _wilson_ci(wins, n_closed)
    print()
    print(ci["summary"])
    print(ci["note"])

    # ── Telegram ─────────────────────────────────────────────────────
    tg = (
        f"<b>30-Gun Degerlendirme ({today})</b>\n"
        f"\n"
        f"Win Rate : {win_rate:.1%}  {'OK' if wr_ok else 'EKSIK'}  (hedef %58)\n"
        f"Sharpe   : {sharpe_str}  {'OK' if sh_ok else 'EKSIK'}  (hedef 1.2)\n"
        f"Islemler : {n_closed}  {'OK' if tr_ok else 'EKSIK'}  (hedef 30)\n"
        f"\n"
        f"Max DD   : {max_dd_pct:.2f}%\n"
        f"Kalan gun: {days_remaining}\n"
        f"Win/Loss : {wins}W / {losses}L\n"
        f"Cum PnL  : {cum_pnl:+,.0f} TL  ({today_pnl:+,.0f} TL bugun)\n"
        f"Balance  : {balance:,.0f} TL  |  Nakit: {cash_pct:.0f}%\n"
        f"\n"
        f"{ci['summary']}\n"
        f"{ci['note']}"
    )
    ok = send_telegram_alert(tg)
    print(f"[Perf] Tracker guncellendi -> {PERF_TRACKER.name}  (TG: {'OK' if ok else 'HATA'})")
    return row


# =====================================================================
# BUILD MESSAGE
# =====================================================================

def build_journal_message(today: str) -> str:
    eq     = _load_equity()
    trades = _load_todays_trades(today)
    open_p = _load_open_positions()
    stats  = _load_all_time_stats()

    equity    = eq["equity"]
    daily_pnl = eq["daily_pnl"]
    cash      = eq["cash"]
    tot_pnl   = equity - CAPITAL

    d_sign = "+" if daily_pnl >= 0 else ""
    t_sign = "+" if tot_pnl   >= 0 else ""
    d_emoji = "📈" if daily_pnl >= 0 else "📉"

    lines = [
        f"🌙 <b>GUN SONU OZETI</b>",
        f"📅 {today}  |  {datetime.now().strftime('%H:%M')}",
        "",
        f"{d_emoji} <b>Portfoy</b>",
        f"  Equity    : {equity:>10,.0f} TL",
        f"  Nakit     : {cash:>10,.0f} TL",
        f"  Gun P&amp;L   : {d_sign}{daily_pnl:>+10,.0f} TL",
        f"  Toplam P&amp;L: {t_sign}{tot_pnl:>+10,.0f} TL ({t_sign}{tot_pnl/CAPITAL*100:.2f}%)",
        "",
    ]

    # Today's closed trades
    if trades:
        lines.append(f"✅ <b>Bugun Kapanan Tradeler ({len(trades)}):</b>")
        day_pnl_sum = sum(t["pnl"] for t in trades)
        for t in trades:
            d   = "L" if t["is_long"] else "S"
            pnl = t["pnl"]
            em  = "+" if pnl >= 0 else ""
            lines.append(
                f"  {t['symbol']:<6} {d}  {t['entry_price']:.2f}->{t['exit_price']:.2f}"
                f"  {em}{pnl:,.0f} TL ({t['pct_return']:+.1f}%)  {t['exit_reason']}"
            )
        sign = "+" if day_pnl_sum >= 0 else ""
        lines.append(f"  Bugun realize: {sign}{day_pnl_sum:,.0f} TL")
        lines.append("")
    else:
        lines += ["  (bugun kapanan trade yok)", ""]

    # Open positions with unrealized P&L
    if open_p:
        lines.append(f"📋 <b>Acik Pozisyonlar ({len(open_p)}):</b>")
        total_unreal = 0.0
        for pos in open_p:
            cur   = pos["current_price"]
            entry = pos["entry_price"]
            stop  = pos["stop_price"]
            d_str = "LONG" if pos["is_long"] else "SHORT"
            if pos["is_long"]:
                pnl_pct = (cur - entry) / entry * 100
                prox    = (entry - cur) / max(entry - stop, 1e-9) * 100
            else:
                pnl_pct = (entry - cur) / entry * 100
                prox    = (cur - entry) / max(stop - entry, 1e-9) * 100
            unreal_tl = pos["size_tl"] * pnl_pct / 100
            total_unreal += unreal_tl
            warn = " ⚠️" if prox > 50 else ""
            lines.append(
                f"  {pos['symbol']:<6} {d_str:<5}  {pnl_pct:+.1f}%"
                f"  ({unreal_tl:+,.0f} TL)  Stop:%{max(0,prox):.0f}{warn}"
            )
        sign = "+" if total_unreal >= 0 else ""
        lines.append(f"  Toplam unrealized: {sign}{total_unreal:,.0f} TL")
        lines.append("")
    else:
        lines += ["  (acik pozisyon yok)", ""]

    # All-time stats
    if stats["n"] > 0:
        ci = _wilson_ci(stats["wins"], stats["n"])
        lines += [
            f"📊 <b>Tum Zamanlar ({stats['n']} trade):</b>",
            f"  Realize P&amp;L  : {stats['total_pnl']:+,.0f} TL",
            f"  Kazanma orani: {stats['win_rate']:.0f}%",
            f"  Ort. getiri  : {stats['avg_pct']:+.1f}% / trade",
            f"",
            f"📈 <b>Istatistiksel Analiz:</b>",
            f"  {ci['summary']}",
            f"  {ci['note']}",
        ]
    else:
        lines.append("  (henuz kapanan trade yok)")

    # OMS order management summary
    if HAS_OMS:
        try:
            df_orders  = _oms.get_todays_orders()
            n_orders   = len(df_orders)
            n_filled   = int((df_orders["status"] == "FILLED").sum())   if n_orders else 0
            n_rejected = int((df_orders["status"] == "REJECTED").sum()) if n_orders else 0
            fill_rate  = _oms.get_fill_rate_today()
            fr_str     = f"{fill_rate:.0%}" if fill_rate is not None else "N/A"
            last_time  = (
                df_orders["created_at"].max()[:19]
                if n_orders else "N/A"
            )
            lines += [
                "",
                "📋 <b>Emir Yonetimi (OMS):</b>",
                f"  Toplam emir : {n_orders}",
                f"  Fill rate   : {fr_str}",
                f"  Rejected    : {n_rejected}",
                f"  Son emir    : {last_time}",
            ]
        except Exception:
            pass

    # Reconciliation summary (DB-only: no in-memory positions at journal time)
    if HAS_RECON:
        try:
            _r_ok, _r_issues = _recon.run()
            _r_high = [i for i in _r_issues if i["severity"] == "HIGH"]
            _r_med  = [i for i in _r_issues if i["severity"] == "MEDIUM"]
            if _r_ok:
                recon_status = "OK -- tutarsizlik yok"
            else:
                recon_status = f"{len(_r_issues)} sorun tespit edildi"
            lines += [
                "",
                "🔍 <b>Reconciliation:</b>",
                f"  Durum : {recon_status}",
            ]
            for i in (_r_high + _r_med):
                lines.append(f"  [{i['severity']}] {i['type']}: {i['msg']}")
        except Exception:
            pass

    return "\n".join(lines)


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":
    today = date.today().isoformat()
    print("=" * 55)
    print(f"BIST GUN SONU RAPORU  |  {today}")
    print("=" * 55)

    msg = build_journal_message(today)
    print(msg)
    print()
    print("Telegram'a gonderiliyor...", end=" ", flush=True)
    ok = send_telegram_alert(msg)
    print("OK" if ok else "HATA")

    print()
    update_performance_tracker(today)

    # ── Auto-fill signal outcomes (3-day horizon) ─────────────────────────
    print()
    print("outcome_filler calistiriliyor...")
    try:
        import subprocess as _sp
        import sys as _sys
        _result = _sp.run(
            [_sys.executable, "outcome_filler.py"],
            capture_output=True, text=True
        )
        print(_result.stdout.strip())
        if _result.returncode != 0 and _result.stderr:
            print(f"[WARN] outcome_filler stderr: {_result.stderr[:200]}")
    except Exception as _e:
        print(f"[WARN] outcome_filler calisilamadi: {_e}")
