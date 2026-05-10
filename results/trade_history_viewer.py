"""
results/trade_history_viewer.py
--------------------------------
Reads paper_trades + order_log from trade_data.db and prints a formatted
trade history table with summary stats. Also writes results/trade_history.csv.

Usage:
    python3.12 results/trade_history_viewer.py
"""

import csv
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent   # one level up from results/
DB_PATH  = BASE_DIR / "trade_data.db"
CSV_OUT  = Path(__file__).parent / "trade_history.csv"

# Turkish exit reason labels
EXIT_TR = {
    "stop_loss":     "Stop Kesildi",
    "trailing_stop": "Trailing Stop",
    "target_hit":    "Hedef Tuttu",
    "model_flip":    "Model Sinyal Dustu",
    "news_exit":     "Haber Cikisi",
    "drawdown_kill": "Drawdown Kill",
}


def _fmt_hold(seconds: float) -> str:
    """Convert seconds to 'Xs YYdk' string."""
    if seconds <= 0:
        return "-"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}s {m}dk"
    return f"{m}dk"


def _parse_entry_time(entry_note: str, entry_date: str) -> str:
    """Extract entry time HH:MM from entry_note JSON, fall back to date."""
    try:
        if entry_note and entry_note.strip().startswith("{"):
            meta = json.loads(entry_note)
            et = meta.get("entry_time", "")
            if et and len(et) >= 16:
                return et[11:16]   # HH:MM
    except Exception:
        pass
    return "--:--"


def _parse_entry_prob(entry_note: str, signal_prob: float) -> tuple[float, str]:
    """Return (prob, regime) from entry_note JSON, fall back to signal_prob."""
    try:
        if entry_note and entry_note.strip().startswith("{"):
            meta = json.loads(entry_note)
            prob   = float(meta.get("prob", signal_prob or 0.0))
            regime = meta.get("reason", "")
            # Pull regime token (last word after RSI value)
            # entry_note reason format: "ML BUY prob=X conf=Y% RSI=Z REGIME"
            parts = regime.split()
            reg = parts[-1] if parts else ""
            return prob, reg
    except Exception:
        pass
    return (signal_prob or 0.0), ""


def _build_neden_aldi(entry_note: str, signal_prob: float, regime: str) -> str:
    """Build 'Neden Aldi' column string."""
    try:
        if entry_note and entry_note.strip().startswith("{"):
            meta   = json.loads(entry_note)
            prob   = float(meta.get("prob", signal_prob or 0.0))
            rsi    = meta.get("rsi", "")
            reg    = (meta.get("reason", "") or "").split()
            reg_str = reg[-1] if reg else (regime or "")
            rsi_str = f" RSI:{rsi:.1f}" if rsi else ""
            return f"Model prob: {prob*100:.1f}%{rsi_str} | rejim: {reg_str}"
    except Exception:
        pass
    prob_val = signal_prob or 0.0
    return f"Model prob: {prob_val*100:.1f}% | rejim: {regime or '-'}"


def _get_exit_time(order_log: dict[str, str], symbol: str, exit_date: str) -> str:
    """Look up SELL order filled_at time for symbol; fall back to date only."""
    key = symbol
    if key in order_log:
        t = order_log[key]
        if len(t) >= 16:
            return t[11:16]   # HH:MM
    return "--:--"


def load_trades() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # -- order_log: latest SELL per symbol for exit time ----------------------
    sell_rows = conn.execute("""
        SELECT symbol, MAX(filled_at) AS filled_at
        FROM order_log
        WHERE direction='SELL' AND status='FILLED'
        GROUP BY symbol
    """).fetchall()
    sell_times: dict[str, str] = {r["symbol"]: r["filled_at"] for r in sell_rows}

    # -- paper_trades ----------------------------------------------------------
    rows = conn.execute("""
        SELECT id, symbol, strategy, entry_date, exit_date,
               entry_price, exit_price, size_tl, is_long,
               pnl, pct_return, exit_reason, signal_prob,
               regime, entry_note, exit_note
        FROM paper_trades
        ORDER BY entry_date ASC, id ASC
    """).fetchall()
    conn.close()

    trades = []
    for r in rows:
        entry_note  = r["entry_note"] or ""
        exit_note   = r["exit_note"]  or ""
        signal_prob = r["signal_prob"] or 0.0
        regime      = r["regime"] or ""

        # Entry time
        entry_time = _parse_entry_time(entry_note, r["entry_date"])

        # Exit time: try order_log first, then exit_note
        exit_time = _get_exit_time(sell_times, r["symbol"], r["exit_date"])
        if exit_time == "--:--":
            # Try to parse from exit_note e.g. "loop_trader model_flip @ 37.90"
            # or "TEST target_hit @ 143.23"  — no time embedded, keep as is
            pass

        # Holding duration
        entry_dt_str = f"{r['entry_date']} {entry_time}:00" if entry_time != "--:--" else f"{r['entry_date']} 10:00:00"
        exit_dt_str  = f"{r['exit_date']} {exit_time}:00"  if exit_time  != "--:--" else f"{r['exit_date']} 18:00:00"
        try:
            entry_dt = datetime.strptime(entry_dt_str, "%Y-%m-%d %H:%M:%S")
            exit_dt  = datetime.strptime(exit_dt_str,  "%Y-%m-%d %H:%M:%S")
            hold_sec = (exit_dt - entry_dt).total_seconds()
        except Exception:
            hold_sec = 0.0

        neden_aldi = _build_neden_aldi(entry_note, signal_prob, regime)
        exit_reason_raw = r["exit_reason"] or ""
        neden_satti = EXIT_TR.get(exit_reason_raw, exit_reason_raw)

        trades.append({
            "tarih":        r["entry_date"],
            "saat":         entry_time,
            "sembol":       r["symbol"],
            "neden_aldi":   neden_aldi,
            "giris_fiyat":  r["entry_price"],
            "cikis_fiyat":  r["exit_price"],
            "cikis_saati":  exit_time,
            "boyut_tl":     r["size_tl"],
            "kz_tl":        r["pnl"],
            "kz_pct":       r["pct_return"],
            "neden_satti":  neden_satti,
            "tutulma":      _fmt_hold(hold_sec),
            "hold_sec":     hold_sec,   # for avg calc
        })

    return trades


def _color(val: float, text: str) -> str:
    """ANSI-color text green if positive, red if negative."""
    if val > 0:
        return f"\033[92m{text}\033[0m"
    if val < 0:
        return f"\033[91m{text}\033[0m"
    return text


def print_table(trades: list[dict]):
    if not trades:
        print("  Kayitli kapali islem yok.")
        return

    # Column widths
    W = {
        "idx":        4,
        "tarih":      11,
        "saat":       6,
        "sembol":     6,
        "neden_aldi": 38,
        "giris":      9,
        "cikis":      9,
        "cikis_s":    7,
        "boyut":      10,
        "kz_tl":      11,
        "kz_pct":     8,
        "neden_s":    20,
        "tutulma":    9,
    }

    def row(vals):
        return (
            f"{vals[0]:<{W['idx']}}"
            f"{vals[1]:<{W['tarih']}}"
            f"{vals[2]:<{W['saat']}}"
            f"{vals[3]:<{W['sembol']}}"
            f"{vals[4]:<{W['neden_aldi']}}"
            f"{vals[5]:>{W['giris']}}"
            f"{vals[6]:>{W['cikis']}}"
            f"{vals[7]:>{W['cikis_s']}}"
            f"{vals[8]:>{W['boyut']}}"
            f"{vals[9]:>{W['kz_tl']}}"
            f"{vals[10]:>{W['kz_pct']}}"
            f"  {vals[11]:<{W['neden_s']}}"
            f"{vals[12]:>{W['tutulma']}}"
        )

    total_w = sum(W.values()) + 4
    sep = "─" * total_w

    print()
    print("=" * total_w)
    print("  KAPALI ISLEM GECMISI")
    print("=" * total_w)
    header = row(["#", "Tarih", "Saat", "Sembol", "Neden Aldi",
                  "Giris TL", "Cikis TL", "C.Saati",
                  "Boyut TL", "K/Z TL", "K/Z %",
                  "Neden Satti", "Tutulma"])
    print(header)
    print(sep)

    for i, t in enumerate(trades, 1):
        kz_tl_str  = f"{t['kz_tl']:+,.2f}"
        kz_pct_str = f"{t['kz_pct']:+.2f}%"

        raw_row = row([
            str(i),
            t["tarih"],
            t["saat"],
            t["sembol"],
            t["neden_aldi"][:37],
            f"{t['giris_fiyat']:.2f}",
            f"{t['cikis_fiyat']:.2f}",
            t["cikis_saati"],
            f"{t['boyut_tl']:,.0f}",
            kz_tl_str,
            kz_pct_str,
            t["neden_satti"][:19],
            t["tutulma"],
        ])

        # Colorize K/Z columns by replacing the plain strings
        colored = _color(t["kz_tl"], kz_tl_str)
        colored_pct = _color(t["kz_pct"], kz_pct_str)
        raw_row = raw_row.replace(kz_tl_str, colored, 1)
        raw_row = raw_row.replace(kz_pct_str, colored_pct, 1)
        print(raw_row)

    print(sep)


def print_summary(trades: list[dict]):
    if not trades:
        return

    total   = len(trades)
    winners = [t for t in trades if t["kz_tl"] > 0]
    losers  = [t for t in trades if t["kz_tl"] < 0]
    total_pnl   = sum(t["kz_tl"] for t in trades)
    best        = max(trades, key=lambda t: t["kz_tl"])
    worst       = min(trades, key=lambda t: t["kz_tl"])
    win_rate    = len(winners) / total * 100 if total else 0
    avg_win     = sum(t["kz_tl"] for t in winners) / len(winners) if winners else 0
    avg_loss    = sum(t["kz_tl"] for t in losers)  / len(losers)  if losers  else 0
    valid_holds = [t["hold_sec"] for t in trades if t["hold_sec"] > 0]
    avg_hold    = sum(valid_holds) / len(valid_holds) if valid_holds else 0

    W = 60
    print()
    print("=" * W)
    print("  OZET")
    print("=" * W)
    print(f"  Toplam islem  : {total}")
    print(f"  Kazanan       : {len(winners)}   Kaybeden: {len(losers)}")
    print(f"  Win rate      : {_color(win_rate - 50, f'{win_rate:.1f}%')}")
    print(f"  Toplam P&L    : {_color(total_pnl, f'{total_pnl:+,.2f} TL')}")
    print(f"  Ort. kazanc   : {_color(avg_win,  f'{avg_win:+,.2f} TL')}")
    print(f"  Ort. kayip    : {_color(avg_loss, f'{avg_loss:+,.2f} TL')}")
    best_pnl_str  = f"{best['kz_tl']:+,.2f} TL"
    worst_pnl_str = f"{worst['kz_tl']:+,.2f} TL"
    print(f"  En iyi islem  : {best['sembol']} {_color(best['kz_tl'],   best_pnl_str)}"
          f"  ({best['tarih']})")
    print(f"  En kotu islem : {worst['sembol']} {_color(worst['kz_tl'], worst_pnl_str)}"
          f"  ({worst['tarih']})")
    print(f"  Ort. tutulma  : {_fmt_hold(avg_hold)}")
    print("=" * W)
    print()


def write_csv(trades: list[dict]):
    fieldnames = [
        "Tarih", "Saat", "Sembol", "Neden Aldi",
        "Giris Fiyati", "Cikis Fiyati", "Cikis Saati",
        "Boyut TL", "Kar/Zarar TL", "Kar/Zarar %",
        "Neden Satti", "Tutulma",
    ]
    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in trades:
            w.writerow({
                "Tarih":          t["tarih"],
                "Saat":           t["saat"],
                "Sembol":         t["sembol"],
                "Neden Aldi":     t["neden_aldi"],
                "Giris Fiyati":   round(t["giris_fiyat"], 4),
                "Cikis Fiyati":   round(t["cikis_fiyat"], 4),
                "Cikis Saati":    t["cikis_saati"],
                "Boyut TL":       round(t["boyut_tl"], 2),
                "Kar/Zarar TL":   round(t["kz_tl"], 2),
                "Kar/Zarar %":    round(t["kz_pct"], 4),
                "Neden Satti":    t["neden_satti"],
                "Tutulma":        t["tutulma"],
            })
    print(f"  CSV yazildi: {CSV_OUT}")


def main():
    if not DB_PATH.exists():
        print(f"DB bulunamadi: {DB_PATH}")
        return

    trades = load_trades()
    print_table(trades)
    print_summary(trades)
    write_csv(trades)


if __name__ == "__main__":
    main()
