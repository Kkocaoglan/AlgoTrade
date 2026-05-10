"""
cost_aware_backtest.py -- BIST Gercekci Maliyet Analizi
---------------------------------------------------------
Ayni 3 strateji uzerinde (backtest.py'den):
  - BRUT  : sifir maliyet (saf sinyal kalitesi)
  - NET   : gercekci BIST maliyetleri (slippage + likidite + komisyon)

Cevap: "Gercekte ne kadar kazanirdim?"

Kullanim:
  python3.12 cost_aware_backtest.py

Ciktilar:
  results/cost_aware_backtest_YYYY-MM-DD.txt   -- tam rapor
  results/cost_aware_backtest_detail.csv        -- trade-level detay
"""

import sys
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CAPITAL = 100_000

# =============================================================================
# BIST GERCEKCI MALIYET PARAMETRELERİ
# =============================================================================

BIST_COSTS = {
    "slippage_entry"  : 0.0010,   # %0.10 giris slippage (gunluk baris)
    "slippage_exit"   : 0.0010,   # %0.10 cikis slippage
    "commission_rate" : 0.0005,   # %0.05 her yon (Deniz Yatirim binde 5)
    "min_commission"  : 2.0,      # min 2 TL per emir
}

# BIST-30 (daha likit, daha dusuk ek slippage)
_BIST30 = {
    "AKBNK", "GARAN", "ISCTR", "YKBNK", "TUPRS",
    "TCELL", "FROTO", "BIMAS", "ENKAI", "PETKM",
    "ASELS", "MGROS", "TAVHL", "EKGYO",
}


def get_liquidity_slippage(symbol: str, size_tl: float) -> float:
    """
    Pozisyon buyuklugune ve hisse likiditesine gore ek slippage.
    BIST-30 hisseleri icin daha dusuk; diger hisseler icin daha yuksek.
    """
    if symbol in _BIST30:
        if size_tl <= 10_000:
            return 0.0005   # %0.05
        if size_tl <= 50_000:
            return 0.0010   # %0.10
        return 0.0020       # %0.20
    else:
        if size_tl <= 10_000:
            return 0.0010   # %0.10
        if size_tl <= 50_000:
            return 0.0020   # %0.20
        return 0.0040       # %0.40


def apply_costs(
    symbol: str,
    entry_price: float,
    exit_price: float,
    size_tl: float,
    is_long: bool = True,
) -> tuple[float, float, float]:
    """
    Gercekci BIST maliyetlerini uygula.

    Returns:
        gross_return : maliyet oncesi getiri (oran, yuzde degil)
        net_return   : maliyet sonrasi getiri (oran)
        total_cost_pct: gross - net (round-trip maliyet orani)
    """
    if entry_price <= 0 or size_tl <= 0:
        return 0.0, 0.0, 0.0

    shares    = size_tl / entry_price
    liq_slip  = get_liquidity_slippage(symbol, size_tl)

    slip_in   = BIST_COSTS["slippage_entry"] + liq_slip
    slip_out  = BIST_COSTS["slippage_exit"]  + liq_slip

    # Ters yonde slippage (her zaman aleyhte)
    if is_long:
        actual_entry = entry_price * (1 + slip_in)   # daha pahali al
        actual_exit  = exit_price  * (1 - slip_out)  # daha ucuz sat
    else:
        actual_entry = entry_price * (1 - slip_in)   # daha ucuz kor-sat
        actual_exit  = exit_price  * (1 + slip_out)  # daha pahali geri al

    # Komisyon (her yon icin, min 2 TL)
    comm_entry = max(
        actual_entry * shares * BIST_COSTS["commission_rate"],
        BIST_COSTS["min_commission"],
    )
    comm_exit = max(
        actual_exit  * shares * BIST_COSTS["commission_rate"],
        BIST_COSTS["min_commission"],
    )
    total_comm = comm_entry + comm_exit

    # PnL hesaplama
    if is_long:
        gross_pnl = (exit_price  - entry_price) * shares
        net_pnl   = (actual_exit - actual_entry) * shares - total_comm
    else:
        gross_pnl = (entry_price  - exit_price)  * shares
        net_pnl   = (actual_entry - actual_exit) * shares - total_comm

    gross_return   = gross_pnl / size_tl
    net_return     = net_pnl   / size_tl
    total_cost_pct = gross_return - net_return

    return gross_return, net_return, total_cost_pct


# =============================================================================
# STRATEJI CALISTIRICI  (backtest.py'i sifir maliyetle kullan)
# =============================================================================

def _run_strategies_zero_cost() -> tuple[list, list]:
    """
    backtest.py'deki 3 stratejiyi SIFIR maliyetle calistir.
    Sonuc: ham sinyal fiyatlarinda trade listesi (gross baseline icin).
    """
    sys.path.insert(0, str(BASE_DIR))
    import backtest as bt

    # Orijinal degerleri sakla
    _orig = {
        "COMMISSION_ENTRY":    bt.COMMISSION_ENTRY,
        "COMMISSION_EXIT":     bt.COMMISSION_EXIT,
        "SLIPPAGE_ENTRY":      bt.SLIPPAGE_ENTRY,
        "SLIPPAGE_EXIT":       bt.SLIPPAGE_EXIT,
        "ENTRY_ATR_SLIP_MULT": bt.ENTRY_ATR_SLIP_MULT,
        "EXIT_ATR_SLIP_MULT":  bt.EXIT_ATR_SLIP_MULT,
    }

    # Sifir maliyet yap
    bt.COMMISSION_ENTRY    = 0.0
    bt.COMMISSION_EXIT     = 0.0
    bt.SLIPPAGE_ENTRY      = 0.0
    bt.SLIPPAGE_EXIT       = 0.0
    bt.ENTRY_ATR_SLIP_MULT = 0.0
    bt.EXIT_ATR_SLIP_MULT  = 0.0

    try:
        print("  Veri yukleniyor...", end=" ", flush=True)
        frames = bt.load_data()
        print(f"OK ({len(frames)} hisse)")

        cap1 = int(CAPITAL * bt.ALLOC[1])
        cap2 = int(CAPITAL * bt.ALLOC[2])
        cap3 = int(CAPITAL * bt.ALLOC[3])

        print(f"  S1 MTF Momentum  (cap={cap1:,} TL)...", end=" ", flush=True)
        t1, _, dates = bt.run_s1(frames, cap1)
        print(f"{len(t1)} trade")

        print(f"  S2 RSI MeanRev.  (cap={cap2:,} TL)...", end=" ", flush=True)
        t2, _, _ = bt.run_s2(frames, cap2)
        print(f"{len(t2)} trade")

        print(f"  S3 BB Breakout   (cap={cap3:,} TL)...", end=" ", flush=True)
        t3, _, _ = bt.run_s3(frames, cap3)
        print(f"{len(t3)} trade")

    finally:
        # Orijinal degerlere geri don
        for k, v in _orig.items():
            setattr(bt, k, v)

    all_trades = t1 + t2 + t3
    return all_trades, dates


# =============================================================================
# MALIYET UYGULA + METRIK HESAPLA
# =============================================================================

def _enrich_with_costs(trades: list) -> pd.DataFrame:
    """Her trade icin gross ve net PnL hesapla."""
    rows = []
    for t in trades:
        sym      = t["symbol"]
        ep       = float(t["entry_p"])
        xp       = float(t["exit_p"])
        size     = float(t["size"])
        is_long  = bool(t.get("is_long", True))

        gross_ret, net_ret, cost_pct = apply_costs(sym, ep, xp, size, is_long)

        rows.append({
            "strategy"   : t.get("strategy", 0),
            "symbol"     : sym,
            "entry_date" : str(t.get("entry_date", ""))[:10],
            "exit_date"  : str(t.get("exit_date", ""))[:10],
            "entry_p"    : round(ep, 4),
            "exit_p"     : round(xp, 4),
            "size_tl"    : round(size, 2),
            "is_long"    : int(is_long),
            "exit_reason": t.get("exit_reason", ""),
            "gross_pnl"  : round(gross_ret * size, 2),
            "net_pnl"    : round(net_ret   * size, 2),
            "cost_tl"    : round(cost_pct  * size, 2),
            "gross_ret"  : round(gross_ret, 6),
            "net_ret"    : round(net_ret,   6),
            "cost_pct"   : round(cost_pct,  6),
            "gross_win"  : int(gross_ret > 0),
            "net_win"    : int(net_ret   > 0),
        })

    return pd.DataFrame(rows)


def _compute_sharpe(daily_rets: pd.Series) -> float:
    """Annualised Sharpe from a series of daily returns."""
    if len(daily_rets) < 5 or daily_rets.std() == 0:
        return 0.0
    return float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))


def _metrics(df: pd.DataFrame, col_ret: str, col_win: str) -> dict:
    """Trade-level and time-aggregated metrics for one cost scenario."""
    n          = len(df)
    win_rate   = df[col_win].mean()
    avg_ret    = df[col_ret].mean()
    total_pnl  = (df[col_ret] * df["size_tl"]).sum()

    # Daily aggregated returns for Sharpe
    daily = (
        df.groupby("exit_date")
        .apply(lambda g: (g[col_ret] * g["size_tl"]).sum() / CAPITAL)
    )
    sharpe = _compute_sharpe(daily)

    # Profit factor
    wins_sum  = (df.loc[df[col_win] == 1, col_ret] * df.loc[df[col_win] == 1, "size_tl"]).sum()
    loss_sum  = abs((df.loc[df[col_win] == 0, col_ret] * df.loc[df[col_win] == 0, "size_tl"]).sum())
    pf        = wins_sum / loss_sum if loss_sum > 0 else float("inf")

    return {
        "n"        : n,
        "win_rate" : win_rate,
        "avg_ret"  : avg_ret,
        "total_pnl": total_pnl,
        "sharpe"   : sharpe,
        "pf"       : pf,
    }


# =============================================================================
# RAPOR
# =============================================================================

def _format_report(df: pd.DataFrame) -> str:
    today     = datetime.now().strftime("%Y-%m-%d")
    date_min  = df["entry_date"].min()
    date_max  = df["exit_date"].max()

    g = _metrics(df, "gross_ret", "gross_win")
    n = _metrics(df, "net_ret",   "net_win")

    # Per-strategy breakdown
    strat_lines = []
    for s_id, s_name in [(1, "S1 MTF Momentum"),
                          (2, "S2 RSI MeanRev."),
                          (3, "S3 BB Breakout")]:
        sub = df[df["strategy"] == s_id]
        if sub.empty:
            continue
        sg = _metrics(sub, "gross_ret", "gross_win")
        sn = _metrics(sub, "net_ret",   "net_win")
        strat_lines.append(
            f"  {s_name:<20} "
            f"N={sg['n']:>4}  "
            f"Brut WR={sg['win_rate']:.0%}  Net WR={sn['win_rate']:.0%}  "
            f"Net PnL={sn['total_pnl']:>+10,.0f} TL"
        )

    # Exit reason breakdown
    exit_cnt  = df["exit_reason"].value_counts().to_dict()
    exit_lines = [f"  {k:<20}: {v}" for k, v in sorted(exit_cnt.items())]

    # Avg cost per round-trip
    avg_cost_pct = df["cost_pct"].mean()

    # Verdict
    verdict = (
        "DEVAM ET -- net edge var"
        if n["win_rate"] >= 0.55 and n["avg_ret"] > 0 else
        "GOZDEN GECIR -- net edge yok veya yetersiz"
    )

    lines = [
        "",
        "=" * 56,
        f"COST-AWARE BACKTEST RAPORU -- {today}",
        "=" * 56,
        f"Test donemi      : {date_min} -> {date_max}",
        f"Toplam islem     : {g['n']}",
        "",
        "[ BRUT  (maliyet yok) ]",
        f"  Win rate        : {g['win_rate']:.1%}",
        f"  Ort. getiri     : {g['avg_ret']:+.3%} / islem",
        f"  Toplam PnL      : {g['total_pnl']:>+,.0f} TL",
        f"  Sharpe          : {g['sharpe']:.2f}",
        f"  Profit factor   : {g['pf']:.2f}",
        "",
        "[ NET   (gercekci BIST maliyetleri dahil) ]",
        f"  Win rate        : {n['win_rate']:.1%}",
        f"  Ort. getiri     : {n['avg_ret']:+.3%} / islem",
        f"  Toplam PnL      : {n['total_pnl']:>+,.0f} TL",
        f"  Sharpe          : {n['sharpe']:.2f}",
        f"  Profit factor   : {n['pf']:.2f}",
        "",
        "[ MALIYET DETAYI ]",
        f"  Giris slippage  : {BIST_COSTS['slippage_entry']:.2%}  + likidite ek",
        f"  Cikis slippage  : {BIST_COSTS['slippage_exit']:.2%}   + likidite ek",
        f"  Komisyon        : {BIST_COSTS['commission_rate']:.3%} her yon (min {BIST_COSTS['min_commission']:.0f} TL)",
        f"  Ort. toplam     : {avg_cost_pct:.3%} / round-trip",
        f"  Brut WR fark    : {(g['win_rate'] - n['win_rate']):+.1%}  (gercek maliyet etkisi)",
        f"  Brut->Net PnL   : {g['total_pnl']:+,.0f} TL -> {n['total_pnl']:+,.0f} TL",
        "",
        "[ STRATEJI KIRILIMI ]",
    ] + strat_lines + [
        "",
        "[ CIKIS DAGILIMI ]",
    ] + exit_lines + [
        "",
        "[ DEGERLENDIRME ]",
        f"  Net WR >= 55%   : {'GECERLI' if n['win_rate'] >= 0.55 else 'BASARISIZ'}  (gercek: {n['win_rate']:.1%})",
        f"  Net Sharpe >= 1 : {'GECERLI' if n['sharpe'] >= 1.0 else 'BASARISIZ'}  (gercek: {n['sharpe']:.2f})",
        f"  Net edge var mi : {'EVET' if n['avg_ret'] > 0 else 'HAYIR'}  (ort. {n['avg_ret']:+.3%}/islem)",
        "",
        f"  Sonuc: {verdict}",
        "",
        "NOT: Brut WR ile Net WR arasindaki fark gercek maliyet etkisidir.",
        "     Bu fark ne kadar buyukse strateji o kadar maliyet hassasidir.",
        "=" * 56,
    ]

    return "\n".join(lines)


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def run_cost_aware_backtest() -> pd.DataFrame:
    print("=" * 56)
    print("Cost-Aware Backtest -- BIST Gercekci Maliyet Analizi")
    print("=" * 56)
    print(f"Maliyet modeli:")
    print(f"  Slippage (her yon) : {BIST_COSTS['slippage_entry']:.2%} + likidite ek")
    print(f"  Komisyon (her yon) : {BIST_COSTS['commission_rate']:.3%} (min {BIST_COSTS['min_commission']:.0f} TL)")
    print()

    # 1. Stratejileri sifir maliyetle calistir
    print("Stratejiler calistiriliyor (sifir maliyet):")
    all_trades, dates = _run_strategies_zero_cost()

    if not all_trades:
        print("[HATA] Hicbir trade olusturulamadi -- DB verisini kontrol et")
        return pd.DataFrame()

    print(f"\nToplam {len(all_trades)} trade -- maliyet uygulanıyor...")

    # 2. Gercekci maliyetleri uygula
    df = _enrich_with_costs(all_trades)

    # 3. Metrik ozeti
    g = _metrics(df, "gross_ret", "gross_win")
    n = _metrics(df, "net_ret",   "net_win")
    avg_cost = df["cost_pct"].mean()

    print()
    print("[ OZET ]")
    print(f"  {'':20} {'BRUT':>10} {'NET':>10}  {'FARK':>8}")
    print(f"  {'Win rate':20} {g['win_rate']:>10.1%} {n['win_rate']:>10.1%}  {(n['win_rate']-g['win_rate']):>+8.1%}")
    print(f"  {'Ort. getiri':20} {g['avg_ret']:>+9.3%} {n['avg_ret']:>+9.3%}  {(n['avg_ret']-g['avg_ret']):>+8.3%}")
    print(f"  {'Toplam PnL (TL)':20} {g['total_pnl']:>+10,.0f} {n['total_pnl']:>+10,.0f}  "
          f"{(n['total_pnl']-g['total_pnl']):>+8,.0f}")
    print(f"  {'Sharpe':20} {g['sharpe']:>10.2f} {n['sharpe']:>10.2f}")
    print(f"  {'Ort. maliyet/RT':20} {avg_cost:>+9.3%}")

    # 4. Tam rapor olustur + kaydet
    report = _format_report(df)
    print(report)

    today    = datetime.now().strftime("%Y-%m-%d")
    txt_path = RESULTS_DIR / f"cost_aware_backtest_{today}.txt"
    csv_path = RESULTS_DIR / "cost_aware_backtest_detail.csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    df.to_csv(csv_path, index=False)

    print(f"Rapor kaydedildi : {txt_path}")
    print(f"Detay CSV        : {csv_path}")

    # 5. Telegram ozeti
    try:
        from telegram_bot import send_telegram_alert
        verdict = (
            "DEVAM ET -- net edge var"
            if n["win_rate"] >= 0.55 and n["avg_ret"] > 0 else
            "GOZDEN GECIR"
        )
        send_telegram_alert(
            f"Cost-Aware Backtest ({today})\n"
            f"Islemler : {len(df)}\n"
            f"Brut WR  : {g['win_rate']:.1%}  |  Net WR: {n['win_rate']:.1%}\n"
            f"Net PnL  : {n['total_pnl']:+,.0f} TL\n"
            f"Net Sharpe: {n['sharpe']:.2f}\n"
            f"Ort. maliyet/RT: {avg_cost:.3%}\n"
            f"Sonuc    : {verdict}"
        )
    except Exception:
        pass

    return df


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    run_cost_aware_backtest()
