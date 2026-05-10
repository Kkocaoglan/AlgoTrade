"""
backtest.py  — Gerçek BIST verisiyle 3 strateji backtesti
────────────────────────────────────────────────────────────────
Kullanım:  py -3.12 backtest.py

Stratejiler:
  S1 — MTF Momentum   : EMA trend filtresi + MACD onayı
  S2 — RSI Mean Rev.  : RSI aşırı satım/alım + VWAP sapma
  S3 — BB Breakout    : Bollinger sıkışma + hacim kırılımı

Çıktı:
  - Terminal: her strateji için metrikler
  - equity_curves.png : 3 strateji + portföy equity eğrisi
  - backtest_results.csv : tüm trade'ler
────────────────────────────────────────────────────────────────
"""

import sqlite3, time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DB_PATH     = Path(__file__).parent / "trade_data.db"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Konfigürasyon ─────────────────────────────────────────────
CAPITAL     = 100_000          # TL
ALLOC       = {1: 0.45, 2: 0.35, 3: 0.20}   # S1 / S2 / S3
COMMISSION  = 0.002            # %0.2 round trip
SLIPPAGE    = 0.001            # %0.1 her taraf
COMMISSION_ENTRY = COMMISSION / 2
COMMISSION_EXIT  = COMMISSION / 2
SLIPPAGE_ENTRY   = SLIPPAGE
SLIPPAGE_EXIT    = SLIPPAGE
ENTRY_ATR_SLIP_MULT = 0.00
EXIT_ATR_SLIP_MULT  = 0.00
USE_GAP_AWARE_EXITS = True
TRAIN_END   = "2024-01-01"     # bu tarihten önce = train, sonra = test
MAX_POS_PCT = 0.15             # tek pozisyon max %15
DAILY_STOP  = -0.03            # günlük -%3 stop

SYMBOLS = [
    "YKBNK","AKBNK","ISCTR","GARAN",
    "TUPRS","PETKM",
    "TAVHL","FROTO",
    "TCELL","ASELS",
    "BIMAS","MGROS",
    "ENKAI","EKGYO",
]

# ── Veri yükleme ──────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    frames = {}
    for sym in SYMBOLS:
        ohlcv = pd.read_sql(
            "SELECT date,open,high,low,close,volume FROM ohlcv "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,))
        ind = pd.read_sql(
            "SELECT date,ema8,ema21,ema50,ema200,rsi14,"
            "macd_line,macd_signal,macd_hist,"
            "atr14,bb_upper,bb_mid,bb_lower,bb_width,"
            "obv,vol_ratio,mtf_trend,rsi_zone,bb_zone,"
            "above_ema200,golden_cross "
            "FROM indicators WHERE symbol=? ORDER BY date",
            conn, params=(sym,))
        df = pd.merge(ohlcv, ind, on="date", how="inner")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in df.select_dtypes(include="object").columns:
            if col not in ("rsi_zone","bb_zone"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[sym] = df
    conn.close()
    return frames

# ── Tek trade P&L hesabı ──────────────────────────────────────
def trade_pnl(entry_p, exit_p, is_long, size_tl):
    """Komisyon dahil P&L döndür."""
    entry_cost = size_tl * COMMISSION_ENTRY
    exit_cost  = size_tl * COMMISSION_EXIT
    if is_long:
        ret = (exit_p - entry_p) / entry_p
    else:
        ret = (entry_p - exit_p) / entry_p
    return size_tl * ret - entry_cost - exit_cost

def _side_slippage_pct(price, atr, side):
    base = SLIPPAGE_ENTRY if side == "entry" else SLIPPAGE_EXIT
    atr_mult = ENTRY_ATR_SLIP_MULT if side == "entry" else EXIT_ATR_SLIP_MULT
    atr_pct = (atr / price) if price and atr else 0.0
    return max(0.0, base + atr_pct * atr_mult)

def apply_fill_price(raw_price, is_long, side, atr=0.0):
    """Apply adverse side-specific slippage to a raw fill price."""
    slip_pct = _side_slippage_pct(raw_price, atr, side)
    if side == "entry":
        adverse_up = is_long
    else:
        adverse_up = not is_long
    return raw_price * (1 + slip_pct if adverse_up else 1 - slip_pct)

def resolve_barrier_exit(row, pos):
    """Resolve gap-aware stop exits from daily OHLC."""
    open_p  = float(row["open"])
    high_p  = float(row["high"])
    low_p   = float(row["low"])
    close_p = float(row["close"])
    stop_p  = pos["stop"]

    if not USE_GAP_AWARE_EXITS:
        stop_hit = (pos["is_long"] and close_p <= stop_p) or \
                   (not pos["is_long"] and close_p >= stop_p)
        return (close_p, "stop") if stop_hit else (None, None)

    if pos["is_long"]:
        if open_p <= stop_p:
            return open_p, "stop_gap"
        if low_p <= stop_p:
            return stop_p, "stop"
    else:
        if open_p >= stop_p:
            return open_p, "stop_gap"
        if high_p >= stop_p:
            return stop_p, "stop"
    return None, None

# ── STRATEJİ 1: MTF Momentum ──────────────────────────────────
def run_s1(frames, capital):
    """
    Giriş  (LONG) : mtf_trend=+1, fiyat EMA200 üstü, MACD hist > 0,
                    RSI 40-65 arası (ne aşırı alım ne aşırı satım)
    Giriş (SHORT) : mtf_trend=-1, fiyat EMA200 altı, MACD hist < 0,
                    RSI 35-60 arası
    Çıkış         : EMA8/EMA21 kesişimi ters döndüğünde
                    veya ATR×2 stop tetiklendiğinde
    Pozisyon büyüklüğü: Kelly / 2
    """
    trades = []
    equity = [capital]
    cash   = capital
    pos    = {}  # sym → {entry_p, is_long, size, stop, entry_date}

    all_dates = sorted(set(
        d for f in frames.values() for d in f.index
    ))

    for dt in all_dates:
        day_pnl = 0

        # Açık pozisyonları güncelle / kapat
        to_close = []
        for sym, p in pos.items():
            if sym not in frames or dt not in frames[sym].index:
                continue
            row   = frames[sym].loc[dt]
            price = float(row["close"])
            atr   = float(row["atr14"]) if pd.notna(row["atr14"]) else 0

            # EMA kesiş ters döndü mü?
            ema8  = row.get("ema8");  ema21 = row.get("ema21")
            cross_exit = False
            if pd.notna(ema8) and pd.notna(ema21):
                if p["is_long"]  and ema8 < ema21: cross_exit = True
                if not p["is_long"] and ema8 > ema21: cross_exit = True

            exit_px, barrier_reason = resolve_barrier_exit(row, p)
            if exit_px is not None or cross_exit:
                raw_exit = exit_px if exit_px is not None else price
                fill_exit = apply_fill_price(raw_exit, p["is_long"], "exit", atr)
                pnl = trade_pnl(p["entry_p"], fill_exit, p["is_long"], p["size"])
                cash += p["size"] + pnl
                day_pnl += pnl
                reason = barrier_reason if exit_px is not None else "cross"
                trades.append({
                    "strategy":1, "symbol":sym,
                    "entry_date": p["entry_date"], "exit_date": dt,
                    "entry_p": p["entry_p"], "exit_p": fill_exit,
                    "is_long": p["is_long"], "pnl": pnl,
                    "size": p["size"], "exit_reason": reason,
                })
                to_close.append(sym)

        for sym in to_close:
            del pos[sym]

        # Yeni giriş sinyali
        if len(pos) < 4:   # max 4 eş zamanlı pozisyon
            for sym, df in frames.items():
                if sym in pos or dt not in df.index:
                    continue
                row = df.loc[dt]
                if pd.isna(row.get("mtf_trend")): continue

                atr  = float(row["atr14"])   if pd.notna(row.get("atr14"))   else 0
                rsi  = float(row["rsi14"])   if pd.notna(row.get("rsi14"))   else 50
                mh   = float(row["macd_hist"])if pd.notna(row.get("macd_hist"))else 0
                ab200= int(row.get("above_ema200", 0))
                mt   = int(row.get("mtf_trend", 0))
                price= float(row["close"])

                is_long = None
                if mt == 1 and ab200 == 1 and mh > 0 and 40 < rsi < 65:
                    is_long = True
                elif mt == -1 and ab200 == 0 and mh < 0 and 35 < rsi < 60:
                    is_long = False

                if is_long is None:
                    continue

                size = min(capital * MAX_POS_PCT,
                           cash * 0.90 / max(1, 4 - len(pos)))
                if size < 1000 or cash < size:
                    continue

                entry_fill = apply_fill_price(price, is_long, "entry", atr)
                stop = entry_fill - 2*atr if is_long else entry_fill + 2*atr
                cash -= size
                pos[sym] = {
                    "entry_p": entry_fill, "is_long": is_long,
                    "size": size, "stop": stop, "entry_date": dt,
                }

        equity.append(cash + sum(
            frames[s].loc[dt, "close"] * p["size"] / p["entry_p"]
            if s in frames and dt in frames[s].index else p["size"]
            for s, p in pos.items()
        ))

    return trades, equity, all_dates

# ── STRATEJİ 2: RSI Mean Reversion ───────────────────────────
def run_s2(frames, capital):
    """
    LONG  : RSI < 32  VE  fiyat BB alt bandına yakın (close < bb_lower * 1.02)
    SHORT : RSI > 68  VE  fiyat BB üst bandına yakın (close > bb_upper * 0.98)
    Çıkış : RSI nötr bölgeye (45-55) girince  VEYA  ATR×1.5 stop
    Az işlem, yüksek kalite — günde max 2 pozisyon
    """
    trades  = []
    equity  = [capital]
    cash    = capital
    pos     = {}

    all_dates = sorted(set(
        d for f in frames.values() for d in f.index
    ))

    for dt in all_dates:
        # Açık pozisyonları kapat
        to_close = []
        for sym, p in pos.items():
            if sym not in frames or dt not in frames[sym].index:
                continue
            row   = frames[sym].loc[dt]
            price = float(row["close"])
            atr   = float(row["atr14"]) if pd.notna(row.get("atr14")) else 0
            rsi   = float(row["rsi14"]) if pd.notna(row.get("rsi14")) else 50

            rsi_exit = (p["is_long"] and rsi > 50) or \
                       (not p["is_long"] and rsi < 50)

            exit_px, barrier_reason = resolve_barrier_exit(row, p)
            if exit_px is not None or rsi_exit:
                raw_exit = exit_px if exit_px is not None else price
                fill_exit = apply_fill_price(raw_exit, p["is_long"], "exit", atr)
                pnl = trade_pnl(p["entry_p"], fill_exit, p["is_long"], p["size"])
                cash += p["size"] + pnl
                trades.append({
                    "strategy":2, "symbol":sym,
                    "entry_date": p["entry_date"], "exit_date": dt,
                    "entry_p": p["entry_p"], "exit_p": fill_exit,
                    "is_long": p["is_long"], "pnl": pnl,
                    "size": p["size"],
                    "exit_reason": barrier_reason if exit_px is not None else "rsi_mean",
                })
                to_close.append(sym)

        for sym in to_close:
            del pos[sym]

        # Yeni giriş
        if len(pos) < 2:
            for sym, df in frames.items():
                if sym in pos or dt not in df.index:
                    continue
                row   = df.loc[dt]
                price = float(row["close"])
                rsi   = float(row["rsi14"])   if pd.notna(row.get("rsi14"))   else 50
                atr   = float(row["atr14"])   if pd.notna(row.get("atr14"))   else 0
                bbl   = float(row["bb_lower"])if pd.notna(row.get("bb_lower"))else price
                bbu   = float(row["bb_upper"])if pd.notna(row.get("bb_upper"))else price
                vol_r = float(row["vol_ratio"])if pd.notna(row.get("vol_ratio"))else 1

                is_long = None
                if rsi < 32 and price < bbl * 1.02 and vol_r > 0.8:
                    is_long = True
                elif rsi > 68 and price > bbu * 0.98 and vol_r > 0.8:
                    is_long = False

                if is_long is None:
                    continue

                size = min(capital * MAX_POS_PCT, cash * 0.45)
                if size < 1000 or cash < size:
                    continue

                entry_fill = apply_fill_price(price, is_long, "entry", atr)
                stop = entry_fill - 1.5*atr if is_long else entry_fill + 1.5*atr
                cash -= size
                pos[sym] = {
                    "entry_p": entry_fill, "is_long": is_long,
                    "size": size, "stop": stop, "entry_date": dt,
                }

        equity.append(cash + sum(
            frames[s].loc[dt,"close"] * p["size"] / p["entry_p"]
            if s in frames and dt in frames[s].index else p["size"]
            for s, p in pos.items()
        ))

    return trades, equity, all_dates

# ── STRATEJİ 3: BB Breakout ───────────────────────────────────
def run_s3(frames, capital):
    """
    Giriş : BB sıkışması (bb_width < 20. persentil)
             + kırılım (close > bb_upper veya < bb_lower)
             + hacim onayı (vol_ratio > 1.5)
    Çıkış : BB genişlediğinde (bb_width > 60. persentil)
             veya ATR×2 stop
    Yüksek volatilite, daha kısa tutma süresi
    """
    trades = []
    equity = [capital]
    cash   = capital
    pos    = {}

    # Her hisse için BB width persentillerini önceden hesapla
    bb_pcts = {}
    for sym, df in frames.items():
        if "bb_width" in df.columns:
            bb_pcts[sym] = {
                "p20": df["bb_width"].quantile(0.20),
                "p60": df["bb_width"].quantile(0.60),
            }

    all_dates = sorted(set(
        d for f in frames.values() for d in f.index
    ))

    for dt in all_dates:
        to_close = []
        for sym, p in pos.items():
            if sym not in frames or dt not in frames[sym].index:
                continue
            row   = frames[sym].loc[dt]
            price = float(row["close"])
            atr   = float(row["atr14"])   if pd.notna(row.get("atr14"))   else 0
            bbw   = float(row["bb_width"]) if pd.notna(row.get("bb_width"))else 10
            p60   = bb_pcts.get(sym, {}).get("p60", 15)

            expand_exit = bbw > p60

            exit_px, barrier_reason = resolve_barrier_exit(row, p)
            if exit_px is not None or expand_exit:
                raw_exit = exit_px if exit_px is not None else price
                fill_exit = apply_fill_price(raw_exit, p["is_long"], "exit", atr)
                pnl = trade_pnl(p["entry_p"], fill_exit, p["is_long"], p["size"])
                cash += p["size"] + pnl
                trades.append({
                    "strategy":3, "symbol":sym,
                    "entry_date": p["entry_date"], "exit_date": dt,
                    "entry_p": p["entry_p"], "exit_p": fill_exit,
                    "is_long": p["is_long"], "pnl": pnl,
                    "size": p["size"],
                    "exit_reason": barrier_reason if exit_px is not None else "bb_expand",
                })
                to_close.append(sym)

        for sym in to_close:
            del pos[sym]

        if len(pos) < 3:
            for sym, df in frames.items():
                if sym in pos or dt not in df.index:
                    continue
                row   = df.loc[dt]
                price = float(row["close"])
                atr   = float(row["atr14"])   if pd.notna(row.get("atr14"))   else 0
                bbw   = float(row["bb_width"]) if pd.notna(row.get("bb_width"))else 10
                bbu   = float(row["bb_upper"]) if pd.notna(row.get("bb_upper"))else price
                bbl   = float(row["bb_lower"]) if pd.notna(row.get("bb_lower"))else price
                vol_r = float(row["vol_ratio"])if pd.notna(row.get("vol_ratio"))else 1
                p20   = bb_pcts.get(sym, {}).get("p20", 5)

                if bbw > p20 * 1.5:   # sıkışma değil
                    continue
                if vol_r < 1.5:        # hacim onayı yok
                    continue

                is_long = None
                if price > bbu:
                    is_long = True
                elif price < bbl:
                    is_long = False

                if is_long is None:
                    continue

                size = min(capital * MAX_POS_PCT, cash * 0.30)
                if size < 1000 or cash < size:
                    continue

                entry_fill = apply_fill_price(price, is_long, "entry", atr)
                stop = entry_fill - 2*atr if is_long else entry_fill + 2*atr
                cash -= size
                pos[sym] = {
                    "entry_p": entry_fill, "is_long": is_long,
                    "size": size, "stop": stop, "entry_date": dt,
                }

        equity.append(cash + sum(
            frames[s].loc[dt,"close"] * p["size"] / p["entry_p"]
            if s in frames and dt in frames[s].index else p["size"]
            for s, p in pos.items()
        ))

    return trades, equity, all_dates

# ── Metrik hesaplama ──────────────────────────────────────────
def metrics(trades, equity, capital, name):
    eq  = np.array(equity[1:])   # ilk eleman başlangıç kapital
    ret = np.diff(equity) / np.array(equity[:-1])

    # Sharpe (günlük, yıllıklandırılmış)
    mu  = ret.mean()
    std = ret.std()
    sharpe = (mu / std * np.sqrt(252)) if std > 0 else 0

    # Sortino
    down = ret[ret < 0]
    sortino = (mu / down.std() * np.sqrt(252)) if len(down) > 1 and down.std() > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = dd.min()

    # Trade istatistikleri
    df_t  = pd.DataFrame(trades)
    if df_t.empty:
        return {"name": name, "trades": 0, "sharpe": 0,
                "sortino": 0, "max_dd": max_dd,
                "total_ret": (eq[-1]/capital-1)*100 if len(eq) else 0,
                "win_rate": 0, "profit_factor": 0, "avg_hold": 0}

    wins  = df_t[df_t["pnl"] > 0]
    loses = df_t[df_t["pnl"] <= 0]
    wr    = len(wins) / len(df_t) * 100
    pf    = (wins["pnl"].sum() / abs(loses["pnl"].sum())
             if len(loses) and loses["pnl"].sum() != 0 else float("inf"))

    df_t["hold_days"] = (pd.to_datetime(df_t["exit_date"]) -
                          pd.to_datetime(df_t["entry_date"])).dt.days
    avg_hold = df_t["hold_days"].mean()

    return {
        "name"         : name,
        "capital"      : capital,
        "final_equity" : eq[-1] if len(eq) else capital,
        "total_ret"    : (eq[-1]/capital - 1)*100 if len(eq) else 0,
        "sharpe"       : sharpe,
        "sortino"      : sortino,
        "max_dd"       : max_dd * 100,
        "trades"       : len(df_t),
        "win_rate"     : wr,
        "profit_factor": pf,
        "avg_hold_days": avg_hold,
        "gross_pnl"    : df_t["pnl"].sum(),
    }

# ── Grafik ────────────────────────────────────────────────────
def plot_equity(dates, eq1, eq2, eq3, cap1, cap2, cap3):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    colors = ["#00d97e", "#9d7ef5", "#f0a020", "#3d9bff"]
    labels = ["S1 MTF Momentum", "S2 RSI Mean Rev.", "S3 BB Breakout"]
    eqs    = [eq1, eq2, eq3]
    caps   = [cap1, cap2, cap3]

    # Bireysel equity eğrileri
    for i, (eq, cap, col, lbl) in enumerate(zip(eqs, caps, colors, labels)):
        ax  = axes[i//2][i%2] if i < 3 else axes[1][1]
        n   = min(len(dates)+1, len(eq))
        xs  = list(dates[:n-1]) if n > 1 else list(dates)
        ys  = eq[1:n] if len(eq) > 1 else eq
        ax.plot(xs, ys, color=col, linewidth=1.2, label=lbl)
        ax.axhline(cap, color="#ffffff30", linewidth=0.8, linestyle="--")
        ax.fill_between(xs, cap, ys,
                        where=[y >= cap for y in ys],
                        color=col, alpha=0.08)
        ax.fill_between(xs, cap, ys,
                        where=[y < cap for y in ys],
                        color="#f03d5a", alpha=0.10)
        ax.set_title(lbl, color="#c9d1d9", fontsize=10, pad=6)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"₺{v/1000:.0f}K"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Birleşik portföy
    ax4 = axes[1][1]
    max_len = min(len(eq1), len(eq2), len(eq3))
    if max_len > 1:
        combined = [e1+e2+e3 for e1,e2,e3
                    in zip(eq1[:max_len], eq2[:max_len], eq3[:max_len])]
        total_cap = cap1+cap2+cap3
        xs = list(dates[:max_len-1])
        ys = combined[1:]
        ax4.plot(xs, ys, color=colors[3], linewidth=1.5)
        ax4.axhline(total_cap, color="#ffffff30",
                    linewidth=0.8, linestyle="--")
        ax4.fill_between(xs, total_cap, ys,
                         where=[y >= total_cap for y in ys],
                         color=colors[3], alpha=0.10)
        ax4.fill_between(xs, total_cap, ys,
                         where=[y < total_cap for y in ys],
                         color="#f03d5a", alpha=0.12)
        ax4.set_title("Birleşik Portföy", color="#c9d1d9",
                      fontsize=10, pad=6)
        ax4.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"₺{v/1000:.0f}K"))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.suptitle("BIST Backtest — Equity Eğrileri (2022-2026)",
                 color="#e6edf3", fontsize=13, y=1.01)
    plt.tight_layout()
    out = RESULTS_DIR / "equity_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return out

# ── Metrikleri yazdır ─────────────────────────────────────────
def print_metrics(m_list):
    print("\n" + "="*68)
    print("BACKTEST SONUÇLARI — Tüm Dönem (2022-2026)")
    print("="*68)
    hdr = f"{'STRATEJİ':<20} {'SERMAYE':>9} {'GETIRI':>8} {'SHARPE':>7} " \
          f"{'SORTINO':>8} {'MAX DD':>8} {'TRADE':>6} {'WIN%':>6} {'PF':>6}"
    print(hdr)
    print("-"*68)
    for m in m_list:
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
        print(
            f"  {m['name']:<18} "
            f"₺{m.get('capital',0)/1000:>6.0f}K "
            f"{m['total_ret']:>+7.1f}% "
            f"{m['sharpe']:>7.2f} "
            f"{m['sortino']:>8.2f} "
            f"{m['max_dd']:>+7.1f}% "
            f"{m['trades']:>6} "
            f"{m['win_rate']:>5.0f}% "
            f"{pf_str:>6}"
        )

    print("\n  Değerlendirme:")
    for m in m_list:
        sh = m["sharpe"]
        dd = abs(m["max_dd"])
        if sh >= 1.5 and dd <= 15:
            verdict = "GUCLU — paper trade'e hazır"
        elif sh >= 1.0 and dd <= 25:
            verdict = "KABUL EDILEBILIR — parametre optimizasyonu yap"
        elif sh >= 0.5:
            verdict = "ZAYIF — strateji revizyonu gerekli"
        else:
            verdict = "BASARISIZ — bu stratejiyi kullanma"
        print(f"  {m['name']:<22} → {verdict}")

    # Portföy özeti
    total_ret = sum(m.get("gross_pnl", 0) for m in m_list)
    total_cap = CAPITAL
    print(f"\n  Portföy net P&L  : ₺{total_ret:>10,.0f}")
    print(f"  Portföy getiri   : %{total_ret/total_cap*100:>+.2f}")

def print_top_trades(all_trades):
    df = pd.DataFrame(all_trades)
    if df.empty:
        return
    print("\n" + "="*68)
    print("EN İYİ 5 TRADE")
    print("="*68)
    top = df.nlargest(5, "pnl")
    for _, r in top.iterrows():
        print(f"  S{r['strategy']} {r['symbol']:<8} "
              f"{'LONG' if r['is_long'] else 'SHORT'} "
              f"{str(r['entry_date'])[:10]} → {str(r['exit_date'])[:10]}  "
              f"P&L: ₺{r['pnl']:>+8,.0f}  ({r['exit_reason']})")

    print("\nEN KÖTÜ 5 TRADE")
    print("="*68)
    bot = df.nsmallest(5, "pnl")
    for _, r in bot.iterrows():
        print(f"  S{r['strategy']} {r['symbol']:<8} "
              f"{'LONG' if r['is_long'] else 'SHORT'} "
              f"{str(r['entry_date'])[:10]} → {str(r['exit_date'])[:10]}  "
              f"P&L: ₺{r['pnl']:>+8,.0f}  ({r['exit_reason']})")

# ── Entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("="*68)
    print("BIST Backtest Motoru v1")
    print(f"Sermaye: ₺{CAPITAL:,}  |  Komisyon: %{COMMISSION*100:.1f}  "
          f"|  Slippage: %{SLIPPAGE*100:.1f}")
    print(f"Dönem  : 2022-01-01 → bugün")
    print("="*68)

    print("\nVeri yükleniyor...", end=" ", flush=True)
    frames = load_data()
    print(f"OK ({len(frames)} hisse)")

    cap1 = int(CAPITAL * ALLOC[1])
    cap2 = int(CAPITAL * ALLOC[2])
    cap3 = int(CAPITAL * ALLOC[3])

    print(f"\nS1 MTF Momentum   (₺{cap1:,}) çalışıyor...", end=" ", flush=True)
    t1, e1, dates = run_s1(frames, cap1)
    print(f"{len(t1)} trade")

    print(f"S2 RSI Mean Rev.  (₺{cap2:,}) çalışıyor...", end=" ", flush=True)
    t2, e2, _     = run_s2(frames, cap2)
    print(f"{len(t2)} trade")

    print(f"S3 BB Breakout    (₺{cap3:,}) çalışıyor...", end=" ", flush=True)
    t3, e3, _     = run_s3(frames, cap3)
    print(f"{len(t3)} trade")

    all_trades = t1 + t2 + t3
    m1 = metrics(t1, e1, cap1, "S1 MTF Momentum")
    m2 = metrics(t2, e2, cap2, "S2 RSI MeanRev.")
    m3 = metrics(t3, e3, cap3, "S3 BB Breakout")

    print_metrics([m1, m2, m3])
    print_top_trades(all_trades)

    # CSV kaydet
    if all_trades:
        csv_path = RESULTS_DIR / "backtest_results.csv"
        pd.DataFrame(all_trades).to_csv(csv_path, index=False)
        print(f"\nTrade detayları: {csv_path}")

    # Grafik
    print("Grafik oluşturuluyor...", end=" ", flush=True)
    try:
        png = plot_equity(dates, e1, e2, e3, cap1, cap2, cap3)
        print(f"OK → {png}")
    except Exception as ex:
        print(f"HATA: {ex}")

    print(f"\nToplam süre: {time.time()-t0:.1f}s")
    print("\nSonraki adım: py -3.12 walk_forward.py")
