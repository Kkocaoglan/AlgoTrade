"""
daily_report.py — Günlük P&L ve Portföy Raporu
────────────────────────────────────────────────────────────────
Kullanim: py -3.12 daily_report.py [--days N]

Cikti:
  - Terminal: ozet tablo
  - results/daily_report_YYYY-MM-DD.txt : metin raporu
  - results/pnl_chart.png               : equity eğrisi
────────────────────────────────────────────────────────────────
"""

import sqlite3, sys, struct
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from telegram_bot import alert_daily_summary as _tg_daily_summary
    HAS_TG = True
except Exception:
    HAS_TG = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

DB_PATH = Path(__file__).parent / "trade_data.db"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

CAPITAL = 100_000

STRATEGY_LABELS = {
    "ML_AL":  "XGBoost AL",
    "ML_SAT": "XGBoost SAT",
    "S1":     "MTF Momentum",
    "S2":     "RSI MeanRev",
    "S3":     "BB Breakout",
}

# ── Yardimci ──────────────────────────────────────────────────
def pct(val, total):
    return f"{val/total*100:+.2f}%" if total else "N/A"

def tl(val):
    return f"{'+'if val>=0 else ''}{val:,.0f} TL"

# ── Veri yükle ────────────────────────────────────────────────
def load_data(conn, days=30):
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        event_date  TEXT,
        event_type  TEXT,
        symbol      TEXT,
        strategy    TEXT,
        details     TEXT
    )""")
    for stmt in [
        "ALTER TABLE paper_positions ADD COLUMN signal_prob REAL",
        "ALTER TABLE paper_positions ADD COLUMN regime TEXT",
        "ALTER TABLE paper_positions ADD COLUMN entry_note TEXT",
        "ALTER TABLE paper_trades ADD COLUMN signal_prob REAL",
        "ALTER TABLE paper_trades ADD COLUMN regime TEXT",
        "ALTER TABLE paper_trades ADD COLUMN entry_note TEXT",
        "ALTER TABLE paper_trades ADD COLUMN exit_note TEXT",
    ]:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    # Equity geçmisi
    eq_df = pd.read_sql("""
        SELECT date, cash, positions_value, total_equity,
               daily_pnl, open_positions
        FROM paper_equity
        WHERE date >= ?
        ORDER BY date
    """, conn, params=(cutoff,))

    # Kapanan tradeler
    trades_df = pd.read_sql("""
        SELECT symbol, strategy, entry_date, exit_date,
               entry_price, exit_price, size_tl, is_long,
               pnl, pct_return, exit_reason,
               COALESCE(signal_prob, 0) AS signal_prob,
               COALESCE(regime, '') AS regime,
               COALESCE(entry_note, '') AS entry_note,
               COALESCE(exit_note, '') AS exit_note
        FROM paper_trades
        WHERE exit_date >= ?
        ORDER BY exit_date DESC
    """, conn, params=(cutoff,))

    # Acik pozisyonlar
    open_df = pd.read_sql("""
        SELECT symbol, strategy, entry_date, entry_price,
               size_tl, is_long, stop_price, target_price, confidence,
               COALESCE(signal_prob, 0) AS signal_prob,
               COALESCE(regime, '') AS regime,
               COALESCE(entry_note, '') AS entry_note
        FROM paper_positions
        WHERE status='open'
        ORDER BY entry_date DESC
    """, conn)

    # confidence bytes->float (numpy float32 SQLite blob olarak saklanir)
    if not open_df.empty and open_df["confidence"].dtype == object:
        open_df["confidence"] = open_df["confidence"].apply(
            lambda v: struct.unpack("<f", v)[0] if isinstance(v, (bytes, bytearray)) else float(v)
        )

    # Tüm zamanlar
    all_trades = pd.read_sql("""
        SELECT symbol, strategy, pnl,
               CASE WHEN pnl > 0 THEN 1 ELSE 0 END as win
        FROM paper_trades
    """, conn)

    events_df = pd.read_sql("""
        SELECT event_date, event_type, symbol, strategy, details
        FROM paper_events
        WHERE event_date >= ?
        ORDER BY id DESC
    """, conn, params=(cutoff,))

    return eq_df, trades_df, open_df, all_trades, events_df

# ── Güncel fiyatlar ───────────────────────────────────────────
def get_current_prices(conn, symbols):
    prices = {}
    for sym in symbols:
        row = conn.execute(
            "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (sym,)).fetchone()
        prices[sym] = row[0] if row else None
    return prices

# ── Strateji bazli P&L özeti ──────────────────────────────────
def strategy_summary(all_trades):
    if all_trades.empty:
        return pd.DataFrame()
    grp = all_trades.groupby("strategy").agg(
        trades   = ("pnl", "count"),
        total_pnl= ("pnl", "sum"),
        win_rate = ("win", "mean"),
        avg_pnl  = ("pnl", "mean"),
        best     = ("pnl", "max"),
        worst    = ("pnl", "min"),
    ).reset_index()
    grp["win_rate"] *= 100
    return grp.sort_values("total_pnl", ascending=False)

# ── Grafik ────────────────────────────────────────────────────
def plot_equity(eq_df):
    if eq_df.empty or len(eq_df) < 2:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    dates  = pd.to_datetime(eq_df["date"])
    equity = eq_df["total_equity"].values

    # Equity curve
    ax1 = axes[0]
    color = "#00d97e"
    ax1.plot(dates, equity, color=color, linewidth=1.5, label="Portföy")
    ax1.axhline(CAPITAL, color="#ffffff30", linewidth=0.8, linestyle="--",
                label=f"Baslangic ({CAPITAL:,} TL)")
    ax1.fill_between(dates, CAPITAL, equity,
                     where=equity >= CAPITAL, color=color, alpha=0.10)
    ax1.fill_between(dates, CAPITAL, equity,
                     where=equity < CAPITAL,  color="#f03d5a", alpha=0.15)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"₺{v/1000:.0f}K"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax1.legend(facecolor="#1c2128", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=8)
    ax1.set_title("Paper Trade — Equity Eğrisi",
                  color="#e6edf3", fontsize=12, pad=8)

    # Günlük P&L bar chart
    ax2 = axes[1]
    daily_pnl = eq_df["daily_pnl"].values
    colors_bar = ["#00d97e" if p >= 0 else "#f03d5a" for p in daily_pnl]
    ax2.bar(dates, daily_pnl, color=colors_bar, alpha=0.8, width=0.8)
    ax2.axhline(0, color="#ffffff30", linewidth=0.6)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"₺{v/1000:.0f}K"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax2.set_title("Günlük P&L", color="#e6edf3", fontsize=10, pad=4)

    plt.tight_layout(h_pad=2)
    out = RESULTS / "pnl_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return out

# ── Drawdown hesapla ──────────────────────────────────────────
def calc_drawdown(equity_arr):
    if len(equity_arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_arr)
    dd   = (equity_arr - peak) / (peak + 1e-9)
    return float(dd.min() * 100)

# ── Sharpe (mevcut veriyle) ───────────────────────────────────
def calc_sharpe(equity_arr):
    if len(equity_arr) < 5:
        return 0.0
    ret = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)
    mu  = ret.mean()
    std = ret.std()
    return float(mu / std * np.sqrt(252)) if std > 0 else 0.0

# ── Makro anlık veri ──────────────────────────────────────────
def fetch_macro_snapshot():
    """USDTRY ve Brent ham petrol degisimini yfinance'tan cek."""
    result = {}
    if not HAS_YF:
        return result
    for key, ticker in [("usdtry", "TRY=X"), ("brent", "BZ=F")]:
        try:
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                last  = float(df["close"].iloc[-1])
                prev  = float(df["close"].iloc[-2])
                chg   = (last - prev) / prev * 100
                result[key] = {"price": last, "change_pct": chg}
        except Exception as e:
            print(f"  [Makro hatasi {ticker}] {e}")
    return result

# ── Sembol bazli performans ───────────────────────────────────
def symbol_performance(all_trades):
    """En iyi ve en kotu sembol bul (kapali tradelerden)."""
    if all_trades.empty or "symbol" not in all_trades.columns:
        return pd.DataFrame()
    grp = all_trades.groupby("symbol").agg(
        trades    = ("pnl", "count"),
        total_pnl = ("pnl", "sum"),
        win_rate  = ("win", "mean"),
    ).reset_index()
    grp["win_rate"] *= 100
    return grp.sort_values("total_pnl", ascending=False)


def current_position_snapshot(open_df, conn):
    if open_df.empty:
        return open_df.copy(), {}

    prices = get_current_prices(conn, open_df["symbol"].tolist())
    pos_df = open_df.copy()
    pos_df["current_price"] = pos_df["symbol"].map(prices).fillna(pos_df["entry_price"])

    def calc_row(row):
        if row["is_long"]:
            pnl_pct = (row["current_price"] / row["entry_price"] - 1) * 100
            market_value = row["size_tl"] * (row["current_price"] / row["entry_price"])
        else:
            pnl_pct = (row["entry_price"] / row["current_price"] - 1) * 100 if row["current_price"] else 0.0
            market_value = row["size_tl"] * (1 + pnl_pct / 100.0)
        pnl_tl = market_value - row["size_tl"]
        stop_gap_pct = abs(row["entry_price"] - row["stop_price"]) / max(row["entry_price"], 1e-9) * 100
        return pd.Series({
            "unrealized_pct": pnl_pct,
            "unrealized_tl": pnl_tl,
            "market_value": market_value,
            "risk_to_stop_pct": stop_gap_pct,
        })

    pos_df[["unrealized_pct", "unrealized_tl", "market_value", "risk_to_stop_pct"]] = pos_df.apply(calc_row, axis=1)
    pos_df["gross_exposure_pct"] = pos_df["size_tl"] / CAPITAL * 100
    pos_df["net_exposure_signed"] = np.where(pos_df["is_long"], pos_df["size_tl"], -pos_df["size_tl"])

    exposure = {
        "gross_tl": float(pos_df["size_tl"].sum()),
        "net_tl": float(pos_df["net_exposure_signed"].sum()),
        "long_tl": float(pos_df.loc[pos_df["is_long"] == 1, "size_tl"].sum()),
        "short_tl": float(pos_df.loc[pos_df["is_long"] == 0, "size_tl"].sum()),
        "largest_position_pct": float(pos_df["gross_exposure_pct"].max()) if not pos_df.empty else 0.0,
        "unrealized_tl": float(pos_df["unrealized_tl"].sum()),
    }
    return pos_df, exposure


def latest_regime_summary(conn, open_positions=None):
    reg_df = pd.read_sql("""
        SELECT o.symbol, i.mtf_trend, i.rsi14, i.above_ema200
        FROM (
            SELECT symbol, MAX(date) AS max_date
            FROM indicators
            GROUP BY symbol
        ) latest
        JOIN indicators i
          ON i.symbol = latest.symbol AND i.date = latest.max_date
        JOIN ohlcv o
          ON o.symbol = i.symbol AND o.date = i.date
    """, conn)
    if reg_df.empty:
        return {}, pd.DataFrame()

    reg_df["regime"] = reg_df["mtf_trend"].map({1: "YUKARI", -1: "ASAGI"}).fillna("YATAY")
    summary = reg_df["regime"].value_counts().to_dict()
    if open_positions is not None and not open_positions.empty:
        merged = open_positions.merge(reg_df[["symbol", "regime"]], on="symbol", how="left", suffixes=("", "_latest"))
    else:
        merged = reg_df
    return summary, merged

# ── Raporu yaz ────────────────────────────────────────────────
def build_report(eq_df, trades_df, open_df, all_trades, conn, days):
    today = date.today().isoformat()
    lines = []
    sep = "=" * 64

    lines += [sep, f"BIST Paper Trade — Günlük Rapor", f"Tarih   : {today}",
              f"Dönem   : Son {days} gün", sep, ""]

    # ── 1. Portföy Özeti ──
    lines += ["1. PORTFÖY ÖZETİ", "-" * 40]
    if not eq_df.empty:
        last    = eq_df.iloc[-1]
        equity  = last["total_equity"]
        cash    = last["cash"]
        pos_val = last["positions_value"]
        n_open  = int(last["open_positions"])
        tot_pnl = equity - CAPITAL
        d_pnl   = last["daily_pnl"]

        arr = eq_df["total_equity"].values
        max_dd  = calc_drawdown(arr)
        sharpe  = calc_sharpe(arr)

        lines += [
            f"  Baslangic sermayesi : {CAPITAL:>12,.0f} TL",
            f"  Nakit               : {cash:>12,.0f} TL",
            f"  Pozisyon degeri     : {pos_val:>12,.0f} TL",
            f"  Toplam Equity       : {equity:>12,.0f} TL",
            f"  Toplam P&L          : {tl(tot_pnl):>12}  ({pct(tot_pnl, CAPITAL)})",
            f"  Günlük P&L          : {tl(d_pnl):>12}",
            f"  Acik Pozisyon       : {n_open}",
            f"  Max Drawdown        : {max_dd:>+.2f}%",
            f"  Sharpe (günlük veri): {sharpe:>.2f}",
        ]
    else:
        lines += ["  (henüz equity kaydı yok — önce paper_trade.py çalistir)"]
    lines.append("")

    # ── 2. Acik Pozisyonlar ──
    lines += ["2. ACIK POZISYONLAR", "-" * 40]
    if not open_df.empty:
        # Anlık fiyatlari al
        prices = get_current_prices(conn, open_df["symbol"].tolist())
        for _, pos in open_df.iterrows():
            cur = prices.get(pos["symbol"]) or pos["entry_price"]
            pnl_pct = (cur / pos["entry_price"] - 1) * 100
            if not pos["is_long"]:
                pnl_pct *= -1
            dir_s = "LONG" if pos["is_long"] else "SHORT"
            lines.append(
                f"  {pos['symbol']:<8} {dir_s:<6} {pos['strategy']:<12}"
                f"  Giris: {pos['entry_price']:>8.2f}"
                f"  Simdiki: {cur:>8.2f}"
                f"  Gerceklesm.: {pnl_pct:>+.1f}%"
                f"  Stop: {pos['stop_price']:>8.2f}"
                f"  Güven: {pos['confidence']:.0f}%"
            )
    else:
        lines += ["  (acik pozisyon yok)"]
    lines.append("")

    # ── 3. Strateji Bazli P&L ──
    lines += ["3. STRATEJİ BAZLI P&L (tüm kapali tradeler)", "-" * 40]
    strat_df = strategy_summary(all_trades)
    if not strat_df.empty:
        lines.append(
            f"  {'STRATEJI':<14} {'TRADE':>6} {'KAZANMA%':>9} "
            f"{'TOPLAM P&L':>12} {'ORT P&L':>9} {'EN IYI':>9} {'EN KÖTÜ':>9}"
        )
        lines.append("  " + "-" * 74)
        for _, row in strat_df.iterrows():
            strat = STRATEGY_LABELS.get(row["strategy"], row["strategy"])
            lines.append(
                f"  {strat:<14} {int(row['trades']):>6}  "
                f"{row['win_rate']:>8.0f}%"
                f"  {row['total_pnl']:>+12,.0f} TL"
                f"  {row['avg_pnl']:>+9,.0f}"
                f"  {row['best']:>+9,.0f}"
                f"  {row['worst']:>+9,.0f}"
            )
        total_pnl_all = all_trades["pnl"].sum()
        total_n       = len(all_trades)
        total_wr      = all_trades["win"].mean() * 100 if total_n else 0
        lines += ["  " + "-" * 74,
                  f"  {'TOPLAM':<14} {total_n:>6}  {total_wr:>8.0f}%"
                  f"  {total_pnl_all:>+12,.0f} TL"]
    else:
        lines += ["  (henüz kapanan trade yok)"]
    lines.append("")

    # ── 4. Son Kapanan Tradeler ──
    lines += [f"4. SON {min(15, len(trades_df))} KAPANAN TRADE", "-" * 40]
    if not trades_df.empty:
        for _, t in trades_df.head(15).iterrows():
            dir_s   = "L" if t["is_long"] else "S"
            strat   = STRATEGY_LABELS.get(t["strategy"], t["strategy"])
            pnl_str = tl(t["pnl"])
            lines.append(
                f"  {t['symbol']:<8} {dir_s} {strat:<12}"
                f"  {t['entry_date']} -> {t['exit_date']}"
                f"  {t['entry_price']:>8.2f} -> {t['exit_price']:>8.2f}"
                f"  P&L: {pnl_str:>12}  ({t['pct_return']:>+.1f}%)"
                f"  {t['exit_reason']}"
            )
    else:
        lines += ["  (yok)"]
    lines.append("")

    # ── 5. Günlük P&L Tablosu ──
    lines += ["5. GÜNLÜK P&L GEÇMİŞİ (son 14 gün)", "-" * 40]
    if not eq_df.empty:
        for _, row in eq_df.tail(14).iterrows():
            bar_len = int(abs(row["daily_pnl"]) / 1000)
            bar_ch  = "+" if row["daily_pnl"] >= 0 else "-"
            bar     = bar_ch * min(bar_len, 20)
            lines.append(
                f"  {row['date']}  "
                f"{tl(row['daily_pnl']):>12}  "
                f"Equity: {row['total_equity']:>10,.0f} TL  "
                f"|{bar}"
            )
    else:
        lines += ["  (yok)"]

    # ── 6. Genel Performans Ozeti ──
    lines += ["6. GENEL PERFORMANS (tum zamanlar)", "-" * 40]
    if not all_trades.empty:
        total_n  = len(all_trades)
        wins     = all_trades["win"].sum()
        wr_total = wins / total_n * 100 if total_n else 0
        pnl_sum  = all_trades["pnl"].sum()
        arr      = eq_df["total_equity"].values if not eq_df.empty else np.array([CAPITAL])
        sharpe   = calc_sharpe(arr)
        lines += [
            f"  Toplam trade       : {total_n}",
            f"  Kazanma orani      : {wr_total:.1f}%",
            f"  Toplam realize P&L : {tl(pnl_sum)}",
            f"  Sharpe (paper)     : {sharpe:.2f}",
        ]

        # En iyi / en kotu sembol
        sym_df = symbol_performance(all_trades)
        if not sym_df.empty:
            best = sym_df.iloc[0]
            worst = sym_df.iloc[-1]
            lines += [
                f"  En iyi sembol      : {best['symbol']}  "
                f"({tl(best['total_pnl'])}, {best['win_rate']:.0f}% kazan, {int(best['trades'])} trade)",
                f"  En kotu sembol     : {worst['symbol']}  "
                f"({tl(worst['total_pnl'])}, {worst['win_rate']:.0f}% kazan, {int(worst['trades'])} trade)",
            ]
    else:
        lines += ["  (henuz kapanan trade yok)"]
    lines.append("")

    # ── 7. Makro Anlık Gorunum ──
    lines += ["7. MAKRO ANLIKTANIM", "-" * 40]
    macro = fetch_macro_snapshot()
    if macro:
        if "usdtry" in macro:
            m = macro["usdtry"]
            sign = "+" if m["change_pct"] >= 0 else ""
            lines.append(f"  USD/TRY  : {m['price']:.4f}  ({sign}{m['change_pct']:.2f}% gun degisimi)")
        if "brent" in macro:
            m = macro["brent"]
            sign = "+" if m["change_pct"] >= 0 else ""
            lines.append(f"  Brent    : ${m['price']:.2f}  ({sign}{m['change_pct']:.2f}% gun degisimi)")
            if m["change_pct"] > 2.0:
                lines.append(f"  [!] MAKRO UYARI: Brent +{m['change_pct']:.1f}% -- TUPRS SHORT pozisyonunu gozden gecir!")
            elif m["change_pct"] < -2.0:
                lines.append(f"  [*] Brent {m['change_pct']:.1f}% -- TUPRS SHORT icin olumlu (petrol dusuyor)")
    else:
        lines += ["  (yfinance yuklu degil veya veri alinamadi)"]
    lines.append("")

    lines += [sep, "Rapor sonu.", sep]
    return "\n".join(lines)


def build_report_v2(eq_df, trades_df, open_df, all_trades, events_df, conn, days):
    today = date.today().isoformat()
    lines = []
    sep = "=" * 64
    open_snap, exposure = current_position_snapshot(open_df, conn)
    regime_summary, regime_df = latest_regime_summary(conn, open_snap)
    realized_total = float(all_trades["pnl"].sum()) if not all_trades.empty else 0.0
    realized_window = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    unrealized_total = exposure.get("unrealized_tl", 0.0)

    lines += [sep, "BIST Paper Trade - Gunluk Rapor", f"Tarih   : {today}",
              f"Donem   : Son {days} gun", sep, ""]

    lines += ["1. EXECUTIVE SUMMARY", "-" * 40]
    if not eq_df.empty:
        last = eq_df.iloc[-1]
        equity = last["total_equity"]
        cash = last["cash"]
        pos_val = last["positions_value"]
        n_open = int(last["open_positions"])
        tot_pnl = equity - CAPITAL
        d_pnl = last["daily_pnl"]
        arr = eq_df["total_equity"].values
        lines += [
            f"  Baslangic sermayesi : {CAPITAL:>12,.0f} TL",
            f"  Toplam Equity       : {equity:>12,.0f} TL",
            f"  Nakit               : {cash:>12,.0f} TL",
            f"  Pozisyon degeri     : {pos_val:>12,.0f} TL",
            f"  Toplam P&L          : {tl(tot_pnl):>12}  ({pct(tot_pnl, CAPITAL)})",
            f"  Gunluk P&L          : {tl(d_pnl):>12}",
            f"  Realize P&L         : {tl(realized_total):>12}",
            f"  Unrealize P&L       : {tl(unrealized_total):>12}",
            f"  Acik Pozisyon       : {n_open}",
            f"  Max Drawdown        : {calc_drawdown(arr):>+.2f}%",
            f"  Sharpe (gunluk veri): {calc_sharpe(arr):>.2f}",
        ]
    else:
        lines += ["  (henuz equity kaydi yok - once paper_trade.py calistir)"]
    lines.append("")

    lines += ["2. EXPOSURE SUMMARY", "-" * 40]
    lines += [
        f"  Gross exposure      : {exposure.get('gross_tl', 0.0):>12,.0f} TL  ({exposure.get('gross_tl', 0.0)/CAPITAL*100:.1f}%)",
        f"  Net exposure        : {exposure.get('net_tl', 0.0):>+12,.0f} TL",
        f"  Long exposure       : {exposure.get('long_tl', 0.0):>12,.0f} TL",
        f"  Short exposure      : {exposure.get('short_tl', 0.0):>12,.0f} TL",
        f"  Largest position    : {exposure.get('largest_position_pct', 0.0):>11.1f}%",
    ]
    if not open_snap.empty:
        strat_exposure = open_snap.groupby("strategy")["size_tl"].sum().sort_values(ascending=False)
        if not strat_exposure.empty:
            lines.append("  Strategy exposure   : " + ", ".join(
                f"{STRATEGY_LABELS.get(k, k)} {v/CAPITAL*100:.1f}%"
                for k, v in strat_exposure.items()
            ))
    lines.append("")

    lines += ["3. RISK SUMMARY", "-" * 40]
    if not open_snap.empty:
        near_stop = int((open_snap["unrealized_pct"].abs() >= open_snap["risk_to_stop_pct"] * 0.5).sum())
        lines += [
            f"  Realize (window)    : {tl(realized_window)}",
            f"  Unrealize (open)    : {tl(unrealized_total)}",
            f"  Stop-near positions : {near_stop}",
            f"  Ortalama stop riski : {open_snap['risk_to_stop_pct'].mean():.2f}%",
            f"  Maks stop riski     : {open_snap['risk_to_stop_pct'].max():.2f}%",
        ]
    else:
        lines += [f"  Realize (window)    : {tl(realized_window)}", "  (acik pozisyon yok)"]
    lines.append("")

    lines += ["4. REGIME SUMMARY", "-" * 40]
    if regime_summary:
        lines.append("  Market regime count : " + ", ".join(
            f"{name} {count}" for name, count in sorted(regime_summary.items())
        ))
        if not open_snap.empty and "regime_latest" in regime_df.columns:
            aligned = regime_df.apply(
                lambda r: (bool(r["is_long"]) and r["regime_latest"] == "YUKARI") or
                          ((not bool(r["is_long"])) and r["regime_latest"] == "ASAGI"),
                axis=1
            ).sum()
            lines.append(f"  Regime alignment    : {aligned}/{len(regime_df)} open positions aligned")
    else:
        lines += ["  (regime verisi yok)"]
    lines.append("")

    lines += ["5. OPEN POSITIONS", "-" * 40]
    if not open_snap.empty:
        for _, pos in open_snap.iterrows():
            dir_s = "LONG" if pos["is_long"] else "SHORT"
            lines.append(
                f"  {pos['symbol']:<8} {dir_s:<6} {pos['strategy']:<12}"
                f"  Giris:{pos['entry_price']:>8.2f}  Simdiki:{pos['current_price']:>8.2f}"
                f"  Unreal:{pos['unrealized_tl']:>+8,.0f} TL ({pos['unrealized_pct']:>+.1f}%)"
            )
            lines.append(
                f"    Boyut:{pos['size_tl']:>10,.0f} TL  Stop:{pos['stop_price']:.2f}  "
                f"Hedef:{pos['target_price']:.2f}  Guven:{float(pos['confidence']):.0f}%  Regime:{(pos.get('regime') or 'N/A')}"
            )
            if pos.get("entry_note"):
                lines.append(f"    Entry thesis: {pos['entry_note']}")
    else:
        lines += ["  (acik pozisyon yok)"]
    lines.append("")

    lines += ["6. STRATEJI BAZLI P&L (tum kapali tradeler)", "-" * 40]
    strat_df = strategy_summary(all_trades)
    if not strat_df.empty:
        lines.append(
            f"  {'STRATEJI':<14} {'TRADE':>6} {'KAZANMA%':>9} "
            f"{'TOPLAM P&L':>12} {'ORT P&L':>9} {'EN IYI':>9} {'EN KOTU':>9}"
        )
        lines.append("  " + "-" * 74)
        for _, row in strat_df.iterrows():
            strat = STRATEGY_LABELS.get(row["strategy"], row["strategy"])
            lines.append(
                f"  {strat:<14} {int(row['trades']):>6}  "
                f"{row['win_rate']:>8.0f}%"
                f"  {row['total_pnl']:>+12,.0f} TL"
                f"  {row['avg_pnl']:>+9,.0f}"
                f"  {row['best']:>+9,.0f}"
                f"  {row['worst']:>+9,.0f}"
            )
    else:
        lines += ["  (henuz kapanan trade yok)"]
    lines.append("")

    lines += [f"7. SON {min(15, len(trades_df))} KAPANAN TRADE", "-" * 40]
    if not trades_df.empty:
        for _, t in trades_df.head(15).iterrows():
            dir_s = "L" if t["is_long"] else "S"
            strat = STRATEGY_LABELS.get(t["strategy"], t["strategy"])
            lines.append(
                f"  {t['symbol']:<8} {dir_s} {strat:<12}"
                f"  {t['entry_date']} -> {t['exit_date']}"
                f"  {t['entry_price']:>8.2f} -> {t['exit_price']:>8.2f}"
                f"  P&L: {tl(t['pnl']):>12}  ({t['pct_return']:>+.1f}%)  {t['exit_reason']}"
            )
            if t["entry_note"]:
                lines.append(f"    Entry: {t['entry_note']}")
            if t["exit_note"]:
                lines.append(f"    Exit : {t['exit_note']}")
    else:
        lines += ["  (yok)"]
    lines.append("")

    lines += [f"8. TRADE REASON LOG (son {min(20, len(events_df))} olay)", "-" * 40]
    if not events_df.empty:
        for _, ev in events_df.head(20).iterrows():
            lines.append(
                f"  {ev['event_date']}  {ev['event_type']:<12} "
                f"{(ev['symbol'] or '-'): <8} {(ev['strategy'] or '-'): <10} {ev['details']}"
            )
    else:
        lines += ["  (event log yok)"]
    lines.append("")

    lines += ["9. GUNLUK P&L GECMISI (son 14 gun)", "-" * 40]
    if not eq_df.empty:
        for _, row in eq_df.tail(14).iterrows():
            bar_len = int(abs(row["daily_pnl"]) / 1000)
            bar = ("+" if row["daily_pnl"] >= 0 else "-") * min(bar_len, 20)
            lines.append(
                f"  {row['date']}  {tl(row['daily_pnl']):>12}  "
                f"Equity: {row['total_equity']:>10,.0f} TL  |{bar}"
            )
    else:
        lines += ["  (yok)"]
    lines.append("")

    lines += ["10. GENEL PERFORMANS (tum zamanlar)", "-" * 40]
    if not all_trades.empty:
        total_n = len(all_trades)
        wins = all_trades["win"].sum()
        wr_total = wins / total_n * 100 if total_n else 0
        arr = eq_df["total_equity"].values if not eq_df.empty else np.array([CAPITAL])
        lines += [
            f"  Toplam trade       : {total_n}",
            f"  Kazanma orani      : {wr_total:.1f}%",
            f"  Toplam realize P&L : {tl(realized_total)}",
            f"  Sharpe (paper)     : {calc_sharpe(arr):.2f}",
        ]
        sym_df = symbol_performance(all_trades)
        if not sym_df.empty:
            best = sym_df.iloc[0]
            worst = sym_df.iloc[-1]
            lines += [
                f"  En iyi sembol      : {best['symbol']} ({tl(best['total_pnl'])}, {best['win_rate']:.0f}% kazan)",
                f"  En kotu sembol     : {worst['symbol']} ({tl(worst['total_pnl'])}, {worst['win_rate']:.0f}% kazan)",
            ]
    else:
        lines += ["  (henuz kapanan trade yok)"]
    lines.append("")

    lines += ["11. MAKRO ANLIK TANIM", "-" * 40]
    macro = fetch_macro_snapshot()
    if macro:
        if "usdtry" in macro:
            m = macro["usdtry"]
            sign = "+" if m["change_pct"] >= 0 else ""
            lines.append(f"  USD/TRY  : {m['price']:.4f}  ({sign}{m['change_pct']:.2f}% gun degisimi)")
        if "brent" in macro:
            m = macro["brent"]
            sign = "+" if m["change_pct"] >= 0 else ""
            lines.append(f"  Brent    : ${m['price']:.2f}  ({sign}{m['change_pct']:.2f}% gun degisimi)")
    else:
        lines += ["  (yfinance yuklu degil veya veri alinamadi)"]
    lines.append("")

    lines += [sep, "Rapor sonu.", sep]
    return "\n".join(lines)

# ── Entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    days = 30
    if "--days" in sys.argv:
        idx = sys.argv.index("--days")
        try:
            days = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass

    conn = sqlite3.connect(DB_PATH)
    eq_df, trades_df, open_df, all_trades, events_df = load_data(conn, days)

    report = build_report_v2(eq_df, trades_df, open_df, all_trades, events_df, conn, days)
    print(report)

    # Raporu dosyaya kaydet
    today = date.today().isoformat()
    rpt_path = RESULTS / f"daily_report_{today}.txt"
    rpt_path.write_text(report, encoding="utf-8")
    print(f"\n  Rapor kaydedildi: {rpt_path}")

    # Grafik
    if not eq_df.empty and len(eq_df) >= 2:
        print("  Grafik olusturuluyor...", end=" ", flush=True)
        chart = plot_equity(eq_df)
        if chart:
            print(f"OK -> {chart}")
        else:
            print("atlandi (yetersiz veri)")
    else:
        print("  Not: Grafik icin once paper_trade.py calistir")

    # Telegram: gunluk ozet
    if HAS_TG:
        print("  Telegram ozeti gonderiliyor...", end=" ", flush=True)
        ok = _tg_daily_summary()
        print("OK" if ok else "HATA")

    conn.close()
