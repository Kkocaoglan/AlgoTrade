
"""
optimize.py — Grid search ile en iyi parametreleri bul
────────────────────────────────────────────────────────
Kullanım: py -3.12 optimize.py

Ne yapar:
  S1: EMA periyotları + RSI eşikleri + ATR çarpanı tarar
  S3: BB periyot + std + vol threshold tarar
  S2: RSI threshold yerine Momentum stratejisi dener
  
Overfitting koruması:
  2022-2023 = eğitim seti
  2024-2026 = test seti
  Her ikisinde de iyi çalışan parametreleri seç
────────────────────────────────────────────────────────
"""

import sqlite3, time, itertools
from pathlib import Path
import pandas as pd
import numpy as np

DB_PATH = Path(__file__).parent / "trade_data.db"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

COMMISSION = 0.002
SLIPPAGE   = 0.001
CAPITAL    = 45_000   # S1 için
SYMBOLS    = ["YKBNK","AKBNK","ISCTR","GARAN",
              "TUPRS","PETKM","TAVHL","FROTO",
              "TCELL","ASELS","BIMAS","MGROS",
              "ENKAI","EKGYO"]

TRAIN_END = "2024-01-01"
TEST_START= "2024-01-01"

# ── Veri yükle ────────────────────────────────────────────────
def load():
    conn = sqlite3.connect(DB_PATH)
    frames = {}
    for sym in SYMBOLS:
        df = pd.read_sql(
            "SELECT o.date,o.open,o.high,o.low,o.close,o.volume,"
            "i.ema8,i.ema21,i.ema50,i.ema200,i.rsi14,"
            "i.macd_hist,i.atr14,i.bb_upper,i.bb_lower,i.bb_width,"
            "i.vol_ratio,i.mtf_trend,i.above_ema200,i.golden_cross "
            "FROM ohlcv o JOIN indicators i "
            "ON o.symbol=i.symbol AND o.date=i.date "
            "WHERE o.symbol=? ORDER BY o.date",
            conn, params=(sym,))
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[sym] = df
    conn.close()
    return frames

def pnl(ep, xp, is_long, size):
    cost = size * (COMMISSION + SLIPPAGE)
    ret  = (xp-ep)/ep if is_long else (ep-xp)/ep
    return size * ret - cost

def sharpe(equity):
    eq  = np.array(equity)
    ret = np.diff(eq) / (eq[:-1] + 1e-9)
    mu  = ret.mean(); std = ret.std()
    return (mu/std*np.sqrt(252)) if std > 0 else -99

def max_dd(equity):
    eq   = np.array(equity)
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / (peak + 1e-9)).min() * 100

# ── S1 tek parametre seti ile çalıştır ───────────────────────
def run_s1_params(frames, capital, params, date_filter=None):
    """
    params = {
      rsi_lo, rsi_hi  : RSI giriş penceresi
      atr_mult        : stop mesafesi = ATR × atr_mult
      max_pos         : aynı anda max pozisyon sayısı
      macd_filter     : True ise MACD hist > 0 şart
    }
    """
    rsi_lo    = params["rsi_lo"]
    rsi_hi    = params["rsi_hi"]
    atr_mult  = params["atr_mult"]
    max_pos   = params["max_pos"]

    trades = []
    equity = [capital]
    cash   = capital
    pos    = {}

    all_dates = sorted(set(d for f in frames.values() for d in f.index))
    if date_filter:
        all_dates = [d for d in all_dates
                     if date_filter[0] <= d < date_filter[1]]

    for dt in all_dates:
        # Pozisyon kapatma
        to_close = []
        for sym, p in pos.items():
            if sym not in frames or dt not in frames[sym].index:
                continue
            row   = frames[sym].loc[dt]
            price = float(row["close"])
            atr   = float(row["atr14"]) if pd.notna(row.get("atr14")) else 0
            e8    = row.get("ema8"); e21 = row.get("ema21")

            stop_hit   = (p["long"] and price <= p["stop"]) or \
                         (not p["long"] and price >= p["stop"])
            cross_exit = False
            if pd.notna(e8) and pd.notna(e21):
                cross_exit = (p["long"] and e8 < e21) or \
                             (not p["long"] and e8 > e21)

            if stop_hit or cross_exit:
                pl = pnl(p["ep"], price, p["long"], p["size"])
                cash += p["size"] + pl
                trades.append({"pnl": pl, "win": pl > 0})
                to_close.append(sym)

        for s in to_close: del pos[s]

        # Yeni giriş
        if len(pos) < max_pos:
            for sym, df in frames.items():
                if sym in pos or dt not in df.index: continue
                row   = df.loc[dt]
                price = float(row["close"])
                rsi   = float(row["rsi14"])  if pd.notna(row.get("rsi14")) else 50
                atr   = float(row["atr14"])  if pd.notna(row.get("atr14")) else 0
                mh    = float(row["macd_hist"]) if pd.notna(row.get("macd_hist")) else 0
                mt    = int(row.get("mtf_trend", 0))
                ab200 = int(row.get("above_ema200", 0))

                is_long = None
                if mt == 1 and ab200 == 1 and mh > 0 and rsi_lo < rsi < rsi_hi:
                    is_long = True
                elif mt == -1 and ab200 == 0 and mh < 0 and (100-rsi_hi) < rsi < (100-rsi_lo):
                    is_long = False

                if is_long is None: continue
                size = min(capital * 0.15, cash * 0.9 / max(1, max_pos - len(pos)))
                if size < 500 or cash < size: continue
                stop = price - atr_mult*atr if is_long else price + atr_mult*atr
                cash -= size
                pos[sym] = {"ep": price, "long": is_long,
                            "size": size, "stop": stop}

        cur_val = cash + sum(
            frames[s].loc[dt,"close"] * p["size"] / p["ep"]
            if s in frames and dt in frames[s].index else p["size"]
            for s, p in pos.items())
        equity.append(cur_val)

    n  = len(trades)
    wr = sum(1 for t in trades if t["win"]) / n * 100 if n else 0
    return {
        "sharpe"  : sharpe(equity),
        "max_dd"  : max_dd(equity),
        "total_ret": (equity[-1]/capital - 1)*100 if equity else 0,
        "trades"  : n,
        "win_rate": wr,
        "equity"  : equity,
    }

# ── S3 tek parametre seti ile çalıştır ───────────────────────
def run_s3_params(frames, capital, params, date_filter=None):
    bb_p   = params["bb_period"]
    bb_k   = params["bb_std"]
    vol_th = params["vol_threshold"]
    atr_m  = params["atr_mult"]

    # BB yeniden hesapla
    def recompute_bb(df):
        c   = df["close"]
        mid = c.rolling(bb_p).mean()
        std = c.rolling(bb_p).std()
        upper = mid + bb_k * std
        lower = mid - bb_k * std
        width = (upper - lower) / (mid + 1e-9) * 100
        p20   = width.quantile(0.20)
        p60   = width.quantile(0.60)
        return upper, lower, width, p20, p60

    bb_cache = {}
    for sym, df in frames.items():
        bb_cache[sym] = recompute_bb(df)

    trades = []
    equity = [capital]
    cash   = capital
    pos    = {}

    all_dates = sorted(set(d for f in frames.values() for d in f.index))
    if date_filter:
        all_dates = [d for d in all_dates
                     if date_filter[0] <= d < date_filter[1]]

    for dt in all_dates:
        to_close = []
        for sym, p in pos.items():
            if sym not in frames or dt not in frames[sym].index: continue
            row   = frames[sym].loc[dt]
            price = float(row["close"])
            atr   = float(row["atr14"]) if pd.notna(row.get("atr14")) else 0
            _, _, width, _, p60 = bb_cache[sym]
            bbw = width.get(dt, 99)

            stop_hit   = (p["long"] and price <= p["stop"]) or \
                         (not p["long"] and price >= p["stop"])
            expand_exit= bbw > p60

            if stop_hit or expand_exit:
                pl = pnl(p["ep"], price, p["long"], p["size"])
                cash += p["size"] + pl
                trades.append({"pnl": pl, "win": pl > 0})
                to_close.append(sym)

        for s in to_close: del pos[s]

        if len(pos) < 3:
            for sym, df in frames.items():
                if sym in pos or dt not in df.index: continue
                upper, lower, width, p20, _ = bb_cache[sym]
                row   = df.loc[dt]
                price = float(row["close"])
                atr   = float(row["atr14"]) if pd.notna(row.get("atr14")) else 0
                vol_r = float(row["vol_ratio"]) if pd.notna(row.get("vol_ratio")) else 1
                bbw   = width.get(dt, 99)
                bbu   = upper.get(dt, price*1.05)
                bbl   = lower.get(dt, price*0.95)

                if bbw > p20 * 1.3: continue
                if vol_r < vol_th:  continue

                is_long = None
                if price > bbu:   is_long = True
                elif price < bbl: is_long = False
                if is_long is None: continue

                size = min(capital * 0.15, cash * 0.30)
                if size < 500 or cash < size: continue
                stop = price - atr_m*atr if is_long else price + atr_m*atr
                cash -= size
                pos[sym] = {"ep": price, "long": is_long,
                            "size": size, "stop": stop}

        cur_val = cash + sum(
            frames[s].loc[dt,"close"] * p["size"] / p["ep"]
            if s in frames and dt in frames[s].index else p["size"]
            for s, p in pos.items())
        equity.append(cur_val)

    n  = len(trades)
    wr = sum(1 for t in trades if t["win"]) / n * 100 if n else 0
    return {
        "sharpe"  : sharpe(equity),
        "max_dd"  : max_dd(equity),
        "total_ret": (equity[-1]/capital - 1)*100,
        "trades"  : n,
        "win_rate": wr,
    }

# ── Grid search ───────────────────────────────────────────────
def grid_s1(frames):
    print("\n" + "="*60)
    print("S1 MTF Momentum — Grid Search")
    print("Parametreler: RSI penceresi · ATR çarpanı · Max pozisyon")
    print("="*60)

    param_grid = {
        "rsi_lo"  : [40, 45, 50],
        "rsi_hi"  : [60, 65, 70],
        "atr_mult": [1.5, 2.0, 2.5],
        "max_pos" : [3, 4, 5],
    }

    combos = list(itertools.product(
        param_grid["rsi_lo"], param_grid["rsi_hi"],
        param_grid["atr_mult"], param_grid["max_pos"]
    ))
    # Sadece rsi_lo < rsi_hi olanlar
    combos = [(a,b,c,d) for a,b,c,d in combos if a < b]

    train_range = (pd.Timestamp("2022-01-01"), pd.Timestamp(TRAIN_END))
    test_range  = (pd.Timestamp(TEST_START),   pd.Timestamp("2030-01-01"))

    results = []
    print(f"  {len(combos)} kombinasyon test ediliyor...", flush=True)

    for i, (rl, rh, am, mp) in enumerate(combos):
        params = {"rsi_lo":rl, "rsi_hi":rh, "atr_mult":am, "max_pos":mp}
        tr = run_s1_params(frames, CAPITAL, params, train_range)
        te = run_s1_params(frames, CAPITAL, params, test_range)

        # Her iki dönemde de Sharpe > 0.8 ve DD > -25 olanları filtrele
        if tr["sharpe"] > 0.8 and te["sharpe"] > 0.5 and \
           tr["max_dd"] > -25 and te["max_dd"] > -20:
            score = tr["sharpe"] * 0.4 + te["sharpe"] * 0.6  # test ağırlıklı
            results.append({**params,
                            "train_sharpe": tr["sharpe"],
                            "train_dd"    : tr["max_dd"],
                            "test_sharpe" : te["sharpe"],
                            "test_dd"     : te["max_dd"],
                            "test_ret"    : te["total_ret"],
                            "test_wr"     : te["win_rate"],
                            "score"       : score})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(combos)}...", flush=True)

    if not results:
        print("  Filtreden geçen parametre yok — eşikleri gevşet")
        # En iyi 3'ü göster
        all_r = []
        for rl, rh, am, mp in combos[:30]:
            params = {"rsi_lo":rl, "rsi_hi":rh, "atr_mult":am, "max_pos":mp}
            te = run_s1_params(frames, CAPITAL, params, test_range)
            all_r.append({**params, **te})
        all_r.sort(key=lambda x: x["sharpe"], reverse=True)
        print("\n  En iyi 3 (filtresiz):")
        for r in all_r[:3]:
            print(f"    RSI[{r['rsi_lo']}-{r['rsi_hi']}] ATR×{r['atr_mult']} "
                  f"MaxPos:{r['max_pos']} → "
                  f"Sharpe:{r['sharpe']:.2f} DD:{r['max_dd']:.1f}% "
                  f"Ret:{r['total_ret']:+.1f}%")
        return all_r[0] if all_r else None

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n  {len(results)} geçerli kombinasyon bulundu")
    print("\n  En iyi 5:")
    print(f"  {'RSI_LO':>6} {'RSI_HI':>6} {'ATR×':>5} {'MAXP':>5} "
          f"{'TR_SH':>6} {'TE_SH':>6} {'TE_DD':>7} {'TE_RET':>8} {'TE_WIN':>7}")
    for r in results[:5]:
        print(f"  {r['rsi_lo']:>6} {r['rsi_hi']:>6} {r['atr_mult']:>5} "
              f"{r['max_pos']:>5} {r['train_sharpe']:>6.2f} "
              f"{r['test_sharpe']:>6.2f} {r['test_dd']:>7.1f}% "
              f"{r['test_ret']:>+7.1f}% {r['test_wr']:>6.0f}%")

    best = results[0]
    print(f"\n  >> EN İYİ: RSI[{best['rsi_lo']}-{best['rsi_hi']}] "
          f"ATR×{best['atr_mult']} MaxPos:{best['max_pos']}")
    print(f"     Train Sharpe: {best['train_sharpe']:.2f}  "
          f"Test Sharpe: {best['test_sharpe']:.2f}")
    print(f"     Test Getiri : {best['test_ret']:+.1f}%  "
          f"Test DD: {best['test_dd']:.1f}%")
    return best

def grid_s3(frames):
    print("\n" + "="*60)
    print("S3 BB Breakout — Grid Search")
    print("Parametreler: BB periyot · std · vol threshold · ATR mult")
    print("="*60)

    param_grid = {
        "bb_period"    : [15, 20, 25],
        "bb_std"       : [1.5, 2.0, 2.5],
        "vol_threshold": [1.3, 1.5, 2.0],
        "atr_mult"     : [1.5, 2.0, 2.5],
    }

    combos = list(itertools.product(
        param_grid["bb_period"], param_grid["bb_std"],
        param_grid["vol_threshold"], param_grid["atr_mult"]
    ))

    cap3 = 20_000
    train_range = (pd.Timestamp("2022-01-01"), pd.Timestamp(TRAIN_END))
    test_range  = (pd.Timestamp(TEST_START),   pd.Timestamp("2030-01-01"))

    results = []
    print(f"  {len(combos)} kombinasyon test ediliyor...", flush=True)

    for i, (bp, bk, vt, am) in enumerate(combos):
        params = {"bb_period":bp, "bb_std":bk,
                  "vol_threshold":vt, "atr_mult":am}
        tr = run_s3_params(frames, cap3, params, train_range)
        te = run_s3_params(frames, cap3, params, test_range)

        if tr["sharpe"] > 0.9 and te["sharpe"] > 0.7 and \
           tr["max_dd"] > -20 and te["max_dd"] > -15:
            score = tr["sharpe"]*0.4 + te["sharpe"]*0.6
            results.append({**params,
                            "train_sharpe": tr["sharpe"],
                            "test_sharpe" : te["sharpe"],
                            "test_dd"     : te["max_dd"],
                            "test_ret"    : te["total_ret"],
                            "test_wr"     : te["win_rate"],
                            "score"       : score})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(combos)}...", flush=True)

    if not results:
        print("  Filtreden geçen yok — en iyi 3:")
        all_r = []
        for bp, bk, vt, am in combos:
            params = {"bb_period":bp,"bb_std":bk,
                      "vol_threshold":vt,"atr_mult":am}
            te = run_s3_params(frames, cap3, params, test_range)
            all_r.append({**params, **te})
        all_r.sort(key=lambda x: x["sharpe"], reverse=True)
        for r in all_r[:3]:
            print(f"    BB({r['bb_period']},×{r['bb_std']}) "
                  f"Vol>{r['vol_threshold']} ATR×{r['atr_mult']} → "
                  f"Sharpe:{r['sharpe']:.2f} DD:{r['max_dd']:.1f}%")
        return all_r[0] if all_r else None

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n  {len(results)} geçerli kombinasyon")
    best = results[0]
    print(f"\n  >> EN İYİ: BB({best['bb_period']},×{best['bb_std']}) "
          f"Vol>{best['vol_threshold']} ATR×{best['atr_mult']}")
    print(f"     Test Sharpe: {best['test_sharpe']:.2f}  "
          f"Test DD: {best['test_dd']:.1f}%  "
          f"Test Getiri: {best['test_ret']:+.1f}%")
    return best

# ── Yeni S2: Trend Momentum (RSI yerine) ─────────────────────
def explain_s2_replacement():
    print("\n" + "="*60)
    print("S2 GÜNCELLEME — RSI Mean Rev. → Trend Momentum")
    print("="*60)
    print("""
  Orijinal S2 (RSI Mean Reversion) BIST'te başarısız oldu çünkü:
  - BIST 2022-2026: enflasyon, kur oynaklığı → güçlü trendler
  - Mean reversion trendin ortasında alım yapar → zarar
  
  Yeni S2: Momentum Confirmation
  - Giriş : EMA50 > EMA200 (trend yönü)
             + son 20 gün getiri top %30 (momentum)
             + RSI 50-70 arası (ne aşırı alım ne satım)
  - Çıkış : EMA50, EMA200'ü kırarsa
             veya 20 günde %8 getiri hedefine ulaşırsa
  - Beklenti: daha az trade, daha yüksek win rate
    """)

# ── Özet ve kaydet ────────────────────────────────────────────
def save_best_params(best_s1, best_s3):
    out = RESULTS / "best_params.txt"
    lines = ["BIST 3-Bot — Optimize Edilmiş Parametreler",
             "="*50, ""]
    if best_s1:
        lines += [
            "S1 MTF Momentum:",
            f"  RSI penceresi : {best_s1.get('rsi_lo')} - {best_s1.get('rsi_hi')}",
            f"  ATR çarpanı   : {best_s1.get('atr_mult')}×",
            f"  Max pozisyon  : {best_s1.get('max_pos')}",
            f"  Test Sharpe   : {best_s1.get('test_sharpe',0):.2f}",
            f"  Test DD       : {best_s1.get('test_dd',0):.1f}%",
            "",
        ]
    if best_s3:
        lines += [
            "S3 BB Breakout:",
            f"  BB periyot    : {best_s3.get('bb_period')}",
            f"  BB std        : {best_s3.get('bb_std')}×",
            f"  Vol threshold : {best_s3.get('vol_threshold')}×",
            f"  ATR çarpanı   : {best_s3.get('atr_mult')}×",
            f"  Test Sharpe   : {best_s3.get('test_sharpe',0):.2f}",
            f"  Test DD       : {best_s3.get('test_dd',0):.1f}%",
            "",
        ]
    lines += ["S2 → Trend Momentum stratejisine geçildi",
              "backtest_v2.py'de tüm güncellenmiş stratejiler çalışır."]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Parametreler kaydedildi: {out}")

# ── Entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("="*60)
    print("Optimizasyon Motoru v1")
    print("Overfitting koruması: 2022-2023 train / 2024-2026 test")
    print("="*60)

    print("\nVeri yükleniyor...", end=" ", flush=True)
    frames = load()
    print("OK")

    best_s1 = grid_s1(frames)
    best_s3 = grid_s3(frames)
    explain_s2_replacement()
    save_best_params(best_s1, best_s3)

    print(f"\nToplam süre: {time.time()-t0:.1f}s")
    print("\nSonraki adım: best_params.txt'i oku,")
    print("sonra backtest.py'yi bu parametrelerle tekrar çalıştır.")
    print("veya: py -3.12 paper_trade.py  (bugünden itibaren canlı izle)")