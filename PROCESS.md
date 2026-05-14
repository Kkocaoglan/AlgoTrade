# PROCESS.md

Bu dosya projenin calistirma runbook'udur. Yeni pipeline, yeni risk kontrolu
veya yeni rapor eklendikce burada guncellenmelidir.

## Aktif Kullanim Durumu

Su an aktif olarak kullandigin crypto tarafinda ana dosya:

```bash
python3.12 crypto_trader.py
```

Tek seferlik guvenli kontrol:

```bash
python3.12 crypto_trader.py --once
```

Dashboard:

```bash
python3.12 crypto_status.py
```

Crypto icin yardimci analiz:

```bash
python3.12 crypto_gate_risk_analysis.py --risk-pcts 0.01,0.02,0.03,0.04
```

Not: Crypto runtime icin aktif kaynak `crypto_trader.py`, ortak config kaynagi
`crypto_config.py`.

## BIST Tek Komutluk Gunluk Akis

BIST icin ana komut:

```bash
python3.12 run_bist_process.py
```

Bu komut crypto gibi terminalde surekli takip eder. Yani BIST icin varsayilan
mod artik devam eden loop'tur.

Bu sira ile calisir:

1. `bist_data_quality.py`
2. `fetch_data.py`
3. `indicators.py`
4. `bist_data_quality.py`
5. `loop_trader.py`

Tek seferlik guvenli kontrol ve simulator calistirmak istersen:

```bash
python3.12 run_bist_process.py --once
```

`--once` modunda 5. adim `loop_trader.py --once` olur ve sonra
`bist_live_wf_sim.py --thresholds 0.60,0.65,0.70,0.73` calisir.

Sadece veri ve indikator guncellemek:

```bash
python3.12 run_bist_process.py --skip-loop --skip-sim
```

Modeli de yeniden egitmek:

```bash
python3.12 run_bist_process.py --train
```

Model egitimiyle beraber threshold dosyasini da guncellemek:

```bash
python3.12 run_bist_process.py --train --optimize-threshold
```

Loop'u calistirmadan sadece analiz:

```bash
python3.12 run_bist_process.py --skip-loop
```

## BIST Modul Rolleri

- `fetch_data.py`: BIST OHLCV verisini gunceller.
- `indicators.py`: teknik indikatorleri ve macro veriyi hesaplar.
- `bist_data_quality.py`: corporate-action / split / bedelsiz gibi fiyat baz kirilmalarini denetler.
- `loop_trader.py`: aktif BIST paper/live-loop karar motoru.
- `bist_live_wf_sim.py`: live-loop kurallarina yakin walk-forward trade simulator.
- `ml_train.py`: BIST model egitimi.
- `optimize_threshold.py`: model precision tabanli threshold yardimcisi. Nihai karar icin simulator metriğiyle beraber okunmali.

## Onemli Notlar

- Mac ortaminda `py -3.12` yoksa `python3.12` kullan.
- Windows tarafinda `py -3.12 run_bist_process.py` kullanilabilir.
- `*.db`, `*.pkl`, `*.csv`, `*.png` dosyalarini elle acip inceleme. Gerekirse script veya kisa inspection komutu kullan.
- BIST icin aktif runtime kaynak `loop_trader.py`, ortak config kaynagi `bist_config.py`.
- Crypto icin aktif runtime kaynak `crypto_trader.py`, ortak config kaynagi `crypto_config.py`.
