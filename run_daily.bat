@echo off
chcp 65001 >nul
echo ============================================================
echo  BIST Algo Trading Sistemi - Gunluk Calistirici
echo ============================================================
echo Tarih: %DATE%  Saat: %TIME%
echo.

cd /d C:\Users\Kayahan\Desktop\CS\Trade

echo [1/5] Veri guncelleniyor...
py -3.12 fetch_data.py
if errorlevel 1 (
    echo [HATA] fetch_data.py basarisiz! Devam ediliyor...
)

echo.
echo [2/5] Indikatörler hesaplaniyor...
py -3.12 indicators.py
if errorlevel 1 (
    echo [HATA] indicators.py basarisiz! Devam ediliyor...
)

echo.
echo [3/5] Paper trade gunluk kosu...
py -3.12 paper_trade.py --run
if errorlevel 1 (
    echo [HATA] paper_trade.py --run basarisiz!
    pause
    exit /b 1
)

echo.
echo [4/5] Gunluk rapor olusturuluyor...
py -3.12 daily_report.py
if errorlevel 1 (
    echo [HATA] daily_report.py basarisiz! Devam ediliyor...
)

echo.
echo [5/5] Haber filtresi ve izleme modu baslatiliyor...
echo      (Ctrl+C ile durdurun)
echo.

REM Haber filtresini arka planda baslat
start "BIST Haber Filtresi" /min py -3.12 news_filter.py

REM Intraday izlemeyi on planda baslat
py -3.12 paper_trade.py --watch

echo.
echo ============================================================
echo  Gunluk calistirici tamamlandi.
echo ============================================================
pause
