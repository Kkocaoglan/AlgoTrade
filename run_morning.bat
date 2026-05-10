@echo off
chcp 65001 >nul
cd /d C:\Users\Kayahan\Desktop\CS\Trade

echo ============================================================
echo  BIST SABAH TARAMASI  %DATE% %TIME%
echo ============================================================

echo [1/5] Veri guncelleniyor...
py -3.12 fetch_data.py
if errorlevel 1 echo [UYARI] fetch_data.py hata verdi, devam ediliyor.

echo.
echo [2/5] Indiktorler...
py -3.12 indicators.py
if errorlevel 1 echo [UYARI] indicators.py hata verdi, devam ediliyor.

echo.
echo [3/5] Pozisyon yonetimi (stop/hedef kontrol)...
py -3.12 paper_trade.py --run
if errorlevel 1 echo [UYARI] paper_trade --run hata verdi, devam ediliyor.

echo.
echo [4/5] Sabah sinyal taramasi (Telegram)...
py -3.12 intraday_scan.py --morning
if errorlevel 1 echo [UYARI] intraday_scan --morning hata verdi.

echo.
echo [5/5] Gunluk rapor...
py -3.12 daily_report.py
if errorlevel 1 echo [UYARI] daily_report.py hata verdi.

echo.
echo Sabah taramasi tamamlandi: %TIME%
