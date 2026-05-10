@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM ============================================================
REM  BIST ALGO - TASK SCHEDULER KURULUM SCRIPTI
REM  Yonetici (Admin) olarak calistirin!
REM  Kullanim: setup_scheduler.bat
REM ============================================================

SET "DIR=C:\Users\Kayahan\Desktop\CS\Trade"

echo ============================================================
echo  BIST Algo - Task Scheduler Kurulumu
echo ============================================================
echo.
echo Gorev klasoru: BIST\
echo Calisma dizini: %DIR%
echo.

REM ── 10:00  Sabah taramasi (fetch + indicators + morning scan) ─
echo [1/18] 10:00 - Sabah taramasi...
schtasks /create /f ^
  /tn "BIST\SabahTarama_1000" ^
  /tr "\"%DIR%\run_morning.bat\"" ^
  /sc daily /st 10:00 ^
  /ru "%USERNAME%" /rl HIGHEST
if errorlevel 1 (echo [HATA] SabahTarama) else (echo [OK] SabahTarama 10:00)

REM ── 10:30 - 18:00  Intraday scan (16 gorev, 30 dk araliklarla) ─
SET IDX=2
FOR %%T IN (
  10:30 11:00 11:30 12:00 12:30
  13:00 13:30 14:00 14:30 15:00
  15:30 16:00 16:30 17:00 17:30
  18:00
) DO (
  SET "SAFE=%%T"
  SET "SAFE=!SAFE::=!"
  echo [!IDX!/18] %%T - Intraday scan...
  schtasks /create /f ^
    /tn "BIST\IntradayScan_!SAFE!" ^
    /tr "\"%DIR%\run_intraday.bat\"" ^
    /sc daily /st %%T ^
    /ru "%USERNAME%" /rl HIGHEST
  if errorlevel 1 (echo [HATA] Intraday %%T) else (echo [OK] Intraday %%T)
  SET /A IDX+=1
)

REM ── 18:30  Gun sonu journal ──────────────────────────────────
echo [18/18] 18:30 - Gun sonu journal...
schtasks /create /f ^
  /tn "BIST\GunSonu_1830" ^
  /tr "\"%DIR%\run_journal.bat\"" ^
  /sc daily /st 18:30 ^
  /ru "%USERNAME%" /rl HIGHEST
if errorlevel 1 (echo [HATA] GunSonu) else (echo [OK] GunSonu 18:30)

echo.
echo ============================================================
echo  Kurulum tamamlandi!
echo ============================================================
echo.
echo Gorevleri kontrol et:
echo   schtasks /query /fo TABLE /tn "BIST\*"
echo.
echo Tek gorev detayi:
echo   schtasks /query /tn "BIST\SabahTarama_1000" /fo LIST /v
echo.
echo Tum BIST gorevlerini sil:
echo   schtasks /delete /f /tn "BIST\SabahTarama_1000"
echo   schtasks /delete /f /tn "BIST\GunSonu_1830"
echo   FOR %%T IN (1030 1100 1130 1200 1230 1300 1330 1400 1430 1500 1530 1600 1630 1700 1730 1800) DO schtasks /delete /f /tn "BIST\IntradayScan_%%T"
echo.
pause
