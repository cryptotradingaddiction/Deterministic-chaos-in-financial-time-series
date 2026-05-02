@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM CORRELATION ENTROPY PIPELINE (K2, all files)
REM Computes K2 from d2.exe's .h2 output (h2(m,r) = ln C_m(r) - ln C_{m+1}(r)).
REM One run per coin:
REM   run2 = per-symbol (TAU_D2_<sym>, W_D2_<sym>) from _per_coin_settings.bat
REM ============================================================================

REM ------------------------------ USER CONFIG ---------------------------------
set TEST_MODE=false
if defined DCH_TEST_MODE set TEST_MODE=%DCH_TEST_MODE%
set DATA_DIR=C:\DCh\data
set RESULTS_DIR=%DATA_DIR%\results
set TISEAN=C:\DCh\Tisean_3.0.0\bin
set GNUPLOT_EXE=C:\Program Files\gnuplot\bin\gnuplot.exe
set PYTHON_EXE=py
set PYTHON_ARGS=-3
set PRINT_RESULTS=C:\DCh\print_results.py
set FILES=BTCUSD_BITSTAMP_1h_complete_logreturns.dat ETHUSD_BITSTAMP_1h_complete_logreturns.dat LTCUSD_BITSTAMP_1h_complete_logreturns.dat XRPUSD_BITSTAMP_1h_complete_logreturns.dat LINKUSD_BITSTAMP_1h_complete_logreturns.dat DOGEUSD_BITSTAMP_1h_complete_logreturns.dat ADAUSD_BITSTAMP_1h_complete_logreturns.dat

REM Per-coin (tau, W) overrides come from the shared settings file.
call "%~dp0_per_coin_settings.bat"

set EMBED=1,3
REM ----------------------------------------------------------------------------

cd /d "%DATA_DIR%" || (echo ERROR: Cannot enter %DATA_DIR% & exit /b 1)

if /i "%TEST_MODE%"=="true" (
    set OUT_ROOT=%RESULTS_DIR%\correlation_entropy_test_2000
    set TMP_ROOT=%DATA_DIR%\results_test_2000
    set TEST_SUFFIX=_test2000
    echo [INFO] TEST MODE - first 2000 lines per file
) else (
    set OUT_ROOT=%RESULTS_DIR%\correlation_entropy_full
    set TMP_ROOT=%DATA_DIR%\results_full
    set TEST_SUFFIX=
    echo [INFO] FULL MODE - using complete files
)

echo [INFO] Output root : %OUT_ROOT%
echo [INFO] m setting   : %EMBED% (components,max_embed -> using m=3 block)
echo [INFO] Per-coin run:
echo [INFO]   run2 = per-symbol (TAU_D2_^<sym^>, W_D2_^<sym^>)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%TMP_ROOT%" mkdir "%TMP_ROOT%"

if exist "%GNUPLOT_EXE%" (
    set HAS_GNUPLOT=true
) else (
    set HAS_GNUPLOT=false
    echo [WARN] gnuplot not found at "%GNUPLOT_EXE%". Plotting will be skipped.
)

set "AGG_FILE=%OUT_ROOT%\_correlation_entropy_summary.txt"
> "%AGG_FILE%" echo symbol,run,tau,W,h2_file

for %%F in (%FILES%) do (
    for /f "tokens=1 delims=_" %%A in ("%%F") do set BASE=%%A
    echo(
    echo ============================================================================
    echo Processing !BASE!
    echo ============================================================================

    set "FULL_DATA=%%F"
    set "CUT_DATA=!FULL_DATA:_logreturns.dat=_logreturns_cut.dat!"
    if exist "!CUT_DATA!" (
        set "FULL_DATA=!CUT_DATA!"
        echo   [INFO] Using liquidity-cut data: !FULL_DATA!
    ) else (
        echo   [INFO] Using raw data: !FULL_DATA!
    )

    if /i "%TEST_MODE%"=="true" (
        set "DATA_FILE=%TMP_ROOT%\tmp_!BASE!%TEST_SUFFIX%.dat"
        powershell -NoProfile -Command "Get-Content -Path '!FULL_DATA!' -TotalCount 2000 | Set-Content -Path '!DATA_FILE!' -Encoding ascii"
    ) else (
        REM Use absolute path so subroutines can pushd into OUT_DIR safely.
        set "DATA_FILE=%DATA_DIR%\!FULL_DATA!"
    )

    if not exist "!DATA_FILE!" (
        echo   [ERROR] Input file missing for !BASE!: !DATA_FILE!
        exit /b 1
    )

    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!DATA_FILE!"

    REM ---- Resolve per-symbol tau/W for run2 ----
    call set "COIN_TAU=%%TAU_D2_!BASE!%%"
    call set "COIN_W=%%W_D2_!BASE!%%"
    if "!COIN_TAU!"=="" set "COIN_TAU=3"
    if "!COIN_W!"==""   set "COIN_W=0"

    call :RUN_K2 "!BASE!" "!DATA_FILE!" "run2_tau!COIN_TAU!_W!COIN_W!" !COIN_TAU! !COIN_W!
    if errorlevel 1 exit /b 1

    set "RUN2_DIR=%OUT_ROOT%\!BASE!_run2_tau!COIN_TAU!_W!COIN_W!"
    set "HYP_DIR=!RUN2_DIR!\hypothesis_k2"
    if not exist "!HYP_DIR!" mkdir "!HYP_DIR!"
    echo   [Hypothesis] K2-only surrogate test ^(tau=!COIN_TAU!, W=!COIN_W!^)
    "%PYTHON_EXE%" %PYTHON_ARGS% "C:\DCh\hypothesis.py" --input "!DATA_FILE!" --base "!BASE!" --delay !COIN_TAU! --theiler !COIN_W! --output_dir "!HYP_DIR!" --test_mode "%TEST_MODE%" --metrics_list "K2"
    if errorlevel 1 exit /b 1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot "!HYP_DIR!\!BASE!_surrogate_summary.txt"
)

if /i "%TEST_MODE%"=="true" (
    del /q "%TMP_ROOT%\tmp_*_test2000.dat" >nul 2>&1
)

echo(
echo ============================================================================
echo Correlation-entropy run completed.
echo Aggregate index: %AGG_FILE%
echo Results root   : %OUT_ROOT%
echo [INFO] Aggregating K2-only surrogate summaries...
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot_aggregate "%OUT_ROOT%"
echo [INFO] Aggregate hypothesis summary: %OUT_ROOT%\_hypothesis_aggregate_summary.txt
echo ============================================================================
exit /b 0


REM ============================================================================
REM :RUN_K2 <BASE> <DATA_FILE> <RUN_ID> <TAU> <W>
REM ============================================================================
:RUN_K2
set "BASE=%~1"
set "DATA_FILE=%~2"
set "RUN_ID=%~3"
set "TAU_DELAY=%~4"
set "THEILER_W=%~5"
set "OUT_DIR=%OUT_ROOT%\!BASE!_!RUN_ID!"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

echo(
echo   --------------------------------------------------
echo   Running K2 analysis: !RUN_ID! (tau=!TAU_DELAY!, W=!THEILER_W!)
echo   Data file : !DATA_FILE!
echo   Output dir: !OUT_DIR!
echo   --------------------------------------------------

echo   [1/2] d2: correlation sums + K2 base curves...
"%TISEAN%\d2.exe" -d!TAU_DELAY! -M%EMBED% -t!THEILER_W! -o "!OUT_DIR!\!BASE!" "!DATA_FILE!"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" h2 "!OUT_DIR!\!BASE!.h2"

>> "%AGG_FILE%" echo !BASE!,!RUN_ID!,!TAU_DELAY!,!THEILER_W!,!OUT_DIR!\!BASE!.h2

echo   [2/2] plot: K2 entropy curves vs ln r for all m...
if /i "%HAS_GNUPLOT%"=="true" (
    "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!OUT_DIR!\!BASE!_K2_all_m.png'; set logscale x; set grid; set xlabel 'ln r'; set ylabel 'K2 entropy (per delay step)'; set title '!BASE! Correlation Entropy K2 (tau=!TAU_DELAY!, W=!THEILER_W!, m=3)%TEST_SUFFIX%'; set key left top font 'Arial,7' vertical maxrows 30; plot '!OUT_DIR!\!BASE!.h2' index 2 using 1:2 with lines lw 1 title 'm=3'" > "!OUT_DIR!\gnuplot.log" 2>&1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_K2_all_m.png"
)
echo(
exit /b 0
