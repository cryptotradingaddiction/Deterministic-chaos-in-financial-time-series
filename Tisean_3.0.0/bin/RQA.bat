@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM RQA PIPELINE (recurrence + RQA metrics, all files)
REM 1) Runs TISEAN recurr for each symbol (+ recurrence plot)
REM 2) Runs Python rqa_values.py for RQA scalar metrics
REM One run per coin:
REM   run2 = per-symbol tau + percentile-based RQA radius
REM ============================================================================

REM ------------------------------ USER CONFIG ---------------------------------
set TEST_MODE=false
if defined DCH_TEST_MODE set TEST_MODE=%DCH_TEST_MODE%
set RUN_HYPOTHESIS=true
if defined DCH_RUN_HYPOTHESIS set RUN_HYPOTHESIS=%DCH_RUN_HYPOTHESIS%
set DATA_DIR=C:\DCh\data
set RESULTS_DIR=%DATA_DIR%\results
set TISEAN=C:\DCh\Tisean_3.0.0\bin
set GNUPLOT_EXE=C:\Program Files\gnuplot\bin\gnuplot.exe
set PYTHON_EXE=py
set PYTHON_ARGS=-3
set PY_RQA_SCRIPT=C:\DCh\rqa_values.py
set RQA_RADIUS_SCRIPT=C:\DCh\rqa_radius.py
set PRINT_RESULTS=C:\DCh\print_results.py
set FILES=BTCUSD_BITSTAMP_1h_complete_logreturns.dat ETHUSD_BITSTAMP_1h_complete_logreturns.dat LTCUSD_BITSTAMP_1h_complete_logreturns.dat XRPUSD_BITSTAMP_1h_complete_logreturns.dat LINKUSD_BITSTAMP_1h_complete_logreturns.dat DOGEUSD_BITSTAMP_1h_complete_logreturns.dat ADAUSD_BITSTAMP_1h_complete_logreturns.dat

REM Per-coin (tau, radius) overrides come from the shared settings file.
call "%~dp0_per_coin_settings.bat"

REM Fixed parameters for recurr. recurr expects -m<components>,<embed_dim>.
set COMPONENTS=1
set EMBED_DIM=3
REM ----------------------------------------------------------------------------

cd /d "%DATA_DIR%" || (echo ERROR: Cannot enter %DATA_DIR% & exit /b 1)

if /i "%TEST_MODE%"=="true" (
    set OUT_ROOT=%RESULTS_DIR%\rqa_test_2000
    set TMP_ROOT=%DATA_DIR%\results_test_2000
    set TEST_SUFFIX=_test2000
    echo [INFO] TEST MODE - first 2000 lines per file
) else (
    set OUT_ROOT=%RESULTS_DIR%\rqa_full
    set TMP_ROOT=%DATA_DIR%\results_full
    set TEST_SUFFIX=
    echo [INFO] FULL MODE - using complete files
)

echo [INFO] Output root : %OUT_ROOT%
echo [INFO] m setting   : components=%COMPONENTS%, embed_dim=%EMBED_DIM%
echo [INFO] Hypothesis  : %RUN_HYPOTHESIS%
echo [INFO] Per-coin run:
echo [INFO]   run2 = per-symbol tau + RQA radius from pairwise-distance percentile

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%TMP_ROOT%" mkdir "%TMP_ROOT%"

if exist "%GNUPLOT_EXE%" (
    set HAS_GNUPLOT=true
) else (
    set HAS_GNUPLOT=false
    echo [WARN] gnuplot not found at "%GNUPLOT_EXE%". Plotting will be skipped.
)

set "AGG_FILE=%OUT_ROOT%\_rqa_summary.txt"
> "%AGG_FILE%" echo symbol,run,tau,radius,rec_file

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
        echo   [ERROR] Required liquidity-cut data missing: !CUT_DATA!
        echo   [ERROR] Run C:\DCh\liquidity.py before this pipeline.
        exit /b 1
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

    REM ---- Resolve per-symbol tau / fallback radius, then compute effective percentile radius ----
    call set "COIN_TAU=%%TAU_RQA_!BASE!%%"
    call set "COIN_RAD=%%RAD_RQA_!BASE!%%"
    call set "COIN_W=%%W_D2_!BASE!%%"
    if "!COIN_TAU!"=="" set "COIN_TAU=3"
    if "!COIN_RAD!"=="" set "COIN_RAD=0.005"
    if "!COIN_W!"=="" set "COIN_W=0"

    set "RADIUS_TMP=%TMP_ROOT%\rqa_radius_!BASE!.tmp"
    "%PYTHON_EXE%" %PYTHON_ARGS% "%RQA_RADIUS_SCRIPT%" --input "!DATA_FILE!" --delay !COIN_TAU! --fallback !COIN_RAD! > "!RADIUS_TMP!"
    if errorlevel 1 (
        echo   [ERROR] Failed to compute percentile RQA radius for !BASE!.
        exit /b 1
    )
    set /p COIN_RAD_EFF=<"!RADIUS_TMP!"
    del /q "!RADIUS_TMP!" >nul 2>&1
    if "!COIN_RAD_EFF!"=="" set "COIN_RAD_EFF=!COIN_RAD!"
    echo   [RQA radius] effective r=!COIN_RAD_EFF! ^(4%% pairwise-distance percentile; fallback=!COIN_RAD!^)

    call :RUN_RQA "!BASE!" "!DATA_FILE!" "run2_tau!COIN_TAU!_r!COIN_RAD_EFF!" !COIN_TAU! !COIN_RAD_EFF!
    if errorlevel 1 exit /b 1

    if /i "%RUN_HYPOTHESIS%"=="true" (
        set "RUN2_DIR=%OUT_ROOT%\!BASE!_run2_tau!COIN_TAU!_r!COIN_RAD_EFF!"
        set "HYP_DIR=!RUN2_DIR!\hypothesis_rqa"
        if not exist "!HYP_DIR!" mkdir "!HYP_DIR!"
        echo   [Hypothesis] RQA scalar summary ^(no bootstrap TS; tau=!COIN_TAU!, r=!COIN_RAD_EFF!, W=!COIN_W!^)
        "%PYTHON_EXE%" %PYTHON_ARGS% "C:\DCh\hypothesis.py" --input "!DATA_FILE!" --base "!BASE!" --delay !COIN_TAU! --theiler !COIN_W! --rqa_radius !COIN_RAD_EFF! --output_dir "!HYP_DIR!" --test_mode "%TEST_MODE%" --metrics_list "RR,DET,LAM,MAXLINE,ENTR,TT,TREND"
        if errorlevel 1 exit /b 1
        "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot "!HYP_DIR!\!BASE!_surrogate_summary.txt"
    ) else (
        echo   [Hypothesis] skipped ^(DCH_RUN_HYPOTHESIS=%RUN_HYPOTHESIS%^)
    )
)

echo(
echo ============================================================================
echo Running Python RQA metrics script: %PY_RQA_SCRIPT%
echo ============================================================================
set "DCH_TEST_MODE=%TEST_MODE%"
"%PYTHON_EXE%" %PYTHON_ARGS% "%PY_RQA_SCRIPT%"
if errorlevel 1 (
    echo [ERROR] Python RQA metrics script failed.
    exit /b 1
)

echo(
echo ============================================================================
echo Per-coin RQA metric summaries
echo ============================================================================
for %%F in (%FILES%) do (
    for /f "tokens=1 delims=_" %%A in ("%%F") do set BASE=%%A
    call set "COIN_TAU=%%TAU_RQA_!BASE!%%"
    call set "COIN_RAD=%%RAD_RQA_!BASE!%%"
    if "!COIN_TAU!"=="" set "COIN_TAU=3"
    if "!COIN_RAD!"=="" set "COIN_RAD=0.005"

    set "FULL_DATA=%%F"
    set "CUT_DATA=!FULL_DATA:_logreturns.dat=_logreturns_cut.dat!"
    if exist "%DATA_DIR%\!CUT_DATA!" (
        set "FULL_DATA=!CUT_DATA!"
    ) else (
        echo [ERROR] Required liquidity-cut data missing: !CUT_DATA!
        echo [ERROR] Run C:\DCh\liquidity.py before this pipeline.
        exit /b 1
    )
    if /i "%TEST_MODE%"=="true" (
        set "DATA_FILE=%TMP_ROOT%\tmp_!BASE!%TEST_SUFFIX%.dat"
    ) else (
        set "DATA_FILE=%DATA_DIR%\!FULL_DATA!"
    )

    set "RADIUS_TMP=%TMP_ROOT%\rqa_radius_!BASE!.tmp"
    "%PYTHON_EXE%" %PYTHON_ARGS% "%RQA_RADIUS_SCRIPT%" --input "!DATA_FILE!" --delay !COIN_TAU! --fallback !COIN_RAD! > "!RADIUS_TMP!"
    if errorlevel 1 exit /b 1
    set /p COIN_RAD_EFF=<"!RADIUS_TMP!"
    del /q "!RADIUS_TMP!" >nul 2>&1
    if "!COIN_RAD_EFF!"=="" set "COIN_RAD_EFF=!COIN_RAD!"

    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" rqa "%OUT_ROOT%\!BASE!_run2_tau!COIN_TAU!_r!COIN_RAD_EFF!\!BASE!_rqa_metrics.txt"
)

if /i "%TEST_MODE%"=="true" (
    del /q "%TMP_ROOT%\tmp_*_test2000.dat" >nul 2>&1
)

echo(
echo ============================================================================
echo RQA run completed.
echo TISEAN recurrence outputs: %OUT_ROOT%
echo Python RQA metrics       : %OUT_ROOT%\*_run2_*\*_rqa_metrics.txt
echo Aggregate index          : %AGG_FILE%
if /i "%RUN_HYPOTHESIS%"=="true" (
    echo [INFO] Aggregating RQA scalar summaries...
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot_aggregate "%OUT_ROOT%"
    echo [INFO] Aggregate hypothesis summary: %OUT_ROOT%\_hypothesis_aggregate_summary.txt
) else (
    echo [INFO] Hypothesis aggregation skipped.
)
echo ============================================================================
exit /b 0


REM ============================================================================
REM :RUN_RQA <BASE> <DATA_FILE> <RUN_ID> <TAU> <RADIUS>
REM ============================================================================
:RUN_RQA
set "BASE=%~1"
set "DATA_FILE=%~2"
set "RUN_ID=%~3"
set "TAU_DELAY=%~4"
set "RADIUS=%~5"
set "OUT_DIR=%OUT_ROOT%\!BASE!_!RUN_ID!"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

echo(
echo   --------------------------------------------------
echo   Running RQA analysis: !RUN_ID! (tau=!TAU_DELAY!, r=!RADIUS!)
echo   Data file : !DATA_FILE!
echo   Output dir: !OUT_DIR!
echo   --------------------------------------------------

echo   [1/2] recurr: recurrence matrix for plot diagnostics...
"%TISEAN%\recurr.exe" -m%COMPONENTS%,%EMBED_DIM% -d!TAU_DELAY! -r!RADIUS! -%%2 -o "!OUT_DIR!\!BASE!_recurr.txt" "!DATA_FILE!"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" rec "!OUT_DIR!\!BASE!_recurr.txt"

>> "%AGG_FILE%" echo !BASE!,!RUN_ID!,!TAU_DELAY!,!RADIUS!,!OUT_DIR!\!BASE!_recurr.txt

echo   [2/2] plot: recurrence matrix diagnostics...
if /i "%HAS_GNUPLOT%"=="true" (
    "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1200,1200 enhanced; set output '!OUT_DIR!\!BASE!_recurrence.png'; unset key; set title '!BASE! Recurrence Plot ^(tau=!TAU_DELAY!, r=!RADIUS!, m=%EMBED_DIM%^) %TEST_SUFFIX%'; set xlabel 'Time index'; set ylabel 'Time index'; plot '!OUT_DIR!\!BASE!_recurr.txt' with dots lc rgb 'black'" > "!OUT_DIR!\gnuplot.log" 2>&1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_recurrence.png"
)
echo(
exit /b 0
