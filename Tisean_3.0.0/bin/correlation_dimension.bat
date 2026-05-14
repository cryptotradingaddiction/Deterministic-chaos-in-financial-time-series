@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM CORRELATION DIMENSION PIPELINE (Takens/Ellner, all files)
REM Uses d2.exe for correlation sums, c2t.exe for Takens' estimator, and Python
REM post-processing for Ellner's extension. Local D2 slopes are diagnostics only.
REM One run per coin:
REM   run2 = per-symbol (TAU_D2_<sym>, W_D2_<sym>) from _per_coin_settings.bat
REM ============================================================================

REM ------------------------------ USER CONFIG ---------------------------------
REM Optional: set DCH_DECISION_ABS_Z_SIGMA=3   -> hypothesis uses |z_sigma|>=3 for decision column (thesis "3 sigma" rule).
set TEST_MODE=false
if defined DCH_TEST_MODE set TEST_MODE=%DCH_TEST_MODE%
set RUN_HYPOTHESIS=true
if defined DCH_RUN_HYPOTHESIS set RUN_HYPOTHESIS=%DCH_RUN_HYPOTHESIS%
set "DIMENSION_METRICS=ELLNER"
if defined DCH_DIMENSION_METRICS set "DIMENSION_METRICS=%DCH_DIMENSION_METRICS%"
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

REM Fixed parameters: m range for d2.exe.
set EMBED=1,3
REM ----------------------------------------------------------------------------

cd /d "%DATA_DIR%" || (echo ERROR: Cannot enter %DATA_DIR% & exit /b 1)

if /i "%TEST_MODE%"=="true" (
    set OUT_ROOT=%RESULTS_DIR%\correlation_dimension_test_2000
    set TMP_ROOT=%DATA_DIR%\results_test_2000
    set TEST_SUFFIX=_test2000
    set PLOT_SUFFIX= test2000
    echo [INFO] TEST MODE - first 2000 lines per file
) else (
    set OUT_ROOT=%RESULTS_DIR%\correlation_dimension_full
    set TMP_ROOT=%DATA_DIR%\results_full
    set TEST_SUFFIX=
    set PLOT_SUFFIX=
    echo [INFO] FULL MODE - using complete files
)

echo [INFO] Output root : %OUT_ROOT%
echo [INFO] m setting   : %EMBED% (components,max_embed -> using m=3 block)
echo [INFO] Hypothesis  : %RUN_HYPOTHESIS%
echo [INFO] Dimension hypothesis metrics: %DIMENSION_METRICS%
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

set "AGG_FILE=%OUT_ROOT%\_correlation_dimension_summary.txt"
> "%AGG_FILE%" echo symbol,run,tau,W,d2_file,takens_file
set "TAKENS_AGG_FILE=%OUT_ROOT%\_takens_summary.csv"
> "%TAKENS_AGG_FILE%" echo symbol,run,tau,W,takens_file,ellner_m3,plateau_points,r_min_m3,r_max_m3

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

    REM ---- Resolve per-symbol tau/W for run2 ----
    call set "COIN_TAU=%%TAU_D2_!BASE!%%"
    call set "COIN_W=%%W_D2_!BASE!%%"
    if "!COIN_TAU!"=="" set "COIN_TAU=3"
    if "!COIN_W!"==""   set "COIN_W=0"

    call :RUN_D2 "!BASE!" "!DATA_FILE!" "run2_tau!COIN_TAU!_W!COIN_W!" !COIN_TAU! !COIN_W!
    if errorlevel 1 exit /b 1

    if /i "%RUN_HYPOTHESIS%"=="true" (
        set "RUN2_DIR=%OUT_ROOT%\!BASE!_run2_tau!COIN_TAU!_W!COIN_W!"
        set "HYP_DIR=!RUN2_DIR!\hypothesis_d2"
        if not exist "!HYP_DIR!" mkdir "!HYP_DIR!"
        echo   [Hypothesis] dimension surrogate test ^(metrics=%DIMENSION_METRICS%, tau=!COIN_TAU!, W=!COIN_W!^)
        "%PYTHON_EXE%" %PYTHON_ARGS% "C:\DCh\hypothesis.py" --input "!DATA_FILE!" --base "!BASE!" --delay !COIN_TAU! --theiler !COIN_W! --output_dir "!HYP_DIR!" --test_mode "%TEST_MODE%" --metrics_list "%DIMENSION_METRICS%"
        if errorlevel 1 exit /b 1
        "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot "!HYP_DIR!\!BASE!_surrogate_summary.txt"
    ) else (
        echo   [Hypothesis] skipped ^(DCH_RUN_HYPOTHESIS=%RUN_HYPOTHESIS%^)
    )
)

if /i "%TEST_MODE%"=="true" (
    del /q "%TMP_ROOT%\tmp_*_test2000.dat" >nul 2>&1
)

echo(
echo ============================================================================
echo Correlation-dimension run completed.
echo Aggregate index: %AGG_FILE%
echo Results root   : %OUT_ROOT%
echo [INFO] Takens summary: %TAKENS_AGG_FILE%
if /i "%RUN_HYPOTHESIS%"=="true" (
    echo [INFO] Aggregating Takens/Ellner surrogate summaries...
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot_aggregate "%OUT_ROOT%"
    echo [INFO] Aggregate hypothesis summary: %OUT_ROOT%\_hypothesis_aggregate_summary.txt
) else (
    echo [INFO] Hypothesis aggregation skipped.
)
echo ============================================================================
exit /b 0


REM ============================================================================
REM :RUN_D2 <BASE> <DATA_FILE> <RUN_ID> <TAU> <W>
REM ============================================================================
:RUN_D2
set "BASE=%~1"
set "DATA_FILE=%~2"
set "RUN_ID=%~3"
set "TAU_DELAY=%~4"
set "THEILER_W=%~5"
set "OUT_DIR=%OUT_ROOT%\!BASE!_!RUN_ID!"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"
del /q "!OUT_DIR!\!BASE!_ellner_all_m.png" >nul 2>&1

echo(
echo   --------------------------------------------------
echo   Running Takens/Ellner dimension analysis: !RUN_ID! (tau=!TAU_DELAY!, W=!THEILER_W!)
echo   Data file : !DATA_FILE!
echo   Output dir: !OUT_DIR!
echo   --------------------------------------------------

echo   [1/3] d2: default-range correlation sums + diagnostic local slopes...
"%TISEAN%\d2.exe" -d!TAU_DELAY! -M%EMBED% -t!THEILER_W! -#100 -N0 -o "!OUT_DIR!\!BASE!" "!DATA_FILE!"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" d2 "!OUT_DIR!\!BASE!.d2"

echo   [2/3] c2t: Takens-Theiler estimator; Ellner uses the detected plateau...
pushd "!OUT_DIR!"
"%TISEAN%\c2t.exe" -V0 -o "!BASE!_takens.dat" "!BASE!.c2"
set C2T_ERR=!errorlevel!
popd
if not "!C2T_ERR!"=="0" exit /b !C2T_ERR!
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" takens "!OUT_DIR!\!BASE!_takens.dat"
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" takens "!OUT_DIR!\!BASE!_takens.dat" > "!OUT_DIR!\!BASE!_takens_summary.txt"
set "TAKENS_VALUE_TMP=!OUT_DIR!\!BASE!_takens_value.tmp"
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" takens_value "!OUT_DIR!\!BASE!_takens.dat" > "!TAKENS_VALUE_TMP!"
for /f "usebackq tokens=1,2,3,4,5 delims=," %%A in ("!TAKENS_VALUE_TMP!") do (
    >> "%TAKENS_AGG_FILE%" echo !BASE!,!RUN_ID!,!TAU_DELAY!,!THEILER_W!,!OUT_DIR!\!BASE!_takens.dat,%%B,%%C,%%D,%%E
)
del /q "!TAKENS_VALUE_TMP!" >nul 2>&1
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" ellner_plot_data "!OUT_DIR!\!BASE!_takens.dat" > "!OUT_DIR!\!BASE!_ellner.dat"
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_ellner.dat"

>> "%AGG_FILE%" echo !BASE!,!RUN_ID!,!TAU_DELAY!,!THEILER_W!,!OUT_DIR!\!BASE!.d2,!OUT_DIR!\!BASE!_takens.dat

REM .c2 and c2t output are the active Takens/Ellner sources; .d2 local slopes are diagnostic.
echo   [3/3] plot: log C, diagnostic local d_2, and Takens curves with Ellner intervals ...
if /i "%HAS_GNUPLOT%"=="true" (
    "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!OUT_DIR!\!BASE!_C2_loglog_all_m.png'; set grid; set xlabel 'ln r'; set ylabel 'ln C^{(m)}(r)'; set title '!BASE! correlation integral ln C^{(m)}(r) vs ln r (tau=!TAU_DELAY!, W=!THEILER_W!)!PLOT_SUFFIX!'; set key left top font 'Arial,7' vertical maxrows 30; plot for [idx=0:2] '!OUT_DIR!\!BASE!.c2' index idx using (log($1)):(log($2)) with lines lw 1 title sprintf('m=%%d', idx+1)" > "!OUT_DIR!\gnuplot.log" 2>&1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_C2_loglog_all_m.png"
    "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!OUT_DIR!\!BASE!_D2_all_m.png'; set grid; set xlabel 'ln r'; set ylabel 'd_2^{(m)}'; set title '!BASE! local correlation dimension d_2^{(m)} (tau=!TAU_DELAY!, W=!THEILER_W!)!PLOT_SUFFIX!'; set key right bottom font 'Arial,7' vertical maxrows 30; plot for [idx=0:2] '!OUT_DIR!\!BASE!.d2' index idx using (log($1)):2 with lines lw 1 title sprintf('m=%%d', idx+1)" >> "!OUT_DIR!\gnuplot.log" 2>&1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_D2_all_m.png"
    "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!OUT_DIR!\!BASE!_takens_all_m.png'; set grid; set xlabel 'ln r'; set ylabel 'dimension estimate'; set title '!BASE! Takens d_2^T with Ellner d_2^E intervals (tau=!TAU_DELAY!, W=!THEILER_W!)!PLOT_SUFFIX!'; set key right bottom font 'Arial,7' vertical maxrows 30; plot for [idx=0:2] '!OUT_DIR!\!BASE!_takens.dat' index idx using (log($1)):2 with lines lw 1 title sprintf('Takens m=%%d', idx+1), '!OUT_DIR!\!BASE!_ellner.dat' index 0 using (log($1)):2 with lines lw 3 dashtype 2 title 'Ellner interval', for [idx=1:2] '!OUT_DIR!\!BASE!_ellner.dat' index idx using (log($1)):2 with lines lw 3 dashtype 2 notitle" >> "!OUT_DIR!\gnuplot.log" 2>&1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_takens_all_m.png"
)
echo(
exit /b 0
