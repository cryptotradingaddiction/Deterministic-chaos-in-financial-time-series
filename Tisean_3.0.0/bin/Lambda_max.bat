@echo off
setlocal enabledelayedexpansion
call "%~dp0_dch_test_env.bat"
call "%~dp0_dch_hypothesis_cli_extra.bat"

REM ============================================================================
REM LARGEST LYAPUNOV PIPELINE (Kantz, all files)
REM Runs lyap_k.exe to compute S(t) divergence curves and plots all m-blocks.
REM One run per coin:
REM   run2 = per-symbol TAU_LLE_<sym> from _per_coin_settings.bat
REM ============================================================================

REM ------------------------------ USER CONFIG ---------------------------------
set TEST_MODE=false
if defined DCH_TEST_MODE set TEST_MODE=%DCH_TEST_MODE%
set RUN_HYPOTHESIS=true
if defined DCH_RUN_HYPOTHESIS set RUN_HYPOTHESIS=%DCH_RUN_HYPOTHESIS%

REM Relocatable paths derived from this .bat file's own location.
REM NB: keep these on separate lines — `pushd ... && set "REPO_ROOT=%CD%" && popd`
REM expands %CD% at parse time (before pushd runs) and yields the *caller's* cwd.
pushd "%~dp0..\.."
set "REPO_ROOT=%CD%"
popd
set "DATA_DIR=%REPO_ROOT%\data"
set "RESULTS_DIR=%DATA_DIR%\results"
set "TISEAN=%~dp0"
if "%TISEAN:~-1%"=="\" set "TISEAN=%TISEAN:~0,-1%"
set "PYTHON_EXE=py"
set "PYTHON_ARGS=-3"
set "PRINT_RESULTS=%REPO_ROOT%\print_results.py"

REM gnuplot: PATH lookup first, then Windows install fallback.
set "GNUPLOT_EXE="
for %%G in (gnuplot.exe) do set "GNUPLOT_EXE=%%~$PATH:G"
if "%GNUPLOT_EXE%"=="" if exist "C:\Program Files\gnuplot\bin\gnuplot.exe" set "GNUPLOT_EXE=C:\Program Files\gnuplot\bin\gnuplot.exe"

REM Per-coin filename list from config_loader.pipeline_logreturn_files.
pushd "%REPO_ROOT%"
for /f "delims=" %%F in ('%PYTHON_EXE% %PYTHON_ARGS% -c "from config_loader import pipeline_logreturn_files; print(' '.join(pipeline_logreturn_files()))"') do set "FILES=%%F"
popd
if "%FILES%"=="" (
    echo [ERROR] Could not derive FILES from config_loader.pipeline_logreturn_files.
    exit /b 1
)

REM Per-coin tau overrides come from the shared settings file.
call "%~dp0_per_coin_settings.bat"

REM Fixed parameters for lyap_k (Kantz algorithm).
REM   M_MIN..M_MAX : embedding sweep for diagnostic plots; OLS slope still extracted at m=M_PRIMARY
REM   STEPS        : lyap_k -n (reference points used to average S(t))
REM   ITER         : lyap_k -s (forward iteration count = length of S(t) curve)
REM   MIN_NEIGHBORS: Python-side filter applied in extract_lle_ols (NOT a lyap_k flag)
set M_MIN=3
set M_MAX=10
set M_PRIMARY=3
set STEPS=500
set ITER=100
set MIN_NEIGHBORS=10
REM ----------------------------------------------------------------------------

cd /d "%DATA_DIR%" || (echo ERROR: Cannot enter %DATA_DIR% & exit /b 1)

if /i "%TEST_MODE%"=="true" (
    set OUT_ROOT=%RESULTS_DIR%\lambda_max_test_%TEST_POINT_COUNT%
    set TMP_ROOT=%DATA_DIR%\results_test_%TEST_POINT_COUNT%
    set TEST_SUFFIX=_test%TEST_POINT_COUNT%
    set PLOT_SUFFIX= test%TEST_POINT_COUNT%
    if not defined DCH_LYAP_STEPS set DCH_LYAP_STEPS=200
    if not defined DCH_LYAP_ITERATIONS set DCH_LYAP_ITERATIONS=30
    if not defined DCH_LYAP_MIN_NEIGHBORS set DCH_LYAP_MIN_NEIGHBORS=3
    set STEPS=%DCH_LYAP_STEPS%
    set ITER=%DCH_LYAP_ITERATIONS%
    set MIN_NEIGHBORS=%DCH_LYAP_MIN_NEIGHBORS%
    echo [INFO] TEST MODE - first %TEST_POINT_COUNT% lines per file
) else (
    set OUT_ROOT=%RESULTS_DIR%\lambda_max_full
    set TMP_ROOT=%DATA_DIR%\results_full
    set TEST_SUFFIX=
    set PLOT_SUFFIX=
    REM Clear any stale test-mode overrides so hypothesis.py uses production
    REM defaults from hypothesis_config (matches lyap_k.exe flags above).
    set "DCH_LYAP_STEPS="
    set "DCH_LYAP_ITERATIONS="
    set "DCH_LYAP_MIN_NEIGHBORS="
    echo [INFO] FULL MODE - using complete files
)

echo [INFO] Output root : %OUT_ROOT%
echo [INFO] m range     : %M_MIN%..%M_MAX% (primary m=%M_PRIMARY%)   r=TISEAN defaults
echo [INFO] lyap_k flags: -n%STEPS% reference points, -s%ITER% S(t) iterations
echo [INFO] Python filter: min_neighbors=%MIN_NEIGHBORS%
echo [INFO] Hypothesis  : %RUN_HYPOTHESIS%
echo [INFO] Per-coin run:
echo [INFO]   run2 = per-symbol (TAU_LLE_^<sym^>)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%TMP_ROOT%" mkdir "%TMP_ROOT%"

if exist "%GNUPLOT_EXE%" (
    set HAS_GNUPLOT=true
) else (
    set HAS_GNUPLOT=false
    echo [WARN] gnuplot not found at "%GNUPLOT_EXE%". Plotting will be skipped.
)

set "AGG_FILE=%OUT_ROOT%\_lambda_max_summary.txt"
> "%AGG_FILE%" echo symbol,run,tau,W,lyap_file

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
        echo   [ERROR] Run liquidity.py from the repository root before this pipeline.
        exit /b 1
    )

    if /i "%TEST_MODE%"=="true" (
        set "DATA_FILE=%TMP_ROOT%\tmp_!BASE!%TEST_SUFFIX%.dat"
        powershell -NoProfile -Command "Get-Content -Path '!FULL_DATA!' -TotalCount %TEST_POINT_COUNT% | Set-Content -Path '!DATA_FILE!' -Encoding ascii"
    ) else (
        REM Use absolute path so subroutines can pushd into OUT_DIR safely.
        set "DATA_FILE=%DATA_DIR%\!FULL_DATA!"
    )

    if not exist "!DATA_FILE!" (
        echo   [ERROR] Input file missing for !BASE!: !DATA_FILE!
        exit /b 1
    )

    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!DATA_FILE!"

    REM ---- Resolve per-symbol tau / Theiler W for run2 ----
    call set "COIN_TAU=%%TAU_LLE_!BASE!%%"
    call set "COIN_W=%%W_D2_!BASE!%%"
    if "!COIN_TAU!"=="" set "COIN_TAU=3"
    if "!COIN_W!"=="" set "COIN_W=0"

    call :RUN_LLE "!BASE!" "!DATA_FILE!" "run2_tau!COIN_TAU!_W!COIN_W!" !COIN_TAU! !COIN_W!
    if errorlevel 1 exit /b 1

    if /i "%RUN_HYPOTHESIS%"=="true" (
        set "RUN2_DIR=%OUT_ROOT%\!BASE!_run2_tau!COIN_TAU!_W!COIN_W!"
        set "HYP_DIR=!RUN2_DIR!\hypothesis_lle"
        if not exist "!HYP_DIR!" mkdir "!HYP_DIR!"
        echo   [Hypothesis] LLE stationary-bootstrap TS test ^(tau=!COIN_TAU!, W=!COIN_W!^)
        "%PYTHON_EXE%" %PYTHON_ARGS% "%REPO_ROOT%\hypothesis.py" --input "!DATA_FILE!" --base "!BASE!" --delay !COIN_TAU! --theiler !COIN_W! --output_dir "!HYP_DIR!" --test_mode "%TEST_MODE%" --metrics_list "LLE" !DCH_HYP_EXTRA!
        if errorlevel 1 exit /b 1
        "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot "!HYP_DIR!\!BASE!_surrogate_summary.txt"
    ) else (
        echo   [Hypothesis] skipped ^(DCH_RUN_HYPOTHESIS=%RUN_HYPOTHESIS%^)
    )
)

if /i "%TEST_MODE%"=="true" (
    del /q "%TMP_ROOT%\tmp_*_test%TEST_POINT_COUNT%.dat" >nul 2>&1
)

echo(
echo ============================================================================
echo Lambda_max run completed.
echo Aggregate index: %AGG_FILE%
echo Results root   : %OUT_ROOT%
if /i "%RUN_HYPOTHESIS%"=="true" (
    echo [INFO] Aggregating LLE bootstrap summaries...
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" boot_aggregate "%OUT_ROOT%"
    echo [INFO] Aggregate hypothesis summary: %OUT_ROOT%\_hypothesis_aggregate_summary.txt
) else (
    echo [INFO] Hypothesis aggregation skipped.
)
echo ============================================================================
exit /b 0


REM ============================================================================
REM :RUN_LLE <BASE> <DATA_FILE> <RUN_ID> <TAU> <W>
REM ============================================================================
:RUN_LLE
set "BASE=%~1"
set "DATA_FILE=%~2"
set "RUN_ID=%~3"
set "TAU_DELAY=%~4"
set "THEILER_W=%~5"
if "!THEILER_W!"=="" set "THEILER_W=0"
set "OUT_DIR=%OUT_ROOT%\!BASE!_!RUN_ID!"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

echo(
echo   --------------------------------------------------
echo   Running LLE analysis: !RUN_ID! (tau=!TAU_DELAY!, W=!THEILER_W!)
echo   Data file : !DATA_FILE!
echo   Output dir: !OUT_DIR!
echo   --------------------------------------------------

echo   [1/2] lyap_k: Kantz S^(t^) divergence curves ^(m=%M_MIN%..%M_MAX%, -n%STEPS% ref pts, -s%ITER% iters, -t!THEILER_W!^)...
"%TISEAN%\lyap_k.exe" -d!TAU_DELAY! -m%M_MIN% -M%M_MAX% -t!THEILER_W! -n%STEPS% -s%ITER% -o "!OUT_DIR!\!BASE!_lyap.txt" "!DATA_FILE!"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" lyap "!OUT_DIR!\!BASE!_lyap.txt"

>> "%AGG_FILE%" echo !BASE!,!RUN_ID!,!TAU_DELAY!,!THEILER_W!,!OUT_DIR!\!BASE!_lyap.txt

REM lyap_k writes one block per (epsilon, dim) header (#epsilon= ... dim= ...).
REM Across m=M_MIN..M_MAX with several epsilon scales, the output therefore has
REM (m_count * eps_count) blocks. Plot all of them; m=%M_PRIMARY% is the active block.
echo   [2/2] plot: Kantz S^(t^) curves across all (epsilon, m) blocks ...
if /i "%HAS_GNUPLOT%"=="true" (
    "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!OUT_DIR!\!BASE!_lyap_St.png'; set title '!BASE! Kantz Lyapunov S(t), m=%M_MIN%..%M_MAX% (primary m=%M_PRIMARY%), tau=!TAU_DELAY!, W=!THEILER_W!!PLOT_SUFFIX!'; set xlabel 'iteration t'; set ylabel 'S(t)'; set grid; set key outside font 'Arial,7' vertical maxrows 30; plot for [i=0:*] '!OUT_DIR!\!BASE!_lyap.txt' index i using 1:2 with lines lw 0.7 title sprintf('block %%d', i)" > "!OUT_DIR!\gnuplot.log" 2>&1
    "%PYTHON_EXE%" %PYTHON_ARGS% "%PRINT_RESULTS%" file "!OUT_DIR!\!BASE!_lyap_St.png"
)
echo(
exit /b 0
