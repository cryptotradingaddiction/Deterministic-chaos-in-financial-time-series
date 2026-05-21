@echo off
setlocal enabledelayedexpansion
call "%~dp0_dch_test_env.bat"

REM ============================================================================
REM THEILER WINDOW PIPELINE (Kantz/Schreiber formula, all coins)
REM ----------------------------------------------------------------------------
REM For every symbol:
REM   1) corr.exe writes the autocorrelation function -> *_acf.dat.
REM   2) stp.exe writes the space-time separation plot -> *_stp.dat
REM      (kept for diagnostic plots only; no longer drives W).
REM   3) detect_theiler.py computes the Theiler window from the textbook
REM      lower bound  W = ceil(tau_a * (2/N) ** (2/m))  (eq. 8.85), where
REM      tau_a is the decorrelation time (first non-positive ACF lag), N is
REM      the series length and m is the embedding dimension. For realistic
REM      N this bound is essentially trivial (raw value < 1), so it is the
REM      practical recommendation from the same reference --
REM   4) --floor_at_tau (sets W_final := TAU_D2_<sym>; project rule W = tau).
REM   5) The Python helper writes a per-coin KEY=VALUE report so the .bat can
REM      log TAU_A, N, W_FORMULA, W_STP (diagnostic), W_FINAL into
REM      ``_theiler_summary.txt``.
REM
REM After the loop the script invokes
REM ``config_loader.sync_per_coin_bat_w_d2_from_theiler_summary`` to write
REM ``W_D2_<sym>`` into ``_per_coin_settings.bat`` for every coin.
REM
REM Honours ``DCH_TEST_MODE=true`` (first DCH_TEST_POINTS lines) like the other batches.
REM ============================================================================

REM ------------------------------ USER CONFIG ---------------------------------
set TEST_MODE=false
if defined DCH_TEST_MODE set TEST_MODE=%DCH_TEST_MODE%

REM Resolve repo root from this .bat file's own location so the project works
REM after a fresh `git clone` to any directory, without editing absolute paths.
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
set "DETECT_SCRIPT=%TISEAN%\detect_theiler.py"

REM gnuplot: try PATH first, fall back to the well-known Windows install path.
set "GNUPLOT_EXE="
for %%G in (gnuplot.exe) do set "GNUPLOT_EXE=%%~$PATH:G"
if "%GNUPLOT_EXE%"=="" if exist "C:\Program Files\gnuplot\bin\gnuplot.exe" set "GNUPLOT_EXE=C:\Program Files\gnuplot\bin\gnuplot.exe"

REM Per-coin filename list comes from config_loader.pipeline_logreturn_files
REM so adding/removing a coin only touches PIPELINE_SYMBOLS in one place.
pushd "%REPO_ROOT%"
for /f "delims=" %%F in ('%PYTHON_EXE% %PYTHON_ARGS% -c "from config_loader import pipeline_logreturn_files; print(' '.join(pipeline_logreturn_files()))"') do set "FILES=%%F"
popd
if "%FILES%"=="" (
    echo [ERROR] Could not derive FILES from config_loader.pipeline_logreturn_files.
    exit /b 1
)

REM Per-coin TAU_D2 (already synchronised with mutual.py) drives the embedding
REM delay used by stp.exe. corr.exe is purely scalar so it does not need tau.
call "%~dp0_per_coin_settings.bat"

REM Fixed parameters for ACF, diagnostic STP plot and the textbook formula.
set EMBED=3
set ACF_LAG=500
set STP_T=500
set STP_PCT=0.05
set DECOR_METHOD=acf_zero
set FLOOR_AT_TAU=true
set DET_THRESHOLD=0.95
set DET_SMOOTH=5
set DET_CHECK=10
set DET_AGGREGATE=max
REM ----------------------------------------------------------------------------

cd /d "%DATA_DIR%" || (echo ERROR: Cannot enter %DATA_DIR% & exit /b 1)

if /i "%TEST_MODE%"=="true" (
    set "OUT_ROOT=%RESULTS_DIR%\theiler_w_test_%TEST_POINT_COUNT%"
    set "TMP_ROOT=%DATA_DIR%\results_test_%TEST_POINT_COUNT%"
    set "TEST_SUFFIX=_test%TEST_POINT_COUNT%"
    set "PLOT_SUFFIX= test%TEST_POINT_COUNT%"
    echo [INFO] TEST MODE - first %TEST_POINT_COUNT% lines per file
) else (
    set "OUT_ROOT=%RESULTS_DIR%\theiler_w"
    set "TMP_ROOT=%DATA_DIR%\results_full"
    set "TEST_SUFFIX="
    set "PLOT_SUFFIX="
    echo [INFO] FULL MODE - using complete files
)

echo [INFO] Output root : %OUT_ROOT%
echo [INFO] m = %EMBED%   ACF lag = %ACF_LAG%   STP t = %STP_T%   STP perc = %STP_PCT%
echo [INFO] formula: W = ceil(tau_a * (2/N)^^(2/m)) (eq. 8.85)   decorrelation: %DECOR_METHOD%   floor_at_tau=%FLOOR_AT_TAU%
echo [INFO] diagnostic STP saturation: threshold=%DET_THRESHOLD%  smooth=%DET_SMOOTH%  check=%DET_CHECK%  aggregate=%DET_AGGREGATE%

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%TMP_ROOT%" mkdir "%TMP_ROOT%"

if exist "%GNUPLOT_EXE%" (
    set "HAS_GNUPLOT=true"
) else (
    set "HAS_GNUPLOT=false"
    echo [WARN] gnuplot not found; plotting will be skipped.
)

set "SUMMARY_FILE=%OUT_ROOT%\_theiler_summary.txt"
> "%SUMMARY_FILE%" echo # theiler-window summary TEST_MODE=%TEST_MODE% formula=ceil(tau_a*(2/N)^^(2/m)) decor=%DECOR_METHOD% floor_at_tau=%FLOOR_AT_TAU%
>> "%SUMMARY_FILE%" echo # columns: symbol  N  tau_d2  tau_a  W_formula  W_stp  W_final
>> "%SUMMARY_FILE%" echo #

for %%F in (%FILES%) do (
    for /f "tokens=1 delims=_" %%A in ("%%F") do set "BASE=%%A"
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
        set "DATA_FILE=%DATA_DIR%\!FULL_DATA!"
    )

    if not exist "!DATA_FILE!" (
        echo   [ERROR] Input file missing for !BASE!: !DATA_FILE!
        exit /b 1
    )

    REM Per-symbol embedding delay (fallback tau=3 matches the other batches).
    call set "COIN_TAU=%%TAU_D2_!BASE!%%"
    if "!COIN_TAU!"=="" set "COIN_TAU=3"

    set "OUT_DIR=%OUT_ROOT%\!BASE!"
    if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"
    set "ACF_FILE=!OUT_DIR!\!BASE!_acf.dat"
    set "STP_FILE=!OUT_DIR!\!BASE!_stp.dat"
    set "REPORT_FILE=!OUT_DIR!\!BASE!_theiler_report.txt"
    set "WFINAL_TMP=!OUT_DIR!\!BASE!_W_final.tmp"

    echo   [1/4] corr.exe: autocorrelation function ^(lag 0..%ACF_LAG%^)...
    "%TISEAN%\corr.exe" -D%ACF_LAG% "!DATA_FILE!" -o "!ACF_FILE!"
    if errorlevel 1 (
        echo   [ERROR] corr.exe failed for !BASE!
        exit /b 1
    )

    echo   [2/4] stp.exe: space-time separation plot ^(tau=!COIN_TAU!, m=%EMBED%^)...
    "%TISEAN%\stp.exe" -d!COIN_TAU! -m%EMBED% -t%STP_T% -%%%STP_PCT% "!DATA_FILE!" -o "!STP_FILE!"
    if errorlevel 1 (
        echo   [ERROR] stp.exe failed for !BASE!
        exit /b 1
    )

    echo   [3/4] detect_theiler.py: W = ceil^(tau_a * ^(2/N^)^^^(2/m^)^) from ACF...
    set "FLOOR_FLAG="
    if /i "%FLOOR_AT_TAU%"=="true" set "FLOOR_FLAG=--floor_at_tau"
    "%PYTHON_EXE%" %PYTHON_ARGS% "%DETECT_SCRIPT%" --acf "!ACF_FILE!" --stp "!STP_FILE!" --data "!DATA_FILE!" --m %EMBED% --tau !COIN_TAU! --decor %DECOR_METHOD% --threshold %DET_THRESHOLD% --smooth %DET_SMOOTH% --check %DET_CHECK% --aggregate %DET_AGGREGATE% --fallback 0 !FLOOR_FLAG! --symbol "!BASE!" --report "!REPORT_FILE!" > "!WFINAL_TMP!"
    if errorlevel 1 (
        echo   [WARN] detect_theiler.py failed for !BASE!; defaulting W=0
        set "W_FINAL=0"
        set "TAU_A=0"
        set "W_FORMULA=0"
        set "W_STP=0"
        set "N_PTS=0"
    ) else (
        set "W_FINAL=0"
        set /p W_FINAL=<"!WFINAL_TMP!"
        set "TAU_A=0"
        set "W_FORMULA=0"
        set "W_STP=0"
        set "N_PTS=0"
        for /f "usebackq tokens=1,* delims==" %%a in ("!REPORT_FILE!") do (
            if /i "%%a"=="TAU_A" set "TAU_A=%%b"
            if /i "%%a"=="W_FORMULA" set "W_FORMULA=%%b"
            if /i "%%a"=="W_STP" set "W_STP=%%b"
            if /i "%%a"=="W_FINAL" set "W_FINAL=%%b"
            if /i "%%a"=="N" set "N_PTS=%%b"
        )
    )
    if exist "!WFINAL_TMP!" del /q "!WFINAL_TMP!" >nul 2>&1
    if "!W_FINAL!"=="" set "W_FINAL=0"
    if "!TAU_A!"=="" set "TAU_A=0"
    if "!W_FORMULA!"=="" set "W_FORMULA=0"
    if "!W_STP!"=="" set "W_STP=0"
    if "!N_PTS!"=="" set "N_PTS=0"

    echo   [4/4] result: tau_a=!TAU_A!  W_formula=!W_FORMULA!  W_stp_diag=!W_STP!  W_final=!W_FINAL!  N=!N_PTS!

    >> "%SUMMARY_FILE%" echo !BASE!  !N_PTS!  !COIN_TAU!  !TAU_A!  !W_FORMULA!  !W_STP!  !W_FINAL!

    if /i "!HAS_GNUPLOT!"=="true" (
        set "PNG_ACF=!OUT_DIR!\!BASE!_acf.png"
        set "PNG_STP_LIN=!OUT_DIR!\!BASE!_stp_linear.png"
        set "PNG_STP_LOG=!OUT_DIR!\!BASE!_stp_log.png"
        echo   [plots] gnuplot: ACF + STP diagnostic PNGs...
        "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!PNG_ACF!'; set grid; set xlabel 'lag'; set ylabel 'ACF'; set title '!BASE! autocorrelation tau_a=!TAU_A! W=!W_FINAL!!PLOT_SUFFIX!'; set xrange [0:%ACF_LAG%]; set arrow from !TAU_A!, graph 0 to !TAU_A!, graph 1 nohead lc rgb 'red' dt 2; set arrow from !W_FINAL!, graph 0 to !W_FINAL!, graph 1 nohead lc rgb 'dark-green' dt 2; plot '!ACF_FILE!' using 1:2 with lines lw 1.5 title 'ACF', 0 with lines lc rgb 'black' notitle" > "!OUT_DIR!\gnuplot.log" 2>&1
        "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!PNG_STP_LIN!'; set grid; set xlabel 'time lag dt'; set ylabel 'spatial distance'; set title '!BASE! STP tau=!COIN_TAU! m=%EMBED% W=!W_FINAL!!PLOT_SUFFIX!'; set key top left font 'Arial,8'; set arrow from !W_FINAL!, graph 0 to !W_FINAL!, graph 1 nohead lc rgb 'dark-green' dt 2; plot for [i=0:*] '!STP_FILE!' index i using 1:2 with lines lw 0.9 title sprintf('band %%d', i)" >> "!OUT_DIR!\gnuplot.log" 2>&1
        if not exist "!PNG_STP_LIN!" (
            echo   [WARN] STP linear plot missing: !PNG_STP_LIN! ^(see gnuplot.log^)
        ) else (
            echo   [OK] STP linear: !PNG_STP_LIN!
        )
        "%GNUPLOT_EXE%" -e "set terminal pngcairo size 1400,900 enhanced font 'Arial,12'; set output '!PNG_STP_LOG!'; set grid; set xlabel 'time lag dt'; set ylabel 'spatial distance'; set logscale y; set title '!BASE! STP log-y tau=!COIN_TAU! m=%EMBED% W=!W_FINAL!!PLOT_SUFFIX!'; set key top left font 'Arial,8'; set arrow from !W_FINAL!, graph 0 to !W_FINAL!, graph 1 nohead lc rgb 'dark-green' dt 2; plot for [i=0:*] '!STP_FILE!' index i using 1:2 with lines lw 0.9 title sprintf('band %%d', i)" >> "!OUT_DIR!\gnuplot.log" 2>&1
        if not exist "!PNG_STP_LOG!" (
            echo   [WARN] STP log plot missing: !PNG_STP_LOG! ^(see gnuplot.log^)
        ) else (
            echo   [OK] STP log-y: !PNG_STP_LOG!
        )
        if not exist "!PNG_ACF!" (
            echo   [WARN] ACF plot missing: !PNG_ACF! ^(see gnuplot.log^)
        ) else (
            echo   [OK] ACF: !PNG_ACF!
        )
    ) else (
        echo   [INFO] gnuplot skipped ^(not found at "%GNUPLOT_EXE%"^)
    )
)

if /i "%TEST_MODE%"=="true" (
    del /q "%TMP_ROOT%\tmp_*_test%TEST_POINT_COUNT%.dat" >nul 2>&1
)

echo(
echo ============================================================================
echo Theiler-window detection completed.
echo Per-coin outputs: %OUT_ROOT%\^<symbol^>\
echo Summary file   : %SUMMARY_FILE%

echo [INFO] Syncing W_D2_^<sym^> := TAU_D2_^<sym^> in _per_coin_settings.bat...
pushd "%REPO_ROOT%" || (
    echo [ERROR] Cannot cd to repo root for sync helper.
    exit /b 1
)
"%PYTHON_EXE%" %PYTHON_ARGS% -c "from config_loader import sync_per_coin_bat_w_d2_from_theiler_summary, load_config; status, n = sync_per_coin_bat_w_d2_from_theiler_summary(load_config()); print('  status=' + status + ', symbols=' + str(n))"
set SYNC_ERR=%errorlevel%
popd
if not "%SYNC_ERR%"=="0" (
    echo [WARN] _per_coin_settings.bat sync helper returned exit code %SYNC_ERR%.
)

echo ============================================================================
exit /b 0
