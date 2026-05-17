@echo off
setlocal
echo NOTE: hypothesis.bat orchestrates the active invariant pipelines (D2, LLE, RQA).
echo Each .bat may call hypothesis.py: stationary-bootstrap TS tests, plus reshuffle/normal/t references.
echo.

echo [1/3] correlation_dimension.bat ...
call "%~dp0correlation_dimension.bat"
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Lambda_max.bat ...
call "%~dp0Lambda_max.bat"
if errorlevel 1 exit /b %errorlevel%

echo [3/3] RQA.bat ...
call "%~dp0RQA.bat"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Distributed hypothesis workflow completed.

echo [4/4] Building Word summary ^(results.docx in results folder^)...
pushd "%~dp0..\.." || ( echo ERROR: Cannot cd to repo root & exit /b 1 )
py -3 documents.py
if errorlevel 1 (
  popd
  exit /b 1
)
for /f "delims=" %%p in ('py -3 -c "from pathlib import Path; from config_loader import get_results_dir, load_config; print(Path(get_results_dir(load_config())) / 'results.docx')"') do set "DOCX_OPEN=%%p"
if exist "%DOCX_OPEN%" (
  start "" "%DOCX_OPEN%"
) else (
  echo WARNING: results.docx not found at "%DOCX_OPEN%"
)
popd

exit /b 0
