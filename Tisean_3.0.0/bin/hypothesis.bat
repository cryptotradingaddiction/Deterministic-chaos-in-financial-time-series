@echo off
setlocal
echo NOTE: hypothesis.bat is a distributed single-surrogate hypothesis wrapper.
echo It runs per-invariant pipelines; each calls hypothesis.py for orig/randperm/normal/t reference runs.
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
exit /b 0
