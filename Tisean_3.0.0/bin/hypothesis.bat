@echo off
setlocal
echo NOTE: hypothesis.bat is a distributed single-surrogate hypothesis wrapper.
echo It runs per-invariant pipelines; each calls hypothesis.py for orig/randperm/normal/t reference runs.
echo.

echo [1/4] correlation_dimension.bat ...
call "%~dp0correlation_dimension.bat"
if errorlevel 1 exit /b %errorlevel%

echo [2/4] correlation_entropy.bat ...
call "%~dp0correlation_entropy.bat"
if errorlevel 1 exit /b %errorlevel%

echo [3/4] Lambda_max.bat ...
call "%~dp0Lambda_max.bat"
if errorlevel 1 exit /b %errorlevel%

echo [4/4] RQA.bat ...
call "%~dp0RQA.bat"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Distributed hypothesis workflow completed.
exit /b 0
