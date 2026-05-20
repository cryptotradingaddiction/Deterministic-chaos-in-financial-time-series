@echo off
REM Shared DCH test-mode defaults (include from TISEAN *.bat after setlocal).
REM ``DCH_TEST_MODE=true`` + first ``DCH_TEST_POINTS`` rows (default 100).
if not defined DCH_TEST_POINTS set DCH_TEST_POINTS=100
set TEST_POINT_COUNT=%DCH_TEST_POINTS%
REM Short-series lyap_k settings when desktop / CLI sets DCH_TEST_MODE=true.
REM   DCH_LYAP_STEPS       -> lyap_k -n (reference points)
REM   DCH_LYAP_ITERATIONS  -> lyap_k -s (S(t) curve length)
REM   DCH_LYAP_MIN_NEIGHBORS -> Python neighbor filter (NOT a lyap_k flag)
if /i "%DCH_TEST_MODE%"=="true" (
    if not defined DCH_LYAP_STEPS set DCH_LYAP_STEPS=200
    if not defined DCH_LYAP_ITERATIONS set DCH_LYAP_ITERATIONS=30
    if not defined DCH_LYAP_MIN_NEIGHBORS set DCH_LYAP_MIN_NEIGHBORS=3
)
