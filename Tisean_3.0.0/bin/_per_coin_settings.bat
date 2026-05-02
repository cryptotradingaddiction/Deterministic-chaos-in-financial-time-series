@echo off
REM ============================================================================
REM PER-COIN OVERRIDES FOR TAU AND THEILER WINDOW
REM ----------------------------------------------------------------------------
REM Single source of truth for the (tau, W) values used by every standalone
REM TISEAN .bat script and by hypothesis.bat. Source from any caller with:
REM
REM     call "%~dp0_per_coin_settings.bat"
REM
REM Each script uses these as the active per-coin parameters (run2 naming kept).
REM
REM ----------------------------------------------------------------------------
REM Quantity groups and the scripts that use them:
REM
REM   TAU_D2_<sym>, W_D2_<sym>     ->  correlation_dimension.bat (D2)
REM                                    correlation_entropy.bat   (K2)
REM                                    hypothesis.bat   tisean branch (D2/K2/Takens)
REM
REM   TAU_LLE_<sym>                ->  Lambda_max.bat            (largest Lyapunov)
REM                                    (lyap_k uses Theiler default)
REM
REM   TAU_RQA_<sym>, RAD_RQA_<sym> ->  RQA.bat                   (recurrence)
REM
REM ----------------------------------------------------------------------------
REM Tip: keep values explicit for every symbol to avoid implicit fallbacks.
REM ============================================================================

REM ---- D2 + K2 + Takens (all run via the same d2.exe pipeline) --------------
set TAU_D2_BTCUSD=3
set TAU_D2_ETHUSD=4
set TAU_D2_LTCUSD=3
set TAU_D2_XRPUSD=3
set TAU_D2_LINKUSD=4
set TAU_D2_DOGEUSD=3
set TAU_D2_ADAUSD=2

set W_D2_BTCUSD=0
set W_D2_ETHUSD=0
set W_D2_LTCUSD=0
set W_D2_XRPUSD=0
set W_D2_LINKUSD=0
set W_D2_DOGEUSD=0
set W_D2_ADAUSD=0

REM ---- Largest Lyapunov exponent (lyap_k) -----------------------------------
set TAU_LLE_BTCUSD=3
set TAU_LLE_ETHUSD=4
set TAU_LLE_LTCUSD=3
set TAU_LLE_XRPUSD=3
set TAU_LLE_LINKUSD=4
set TAU_LLE_DOGEUSD=3
set TAU_LLE_ADAUSD=2

REM ---- RQA (recurr) - tau and recurrence threshold radius -------------------
set TAU_RQA_BTCUSD=3
set TAU_RQA_ETHUSD=4
set TAU_RQA_LTCUSD=3
set TAU_RQA_XRPUSD=3
set TAU_RQA_LINKUSD=4
set TAU_RQA_DOGEUSD=3
set TAU_RQA_ADAUSD=2

set RAD_RQA_BTCUSD=0.01
set RAD_RQA_ETHUSD=0.01
set RAD_RQA_LTCUSD=0.01
set RAD_RQA_XRPUSD=0.01
set RAD_RQA_LINKUSD=0.01
set RAD_RQA_DOGEUSD=0.01
set RAD_RQA_ADAUSD=0.01

exit /b 0
