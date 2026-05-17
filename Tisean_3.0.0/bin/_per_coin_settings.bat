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
REM   TAU_D2_<sym>, W_D2_<sym>     ->  correlation_dimension.bat (Takens/Ellner)
REM                                    hypothesis.bat   TISEAN dimension branch
REM
REM   TAU_LLE_<sym>                ->  Lambda_max.bat            (largest Lyapunov)
REM                                    (lyap_k uses Theiler default)
REM
REM   TAU_RQA_<sym>, RAD_RQA_<sym> ->  RQA.bat                   (recurrence)
REM
REM ----------------------------------------------------------------------------
REM Each TAU_* below is overwritten when you run mutual.py (first MI minimum → _mi_summary.txt).
REM ----------------------------------------------------------------------------

REM ---- Takens/Ellner dimension settings (d2.exe + c2t.exe pipeline) ----------
set TAU_D2_BTCUSD=5
set TAU_D2_ETHUSD=3
set TAU_D2_LTCUSD=3
set TAU_D2_XRPUSD=2
set TAU_D2_LINKUSD=2
set TAU_D2_DOGEUSD=3
set TAU_D2_ADAUSD=2
set W_D2_BTCUSD=5
set W_D2_ETHUSD=3
set W_D2_LTCUSD=3
set W_D2_XRPUSD=2
set W_D2_LINKUSD=2
set W_D2_DOGEUSD=3
set W_D2_ADAUSD=2
REM ---- Largest Lyapunov exponent (lyap_k) -----------------------------------
set TAU_LLE_BTCUSD=5
set TAU_LLE_ETHUSD=3
set TAU_LLE_LTCUSD=3
set TAU_LLE_XRPUSD=2
set TAU_LLE_LINKUSD=2
set TAU_LLE_DOGEUSD=3
set TAU_LLE_ADAUSD=2
REM ---- RQA (recurr) - tau and recurrence threshold radius -------------------
set TAU_RQA_BTCUSD=5
set TAU_RQA_ETHUSD=3
set TAU_RQA_LTCUSD=3
set TAU_RQA_XRPUSD=2
set TAU_RQA_LINKUSD=2
set TAU_RQA_DOGEUSD=3
set TAU_RQA_ADAUSD=2
set RAD_RQA_BTCUSD=0.005
set RAD_RQA_ETHUSD=0.005
set RAD_RQA_LTCUSD=0.005
set RAD_RQA_XRPUSD=0.005
set RAD_RQA_LINKUSD=0.005
set RAD_RQA_DOGEUSD=0.005
set RAD_RQA_ADAUSD=0.005

exit /b 0