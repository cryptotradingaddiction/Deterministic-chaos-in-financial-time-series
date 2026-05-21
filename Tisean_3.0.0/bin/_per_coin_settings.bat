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
REM   TAU_LLE_<sym>, W_D2_<sym>    ->  Lambda_max.bat (lyap_k -d -t uses same tau/W as d2)
REM
REM   TAU_RQA_<sym>, RAD_RQA_<sym> ->  RQA.bat                   (recurrence)
REM
REM ----------------------------------------------------------------------------
REM Each TAU_* below is overwritten when you run mutual.py (first MI minimum → _mi_summary.txt).
REM ----------------------------------------------------------------------------

REM ---- Takens/Ellner dimension settings (d2.exe + c2t.exe pipeline) ----------
set TAU_D2_BTCUSD=4
set TAU_D2_ETHUSD=4
set TAU_D2_LTCUSD=4
set TAU_D2_XRPUSD=4
set TAU_D2_LINKUSD=18
set TAU_D2_DOGEUSD=2
set TAU_D2_ADAUSD=34
set W_D2_BTCUSD=4
set W_D2_ETHUSD=4
set W_D2_LTCUSD=4
set W_D2_XRPUSD=4
set W_D2_LINKUSD=18
set W_D2_DOGEUSD=2
set W_D2_ADAUSD=34
REM ---- Largest Lyapunov exponent (lyap_k) -----------------------------------
set TAU_LLE_BTCUSD=4
set TAU_LLE_ETHUSD=4
set TAU_LLE_LTCUSD=4
set TAU_LLE_XRPUSD=4
set TAU_LLE_LINKUSD=18
set TAU_LLE_DOGEUSD=2
set TAU_LLE_ADAUSD=34
REM ---- RQA (recurr) - tau and recurrence threshold radius -------------------
set TAU_RQA_BTCUSD=4
set TAU_RQA_ETHUSD=4
set TAU_RQA_LTCUSD=4
set TAU_RQA_XRPUSD=4
set TAU_RQA_LINKUSD=18
set TAU_RQA_DOGEUSD=2
set TAU_RQA_ADAUSD=34
set RAD_RQA_BTCUSD=0.005
set RAD_RQA_ETHUSD=0.005
set RAD_RQA_LTCUSD=0.005
set RAD_RQA_XRPUSD=0.005
set RAD_RQA_LINKUSD=0.005
set RAD_RQA_DOGEUSD=0.005
set RAD_RQA_ADAUSD=0.005

exit /b 0