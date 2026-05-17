@echo off
REM Extra hypothesis.py CLI flags from environment (include after setlocal EnableDelayedExpansion).
REM Desktop sets DCH_BOOTSTRAP_SAMPLES; optional DCH_STATIONARY_BLOCK_MEAN (0 = sqrt(n)).
set "DCH_HYP_EXTRA="
if defined DCH_BOOTSTRAP_SAMPLES set "DCH_HYP_EXTRA=!DCH_HYP_EXTRA! --bootstrap_samples !DCH_BOOTSTRAP_SAMPLES!"
if defined DCH_STATIONARY_BLOCK_MEAN set "DCH_HYP_EXTRA=!DCH_HYP_EXTRA! --stationary_block_mean !DCH_STATIONARY_BLOCK_MEAN!"
