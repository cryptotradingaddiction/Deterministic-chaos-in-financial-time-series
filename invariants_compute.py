"""Dispatch invariant computation for one in-memory series."""

import glob
import logging
import os
import subprocess

import numpy as np

from hypothesis_config import (
    DEFAULT_RQA_RADIUS,
    RQA_EMBEDDING_DIM,
    RQA_KEYS,
    RQA_RADIUS_PERCENTILE_DEFAULT,
)
from invariants_correlation import compute_ellner_from_c2, extract_takens_plateau
from invariants_lyapunov import extract_lle_mean_std
from invariants_rqa import compute_percentile_radius, compute_pyrqa_metrics
from tisean_io import run_c2t, run_d2, run_lyap_k

logger = logging.getLogger(__name__)


def compute_invariants(series_array, output_dir, label, delay, theiler,
                       metric_names, rqa_radius=None, series_std_fallback=np.nan,
                       rqa_radius_mode="percentile",
                       rqa_percentile=RQA_RADIUS_PERCENTILE_DEFAULT,
                       rqa_radius_log=None):
    """Compute invariants for an in-memory series. Returns (mean_dict, sd_dict, n_dict).

    Value / SD / n sources:
      TAKENS — plateau mean of the c2t Takens-Theiler curve d_2^(T)(r') for
               embedding m=3. Its plateau end-points also define r_min/r_max.
      ELLNER — Ellner extension (eq. 8.78) evaluated on .c2 between the same
               r_min and r_max. The reported SD/n come from the Takens plateau
               dispersion and serve as an orientation for the interval quality.
      LLE   — median lyap_k S(t) slope across epsilon blocks at m=3
      RQA   — one metric value computed on the full time series

    For RQA, the recurrence radius is selected dynamically when
    `rqa_radius_mode == 'percentile'` (default) as `rqa_percentile`-th percentile
    of pairwise Euclidean distances between embedded state vectors. If the
    percentile calculation cannot be performed, the function falls back to the
    explicit `rqa_radius` value (or `DEFAULT_RQA_RADIUS` if that is also missing).
    """
    metric_names = list(metric_names)

    # Work out which external tools are actually needed. This matters because
    # hypothesis.py is called many times during stationary bootstrap; skipping
    # unused executables saves a lot of runtime.
    need_takens = "TAKENS" in metric_names
    need_ellner = "ELLNER" in metric_names
    need_lle = "LLE" in metric_names
    need_rqa = any(k in metric_names for k in RQA_KEYS)

    # Every in-memory pseudo-series is written to a temporary .dat file because
    # TISEAN tools are command-line programs that operate on files, not arrays.
    prefix = os.path.join(output_dir, label)
    data_file = prefix + ".dat"
    np.savetxt(data_file, series_array)

    # The three output dictionaries share the same metric keys:
    #   out     -> point estimate used in comparisons,
    #   out_std -> within-curve/plateau SD where meaningful,
    #   out_n   -> number of values behind that SD.
    #
    # For bootstrap TS decisions, the important SD is later computed across the
    # B bootstrap point estimates, not from out_std.
    out = {k: np.nan for k in metric_names}
    out_std = {k: np.nan for k in metric_names}
    out_n = {k: 0 for k in metric_names}

    try:
        d2_file = h2_file = c2_file = None
        if need_takens or need_ellner:
            # Dimension metrics share one d2.exe call. We keep TISEAN's default
            # radius scan so c2t can build the full Takens curve; the practical
            # scale choice is then made by plateau selection on d_2^(T)(r').
            d2_file, h2_file, c2_file = run_d2(
                data_file, delay, theiler, prefix,
            )
        if (need_takens or need_ellner) and c2_file:
            # One c2t run supplies both TAKENS and ELLNER. TAKENS uses the
            # plateau mean directly. ELLNER reuses the plateau endpoints and
            # integrates the original .c2 correlation integral over that interval.
            #
            # Book mapping in this block:
            #   1. run_c2t() gives the Takens curve d_2^(T)(r') from (8.75)-(8.76).
            #   2. extract_takens_plateau() implements the recommended plateau
            #      search in d_2^(T)(r') vs ln(r') after (8.77).
            #   3. compute_ellner_from_c2() applies Ellner's finite-interval
            #      correction (8.78) on the same scaling region.
            takens_file = prefix + "_takens.dat"
            run_c2t(c2_file, takens_file)
            takens_mean, takens_sd, n_val, r_min, r_max = extract_takens_plateau(takens_file)
            if need_takens:
                out["TAKENS"] = takens_mean
                out_std["TAKENS"] = takens_sd
                out_n["TAKENS"] = int(n_val)
            if np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
                ellner = compute_ellner_from_c2(c2_file, r_min, r_max)
            else:
                ellner = np.nan
            if need_ellner:
                out["ELLNER"] = ellner
                out_std["ELLNER"] = takens_sd
                out_n["ELLNER"] = int(n_val) if np.isfinite(ellner) else 0
        if need_lle:
            # LLE is one scalar slope estimate. It is recomputed for
            # original/reference/bootstrap series when LLE is selected.
            #
            # Book mapping:
            #   run_lyap_k() estimates S(t), the averaged logarithmic divergence
            #   from (8.95), under the exponential-separation model (8.94).
            #   extract_lle_mean_std() then finds the linear part of S(t) and
            #   returns its slope as lambda_max.
            lyap_file = prefix + "_lyap.txt"
            run_lyap_k(data_file, delay, theiler, lyap_file)
            mu, sg, nn = extract_lle_mean_std(lyap_file)
            out["LLE"], out_std["LLE"] = mu, sg
            out_n["LLE"] = nn
        if need_rqa:
            # ``percentile``: recompute r on each series (bootstrap locks orig's r).
            # ``fixed``: use ``--rqa_radius`` (RQA.bat passes r from rqa_radius.py).
            r_eff = None
            source = "fixed"
            if str(rqa_radius_mode).lower() == "percentile":
                r_eff = compute_percentile_radius(
                    series_array,
                    delay=delay,
                    m=RQA_EMBEDDING_DIM,
                    percentile=rqa_percentile,
                )
                source = f"percentile({rqa_percentile:g}%)"
                if not (np.isfinite(r_eff) and r_eff > 0.0):
                    r_eff = None
                    source = "fixed (percentile failed)"
            if r_eff is None:
                r_eff = (
                    float(rqa_radius)
                    if rqa_radius is not None and np.isfinite(rqa_radius) and rqa_radius > 0.0
                    else float(DEFAULT_RQA_RADIUS)
                )
                if str(rqa_radius_mode).lower() != "percentile":
                    source = (
                        "fixed (--rqa_radius)"
                        if rqa_radius is not None and np.isfinite(rqa_radius) and rqa_radius > 0.0
                        else "fixed (default)"
                    )
            print(
                f"     RQA radius for label={label}: r={r_eff:.6g}  "
                f"({source}, m={RQA_EMBEDDING_DIM}, tau={delay})"
            )
            if isinstance(rqa_radius_log, dict):
                rqa_radius_log[label] = {"radius": float(r_eff), "source": source}
            rqa_values = compute_pyrqa_metrics(series_array, delay, theiler, r_eff)
            for k in RQA_KEYS:
                if k in out:
                    value = float(rqa_values.get(k, np.nan))
                    out[k] = value
                    out_std[k] = np.nan
                    out_n[k] = 1 if np.isfinite(value) else 0
    except subprocess.CalledProcessError:
        logger.exception("TISEAN failed for label=%s", label)
    except Exception:
        logger.exception("Error computing invariants for label=%s", label)
    finally:
        # Each bootstrap iteration writes several temporary files. Remove every
        # file sharing the prefix so long runs do not fill the result directory.
        for tmp in glob.glob(prefix + "*"):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return out, out_std, out_n
