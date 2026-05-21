"""RQA embedding, percentile radius, PyRQA metrics, and custom TREND."""

import logging

import numpy as np
from scipy.spatial.distance import pdist
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries

from hypothesis_config import (
    DEFAULT_RQA_RADIUS,
    RQA_EMBEDDING_DIM,
    RQA_KEYS,
    RQA_RADIUS_MAX_VECTORS,
    RQA_RADIUS_PERCENTILE_DEFAULT,
    RQA_RADIUS_SAMPLE_SEED,
)

logger = logging.getLogger(__name__)


def compute_percentile_radius(
    series,
    delay=1,
    m=RQA_EMBEDDING_DIM,
    percentile=RQA_RADIUS_PERCENTILE_DEFAULT,
    max_vectors=RQA_RADIUS_MAX_VECTORS,
    seed=RQA_RADIUS_SAMPLE_SEED,
):
    """Recurrence threshold as a percentile of pairwise Euclidean distances.

    Reconstructs the (m, delay) embedded state space and returns the requested
    percentile of pairwise Euclidean distances between embedded vectors. For
    long series the embedded matrix is randomly subsampled (without replacement,
    fixed RNG seed for reproducibility) down to ``max_vectors`` rows to keep
    pdist memory bounded.

    Returns NaN when the embedding cannot be built or when fewer than two
    embedded vectors are available.

    ``seed`` is intentionally a fixed default (``RQA_RADIUS_SAMPLE_SEED``) so
    repeat runs on the same series produce the same radius, and so the
    subsample comparison across coins / surrogates uses the same generator
    state. Callers that want a different RNG (e.g. sensitivity studies) can
    override it explicitly.
    """
    delay = int(delay) if delay and int(delay) > 0 else 1
    m = int(m) if m and int(m) > 0 else 1
    embedded = embed_series(series, delay=delay, m=m)
    if max_vectors and embedded.shape[0] > max_vectors:
        # pdist is quadratic in the number of vectors. Subsampling keeps memory
        # and runtime predictable while preserving a stable radius estimate.
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(embedded.shape[0], size=max_vectors, replace=False)
        embedded = embedded[sample_idx]
    if embedded.shape[0] < 2:
        return float("nan")
    distances = pdist(embedded, metric="euclidean")
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, percentile))


def embed_series(series, delay=1, m=RQA_EMBEDDING_DIM):
    """Construct a standard delay-embedding matrix from a scalar series.

    Row i is (x_i, x_{i+delay}, ..., x_{i+(m-1)delay}). Non-finite input values
    are dropped before embedding so downstream distance calculations do not
    propagate NaNs.
    """
    data = np.asarray(series, dtype=float)
    finite_mask = np.isfinite(data)
    if not finite_mask.all():
        data = data[finite_mask]
    delay = int(delay) if delay and int(delay) > 0 else 1
    m = int(m) if m and int(m) > 0 else 1
    span = (m - 1) * delay
    if data.size <= span:
        return np.empty((0, m), dtype=float)
    n_vec = data.size - span
    offsets = np.arange(m, dtype=np.int64) * delay
    return data[np.arange(n_vec, dtype=np.int64)[:, None] + offsets[None, :]]


def rqa_recurrence_matrix(series, delay, radius, m=RQA_EMBEDDING_DIM):
    """Build a boolean recurrence matrix for diagnostics.

    This helper is intentionally simple and explicit. It is not used for the
    core PyRQA metrics, but it is useful for manual checks and for code that
    needs a concrete recurrence matrix rather than PyRQA's internal structures.
    """
    embedded = embed_series(series, delay=delay, m=m)
    if embedded.shape[0] < 2:
        return np.zeros((0, 0), dtype=bool)
    n = embedded.shape[0]
    recurrence = np.zeros((n, n), dtype=bool)
    radius = float(radius)
    # Row-wise computation avoids constructing an N x N x m temporary array.
    for i in range(n):
        diff = embedded - embedded[i]
        recurrence[i, :] = np.sqrt(np.sum(diff * diff, axis=1)) <= radius
    return recurrence


def tisean_theiler_min_diagonal_k(theiler_w) -> int:
    """First diagonal index k to include when TISEAN uses ``-t W``.

    TISEAN ``d2.exe`` / ``lyap_k.exe`` exclude pairs with ``|i-j| <= W`` (strict
    ``> W`` for inclusion). PyRQA ``theiler_corrector`` excludes ``|i-j| < value``,
    so pass ``W+1`` there. For TREND, the loop over off-LOI diagonals must start
    at ``k = W+1``, not ``k = W``.
    """
    try:
        w = int(theiler_w)
    except (TypeError, ValueError):
        w = 0
    return max(1, w + 1)


def compute_rqa_trend(series, delay, radius, min_k=1, m=RQA_EMBEDDING_DIM):
    """RQA TREND as linear slope of recurrence density along off-LOI diagonals.

    For each diagonal distance k from the line of identity, combine the +k and
    -k diagonals and compute their recurrence density. TREND is the weighted
    slope of that density as a function of k. ``min_k`` is the smallest diagonal
    distance *included* (not TISEAN's ``-t W`` itself); use
    ``tisean_theiler_min_diagonal_k(W)`` to match ``d2.exe`` / ``lyap_k.exe``.
    """
    embedded = embed_series(series, delay=delay, m=m)
    n = embedded.shape[0]
    if n < 3:
        return np.nan
    try:
        min_k = int(min_k)
    except Exception:
        min_k = 1
    min_k = max(1, min_k)
    max_k = max(min_k + 1, n // 10)
    max_k = min(max_k, n - 1)
    if max_k <= min_k:
        return np.nan
    xs = []
    ys = []
    radius = float(radius)
    for k in range(min_k, max_k + 1):
        # The +k and -k diagonals contain the same distances for a symmetric
        # recurrence matrix, so computing embedded[:-k] vs embedded[k:] captures
        # the off-identity diagonal density without materializing the full matrix.
        diff = embedded[:-k] - embedded[k:]
        if diff.size == 0:
            continue
        recurrent = np.sqrt(np.sum(diff * diff, axis=1)) <= radius
        density = float(np.mean(recurrent))
        xs.append(float(k))
        ys.append(density)
    if len(xs) < 2:
        return np.nan
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    n_tilde = float(max_k)
    weights = n_tilde - xs_arr
    denom = float(np.sum(weights ** 2))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    ys_centered = ys_arr - ys_arr.mean()
    slope = float(np.sum(weights * ys_centered) / denom)
    return slope


def format_rqa_radius(radius):
    """Stable string representation used in BAT run IDs and CLI arguments.

    A fixed precision prevents tiny floating-point formatting differences from
    creating different output-folder names for the same effective radius.
    """
    if radius is None or not np.isfinite(radius):
        return f"{DEFAULT_RQA_RADIUS:.10g}"
    return f"{float(radius):.10g}"


def compute_pyrqa_metrics(series, delay, theiler, radius=None):
    """Compute PyRQA scalar metrics for the full series.

    RQA is intentionally not segmented in this pipeline. Each metric is one
    scalar for the whole original series.
    ``theiler`` is TISEAN's ``-t W`` from ``theilers_w.bat``. PyRQA excludes
    ``|i-j| < theiler_corrector``, so we pass ``W+1``. TREND uses the same rule
    via ``tisean_theiler_min_diagonal_k(W)`` (first diagonal ``k = W+1``).
    TREND is added manually because the installed PyRQA API does not expose the
    exact off-identity diagonal-density slope needed here.
    """
    r_eff = float(DEFAULT_RQA_RADIUS if radius is None else radius)
    try:
        w_eff = int(theiler) if theiler is not None else 0
    except (TypeError, ValueError):
        w_eff = 0
    if w_eff < 0:
        w_eff = 0
    try:
        ts = TimeSeries(series, embedding_dimension=RQA_EMBEDDING_DIM, time_delay=delay)
        pyrqa_theiler = tisean_theiler_min_diagonal_k(w_eff)
        settings = Settings(ts, analysis_type=Classic,
                            neighbourhood=FixedRadius(r_eff),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=pyrqa_theiler)
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        trend = compute_rqa_trend(
            series, delay=delay, radius=r_eff, min_k=pyrqa_theiler, m=RQA_EMBEDDING_DIM
        )
        return {
            "RR": float(result.recurrence_rate),
            "DET": float(result.determinism),
            "LAM": float(result.laminarity),
            "MAXLINE": float(result.longest_diagonal_line),
            "ENTR": float(result.entropy_diagonal_lines),
            "TT": float(result.trapping_time),
            "TREND": float(trend),
        }
    except Exception:
        # PyRQA can fail silently for too-short series, OpenCL/GPU issues, or
        # extreme radii. Log at exception level so bootstrap "no sd" outcomes
        # are diagnosable rather than mysteriously NaN. Cost is one stack
        # trace per failed series; turn down with logging.getLogger to WARNING.
        logger.exception(
            "PyRQA computation failed (N=%d, delay=%s, theiler=%s, radius=%s); "
            "returning NaN metrics.",
            len(series) if hasattr(series, "__len__") else -1,
            delay, theiler, radius,
        )
        return {k: np.nan for k in RQA_KEYS}
