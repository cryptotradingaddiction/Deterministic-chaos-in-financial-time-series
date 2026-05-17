# ---------------------------------------------------------------------------
# Surrogate and reference series generation
# ---------------------------------------------------------------------------
#
# The current test uses exactly one fully reshuffled series as the independent
# noise reference. Gaussian and Student-t series are computed only as descriptive
# benchmarks with the same first two moments as the original log-return series.
# The stationary bootstrap samples are produced in surrogate_sampling.py because
# they are reused as a standalone sampling primitive.
import numpy as np
from hypothesis_config import T_DOF

def generate_single_surrogate(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of the original series (randperm, no blocking).

    This destroys temporal ordering while preserving the empirical marginal
    distribution exactly. It is the "reshuffle" value in the TS denominator.
    """
    return rng.permutation(data)


def generate_normal_series(
    mu: float, sigma: float, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian white-noise reference with the original mean and SD."""
    return rng.normal(mu, sigma, n)


def generate_t_series(
    mu: float,
    sigma: float,
    n: int,
    rng: np.random.Generator,
    dof: float = T_DOF,
) -> np.ndarray:
    """t(dof) series scaled to (mu, sigma).

    t(dof) has zero mean and variance dof/(dof-2) for dof>2, so
    scale raw draws by sigma / sqrt(dof/(dof-2)) and shift by mu.
    """
    t_raw = rng.standard_t(dof, n)
    t_sd = np.sqrt(dof / (dof - 2.0))
    return mu + sigma * (t_raw / t_sd)
