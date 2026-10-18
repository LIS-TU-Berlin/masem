from typing import Literal, Sequence

import numpy as onp
from chex import Array
from scipy.stats import ttest_ind, trim_mean
from scipy.stats import sem, bootstrap


def welch_pvalue(
    x: onp.ndarray,
    y: onp.ndarray,
    side: Literal["two-sided", "less", "greater"] = "two-sided"
) -> float:
    x = onp.asarray(x, float)
    y = onp.asarray(y, float)
    x = x[onp.isfinite(x)]
    y = y[onp.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return onp.nan

    t_res = ttest_ind(x, y, equal_var=False, alternative=side)
    return t_res.pvalue  #, t_res.statistic


def compute_iqm(x: Array, trim_frac=0.25, axis: int = 0):
    flat = onp.asarray(x)
    return trim_mean(flat, trim_frac, axis=axis)


def mean_ci(
    x: Sequence,
    cval: float = 0.95,
    axis: int = 0,
    use_iqm: bool = False,
    method: Literal["bca", "percentile", "basic"] = "percentile",
):
    stat = onp.mean if not use_iqm else compute_iqm
    res = bootstrap(
        (x,),
        statistic=stat,
        confidence_level=cval,
        #n_resamples=10_000,
        method=method,
        random_state=0,
        axis=axis
    )
    mean = stat(x, axis=axis)
    serr = sem(x, axis=axis, ddof=1)

    return mean, serr, res.confidence_interval.low, res.confidence_interval.high
