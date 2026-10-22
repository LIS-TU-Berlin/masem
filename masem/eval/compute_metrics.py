import jax
import jax.numpy as jnp
import numpy as onp
from chex import Array, PRNGKey
from tqdm import tqdm

from masem.constants import FEAS_THRESH
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.eval.mp_metrics import grid_coverage, path_ht_entropy, trajectory_entropy
from masem.eval.performance_metrics import (
    approx_entropy,
    approx_feasible_entropy,
    constraint_violations,
    energy_distance,
    expected_K_error,
    get_multi_k_ess,
    moment_errors,
    pwd_hist_KL,
    sinkhorn_divergence,
)
from masem.utils import linear_project


def get_metrics_fn(dist_name: str):
    if dist_name in ("uniform_ball40", "sine", "seven_lobes", "swiss_roll",
                     "uniform_ball2_connected", "uniform_ball2_disconnected") \
                     or dist_name.startswith("embedded_disks_nonlin"):
        return compute_metrics_2d
    elif dist_name.startswith("motion_planning"):
        return compute_metrics_motion_plan
    elif dist_name in ("embedded_disks", "disconnected_stress", "disconnected_stress_2d", "highdim_stress"):
        return compute_metrics_scaling
    else:
        return compute_minimum_metrics


def compute_metrics_2d(
    rng: PRNGKey,
    steps: Array,   # x samples
    dist: ConstrainedDistribution,
    n_gt: int|None = None,
    feas_tol: float = FEAS_THRESH,
    pwd_max_dist: float|None = None,
    ess_k: list | tuple = (4, 8, 16),
    mu_ess: float | None = None
) -> dict:
    ess_k = jnp.asarray(ess_k)
    n_steps = steps.shape[0]
    suffixes = ["_raw", "_proj"]
    metrics = {
        k + s: onp.empty(n_steps) for k in
         ("n_feas", "slack", "sd", "ed", "fm", "sm", "entropy", "feasible_entropy", "kl") for s in suffixes
    }
    for s in suffixes:
        for k in ess_k:
            metrics[f"ess_{k}" + s] = onp.empty(n_steps)
            metrics[f"ess_raw_{k}" + s] = onp.empty(n_steps)
            metrics[f"ess_w_{k}" + s] = onp.empty(n_steps)

    mu_ess = mu_ess if mu_ess is not None else 1e3 / dist.dim
    comp_ess_fn = get_multi_k_ess(dist, feas_tol=feas_tol, mu_slack=mu_ess)

    # custom_metrics = dist.custom_metrics
    # metrics.update({k + s: onp.empty(n_steps) for k in custom_metrics for s in suffixes})
    if pwd_max_dist is None:
        pwd_max_dist = float(dist.bounds[1] - dist.bounds[0])

    rngs = jax.random.split(rng, n_steps + 1)
    if n_gt is None:
        n_gt = steps.shape[1]
    try:
        gt_samples = dist.sample(rngs[0], n_samples=n_gt, load=True)
    except:
        print("Failed to load ground truth samples, generating new ones.")
        gt_samples = dist.sample(rngs[0], n_samples=n_gt, load=False)

    for i in tqdm(range(n_steps), colour="blue", desc="Computing Metrics"):
        x_raw = steps[i]
        x_proj = linear_project(dist, x_raw)

        for suffix, x in (("_raw", x_raw), ("_proj", x_proj)):
            metrics["n_feas" + suffix][i] = dist.feasible(x, tol=feas_tol).sum()
            metrics["slack" + suffix][i] = constraint_violations(dist, x)

            metrics["sd" + suffix][i] = sinkhorn_divergence(gt_samples, x)
            metrics["entropy" + suffix][i] = approx_entropy(rngs[i], x)
            metrics["feasible_entropy" + suffix][i] = approx_feasible_entropy(rngs[i], x, dist, tol=feas_tol)
            metrics["kl" + suffix][i] = pwd_hist_KL(gt_samples, x, pwd_max_dist)

            metrics["ed" + suffix][i] = energy_distance(gt_samples, x)[1]

            # ess, ess_raw, ess_w = comp_ess_fn(x)
            # for j, k in enumerate(ess_k):
            #     metrics[f"ess_{k}" + suffix][i] = ess[j]
            #     metrics[f"ess_raw_{k}" + suffix][i] = ess_raw[j]
            #     metrics[f"ess_w_{k}" + suffix][i] = ess_w[j]

            fm, sm = moment_errors(gt_samples, x)
            metrics["fm" + suffix][i] = fm
            metrics["sm" + suffix][i] = sm

            # for k, v in custom_metrics.items():
            #     metrics[k + suffix][i] = v(x)

    return metrics


def compute_metrics_scaling(
    rng: PRNGKey,
    steps: Array,   # x samples
    dist: ConstrainedDistribution,
    n_gt: int = 2_000,
    feas_tol: float = FEAS_THRESH,
    pwd_max_dist: float|None = None,
    ess_k: list | tuple = (4, 8, 16),
    mu_ess: float | None = None
) -> dict:
    ess_k = jnp.asarray(ess_k)
    n_steps = steps.shape[0]
    suffixes = ["_raw", "_proj"]
    metrics = {
        k + s: onp.empty(n_steps) for k in
         ("n_feas", "slack", "sd", "ed", "fm", "sm",
          "entropy", "feasible_entropy", "kl", "exp_K", "comp_kl", "comp_tv") for s in suffixes
    }
    for s in suffixes:
        for k in ess_k:
            metrics[f"ess_{k}" + s] = onp.empty(n_steps)
            metrics[f"ess_raw_{k}" + s] = onp.empty(n_steps)
            metrics[f"ess_w_{k}" + s] = onp.empty(n_steps)

    mu_ess = mu_ess if mu_ess is not None else 1e3 / dist.dim
    comp_ess_fn = get_multi_k_ess(dist, feas_tol=feas_tol, mu_slack=mu_ess)
    if pwd_max_dist is None:
        pwd_max_dist = float(dist.bounds[1] - dist.bounds[0])

    rng_k, *rngs = jax.random.split(rng, n_steps + 2)
    gt_samples = dist.sample(rngs[0], n_samples=n_gt, load=True)
    true_w = dist.estimate_component_weights(rng_k, n_mc=500_000)

    for i in tqdm(range(n_steps), colour="blue", desc="Computing Metrics"):
        x_raw = steps[i]
        x_proj = linear_project(dist, x_raw)

        for suffix, x in (("_raw", x_raw), ("_proj", x_proj)):
            metrics["n_feas" + suffix][i] = dist.feasible(x, tol=feas_tol).sum()
            metrics["slack" + suffix][i] = constraint_violations(dist, x)

            metrics["sd" + suffix][i] = sinkhorn_divergence(gt_samples, x)
            metrics["entropy" + suffix][i] = approx_entropy(rngs[i], x)
            metrics["feasible_entropy" + suffix][i] = approx_feasible_entropy(rngs[i], x, dist, tol=feas_tol)
            metrics["kl" + suffix][i] = pwd_hist_KL(gt_samples, x, pwd_max_dist)

            metrics["ed" + suffix][i] = energy_distance(gt_samples, x)[1]

            fm, sm = moment_errors(gt_samples, x)
            metrics["fm" + suffix][i] = fm
            metrics["sm" + suffix][i] = sm

            metrics["exp_K" + suffix][i] = expected_K_error(gt_samples, x)

            kl, tv = dist.component_kl_tv(x, true_weights=true_w)
            metrics["comp_kl" + suffix][i] = kl
            metrics["comp_tv" + suffix][i] = tv

            ess, ess_raw, ess_w = comp_ess_fn(x)
            for j, k in enumerate(ess_k):
                metrics[f"ess_{k}" + suffix][i] = ess[j]
                metrics[f"ess_raw_{k}" + suffix][i] = ess_raw[j]
                metrics[f"ess_w_{k}" + suffix][i] = ess_w[j]

    return metrics


def compute_metrics_motion_plan(
    rng: PRNGKey,
    steps: Array,   # x samples
    dist: ConstrainedDistribution,
    n_gt: int = 2_000,
    feas_tol: float = FEAS_THRESH,
    pwd_max_dist: float|None = None,
    ess_k: list | tuple = (4, 8, 16),
    mu_ess: float | None = None
) -> dict:
    ess_k = jnp.asarray(ess_k)
    n_steps = steps.shape[0]
    suffixes = ["_raw", "_proj"]
    metrics = {
        k + s: onp.empty(n_steps) for k in
        ("n_feas", "slack", "entropy", "feasible_entropy", "traj_entropy", "grid_cov", "grid_cov_entropy", "path_ht_entropy")
        for s in suffixes
    }
    for s in suffixes:
        for k in ess_k:
            metrics[f"ess_{k}" + s] = onp.empty(n_steps)
            metrics[f"ess_raw_{k}" + s] = onp.empty(n_steps)
            metrics[f"ess_w_{k}" + s] = onp.empty(n_steps)

    mu_ess = mu_ess if mu_ess is not None else 1e3 / dist.dim
    comp_ess_fn = get_multi_k_ess(dist, feas_tol=feas_tol, mu_slack=mu_ess)

    rngs = jax.random.split(rng, n_steps + 1)

    for i in tqdm(range(n_steps), colour="blue", desc="Computing Metrics"):
        x_raw = steps[i]
        x_proj = linear_project(dist, x_raw)

        for suffix, x in (("_raw", x_raw), ("_proj", x_proj)):
            metrics["n_feas" + suffix][i] = dist.feasible(x, tol=feas_tol).sum()
            metrics["slack" + suffix][i] = constraint_violations(dist, x)
            metrics["entropy" + suffix][i] = approx_entropy(rngs[i], x)
            metrics["feasible_entropy" + suffix][i] = approx_feasible_entropy(rngs[i], x, dist, tol=feas_tol)
            metrics["traj_entropy" + suffix][i] = trajectory_entropy(rngs[i], x, dist, tol=feas_tol)
            cov_frac, cov_ent = grid_coverage(x, dist, tol=feas_tol)
            metrics["grid_cov" + suffix][i] = cov_frac
            metrics["grid_cov_entropy" + suffix][i] = cov_ent
            metrics["path_ht_entropy" + suffix][i] = path_ht_entropy(x, dist)
            ess, ess_raw, ess_w = comp_ess_fn(x)
            for j, k in enumerate(ess_k):
                metrics[f"ess_{k}" + suffix][i] = ess[j]
                metrics[f"ess_raw_{k}" + suffix][i] = ess_raw[j]
                metrics[f"ess_w_{k}" + suffix][i] = ess_w[j]

    return metrics


def compute_minimum_metrics(
    rng: PRNGKey,
    steps: Array,   # x samples
    dist: ConstrainedDistribution,
    n_gt: int = 2_000,
    feas_tol: float = FEAS_THRESH,
) -> dict:
    n_steps = steps.shape[0]
    suffixes = ["_raw", "_proj"]
    metrics = {
        k + s: onp.empty(n_steps) for k in
        ("n_feas", "slack", "entropy", "feasible_entropy") for s in suffixes
    }

    rngs = jax.random.split(rng, n_steps + 1)

    for i in tqdm(range(n_steps), colour="blue", desc="Computing Metrics"):
        x_raw = steps[i]
        x_proj = linear_project(dist, x_raw)

        for suffix, x in (("_raw", x_raw), ("_proj", x_proj)):
            metrics["n_feas" + suffix][i] = dist.feasible(x, tol=feas_tol).sum()
            metrics["slack" + suffix][i] = constraint_violations(dist, x)
            metrics["entropy" + suffix][i] = approx_entropy(rngs[i], x)
            metrics["feasible_entropy" + suffix][i] = approx_feasible_entropy(rngs[i], x, dist, tol=feas_tol)

    return metrics
