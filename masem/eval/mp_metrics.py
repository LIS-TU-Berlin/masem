import jax
import jax.numpy as jnp
import numpy as onp
from chex import Array, PRNGKey

from masem.constants import FEAS_THRESH
from masem.distributions.motion_planning.motion_planning import MotionPlanning
from masem.eval.performance_metrics import approx_entropy


def trajectory_entropy(
    rng: PRNGKey,
    x: Array,
    dist: MotionPlanning,
    tol: float = FEAS_THRESH,
    k_range: tuple = (2, 4, 8),
    n: int = 300,
    draws: int = 5
) -> float:
    """KNN entropy of feasible trajectories in B-spline path space."""
    task_dim = 2
    feasible_mask = dist.feasible(x, tol=tol)
    n_feas = int(feasible_mask.sum())
    if n_feas < 10:
        return 0.
    x_feas = x[feasible_mask]
    waypts = x_feas.reshape(n_feas, -1, task_dim)
    knots = dist.get_knots(waypts)
    paths = dist.Phi @ knots
    paths_flat = paths.reshape(n_feas, -1)

    sub_n = min(n, n_feas)
    rngs = jax.random.split(rng, draws)
    ents = []
    for r in rngs:
        samp = jax.random.choice(r, paths_flat, shape=(sub_n,), replace=False)
        ents.append(approx_entropy(r, samp, k_range, n=sub_n))
    return float(jnp.asarray(ents).mean())


def grid_coverage(x: Array, dist: MotionPlanning, n_cells: int = 50, tol: float = FEAS_THRESH):
    """Grid-based 2D coverage of feasible trajectories.

    Returns (coverage_fraction, coverage_entropy):
      coverage_fraction – fraction of n_cells x n_cells cells visited by at least
                          one feasible trajectory.
      coverage_entropy  – entropy of per-cell trajectory-count distribution,
                          normalized by log(n_occupied) so 1.0 = perfectly uniform
                          across all visited cells.
    """
    task_dim = 2
    feasible_mask = dist.feasible(x, tol=tol)
    n_feas = int(feasible_mask.sum())
    if n_feas == 0:
        return 0., 0.

    waypts = x[feasible_mask].reshape(n_feas, -1, task_dim)
    knots = dist.get_knots(waypts)
    paths = onp.array(dist.Phi @ knots)  # (n_feas, n_eval, task_dim)
    n_eval = paths.shape[1]

    lo, hi = float(dist.bounds[0]), float(dist.bounds[1])
    pts = paths.reshape(-1, 2)
    xi = onp.clip(((pts[:, 0] - lo) / (hi - lo) * n_cells).astype(int), 0, n_cells - 1)
    yi = onp.clip(((pts[:, 1] - lo) / (hi - lo) * n_cells).astype(int), 0, n_cells - 1)
    cell_ids = xi * n_cells + yi  # (n_feas * n_eval,)

    total_cells = n_cells * n_cells
    counts = onp.zeros(total_cells, dtype=float)
    for i in range(n_feas):
        cells_i = onp.unique(cell_ids[i * n_eval:(i + 1) * n_eval])
        counts[cells_i] += 1.0

    n_occupied = int((counts > 0).sum())
    coverage_fraction = n_occupied / total_cells

    p = counts[counts > 0]
    p = p / p.sum()
    raw_entropy = float(-onp.sum(p * onp.log(p + 1e-10)))
    coverage_entropy = raw_entropy / onp.log(n_occupied + 1e-10) if n_occupied > 1 else 0.

    return coverage_fraction, float(coverage_entropy)


def path_ht_entropy(x: Array, dist: MotionPlanning, tol: float = 1e-3):
    feasible_mask = dist.feasible(x, tol=tol)
    if int(feasible_mask.sum()) == 0:
        return 0.
    ht_class = dist.path_homotopy_class(x[feasible_mask])
    p = jnp.bincount(ht_class)
    p = p / p.sum()
    ent = jnp.sum(jnp.where(p > 0, -p * jnp.log(p), 0.))

    return ent
