from functools import partial

import jax
from chex import Array, PRNGKey
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence as ott_sd
from scipy.special import digamma, gammaln

from masem.constants import FEAS_THRESH
from masem.distributions.constrained_dist import ConstrainedDistribution


def _pwd_chunked(
    X: Array,
    Y: Array,
    chunk_size: int,
) -> Array:
    n, d = X.shape
    n_chunks = (n + chunk_size - 1) // chunk_size
    pad = n_chunks * chunk_size - n

    X_pad = jnp.pad(X, ((0, pad), (0, 0)))

    Y2 = jnp.sum(Y * Y, axis=1)[None, :]

    def body(carry, chunk_idx):
        start = chunk_idx * chunk_size
        Xc = jax.lax.dynamic_slice(X_pad, (start, 0), (chunk_size, d))

        valid = start + jnp.arange(chunk_size) < n

        X2 = jnp.sum(Xc * Xc, axis=1)[:, None]
        D2 = X2 + Y2 - 2.0 * (Xc @ Y.T)

        D = jnp.sqrt(jnp.maximum(D2, 0.0))
        D = jnp.where(valid[:, None], D, 0.0)

        return carry + D.sum(), None

    total, _ = jax.lax.scan(
        body,
        jnp.array(0.0, dtype=X.dtype),
        jnp.arange(n_chunks),
    )

    return total


def get_multi_k_ess(
    dist: ConstrainedDistribution,
    feas_tol: float,
    ks: tuple[int, ...] = (4, 8, 16),
    knn_tau: float = 1.0,
    knn_eps: float = 1e-6,
    use_kmean: bool = False,
    mu_slack: float = 1e3,
):
    import numpy as onp

    ks = onp.asarray(ks, dtype=int)

    def _knn_log_radii(pts: Array, valid_ks: Array) -> Array:
        """kNN log-radius log-weights of `pts` for each k in `valid_ks`.

        Returns array of shape (n_pts, len(valid_ks)).
        """
        n_pts = pts.shape[0]
        k_max = int(valid_ks.max())

        # compute pwd
        x2 = jnp.sum(pts**2, axis=1, keepdims=True)
        d2 = jnp.maximum(x2 + x2.T - 2.0 * pts @ pts.T, 0.0)
        d2 = d2.at[jnp.diag_indices(n_pts)].set(jnp.inf)
        nearest_d2 = -jax.lax.top_k(-d2, k_max)[0]
        nearest_r = jnp.sqrt(nearest_d2)

        if use_kmean:
            radius_scores = (
                jnp.cumsum(nearest_r, axis=1) / jnp.arange(
                    1,
                    k_max + 1,
                    dtype=pts.dtype,
                )[None, :]
            )
        else:
            radius_scores = nearest_r

        selected_radii = radius_scores[:, jnp.asarray(valid_ks - 1)]
        return knn_tau * jnp.log(jnp.maximum(selected_radii, knn_eps))

    def ess_fn(x: Array) -> tuple[Array, Array, Array]:
        n_total = x.shape[0]
        zeros = jnp.zeros(len(ks), dtype=x.dtype)

        # --- feasible ESS: reverse ESS on the strictly feasible subset ---
        feasible_mask = dist.feasible(x, tol=feas_tol)
        n_feas = int(feasible_mask.sum())
        raw_ess = zeros
        adjusted_ess = zeros

        # Need at least k + 1 points because self is excluded.
        valid_idx = onp.flatnonzero(ks < n_feas)
        if len(valid_idx) > 0:
            feasible_x = x[feasible_mask]
            log_weights = _knn_log_radii(feasible_x, ks[valid_idx])
            raw_valid = jax.vmap(
                compute_reverse_ess,
                in_axes=1,
                out_axes=0,
            )(log_weights)
            raw_ess = raw_ess.at[jnp.asarray(valid_idx)].set(raw_valid)
            adjusted_ess = (n_feas / n_total) * raw_ess

        # --- slack-weighted ESS: keep all samples, softly downweight
        # infeasible ones via the slack penalty used in knn_nhr.resample. ---
        weighted_ess = zeros
        w_valid_idx = onp.flatnonzero(ks < n_total)
        if len(w_valid_idx) > 0:
            log_weights = _knn_log_radii(x, ks[w_valid_idx])
            slack_penalty = jnp.sum(dist.slack_vec(x) ** 2, axis=-1)
            log_weights = log_weights - mu_slack * slack_penalty[:, None]
            w_valid = jax.vmap(
                compute_reverse_ess,
                in_axes=1,
                out_axes=0,
            )(log_weights)
            weighted_ess = weighted_ess.at[jnp.asarray(w_valid_idx)].set(w_valid)

        return adjusted_ess, raw_ess, weighted_ess

    return ess_fn


def compute_reverse_ess(log_weights):
    """Taken from DBS paper source:

    https://github.com/DenisBless/UnderdampedDiffusionBridges/blob/master/algorithms/common/eval_methods/utils.py
    """
    # Subtract the maximum log weight for numerical stability
    n_weights = log_weights.shape[0]
    max_log_weight = jnp.max(log_weights)
    stable_log_weights = log_weights - max_log_weight

    # Compute the importance weights in a numerically stable way
    is_weights = jnp.exp(stable_log_weights)

    # Compute the sums needed for ESS
    sum_is_weights = jnp.sum(is_weights)
    sum_is_weights_squared = jnp.sum(is_weights ** 2)

    # Calculate the effective sample size (ESS)
    ess = (sum_is_weights ** 2) / (n_weights * sum_is_weights_squared)

    return ess


@jax.jit
def sinkhorn_divergence(gt_samples: Array, samples: Array):
    """Taken from SCLD paper:

    https://github.com/anonymous3141/SCLD/tree/master
    """
    gt_samples = jnp.astype(gt_samples, samples.dtype)
    geom = pointcloud.PointCloud(gt_samples, samples, epsilon=5e-3)
    sd, _ = ott_sd.sinkhorn_divergence(
        geom,
        x=geom.x,
        y=geom.y,
    )

    return sd


@partial(jax.jit, static_argnums=(2, 3, 4))
def energy_distance(
    gt_samples: Array,
    samples: Array,
    unbiased: bool = True,
    eps: float = 1e-12,
    chunk_size: int = 512,
):
    X = gt_samples.astype(samples.dtype)
    Y = jnp.asarray(samples)

    n = X.shape[0]
    m = Y.shape[0]

    sum_xy = _pwd_chunked(X, Y, chunk_size)
    sum_xx = _pwd_chunked(X, X, chunk_size)
    sum_yy = _pwd_chunked(Y, Y, chunk_size)

    n_f = jnp.asarray(n, dtype=Y.dtype)
    m_f = jnp.asarray(m, dtype=Y.dtype)

    term_xy = 2.0 * sum_xy / (n_f * m_f)

    if unbiased:
        # Diagonal self-distances are already exactly zero, so no correction needed.
        term_xx = sum_xx / (n_f * (n_f - 1.0))
        term_yy = sum_yy / (m_f * (m_f - 1.0))
    else:
        term_xx = sum_xx / (n_f * n_f)
        term_yy = sum_yy / (m_f * m_f)

    ed2_raw = term_xy - term_xx - term_yy
    # The unbiased estimator may be negative for finite samples.
    # Clamp only for the displayed distance, not for the returned squared estimate.
    ed = jnp.sqrt(jnp.maximum(ed2_raw, 0.0) + eps)

    return ed2_raw, ed

@jax.jit
def moment_errors(gt_samples: Array, samples: Array) -> tuple[Array, Array]:
    def get_cov(x):
        centered = x - jnp.mean(x, axis=0)
        return jnp.dot(centered.T, centered) / (x.shape[0] - 1)

    mu_x = jnp.mean(samples, axis=0)
    mu_gt = jnp.mean(gt_samples, axis=0)
    cov_x = get_cov(samples)
    cov_gt = get_cov(gt_samples)
    first_moment_error = jnp.linalg.norm(mu_x - mu_gt)
    second_moment_error = jnp.linalg.norm(cov_x - cov_gt, ord='fro')

    return first_moment_error, second_moment_error


@partial(jax.jit, static_argnums=(0,))
def constraint_violations(dist: ConstrainedDistribution, samples: Array):
    """Expected max slack as mean max violation over all samples."""
    ineq_cons = dist.g(samples).clip(min=0.)
    eq_cons = dist.h(samples)
    max_g = jnp.max(ineq_cons, axis=-1) if ineq_cons.size > 0 else 0.
    max_h = jnp.max(jnp.abs(eq_cons), axis=-1) if eq_cons.size > 0 else 0.
    max_v = jnp.maximum(max_g, max_h)
    #nc = ineq_cons.size + eq_cons.size
    #eq_v = jnp.sum(ineq_cons) + jnp.abs(eq_cons).sum()
    #return 0. if nc == 0. else eq_v / nc
    return max_v.mean()


@partial(jax.jit, static_argnames=["n", "k", "d"])
def knn_entropy(rng: PRNGKey, samples: Array, k: int = 5, n: int = 1000, d: int|None = None) -> Array:
    """KL-Estimator for entropy."""
    if samples.shape[0] > n:
        samples = jax.random.choice(rng, samples, shape=(n,), replace=False)

    n_eff = samples.shape[0]
    if d is None:
        d = samples.shape[1]
    dists = jnp.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)

    k_nearest = -jax.lax.top_k(-dists, k + 1)[0][:, -1]
    mean_log_r = jnp.mean(jnp.log(k_nearest + 1e-10))
    log_unit_ball_volume = (d / 2) * jnp.log(jnp.pi) - gammaln(d / 2 + 1)

    return digamma(n_eff) - digamma(k) + log_unit_ball_volume + d * mean_log_r


def pwd_hist(samples: Array, vmin=0.0, vmax=2.0, n_bins=50):
    pairwise_sq = jnp.sum((samples[:, None] - samples[None, :]) ** 2, axis=-1)
    pairwise_dists = jnp.sqrt(jnp.maximum(pairwise_sq, 0.0))   # underflow protection for bad runs
    pairwise_dists = pairwise_dists[jnp.triu_indices(len(samples), k=1)]
    # put out of range values into edge bins for vmax or vmin
    pairwise_dists = jnp.clip(pairwise_dists, vmin, jnp.nextafter(vmax, vmin))

    hist, _ = jnp.histogram(
        pairwise_dists,
        bins=n_bins,
        range=(vmin, vmax),
        density=True,
    )
    return hist


@partial(jax.jit, static_argnums=(3,))
def pwd_hist_KL(gt: Array, samples: Array, vmax: float, n_bins: int = 50):
    gt_hist = pwd_hist(gt, vmin=0.0, vmax=vmax, n_bins=n_bins)
    samples_hist = pwd_hist(samples, vmin=0.0, vmax=vmax, n_bins=n_bins)
    return KL_hist_forward(gt_hist, samples_hist, vmax / n_bins)  # fwd kl better to detect mode loss


def KL_hist_forward(p_true: Array, q_approx: Array, dx: float, eps: float = 1e-10):
    return jnp.sum(
        jnp.where(
            p_true > 0,
            p_true * jnp.log(p_true / (q_approx + eps)),
            0.0,
        )
    ) * dx


def KL_hist_reverse(q_approx: Array, p_true: Array, dx: float, eps: float = 1e-10):
    return jnp.sum(
        jnp.where(
            q_approx > 0,
            q_approx * jnp.log(q_approx / (p_true + eps)),
            0.0,
        )
    ) * dx


@partial(jax.jit, static_argnames=["k_range", "n", "d"])
def approx_entropy(rng, x, k_range: tuple = (2, 4, 8), n=1000, d: int|None = None):
    rngs = jax.random.split(rng, len(k_range))
    # return jax.vmap(lambda r, k: knn_entropy(r, x, k))(rngs, k_range)
    ents = jnp.asarray([knn_entropy(r, x, k, n=n, d=d) for (r, k) in zip(rngs, k_range)])
    return ents.mean()


def approx_feasible_entropy(
    rng,
    x,
    dist: ConstrainedDistribution,
    k_range: tuple = (2, 4, 8),
    n=100,
    tol=FEAS_THRESH,
    draws=10,
    d: int|None = None,
    offset: float = 0
):
    feasible_mask = dist.feasible(x, tol=tol)
    n_feas = int(feasible_mask.sum())
    if n_feas < n:        # big problem with KL-estimator: depends on n, meaning you cant really compare entropy across different n.
        return jnp.nan    # so we should ensure we use the same n every time.

    ents = []
    rngs = jax.random.split(rng, draws)
    feasible_x = x[feasible_mask]

    for r in rngs:
        samples = jax.random.choice(r, feasible_x, shape=(n,), replace=False)
        ents.append(approx_entropy(r, samples, k_range, n=n, d=d))

    return (jnp.asarray(ents).mean() + offset) * (n_feas / x.shape[0])


@jax.jit
def K_fn(x: Array) -> Array:
    """K(x) = sin(x1)*exp(x2) + log(|x3|+1)*tanh(x4) + prod_{i=5..9} cos(xi), 1-indexed."""
    return (
        jnp.sin(x[..., 0]) * jnp.exp(x[..., 1])
        + jnp.log(jnp.abs(x[..., 2]) + 1.0) * jnp.tanh(x[..., 3])
        + jnp.prod(jnp.cos(x[..., 4:9]), axis=-1)
    )


@jax.jit
def expected_K_error(gt_samples: Array, samples: Array) -> Array:
    """Absolute error in E[K(x)] between approximate and ground-truth samples."""
    return jnp.abs(jnp.mean(K_fn(samples)) - jnp.mean(K_fn(gt_samples)))