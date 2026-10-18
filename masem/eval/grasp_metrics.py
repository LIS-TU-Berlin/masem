from functools import partial

import jax
import jax.numpy as jnp
from chex import Array

from masem.distributions.capsule_grasp.capsule_grasp import CapsuleGrasp
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.eval.performance_metrics import approx_entropy


@partial(jax.jit, static_argnums=(1,))
def n_working_entropy(x: Array, dist: CapsuleGrasp) -> float:
    if x.ndim == 1:
        x = x[None]
    x = x.reshape(x.shape[0], -1, 7)
    working = x[:, :, 6] > 0
    num_working = jnp.sum(working, axis=1)
    counts = jnp.bincount(num_working, length=dist.num_fingers)[dist.min_fingers:]
    probs = counts / jnp.sum(counts)
    ent = -jnp.sum(probs * jnp.log(probs + 1e-8))
    return ent


@partial(jax.jit, static_argnums=(1,))
def finger_broken_fraction_entropy(x: Array, dist: CapsuleGrasp) -> float:
    if x.ndim == 1:
        x = x[None]
    x = x.reshape(x.shape[0], dist.num_fingers, 7)
    working = x[:, :, 6] > 0
    broken_fraction = 1.0 - jnp.mean(working, axis=0)
    ent = -jnp.sum(broken_fraction * jnp.log(broken_fraction + 1e-8))
    return ent


@partial(jax.jit, static_argnums=(1,))
def n_working_fraction_max_diff(x: Array, dist: CapsuleGrasp) -> float:
    if x.ndim == 1:
        x = x[None]
    x = x.reshape(x.shape[0], -1, 7)
    working = x[:, :, 6] > 0
    num_working = jnp.sum(working, axis=1)
    counts = jnp.bincount(num_working, length=dist.num_fingers)[dist.min_fingers:]
    frac_working = counts / jnp.sum(counts)
    max_diff = jnp.max(frac_working) - jnp.min(frac_working)
    return max_diff


def approx_feasible_pos_entropy(
    rng,
    x,
    dist: ConstrainedDistribution,
    k_range: tuple = (2, 4, 8),
    n=100,
    tol=1e-3,
    draws=10,
):
    feasible_mask = dist.feasible(x, tol=tol)
    n_feas = int(feasible_mask.sum())
    if n_feas < n:        # big problem with KL-estimator: depends on n, meaning you cant really compare entropy across different n.
        return jnp.nan    # so we should ensure we use the same n every time.

    ents = []
    rngs = jax.random.split(rng, draws)
    feasible_x = x[feasible_mask]
    feasible_pos = dist.get_block_pos(feasible_x)

    for r in rngs:
        samples = jax.random.choice(r, feasible_pos, shape=(n,), replace=False)
        ents.append(approx_entropy(r, samples, k_range, n=n))

    return jnp.asarray(ents).mean() * (n_feas / x.shape[0])
