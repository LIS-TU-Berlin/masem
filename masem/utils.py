from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
from chex import Array, Numeric
from matplotlib import colormaps

from masem.distributions.constrained_dist import ConstrainedDistribution


@partial(jax.jit, static_argnums=(0,))
def linear_project(dist: ConstrainedDistribution, x: jnp.ndarray) -> jnp.ndarray:
    """One Newton step projecting x onto h(x)=0."""
    ev = jax.vmap(dist.eval)(x)
    m = ev.jac_h.shape[1]
    if m == 0:
        return x
    damp = dist.domain_tol
    def proj_one(_x, Jh, h):
        gram = Jh @ Jh.T + damp * jnp.eye(m)
        return _x - Jh.T @ jnp.linalg.solve(gram, h)
    return jax.vmap(proj_one)(x, ev.jac_h, ev.h)


def safe_norm(
    x: Array,
    min_norm: Numeric = 1e-12,
    ord: Optional[Union[int, float, str]] = None,  # pylint: disable=redefined-builtin
    axis: Union[None, tuple[int, ...], int] = None,
    keepdims: bool = False,
) -> Array:
    """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

    From:
    https://github.com/google-deepmind/optax/blob/main/optax/_src/numerics.py#L48#L83

    The gradients of `jnp.maximum(jnp.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
    because jax will evaluate both branches of the `jnp.maximum`. This function
    will instead return the correct gradient of 0.0 also in such setting.

    Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
    ord: {non-zero int, inf, -inf, 'fro', 'nuc'}, optional. Order of the norm.
      inf means numpy's inf object. The default is None.
    axis: {None, int, 2-tuple of ints}, optional. If axis is an integer, it
      specifies the axis of x along which to compute the vector norms. If axis
      is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
      norms of these matrices are computed. If axis is None then either a vector
      norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The
      default is None.
    keepdims: bool, optional. If this is set to True, the axes which are normed
      over are left in the result as dimensions with size one. With this option
      the result will broadcast correctly against the original x.

    Returns:
    The safe norm of the input vector, accounting for correct gradient.
    """
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
    norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
    masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return jnp.where(norm <= min_norm, min_norm, masked_norm)


def chol_inv(G: Array) -> Array:
    """Compute inverse via cholesky decomposition."""
    L = jax.scipy.linalg.cho_factor(G, lower=True, check_finite=False)
    I = jnp.eye(G.shape[-1])
    return jax.scipy.linalg.cho_solve(L, I, check_finite=False)


def inverse_softplus(x):
    # Numerically stable implementation of inverse softplus
    # Threshold above which the approximation log(e^x - 1) ≈ x is used
    threshold = 20.0
    return jnp.where(x > threshold, x, jnp.log(jnp.expm1(x)))


def reset_device_memory():
    """Free all tracked DeviceArray memory and delete objects."""
    backend = jax.extend.backend.get_backend()
    for buf in backend.live_buffers():
        buf.delete()


def extract_last_entry(dictionary: dict):
    last_entries = {}
    for key, value in dictionary.items():
        try:
            last_entries[key] = value[-min(len(value), 1)]
        except Exception as e:
            # print(f"Exception when extracting last entry for {key}: {e}")
            pass
    return last_entries


def reverse_transition_params(transition_params):
    flattened_params, tree = jax.tree_util.tree_flatten(transition_params, is_leaf=None)
    reversed_flattened_params = list(
        map(lambda w: jnp.flip(w, axis=0), flattened_params)
    )
    return jax.tree_util.tree_unflatten(tree, reversed_flattened_params)


def cmapper(name: str, cmap_name: str = "tab10"):
    map = colormaps[cmap_name]

    match name:
        case "nhr":
            return "#D43244", ":", "NHR"
        case "knn_nhr":
            return "#D43244", "-", "MASEM-NHR (Ours)"
        case "cluster_nhr":
            return map(5), "-.", "Cluster-NHR"
        case "olla":
            return "purple", ":", "OLLA"
        case "knn_olla":
            return "purple", "-", "MASEM-OLLA (Ours)"
        case "cghmc":
            return "#2D7D6B", ":", "CGHMC"
        case "soft_scmc":
            return "#2279B5", "-.", "SCMC"
        case _:
            raise NotImplementedError(f"{name} sampler not implemented!")


def seed_all(seed: int):
    import random
    import numpy as np

    # numpy
    random.seed(seed)
    np.random.seed(seed)


def get_dist(name: str, dim: int, *args, **kwargs):

    match name:
        case "uniform_ball2_connected":
            from masem.distributions.constrained.uniform_ball2 import UniformBall2
            centeroffset = kwargs.pop("center_offset", 0.6)
            return UniformBall2(dim=dim, *args, center_offset=centeroffset, **kwargs)
        case "uniform_ball2_disconnected":
            from masem.distributions.constrained.uniform_ball2 import UniformBall2
            centeroffset = kwargs.pop("center_offset", 1.35)
            return UniformBall2(dim=dim, *args, center_offset=centeroffset, **kwargs)
        case "seven_lobes":
            from masem.distributions.constrained.seven_lobes import SevenLobesUniform
            return SevenLobesUniform(dim=dim)
        case "disconnected_stress":
            from masem.distributions.constrained.disconnected_stresstest import DisconnectedStress
            return DisconnectedStress(dim=dim, *args, **kwargs)
        case "disconnected_stress_2d":
            from masem.distributions.constrained.disconnected_stresstest import DisconnectedStress
            return DisconnectedStress(dim=dim, *args, **kwargs)
        case "motion_planning_rand":
            from masem.distributions.motion_planning.motion_planning import RandomObstacleMotionPlanning
            return RandomObstacleMotionPlanning(*args, **kwargs)
        case "motion_planning_grid":
            from masem.distributions.motion_planning.motion_planning import GridObstacleMotionPlanning
            return GridObstacleMotionPlanning(*args, **kwargs)
        case "sine":
            from masem.distributions.constrained.growing_sine import GrowingSineDistribution
            return GrowingSineDistribution(*args, **kwargs)
        case "swiss_roll":
            from masem.distributions.constrained.swiss_roll import CircleSwissRoll
            return CircleSwissRoll(*args, **kwargs)
        case "capsule_grasp":
            from masem.distributions.capsule_grasp.capsule_grasp import CapsuleGrasp
            assert dim % 6 == 0, "CapsuleGrasp dim must be multiple of 6."
            return CapsuleGrasp(*args, num_fingers=dim//6, **kwargs)
        case _:
            raise NotImplementedError(f"{name} density not implemented!")


def plot_name(abbrev: str):
    if "sd" in abbrev:
        return r"$\mathcal{W}^2_2\downarrow$"
    elif "ed" in abbrev:
        return r"$\mathrm{ED}\downarrow$"
    elif "kl" in abbrev.lower():
        return r"$\text{KL}\downarrow$"
    elif "slack" in abbrev:
        return r"$\text{Slack}\downarrow$"
    elif "n_feas" in abbrev:
        return "N Feas"
    elif "entropy" in abbrev:
        return r"$\mathrm{EMC}\uparrow$"
    elif "fm" in abbrev:
        return r"$\mathbb{E}~[x]$ MSE $\downarrow$"
    elif "sm" in abbrev:
        return r"$\mathbb{V}~[x]$ MSE $\downarrow$"
    elif "mst_size" in abbrev:
        return "MST Size"
    elif "mean_logp" in abbrev:
        return r"$\log p(x)$"
    elif "comp_tv" in abbrev:
        return r"Comp. Volume $\text{TV}\downarrow$"
    elif "wallclock" in abbrev:
        return r"Wallclock $(s)\downarrow$"
    else:
        return abbrev
