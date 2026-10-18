from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array, PRNGKey

from komino.constants import SAMPLE_PATH
from komino.distributions.constrained.halfspace import get_bound_contraints, HalfspaceIneq
from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.distributions.constraint import Constraint, ConstraintSet
from komino.distributions.unconstrained.uniform import Uniform


class GrowingSineConstraint(Constraint):
    def __init__(self):
        super().__init__()
        self.dim = 2

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        out = x[:, 1] - jnp.exp(-0.15 * x[:, 0]) * jnp.sin(x[:, 0])

        return out if batched else out[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x_input = x[None]
        else:
            x_input = x

        t = x_input[:, 0]

        dg_dt = -jnp.exp(-0.15 * t) * (jnp.cos(t) - 0.15 * jnp.sin(t))
        dg_dy = jnp.ones_like(t)

        out = jnp.stack([dg_dt, dg_dy], axis=-1)
        return out if batched else out[0]


class GrowingSineDistribution(ConstrainedDistribution):
    def __init__(self, bounds: tuple[int, int] = (-20., 20.)):
        self.dim = 2
        self.bounds = jnp.asarray(bounds)

        base = Uniform(bounds=self.bounds, dim=self.dim)

        sine_constraint = GrowingSineConstraint()
        y_constraint = HalfspaceIneq(a=jnp.array([0., -1.]), b=0.)
        outer_constraints = get_bound_contraints(bounds, self.dim)
        cons = ConstraintSet([sine_constraint, y_constraint, *outer_constraints])

        super().__init__(base, cons)

        # Precompute max arclength Jacobian over the feasible arc.
        n_grid = 10_000
        t_grid = jnp.linspace(float(self.bounds[0]), float(self.bounds[1]), n_grid)
        y_grid = jnp.exp(-0.15 * t_grid) * jnp.sin(t_grid)
        pts_grid = jnp.stack([t_grid, y_grid], axis=-1)
        dy_dt_grid = jnp.exp(-0.15 * t_grid) * (jnp.cos(t_grid) - 0.15 * jnp.sin(t_grid))
        ds_grid = jnp.sqrt(1.0 + dy_dt_grid ** 2)
        feas_grid = self.feasible(pts_grid, self.domain_tol)
        self.max_ds = jnp.max(jnp.where(feas_grid, ds_grid, 0.0)) * 1.05

    @property
    def init_scale(self) -> float:
        return (self.bounds[1] - self.bounds[0]) / 4

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        """Sample uniformly w.r.t. arclength on the feasible portion of the sine curve."""
        if load:
            try:
                out = jnp.load(SAMPLE_PATH / f"growing_sine_{n_samples}s_2d.npy")
                return out
            except FileNotFoundError as e:
                print(e)
                exit(1)

        out = jnp.zeros((n_samples, 2))
        bsz = int(min(max(2048, n_samples), 16384))
        t_min, t_max = float(self.bounds[0]), float(self.bounds[1])
        max_ds = self.max_ds

        def cond(state):
            _, _, curr = state
            return curr < n_samples

        def body(state):
            key, samples, curr = state
            key, r_t, r_accept = jr.split(key, 3)

            t = jr.uniform(r_t, (bsz,), minval=t_min, maxval=t_max)
            y = jnp.exp(-0.15 * t) * jnp.sin(t)
            proposals = jnp.stack([t, y], axis=-1)

            dy_dt = jnp.exp(-0.15 * t) * (jnp.cos(t) - 0.15 * jnp.sin(t))
            ds_dt = jnp.sqrt(1.0 + dy_dt ** 2)

            u = jr.uniform(r_accept, (bsz,))
            ok_arclength = u < ds_dt / max_ds

            ok_feas = self.feasible(proposals, self.domain_tol)
            ok = ok_arclength & ok_feas

            cs = jnp.cumsum(ok)
            sample_idx = curr + cs - 1
            sample_idx = jnp.where(ok & (sample_idx < n_samples), sample_idx, n_samples)
            samples = samples.at[sample_idx].set(proposals, mode="drop")
            curr = jnp.minimum(n_samples, curr + cs[-1])

            return key, samples, curr

        _, out, _ = jax.lax.while_loop(cond, body, (rng, out, 0))
        return out

    def save_samples(self, samples: Array, n_samples: int | None = None) -> None:
        if n_samples is None:
            n_samples = int(samples.shape[0])

        path = SAMPLE_PATH / f"growing_sine_{n_samples}s_{self.dim}d.npy"
        jnp.save(path, jnp.asarray(samples))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    rng = jax.random.PRNGKey(0)
    dist = GrowingSineDistribution()
    samples = dist.sample(rng, n_samples=2_000, load=False)
    dist.save_samples(samples)
