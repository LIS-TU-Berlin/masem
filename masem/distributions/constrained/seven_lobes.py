"""Based on:

https://github.com/KraitGit/OLLA/blob/main/src/constraints/mix_gaussian.py


MIT License

Copyright (c) 2025 Kijung Jeon
2026 Cornelius Braun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array, PRNGKey

from masem.constants import SAMPLE_PATH
from masem.distributions.constrained.halfspace import get_bound_contraints
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.distributions.constraint import Constraint, ConstraintSet
from masem.distributions.unconstrained.uniform import Uniform


class StarConstraint(Constraint):
    def __init__(self, a: float = 1.0, k: float = 7.0) -> None:
        self.dim = 2
        self.k = k
        self.a = a

    @property
    def is_equality(self) -> bool:
        return True

    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        theta = jnp.atan2(x[:, 1], x[:, 0])
        r_boundary = 3.0 + self.a * jnp.cos(self.k * theta)
        out = jnp.linalg.norm(x, axis=-1) - r_boundary
        return out if batched else out[0]

    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        r2 = jnp.sum(x ** 2, axis=-1, keepdims=True)
        r = jnp.sqrt(r2 + 1e-12)
        theta = jnp.atan2(x[:, 1], x[:, 0])
        dr_dx = x / r
        dtheta_dx = jnp.stack([-x[:, 1], x[:, 0]], axis=-1) / r2
        dboundary_dtheta = -self.a * self.k * jnp.sin(self.k * theta)[:, None]
        dboundary_dx = dboundary_dtheta * dtheta_dx
        g = dr_dx - dboundary_dx

        return g if batched else g[0]

    def analytical_intersections(
        self, x: Array, dir: Array, tol: float = 1e-12
    ) -> Array:
        pass


class PolynIneqConstraint(Constraint):
    def __init__(self) -> None:
        self.dim = 2

    @property
    def is_equality(self) -> bool:
        return False

    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        out = (x[:, 0] - 2.0) ** 2 - 5 * x[:, 0] * (x[:, 1] ** 3) + 0.5 * x[:, 1] ** 5 - 40.0
        return out if batched else out[0]

    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        x0 = x[:, 0]
        x1 = x[:, 1]
        df_dx0 = 2.0 * (x0 - 2.0) - 5.0 * (x1 ** 3)
        df_dx1 = -15.0 * x0 * (x1 ** 2) + 2.5 * (x1 ** 4)
        g = jnp.stack([df_dx0, df_dx1], axis=-1)

        return g if batched else g[0]

    def analytical_intersections(
        self, x: Array, dir: Array, tol: float = 1e-12
    ) -> Array:
        pass


class SevenLobesUniform(ConstrainedDistribution):
    def __init__(
        self,
        dim: int = 2,   # pholder
        bounds: tuple[float, float] = (-4.5, 4.5)
    ):
        self.a = 1.0
        self.k = 7.0
        base = Uniform(
            bounds=tuple([0.6 * b for b in bounds]),
            dim=dim,
        )
        cons = ConstraintSet([
            StarConstraint(a=self.a, k=self.k),
            PolynIneqConstraint(),
            *get_bound_contraints(bounds, dim)
        ])
        super().__init__(base, cons)
        self.bounds = bounds

        # Precompute the maximum arclength Jacobian over the feasible arc.
        # ds/dtheta = sqrt(r^2 + r'^2) is non-uniform
        n_grid = 10_000
        thetas_grid = jnp.linspace(0, 2 * jnp.pi, n_grid, endpoint=False)
        r_grid = 3.0 + self.a * jnp.cos(self.k * thetas_grid)
        pts_grid = jnp.column_stack([r_grid * jnp.cos(thetas_grid), r_grid * jnp.sin(thetas_grid)])
        dr_dt_grid = -self.a * self.k * jnp.sin(self.k * thetas_grid)
        ds_grid = jnp.sqrt(r_grid ** 2 + dr_dt_grid ** 2)
        feas_grid = self.feasible(pts_grid, self.domain_tol)
        self.max_ds = jnp.max(jnp.where(feas_grid, ds_grid, 0.0)) * 1.05

    @property
    def init_scale(self) -> float:
        return (self.bounds[1] - self.bounds[0]) / 4

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        """Sample uniformly w.r.t. arclength on the feasible portion of the star curve.

        Proposal: theta ~ Uniform[0, 2pi]
        Accept:   feasible(x(theta))  AND  u < ds/dtheta / max_ds w jacobian correction.
        """
        if load:
            try:
                out = jnp.load(SAMPLE_PATH / f"seven_lobes_uniform_{n_samples}s_{self.dim}d.npy")
                return out
            except FileNotFoundError as e:
                print(e)
                exit(1)

        rng, r0 = jr.split(rng)
        out = jnp.zeros((n_samples, 2))
        bsz = int(min(max(2048, n_samples), 16384))
        a, k = self.a, self.k
        max_ds = self.max_ds

        def cond(state):
            _, _, curr = state
            return curr < n_samples

        def body(state):
            key, samples, curr = state
            key, r_theta, r_accept = jr.split(key, 3)

            # Propose points on the star curve via polar parametrisation
            theta = jr.uniform(r_theta, (bsz,), minval=0., maxval=2. * jnp.pi)
            r_val = 3.0 + a * jnp.cos(k * theta)
            proposals = jnp.column_stack([r_val * jnp.cos(theta), r_val * jnp.sin(theta)])

            # Arclength Jacobian correction
            dr_dt = -a * k * jnp.sin(k * theta)
            ds_dtheta = jnp.sqrt(r_val ** 2 + dr_dt ** 2)
            u = jr.uniform(r_accept, (bsz,))
            ok_arclength = u < ds_dtheta / max_ds

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

        path = SAMPLE_PATH / f"seven_lobes_uniform_{n_samples}s_{self.dim}d.npy"
        jnp.save(path, jnp.asarray(samples))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    rng = jax.random.PRNGKey(0)
    dist = SevenLobesUniform()
    samples = dist.sample(rng, n_samples=2_000, load=False)
    #jnp.save(SAMPLE_PATH / f"seven_lobes_uniform_2000s_2d.npy", samples)
    dist.plot2d(samples)
    dist.save_samples(samples)
