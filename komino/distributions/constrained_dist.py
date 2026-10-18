from typing import Optional

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from functools import partial

from jax.nn import relu
from matplotlib.axes import Axes

from komino.distributions.constraint import ConstraintSet
from komino.distributions.distribution import Distribution
from komino.types import Eval


class ConstrainedDistribution(Distribution):
    def __init__(self, base: Distribution, constraints: ConstraintSet):
        self.bounds = base.bounds
        self.base = base
        self.constraints = constraints
        self.dim = base.dim
        self.logprob_min = self.base.log_prob_min
        self.domain_tol = self.base.domain_tol
        self.eps = self.base.eps

        self.vmap_blogp = jax.vmap(self.base.unnorm_log_prob)
        self.vmap_bscore = jax.vmap(self.base.score)

    @partial(jax.jit, static_argnums=(0,))
    def unnorm_log_prob(self, x: Array) -> Array:
        logp = self.base_unnorm_log_prob(x)
        mask = self.feasible(x, self.domain_tol)
        return jnp.where(mask, logp, self.logprob_min)

    @partial(jax.jit, static_argnums=(0,))
    def score(self, x: Array) -> Array:
        """Be flat in infeasible regions."""
        s = self.base_score(x)
        mask = self.feasible(x, self.domain_tol)
        m = mask if mask.ndim == s.ndim else mask[..., None]
        return jnp.where(m, s, jnp.zeros_like(s))

    @partial(jax.jit, static_argnums=(0,))
    def base_unnorm_log_prob(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        logp = self.vmap_blogp(x)
        if not batched:
            logp = jnp.squeeze(logp, axis=0)
        return logp

    @partial(jax.jit, static_argnums=(0,))
    def base_score(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        score = self.vmap_bscore(x)
        if not batched:
            score = jnp.squeeze(score, axis=0)
        return score

    @partial(jax.jit, static_argnums=(0,))
    def base_val_and_score(self, x: Array) -> tuple[Array, Array]:
        """Returns (unnorm_log_prob, score) in a single forward+backward pass."""
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        logp, score = jax.vmap(jax.value_and_grad(self.base.unnorm_log_prob))(x)
        if not batched:
            logp = jnp.squeeze(logp, axis=0)
            score = jnp.squeeze(score, axis=0)
        return logp, score

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        from warnings import warn

        warn(
            "Sampling GT from constrained distribution not supported; returning GT samples from base distribution."
        )
        return self.base.sample(rng, n_samples)

    @partial(jax.jit, static_argnums=(0,))
    def g(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        g = self.constraints.g(x)
        if not batched:
            g = jnp.squeeze(g, axis=0)
        return g

    @partial(jax.jit, static_argnums=(0,))
    def Jg(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        Jg = self.constraints.Jg(x)
        if not batched:
            Jg = jnp.squeeze(Jg, axis=0)
        return Jg

    @partial(jax.jit, static_argnums=(0,))
    def h(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        h = self.constraints.h(x)
        if not batched:
            h = jnp.squeeze(h, axis=0)
        return h

    @partial(jax.jit, static_argnums=(0,))
    def Jh(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        Jh = self.constraints.Jh(x)
        if not batched:
            Jh = jnp.squeeze(Jh, axis=0)
        return Jh

    @partial(jax.jit, static_argnums=(0,))
    def h_and_Jh(self, x: Array) -> tuple[Array, Array]:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        h_vals, Jh = self.constraints.h_and_Jh(x)
        if not batched:
            h_vals = jnp.squeeze(h_vals, axis=0)
            Jh = jnp.squeeze(Jh, axis=0)
        return h_vals, Jh

    @partial(jax.jit, static_argnums=(0,))
    def feasible(self, x: Array, tol: float = 1e-6) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        feas = jax.vmap(self.constraints.feasible, (0, None))(x, tol)
        if not batched:
            feas = jnp.squeeze(feas, axis=0)
        return feas

    @partial(jax.jit, static_argnums=(0,))
    def slack_vec(self, x: Array) -> Array:
        g = self.g(x)
        h = self.h(x)
        return jnp.concatenate([relu(g - self.domain_tol), jnp.abs(h)], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def eval(self, x: Array) -> Eval:
        """Returns compact eval object that contains all info for sampling.

        logp: unnormalized log density of unconstrained target.
        score: score of unconstrained target.
        g: value of inequality constraint fcts.
        h: value of equality constraint fcts.
        Jg: Jac of inequality constraints.
        Jh: Jac of equality constraints.
        """
        g = self.g(x)
        h = self.h(x)
        jac_g = self.Jg(x)
        jac_h = self.Jh(x)
        mask = g >= self.domain_tol
        slack = jnp.concatenate([relu(g - self.domain_tol), jnp.abs(h)], axis=-1)
        jac_s_g = jac_g * mask[..., None]
        jac_s_h = jac_h * jnp.sign(h)[..., None]
        jac_slack = jnp.concatenate([jac_s_g, jac_s_h], axis=-2)
        return Eval(
            logp=self.unnorm_log_prob(x),
            score=self.score(x),
            base_logp=self.base_unnorm_log_prob(x),
            base_score=self.base_score(x),
            g=g,
            h=h,
            jac_g=jac_g,
            jac_h=jac_h,
            slack=slack,
            jac_slack=jac_slack,
        )

    @property
    def custom_metrics(self):
        return {}

    @property
    def init_scale(self) -> float:
        return (self.bounds[1] - self.bounds[0]) / 4

    @property
    def neq(self) -> int:
        return len(self.constraints.eq)

    @property
    def nineq(self) -> int:
        return len(self.constraints.ineq)

    def plot2d(
        self,
        samples: Array = None,
        ax: Axes = None,
        bounds: Optional[tuple[float | int, float | int]] = None,
        bounds_2: Optional[tuple[float | int, float | int]] = None,
        grid_width_n_points: int = 20,
        grid_width_n_points_2: int = None,
        n_contour_levels: int = None,
        plot_marginal_dims: tuple[int] = (0, 1),
        markersize: float = 2.0,
        alpha: float = 0.6,
        cmap_alpha: float = 0.15,
        plt_show: bool = True,
        title: str = None,
        filled: bool = True,
        cmap: str = "Blues",
    ) -> Axes:
        import matplotlib.pyplot as plt
        import numpy as onp

        if ax is None:
            _, ax = plt.subplots(1, 1)

        if bounds is None:
            bounds = self.bounds
        if bounds_2 is None:
            bounds_2 = bounds
        if grid_width_n_points_2 is None:
            grid_width_n_points_2 = grid_width_n_points

        if self.dim == 2:
            x1_vals = jnp.linspace(bounds[0], bounds[1], grid_width_n_points)
            x2_vals = jnp.linspace(bounds_2[0], bounds_2[1], grid_width_n_points_2)
            X, Y = jnp.meshgrid(x1_vals, x2_vals)
            grid_pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
            X_np, Y_np = onp.array(X), onp.array(Y)
            shape2d = (grid_width_n_points_2, grid_width_n_points)

            for c in self.constraints.eq:
                vals = c.value(grid_pts)
                if vals.ndim > 1:
                    vals = vals.squeeze(-1)
                vals_2d = onp.array(vals.reshape(shape2d))
                ax.contour(
                    X_np, Y_np, vals_2d,
                    levels=[0], colors=["black"], linestyles=["solid"], linewidths=1.5, zorder=2,
                )

            # Skip bound constraints; shade green where ALL non-box ineqs are satisfied
            nonbox_ineqs = self.constraints.ineq[:-int(2 * self.dim)]
            if nonbox_ineqs:
                grid_vals = []
                for c in nonbox_ineqs:
                    vals = c.value(grid_pts)
                    if vals.ndim > 1:
                        vals = vals.squeeze(-1)
                    grid_vals.append(onp.array(vals.reshape(shape2d)))

                joint = onp.maximum.reduce(grid_vals)
                ax.contourf(
                    X_np, Y_np, joint,
                    levels=[joint.min() - 1, 0.], colors=["green"], alpha=cmap_alpha, zorder=1,
                )
                for vals_2d in grid_vals:
                    ax.contour(
                        X_np, Y_np, vals_2d,
                        levels=[0], colors=["black"], linestyles=["dashed"], linewidths=1.5, zorder=2,
                    )

        elif samples is None:
            return ax

        if title is not None:
            ax.set_title(title)

        if samples is not None:
            ax.scatter(
                onp.clip(samples[:, plot_marginal_dims[0]], bounds[0], bounds[1]),
                onp.clip(samples[:, plot_marginal_dims[1]], bounds_2[0], bounds_2[1]),
                s=markersize, alpha=alpha, zorder=3, c="#2688c9", edgecolors="#1f77b3",
            )
            ax.axis("equal")

        if plt_show:
            plt.show()

        return ax
