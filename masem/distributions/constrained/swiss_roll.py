from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from chex import Array, PRNGKey
from matplotlib.axes import Axes

from masem.constants import SAMPLE_PATH
from masem.distributions.constrained.halfspace import get_bound_contraints
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.distributions.constraint import Constraint, ConstraintSet
from masem.distributions.unconstrained.uniform import Uniform


def _vmap_norm(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis) + eps)


def _make_random_circles(
    key: PRNGKey,
    n_circles: int,
    x_range: tuple[float, float] = (-7.5, -1.8),
    y_range: tuple[float, float] = (-4.8, 4.8),
    r_range: tuple[float, float] = (0.45, 1.15),
    min_gap: float = 0.35,
    max_tries: int = 5000,
) -> tuple[Array, Array]:
    centers = []
    radii = []
    k = key
    tries = 0
    while len(centers) < n_circles and tries < max_tries:
        tries += 1
        k, k1, k2, kr = jr.split(k, 4)
        c = jnp.array([
            jr.uniform(k1, (), minval=x_range[0], maxval=x_range[1]),
            jr.uniform(k2, (), minval=y_range[0], maxval=y_range[1]),
        ])
        r = jr.uniform(kr, (), minval=r_range[0], maxval=r_range[1])

        ok = True
        for cj, rj in zip(centers, radii):
            d = float(jnp.linalg.norm(c - cj))
            if d <= float(r + rj + min_gap):
                ok = False
                break
        if ok:
            centers.append(c)
            radii.append(r)

    if len(centers) < n_circles:
        raise RuntimeError(
            "Could not place all circles without overlap; "
            "relax min_gap or enlarge the left-half box."
        )

    return jnp.stack(centers, axis=0), jnp.asarray(radii)


class SmoothBottomLeftCutConstraint(Constraint):
    """Inequality constraint removing a smooth lower-left region.

    Feasible set: g(x) <= 0
    where
        g(x, y) = softplus(k(x0 - x)) * softplus(k(y0 - y)) - tau
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        k: float = 3.0,
        tau: float = 0.03,
    ) -> None:
        self.dim = 2
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.k = float(k)
        self.tau = float(tau)

    @property
    def is_equality(self) -> bool:
        return False

    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]

        xx, yy = x[:, 0], x[:, 1]
        sx = jax.nn.softplus(self.k * (self.x0 - xx))
        sy = jax.nn.softplus(self.k * (self.y0 - yy))
        out = sx * sy - self.tau
        return out if batched else out[0]

    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]

        xx, yy = x[:, 0], x[:, 1]

        ax = self.k * (self.x0 - xx)
        ay = self.k * (self.y0 - yy)

        sx = jax.nn.softplus(ax)
        sy = jax.nn.softplus(ay)

        # d/dx softplus(k(x0-x)) = -k * sigmoid(k(x0-x))
        dsx_dx = -self.k * jax.nn.sigmoid(ax)
        dsy_dy = -self.k * jax.nn.sigmoid(ay)

        gx = dsx_dx * sy
        gy = sx * dsy_dy

        out = jnp.stack([gx, gy], axis=-1)
        return out if batched else out[0]

    def analytical_intersections(
        self, x: Array, dir: Array, tol: float = 1e-12
    ) -> Array:
        pass


class CircleSwissRollConstraint(Constraint):
    """Equality constraint whose zero set is a union of circles and one 2D swiss-roll.

    Circles:
        ||x - c_k|| - r_k = 0

    Swiss roll (Archimedean spiral):
        gamma(t) = c_spiral + (a + b t) [cos t, sin t],  t in [t_min, t_max]

    The constraint value is the *signed* residual of the closest component
    (selected by argmin of the unsigned residual):
        active(x) = argmin over { ||x-c_k|| - r_k, rho - (a + b(theta + 2πm)),
                                  distance to spiral endpoints }
        value(x) = signed_residual_of(active(x))
    Feasible points are exactly those where the active signed residual is zero
    (i.e. on any circle or on the spiral segment). Locally on each component
    the gradient is the unit (or near-unit) outward normal, so LICQ holds on
    the entire feasible set. The value is discontinuous across Voronoi-like
    cell boundaries between components, but those boundaries do not intersect
    the feasible manifold.
    """

    def __init__(
        self,
        circle_centers: Array,
        circle_radii: Array,
        spiral_center: tuple[float, float] = (4.8, -0.4),
        spiral_a: float = 0.45,
        spiral_b: float = 0.33,
        spiral_t_min: float = 0.9,
        spiral_t_max: float = 3.9 * jnp.pi,
    ) -> None:
        self.dim = 2
        self.circle_centers = jnp.asarray(circle_centers)
        self.circle_radii = jnp.asarray(circle_radii)
        self.n_circles = int(self.circle_radii.shape[0])

        self.spiral_center = jnp.asarray(spiral_center)
        self.spiral_a = float(spiral_a)
        self.spiral_b = float(spiral_b)
        self.spiral_t_min = float(spiral_t_min)
        self.spiral_t_max = float(spiral_t_max)

        m_min = int(jnp.floor((self.spiral_t_min - jnp.pi) / (2.0 * jnp.pi))) - 1
        m_max = int(jnp.ceil((self.spiral_t_max + jnp.pi) / (2.0 * jnp.pi))) + 1
        self.branch_ms = jnp.arange(m_min, m_max + 1)

    @property
    def is_equality(self) -> bool:
        return True

    def _spiral_point(self, t: Array) -> Array:
        r = self.spiral_a + self.spiral_b * t
        return self.spiral_center + jnp.stack([r * jnp.cos(t), r * jnp.sin(t)], axis=-1)

    def _circle_values(self, x: Array) -> Array:
        diff = x[:, None, :] - self.circle_centers[None, :, :]
        d = _vmap_norm(diff, axis=-1)
        return d - self.circle_radii[None, :]

    def _circle_grads(self, x: Array) -> Array:
        diff = x[:, None, :] - self.circle_centers[None, :, :]
        d = _vmap_norm(diff, axis=-1)[..., None]
        return diff / d

    def _spiral_branch_values_and_grads(self, x: Array) -> tuple[Array, Array]:
        z = x - self.spiral_center[None, :]
        zx, zy = z[:, 0], z[:, 1]
        rho = _vmap_norm(z, axis=-1)
        rho2 = jnp.sum(z ** 2, axis=-1) + 1e-12
        theta = jnp.atan2(zy, zx)

        t_cands = theta[:, None] + 2.0 * jnp.pi * self.branch_ms[None, :]
        valid = (t_cands >= self.spiral_t_min) & (t_cands <= self.spiral_t_max)

        signed_res = rho[:, None] - (self.spiral_a + self.spiral_b * t_cands)
        vals = jnp.where(valid, signed_res, jnp.inf)

        drho = z / rho[:, None]
        dtheta = jnp.stack([-zy, zx], axis=-1) / rho2[:, None]
        common_grad = drho - self.spiral_b * dtheta
        grads = jnp.broadcast_to(
            common_grad[:, None, :],
            t_cands.shape + (x.shape[-1],),
        )
        grads = jnp.where(valid[..., None], grads, 0.0)

        p0 = self._spiral_point(jnp.asarray(self.spiral_t_min))
        p1 = self._spiral_point(jnp.asarray(self.spiral_t_max))
        d0 = x - p0[None, :]
        d1 = x - p1[None, :]
        v0 = _vmap_norm(d0, axis=-1)
        v1 = _vmap_norm(d1, axis=-1)
        g0 = d0 / v0[:, None]
        g1 = d1 / v1[:, None]

        vals = jnp.concatenate([vals, v0[:, None], v1[:, None]], axis=1)
        grads = jnp.concatenate([grads, g0[:, None, :], g1[:, None, :]], axis=1)
        return vals, grads

    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]

        circle_signed = self._circle_values(x)
        spiral_signed, _ = self._spiral_branch_values_and_grads(x)
        all_signed = jnp.concatenate([circle_signed, spiral_signed], axis=1)
        idx = jnp.argmin(jnp.abs(all_signed), axis=1)
        out = all_signed[jnp.arange(x.shape[0]), idx]
        return out if batched else out[0]

    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]

        circle_signed = self._circle_values(x)
        circle_grads = self._circle_grads(x)
        spiral_signed, spiral_grads = self._spiral_branch_values_and_grads(x)

        all_signed = jnp.concatenate([circle_signed, spiral_signed], axis=1)
        all_grads = jnp.concatenate([circle_grads, spiral_grads], axis=1)
        idx = jnp.argmin(jnp.abs(all_signed), axis=1)
        out = all_grads[jnp.arange(x.shape[0]), idx]
        return out if batched else out[0]

    def analytical_intersections(
        self, x: Array, dir: Array, tol: float = 1e-12
    ) -> Array:
        pass


class CircleSwissRoll(ConstrainedDistribution):
    """Uniform version on the same disconnected manifold."""

    def __init__(
        self,
        dim: int = 2,
        bounds: tuple[float, float] = (-10., 10.),
        n_circles: int = 6,
        layout_seed: int = 0,
    ):
        assert dim == 2, "CircleSwissRollUniform is a 2D benchmark."
        self.bounds = bounds
        self.dim = dim



        circle_centers, circle_radii = _make_random_circles(
            jr.PRNGKey(layout_seed),
            n_circles=n_circles,
            x_range=(bounds[0] + 1.0, bounds[0] * 0.35),
            y_range=(bounds[0] * 0.55, bounds[1] * 0.55),
            r_range=(0.45, 1.15),
            min_gap=0.35,
        )

        self.circle_centers = circle_centers
        self.circle_radii = circle_radii
        self.spiral_center = jnp.array((4.8, -0.4))
        self.spiral_a = 0.45
        self.spiral_b = 0.33
        self.spiral_t_min = 0.9
        self.spiral_t_max = float(3.9 * jnp.pi)

        base = Uniform(
            bounds=tuple(bounds),
            dim=dim,
        )
        cons = ConstraintSet([
            CircleSwissRollConstraint(
                circle_centers=self.circle_centers,
                circle_radii=self.circle_radii,
                spiral_center=tuple(self.spiral_center.tolist()),
                spiral_a=self.spiral_a,
                spiral_b=self.spiral_b,
                spiral_t_min=self.spiral_t_min,
                spiral_t_max=self.spiral_t_max,
            ),
            SmoothBottomLeftCutConstraint(
                x0=-3.0,
                y0=0.0,
                k=1.0,
                tau=0.1,
            ),
            *get_bound_contraints(bounds, dim),
        ])
        super().__init__(base, cons)

        pts = []
        log_jacs = []

        phi = jnp.linspace(0.0, 2.0 * jnp.pi, 1200, endpoint=False)
        for c, r in zip(self.circle_centers, self.circle_radii):
            circ = c[None, :] + r * jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)
            pts.append(circ)
            log_jacs.append(jnp.full((phi.shape[0],), jnp.log(r + 1e-12)))

        t = jnp.linspace(self.spiral_t_min, self.spiral_t_max, 4000)
        rr = self.spiral_a + self.spiral_b * t
        spiral = self.spiral_center[None, :] + rr[:, None] * jnp.stack(
            [jnp.cos(t), jnp.sin(t)], axis=-1
        )
        ds_dt = jnp.sqrt(rr ** 2 + self.spiral_b ** 2)
        pts.append(spiral)
        log_jacs.append(jnp.log(ds_dt + 1e-12))

        pts = jnp.concatenate(pts, axis=0)
        log_jacs = jnp.concatenate(log_jacs, axis=0)
        log_weights = base.unnorm_log_prob(pts) + log_jacs
        self.max_logp = jnp.max(log_weights) + 0.1

        self.component_param_lengths = jnp.concatenate([
            jnp.full((n_circles,), 2.0 * jnp.pi),
            jnp.asarray([self.spiral_t_max - self.spiral_t_min]),
        ])
        self.component_probs = self.component_param_lengths / jnp.sum(
            self.component_param_lengths
        )

    @property
    def init_scale(self) -> float:
        return (self.bounds[1] - self.bounds[0]) / 4

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        if load:
            try:
                out = jnp.load(
                    SAMPLE_PATH / f"circle_swiss_roll_{n_samples}s_{self.dim}d.npy"
                )
                return out
            except FileNotFoundError as e:
                print(e)
                exit(1)

        rng, r0 = jr.split(rng)
        proto = self.base.sample(r0, 1)
        out = jnp.zeros((n_samples,) + proto.shape[1:])
        bsz = int(min(max(2048, n_samples), 16384))
        n_c = self.circle_radii.shape[0]

        def cond(state):
            _, _, curr_n = state
            return curr_n < n_samples

        def body(state):
            key, samples, curr_n = state
            key, r_comp, r_param, r_accept = jr.split(key, 4)

            comp = jr.choice(r_comp, n_c + 1, shape=(bsz,), p=self.component_probs)
            is_circle = comp < n_c

            phi = jr.uniform(r_param, shape=(bsz,), minval=0.0, maxval=2.0 * jnp.pi)
            c_idx = jnp.clip(comp, 0, n_c - 1)
            c = self.circle_centers[c_idx]
            rad = self.circle_radii[c_idx]
            circ_prop = c + rad[:, None] * jnp.stack(
                [jnp.cos(phi), jnp.sin(phi)], axis=-1
            )
            circ_logjac = jnp.log(rad + 1e-12)

            t = jr.uniform(
                r_param,
                shape=(bsz,),
                minval=self.spiral_t_min,
                maxval=self.spiral_t_max,
            )
            rr = self.spiral_a + self.spiral_b * t
            spiral_prop = self.spiral_center[None, :] + rr[:, None] * jnp.stack(
                [jnp.cos(t), jnp.sin(t)], axis=-1
            )
            spiral_logjac = jnp.log(jnp.sqrt(rr ** 2 + self.spiral_b ** 2) + 1e-12)

            proposals = jnp.where(is_circle[:, None], circ_prop, spiral_prop)
            log_jac = jnp.where(is_circle, circ_logjac, spiral_logjac)
            log_accept = log_jac - self.max_logp
            u = jr.uniform(r_accept, (bsz,), minval=1e-8, maxval=1.0)
            ok_mog = jnp.log(u) <= log_accept

            ok_feas = self.feasible(proposals, tol=self.domain_tol)
            ok = ok_mog & ok_feas

            cs = jnp.cumsum(ok)
            sample_idx = curr_n + cs - 1
            sample_idx = jnp.where(ok & (sample_idx < n_samples), sample_idx, n_samples)
            samples = samples.at[sample_idx].set(proposals, mode="drop")
            curr_n = jnp.minimum(n_samples, curr_n + cs[-1])
            return key, samples, curr_n

        _, out, _ = jax.lax.while_loop(cond, body, (rng, out, 0))
        return out

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

        if bounds is None:
            bounds = self.bounds
        if bounds_2 is None:
            bounds_2 = bounds

        # The signed-residual equality flips sign across Voronoi-like switch
        # boundaries between adjacent components, which makes contour() draw
        # spurious zero-level lines. We draw the true zero set (circles + spiral)
        # parametrically below, so suppress the grid-based equality contour here.
        saved_eq = self.constraints.eq
        self.constraints.eq = []
        try:
            ax = super().plot2d(
                samples=samples, ax=ax,
                bounds=bounds, bounds_2=bounds_2,
                grid_width_n_points=grid_width_n_points,
                grid_width_n_points_2=grid_width_n_points_2,
                n_contour_levels=n_contour_levels,
                plot_marginal_dims=plot_marginal_dims,
                markersize=markersize, alpha=alpha,
                cmap_alpha=cmap_alpha, plt_show=False,
                title=title, filled=filled, cmap=cmap,
            )
        finally:
            self.constraints.eq = saved_eq

        # Draw circles and spiral parametrically (exact, on top of grid contours)
        phi = onp.linspace(0.0, 2.0 * onp.pi, 800, endpoint=True)
        for c, r in zip(onp.array(self.circle_centers), onp.array(self.circle_radii)):
            ax.plot(
                c[0] + r * onp.cos(phi), c[1] + r * onp.sin(phi),
                    color="black", linewidth=1.5, zorder=2
            )

        t = onp.linspace(self.spiral_t_min, self.spiral_t_max, 2000)
        rr = self.spiral_a + self.spiral_b * t
        sc = onp.array(self.spiral_center)
        ax.plot(
            sc[0] + rr * onp.cos(t), sc[1] + rr * onp.sin(t),
                color="black", linewidth=1.5, zorder=2
        )

        ax.set_xlim(bounds)
        ax.set_ylim(bounds_2)

        if plt_show:
            plt.show()

        return ax

    def save_samples(self, samples: Array, n_samples: int | None = None) -> None:
        if n_samples is None:
            n_samples = int(samples.shape[0])

        path = SAMPLE_PATH / f"circle_swiss_roll_{n_samples}s_{self.dim}d.npy"
        jnp.save(path, jnp.asarray(samples))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    rng = jax.random.PRNGKey(0)
    dist = CircleSwissRoll()
    samples = dist.sample(rng, n_samples=2_000, load=False)

    dist.plot2d(samples, plt_show=True, grid_width_n_points=100)
    dist.save_samples(samples)
