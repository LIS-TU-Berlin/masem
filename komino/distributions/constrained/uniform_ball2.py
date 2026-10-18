from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as onp

from functools import partial
from chex import Array, PRNGKey
from matplotlib.axes import Axes

from komino.constants import SAMPLE_PATH
from komino.distributions.constrained.halfspace import get_bound_contraints
from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.distributions.constraint import Constraint, ConstraintSet
from komino.distributions.unconstrained.uniform import Uniform


class SphereEq(Constraint):
    """Sphere equality constraint: ||x|| = radius."""

    def __init__(self, radius: float):
        self.radius = float(radius)

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        return jnp.linalg.norm(x, axis=-1) - self.radius


class TwoDisksIneq(Constraint):
    """Union of two spherical caps on a sphere."""

    def __init__(
        self,
        center_dirs: Array,
        angular_radii: Array,
        sphere_radius: float,
    ):
        center_dirs = jnp.asarray(center_dirs, dtype=float)
        center_dirs = center_dirs / jnp.linalg.norm(center_dirs, axis=-1, keepdims=True)

        self.center_dirs = center_dirs
        self.angular_radii = jnp.asarray(angular_radii, dtype=float)
        self.cos_radii = jnp.cos(self.angular_radii)
        self.sphere_radius = float(sphere_radius)

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        x_dir = x / self.sphere_radius
        dots = x_dir @ self.center_dirs.T

        vals = self.cos_radii[None, :] - dots
        out = jnp.min(vals, axis=-1)

        return out if batched else out[0]


class UniformBall2(ConstrainedDistribution):
    """Uniform distribution on two spherical caps with different radii."""

    def __init__(
        self,
        dim: int = 3,
        sphere_radius: float = 2.5,
        r_small: float = 0.2,
        r_large: float = 0.60,
        center_offset: float = 0.6,
        # center_offset: float = 1.35,
        center_tilt: float = onp.pi / 4.0,
        bounds: tuple[float, float] | None = (-5.0, 5.0),
    ):
        if dim != 3:
            raise ValueError("UniformSphereTwoCaps is only implemented for dim=3.")
        if bounds is None:
            bounds = (-1.05 * sphere_radius, 1.05 * sphere_radius)

        self.sphere_radius = float(sphere_radius)
        self.r_small = float(r_small)
        self.r_large = float(r_large)
        self.center_offset = float(center_offset)
        self.center_tilt = float(center_tilt)


        bounds_arr = jnp.asarray(bounds, dtype=float)
        base = Uniform(bounds=bounds_arr, dim=dim)
        base.log_prob_min = -10_000.0
        outer_constraints = get_bound_contraints(bounds_arr, dim)

        # Disk center
        half_sep = 0.5 * self.center_offset
        base_mu1 = jnp.array([jnp.cos(-half_sep), jnp.sin(-half_sep), 0.0])
        base_mu2 = jnp.array([jnp.cos(half_sep), jnp.sin(half_sep), 0.0])
        tilt = self.center_tilt
        rot_x = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, jnp.cos(tilt), -jnp.sin(tilt)],
                [0.0, jnp.sin(tilt), jnp.cos(tilt)],
            ]
        )
        mu1 = rot_x @ base_mu1
        mu2 = rot_x @ base_mu2
        center_dirs = jnp.stack([mu1, mu2], axis=0)
        angular_radii = jnp.array([self.r_small, self.r_large], dtype=float)

        sphere_eq = SphereEq(radius=self.sphere_radius)
        cap_constraint = TwoDisksIneq(
            center_dirs=center_dirs,
            angular_radii=angular_radii,
            sphere_radius=self.sphere_radius,
        )

        cons = ConstraintSet([sphere_eq, cap_constraint, *outer_constraints])
        super().__init__(base, cons)

        self.sphere_eq = sphere_eq
        self.cap_constraint = cap_constraint
        self._center_dirs = center_dirs
        self._angular_radii = angular_radii
        self._cos_radii = jnp.cos(angular_radii)

        # Spherical cap area:
        #
        #     area(theta) = 2 pi R^2 (1 - cos(theta)).
        #
        # Constants cancel in the categorical weights.
        log_w = jnp.log1p(-jnp.cos(angular_radii))
        self._log_alpha = log_w - jsp.special.logsumexp(log_w)

    @property
    def init_scale(self) -> float:
        # return self.sphere_radius / 2.0
        return (self.bounds[1] - self.bounds[0]) / 4

    def _cap_tangent_bases(self) -> tuple[Array, Array]:
        """Stable tangent bases at the two cap centers.

        This version does not assume that the centers are on the equator.
        """
        centers = self._center_dirs

        z_axis = jnp.array([0.0, 0.0, 1.0])
        y_axis = jnp.array([0.0, 1.0, 0.0])

        dots_z = jnp.abs(centers @ z_axis)
        refs = jnp.where(dots_z[:, None] < 0.9, z_axis[None, :], y_axis[None, :])

        e1 = jnp.cross(refs, centers)
        e1 = e1 / jnp.linalg.norm(e1, axis=-1, keepdims=True)

        e2 = jnp.cross(centers, e1)
        e2 = e2 / jnp.linalg.norm(e2, axis=-1, keepdims=True)

        return e1, e2

    def _sample_selected_caps(self, rng: PRNGKey, cap_idx: Array) -> Array:
        """Sample uniformly from selected spherical caps.

        For a cap centered at the north pole:

            dS = R^2 sin(alpha) d_alpha d_phi.

        Therefore cos(alpha) is uniform on [cos(theta), 1].
        """
        n = cap_idx.shape[0]
        key_cos, key_phi = jr.split(rng)

        centers = self._center_dirs[cap_idx]
        theta = self._angular_radii[cap_idx]
        cos_theta = jnp.cos(theta)

        u = jr.uniform(key_cos, shape=(n,))
        cos_alpha = 1.0 - u * (1.0 - cos_theta)
        sin_alpha = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_alpha**2))

        phi = jr.uniform(
            key_phi,
            shape=(n,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
        )

        e1_all, e2_all = self._cap_tangent_bases()
        e1 = e1_all[cap_idx]
        e2 = e2_all[cap_idx]

        dirs = (
            cos_alpha[:, None] * centers
            + sin_alpha[:, None]
            * (
                jnp.cos(phi)[:, None] * e1
                + jnp.sin(phi)[:, None] * e2
            )
        )

        return self.sphere_radius * dirs

    def _cover_count(self, x: Array) -> Array:
        x_dir = x / self.sphere_radius
        dots = x_dir @ self._center_dirs.T
        inside = dots >= self._cos_radii[None, :]
        return jnp.sum(inside, axis=-1)

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        """Exact i.i.d. surface-uniform samples from the cap union."""
        if load:
            path = (
                SAMPLE_PATH
                / f"uniform_sphere_two_caps_"
                  f"{self.center_offset:.3f}offset_"
                  f"{self.r_small:.3f}small_"
                  f"{self.r_large:.3f}large_"
                  f"{n_samples}s.npy"
            )
            try:
                return jnp.asarray(onp.load(path))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load samples from {path}") from e

        rng, r0 = jr.split(rng)
        proto = self.base.sample(r0, 1)
        out = jnp.zeros((n_samples,) + proto.shape[1:], dtype=proto.dtype)

        bsz = int(min(max(2048, n_samples), 16384))

        def cond(state):
            _, _, curr = state
            return curr < n_samples

        def body(state):
            key, samples, curr = state
            key, k_idx, k_cap, k_overlap = jr.split(key, 4)

            # 1. Choose a cap proportional to its spherical surface area.
            idx = jr.categorical(k_idx, self._log_alpha, shape=(bsz,))

            # 2. Sample uniformly from that cap with respect to surface area.
            proposals = self._sample_selected_caps(k_cap, idx)

            # 3. Correct for overlap multiplicity.
            cover = self._cover_count(proposals)
            u_overlap = jr.uniform(k_overlap, shape=(bsz,))
            ok_overlap = u_overlap < (1.0 / jnp.maximum(cover, 1))

            # 4. Numerical feasibility check.
            ok_feas = self.feasible(proposals, self.domain_tol)

            ok = ok_overlap & ok_feas

            cs = jnp.cumsum(ok.astype(jnp.int32))
            sample_idx = curr + cs - 1
            sample_idx = jnp.where(
                ok & (sample_idx < n_samples),
                sample_idx,
                n_samples,
            )

            samples = samples.at[sample_idx].set(proposals, mode="drop")
            curr = jnp.minimum(n_samples, curr + cs[-1])

            return key, samples, curr

        _, out, _ = jax.lax.while_loop(cond, body, (rng, out, 0))
        return out

    def save_samples(self, samples: Array, n_samples: int | None = None) -> None:
        if n_samples is None:
            n_samples = int(samples.shape[0])

        path = (
            SAMPLE_PATH
            / f"uniform_sphere_two_caps_"
              f"{self.center_offset:.3f}offset_"
              f"{self.r_small:.3f}small_"
              f"{self.r_large:.3f}large_"
              f"{n_samples}s.npy"
        )

        onp.save(path, onp.asarray(samples))

    def _lambert_basis(self) -> tuple[Array, Array, Array]:
        # Bisector of the two cap centers; by symmetry of the placement
        # (centers on the equator, mirrored about +x), p = (1, 0, 0).
        p = jnp.array([1.0, 0.0, 0.0])
        e1 = jnp.array([0.0, 1.0, 0.0])
        e2 = jnp.array([0.0, 0.0, 1.0])
        return p, e1, e2

    def _flat_plot_coords(self, x: Array) -> Array:
        """Lambert azimuthal equal-area projection of sphere points to 2D.

        Uniform samples on the sphere stay uniform under this map, and a
        single global projection (rather than per-cap stitching) preserves
        the relative geometry of the two caps.
        """
        R = self.sphere_radius
        p, e1, e2 = self._lambert_basis()

        q = x / R
        cos_alpha = jnp.clip(q @ p, -1.0 + 1e-12, 1.0)

        tang = q - cos_alpha[:, None] * p[None, :]
        tang_norm = jnp.linalg.norm(tang, axis=-1, keepdims=True)
        tang_dir = tang / jnp.maximum(tang_norm, 1e-12)

        rho = 2.0 * R * jnp.sin(0.5 * jnp.arccos(cos_alpha))
        u = rho * (tang_dir @ e1)
        v = rho * (tang_dir @ e2)
        return jnp.stack([u, v], axis=-1)

    def _inverse_lambert(self, uv: Array) -> Array:
        """Inverse Lambert: 2D plane coords -> point on the sphere."""
        R = self.sphere_radius
        p, e1, e2 = self._lambert_basis()

        rho = jnp.linalg.norm(uv, axis=-1, keepdims=True)
        rho_safe = jnp.maximum(rho, 1e-12)

        alpha = 2.0 * jnp.arcsin(jnp.clip(rho / (2.0 * R), 0.0, 1.0))
        cos_alpha = jnp.cos(alpha)
        sin_alpha = jnp.sin(alpha)

        tang_dir = uv / rho_safe
        q = (
            cos_alpha * p[None, :]
            + sin_alpha
            * (
                tang_dir[:, 0:1] * e1[None, :]
                + tang_dir[:, 1:2] * e2[None, :]
            )
        )
        return R * q

    def _cap_boundary_2d(self, n_phi: int = 256) -> Array:
        """Lambert image of the cap boundaries, used for tight view bounds."""
        R = self.sphere_radius
        e1_all, e2_all = self._cap_tangent_bases()
        phi = jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False)
        cphi = jnp.cos(phi)[:, None]
        sphi = jnp.sin(phi)[:, None]

        out = []
        for k in range(2):
            mu = self._center_dirs[k]
            theta = self._angular_radii[k]
            boundary = R * (
                jnp.cos(theta) * mu[None, :]
                + jnp.sin(theta)
                * (cphi * e1_all[k][None, :] + sphi * e2_all[k][None, :])
            )
            out.append(self._flat_plot_coords(boundary))
        return jnp.concatenate(out, axis=0)

    def plot2d(
        self,
        samples: Array = None,
        ax: Axes = None,
        bounds: Optional[tuple[float | int, float | int]] = None,
        bounds_2: Optional[tuple[float | int, float | int]] = None,
        grid_width_n_points: int = 500,
        grid_width_n_points_2: int = 500,
        n_contour_levels: int = None,
        plot_marginal_dims: tuple[int] = (0, 1),
        markersize: float = 4.0,
        alpha: float = 0.6,
        cmap_alpha: float = 0.15,
        plt_show: bool = True,
        title: str = None,
        filled: bool = True,
        cmap: str = "Blues",
        show_support: bool = True,
        show_manifold: bool = True,
    ):
        """Area-preserving 2D visualization via Lambert azimuthal equal-area.

        Single global projection from the bisector of the cap centers, so
        uniform samples on the cap union appear uniform in the plane.
        Off-bisector caps map to ovals (rather than circles): this is the
        honest curvature signature of the underlying spherical embedding.

        With ``show_manifold=True``, also draws the Lambert image of the
        sphere itself (a circle of radius 2R) as a thin grey line, and
        widens the view to include it.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1)

        R = self.sphere_radius
        R_manifold = jnp.sqrt(2.0) * R

        boundary_2d = onp.asarray(self._cap_boundary_2d())
        bx_min, bx_max = float(boundary_2d[:, 0].min()), float(boundary_2d[:, 0].max())
        by_min, by_max = float(boundary_2d[:, 1].min()), float(boundary_2d[:, 1].max())
        pad = 0.06 * max(bx_max - bx_min, by_max - by_min)

        if show_manifold:
            bx_min = min(bx_min, -R_manifold)
            bx_max = max(bx_max, R_manifold)
            by_min = min(by_min, -R_manifold)
            by_max = max(by_max, R_manifold)

        if bounds is None:
            x_min, x_max = bx_min - pad, bx_max + pad
        else:
            x_min, x_max = bounds

        if bounds_2 is None:
            if bounds is None:
                y_min, y_max = by_min - pad, by_max + pad
            else:
                y_min, y_max = bounds
        else:
            y_min, y_max = bounds_2

        if grid_width_n_points_2 is None:
            grid_width_n_points_2 = grid_width_n_points

        if show_manifold:
            phi = onp.linspace(0.0, 2.0 * onp.pi, 360)
            ax.plot(
                R_manifold * onp.cos(phi),
                R_manifold * onp.sin(phi),
                color="black",
                linewidth=1.5,
                zorder=0,
            )

        if show_support:
            xs = onp.linspace(x_min, x_max, grid_width_n_points)
            ys = onp.linspace(y_min, y_max, grid_width_n_points_2)
            XX, YY = onp.meshgrid(xs, ys)
            UV = onp.stack([XX.ravel(), YY.ravel()], axis=-1)

            # Lift each pixel back to the sphere; test the cap inequalities there.
            Q = onp.asarray(self._inverse_lambert(jnp.asarray(UV)))
            dots = (Q / R) @ onp.asarray(self._center_dirs).T
            inside = onp.any(
                dots >= onp.asarray(self._cos_radii)[None, :],
                axis=-1,
            )
            inside = inside.reshape(XX.shape).astype(float)

            if filled:
                ax.contourf(
                    XX,
                    YY,
                    inside,
                    levels=[0.5, 1.5],
                    colors=["green"],
                    alpha=cmap_alpha,
                    zorder=1,
                )

            ax.contour(
                XX,
                YY,
                inside,
                levels=[0.5],
                colors="black",
                linewidths=1.5,
                linestyles="dashed",
                zorder=2,
            )

        if samples is not None:
            coords = onp.asarray(self._flat_plot_coords(samples))
            coords_x = onp.clip(
                coords[:, plot_marginal_dims[0]],
                x_min,
                x_max,
            )
            coords_y = onp.clip(
                coords[:, plot_marginal_dims[1]],
                y_min,
                y_max,
            )

            ax.scatter(
                coords_x,
                coords_y,
                s=markersize,
                alpha=alpha,
                zorder=3,
                c="#2688c9", edgecolors="#1f77b3"
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        if title is not None:
            ax.set_title(title)

        if plt_show:
            plt.show()

        return ax

    def plot3d(
        self,
        samples: Array = None,
        ax: Axes = None,
        sphere_grid_n: int = 160,
        markersize: float = 4.0,
        alpha: float = 0.75,
        manifold_alpha: float = 0.08,
        support_alpha: float = 0.55,
        wire_alpha: float = 0.2,
        wire_lw: float = 0.2,
        show_full_sphere: bool = True,
        show_support: bool = True,
        show_samples: bool = True,
        show_centers: bool = False,
        show_axes: bool = False,
        show_wireframe: bool = False,
        elev: float = 10.0,
        azim: float = 0.0,
        title: str = None,
        plt_show: bool = True,
        surface_shrink: float = 0.03,
        rasterize: bool = True,
    ):
        """3D visualization of the spherical-cap benchmark.

        This plot is intended for diagnostics and paper figures.

        It shows:
            - the sphere manifold as a faint translucent surface,
            - the feasible cap union as a green surface patch,
            - samples in their true ambient R^3 coordinates.

        Unlike the flat plot, this does not remap or clip samples.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(4.0, 4.0))
            ax = fig.add_subplot(111, projection="3d")

        R = self.sphere_radius
        R_plot = R * (1.0 - surface_shrink)

        lon = onp.linspace(-onp.pi, onp.pi, sphere_grid_n)
        lat = onp.linspace(-0.5 * onp.pi, 0.5 * onp.pi, sphere_grid_n // 2)
        Lon, Lat = onp.meshgrid(lon, lat)

        # Shrunken surfaces for plotting.
        X = R_plot * onp.cos(Lat) * onp.cos(Lon)
        Y = R_plot * onp.cos(Lat) * onp.sin(Lon)
        Z = R_plot * onp.sin(Lat)

        # True-radius grid for feasibility mask.
        X_true = R * onp.cos(Lat) * onp.cos(Lon)
        Y_true = R * onp.cos(Lat) * onp.sin(Lon)
        Z_true = R * onp.sin(Lat)

        grid_true = onp.stack(
            [X_true.ravel(), Y_true.ravel(), Z_true.ravel()],
            axis=-1,
        )
        feasible = onp.asarray(self.feasible(jnp.asarray(grid_true), self.domain_tol))
        feasible = feasible.reshape(X.shape)

        # ------------------------------------------------------------------
        # Full sphere manifold.
        # ------------------------------------------------------------------
        if show_full_sphere:
            ax.plot_surface(
                X,
                Y,
                Z,
                rstride=2,
                cstride=2,
                color="lightgray",
                alpha=manifold_alpha,
                linewidth=0.0,
                antialiased=True,
                shade=True,
                zorder=0,
                rasterized=rasterize
            )

        # ------------------------------------------------------------------
        # Feasible spherical-cap union.
        # ------------------------------------------------------------------
        if show_support:
            Xs = onp.where(feasible, X, onp.nan)
            Ys = onp.where(feasible, Y, onp.nan)
            Zs = onp.where(feasible, Z, onp.nan)

            ax.plot_surface(
                Xs,
                Ys,
                Zs,
                rstride=1,
                cstride=1,
                color="green",
                alpha=support_alpha,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                zorder=1,
                rasterized=rasterize
            )

        # ------------------------------------------------------------------
        # Samples in true ambient coordinates.
        # ------------------------------------------------------------------
        if samples is not None and show_samples:
            s = onp.asarray(samples)
            ax.scatter(
                s[:, 0],
                s[:, 1],
                s[:, 2],
                s=markersize,
                alpha=alpha,
                c="none",
                edgecolors="none",
                depthshade=False,
                zorder=10,  # TODO: how to make sure they are in foreground?
                rasterized=rasterize
            )

        # ------------------------------------------------------------------
        # Optional cap centers.
        # ------------------------------------------------------------------
        if show_centers:
            centers = R * onp.asarray(self._center_dirs)

            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                s=60,
                marker="x",
                c="black",
                linewidths=2.0,
                depthshade=False,
                zorder=4,
            )

        # wireframe
        if show_wireframe:
            ax.plot_wireframe(
                X,
                Y,
                Z,
                rstride=max(1, X.shape[0] // 12),
                cstride=max(1, X.shape[1] // 12),
                color="0.55",
                linewidth=wire_lw,
                alpha=wire_alpha,
                zorder=0,
                rasterized=True,
            )

        # ------------------------------------------------------------------
        # Camera and equal 3D aspect.
        # ------------------------------------------------------------------
        ax.view_init(elev=elev, azim=azim)

        lim = 1.05 * R
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1.0, 1.0, 1.0))

        if not show_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$x_3$")

        if title is not None:
            ax.set_title(title, pad=0)

        if plt_show:
            plt.show()

        return ax


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    rng = jr.PRNGKey(0)

    # Connected: center_offset < r_small + r_large
    connected = UniformBall2()
    samples_connected = connected.sample(rng, n_samples=2_000)
    connected.plot2d(samples_connected)
    connected.plot3d(
        samples_connected,
        elev=24,
        azim=-55,
        title="Connected",
    )
    connected.save_samples(samples_connected)

    rng, subkey = jr.split(rng)
    disconnected = UniformBall2(
        center_offset=1.35,
    )
    samples_disconnected = disconnected.sample(subkey, n_samples=2_000)
    disconnected.plot2d(samples_disconnected)
    disconnected.plot3d(
        samples_disconnected,
        elev=24,
        azim=-55,
        title="Disconnected",
    )
    disconnected.save_samples(samples_disconnected)
