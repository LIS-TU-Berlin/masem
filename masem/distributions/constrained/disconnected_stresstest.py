from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from chex import Array, PRNGKey

from masem.constants import SAMPLE_PATH
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.distributions.constraint import Constraint, ConstraintSet
from masem.distributions.unconstrained.uniform import Uniform


class NullspaceUnionSpheresEq(Constraint):
    """Union-of-spheres equality in nullspace coordinates."""

    def __init__(self, x_perp: Array, nsp_basis: Array, centers_z: Array, radii: Array):
        self.x_perp = jnp.asarray(x_perp)
        self.nsp_basis = jnp.asarray(nsp_basis)
        self.centers_z = jnp.asarray(centers_z)
        self.radii = jnp.asarray(radii)

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        z = (x - self.x_perp) @ self.nsp_basis
        dists = z[:, None, :] - self.centers_z[None, :, :]
        d2 = jnp.sum(dists ** 2, axis=-1) - self.radii[None, :] ** 2

        # Pick nearest sphere equation by absolute residual.
        idx = jax.lax.stop_gradient(jnp.argmin(jnp.abs(d2), axis=-1))
        out = d2[jnp.arange(d2.shape[0]), idx]

        return out if batched else out[0]


class NullspaceOutsideBallIneq(Constraint):
    """Ball-cutout in nullspace: feasible region is ||z - c||^2 >= r^2."""

    def __init__(self, x_perp: Array, nsp_basis: Array, center_z: Array, radius: float):
        self.x_perp = jnp.asarray(x_perp)
        self.nsp_basis = jnp.asarray(nsp_basis)
        self.center_z = jnp.asarray(center_z)
        self.radius = float(radius)

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        z = (x - self.x_perp) @ self.nsp_basis
        out = self.radius ** 2 - jnp.sum((z - self.center_z) ** 2, axis=-1)

        return out if batched else out[0]


class LinearEqsVec(Constraint):
    """Vectorized linear equality A x - b = 0, returning all n_eq residuals at once.

    Replaces a list of scalar LineEq instances; collapses the JIT trace from
    n_eq separate value()/grad() calls to a single matmul.
    """

    def __init__(self, A: Array, b: Array):
        self.A = jnp.asarray(A)
        self.b = jnp.asarray(b)
        self.n_eq = int(self.A.shape[0])
        self.dim = int(self.A.shape[1])

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        return x @ self.A.T - self.b

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        if x.ndim == 1:
            return self.A
        return jnp.broadcast_to(
            self.A[None, :, :], (x.shape[0], self.n_eq, self.dim)
        )


class BoundsBoxIneq(Constraint):
    """Vectorized axis-aligned box: lower <= x_i <= upper for all i.

    Returns 2*dim residuals. Replaces 2*dim HalfspaceIneq instances.
    """

    def __init__(self, lower: float, upper: float, dim: int):
        self.lower = float(lower)
        self.upper = float(upper)
        self.dim = int(dim)
        I = jnp.eye(self.dim)
        self._J = jnp.concatenate([I, -I], axis=0)  # (2*dim, dim)

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        return jnp.concatenate([x - self.upper, self.lower - x], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._J
        return jnp.broadcast_to(
            self._J[None, :, :], (x.shape[0], 2 * self.dim, self.dim)
        )


class DisconnectedStress(ConstrainedDistribution):

    def __init__(
        self,
        *,
        dim: int = 10,
        n_eq: int = 7,
        n_ineq: int = 0,
        n_components: int = 5,
        radius: float = 0.25,
        radius_jitter: float = 0.5,
        center_scale: float = 0.2,
        obstacle_radius_frac: float = 0.5,
        seed: int = 123,
        bounds: tuple[float, float] = (-28., 28.),
        random_orientation: bool = True,
        center_mode: str = "random",
        affine_offset: bool = False,
        affine_offset_scale: float = 0.1,
    ):
        """Construct the disconnected stress benchmark.

        Parameters
        ----------
        dim:
            Ambient dimension.

        n_eq:
            Number of linear equality constraints. The nullspace dimension is
                p = dim - n_eq.
            The sphere surface dimension is p - 1.

        n_ineq:
            Number of optional ball cutouts.

        n_components:
            Number of disconnected sphere components.

        radius:
            Base radius of each sphere.

        radius_jitter:
            Relative random jitter in sphere radii. Use 0.0 for clean scaling.

        center_scale:
            If center_mode="ring", this is the radius of the ring on which
            component centers are placed in latent coordinates.
            If center_mode="random", this is the scale of uniform.

        obstacle_radius_frac:
            Cutout radius as a fraction of the corresponding component radius.

        bounds:
            Fixed ambient box bounds. Unlike the old version, these do not grow
            with dim.

        random_orientation:
            If False, the nullspace is the coordinate subspace x[:p] and the
            linear constraints are x[p:] = 0.

            If True, the same latent geometry is randomly rotated into R^dim.

        center_mode:
            "ring" gives deterministic, non-overlapping, dimension-stable
            centers.
            "random" uses rejection placement.

        affine_offset:
            If False, uses A x = 0, so x_perp = 0.
            If True, uses a small random offset b. This is less scaling-friendly
            because ||x_perp|| can grow with n_eq.

        affine_offset_scale:
            Scale of b if affine_offset=True.
        """
        assert dim > n_eq, "Need nullspace dimension p = dim - n_eq >= 1"

        p = dim - n_eq
        key = jr.PRNGKey(seed)
        k_embed, kb, kr, kc, ko = jr.split(key, 5)

        lower, upper = bounds
        bounds_arr = jnp.asarray([lower, upper])

        # ------------------------------------------------------------------
        # Embedding: construct A x = b and nullspace basis.
        # ------------------------------------------------------------------
        if n_eq > 0:
            if random_orientation:
                Q, _ = jnp.linalg.qr(jr.normal(k_embed, (dim, dim)))
                nsp_basis = Q[:, :p]       # (dim, p)
                A = Q[:, p:].T             # (n_eq, dim)
            else:
                nsp_basis = jnp.concatenate(
                    [
                        jnp.eye(p),
                        jnp.zeros((dim - p, p)),
                    ],
                    axis=0,
                )
                A = jnp.concatenate(
                    [
                        jnp.zeros((dim - p, p)),
                        jnp.eye(dim - p),
                    ],
                    axis=1,
                )

            if affine_offset:
                b = jr.normal(kb, (n_eq,)) * affine_offset_scale
                # x_perp = A.T @ b is the minimum-norm solution.
                x_perp = A.T @ b
            else:
                b = jnp.zeros((n_eq,))
                x_perp = jnp.zeros(dim)

        else:
            A = jnp.zeros((0, dim))
            b = jnp.zeros((0,))
            x_perp = jnp.zeros(dim)
            nsp_basis = jnp.eye(dim)

        # ------------------------------------------------------------------
        # Radii.
        # ------------------------------------------------------------------
        jitter = jr.uniform(
            kr,
            (n_components,),
            minval=-radius_jitter,
            maxval=radius_jitter,
        )
        radii = jnp.maximum(radius * (1.0 + jitter), 1e-6)
        #print(radii, jitter)

        # ------------------------------------------------------------------
        # Component centers in nullspace coordinates.
        # ------------------------------------------------------------------
        if center_mode == "ring":
            centers_z = self._ring_centers(
                n_components=n_components,
                p=p,
                radii=radii,
                center_norm=float(center_scale),
            )
        elif center_mode == "random":
            centers_z = self._place_centers(
                kc,
                p=p,
                radii=radii,
                scale=float(center_scale),
            )
        else:
            raise ValueError(
                f"Unknown center_mode={center_mode!r}. "
                "Expected 'ring' or 'random'."
            )

        # ------------------------------------------------------------------
        # Validate that the target fits in the fixed box.
        # ------------------------------------------------------------------
        max_latent_extent = float(jnp.max(jnp.linalg.norm(centers_z, axis=-1) + radii))
        max_offset = float(jnp.linalg.norm(x_perp))

        # For random orientations this is conservative but fine. Since N has
        # orthonormal columns, ||N z|| = ||z||, so each coordinate is bounded by
        # the vector norm. For coordinate orientation, this is also safe.
        required_abs_bound = max_offset + max_latent_extent
        if required_abs_bound > min(abs(float(lower)), abs(float(upper))):
            raise ValueError(
                "The target may not fit inside the requested fixed bounds. "
                f"Need approximately +/-{required_abs_bound:.4f}, "
                f"but got bounds={bounds}. Increase bounds, decrease "
                "center_scale, decrease radius, or set affine_offset=False."
            )

        # ------------------------------------------------------------------
        # Optional ball obstacles / cutouts.
        # ------------------------------------------------------------------
        if n_ineq > 0:
            k_comp, k_dir = jr.split(ko)
            comp_idx = jr.randint(k_comp, (n_ineq,), 0, n_components)

            dirs = jr.normal(k_dir, (n_ineq, p))
            dirs /= jnp.maximum(
                jnp.linalg.norm(dirs, axis=-1, keepdims=True),
                1e-30,
            )

            # Place obstacle centers on sphere surfaces.
            obs_centers = centers_z[comp_idx] + radii[comp_idx, None] * dirs
            obs_radii = obstacle_radius_frac * radii[comp_idx]
        else:
            obs_centers = jnp.zeros((0, p))
            obs_radii = jnp.zeros((0,))

        # ------------------------------------------------------------------
        # Assemble constraints.
        # ------------------------------------------------------------------
        lin_eqs = [LinearEqsVec(A, b)] if n_eq > 0 else []

        union_eq = NullspaceUnionSpheresEq(
            x_perp=x_perp,
            nsp_basis=nsp_basis,
            centers_z=centers_z,
            radii=radii,
        )

        ineqs = [
            NullspaceOutsideBallIneq(
                x_perp=x_perp,
                nsp_basis=nsp_basis,
                center_z=obs_centers[j],
                radius=float(obs_radii[j]),
            )
            for j in range(n_ineq)
        ]

        cons = ConstraintSet(
            [
                *lin_eqs,
                union_eq,
                *ineqs,
                BoundsBoxIneq(float(lower), float(upper), dim),
            ]
        )

        base = Uniform(bounds=bounds_arr, dim=dim)
        base.log_prob_min = -10_000.0

        super().__init__(base=base, constraints=cons)

        # ------------------------------------------------------------------
        # Metadata.
        # ------------------------------------------------------------------
        self.dim = dim
        self.p = p
        self.intrinsic_dim = p - 1
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.n_components = n_components
        self.radius = radius
        self.radius_jitter = radius_jitter
        self.center_scale = center_scale
        self.obstacle_radius_frac = obstacle_radius_frac
        self.seed = seed
        self.random_orientation = random_orientation
        self.center_mode = center_mode
        self.affine_offset = affine_offset
        self.affine_offset_scale = affine_offset_scale

        self.x_perp = x_perp
        self.nsp_basis = nsp_basis
        self.A = A
        self.b = b
        self.centers_z = centers_z
        self.radii = radii
        self.obs_centers = obs_centers
        self.obs_radii = obs_radii
        self.bounds = bounds_arr

        log_w = (p - 1) * jnp.log(radii + 1e-30)
        self.log_component_weights = log_w - jsp.special.logsumexp(log_w)
        self.component_weights = jnp.exp(self.log_component_weights)

    @property
    def init_scale(self) -> float:
        return float(self.bounds[1] - self.bounds[0]) / 4.0

    @staticmethod
    def _ring_centers(
        *,
        n_components: int,
        p: int,
        radii: Array,
        center_norm: float,
    ) -> Array:
        """Deterministically place centers on a ring.

        This is the recommended placement for scaling experiments because the
        geometry remains comparable as dim changes.

        Requires p >= 2.
        """
        if p < 2:
            raise ValueError(
                "center_mode='ring' requires nullspace dimension p >= 2. "
                "Use p >= 2, i.e. dim - n_eq >= 2."
            )

        if n_components < 1:
            raise ValueError("n_components must be positive.")

        angles = jnp.linspace(
            0.0,
            2.0 * jnp.pi,
            n_components,
            endpoint=False,
        )

        centers = jnp.zeros((n_components, p))
        centers = centers.at[:, 0].set(center_norm * jnp.cos(angles))
        centers = centers.at[:, 1].set(center_norm * jnp.sin(angles))

        if n_components > 1:
            # Adjacent center distance on the ring.
            min_center_dist = 2.0 * center_norm * jnp.sin(jnp.pi / n_components)

            # Conservative pairwise non-overlap condition.
            max_pair_radius_sum = 2.0 * float(jnp.max(radii))

            if float(min_center_dist) <= max_pair_radius_sum:
                raise ValueError(
                    "Sphere components overlap or touch. "
                    f"Minimum ring center distance is {float(min_center_dist):.4f}, "
                    f"while conservative radius sum is {max_pair_radius_sum:.4f}. "
                    "Decrease radius, decrease n_components, or increase center_scale."
                )

        return centers

    def _place_centers(self, key: PRNGKey, *, p: int, radii: Array, scale: float) -> Array:
        """Randomly place K sphere centers with pairwise separation.

        This is closer to the old behavior. It is less ideal for clean scaling
        because random geometry can vary across dimension.
        """
        K = int(radii.shape[0])
        centers = []

        for i in range(K):
            for _ in range(50_000):
                key, sub = jr.split(key)
                #c = jr.normal(sub, (p,)) * scale
                c = jr.uniform(
                    sub, (p,), minval=self.bounds[0] * scale, maxval=self.bounds[1] * scale
                )

                if all(
                    float(jnp.linalg.norm(c - c_old))
                    > 2.0 * (float(radii[i]) + float(radii[j]))
                    for j, c_old in enumerate(centers)
                ):
                    centers.append(c)
                    break
            else:
                raise RuntimeError(
                    f"Could not place component {i}. "
                    "Increase center_scale or decrease n_components."
                )

        return jnp.stack(centers)

    def latent_coords(self, samples: Array) -> Array:
        """Map ambient samples to latent/nullspace coordinates."""
        samples = jnp.atleast_2d(samples)
        return (samples - self.x_perp) @ self.nsp_basis

    def _sample_full_union_proposal(self, rng: PRNGKey, n: int) -> tuple[Array, Array]:
        """Sample from the uncut union of spheres and return component labels."""
        k_idx, k_dir = jr.split(rng)

        idx = jr.categorical(
            k_idx,
            self.log_component_weights,
            shape=(n,),
        )

        g = jr.normal(k_dir, (n, self.p))
        u = g / jnp.maximum(
            jnp.linalg.norm(g, axis=-1, keepdims=True),
            1e-30,
        )

        z = self.centers_z[idx] + self.radii[idx, None] * u
        x = self.x_perp[None, :] + z @ self.nsp_basis.T

        return x, idx

    def _propose(self, rng: PRNGKey, n: int) -> Array:
        x, _ = self._sample_full_union_proposal(rng, n)
        return x

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        """Exact i.i.d. samples.

        If n_ineq == 0, sampling is direct.

        If n_ineq > 0, samples are obtained by rejection from the sphere union.
        """
        if load:
            path = (
                SAMPLE_PATH
                / f"disconnected_stress_{n_samples}s"
                  f"_{self.dim}d"
                  f"_{self.n_eq}eq"
                  f"_{self.n_ineq}ineq"
                  f"_{self.n_components}K"
                  f"_{self.center_mode}"
                  f"_{'rot' if self.random_orientation else 'coord'}.npy"
            )
            return jnp.load(path)

        if self.n_ineq == 0:
            return self._propose(rng, n_samples)

        bsz = min(max(2048, n_samples), 16384)
        out = jnp.zeros((n_samples, self.dim))

        def cond(state):
            _, _, curr = state
            return curr < n_samples

        def body(state):
            key, samples, curr = state
            key, sub = jr.split(key)

            proposals = self._propose(sub, bsz)
            ok = self.feasible(proposals, self.domain_tol)

            cs = jnp.cumsum(ok.astype(jnp.int32))

            sample_idx = jnp.where(
                ok & (curr + cs - 1 < n_samples),
                curr + cs - 1,
                n_samples,
            )

            samples = samples.at[sample_idx].set(proposals, mode="drop")
            curr = jnp.minimum(n_samples, curr + cs[-1])

            return key, samples, curr

        _, out, _ = jax.lax.while_loop(
            cond,
            body,
            (rng, out, jnp.array(0, jnp.int32)),
        )

        return out

    def save_samples(self, samples: Array) -> None:
        path = (
            SAMPLE_PATH
            / f"disconnected_stress_{samples.shape[0]}s"
              f"_{self.dim}d"
              f"_{self.n_eq}eq"
              f"_{self.n_ineq}ineq"
              f"_{self.n_components}K"
              f"_{self.center_mode}"
              f"_{'rot' if self.random_orientation else 'coord'}.npy"
        )
        jnp.save(path, samples)

    def estimate_component_weights(self, rng: PRNGKey, n_mc: int = 200_000) -> Array:
        if self.n_ineq == 0:
            return self.component_weights

        x, idx = self._sample_full_union_proposal(rng, n_mc)
        ok = self.feasible(x, self.domain_tol)

        counts = jnp.bincount(idx[ok], length=self.n_components)
        total = jnp.sum(counts)

        return counts / jnp.maximum(total, 1)

    def component_index(self, samples: Array) -> Array:
        samples = jnp.atleast_2d(samples)
        z = (samples - self.x_perp) @ self.nsp_basis
        dz = z[:, None, :] - self.centers_z[None, :, :]
        boundary_err = jnp.abs(
            jnp.sum(dz * dz, axis=-1) - self.radii[None, :] ** 2
        )
        return jnp.argmin(boundary_err, axis=-1)

    def empirical_component_frequencies(self, samples: Array) -> Array:
        idx = self.component_index(samples)
        counts = jnp.bincount(idx, length=self.n_components)
        return counts / jnp.maximum(samples.shape[0], 1)

    @partial(jax.jit, static_argnums=(0,))
    def component_kl_tv(self, samples: Array, true_weights: Array | None = None) -> tuple[Array, Array]:
        """KL(empirical || true) over component labels.
        For n_ineq == 0, true_weights defaults to analytic r_i^(p-1) weights.
        For n_ineq > 0, pass estimate_component_weights(...) as true_weights.
        """
        q = self.empirical_component_frequencies(samples)
        p = self.component_weights if true_weights is None else true_weights
        return jnp.sum(q * (jnp.log(q + 1e-30) - jnp.log(p + 1e-30))), 0.5 * jnp.sum(jnp.abs(q - p))

    def plot2d(
        self,
        samples: Array = None,
        ax=None,
        bounds=None,
        plot_marginal_dims: tuple[int, int] = (0, 1),
        plot_latent: bool = True,
        markersize: float = 2.0,
        alpha: float = 0.6,
        plt_show: bool = True,
        title: str | None = None,
        n_component_points: int = 2_000,
        component_alpha: float = 0.08,
        **kwargs,
    ):
        """Plot latent coordinates or ambient coordinate marginals.

        For high-dimensional scaling plots, plot_latent=True is usually the
        most interpretable because it shows the fixed target geometry.
        """
        import matplotlib.pyplot as plt
        import numpy as onp

        if ax is None:
            _, ax = plt.subplots(1, 1)

        if bounds is None:
            bounds = self.bounds

        rng = jr.PRNGKey(0)

        if plot_latent:
            if self.p < 2:
                raise ValueError("Latent 2D plot requires p >= 2.")

            d0, d1 = 0, 1

            for i in range(self.n_components):
                rng, sub = jr.split(rng)

                g = jr.normal(sub, (n_component_points, self.p))
                u = g / jnp.maximum(
                    jnp.linalg.norm(g, axis=-1, keepdims=True),
                    1e-30,
                )

                z = self.centers_z[i] + self.radii[i] * u

                if self.n_ineq > 0:
                    x = self.x_perp[None, :] + z @ self.nsp_basis.T
                    ok = self.feasible(x, self.domain_tol)
                    z = z[ok]

                if z.shape[0] == 0:
                    continue

                pts = onp.asarray(z[:, [d0, d1]])

                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=0.5,
                    alpha=component_alpha,
                    color="black",
                    linewidths=0,
                    zorder=1,
                )

            if samples is not None:
                z_samples = self.latent_coords(samples)
                z_np = onp.asarray(z_samples)
                comp_idx = onp.asarray(self.component_index(samples))

                ax.scatter(
                    z_np[:, d0],
                    z_np[:, d1],
                    s=markersize,
                    alpha=alpha,
                    c=comp_idx,
                    linewidths=0.3,
                    zorder=3,
                )

            max_extent = float(jnp.max(jnp.linalg.norm(self.centers_z, axis=-1) + self.radii))
            lim = max_extent + 0.25
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(r"$z_0$")
            ax.set_ylabel(r"$z_1$")

        else:
            d0, d1 = plot_marginal_dims

            for i in range(self.n_components):
                rng, sub = jr.split(rng)

                g = jr.normal(sub, (n_component_points, self.p))
                u = g / jnp.maximum(
                    jnp.linalg.norm(g, axis=-1, keepdims=True),
                    1e-30,
                )

                z = self.centers_z[i] + self.radii[i] * u
                x = self.x_perp[None, :] + z @ self.nsp_basis.T

                if self.n_ineq > 0:
                    ok = self.feasible(x, self.domain_tol)
                    x = x[ok]

                if x.shape[0] == 0:
                    continue

                pts = onp.asarray(x[:, [d0, d1]])

                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=0.5,
                    alpha=component_alpha,
                    color="black",
                    linewidths=0,
                    zorder=1,
                )

            if samples is not None:
                samples_np = onp.asarray(jnp.atleast_2d(samples))
                comp_idx = onp.asarray(self.component_index(samples))

                ax.scatter(
                    samples_np[:, d0],
                    samples_np[:, d1],
                    s=markersize,
                    alpha=alpha,
                    c=comp_idx,
                    linewidths=0.3,
                    zorder=3,
                )

            ax.set_xlim(float(bounds[0]), float(bounds[1]))
            ax.set_ylim(float(bounds[0]), float(bounds[1]))
            ax.set_xlabel(f"$x_{{{d0}}}$")
            ax.set_ylabel(f"$x_{{{d1}}}$")

        ax.set_aspect("equal", adjustable="box")

        if title is not None:
            ax.set_title(title)

        if plt_show:
            plt.show()

        return ax


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # dim = 200
    for dim in [10]:
        intrinsic_dim = 2
        p = intrinsic_dim + 1
        # n_eq = 5
        n_eq = dim - p
        n_ineq = 5
        n_comp = 5
        print(dim, n_eq, n_ineq, n_comp)
        rng = jr.PRNGKey(0)
        dist = DisconnectedStress(dim=dim, n_eq=n_eq, n_components=n_comp, n_ineq=n_ineq)
        samples = dist.sample(rng, n_samples=2_000)
        dist.plot2d(samples, plt_show=True)
        dist.save_samples(samples)
