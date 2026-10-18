from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from jax import random
from jax.nn import relu
from matplotlib.axes import Axes

from komino.constants import SAMPLE_PATH
from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.distributions.constraint import Constraint, ConstraintSet
from komino.distributions.unconstrained.uniform import Uniform
from komino.distributions.constrained.halfspace import get_bound_contraints
from komino.utils import safe_norm


# Safety factor applied to the empirical sup of the area element used as rejection envelope.
ENVELOPE_SAFETY = 1.25
ENVELOPE_PROBES = 200_000


def poly_from_coeffs(coeffs: Array, u: Array) -> Array:
    """Evaluate the graph polynomial at a single chart point.

    Args:
        coeffs: (m, degree + 1, 2) polynomial coefficients.
        u: (2,) chart coordinates.

    Returns:
        (m,) graph coordinates, one product of two univariate polynomials per axis.
    """
    powers = jnp.arange(1, coeffs.shape[1])                                          # (degree,)
    u_powers = u[None, :] ** powers[:, None]                                         # (degree, 2)
    poly_axis = jnp.sum(u_powers[None] * coeffs[:, 1:, :], axis=1) + coeffs[:, 0, :]  # (m, 2)
    return poly_axis[:, 0] * poly_axis[:, 1]                                          # (m,)


def area_element_from_coeffs(coeffs: Array, u: Array) -> Array:
    """Volume distortion sqrt(det(I + Jp^T Jp)) of the graph lift u -> (u, p(u)).

    This is the co-area factor of the equality constraint: in the rotated frame the
    constraint Jacobian is Jh = [-Jp, I], so det(Jh Jh^T) = det(I + Jp^T Jp). Sampling
    uniformly in the chart therefore yields density proportional to 1 / this factor
    w.r.t. the surface measure on the manifold, which is why ground-truth samples need
    to be reweighted by it. Note this is a first-derivative quantity: curvature enters
    the ambient tube volume only at O(eps^2), and is not the relevant correction.

    Args:
        coeffs: (m, degree + 1, 2) polynomial coefficients.
        u: (2,) chart coordinates.

    Returns:
        Scalar area element at u.
    """
    jac = jax.jacfwd(poly_from_coeffs, argnums=1)(coeffs, u)  # (m, 2)
    g = jnp.eye(2) + jac.T @ jac                              # (2, 2)
    return jnp.sqrt(jnp.linalg.det(g))


class EmbeddedDiskConst(Constraint):
    def __init__(self,
            key: PRNGKey,
            radius: float = 0.3,
            bounds: tuple[float] = (-1.5,1.5),
            dim: int = 10):
        key_matrix, key_center = random.split(key)
        self.rotate = random.orthogonal(key_matrix, dim)
        self.center = random.uniform(key_center, (1,2),
                        minval=bounds[0], maxval=bounds[1])
        self.radius = radius

    @property
    def is_equality(self):
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array):
        if x.ndim == 1:
            x = x[None, :]
        x_rotated = (x @ self.rotate)
        disk_constr = safe_norm(x_rotated[:, :2] - self.center, axis=1, keepdims=True) - self.radius  # noqa: E501
        manifold_constr = safe_norm(x_rotated[:, 2:], axis=1, keepdims=True)
        return relu(disk_constr) + manifold_constr


class EmbeddedDisksNonlinConst(Constraint):
    """Embedd plane as graph of polyniomial, then rotate."""
    def __init__(
            self,
            key: PRNGKey,
            radius: float = 0.3,
            bounds: tuple[float] = (-0.7, 0.7),
            dim: int = 10,
            degree: int = 3
        ):
        key_matrix, key_center, key_poly = random.split(key, 3)
        self.rotate = random.orthogonal(key_matrix, dim)
        self.center = random.uniform(key_center, (1,2),
                        minval=bounds[0], maxval=bounds[1])
        self.poly_coeffs = random.uniform(key_poly, (dim-2, degree + 1, 2), minval=-1.0, maxval=1.0)
        self.radius = radius
    
    @partial(jax.jit, static_argnums=(0,))
    def _poly(self, x):
        # x: (n, 2)
        # coeffs: (dim-2, degree + 1, 2)
        # output: (n, dim-2)
        powers = jnp.arange(1, self.poly_coeffs.shape[1])  # (degree,)
        x_powers = x[:, None, :] ** powers[None, :, None]  # (n, degree, 2)
        poly_axis = jnp.sum(x_powers[:, None, :, :] * self.poly_coeffs[None, :, 1:, :], axis=2)\
              + self.poly_coeffs[None, :, 0, :]  # (n, dim-2, 2)
        poly = poly_axis[:, :, 0] * poly_axis[:, :, 1]  # (n, dim-2)
        return poly

    @property
    def is_equality(self):
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array):
        if x.ndim == 1:
            x = x[None, :]
        x_rotated = (x @ self.rotate)
        poly_graph = self._poly(x_rotated[:, :2])
        disk_constr = safe_norm(x_rotated[:, :2] - self.center, axis=1, keepdims=True) - self.radius
        manifold_constr = safe_norm(x_rotated[:, 2:] - poly_graph, axis=1, keepdims=True)
        return relu(disk_constr) + manifold_constr

    @partial(jax.jit, static_argnums=(0,))
    def _poly_single(self, x: Array):
        # x: (2,) -> (dim-2,)
        return self._poly(x[None, :])[0]

    @partial(jax.jit, static_argnums=(0, 2))
    def curvature(self, x: Array, kind: str = "hessian"):
        """Returns ||H|| at x for x in R^2."""
        if x.ndim == 1:
            x = x[None, :]
        return jax.vmap(lambda u: safe_norm(jax.hessian(self._poly_single)(u)))(x)

    @partial(jax.jit, static_argnums=(0,))
    def area_element(self, x: Array):
        """Returns sqrt(det(I + Jp^T Jp)) at x for x in R^2, i.e. (n, 2) -> (n,)."""
        if x.ndim == 1:
            x = x[None, :]
        return jax.vmap(partial(area_element_from_coeffs, self.poly_coeffs))(x)


class EmbeddedDisksOrConst(Constraint):
    def __init__(self,
            constraints: list[EmbeddedDiskConst]
        ):
        self.constraints = constraints
        print(f"EmbeddedDisksOrConst with {len(constraints)} disks.")

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        if x.ndim == 1:
            x = x[None, :]
        vals = jnp.stack([c.value(x) for c in self.constraints], axis=1)
        return jnp.min(vals, axis=1)


class EmbeddedDisks(ConstrainedDistribution):
    def __init__(self,
        num_disks: int = 5,
        dim: int = 10,
        radius: float = 0.3,
        center_bounds: tuple[float] = (-0.7, 0.7),
        seed: int = 0,
        polynom_degree: int|None = None,
        **kwargs
    ):
        self.dim = dim
        rng = random.PRNGKey(seed)
        keys = random.split(rng, num_disks)
        self.radius = radius
        if polynom_degree is not None:
            self.disk_constr = [
                EmbeddedDisksNonlinConst(key, radius, center_bounds, dim, polynom_degree)
                for key in keys
            ]
            self.bounds = (-(polynom_degree + 1), (polynom_degree + 1))
            assert_str = "To ensure polynomials are bounded, disks must be in [-1,1]^2."
            assert center_bounds[0] - radius >= -1, assert_str
            assert center_bounds[1] + radius <= 1, assert_str
        else:
            self.disk_constr = [
                EmbeddedDiskConst(key, radius, center_bounds, dim) for key in keys
            ]
            self.bounds = (center_bounds[0] - radius, center_bounds[1] + radius)
        self.polynom_degree = polynom_degree
        constr = EmbeddedDisksOrConst(self.disk_constr)
        base = Uniform(self.bounds, dim)
        super().__init__(base, ConstraintSet([constr, *get_bound_contraints(self.bounds, dim)]))
        # Probing the area element costs a few seconds at high degree and is only needed
        # for ground-truth sampling, so it is deferred until the first `sample` call.
        self._seed = seed
        self._area_probe = None

    def _probe(self) -> tuple[Array, Array]:
        if self._area_probe is None:
            self._area_probe = self._probe_area_element(random.PRNGKey(self._seed + 1))
        return self._area_probe

    @property
    def _disk_weights(self) -> Array:
        """True per-disk surface-area fractions (diagnostic; see `_probe_area_element`)."""
        return self._probe()[0]

    @property
    def _area_sup(self) -> Array:
        """Per-disk rejection envelope for the area element."""
        return self._probe()[1]

    def _chart_points(self, rng: PRNGKey, disk_idx: int, n_samples: int) -> Array:
        """Uniform points on disk `disk_idx` in chart coordinates, (n_samples, 2)."""
        rng_r, rng_theta = random.split(rng)
        r = jnp.sqrt(random.uniform(rng_r, (n_samples, 1))) * self.radius
        theta = random.uniform(rng_theta, (n_samples, 1), maxval=2 * jnp.pi)
        return self.disk_constr[disk_idx].center + jnp.hstack([
            r * jnp.cos(theta),
            r * jnp.sin(theta)
        ])

    def _probe_area_element(self, rng: PRNGKey) -> tuple[Array, Array]:
        """Per-disk surface area (as mixture weights) and rejection envelope.

        The disks carry equal chart area but, once lifted through their polynomials,
        very different surface areas, so ground-truth samples must not pick a disk
        uniformly. The weights returned here are the true area fractions and are used
        to verify the sampler -- note they are *not* the proposal weights, since
        rejection with a per-disk envelope reshapes the mixture as well (see
        `_rejection_sample_chart`).

        The envelope is the empirical sup times ENVELOPE_SAFETY; it must be an upper
        bound for the rejection sampler to be exact (see `validate_envelope`).
        """
        if self.polynom_degree is None:
            k = len(self.disk_constr)
            return jnp.full((k,), 1.0 / k), jnp.ones((k,))

        probes = [
            self.disk_constr[i].area_element(self._chart_points(key, i, ENVELOPE_PROBES))
            for i, key in enumerate(random.split(rng, len(self.disk_constr)))
        ]
        areas = jnp.array([p.mean() for p in probes])
        sup = jnp.array([p.max() for p in probes]) * ENVELOPE_SAFETY
        return areas / areas.sum(), sup

    def validate_envelope(self, rng: PRNGKey, n_samples: int = 1_000_000) -> Array:
        """Largest area_element / envelope ratio over fresh probes; must stay below 1."""
        if self.polynom_degree is None:
            return jnp.zeros((len(self.disk_constr),))
        return jnp.array([
            self.disk_constr[i].area_element(self._chart_points(key, i, n_samples)).max()
            / self._area_sup[i]
            for i, key in enumerate(random.split(rng, len(self.disk_constr)))
        ])

    def get_curv_info(self, rng, n_samples=2000):
        max_curv = []
        avg_curve = []
        for i in range(len(self.disk_constr)):
            rng_i, rng = random.split(rng)
            curvatures = self.disk_constr[i].curvature(self._chart_points(rng_i, i, n_samples))
            max_curv.append(jnp.max(curvatures))
            avg_curve.append(jnp.mean(curvatures))
        return jnp.array(max_curv), jnp.array(avg_curve)

    def get_area_info(self, rng, n_samples=2000):
        """Per-disk (max, mean) area element -- the density distortion the GT corrects for."""
        max_area = []
        min_area = []
        avg_area = []
        for i in range(len(self.disk_constr)):
            rng_i, rng = random.split(rng)
            areas = self.disk_constr[i].area_element(self._chart_points(rng_i, i, n_samples))
            max_area.append(jnp.max(areas))
            min_area.append(jnp.min(areas))
            avg_area.append(jnp.mean(areas))
        return jnp.array(min_area), jnp.array(max_area), jnp.array(avg_area)

    @property
    def sample_name(self) -> str:
        deg = "lin" if self.polynom_degree is None else f"deg{self.polynom_degree}"
        return f"embed_disk_{self.dim}d_{len(self.disk_constr)}K_{deg}"

    @property
    def init_scale(self) -> float:
        return (self.bounds[1] - self.bounds[0]) / 4

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False):
        """Ground-truth samples, uniform w.r.t. the surface measure on the manifold.

        For the linear disks the graph lift is an isometry, so uniform chart coordinates
        are already uniform on the manifold. For the polynomial disks the lift stretches
        area, so both the disk mixture weights and the within-disk density have to be
        corrected by the area element -- done here by rejection sampling.

        The returned points lie on the manifold up to floating point: |h| ~ 1e-12 under
        x64, but ~1e-2 at degree 6 in float32, which is above FEAS_THRESH. Enable x64
        when generating or evaluating these samples (see `__main__`).
        """
        if load:
            try:
                out = jnp.load(SAMPLE_PATH / f"{self.sample_name}_{n_samples}s.npy")
                return out
            except FileNotFoundError as e:
                print(e)
                exit(1)

        rotates = jnp.stack([d.rotate for d in self.disk_constr], axis=0)         # (K, dim, dim)
        centers = jnp.concatenate([d.center for d in self.disk_constr], axis=0)   # (K, 2)

        if self.polynom_degree is None:
            key_idx, key_chart = random.split(rng)
            disk_indices = random.randint(key_idx, (n_samples,), 0, len(self.disk_constr))
            chart = centers[disk_indices] + self._polar_offsets(key_chart, n_samples)
            samples = jnp.zeros((n_samples, self.dim)).at[:, :2].set(chart)
            return jnp.einsum('nd,nde->ne', samples,
                              jnp.swapaxes(rotates[disk_indices], -1, -2))

        chart, disk_indices = self._rejection_sample_chart(rng, n_samples, self._area_sup)
        poly_coeffs = jnp.stack([d.poly_coeffs for d in self.disk_constr], axis=0)
        poly = jax.vmap(poly_from_coeffs)(poly_coeffs[disk_indices], chart)  # (n, dim-2)
        samples = jnp.concatenate([chart, poly], axis=1)                    # (n, dim)
        return jnp.einsum('nd,nde->ne', samples, jnp.swapaxes(rotates[disk_indices], -1, -2))

    def _polar_offsets(self, rng: PRNGKey, n_samples: int) -> Array:
        """Uniform offsets within a disk of `self.radius`, (n_samples, 2)."""
        key_r, key_theta = random.split(rng)
        r = jnp.sqrt(random.uniform(key_r, (n_samples, 1))) * self.radius
        theta = random.uniform(key_theta, (n_samples, 1), maxval=2 * jnp.pi)
        return jnp.hstack([r * jnp.cos(theta), r * jnp.sin(theta)])

    @partial(jax.jit, static_argnums=(0, 2))
    def _rejection_sample_chart(
            self, rng: PRNGKey, n_samples: int, area_sup: Array
        ) -> tuple[Array, Array]:
        """Chart points with density proportional to the area element, plus their disk.

        A disk is proposed with probability proportional to its envelope `_area_sup[i]`
        and a uniform chart point on it is accepted with probability
        `area_element(u) / _area_sup[i]`, so the accepted density over (disk, u) is
        proportional to `area_element(u)` with the same constant for every disk. That
        makes the lift uniform w.r.t. the surface measure on the union of the disks and
        fixes the mixture weights as a side effect -- proposing disks by area instead
        would leave a residual `E_i[area] / sup_i` factor in the mixture.

        Follows the fixed-buffer while_loop pattern of `GrowingSineDistribution.sample`.
        """
        centers = jnp.concatenate([d.center for d in self.disk_constr], axis=0)   # (K, 2)
        poly_coeffs = jnp.stack([d.poly_coeffs for d in self.disk_constr], axis=0)
        n_disks = len(self.disk_constr)
        bsz = int(min(max(2048, n_samples), 16384))
        proposal_weights = area_sup / area_sup.sum()

        def cond(state):
            _, _, _, curr = state
            return curr < n_samples

        def body(state):
            key, chart, disks, curr = state
            key, k_idx, k_chart, k_accept = random.split(key, 4)

            idx = random.choice(k_idx, n_disks, (bsz,), p=proposal_weights)
            proposals = centers[idx] + self._polar_offsets(k_chart, bsz)
            areas = jax.vmap(area_element_from_coeffs)(poly_coeffs[idx], proposals)
            ok = random.uniform(k_accept, (bsz,)) * area_sup[idx] < areas

            cs = jnp.cumsum(ok)
            sample_idx = curr + cs - 1
            sample_idx = jnp.where(ok & (sample_idx < n_samples), sample_idx, n_samples)
            chart = chart.at[sample_idx].set(proposals, mode="drop")
            disks = disks.at[sample_idx].set(idx, mode="drop")
            curr = jnp.minimum(n_samples, curr + cs[-1])

            return key, chart, disks, curr

        init = (rng, jnp.zeros((n_samples, 2)),
                jnp.zeros((n_samples,), dtype=jnp.result_type(int)), 0)
        _, chart, disks, _ = jax.lax.while_loop(cond, body, init)
        return chart, disks

    def save_samples(self, samples: Array) -> None:
        path = SAMPLE_PATH / f"{self.sample_name}_{samples.shape[0]}s.npy"
        jnp.save(path, jnp.asarray(samples))


    def plot2d(
        self,
        samples: Array = None,
        ax: Axes = None,
        markersize: float = 10.0,
        alpha: float = 0.6,
        plt_show: bool = True,
        title: str = None,
        big_symbols = True,
        **kwargs
    ) -> Axes:
        import matplotlib.pyplot as plt
        if self.polynom_degree is not None:
            Warning("Plotting EmbeddedDisks with polynomials is not yet implemented, plot is nonsense.")
            return ax

        symbols_big = ["o", "v", "*", "X", "d", "p"]
        symbolds_thin = [".", "1", "+", "x", "|", "_"]
        symbols = symbols_big if big_symbols else symbolds_thin
        assert len(self.disk_constr) <= len(symbols), "Not enough symbols for the number of disks."

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if samples.ndim == 1:
            samples = samples[None, :]
        constr_vals = jnp.stack([c.value(samples) for c in self.disk_constr], axis=1)
        disk_idx = jnp.argmin(constr_vals, axis=1)
        for i in range(len(self.disk_constr)):
            disk_samples = samples[(disk_idx == i).squeeze()]
            disk_samples = disk_samples @ self.disk_constr[i].rotate
            dist_to_plane = safe_norm(disk_samples[:, 2:], axis=1)
            ax.scatter(
                disk_samples[:, 0],
                disk_samples[:, 1],
                s=markersize,
                marker=symbols[i % len(symbols)],
                alpha=alpha * jnp.exp(-dist_to_plane).squeeze(),
                label=f'Disk {i}',
                cmap='cividis',
                c=-dist_to_plane,
                vmin=-1,
                vmax=0
            )
        ax.set_aspect('equal')
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        if title is not None:
            ax.set_title(title)
        if plt_show:
            plt.show()
        return ax


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    n_samples = 2_000
    rng = jax.random.PRNGKey(0)

    for dim, degree in [(10, None), (16, None), *[(16, d) for d in range(1, 7)]]:
        for num_disks in [10,15,20]:
            rng, rng_s, rng_v = jax.random.split(rng, 3)
            dist = EmbeddedDisks(dim=dim, polynom_degree=degree, num_disks=num_disks)
            samples = dist.sample(rng_s, n_samples=n_samples, load=False)
            dist.save_samples(samples)

            ratio = dist.validate_envelope(rng_v, n_samples=200_000)
            assert jnp.all(ratio < 1.0), f"envelope exceeded for {dist.sample_name}: {ratio}"
            print(f"{dist.sample_name}: weights={jnp.round(dist._disk_weights, 3)} "
                f"max_envelope_ratio={ratio.max():.3f}")
