import jax.numpy as jnp
from chex import Array

from masem.distributions.constraint import Constraint


def get_bound_contraints(bounds: tuple[float, float], dim: int) -> list[Constraint]:
    constraints = []
    one_hot = jnp.eye(dim)
    for d in range(dim):
        lo_bound = HalfspaceIneq(a=-one_hot[d], b=-bounds[0])
        hi_bound = HalfspaceIneq(a=one_hot[d], b=bounds[1])
        constraints.extend([lo_bound, hi_bound])
    return constraints


class HalfspaceIneq(Constraint):
    def __init__(self, a: Array, b: float | Array):
        """Affine half space constraint a^Tx <= b"""
        self.a = jnp.asarray(a)
        self.b = jnp.asarray(b)

    @property
    def is_equality(self) -> bool:
        return False

    def value(self, x: Array) -> Array:
        return jnp.einsum("...d,d->...", x, self.a) - self.b

    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        out = jnp.ones(x.shape) * self.a
        return out if batched else out[0]

    def analytical_intersections(
        self, x: Array, dir: Array, tol: float = 1e-12
    ) -> Array:
        """T = (b - aTx) / aTd"""
        batx = self.b - jnp.einsum("d,d->", x, self.a)
        atd = jnp.einsum("d,d->", dir, self.a)
        parallel = jnp.abs(atd) < tol
        t = batx / (atd + tol)
        return jnp.where(parallel, jnp.zeros((0,)), jnp.array([t]))
