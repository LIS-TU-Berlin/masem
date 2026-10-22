from functools import partial

import jax
import jax.numpy as jnp
from chex import Array

from masem.distributions.constraint import Constraint


class SphereCollisionConstraint(Constraint):
    def __init__(self, center: Array, radius: float, ndof: int):
        self.center = center
        self.radius = radius
        self.ndof = ndof

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]
        bsz = x.shape[0]

        x = x.reshape(bsz, -1, self.ndof)
        dx = x - self.center[None, None, :]
        g = self.radius**2 - jnp.einsum("...d,...d->...", dx, dx)
        return g if batched else g[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]
        bsz = x.shape[0]

        x = x.reshape(bsz, -1, self.ndof)
        jac = -2.0 * (x - self.center[None, None, :])
        nt = x.shape[1]
        full_jac = jnp.einsum("btd,tT->btTd", jac, jnp.eye(nt))
        full_jac = full_jac.reshape(bsz, nt, nt * self.ndof)

        return full_jac if batched else full_jac[0]


class WaypointToPathConstraint(Constraint):
    def __init__(
        self,
        inner: Constraint,
        q0: Array,
        invPhi: Array,
        Phi_path: Array,
        ndof: int,
    ):
        self.inner = inner
        self.q0 = jnp.asarray(q0)
        self.invPhi = jnp.asarray(invPhi)
        self.Phi_path = jnp.asarray(Phi_path)
        self.ndof = ndof

        # Split into constant offset from q0 and linear part from waypoints
        self.Bmat = self.Phi_path @ self.invPhi
        self.b0 = self.Bmat[:, 0]
        self.A = self.Bmat[:, 1:]
        self.T, self.N = self.A.shape

        # Precompute d(path_flat)/d(waypts_flat) = kron(A, I_D)
        self.J_path = jnp.kron(self.A, jnp.eye(self.ndof))

    @property
    def is_equality(self) -> bool:
        return self.inner.is_equality

    def _path_flat(self, x: Array) -> Array:
        batched = (x.ndim == 2)
        if not batched:
            x = x[None, :]

        B = x.shape[0]
        waypts = x.reshape(B, self.N, self.ndof)
        offset = (self.b0[:, None] * self.q0[None, :])[None, :, :]
        path = offset + jnp.einsum("tn,bnd->btd", self.A, waypts)

        return path.reshape(B, self.T * self.ndof)

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = (x.ndim == 2)
        if not batched:
            x = x[None, :]

        path_flat = self._path_flat(x)
        g = self.inner.value(path_flat)
        return g if batched else g[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        batched = (x.ndim == 2)
        if not batched:
            x = x[None, :]

        path_flat = self._path_flat(x)
        inner_jac = self.inner.grad(path_flat)
        # Chain rule: (B,T,T*D) @ (T*D, N*D) -> (B,T,N*D)
        jac = jnp.einsum("bti,ij->btj", inner_jac, self.J_path)

        return jac if batched else jac[0]
