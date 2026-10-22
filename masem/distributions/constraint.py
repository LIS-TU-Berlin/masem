from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from chex import Array


class Constraint(ABC):
    @property
    @abstractmethod
    def is_equality(self) -> bool:
        pass

    @abstractmethod
    def value(self, x: Array) -> Array:
        """g(x) or h(x)"""
        pass

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        return jax.vmap(jax.grad(lambda x: self.value(x).reshape(())))(x)


class InvertedConstraint(Constraint):
    def __init__(self, constraint: Constraint):
        self.constraint = constraint
        if constraint.is_equality:
            raise ValueError("Cannot invert an equality constraint.")
            # For equalities we could do (h < 0 or h > 0) with OrConstraint

    @property
    def is_equality(self) -> bool:
        return False

    def value(self, x: Array) -> Array:
        return -self.constraint.value(x)

    def grad(self, x: Array) -> Array:
        return -self.constraint.grad(x)


class ConstraintSet:
    def __init__(self, constraints: list[Constraint]):
        self.eq = [c for c in constraints if c.is_equality]
        self.ineq = [c for c in constraints if not c.is_equality]

    @staticmethod
    def _val_to_vec(v: Array) -> Array:
        if v.ndim == 0:
            return v[None]
        if v.ndim == 1:
            return v[..., None]
        return v

    @staticmethod
    def _jac_to_mat(J: Array, dim: int) -> Array:
        if J.ndim == 1 and J.shape[0] == dim:
            return J[None, :]
        if J.ndim == 2 and J.shape[-1] == dim:
            return J[:, None, :]
        return J

    def _cat_vals(self, vals: list[Array], x: Array) -> Array:
        if not vals:
            return jnp.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
        vals = [self._val_to_vec(v) for v in vals]
        return jnp.concatenate(vals, axis=-1)

    def _cat_jacs(self, grads: list[Array], x: Array) -> Array:
        if not grads:
            return jnp.zeros(x.shape[:-1] + (0, x.shape[-1]), dtype=x.dtype)
        dim = x.shape[-1]
        grads = [self._jac_to_mat(J, dim) for J in grads]
        return jnp.concatenate(grads, axis=-2)

    def g(self, x: Array) -> Array:
        """Ineq g(x) <= 0."""
        return self._cat_vals([c.value(x) for c in self.ineq], x)

    def Jg(self, x: Array) -> Array:
        """Jac of g, shape is (batch, n, dim)"""
        return self._cat_jacs([c.grad(x) for c in self.ineq], x)

    def h(self, x: Array) -> Array:
        """equality: h(x) = 0."""
        return self._cat_vals([c.value(x) for c in self.eq], x)

    def Jh(self, x: Array) -> Array:
        return self._cat_jacs([c.grad(x) for c in self.eq], x)

    def h_and_Jh(self, x: Array) -> tuple[Array, Array]:
        """Returns (h, Jh) sharing fwd pass via value_and_grad."""
        if not self.eq:
            return (
                jnp.zeros(x.shape[:-1] + (0,), dtype=x.dtype),
                jnp.zeros(x.shape[:-1] + (0, x.shape[-1]), dtype=x.dtype),
            )
        return self.h(x), self.Jh(x)

    def feasible(self, x: Array, tol: float = 1e-6) -> Array:
        x = jnp.atleast_2d(x)
        bsz = x.shape[0]
        cond = self.g(x) <= tol
        ineq_ok = (
            jnp.all(cond.reshape(bsz, -1), axis=-1)
            if self.ineq
            else jnp.ones(x.shape[:-1], dtype=bool)
        )
        cond = jnp.abs(self.h(x)) <= tol
        eq_ok = (
            jnp.all(cond.reshape(bsz, -1), axis=-1)
            if self.eq
            else jnp.ones(x.shape[:-1], dtype=bool)
        )
        return jnp.logical_and(ineq_ok, eq_ok).squeeze()

    def analytical_intersections(self, x: Array, dir: Array) -> Array:
        parts = [c.analytical_intersections(x, dir) for c in self.ineq] + [
            c.analytical_intersections(x, dir) for c in self.eq
        ]

        if not parts:
            return jnp.zeros((0,), dtype=x.dtype)

        all_t = jnp.concatenate(parts, axis=0)
        return all_t
