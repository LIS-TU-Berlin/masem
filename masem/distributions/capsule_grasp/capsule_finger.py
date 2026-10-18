"""Dynamically generate constraints for each finger.

We create the finger constraints dynamically by adding different value functions
to the base class, which registers the values and provides helpers. Each value
function is wrapped, where the wrapper takes care of the contact mode and selects
the correct finger vector. The constraints are defined in get_constraints_data().
"""

from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
from chex import Array

from masem.distributions.constraint import Constraint
from masem.utils import safe_norm


class ContactMode(Enum):
    NONE = 0
    BASE = 1
    SIDE = 2


def create_capsule_finger_constraints(
    index: int,
    length: float,
    radius: float,
    orientation: Array,
    max_force: float,
    min_force: float,
    cone_coeff: float
):
    # f, is_equal
    c_spec = (
        ("poa_pos", True),
        ("force_cone_constraint", False),
        ("force_norm", False),
        ("force_norm_min", False),   # removes local opt. of satisfying force constraints w/o force
        ("infeasible_region", False)
    )

    constraints = []
    for method_name, is_equal in c_spec:
        constr = FingerConstraintBC(
            index=index,
            is_equality=is_equal,
            length=length,
            radius=radius,
            orientation=orientation,
            max_force=max_force,
            min_force=min_force,
            cone_coeff=cone_coeff,
            method=method_name
        )
        constraints.append(constr)

    return constraints


class FingerConstraintBC(Constraint):
    def __init__(
        self,
        index: int,
        is_equality: bool,
        length: float,
        radius: float,
        orientation: Array,
        max_force: float,
        min_force: float,
        cone_coeff: float,
        method: str
    ):
        r"""Each finger is 6d and encodes x = (POA loc, POA force) \in R^6.

        The base or side contact is constraint to be XOR.
        """
        self.index = index
        self._is_equality = is_equality
        self.length = length
        self.radius = radius
        self.orientation = orientation
        self.max_force = max_force
        self.min_force = min_force
        self.cone_coeff = cone_coeff
        self._value = getattr(self, method)

    @property
    def is_equality(self) -> bool:
        return self._is_equality

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array):
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        x = x.reshape(x.shape[0], -1, 6)[:, self.index, :]
        val = self._value(x)

        return val if batched else val[0]

    def _get_axis_pos(self, x: Array):
        poa = x[:, :3]
        axis_pos = jax.lax.clamp(-self.length/2, jnp.inner(poa, self.orientation), self.length/2)
        return axis_pos

    def _get_center_axis_vector(self, x: Array):
        poa = x[:, :3]
        axis_pos = self._get_axis_pos(x)
        center_axis_vec = poa - axis_pos * self.orientation
        return center_axis_vec

    def _calculate_normal(self, x: Array):
        center_axis_vec = self._get_center_axis_vector(x)
        return center_axis_vec / self.radius

    def force_cone_constraint(self, x: Array):
        normal = self._calculate_normal(x)
        force = x[:, 3:6]
        force_normal = jnp.einsum('bi,bi->b', force, normal)
        force_tangential = safe_norm(force - force_normal[:,None] * normal, axis=-1)
        return force_tangential + self.cone_coeff * force_normal

    def poa_pos(self, x: Array):
        center_axis_vec = self._get_center_axis_vector(x)
        return safe_norm(center_axis_vec, axis=-1) - self.radius

    def force_norm(self, x: Array):
        return safe_norm(x[:, 3:6], axis=-1) - self.max_force

    def force_norm_min(self, x: Array):
        return self.min_force - safe_norm(x[:, 3:6], axis=-1)

    def infeasible_region(self, x: Array, offset: float = 0., width: float = 0.25):
        axis_pos = self._get_axis_pos(x)
        return -jnp.abs(axis_pos - offset) + width/2
