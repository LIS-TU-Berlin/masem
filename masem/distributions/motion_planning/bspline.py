import jax
import chex
import jax.numpy as jnp
from functools import partial


def _safe_div(numer: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.cond(
        denom == 0.0,
        lambda _: jnp.array(0.0, dtype=numer.dtype),
        lambda _: numer / denom,
        operand=None,
    )


def compute_knots(n: int, degree: int = 3) -> chex.Array:
    """Get the knot vector for a given number of points.

    Parameters:
        n (int): Number of evaluation points.
        degree (int): Degree of the B-spline.

    Returns:
        chex.Array: Knot vector.
    """
    u = jnp.linspace(0., 1., n, True)
    if degree % 2 == 0:
        middle_knots = (u[:-1] + u[1:]) / 2.0
        middle_knots = jnp.append(middle_knots, u[-1])
    else:
        middle_knots = u
    kv = jnp.r_[(u[0],) * degree, middle_knots, (u[-1],) * degree]
    return kv


def clamp_ctrls(x: chex.Array, degree: int = 2) -> chex.Array:
    """Clamp a vector of controls for inference.

    Parameters:
        x (chex.Array): Control vector.
        degree (int): Degree of the B-spline.

    Returns:
        chex.Array: Clamped control vector with multiplicities at start and end.
    """
    if x.ndim == 2:
        x = x[jnp.newaxis, :, :]

    nrep = degree // 2
    first_points = jnp.tile(x[:, 0, :][:, jnp.newaxis, :], (1, nrep, 1))
    last_points = jnp.tile(x[:, -1, :][:, jnp.newaxis, :], (1, nrep, 1))
    ctrl_points = jnp.concatenate([first_points, x, last_points], axis=1)

    return jnp.hstack(ctrl_points)


class BSpline:
    def __init__(self, n_in: int, n_out: int, degree: int = 3, order: int = 2):
        """Initialize the B-spline.

        Parameters:
            n_in (int): Number of input points = number of knots.
            n_out (int): Number of output points = number of points at which we eval the spline.
            degree (int): Degree of the B-spline.
            order (int): Order of derivatives to compute.
        """
        self.knots = compute_knots(n_in, degree)
        self.degree = degree
        self.t_values = jnp.linspace(self.knots[degree], self.knots[-degree - 1], n_out, True)
        self.n = len(self.knots) - degree - 1
        self.order = order
        self.Phi = self.compute_basis_matrix()
        self.dPhi = self.compute_derivative_matrix(1) if order > 0 else None
        self.ddPhi = self.compute_derivative_matrix(2) if order > 1 else None

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Evaluate the B-spline and its derivatives at the given control points.

        Parameters:
            x (chex.Array): Control points for the B-spline.

        Returns:
            tuple[chex.Array, chex.Array, chex.Array]: Evaluated spline, first derivative, and second derivative.
        """
        xc = clamp_ctrls(x, self.degree)

        y = self.Phi @ xc
        yDot = self.dPhi @ xc if self.order > 0 else None
        ydDot = self.ddPhi @ xc if self.order > 1 else None

        if x.ndim > 2:
            y = y.reshape(-1, x.shape[0], x.shape[-1]).transpose(1, 0, 2)
            if self.order > 0:
                yDot = yDot.reshape(-1, x.shape[0], x.shape[-1]).transpose(1, 0, 2)
            if self.order > 1:
                ydDot = ydDot.reshape(-1, x.shape[0], x.shape[-1]).transpose(1, 0, 2)

        return y, yDot, ydDot

    def compute_basis_matrix(self) -> chex.Array:
        """Compute the basis matrix for the B-spline.

        Returns:
            chex.Array: Basis matrix.
        """
        basis_matrix = jax.vmap(
            lambda i: jax.vmap(lambda t: self.basis_function(t, self.degree, i))(self.t_values)
        )(jnp.arange(self.n))

        return basis_matrix.T

    def compute_derivative_matrix(self, order: int = 1) -> chex.Array:
        """Compute the derivative matrix for the B-spline.

        Parameters:
            order (int): Order of the derivative.

        Returns:
            chex.Array: Derivative matrix.
        """
        derivative_matrix = jax.vmap(
            lambda i: jax.vmap(lambda t: self.derivative_basis_function(t, self.degree, i, order))(
                self.t_values)
        )(jnp.arange(self.n))

        return derivative_matrix.T

    def basis_function(self, t: float, k: int, i: int) -> float:
        """Cox-de Boor recursion formula for B-spline basis function.

        Parameters:
            t (float): Parameter value.
            k (int): Degree of the basis function.
            i (int): Knot span index.

        Returns:
            float: Basis function value at t.
        """
        knots = self.knots

        if k == 0:
            in_span = (knots[i] <= t) & (t < knots[i + 1])
            at_right_endpoint = (t == knots[-1]) & (i == self.n - 1)
            return jnp.where(in_span | at_right_endpoint, 1.0, 0.0)

        # Recursive case
        denom_left = knots[i + k] - knots[i]
        denom_right = knots[i + k + 1] - knots[i + 1]

        # left term: ((t - t_i) / (t_{i+k} - t_i)) * N_{i,k-1}(t)
        left_coeff = _safe_div(t - knots[i], denom_left)
        left = left_coeff * self.basis_function(t, k - 1, i)

        # right term: ((t_{i+k+1} - t) / (t_{i+k+1} - t_{i+1})) * N_{i+1,k-1}(t)
        right_coeff = _safe_div(knots[i + k + 1] - t, denom_right)
        right = right_coeff * self.basis_function(t, k - 1, i + 1)

        return left + right

    def derivative_basis_function(self, t: float, k: int, i: int, order: int = 1) -> float:
        """Compute the derivative of the B-spline basis function.

        Parameters:
            t (float): Parameter value.
            k (int): Degree of the basis function.
            i (int): Knot span index.
            order (int): Order of the derivative.

        Returns:
            float: Derivative of the basis function at t.
        """
        if order == 0:
            return self.basis_function(t, k, i)
        if k == 0:
            return 0.0

        knots = self.knots
        denom_left = knots[i + k] - knots[i]
        denom_right = knots[i + k + 1] - knots[i + 1]

        # d/dt N_{i,k}(t) = k/(t_{i+k}-t_i) * N_{i,k-1}(t) - k/(t_{i+k+1}-t_{i+1}) * N_{i+1,k-1}(t)
        # For higher derivatives, apply derivative recurrence to the (k-1) basis derivatives.
        left_scale = _safe_div(jnp.array(k, dtype=knots.dtype), denom_left)
        right_scale = _safe_div(jnp.array(k, dtype=knots.dtype), denom_right)

        left = left_scale * self.derivative_basis_function(t, k - 1, i, order - 1)
        right = right_scale * self.derivative_basis_function(t, k - 1, i + 1, order - 1)

        return left - right
