from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.sparse.linalg import cg
from chex import Array, PRNGKey

from masem.distributions.constrained_dist import ConstrainedDistribution


class GaussNewtonInitializer:
    def __init__(
        self,
        dist: ConstrainedDistribution,
        initial_scale: float,
        n_chains: int,
        alpha: float = 0.5,
        sigma: float = 1e-2,
        step_max: float = 1.0,
        tol: float = 1e-6,
        cg_maxiter: int = 50,
    ):
        self.dist = dist
        self.initial_scale = initial_scale
        self.n_chains = n_chains
        self.alpha = alpha
        self.sigma = sigma
        self.step_max = step_max
        self.tol = tol
        self.cg_maxiter = cg_maxiter

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng: PRNGKey) -> Array:
        return jr.normal(rng, (self.n_chains, self.dist.dim)) * self.initial_scale
    
    @partial(jax.jit, static_argnums=(0,))
    def _gauss_newton_step(self, rng: PRNGKey, x: Array) -> Array:
        x += jr.normal(rng, x.shape) * self.sigma

        _, vjp_fun = jax.vjp(self.dist.slack_vec, x)
        JT = lambda w: vjp_fun(w)[0]
        J = lambda v: jax.jvp(self.dist.slack_vec, (x,), (v,))[1]

        A = lambda v: JT(J(v)) + self.tol * v
        b = -JT(self.dist.slack_vec(x))

        delta, _ = cg(A, b, maxiter=self.cg_maxiter)
        step = self.alpha * delta
        step *= jnp.minimum(
            1.0, self.step_max / (jnp.linalg.norm(step, keepdims=True) + self.tol)
        )
        return x + step
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, rng: PRNGKey, x: Array) -> Array:
        rng = jr.split(rng, self.n_chains)
        return jax.vmap(self._gauss_newton_step)(rng, x)