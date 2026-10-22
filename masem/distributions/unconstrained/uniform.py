from functools import partial

import jax.numpy as jnp
import jax.random
from chex import Array, PRNGKey

from masem.constants import SAMPLE_PATH
from masem.distributions.distribution import Distribution


class Uniform(Distribution):
    """Uniform distribution over a box"""

    def __init__(self, bounds: tuple[float, float] = (-10.0, 10.0), dim: int = 2):
        self.dim = dim
        self.bounds = bounds
        self.log_prob_min = -40.

    @partial(jax.jit, static_argnums=(0,))
    def unnorm_log_prob(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        logp = jnp.ones((x.shape[0],))
        if not batched:
            logp = logp.squeeze(0)
        return logp

    @partial(jax.jit, static_argnums=(0,))
    def score(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        score_ = jnp.zeros((x.shape[0], x.shape[-1]))
        if not batched:
            score_ = jnp.squeeze(score_, axis=0)
        return score_

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        if load:
            try:
                out = jnp.load(SAMPLE_PATH / f"uniform_{n_samples}s_{self.dim}d.npy")
                return out
            except FileNotFoundError as e:
                print(e)
                exit(1)
        x = jax.random.uniform(rng, (n_samples, self.dim))
        return self.bounds[0] + x * (self.bounds[1] - self.bounds[0])
