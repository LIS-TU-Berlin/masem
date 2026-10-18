"""MIT License

Copyright (c) 2025 Kijung Jeon
2026 Cornelius Braun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array, PRNGKey, dataclass
from jax.scipy.sparse.linalg import cg

from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.samplers.sampler import Sampler
from masem.types import Eval


@dataclass
class OLLAState:
    x: Array
    ev: Eval
    step_count: int = 0


class OLLA(Sampler):
    """OLLA (Zheng et al , 2025).

    Implements the OLLA-H version with Hutchinsons trace estimator for curvature correction

    Based on: https://github.com/KraitGit/OLLA/blob/main/src/samplers/olla_h.py
    """

    def __init__(
        self,
        target_dist: ConstrainedDistribution,
        n_chains: int,
        step_size: float = 5e-4,
        alpha: float = 2e3,
        max_norm: float = 100.,
        proj_damp: float = 1e-6,
        n_hutch_samples: int = 5,
        cg_maxiter: int = 20,
    ):
        super().__init__()
        self.target_dist = target_dist
        self.dim = target_dist.dim
        self.n_chains = int(n_chains)
        self.step_size = float(step_size)
        self.alpha = float(alpha)
        self.noise_scale = jnp.sqrt(2.0 * self.step_size)
        self.proj_damp = float(proj_damp)
        self.n_hutch_samples = int(n_hutch_samples)
        self.max_norm = float(max_norm)
        self.cg_maxiter = int(cg_maxiter)
        self.bounds = target_dist.bounds

        def c_full(x: Array) -> Array:
            ev: Eval = self.target_dist.eval(x)
            g = ev.g if ev.g is not None else jnp.zeros((0,))
            h = ev.h if ev.h is not None else jnp.zeros((0,))
            return jnp.concatenate([g, h], axis=-1)

        self._c_full = c_full

    def initialize(self, rng: PRNGKey, x0: Array) -> OLLAState:
        ev0 = jax.vmap(self.get_eval)(x0)
        return OLLAState(x=x0, step_count=0, ev=ev0)

    def _hutchinson_trace_vec(self, rng: PRNGKey, x: Array, Js: Array) -> Array:
        """Estimates Trace(P @ Hess c_i) for all constraints i via nested JVP."""
        d = Js.shape[1]
        proj_damp = self.proj_damp
        matvec = lambda v: Js @ (Js.T @ v) + proj_damp * v

        def single_sample_trace(key):
            u = jr.bernoulli(key, shape=(d,)).astype(x.dtype) * 2 - 1
            Ju, _ = cg(matvec, Js @ u, maxiter=self.cg_maxiter)
            Pu = u - Js.T @ Ju
            _, hvp_all = jax.jvp(
                lambda _x: jax.jvp(self._c_full, (_x,), (u,))[1],
                (x,), (Pu,)
            )
            return hvp_all

        keys = jr.split(rng, self.n_hutch_samples)
        trace_samples = jax.vmap(single_sample_trace)(keys)
        return jnp.mean(trace_samples, axis=0)

    def _step_one(self, rng: PRNGKey, x: Array, ev: Eval) -> Array:
        rng_hutch, rng_noise = jr.split(rng)
        score_f = ev.base_score

        Js = ev.jac_slack
        n_active = Js.shape[0]
        if n_active == 0:
            xi = jr.normal(rng_noise, x.shape)
            return x + self.step_size * score_f + self.noise_scale * xi

        matvec = lambda v: Js @ (Js.T @ v) + self.proj_damp * v

        if self.n_hutch_samples > 0:
            mask_g = (ev.g <= self.target_dist.eps).astype(x.dtype)
            selection = jnp.concatenate([mask_g, jnp.ones((ev.h.shape[0],))], axis=0)
            trace_vec = self._hutchinson_trace_vec(rng_hutch, x, Js) * selection
        else:
            trace_vec = jnp.zeros(n_active)

        rhs = Js @ score_f + self.alpha * ev.slack + trace_vec
        lam, _ = cg(matvec, rhs, maxiter=self.cg_maxiter)
        drift = score_f - Js.T @ lam
        drift *= jnp.minimum(1.0, self.max_norm / (jnp.linalg.norm(drift) + 1e-8))

        rnd_dir = jr.normal(rng_noise, x.shape)
        lam_noise, _ = cg(matvec, Js @ rnd_dir, maxiter=self.cg_maxiter)
        xi_projected = rnd_dir - Js.T @ lam_noise

        return x + self.step_size * drift + self.noise_scale * xi_projected

    @partial(jax.jit, static_argnames=("self",))
    def step(self, rng: PRNGKey, state: OLLAState) -> OLLAState:
        x = state.x
        rngs = jr.split(rng, x.shape[0])
        x_next = jax.vmap(self._step_one, in_axes=(0, 0, 0), out_axes=0)(rngs, x, state.ev)
        x_next = jnp.clip(x_next, *self.bounds)
        ev = jax.vmap(self.target_dist.eval)(x_next)

        return OLLAState(
            x=x_next, step_count=state.step_count + self.n_hutch_samples + 1, ev=ev
        )
