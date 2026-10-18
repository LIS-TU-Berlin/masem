from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from chex import Array, PRNGKey, dataclass
from jax import random
from jaxopt import Bisection

from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.samplers.nhr import NHR, NhrState
from masem.samplers.sampler import ResamplingSampler
from masem.samplers.utils import knn_log_radius_topk
from masem.types import Eval


@dataclass
class KnnNhrSmcState:
    x: Array
    last_eval: Eval
    log_weights: Array
    temperature: Array
    step_count: int = 0


class KnnNhr(NHR, ResamplingSampler):

    def __init__(
        self,
        target_dist: ConstrainedDistribution,
        n_chains: int,
        n_down: int,
        max_step: float = 1.0,
        mu_slack: float = 1e3,
        max_ls_iters: int = 1,
        slack_margin: float = 1e-1,
        knn_tau: float = 0.5,
        knn_k: int = 10,
        knn_eps: float = 1e-6,
        n_inner: int = 50,
        p_tau = 0.0,
        use_kmean: bool = True
    ):
        super().__init__(
            target_dist,
            n_chains,
            n_down,
            max_step,
            mu_slack,
            max_ls_iters,
            slack_margin
        )
        self.knn_tau = knn_tau
        self.knn_k = int(knn_k)
        self.knn_eps = knn_eps
        self.p_tau = p_tau
        self.use_kmean = use_kmean

        self.n_down = n_down
        self.n_inner = n_inner
        #self.mu_slack2 = mu_slack

    @partial(jax.jit, static_argnames=("self",))
    def resample(self, rng: PRNGKey, state: NhrState) -> NhrState:
        x, ev = state.x, state.last_eval
        log_r = knn_log_radius_topk(x, k=self.knn_k, eps=self.knn_eps, use_kmean=self.use_kmean)
        logw = self.knn_tau * log_r
        logw = logw + ev.base_logp * self.p_tau
        slack_penalty = jnp.sum(ev.slack ** 2, axis=-1)
        logw -= self.mu_slack * slack_penalty
        w = jax.nn.softmax(logw)
        idx = resampling.stratified(rng, w, x.shape[0])
        x_rs = x[idx]
        ev_rs = jax.tree.map(lambda a: a[idx] if isinstance(a, jax.Array) else a, ev)
        return state.replace(x=x_rs, last_eval=ev_rs)


class KnnNhrSmc(KnnNhr):
    """Three-phase sampler: downhill → uniform coverage (NHR) → SMC annealing (NHR).

    Phase 1 (step_count < n_down): downhill Gauss-Newton steps to reach feasibility.
    Phase 2 (n_down <= step_count < n_uniform): NHR steps (temperature=0) with kNN
        resampling every n_rejuv steps to drive particles toward uniform on the feasible set.
    Phase 3 (step_count >= n_uniform): SMC — every n_rejuv steps: geometric weight
        update (bisection or fixed delta_t) + stratified resample; otherwise NHR
        jitter at the current temperature.
    """

    def __init__(
        self,
        target_dist: ConstrainedDistribution,
        n_chains: int,
        n_total: int,
        n_down: int,
        max_step: float = 1.0,
        mu_slack: float = 1e3,
        max_ls_iters: int = 1,
        slack_margin: float = 1e-1,
        knn_tau: float = 1.0,
        knn_k: int = 10,
        knn_eps: float = 1e-6,
        n_uniform: int = 500,
        ess_threshold: float = 0.8,
        use_bisection: bool = False,
        bisection_steps: int = 30,
        slack_alpha: float = 2e3,
    ):
        super().__init__(
            target_dist, n_chains, n_down, max_step, mu_slack,
            max_ls_iters, slack_margin, knn_tau, knn_k, knn_eps,
        )
        self.n_uniform = n_uniform
        n_smc_steps = n_total - n_uniform
        self.delta_t = 1.0 / n_smc_steps
        self.ess_threshold = ess_threshold
        self.use_bisection = use_bisection
        self.bisection_steps = bisection_steps
        self.slack_alpha = slack_alpha

    def initialize(self, rng: PRNGKey, x0: Array) -> KnnNhrSmcState:
        ev0 = jax.vmap(self.get_eval)(x0)
        return KnnNhrSmcState(
            x=x0,
            last_eval=ev0,
            log_weights=jnp.zeros(self.n_chains),
            temperature=0.,
            step_count=0,
        )

    def _ess_target_from_dt(self, log_weights: Array, base_logp: Array, dt: Array) -> Array:
        log_w = log_weights + dt * base_logp
        log_w_norm = log_w - jax.nn.logsumexp(log_w)
        ess = 1.0 / jnp.sum(jnp.square(jnp.exp(log_w_norm)))
        return ess - self.ess_threshold * self.n_chains

    def _nhr_step_one(self, rng: PRNGKey, x: Array, ev: Eval, temperature: Array) -> Tuple[Array, Eval]:
        """Single NHR step at the given annealing temperature."""
        return self._nhr_step(
            rng, x, ev,
            self.get_eval,
            self.bounds[0],
            self.bounds[1],
            max_step=self.max_step,
            max_ls_iters=self.max_ls_iters,
            mu_slack=self.mu_slack,
            temperature=temperature,
        )

    def _smc_step(
        self,
        rng: PRNGKey,
        x: Array,
        ev: Eval,
        log_weights: Array,
        temperature: Array,
    ) -> Tuple[Array, Eval, Array, Array]:
        """One SMC step: advance temperature, accumulate IS weight, NHR rejuvenation."""
        delta_t_max = jnp.minimum(temperature + self.delta_t, 1.0) - temperature
        base_logp = ev.base_logp.flatten()
        if self.use_bisection:
            ess_fn = lambda dt: self._ess_target_from_dt(log_weights, base_logp, dt)
            ess_at_max = ess_fn(delta_t_max)
            delta_t = jax.lax.cond(
                ess_at_max >= 0.0,
                lambda: delta_t_max,
                lambda: Bisection(
                    ess_fn, 0., delta_t_max,
                    maxiter=self.bisection_steps, check_bracket=False,
                ).run().params,
            )
        else:
            delta_t = delta_t_max

        t_new = temperature + delta_t
        log_w_new = log_weights + delta_t * base_logp

        rng_chains = jax.random.split(rng, self.n_chains)
        x_new, ev_new = jax.vmap(
            lambda rk, xi, evi: self._nhr_step_one(rk, xi, evi, t_new)
        )(rng_chains, x, ev)
        x_new = x_new.clip(*self.bounds)
        return x_new, ev_new, log_w_new, t_new

    def resample(self, rng: PRNGKey, state: KnnNhrSmcState) -> KnnNhrSmcState:
        nhr_state = super().resample(rng, state)
        return state.replace(
            x=nhr_state.x,
            last_eval=nhr_state.last_eval,
            log_weights=jnp.zeros(self.n_chains)
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(self, rng: PRNGKey, state: KnnNhrSmcState) -> KnnNhrSmcState:
        def do_uniform_nhr(_):
            rng_chains = random.split(rng, self.n_chains)
            x_new, ev_new = jax.vmap(
                lambda rk, xi, evi: self._nhr_step_one(rk, xi, evi, 0.)
            )(rng_chains, state.x, state.last_eval)
            return state.replace(
                x=x_new.clip(*self.bounds),
                last_eval=ev_new,
                step_count=state.step_count + 1 + self.max_ls_iters,
            )

        def do_smc(_):
            x_new, ev_new, log_w_new, t_new = self._smc_step(
                rng, state.x, state.last_eval, state.log_weights, state.temperature,
            )
            return KnnNhrSmcState(
                x=x_new, last_eval=ev_new, log_weights=log_w_new,
                temperature=t_new, step_count=state.step_count + 1 + self.max_ls_iters,
            )

        return jax.lax.cond(
            state.step_count < self.n_uniform,
            do_uniform_nhr,
            do_smc,
            operand=None,
        )
