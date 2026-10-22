from functools import partial

import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from chex import PRNGKey

from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.samplers.nhr import NHR, NhrState
from masem.samplers.sampler import ResamplingSampler
from masem.samplers.utils import knn_log_radius_topk


class MasemNHR(NHR, ResamplingSampler):

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
