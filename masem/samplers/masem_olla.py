from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from chex import Array, PRNGKey
from jax.scipy.sparse.linalg import cg

from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.samplers.olla import OLLA, OLLAState
from masem.samplers.sampler import ResamplingSampler
from masem.samplers.utils import knn_log_radius_topk
from masem.types import Eval


class MasemOlla(OLLA, ResamplingSampler):

    def __init__(
        self,
        target_dist: ConstrainedDistribution,
        n_chains: int,
        n_down: int,
        step_size: float = 5e-4,
        alpha: float = 2e3,
        max_norm: float = 1e4,
        proj_damp: float = 1e-6,
        n_hutch_samples: int = 0,
        knn_tau: float = 1.0,
        knn_k: int = 10,
        knn_eps: float = 1e-6,
        n_inner: int = 50,
        p_tau: float = 0.0,
        max_step: Optional[float] = None,
        mu_slack: float = 1e3,
        use_kmean: bool = True
    ):
        super().__init__(target_dist, n_chains, step_size, alpha, max_norm, proj_damp, n_hutch_samples)
        self.knn_tau = knn_tau
        self.knn_k = int(knn_k)
        self.knn_eps = knn_eps
        self.p_tau = p_tau
        self.max_step = max_step or 1.0  # TODO: safe?
        self.n_down = n_down
        self.n_inner = n_inner
        self.use_kmean = use_kmean
        self.mu_slack = mu_slack

        self._olla_step = jax.jit(jax.vmap(
            lambda rk, xi, evi: self._olla_step_one(rk, xi, evi, 1.)
        ))

    def _olla_step_one(self, rng: PRNGKey, x: Array, ev: Eval, temperature: Array) -> tuple[Array, Eval]:
        """Temperature-scaled OLLA step targeting p_t ∝ exp(t * base_logp)."""
        rng_hutch, rng_noise = jax.random.split(rng)
        Js = ev.jac_slack
        n_active = Js.shape[0]

        if n_active == 0:
            rnd = jax.random.normal(rng_noise, x.shape)
            x_new = x + self.step_size * temperature * ev.base_score + self.noise_scale * rnd
            return x_new, self.target_dist.eval(x_new)

        matvec = lambda v: Js @ (Js.T @ v) + self.proj_damp * v

        if self.n_hutch_samples > 0:
            mask_g = (ev.g <= self.target_dist.eps).astype(x.dtype)
            selection = jnp.concatenate([mask_g, jnp.ones((ev.h.shape[0],))], axis=0)
            trace_vec = self._hutchinson_trace_vec(rng_hutch, x, Js) * selection
        else:
            trace_vec = jnp.zeros(n_active)

        t_score = temperature * ev.base_score
        rhs = Js @ t_score + self.alpha * ev.slack + trace_vec
        lam, _ = cg(matvec, rhs, maxiter=self.cg_maxiter)
        drift = t_score - Js.T @ lam
        drift *= jnp.minimum(1.0, self.max_norm / (jnp.linalg.norm(drift) + 1e-8))

        xi = jax.random.normal(rng_noise, x.shape)
        lam_noise, _ = cg(matvec, Js @ xi, maxiter=self.cg_maxiter)
        x_new = x + self.step_size * drift + self.noise_scale * (xi - Js.T @ lam_noise)
        return x_new, self.target_dist.eval(x_new)

    @partial(jax.jit, static_argnames=("self",))
    def projection_step(self, rng: PRNGKey, state: OLLAState) -> OLLAState:
        tol = self.target_dist.domain_tol

        def _gn_step(x):
            slack, vjp_fun = jax.vjp(self.target_dist.slack_vec, x)
            JT = lambda w: vjp_fun(w)[0]
            J = lambda v: jax.jvp(self.target_dist.slack_vec, (x,), (v,))[1]
            delta, _ = cg(lambda v: JT(J(v)) + tol * v, -JT(slack), maxiter=50)
            return x + delta * jnp.minimum(1.0, self.max_step / (jnp.linalg.norm(delta) + tol))

        x_new = jax.vmap(_gn_step)(state.x)
        x_new = x_new.clip(*self.bounds)
        ev_new = jax.vmap(self.target_dist.eval)(x_new)
        return state.replace(x=x_new, ev=ev_new, step_count=state.step_count + 1)

    @partial(jax.jit, static_argnames=("self",))
    def resample(self, rng: PRNGKey, state: OLLAState) -> OLLAState:
        x, ev = state.x, state.ev
        log_r = knn_log_radius_topk(x, k=self.knn_k, eps=self.knn_eps, use_kmean=self.use_kmean)
        logw = self.knn_tau * log_r
        logw = logw + ev.base_logp * self.p_tau
        slack_penalty = jnp.sum(ev.slack ** 2, axis=-1)
        logw -= self.mu_slack * slack_penalty
        w = jax.nn.softmax(logw)
        idx = resampling.stratified(rng, w, x.shape[0])
        x_rs = x[idx]
        ev_rs = jax.tree.map(lambda a: a[idx] if isinstance(a, jax.Array) else a, ev)
        return state.replace(x=x_rs, ev=ev_rs)

    # def resample_infeasible(self, rng: PRNGKey, state: OLLAState, tol: float = 1e-2) -> OLLAState:
    #     feasible_mask = self.target_dist.feasible(state.x, tol=tol)
    #     n_feas = feasible_mask.sum()
    #     if n_feas == 0:
    #         Warning("No feasible samples, returning state unchanged.")
    #         return state
    #     if n_feas == self.n_chains:
    #         return state
    #
    #     feasible_idx = jnp.where(feasible_mask)[0]
    #     new_idx = jax.random.choice(rng, feasible_idx, shape=(self.n_chains - n_feas,), replace=True)
    #     idx = jnp.concat([feasible_idx, new_idx])
    #     x, ev = state.x, state.last_eval
    #     x_rs = x[idx]
    #     ev_rs = jax.tree.map(lambda a: a[idx] if isinstance(a, jax.Array) else a, ev)
    #     return state.replace(x=x_rs, last_eval=ev_rs)

    @partial(jax.jit, static_argnames=("self",))
    def step(self, rng: PRNGKey, state: OLLAState) -> OLLAState:
        rngs = jax.random.split(rng, self.n_chains)
        x_new, ev_new = self._olla_step(rngs, state.x, state.ev)
        x_new = x_new.clip(*self.bounds)
        return state.replace(x=x_new, ev=ev_new, step_count=state.step_count + 1)
