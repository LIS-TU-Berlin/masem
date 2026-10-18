from functools import partial
from typing import Optional

import jax
import jax.random as jr
import jax.numpy as jnp
from chex import Array, PRNGKey, dataclass
from jaxopt import Bisection
from blackjax.smc import resampling

from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.samplers.sampler import Sampler
from komino.types import Eval


def broadcast_mask(mask: Array, x: Array) -> Array:
    return mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))


@dataclass
class ScmcState:
    x: Array
    ev: Eval
    log_p: Array
    weights: Array
    mu: float
    q_prop: Array
    level_count: int = 0
    advance_count: int = 0
    step_count: int = 0   # total count of target evals


class SoftSCMC(Sampler):
    """SCMC implementation for Golchi et al (2016).

    Based on their R code:
    https://github.com/sgolchi/SCMC-/blob/master/SCMC.R

    NOTE: this implementation follows the convention of
            p(x) \propto f(x)*slack(x)/mu
        This is different from the usual f(x)slack(x)^mu!
    """

    def __init__(
        self,
        target_dist: ConstrainedDistribution,
        n_chains: int,
        steps_per_level: int,
        bisection_steps: int = 60,
        mu_min: float = 1e-3,
        mu_max: float = 1e3,
        step_exp: float = 1.0,
        ess_target: Optional[float] = None,
    ):
        super().__init__()
        self.target_dist = target_dist
        self.dim = target_dist.dim
        self.n_bound = int(2 * self.dim)
        self.n_chains = n_chains
        self.steps_per_level = steps_per_level
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.step_exp = step_exp
        self.bisection_steps = bisection_steps
        self.ess_target = ess_target * self.n_chains if ess_target is not None else self.n_chains / 2

    def initialize(self, rng: PRNGKey, x0: Array) -> ScmcState:
        ev0 = jax.vmap(self.get_eval)(x0)
        return ScmcState(
            x=x0,
            ev=ev0,
            weights=jnp.ones(self.n_chains) / self.n_chains,
            mu=self.mu_max,
            log_p=self._soft_logp(ev0, mu=self.mu_max),
            q_prop=jnp.std(x0, axis=0, ddof=1)  # pholder
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(self, rng: PRNGKey, state: ScmcState) -> ScmcState:
        """Either we advance the sequence or we use Gibbs sampling for rejuvenation."""
        return jax.lax.cond(
            state.advance_count % self.steps_per_level == 0,
            self._advance_smc,
            self._gibbs_sample,
            rng, state
        )

    def _advance_smc(self, rng: PRNGKey, state: ScmcState):
        """The actual SMC step:

        1. Find mu (ie advance)
        2. Recompute weights
        3. Resample
        """
        t_next = state.level_count + 1
        log_p0 = state.log_p
        log_w0 = jnp.log(state.weights)

        def _ess_from_logp(lp: Array) -> Array:
            log_w = log_w0 + (lp - log_p0)
            log_w = log_w - jnp.max(log_w)
            log_norm = jax.scipy.special.logsumexp(log_w)
            log_ess = 2.0 * log_norm - jax.scipy.special.logsumexp(2.0 * log_w)
            return jnp.exp(log_ess) - self.ess_target

        bisection_target = lambda mu: _ess_from_logp(self._soft_logp(state.ev, mu))

        # 1. find next mu: check mu_min first; if ESS sufficient skip bisection
        logp_at_mu_min = self._soft_logp(state.ev, self.mu_min)
        ess_final = _ess_from_logp(logp_at_mu_min)

        def mu_optimization():
            lo = jnp.where(t_next < 3, 0.1, self.mu_min)
            hi = jnp.where(t_next != 2, state.mu, 1e5)
            bisection = Bisection(bisection_target, lo, hi, maxiter=self.bisection_steps, check_bracket=False)
            return bisection.run().params

        mu_next = jax.lax.cond(ess_final >= 0, lambda: self.mu_min, mu_optimization)
        mu_next = jnp.clip(mu_next, self.mu_min, self.mu_max)

        # 2. Recompute weights for new mu — reuse logp_at_mu_min when mu_next == mu_min
        logp_new = jax.lax.cond(
            ess_final >= 0,
            lambda: logp_at_mu_min,
            lambda: self._soft_logp(state.ev, mu_next),
        )
        log_w_new = log_w0 + (logp_new - log_p0)
        log_w_new = log_w_new - jnp.max(log_w_new)
        w_new = jnp.exp(log_w_new)
        w_new = w_new / jnp.sum(w_new)

        # 3. Resample
        idx = resampling.stratified(rng, w_new, self.n_chains)
        resamp_x = state.x[idx]
        resamp_logp = logp_new[idx]
        resamp_ev = jax.tree.map(lambda x: x[idx] if isinstance(x, jax.Array) else x, state.ev)
        # After resampling, we will use w from original samples
        resamp_w = jnp.ones((self.n_chains,)) / self.n_chains
        q_prop = jnp.std(resamp_x, axis=0, ddof=1) / (t_next ** self.step_exp)

        return ScmcState(
            x=resamp_x,
            weights=resamp_w,
            q_prop=q_prop,
            log_p=resamp_logp,
            mu=mu_next,
            ev=resamp_ev,
            step_count=state.step_count,
            level_count=t_next,
            advance_count=state.advance_count + 1,
        )

    def _gibbs_sample(self, rng: PRNGKey, state: ScmcState) -> ScmcState:
        """
        One full Metropolis-within-Gibbs sweep over all dimensions d=0..D-1,
        updating ALL particles in parallel for each dimension.
        """
        def update1d(d, carry):
            x_cur, logp_cur, ev_cur, key_cur = carry
            key_cur, key_delta, key_u = jr.split(key_cur, 3)
            delta = jr.normal(key_delta, shape=(self.n_chains,)) * state.q_prop[d]
            x_prop = x_cur.at[:, d].add(delta)
            ev_prop = jax.vmap(self.target_dist.eval)(x_prop)
            logp_prop = self._soft_logp(ev_prop, state.mu)
            log_alpha = jnp.minimum(0., logp_prop - logp_cur)
            u = jr.uniform(key_u, shape=(self.n_chains,))
            mh_accept = jnp.log(u) <= log_alpha
            x_next = jnp.where(broadcast_mask(mh_accept, x_prop), x_prop, x_cur)
            ev_next = jax.tree.map(
                lambda a_new, a_old: jnp.where(broadcast_mask(mh_accept, a_new), a_new, a_old),
                ev_prop,
                ev_cur,
            )
            logp_next = jnp.where(mh_accept, logp_prop, logp_cur)
            return x_next, logp_next, ev_next, key_cur

        # Iterate dimensions sequentially (true Gibbs ordering over d)
        x_out, logp_out, ev_out, _ = jax.lax.fori_loop(
            0,
            self.target_dist.dim,
            update1d,
            (state.x, state.log_p, state.ev, rng),
        )

        return ScmcState(
            x=x_out,
            ev=ev_out,
            weights=state.weights,
            log_p=logp_out,
            mu=state.mu,
            q_prop=state.q_prop,
            step_count=state.step_count+self.target_dist.dim,
            advance_count=state.advance_count + 1,
        )

    def _soft_logp(self, ev: Eval, mu: float) -> Array:
        """We divide the slack by mu in the original implementation.
        --> we want to anneal this to 0.
        """
        base_logp = jnp.ravel(ev.base_logp)
        log_Phi_g = jnp.zeros_like(base_logp)
        log_Phi_h = jnp.zeros_like(base_logp)
        log_Phi_bound = jnp.zeros_like(base_logp)

        # Inequality constraints.
        if ev.g.ndim > 0 and ev.g.shape[-1] > 0:
            g = ev.g
            if self.n_bound > 0:
                # Last 2 * dim inequality constraints are assumed to be box bounds.
                g_soft = g[..., :-self.n_bound]
                g_bound = g[..., -self.n_bound:]
                z_bound = -g_bound / self.mu_min
                log_Phi_bound = jax.scipy.special.log_ndtr(z_bound).sum(axis=-1)
            else:
                g_soft = g

            # Softly anneal only the non-bound inequalities.
            if g_soft.shape[-1] > 0:
                z_g = -g_soft / mu
                log_Phi_g = jax.scipy.special.log_ndtr(z_g).sum(axis=-1)

        # Equality constraints
        if ev.h.ndim > 0 and ev.h.shape[-1] > 0:
            z_h = -jnp.abs(ev.h) / mu
            log_Phi_h = jax.scipy.special.log_ndtr(z_h).sum(axis=-1)

        logp = base_logp + log_Phi_g + log_Phi_h + log_Phi_bound

        return jnp.ravel(logp)
