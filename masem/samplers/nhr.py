from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from chex import Array, PRNGKey, dataclass
from jax import random
from jax.scipy.sparse.linalg import cg

from masem.constants import FEAS_THRESH
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.samplers.sampler import Sampler
from masem.types import Eval


@dataclass
class NhrState:
    x: Array
    last_eval: Eval
    step_count: int = 0


class NHR(Sampler):
    """Non-linear hit-and-run sampler for constrained sampling on connected support.

    Based on: Toussaint et al's cpp from https://arxiv.org/pdf/2407.03035
    """

    def __init__(
        self,
        target_dist: ConstrainedDistribution,
        n_chains: int,
        n_down: int,
        max_step: float = 1.0,
        mu_slack: float = 1e3,
        max_ls_iters: int = 1,
        slack_margin: float = 1e-1,
        cg_maxiter: int = 50,
    ):
        super().__init__()
        self.target_dist = target_dist
        self.dim = target_dist.dim
        self.n_chains = n_chains
        self.n_down = n_down
        self.max_step = float(max_step)
        self.mu_slack = float(mu_slack)
        #self.mu_slack = 1e3 / self.dim #monkey patch during rebuttal to ablate mu
        self.max_ls_iters = int(max_ls_iters)
        self.bounds = target_dist.bounds
        self.slack_margin = slack_margin
        self.cg_maxiter = int(cg_maxiter)

    def initialize(self, rng: PRNGKey, x0: Array) -> NhrState:
        eval0 = jax.vmap(self.get_eval)(x0)
        return NhrState(x=x0, last_eval=eval0, step_count=0)

    @partial(jax.jit, static_argnames=("self",))
    def _downhill_step(
        self,
        rng: PRNGKey,
        x: Array,
        alpha: float = 1.0,
        max_step: float = 1.0,
        sigma: float = 1e-2,
        tol: float = 1e-8,
    ) -> Array:
        x += random.normal(rng, shape=x.shape) * sigma

        slack, vjp_fun = jax.vjp(self.target_dist.slack_vec, x)
        JT = lambda w: vjp_fun(w)[0]
        J = lambda v: jax.jvp(self.target_dist.slack_vec, (x,), (v,))[1]

        A = lambda v: JT(J(v)) + tol * v
        b = -JT(slack)

        delta, _ = cg(A, b, maxiter=self.cg_maxiter)  # solve system w conjugate grad for speed
        step = alpha * delta
        step *= jnp.minimum(
            1.0, max_step / (jnp.linalg.norm(step, keepdims=True) + tol)
        )
        return x + step

    def _project_tangent(self, x: Array, v: Array, ev: Eval, tol: float) -> Array:
        """Project v onto the tangent space of the equality constraints at x."""
        m = ev.jac_h.shape[0]
        if m == 0:
            return v

        # implicit projection
        gram = ev.jac_h @ ev.jac_h.T + tol * jnp.eye(m)
        alpha = ev.jac_h @ v
        cho = jax.scipy.linalg.cho_factor(gram)
        lam = jax.scipy.linalg.cho_solve(cho, alpha)

        return v - ev.jac_h.T @ lam

    def _nhr_step(
        self,
        rng: PRNGKey,
        x: Array,
        ev: Eval,
        eval_fn: Callable,
        bounds_lo: Optional[Array],
        bounds_hi: Optional[Array],
        max_step: float = 1.0,
        max_ls_iters: int = 10,
        mu_slack: float = 1e3,
        eps_slack: float = 0.0,
        temperature: float = 1.0,
        tol: float = 1e-8,
    ) -> Tuple[Array, Eval]:
        # Extra slack step on new proposal; does nothing if no slack
        rng, rng_d = jax.random.split(rng)
        x = self._downhill_step(
            rng_d,
            x,
            max_step=self.max_step,
            tol=self.target_dist.domain_tol,
            sigma=0.
        )
        ev = self.get_eval(x, epsilon_margin=eps_slack)

        # Project a random direction to the tangent space of equalities
        rng, k_dir, k_beta = random.split(rng, 3)
        rnd_dir = jax.random.normal(k_dir, shape=(self.dim,))
        rnd_dir = self._project_tangent(x, rnd_dir, ev, tol)
        rnd_dir /= jnp.linalg.norm(rnd_dir, keepdims=True) + tol

        # Clip bounds init
        t_lo, t_hi = self._clip_betas(bounds_lo - x, -rnd_dir, -max_step, max_step)
        t_lo, t_hi = self._clip_betas(x - bounds_hi, rnd_dir, t_lo, t_hi)

        # Line search iterations
        def ls_body(i, carry):
            rng, x_in, ev_in, t_lo_in, t_hi_in = carry
            # only clip by active inequality constraints (g > 0); zero out inactive rows
            active_ineq = (ev_in.g > 0).astype(ev_in.g.dtype)
            ev_g = jnp.concatenate(
                [ev_in.g * active_ineq, ev_in.h - self.slack_margin, -ev_in.h - self.slack_margin], axis=-1
            )
            ev_jac_g = jnp.concatenate(
                [ev_in.jac_g * active_ineq[:, None], ev_in.jac_h, -ev_in.jac_h], axis=-2
            )
            t_lo, t_hi = self._clip_betas(
                ev_g, ev_jac_g @ rnd_dir, t_lo_in, t_hi_in
            )
            interval_empty = t_lo >= t_hi

            def _stay(rng):
                return rng, x_in, ev_in, t_lo_in, t_hi_in

            def _propose_and_reject(rng):
                rng, rng_beta, rng_mh = random.split(rng, 3)
                u = random.uniform(rng_beta, ())
                beta = t_lo + u * (t_hi - t_lo)
                x_prop = x_in + beta * rnd_dir
                ev_prop = eval_fn(x_prop, eps_slack)

                # Compute reverse-proposal interval at x_prop for correct MH ratio
                t_lo_rev, t_hi_rev = self._clip_betas(bounds_lo - x_prop, -rnd_dir, -max_step, max_step)
                t_lo_rev, t_hi_rev = self._clip_betas(x_prop - bounds_hi, rnd_dir, t_lo_rev, t_hi_rev)
                active_rev = (ev_prop.g > 0).astype(ev_prop.g.dtype)
                ev_g_rev = jnp.concatenate(
                    [ev_prop.g * active_rev, ev_prop.h - self.slack_margin, -ev_prop.h - self.slack_margin], axis=-1
                )
                ev_jac_g_rev = jnp.concatenate(
                    [ev_prop.jac_g * active_rev[:, None], ev_prop.jac_h, -ev_prop.jac_h], axis=-2
                )
                t_lo_rev, t_hi_rev = self._clip_betas(
                    ev_g_rev, ev_jac_g_rev @ rnd_dir, t_lo_rev, t_hi_rev
                )

                # MH reject based on kernel and slack; use mu to add extra weight on slack
                energy_x = jnp.sum(-temperature * ev_in.base_logp) + mu_slack * jnp.sum(ev_in.slack)
                energy_xprop = jnp.sum(-temperature * ev_prop.base_logp) + mu_slack * jnp.sum(ev_prop.slack)
                log_q_in = -jnp.log(jnp.maximum(t_hi_rev - t_lo_rev, tol))
                log_q_prop = -jnp.log(jnp.maximum(t_hi - t_lo, tol))
                log_alpha = jnp.minimum(
                    0.0, energy_x - energy_xprop + log_q_in - log_q_prop
                )
                mh_kernel = jnp.log(random.uniform(rng_mh, (), minval=tol)) < log_alpha
                mh_accept = mh_kernel & (t_lo < t_hi) #& ineq_ok & slack_ok
                x_next = jnp.where(mh_accept, x_prop, x_in)
                ev_next = jax.tree.map(
                    lambda a_new, a_old: jnp.where(mh_accept, a_new, a_old),
                    ev_prop,
                    ev_in,
                )
                return rng, x_next, ev_next, t_lo, t_hi

            # Do propose and reject if we have an interval
            rng, x_next, ev_next, tl, th = jax.lax.cond(
                interval_empty, _stay, _propose_and_reject, operand=rng
            )

            return rng, x_next, ev_next, tl, th

        rng, x_out, ev_out, _, _ = jax.lax.fori_loop(
            0, max_ls_iters, ls_body, (rng, x, ev, t_lo, t_hi)
        )

        return x_out, ev_out

    @partial(jax.jit, static_argnames=("self",))
    def projection_step(self, rng: PRNGKey, state: NhrState) -> NhrState:
        rng_chains = jax.random.split(rng, self.n_chains)
        x_next = jax.vmap(
            lambda _r, _x: self._downhill_step(
                _r,
                _x,
                max_step=self.max_step,
                tol=self.target_dist.domain_tol,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )(rng_chains, state.x)
        x_next = x_next.clip(*self.bounds)
        ev_next = jax.vmap(self.get_eval)(x_next)

        return NhrState(
            x=x_next,
            last_eval=ev_next,
            step_count=state.step_count + 1
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(self, rng: PRNGKey, state: NhrState) -> NhrState:
        rng_chains = jax.random.split(rng, self.n_chains)

        def one_chain(rk, xi, evi):
            return self._nhr_step(
                rk,
                xi,
                evi,
                self.get_eval,
                self.bounds[0],
                self.bounds[1],
                max_step=self.max_step,
                max_ls_iters=self.max_ls_iters,
                mu_slack=self.mu_slack,
            )

        x_new, ev_new = jax.vmap(one_chain)(rng_chains, state.x, state.last_eval)
        x_new = x_new.clip(*self.bounds)
        return NhrState(
            x=x_new,
            last_eval=ev_new,
            step_count=state.step_count + 1 + self.max_ls_iters
        )

    @staticmethod
    def _clip_betas(
        gbar: Array, direction: Array, beta_lo: float, beta_hi: float, tol: float = 1e-6
    ) -> tuple[float | Array, float | Array]:
        """Solves for max step along direction that is feasible using linearization gbar.

        Given gbar = g(x) + nabla g(x)T d * t --> t leq frac{-g(x)}{nabla g(x) ^ T}
        """
        beta = -gbar / (
            direction + (direction == 0.0) * jnp.sign(direction + tol) * tol
        )
        dir_active = jnp.abs(direction) > tol
        beta_lo_cand = jnp.where(dir_active & (direction < 0.0), beta, beta_lo)
        beta_up_cand = jnp.where(dir_active & (direction > 0.0), beta, beta_hi)
        new_beta_lo = jnp.maximum(beta_lo, jnp.max(beta_lo_cand))  # clip over all dims
        new_beta_hi = jnp.minimum(beta_hi, jnp.min(beta_up_cand))

        return new_beta_lo, new_beta_hi

    @partial(jax.jit, static_argnames=("self",))
    def resample_infeasible(self, rng: PRNGKey, state: NhrState) -> NhrState:
        """Reject the infeas points post projection."""
        x, ev = state.x, state.last_eval
        feasible = jnp.max(ev.slack, axis=-1) < FEAS_THRESH
        any_feasible = jnp.any(feasible)
        w = jnp.where(feasible, 1.0, 0.0)
        w = jnp.where(any_feasible, w / jnp.sum(w), jnp.ones(self.n_chains) / self.n_chains)
        idx = resampling.stratified(rng, w, self.n_chains)

        def _replace(a):
            keep_mask = feasible.reshape((-1,) + (1,) * (a.ndim - 1))
            return jnp.where(keep_mask, a, a[idx])

        x_rs = _replace(x)
        ev_rs = jax.tree.map(lambda a: _replace(a) if isinstance(a, jax.Array) else a, ev)

        return state.replace(x=x_rs, last_eval=ev_rs)
