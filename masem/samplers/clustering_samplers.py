from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, dataclass
from jax import random
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.samplers.nhr import NHR, NhrState
from masem.samplers.sampler import ResamplingSampler
from masem.types import Eval


@dataclass
class ClusterNhrState:
    x: Array
    last_eval: Eval
    cluster_ids: Array
    log_weights: Array
    step_count: int = 0


class ClusterNhr(NHR, ResamplingSampler):
    """NHR with linkage-clustering stratified resampling.

    Particles are split into K fixed clusters after projection.
    Each resample call re-clusters, allocates n_target=N//K particles per cluster (equal allocation
    with remainder distributed to first clusters), and assigns IS weights
    log(V_k / n_target) where V_k is estimated from intra-cluster distances.
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
        n_inner: int = 50,
        cluster_threshold: float = 1.0,
        p_tau: float = 0.0,  # weight for base log p in non-uniform case
        linkage_method: str = 'ward',
    ):
        super().__init__(target_dist, n_chains, n_down, max_step, mu_slack, max_ls_iters, slack_margin)
        self.cluster_threshold = float(cluster_threshold * (self.bounds[1] - self.bounds[0]) * np.sqrt(self.dim / 2))
        self.p_tau = p_tau
        self.linkage_method = linkage_method

        self.n_down = n_down
        self.n_inner = n_inner

    def initialize(self, rng: PRNGKey, x0: Array) -> ClusterNhrState:
        state = super().initialize(rng, x0)
        return ClusterNhrState(
            x=state.x,
            last_eval=state.last_eval,
            cluster_ids=jnp.zeros(self.n_chains, dtype=jnp.int32),
            log_weights=jnp.zeros(self.n_chains),
            step_count=0,
        )

    def projection_step(self, rng: PRNGKey, state: ClusterNhrState) -> ClusterNhrState:
        rng_chains = random.split(rng, self.n_chains)
        x_new = jax.vmap(
            lambda _r, _x: self._downhill_step(
                _r, _x,
                max_step=self.max_step,
                tol=self.target_dist.domain_tol,
            )
        )(rng_chains, state.x)
        x_new = x_new.clip(*self.bounds)
        ev_new = jax.vmap(self.get_eval)(x_new)
        return state.replace(x=x_new, last_eval=ev_new, step_count=state.step_count + 1)

    def resample(self, rng: PRNGKey, state: ClusterNhrState) -> ClusterNhrState:
        N = self.n_chains
        x_np = np.array(state.x)
        base_logp_np = np.array(state.last_eval.base_logp).flatten()
        slack_sq_np = np.array(jnp.sum(state.last_eval.slack ** 2, axis=-1))  # (N,)

        # Discard infeasible samples before clustering; fall back to all if none are feasible
        feas_idx = np.where(slack_sq_np < 1e-12)[0]
        if len(feas_idx) == 0:
            feas_idx = np.arange(N)
        x_feas = x_np[feas_idx]
        base_logp_feas = base_logp_np[feas_idx]
        n_feas = len(feas_idx)

        # Unsupervised linkage clustering on feasible particles only
        if n_feas > 1:
            Z = linkage(pdist(x_feas), method=self.linkage_method)
            labels = fcluster(Z, t=self.cluster_threshold, criterion='distance') - 1
        else:
            labels = np.zeros(n_feas, dtype=int)
        K = int(labels.max()) + 1

        n_per_base = N // K
        remainder = N % K
        rng_np = np.random.default_rng(int(rng[0]))   # bit hacky but works
        chosen_indices = []
        cluster_ids_out = []
        log_weights_out = []

        for k in range(K):
            n_target = n_per_base + (1 if k < remainder else 0)
            idx_k = np.where(labels == k)[0]  # indices into x_feas

            if len(idx_k) == 0:
                idx_k = np.arange(n_feas)

            n_k = len(idx_k)
            x_k = x_feas[idx_k]

            # Estimate volumes via pwds inside cluster
            mean_dist = pdist(x_k).mean() if n_k > 1 else 1e-6
            vol_k = max(mean_dist ** self.dim, 1e-12)   # TODO: do we need to put manifold dim here or does it cancel?

            # Pure component-level IS weight; all chosen particles are feasible
            log_w_k = np.log(vol_k) - np.log(n_target)

            # Within-cluster sampling probabilities
            logw_local = base_logp_feas[idx_k] * self.p_tau
            logw_local -= logw_local.max()
            w_local = np.exp(logw_local)
            w_local /= w_local.sum()

            sub = rng_np.choice(n_k, size=n_target, replace=(n_k < n_target), p=w_local)
            chosen_indices.append(feas_idx[idx_k[sub]])  # map back to original indices
            cluster_ids_out.extend([k] * n_target)
            log_weights_out.extend([log_w_k] * n_target)

        jax_idx = jnp.array(np.concatenate(chosen_indices))
        ev_new = jax.tree.map(
            lambda a: a[jax_idx] if isinstance(a, jax.Array) else a,
            state.last_eval,
        )
        return state.replace(
            x=state.x[jax_idx],
            last_eval=ev_new,
            cluster_ids=jnp.array(cluster_ids_out, dtype=jnp.int32),
            log_weights=jnp.array(log_weights_out, dtype=jnp.float32),
        )

    def resample_infeasible(self, rng: PRNGKey, state: ClusterNhrState, tol: float = 1e-2) -> ClusterNhrState:
        """Replaces infeasible samples with feasible ones, if any. No clustering."""
        feasible_mask = self.target_dist.feasible(state.x, tol=tol)
        n_feas = feasible_mask.sum()
        if n_feas == 0:
            Warning("No feasible samples, returning state unchanged.")
            return state
        if n_feas == self.n_chains:
            return state

        feasible_idx = jnp.where(feasible_mask)[0]
        new_idx = jax.random.choice(rng, feasible_idx, shape=(self.n_chains - n_feas,), replace=True)
        idx = jnp.concat([feasible_idx, new_idx])
        x, ev = state.x, state.last_eval
        x_rs = x[idx]
        ev_rs = jax.tree.map(lambda a: a[idx] if isinstance(a, jax.Array) else a, ev)
        return state.replace(x=x_rs, last_eval=ev_rs)

    @partial(jax.jit, static_argnames=("self",))
    def step(self, rng: PRNGKey, state: ClusterNhrState) -> ClusterNhrState:
        def one_chain(rk, xi, evi):
            return self._nhr_step(
                rk, xi, evi,
                self.get_eval,
                self.bounds[0],
                self.bounds[1],
                max_step=self.max_step,
                max_ls_iters=self.max_ls_iters,
                mu_slack=self.mu_slack,
            )
        rng_chains = random.split(rng, self.n_chains)
        x_new, ev_new = jax.vmap(one_chain)(rng_chains, state.x.clip(*self.bounds), state.last_eval)

        return state.replace(x=x_new, last_eval=ev_new, step_count=state.step_count + 1 + self.max_ls_iters)
