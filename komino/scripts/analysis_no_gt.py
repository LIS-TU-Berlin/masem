from dotenv import load_dotenv
load_dotenv()

import os
from time import time

import hydra
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
from omegaconf import OmegaConf
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from komino.constants import REPO_PATH
from komino.utils import seed_all
from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.samplers.sampler import Sampler
from komino.eval.performance_metrics import constraint_violations


OmegaConf.register_new_resolver("floordiv", lambda a, b: int(a) // int(b))
OmegaConf.register_new_resolver("mult", lambda a, b: a * b)
OmegaConf.register_new_resolver("sqrt", lambda a: a ** 0.5)
OmegaConf.register_new_resolver("eval", eval)


def mst_size(x: onp.ndarray) -> float:
    """Total edge weight of the Euclidean minimum spanning tree of x (N, d)."""
    dist_matrix = squareform(pdist(x, metric="euclidean"))
    mst = minimum_spanning_tree(dist_matrix)
    return float(mst.data.sum())


@hydra.main(config_path=str(REPO_PATH.parent / "configs"), config_name="base_cfg")
def sample_robot(cfg: OmegaConf):
    # 0. create sampler & dist
    seed_all(cfg.seed)
    rng = jr.PRNGKey(cfg.seed)
    rng, rng_init = jr.split(rng)
    dist: ConstrainedDistribution = hydra.utils.instantiate(cfg.target.fn)
    sampler: Sampler = hydra.utils.instantiate(
        cfg.algorithm.fn, target_dist=dist
    )
    state = sampler.initialize(rng_init, initial_scale=cfg.target.all.initial_scale)

    # 1. Run sampler
    n_iter = cfg.algorithm.n_iter
    eval_every = cfg.n_evals
    log_iters = onp.arange(0, n_iter + 1, eval_every)
    log_iters[-1] = n_iter
    times = jnp.zeros((len(log_iters)))
    iters = jnp.zeros((len(log_iters)))

    # dummy pre-step for compilation -- not used for computation
    sampler.step(rng, state)

    # Log initial
    init_samples = sampler.get_samples(state)
    steps = jnp.empty((len(log_iters), *init_samples.shape))
    steps = steps.at[0].set(init_samples)
    k = 1
    t_start = time()
    pbar = tqdm(total=n_iter, colour="blue", desc="Sampling")
    prev_sc = 0
    while True:
        rng, rng_step = jr.split(rng)
        state = sampler.step(rng_step, state)
        sc = int(state.step_count)
        pbar.update(sc - prev_sc)
        prev_sc = sc
        while k < len(log_iters) and sc >= log_iters[k]:
            target_samples = sampler.get_samples(state)
            steps = steps.at[k].set(target_samples)
            times = times.at[k].set(time() - t_start)
            iters = iters.at[k].set(sc)
            k += 1
        if sc >= n_iter:
            break
    pbar.close()

    # 2. Compute metrics (no ground truth)
    metrics = {
        "n_feas": onp.empty((steps.shape[0])),
        "slack": onp.empty((steps.shape[0])),
        "mean_logp": onp.empty((steps.shape[0])),
        "mst_size": onp.empty((steps.shape[0])),
    }
    for i in tqdm(range(steps.shape[0]), colour="blue", desc="Computing Metrics"):
        x = steps[i]
        feas = dist.feasible(x, tol=1e-3)
        metrics["n_feas"][i] = float(jnp.sum(feas))
        metrics["slack"][i] = float(constraint_violations(dist, x))

        # Expected log probability over feasible samples only
        logp = dist.base_unnorm_log_prob(x)
        feas_mask = onp.array(feas, dtype=bool)
        if feas_mask.any():
            metrics["mean_logp"][i] = float(jnp.mean(logp[feas_mask]))
        else:
            metrics["mean_logp"][i] = float("nan")

        # MST size as a proxy for sample set diversity
        x_np = onp.array(x)
        metrics["mst_size"][i] = mst_size(x_np)

    # Log timing
    metrics["n_eval"] = state.step_count
    metrics["wallclock"] = onp.array(times)
    metrics["iters"] = onp.array(iters)

    # 3. Store metrics & final samples
    res_dir = REPO_PATH.parent / cfg.result_dir
    print("Saving to: ", res_dir)
    os.makedirs(res_dir, exist_ok=True)

    if cfg.save_samples:
        onp.save(res_dir / "samples.npy", onp.array(steps[-1]))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(res_dir / "metric_summary.csv")

    OmegaConf.save(cfg, res_dir / "config.yaml")


if __name__ == "__main__":
    sample_robot()
