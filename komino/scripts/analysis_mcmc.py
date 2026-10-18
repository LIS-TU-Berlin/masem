from dotenv import load_dotenv
load_dotenv()

import os
from time import time

import hydra
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import wandb
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm

from komino.constants import REPO_PATH
from komino.eval.compute_metrics import get_metrics_fn
from komino.utils import seed_all, linear_project, reset_device_memory
from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.samplers.sampler import Sampler

OmegaConf.register_new_resolver("eval", eval)


def setup(cfg: OmegaConf):
    if use_wandb := cfg.get("use_wandb", False):
        wandb.init(
            name=f"{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}D_seed{cfg.seed}",
            entity=os.environ.get("WANDB_ENTITY"),
            project=cfg.project_name,
            group=f"{cfg.algorithm.name}",
            job_type=f"{cfg.target.name}_{cfg.target.dim}D",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    # 0. create sampler & dist
    seed_all(cfg.seed)
    rng = jr.PRNGKey(cfg.seed)
    rng, rng_init = jr.split(rng)
    dist: ConstrainedDistribution = hydra.utils.instantiate(cfg.target.fn)
    sampler: Sampler = hydra.utils.instantiate(
        cfg.algorithm.fn, target_dist=dist
    )
    x0 = jr.normal(rng_init, (cfg.eval_samples, dist.dim)) * dist.init_scale
    state = sampler.initialize(rng_init, x0=x0)

    # # Log the component memberships
    # if hasattr(dist, "empirical_component_frequencies"):
    #     q0 = onp.asarray(dist.empirical_component_frequencies(x0))
    #     onp.save(data_path / "alpha0_components.npy", q0)

    return rng, dist, sampler, state, use_wandb, get_metrics_fn(cfg.target.name)


@hydra.main(config_path=str(REPO_PATH.parent / "configs"), config_name="base_cfg", version_base="1.2")
def sample_mcmc(cfg: OmegaConf):
    rng, dist, sampler, state, use_wandb, metrics_fn = setup(cfg)

    # 1. Run MCMC
    n_iter = cfg.algorithm.n_iter
    eval_every = cfg.eval_every
    log_iters = onp.arange(0, n_iter + 1, eval_every)
    log_iters[-1] = n_iter
    times = jnp.zeros((len(log_iters)))
    iters = jnp.zeros((len(log_iters)))

    allow_final_mixing = cfg.get("allow_final_mixing", False)

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
        # log whenever step_count has crossed the next scheduled checkpoint
        while k < len(log_iters) and sc >= log_iters[k]:
            target_samples = sampler.get_samples(state)
            steps = steps.at[k].set(target_samples)
            times = times.at[k].set(time() - t_start)
            iters = iters.at[k].set(sc)
            k += 1
        if sc >= n_iter:
            target_samples = sampler.get_samples(state)
            steps = steps.at[k].set(target_samples)
            times = times.at[k].set(time() - t_start)
            iters = iters.at[k].set(sc)
            break
    pbar.close()

    # 2. compute metrics
    metrics = metrics_fn(rng, steps, dist)
    metrics["n_eval"] = state.step_count
    metrics["wallclock"] = times
    metrics["iters"] = iters

    # metrics after mixing
    # technically useless here, but for comparison with resampling
    if allow_final_mixing:
        for _ in tqdm(range(50), desc="Final mixing", colour="blue"):
            rng, rng_step = jr.split(rng)
            state = sampler.step(rng_step, state)

    # 4. Store metrics & perhaps final samples
    res_dir = REPO_PATH.parent / cfg.result_dir
    print("Saving to: ", res_dir)
    os.makedirs(res_dir, exist_ok=True)

    if cfg.save_samples:
        final_samples = steps[-1]
        final_samples_proj = linear_project(dist, final_samples)
        onp.save(res_dir / "samples.npy", final_samples)
        onp.save(res_dir / "samples_proj.npy", final_samples_proj)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(res_dir / "metric_summary.csv")
    OmegaConf.save(cfg, res_dir / "config.yaml")

    if use_wandb:   # awkward wandb cannot accept logs from multiple steps at once
        for i in tqdm(range(iters.shape[0]), colour="blue", desc="Upload"):
            mi = {k: v[i] if onp.ndim(v) > 0 else v for (k, v) in metrics.items()}
            wandb.log(mi, step=i)
        


        # get auc for hparam tuning
        if "sd_proj" in metrics:
            auc = onp.trapezoid(metrics["sd_proj"], iters)
            norm_auc = auc / (iters[-1] - iters[0])
            wandb.run.summary["eval/sd_norm_auc"] = float(norm_auc)

        if allow_final_mixing:
            final_metrics = metrics_fn(rng, sampler.get_samples(state)[None], dist)
            for k, v in final_metrics.items():
                wandb.run.summary["eval/"+k+"_am"] = float(v)

        wandb.finish()


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    sample_mcmc()

    reset_device_memory()
