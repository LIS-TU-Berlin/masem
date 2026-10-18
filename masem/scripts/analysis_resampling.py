from dotenv import load_dotenv

load_dotenv()

import os
from time import time

import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
import pandas as pd
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from masem.distributions.capsule_grasp.capsule_grasp import CapsuleGrasp
from masem.constants import REPO_PATH
from masem.scripts.analysis_mcmc import setup
from masem.utils import linear_project, reset_device_memory


@hydra.main(config_path=str(REPO_PATH.parent / "configs"), config_name="base_cfg", version_base="1.2")
def resample_mcmc(cfg: OmegaConf):
    rng, dist, sampler, state, use_wandb, metrics_fn = setup(cfg)

    # 1. Run MCMC
    n_project = sampler.n_down
    n_inner = getattr(sampler, "n_inner", None)  # None for non-knn samplers
    n_iter_total = cfg.algorithm.n_iter
    eval_every = cfg.eval_every
    do_periodic_resample = n_inner is not None
    resamp_infeas = cfg.algorithm.get("resample_infeasible", False) #and not do_periodic_resample
    allow_final_mixing = cfg.get("allow_final_mixing", False)

    log_iters = onp.arange(0, n_iter_total + 1, eval_every)
    log_iters[-1] = n_iter_total
    times = jnp.zeros((len(log_iters)))
    iters = jnp.zeros((len(log_iters)))

    # dummy pre-step for compilation -- not used for computation
    sampler.step(rng, state)
    if hasattr(sampler, "projection_step"):
        sampler.projection_step(rng, state)
    if hasattr(sampler, "resample"):
        sampler.resample(rng, state)
    if hasattr(sampler, "resample_infeasible"):
        sampler.resample_infeasible(rng, state)

    # Log initial
    init_samples = sampler.get_samples(state)
    steps = jnp.empty((len(log_iters), *init_samples.shape))
    steps = steps.at[0].set(init_samples)
    k = 1
    t_start = time()
    pbar = tqdm(total=n_iter_total, colour="blue", desc="Projecting")
    prev_sc = 0

    # plot_and_print(sampler, dist, state, 0)

    # Phase 1: Projection
    while True:
        rng, rng_step = jr.split(rng)
        state = sampler.projection_step(rng_step, state)
        sc = int(state.step_count)
        pbar.update(sc - prev_sc)
        prev_sc = sc
        while k < len(log_iters) and sc >= log_iters[k]:
            target_samples = sampler.get_samples(state)
            steps = steps.at[k].set(target_samples)
            times = times.at[k].set(time() - t_start)
            iters = iters.at[k].set(sc)
            k += 1
        if sc >= n_project:
            break

    # Single resample after projection to remove infeasible particles
    if resamp_infeas and hasattr(sampler, "resample_infeasible"):
        print("Resamp infeas")
        rng, rng_re = jr.split(rng)
        state = sampler.resample_infeasible(rng_re, state)

    # plot_and_print(sampler, dist, state, state.step_count)
    has_comp_freq = hasattr(dist, "empirical_component_frequencies")
    comp_freq0 = None
    if has_comp_freq:
        comp_freq0 = onp.asarray(
            dist.empirical_component_frequencies(sampler.get_samples(state))
        )

    # Phase 2: Sampling (with periodic resampling for knn samplers)
    pbar.set_description("Sampling")
    sampling_steps = 0
    while True:
        rng, rng_step = jr.split(rng)
        if do_periodic_resample and (sampling_steps + 1) % n_inner == 0:
            rng, rng_re = jr.split(rng)
            state = sampler.resample(rng_re, state)
        state = sampler.step(rng_step, state)
        sampling_steps += 1
        sc = int(state.step_count)
        pbar.update(sc - prev_sc)
        prev_sc = sc
        while k < len(log_iters) and sc >= log_iters[k]:
            target_samples = sampler.get_samples(state)
            steps = steps.at[k].set(target_samples)
            times = times.at[k].set(time() - t_start)
            iters = iters.at[k].set(sc)
            k += 1
        if sc >= n_iter_total:
            target_samples = sampler.get_samples(state)
            steps = steps.at[k].set(target_samples)
            times = times.at[k].set(time() - t_start)
            iters = iters.at[k].set(sc)
            break

        # if (sampling_steps + 1) % 200 == 0:
        #     plot_and_print(sampler, dist, state, state.step_count)

    pbar.close()

    # 2. compute metrics
    metrics = metrics_fn(rng, steps, dist)
    metrics["n_eval"] = state.step_count
    metrics["wallclock"] = times
    metrics["iters"] = iters

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
    if has_comp_freq:
        onp.save(res_dir / "alpha0_components.npy", comp_freq0)

    if use_wandb:   # awkward wandb cannot accept logs from multiple steps at once
        for i in tqdm(range(iters.shape[0]), colour="blue", desc="Upload"):
            mi = {k: v[i] if onp.ndim(v) > 0 else v for (k, v) in metrics.items()}
            wandb.log(mi, step=i)

        # metrics after mixing
        if allow_final_mixing:
            final_metrics = metrics_fn(rng, sampler.get_samples(state)[None], dist)
            for k, v in final_metrics.items():
                wandb.run.summary["eval/"+k+"_am"] = float(v)

        # get auc for hparam tuning
        if "sd_proj" in metrics:
            auc = onp.trapezoid(metrics["sd_proj"], iters)
            norm_auc = auc / (iters[-1] - iters[0])
            wandb.run.summary["eval/sd_norm_auc"] = float(norm_auc)

        if type(dist) == CapsuleGrasp:
            auc = onp.trapezoid(metrics["n_working_fraction_max_diff"], iters)
            norm_auc = auc / (iters[-1] - iters[0])
            wandb.run.summary["eval/working_frac_maxdiff_norm_auc"] = float(norm_auc)

        wandb.finish()


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    resample_mcmc()

    reset_device_memory()
