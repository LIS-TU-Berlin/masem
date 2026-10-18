# Manifold Sampling via Entropy Maximization

This repository accompanies the paper "[Manifold Sampling via Entropy Maximization](https://arxiv.org/pdf/2605.12338)" [[`NeurIPS 2026`](https://openreview.net/forum?id=mAVz31YJlN), [`BibTeX`](#references)].

## Repo structure
The code in this repo is for sampling on manifolds that are defined by constraint functions $g(x) \leq 0$ and $h(x) = 0$.

We provide implementations of the following samplers:

| **Acronym**                                          | **Method**                        | **Reference**                                              | ID                    |
|------------------------------------------------------|-----------------------------------|------------------------------------------------------------|-----------------------|
| [NHR](masem/samplers/nhr.py)                         | Non-Linear Hit-and-Run            | [Toussaint et al., 2024](https://arxiv.org/abs/2407.03035) | nhr                   |
| [OLLA](masem/samplers/olla.py)                       | Uncorrected Hamiltonian Annealing | [Jeon et al., 2025](https://arxiv.org/abs/2510.22044)      | olla                  |
| [SCMC](masem/samplers/soft_scmc.py)                  | Time-Reversed Diffusion Sampler   | [Golchi et al., 2014](https://arxiv.org/abs/1410.8209)     | soft_scmc             |
| [MASEM-NHR](masem/samplers/masem_nhr.py)             | Masem-NHR                         | Ours                                                       | masem_nhr / knn_nhr   |
| [MASEM-OLLA](masem/samplers/masem_olla.py)           | Masem-OLLA                        | Ours                                                       | masem_olla / knn_olla |


## How to use

### Configuration
We use [`hydra`](https://hydra.cc/) for config management. The [base configuration](configs/base_conf.yaml) file sets 
parameters that are agnostic to the specific choice of algorithm and target density.

Environment setup:
```bash
# Setup virtual env
uv venv --python 3.11

# Install dependencies
uv pip install -e .
```

### Running experiments
There are 2 types of experiments: experiments for synthetic benchmarks and experiments for robotics benchmarks.
For synthetic benchmarks you must have ground truth samples stored in `masem/distributions/data` for the scripts to run.
We provide this data on [TBD](). 
Alternatively you can regenerate it using the `main()` in the corresponding files.

To run the analysis and compute metircs automatically after sampling, use:

```bash
python masem/scripts/run_analysis.py algorithm=nhr target=seven_lobes seed=123
```

## Conventions -- TODO: check
- $\log p(x) \in \mathbb{R}^{b}$ for $x \in \mathbb{R}^{b \times d}$
- $\nabla \log p(x) \in \mathbb{R}^{b \times d}$ for $x \in \mathbb{R}^{b \times d}$
- $Jac~g(x) \in \mathbb{R}^{b \times n_c \times d}$ where $n_c$ is the cardinality of consrtaint set

## References
If you use parts of this codebase in your research, please cite us using the following BibTeX entries.
```
@inproceedings{
    braun2026manifold,
    title={Manifold Sampling via Entropy Maximization},
    author={Cornelius V. Braun and Tilman Burghoff and Marc Toussaint},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=mAVz31YJlN}
}
```
