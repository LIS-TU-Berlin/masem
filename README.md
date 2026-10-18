# Constrained Sampling

This repository accompanies the paper "[Underdamped Diffusion Bridges with Applications to Sampling](https://openreview.net/forum?id=mAVz31YJlN)" [[`NeurIPS 2026`](https://openreview.net/forum?id=mAVz31YJlN), [`BibTeX`](#references)].

Contains code for sampling on manifolds that are defined by constraint functions $g(x) \leq 0$ and $h(x) = 0$.

## Repo structure
TBD.

## How to use
TBD.

```bash
# Setup virtual env
uv venv --python 3.11

# Install dependencies
uv pip install -e .
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
