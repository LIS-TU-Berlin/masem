# Constrained Sampling

Contains code for sampling on manifolds that are defined by constraint functions $g(x) \leq 0$ and $h(x) = 0$.

## Repo structure
TBD.

## How to use
TBD.

```bash
uv venv --python 3.11
```

## Conventions -- TODO: check
- $\log p(x) \in \mathbb{R}^{b}$ for $x \in \mathbb{R}^{b \times d}$
- $\nabla \log p(x) \in \mathbb{R}^{b \times d}$ for $x \in \mathbb{R}^{b \times d}$
- $Jac~g(x) \in \mathbb{R}^{b \times n_c \times d}$ where $n_c$ is the cardinality of consrtaint set

