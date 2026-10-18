from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from chex import Array, PRNGKey
from matplotlib.axes import Axes

from komino.types import Eval


class Distribution(ABC):
    dim: int
    bounds: tuple = (-10.0, 10.0)
    log_prob_min: float = -10_000.  # for numerical stability
    eps: float = 1e-8
    domain_tol: float | None = 1e-5
    log_Z: float | None = None

    @abstractmethod
    def unnorm_log_prob(self, x: Array) -> Array:
        pass

    @abstractmethod
    def score(self, x: Array) -> Array:
        pass

    @abstractmethod
    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        pass

    def eval(self, x: Array) -> Eval:
        return Eval(logp=self.unnorm_log_prob(x), score=self.score(x))

    def plot2d(
        self,
        samples: Array = None,
        ax: Axes = None,
        bounds: Optional[tuple[float | int, float | int]] = None,
        bounds_2: Optional[tuple[float | int, float | int]] = None,
        grid_width_n_points: int = 20,
        grid_width_n_points_2: int = None,
        n_contour_levels: int = None,
        plot_marginal_dims: tuple[int] = (0, 1),
        markersize: float = 2.0,
        alpha: float = 0.6,
        cmap_alpha: float = 0.4,
        plt_show: bool = True,
        title: str = None,
        filled: bool = True,
        cmap: str = "Blues",
    ) -> Axes:
        from itertools import product

        if ax is None:
            fig, ax = plt.subplots(1)

        if bounds is None:
            bounds = self.bounds
        if bounds_2 is None:
            bounds_2 = bounds
        if grid_width_n_points_2 is None:
            grid_width_n_points_2 = grid_width_n_points

        # Eval log prob
        if self.dim == 2:
            x_points_dim1 = jnp.linspace(bounds[0], bounds[1], grid_width_n_points)
            x_points_dim2 = jnp.linspace(bounds_2[0], bounds_2[1], grid_width_n_points_2)
            x_points = jnp.asarray(list(product(x_points_dim1, x_points_dim2)))
            log_p_x = jax.vmap(self.unnorm_log_prob)(x_points)
            log_p_x = jnp.clip(log_p_x, min=-40.0, max=40.0)
            log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points_2))

            # convert to np for plotting
            x_points_dim1 = x_points[:, 0].reshape(
                (grid_width_n_points, grid_width_n_points_2)
            )
            x_points_dim2 = x_points[:, 1].reshape(
                (grid_width_n_points, grid_width_n_points_2)
            )
            x_points_dim1 = onp.array(x_points_dim1)
            x_points_dim2 = onp.array(x_points_dim2)
            log_p_x = onp.array(log_p_x)

            pmax = onp.nanmax(log_p_x)
            pmin = onp.nanmin(log_p_x)
            plt_fc = ax.contourf if filled else ax.contour
            if n_contour_levels is not None:
                plt_fc(
                    x_points_dim1,
                    x_points_dim2,
                    log_p_x,
                    vmin=pmin,
                    vmax=pmax,
                    levels=onp.linspace(pmin, pmax, n_contour_levels),
                    cmap=cmap,
                    alpha=cmap_alpha
                )
            else:
                plt_fc(
                    x_points_dim1,
                    x_points_dim2,
                    log_p_x,
                    vmin=pmin,
                    vmax=pmax,
                    cmap=cmap,
                    zorder=0,
                    alpha=cmap_alpha
                )
        elif samples is None:
            return ax

        if title is not None:
            ax.set_title(title)

        if samples is not None:
            ax.scatter(
                # samples[:, plot_marginal_dims[0]],
                # samples[:, plot_marginal_dims[1]],
                onp.clip(
                    samples[:, plot_marginal_dims[0]], bounds[0], bounds[1]
                ),
                onp.clip(
                    samples[:, plot_marginal_dims[1]], bounds_2[0], bounds_2[1]
                ),
                s=markersize,
                alpha=alpha,
                zorder=2,
                c="#e35867", edgecolors="#de4c5a"
            )
            ax.axis("equal")

        if plt_show:
            plt.show()

        return ax
