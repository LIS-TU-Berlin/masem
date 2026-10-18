from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from matplotlib.axes import Axes

from masem.distributions.constrained.halfspace import get_bound_contraints
from masem.distributions.constrained_dist import ConstrainedDistribution
from masem.distributions.constraint import ConstraintSet, Constraint
from masem.distributions.distribution import Distribution
from masem.distributions.motion_planning.bspline import BSpline
from masem.distributions.motion_planning.collision_constraint import WaypointToPathConstraint, \
    SphereCollisionConstraint
from masem.distributions.unconstrained.uniform import Uniform


def get_knots(q0: Array, X: Array, invPhi: Array) -> Array:
    q0 = jnp.asarray(q0)
    X = jnp.asarray(X)
    if q0.ndim == 1:
        q0 = q0[None, None, :]

    q0 = jnp.broadcast_to(q0, (X.shape[0], 1, X.shape[-1]))
    full_waypts = jnp.concatenate((q0, X), axis=-2)

    return invPhi @ full_waypts


class SosPathCost(Distribution):
    """Unnormalized log-prob = -cost.

    order=0: path length (sum ||x_{t+1}-x_t||)
    order=1: squared velocity effort  ∫ ||ẋ||^2 dt
    order=2: squared acceleration effort ∫ ||ẍ||^2 dt
    """

    def __init__(
        self,
        n_knots: int,
        task_dim: int,
        q0: Array,
        order: int = 0,
        bounds: tuple = (-10.0, 10.0),
        n_eval: int = 100,
        degree: int = 3,
    ):
        self.dim = int(n_knots * task_dim)
        self.n_knots = n_knots
        self.task_dim = task_dim
        self.bounds = bounds
        self.order = int(order)
        self.q0 = jnp.asarray(q0)

        if self.order not in (0, 1, 2):
            raise ValueError(f"order must be 0, 1, or 2; got {self.order}")

        n_interp = n_knots + 1
        reverse_spline = BSpline(n_in=n_interp, n_out=n_interp, degree=degree, order=0)
        self._invPhi = jnp.linalg.pinv(jnp.asarray(reverse_spline.Phi))

        # Build evaluation matrices for cost
        eval_spline = BSpline(n_in=n_interp, n_out=n_eval, degree=degree, order=2)
        self.Phi_pos = jnp.asarray(eval_spline.Phi)
        self.Phi_vel = jnp.asarray(eval_spline.dPhi)
        self.Phi_acc = jnp.asarray(eval_spline.ddPhi)

        t = jnp.asarray(eval_spline.t_values)
        self.dt = (t[-1] - t[0]) / (t.shape[0] - 1)

        self._get_knots = lambda waypts: get_knots(self.q0, waypts, self._invPhi)

    @partial(jax.jit, static_argnums=(0,))
    def unnorm_log_prob(self, x: Array) -> Array:
        batched = (x.ndim == 2)
        if not batched:
            x = x[None, :]

        waypts = x.reshape(x.shape[0], self.n_knots, self.task_dim)
        ctrl = self._get_knots(waypts)

        if self.order == 0:
            # Path length from position samples
            pos = self.Phi_pos @ ctrl
            dpos = pos[:, 1:, :] - pos[:, :-1, :]
            cost = jnp.mean(dpos ** 2, axis=(-2, -1))

        elif self.order == 1:
            vel = self.Phi_vel @ ctrl
            cost = 5e-2 * jnp.mean(vel ** 2, axis=(-2, -1))  # mean is numerically more stable because we dont scale with n_eval

        else:  # self.order == 2
            acc = self.Phi_acc @ ctrl
            cost = 5e-2 * jnp.mean(acc ** 2, axis=(-2, -1))

        logp = -cost  # high log-prob for low cost
        return logp if batched else logp[0]

    @partial(jax.jit, static_argnums=(0,))
    def score(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        out = jax.grad(lambda xb: jnp.sum(self.unnorm_log_prob(xb)))(x)
        return out if batched else out[0]

    def sample(self, rng: PRNGKey, n_samples: int, load: bool = False) -> Array:
        pass


class DimEqConstraint(Constraint):
    def __init__(self, time_mask: Array, ndof: int, dim_idx: int, target_val: float):
        """
        time_maks: binary time idx at which the constraint should hold!
        dim_idx: dim idx at which constraint should hold
        ndof: number of dofs of the robot
        """
        self.ndof = ndof
        self.dim_idx = dim_idx
        self.target_val = target_val
        self.time_mask = jnp.asarray(time_mask).astype(bool)
        self.nt = time_mask.shape[0]

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        # Note this is slightly different to the non-robotics problems:
        #  here we actually return a vector of shape (self.nt * dn) for each x
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        bsz = x.shape[0]
        x = x.reshape(bsz, self.nt, -1)
        out = x[..., self.dim_idx] - self.target_val
        out = jnp.where(self.time_mask[None, :], out, 0.0)

        return out if batched else out[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None, :]
        bsz = x.shape[0]

        dof_enc = jax.nn.one_hot(self.dim_idx, self.ndof, dtype=x.dtype)
        jac = jnp.eye(self.nt)[:, :, None] * dof_enc[None, None, :]
        jac = jac.reshape(self.nt, self.nt * self.ndof)
        jac = jnp.broadcast_to(jac, (bsz, self.nt, self.nt * self.ndof))
        jac = jnp.where(self.time_mask[None, :, None], jac, 0.0)

        return jac if batched else jac[0]


class MonotonicDimConstraint(Constraint):
    """Enforces monotonic non-decreasing progress along a given dimension."""

    def __init__(self, nt: int, ndof: int, dim_idx: int, q0: Array):
        self.nt = int(nt)
        self.ndof = int(ndof)
        self.dim_idx = int(dim_idx)
        self.q0 = jnp.asarray(q0)

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        bsz = x.shape[0]
        x = x.reshape(bsz, self.nt, self.ndof)

        xs = x[..., self.dim_idx]
        q0_d = jnp.broadcast_to(self.q0[self.dim_idx], (bsz, 1))
        prev = jnp.concatenate([q0_d, xs[:, :-1]], axis=1)
        out = prev - xs

        return out if batched else out[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        bsz = x.shape[0]

        T, D = self.nt, self.ndof
        dof_enc = jax.nn.one_hot(self.dim_idx, D, dtype=x.dtype)

        eye = jnp.eye(T, dtype=x.dtype)
        neg_self = -eye
        pos_prev = jnp.zeros((T, T), dtype=x.dtype).at[1:, :-1].set(
            jnp.eye(T - 1, dtype=x.dtype)
        )
        jac_time = neg_self + pos_prev

        jac = jac_time[:, :, None] * dof_enc[None, None, :]
        jac = jac.reshape(T, T * D)
        jac = jnp.broadcast_to(jac, (bsz, T, T * D))

        return jac if batched else jac[0]


class MaxStepConstraint(Constraint):
    """Limits the gap between consecutive points along the path."""

    def __init__(self, nt: int, ndof: int, max_step: float, q0: Array):
        self.nt = int(nt)
        self.ndof = int(ndof)
        self.max_step = float(max_step)
        self.q0 = jnp.asarray(q0)

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        bsz = x.shape[0]
        x = x.reshape(bsz, self.nt, self.ndof)

        q0_b = jnp.broadcast_to(self.q0[None, None, :], (bsz, 1, self.ndof))
        prev = jnp.concatenate([q0_b, x[:, :-1, :]], axis=1)
        dx = x - prev
        sq = jnp.sum(dx * dx, axis=-1)
        out = sq - self.max_step ** 2

        return out if batched else out[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        # Use autodiff rather than hand-rolling — cheap, and avoids a second
        # source of bugs for what is essentially a quadratic form.
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        jac = jax.vmap(jax.jacrev(lambda xi: self.value(xi[None])[0]))(x)
        return jac if batched else jac[0]


class MaxAccConstraint(Constraint):
    """Limits the change in step between consecutive segments along the path.

    Discrete second difference with q0 prepended:
        a_k = x_{k+1} - 2 x_k + x_{k-1}
    where x_{-1} = q0. Yields nt - 1 inequalities.
    """

    def __init__(self, nt: int, ndof: int, max_acc: float, q0: Array):
        self.nt = int(nt)
        self.ndof = int(ndof)
        self.max_acc = float(max_acc)
        self.q0 = jnp.asarray(q0)

    @property
    def is_equality(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        bsz = x.shape[0]
        x = x.reshape(bsz, self.nt, self.ndof)

        q0_b = jnp.broadcast_to(self.q0[None, None, :], (bsz, 1, self.ndof))
        x_aug = jnp.concatenate([q0_b, x], axis=1)
        acc = x_aug[:, 2:, :] - 2.0 * x_aug[:, 1:-1, :] + x_aug[:, :-2, :]
        sq = jnp.sum(acc * acc, axis=-1)
        out = sq - self.max_acc ** 2

        return out if batched else out[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        jac = jax.vmap(jax.jacrev(lambda xi: self.value(xi[None])[0]))(x)
        return jac if batched else jac[0]


class MotionPlanning(ConstrainedDistribution):
    """Motion Planning constrained objective.

    We optimize waypoints directly by solving for the corresponding spline control points.
    We use SOS objective and guide the optimization via constraints:
        - bound constraints
        - obstacle inequality constraints.
        - final point at goal constraint.
        - any others?
    """

    def __init__(
        self,
        n_knots: int = 3,
        n_eval: int = 40,
        bot_rad: float = 1e-1,
        obs_rad: float = 4e-1,
        n_layers: int = 3,
        bounds: tuple[float, float] = (-4., 4.),
        sos_order: Optional[int] = None,
        max_step: Optional[float] = None,
        max_acc: Optional[float] = None,
    ):
        self._task_dim = 2
        self.dim = int(n_knots * self._task_dim)
        self.bot_rad = bot_rad
        self.obs_rad = obs_rad
        self.n_eval = n_eval
        self.n_layers = n_layers
        self.n_knots = n_knots
        self.max_step = max_step
        self.max_acc = max_acc
        self.bounds = bounds
        self.q0 = jnp.array([0.95 * self.bounds[0], 0.])

        if sos_order is not None:
            base = SosPathCost(n_knots, self._task_dim, q0=self.q0, bounds=self.bounds, order=sos_order)
            invPhi_for_cons = base._invPhi
            Phi_pos_for_cons = base.Phi_pos
        else:
            base = Uniform(bounds=bounds, dim=self.dim)
            n_interp = n_knots + 1
            reverse = BSpline(n_in=n_interp, n_out=n_interp, degree=3, order=0)
            invPhi_for_cons = jnp.linalg.pinv(jnp.asarray(reverse.Phi))
            eval_for_cons = BSpline(n_in=n_interp, n_out=n_eval, degree=3, order=0)
            Phi_pos_for_cons = jnp.asarray(eval_for_cons.Phi)

        eqs, ineqs = self._create_obstacles(invPhi_for_cons, Phi_pos_for_cons)
        ineqs.append(  # Monotonic x-progress along the path.
            WaypointToPathConstraint(
                inner=MonotonicDimConstraint(nt=self.n_eval, ndof=self._task_dim, dim_idx=0, q0=self.q0),
                q0=self.q0,
                invPhi=invPhi_for_cons,
                Phi_path=Phi_pos_for_cons,
                ndof=self._task_dim,
            )
        )
        if self.max_step is not None:
            ineqs.append(  # velocity constraint to reduce gaps
                WaypointToPathConstraint(
                    inner=MaxStepConstraint(
                        nt=self.n_eval, ndof=self._task_dim, max_step=self.max_step, q0=self.q0
                    ),
                    q0=self.q0,
                    invPhi=invPhi_for_cons,
                    Phi_path=Phi_pos_for_cons,
                    ndof=self._task_dim,
                )
            )
        if self.max_acc is not None:
            ineqs.append(  # acceleration constraint to suppress kinks
                WaypointToPathConstraint(
                    inner=MaxAccConstraint(
                        nt=self.n_eval, ndof=self._task_dim, max_acc=self.max_acc, q0=self.q0
                    ),
                    q0=self.q0,
                    invPhi=invPhi_for_cons,
                    Phi_path=Phi_pos_for_cons,
                    ndof=self._task_dim,
                )
            )
        ineqs.extend(get_bound_contraints(bounds, self.dim))

        cons = ConstraintSet([*ineqs, *eqs])
        super().__init__(base, cons)

        reverse_spline = BSpline(n_in=n_knots + 1, n_out=n_knots + 1)  # +1 knot because of q0
        eval_spline = BSpline(n_in=n_knots + 1, n_out=n_eval)
        self.Phi = jnp.asarray(eval_spline.Phi)
        invPhi = jnp.linalg.pinv(reverse_spline.Phi)
        self.get_knots = lambda waypts: get_knots(self.q0, waypts, invPhi)

        ev = self.eval(jnp.zeros(self.dim))
        self._neq = ev.h.shape[-1]
        self._nineq = ev.g.shape[-1]

    @property
    def init_scale(self) -> float:
        return 1.

    @property
    def neq(self) -> int:
        return self._neq

    @property
    def nineq(self) -> int:
        return self._nineq

    def _create_obstacles(self, invPhi_for_cons, Phi_pos_for_cons):
        x_start = 0.5 * self.bounds[0]
        level_dist = (self.bounds[1] - self.bounds[0]) / (self.n_layers + 1)
        self.obs_locs = []

        final_waypt_mask = jnp.zeros(self.n_knots)
        final_waypt_mask = final_waypt_mask.at[-1].set(1.)
        eqs = [
            DimEqConstraint(final_waypt_mask, ndof=2, dim_idx=0, target_val=self.bounds[1] * 0.9),
        ]

        ineqs = []
        for i in range(self.n_layers):
            for y in range(i + 1):
                circle_loc = jnp.array([x_start + i * level_dist, 4 * (2 * y - i) * self.obs_rad])
                ineqs.append(
                    WaypointToPathConstraint(
                        inner=SphereCollisionConstraint(circle_loc, self.obs_rad, ndof=2),
                        q0=self.q0,
                        invPhi=invPhi_for_cons,
                        Phi_path=Phi_pos_for_cons,  # evaluate collisions along path samples
                        ndof=2,
                    )
                )
                self.obs_locs.append(circle_loc)

        return eqs, ineqs

    def path_homotopy_class(self, x: Array) -> int:
        sig = self.path_signature(x)
        _, inverse = jnp.unique(sig, return_inverse=True, axis=0)
        return inverse

    @partial(jax.jit, static_argnums=(0,))
    def path_signature(self, x: Array) -> Array:
        """for each trajectory and obstacle compute whether it passes above or below"""
        waypts = x.reshape(x.shape[0], -1, 2)
        knots = self.get_knots(waypts)
        paths = self.Phi @ knots
        obstacles = jnp.array(self.obs_locs)
        passing = jnp.argmin(jnp.abs(paths[:, :, None, 0] - obstacles[None, None, :, 0]), axis=1)  # (n_traj, n_obs)

        def signature_fn(x, p, o):
            # x: (n_eval, 2), p: (n_obs,), o: (n_obs, 2)
            # output: (n_obs,)
            obs_y = o[:, 1]
            path_y = x[p, 1]
            return path_y > obs_y

        signature = jax.vmap(signature_fn, in_axes=(0, 0, None))(paths, passing, obstacles)
        return signature

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
        markersize: float = 1.0,
        alpha: float = 0.6,
        cmap_alpha: float = 0.4,
        plt_show: bool = True,
        title: str = None,
        filled: bool = True,
        cmap: str = "Blues",
        color_homotopy_classes: bool = False,
        color_feasibility: bool = True,
        draw_obstacles=True
    ) -> Axes:
        import numpy as onp
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle

        if ax is None:
            _, ax = plt.subplots(1)

        if bounds is None:
            bounds = self.bounds
        if bounds_2 is None:
            bounds_2 = bounds

        # plot obstacles
        if draw_obstacles and getattr(self, "obs_locs", None) is not None and len(self.obs_locs) > 0:
            obs = onp.asarray(jnp.stack(self.obs_locs) if isinstance(self.obs_locs, list) else self.obs_locs)
            for c in obs:
                ax.add_patch(
                    Circle(
                        (float(c[0]), float(c[1])),
                        radius=float(self.obs_rad),
                        fill=True,
                        alpha=0.25,
                        linewidth=1.0,
                        zorder=1,
                        color="black"
                    )
                )

        # TODO: standardize this better; ie, plot goal based on constraints automatically
        if getattr(self, "q0", None) is not None:
            ax.scatter([float(self.q0[0])], [float(self.q0[1])], s=30.0, zorder=3, marker="o")
        ax.axvline(  # goal
            self.bounds[1] * 0.9, color="black", linewidth=2.
        )

        if samples is not None:
            if samples.ndim == 1:
                samples = samples[None, :]

            # logp = self.unnorm_log_prob(samples)
            # bin_logp = jnp.where(logp < logp.min(), 0., 1.)
            def _get_paths(samples):
                print(samples.shape)
                waypts = samples.reshape(samples.shape[0], -1, 2)
                knots = self.get_knots(waypts)
                paths = self.Phi @ knots
                paths = onp.array(paths)
                paths[..., 0] = onp.clip(paths[..., 0], bounds[0], bounds[1])
                paths[..., 1] = onp.clip(paths[..., 1], bounds_2[0], bounds_2[1])
                return paths, waypts
            # scatter collision eval points

            if color_homotopy_classes and len(self.obs_locs) > 0:
                signatures = self.path_signature(samples)
                class_id = signatures @ (2 ** jnp.arange(signatures.shape[-1]))
                n_classes = 2 ** len(self.obs_locs)
                cmap = plt.get_cmap("Set1" if n_classes <= 9 else "tab20" if n_classes <= 20 else "hsv")
                c = cmap(class_id / n_classes)
            if color_feasibility:
                c = "#1f77b3"
                feas_mask = self.feasible(samples)
                if feas_mask.sum() > 0:
                    feas_samples = samples[feas_mask]
                    print(len(feas_samples))
                    paths, waypts = _get_paths(feas_samples)
                    lc = LineCollection(paths, linewidths=1.0, alpha=alpha, zorder=3, facecolors='none', edgecolors=c, rasterized=True)
                    ax.add_collection(lc)

                c = "#7A1311"
                if feas_mask.sum() < samples.shape[0]:
                    infeas_samples = samples[~feas_mask]
                    paths, waypts = _get_paths(infeas_samples)
                    lc = LineCollection(paths, linewidths=1.0, alpha=alpha * 0.5, zorder=2, facecolors='none', edgecolors='red', rasterized=True)
                    ax.add_collection(lc)
            # lc.set_array(bin_logp)
            # fig = ax.get_figure()
            # fig.colorbar(lc, ax=ax)
            else:
                paths, waypts = _get_paths(samples)
                lc = LineCollection(paths, linewidths=1.0, alpha=alpha, zorder=3, facecolors='none', edgecolors=c, rasterized=True)
                ax.add_collection(lc)
                for t in range(waypts.shape[1]):
                    ax.scatter(
                        waypts[:, t, 0],
                        waypts[:, t, 1],
                        s=markersize,
                        alpha=alpha,
                        zorder=3,
                        c="salmon",
                        rasterized=True,
                    )

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds_2[0], bounds_2[1])
        ax.set_aspect("equal", adjustable="box")

        if title is not None:
            ax.set_title(title)

        if plt_show:
            plt.show()

        return ax


class RandomObstacleMotionPlanning(MotionPlanning):
    """Motion Planning with randomly placed obstacles of varying radii."""

    def __init__(
        self,
        n_knots: int = 3,
        n_eval: int = 40,
        bot_rad: float = 1e-1,
        n_obstacles: int = 20,
        obs_rad_range: tuple[float, float] = (2e-1, 5e-1),
        bounds: tuple[float, float] = (-4., 4.),
        sos_order: Optional[int] = None,
        seed: int = 0,
        max_tries_per_obs: int = 200,
        clearance: float = 0.2,
        max_step: Optional[float] = 1.0,  # TODO: test 2.5
        max_acc: Optional[float] = 0.65,
    ):
        self._seed = seed
        self._n_obstacles = n_obstacles
        self._obs_rad_range = obs_rad_range
        self._clearance = clearance
        self._max_tries_per_obs = max_tries_per_obs
        self._goal = jnp.array([0.9 * bounds[1], 0.])

        super().__init__(
            n_knots, n_eval, bot_rad,
            obs_rad=obs_rad_range[0], n_layers=0,
            bounds=bounds, sos_order=sos_order, max_step=max_step, max_acc=max_acc
        )

    def _create_obstacles(self, invPhi_for_cons, Phi_pos_for_cons):
        obs_locs, obs_radii = self._sample_obstacles(
            rng=jax.random.PRNGKey(self._seed),
            n_obstacles=self._n_obstacles,
            obs_rad_range=self._obs_rad_range,
            bounds=self.bounds,
            q0=self.q0,
            goal=self._goal,
            bot_rad=self.bot_rad,
            clearance=self._clearance,
            max_tries_per_obs=self._max_tries_per_obs,
        )
        # Store per-obstacle radius
        self.obs_locs = [jnp.asarray(c) for c in obs_locs]
        self.obs_radii = jnp.asarray(obs_radii)
        self.obs_rad = float(jnp.mean(self.obs_radii))

        # Final waypoint equality at goal x.
        final_waypt_mask = jnp.zeros(self.n_knots).at[-1].set(1.)
        eqs = [
            DimEqConstraint(final_waypt_mask, ndof=2, dim_idx=0, target_val=self.bounds[1] * 0.9),
        ]

        # Collision constraints
        ineqs = []
        for c, r in zip(self.obs_locs, obs_radii):
            ineqs.append(
                WaypointToPathConstraint(
                    inner=SphereCollisionConstraint(c, float(r) + self.bot_rad, ndof=2),
                    q0=self.q0,
                    invPhi=invPhi_for_cons,
                    Phi_path=Phi_pos_for_cons,
                    ndof=2,
                )
            )

        return eqs, ineqs

    @staticmethod
    def _sample_obstacles(
        rng: PRNGKey,
        n_obstacles: int,
        obs_rad_range: tuple[float, float],
        bounds: tuple[float, float],
        q0: Array,
        goal: Array,
        bot_rad: float,
        clearance: float,
        max_tries_per_obs: int,
    ) -> tuple[list[Array], list[float]]:
        import numpy as onp

        # Sample in a corridor that excludes a margin around start/goal.
        lo_x, hi_x = float(bounds[0]) + 0.25, float(bounds[1]) - 0.25
        lo_y, hi_y = float(bounds[0]) + 0.25, float(bounds[1]) - 0.25
        rmin, rmax = float(obs_rad_range[0]), float(obs_rad_range[1])

        q0_np = onp.asarray(q0)
        goal_np = onp.asarray(goal)

        locs: list[onp.ndarray] = []
        radii: list[float] = []

        rng_np = onp.random.default_rng(int(jax.random.randint(rng, (), 0, 2 ** 31 - 1)))

        for _ in range(n_obstacles):
            placed = False
            for _ in range(max_tries_per_obs):
                r = rng_np.uniform(rmin, rmax)
                cx = rng_np.uniform(lo_x, hi_x)
                cy = rng_np.uniform(lo_y, hi_y)
                c = onp.array([cx, cy])

                # Keep clear of start and goal.
                if onp.linalg.norm(c - q0_np) < r + bot_rad + clearance + 0.1:
                    continue
                if onp.linalg.norm(c - goal_np) < r + bot_rad + clearance + 0.1:
                    continue

                # Avoid overlap with previously placed obstacles.
                ok = True
                for c_prev, r_prev in zip(locs, radii):
                    if onp.linalg.norm(c - c_prev) < r + r_prev + clearance:
                        ok = False
                        break
                if not ok:
                    continue

                locs.append(c)
                radii.append(float(r))
                placed = True
                break

            if not placed:
                continue

        return [jnp.asarray(c) for c in locs], radii

    def plot2d(self, *args, **kwargs) -> Axes:
        # Parent uses a single self.obs_rad for all circles; override to draw per-obstacle radii.
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        ax = kwargs.get("ax", None)
        if ax is None:
            _, ax = plt.subplots(1)
            kwargs["ax"] = ax

        # Draw obstacles with their own radii first, then let the parent draw the rest.
        for c, r in zip(self.obs_locs, [float(x) for x in self.obs_radii]):
            ax.add_patch(
                Circle(
                    (float(c[0]), float(c[1])),
                    radius=r,
                    fill=True,
                    alpha=0.25,
                    linewidth=1.0,
                    zorder=1,
                    color="black",
                )
            )

        # Temporarily hide obs_locs so parent doesn't redraw them at the wrong radius.
        out = super().plot2d(*args, draw_obstacles=False, **kwargs)
        return out


class GridObstacleMotionPlanning(MotionPlanning):
    """Motion Planning with obstacles placed on a regular NxM grid."""

    def __init__(
        self,
        n_knots: int = 3,
        n_eval: int = 40,
        bot_rad: float = 1e-1,
        obs_rad: float = 5e-1,
        grid_shape: tuple[int, int] = (4, 4),
        bounds: tuple[float, float] = (-4., 4.),
        sos_order: Optional[int] = None,
        inset: float = 1.2, #1.22,
        max_step: Optional[float] = 1.0,
        max_acc: Optional[float] = 0.65,
    ):
        self._grid_shape = grid_shape
        self._inset = inset
        super().__init__(
            n_knots, n_eval, bot_rad,
            obs_rad=obs_rad, n_layers=0,
            bounds=bounds, sos_order=sos_order, max_step=max_step, max_acc=max_acc
        )

    def _create_obstacles(self, invPhi_for_cons, Phi_pos_for_cons):
        final_waypt_mask = jnp.zeros(self.n_knots).at[-1].set(1.)
        eqs = [
            DimEqConstraint(final_waypt_mask, ndof=2, dim_idx=0, target_val=self.bounds[1] * 0.9),
        ]

        # Build grid of obstacle centers.
        ineqs = []
        nx, ny = int(self._grid_shape[0]), int(self._grid_shape[1])
        lo, hi = float(self.bounds[0]) + self._inset, float(self.bounds[1]) - self._inset
        xs = jnp.linspace(lo, hi, nx) if nx > 1 else jnp.array([0.5 * (lo + hi)])
        ys = jnp.linspace(lo, hi, ny) if ny > 1 else jnp.array([0.5 * (lo + hi)])
        self.obs_locs = [jnp.array([float(x), float(y)]) for x in xs for y in ys]

        for c in self.obs_locs:
            ineqs.append(
                WaypointToPathConstraint(
                    inner=SphereCollisionConstraint(c, self.obs_rad, ndof=2),
                    q0=self.q0,
                    invPhi=invPhi_for_cons,
                    Phi_path=Phi_pos_for_cons,
                    ndof=2,
                )
            )

        return eqs, ineqs


if __name__ == "__main__":
    mp = RandomObstacleMotionPlanning()
    print(mp.neq, mp.nineq, mp.dim)
