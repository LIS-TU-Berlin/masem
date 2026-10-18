from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from matplotlib.axes import Axes
from jax.scipy.spatial.transform import Rotation

from komino.distributions.constrained.halfspace import get_bound_contraints
from komino.distributions.constrained_dist import ConstrainedDistribution
from komino.distributions.constraint import Constraint, ConstraintSet
from komino.distributions.capsule_grasp.capsule_finger import create_capsule_finger_constraints
from komino.distributions.unconstrained.uniform import Uniform


class ForceEquilibrium(Constraint):
    def __init__(self, force_idx: int, offset: float = 0.0):
        self.force_idx = force_idx
        self.offset = offset

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        x = x.reshape(x.shape[0], -1, 6)
        out = jnp.sum(x[:, :, self.force_idx + 3], axis=1) + self.offset
        return out if batched else out[0]

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, x: Array) -> Array:
        if x.ndim == 1:
            x = x[None]
        x = x.reshape(x.shape[0], -1, 6)
        grads = jnp.zeros_like(x)
        grads = grads.at[:, :, self.force_idx + 3].set(1.0)
        return grads.reshape(x.shape[0], -1)


class TorqueEquilibrium(Constraint):
    def __init__(self, torque_idx: int):
        self.torque_idx = torque_idx

    @property
    def is_equality(self) -> bool:
        return True

    @partial(jax.jit, static_argnums=(0,))
    def value(self, x: Array) -> Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        x = x.reshape(x.shape[0], -1, 6)
        torque = jnp.cross(
            x[:, :, 0:3],
            x[:, :, 3:6],
            axis=-1
        )
        out = jnp.sum(torque[:, :, self.torque_idx], axis=1)
        return out if batched else out[0]


class CapsuleGrasp(ConstrainedDistribution):
    def __init__(
        self,
        num_fingers: int = 3,
        length: float = 1.0,
        radius: float = 0.25,
        gravity: float | Array = 1.0,
        orientation: Array = jnp.array([[1.0,1.0, 1.0]]),
        max_force: float = 1.0,
        min_force: float = 0.1,
        cone_coeff: float = 1.0, # 45 deg cone
        bounds: tuple[Array, Array] = (-1.0, 1.0),
        **kwargs
    ):

        if orientation.ndim == 1:
            orientation = orientation[None, :]
        orientation = orientation / jnp.linalg.norm(orientation)

        if isinstance(gravity, (float, int)):
            gravity = jnp.array([0.0, 0.0, -gravity])

        self.length = length
        self.radius = radius
        self.orientation = orientation

        constraints = []
        # finger constraints
        for i in range(num_fingers):
            finger_constraints = create_capsule_finger_constraints(
                index=i,
                length=length,
                radius=radius,
                orientation=orientation,
                max_force=max_force,
                min_force=min_force,
                cone_coeff=cone_coeff,
            )
            constraints.extend(finger_constraints)

        # wrench equilibrium
        for i in range(3):
            constraints.append(ForceEquilibrium(force_idx=i, offset=gravity[i]))
            constraints.append(TorqueEquilibrium(torque_idx=i))

        dim = num_fingers * 6
        cons = ConstraintSet(constraints + get_bound_contraints(bounds, dim))
        base = Uniform(bounds, dim)
        super().__init__(base, cons)

    @property
    def init_scale(self) -> float:
        return (self.bounds[1] - self.bounds[0]) / 4

    def _plot_capsule(self, ax, radius=1.0, length=4.0, center=(0,0,0), direction=(0,0,1),
                    resolution=50, color='blue', alpha=0.3):
        """Plot a capsule in a 3D matplotlib axis.

        Parameters:
            ax          : matplotlib 3D axis
            radius      : radius of capsule
            length      : length of cylindrical part (excluding caps)
            center      : (x,y,z) center of capsule
            direction   : normalized vector for capsule axis
            resolution  : mesh resolution
        """
        # Normalize direction vector
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        # --- Build rotation matrix (align z-axis to direction) ---
        z_axis = np.array([0, 0, 1.0])

        if np.allclose(direction, z_axis):
            R = np.eye(3)
        elif np.allclose(direction, -z_axis):
            R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        else:
            v = np.cross(z_axis, direction)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, direction)

            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])

            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

        # --- Cylinder ---
        z = np.linspace(-length/2, length/2, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        theta, z = np.meshgrid(theta, z)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        cyl = np.stack([x, y, z], axis=-1)

        # --- Hemispheres ---
        phi = np.linspace(0, np.pi/2, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)

        xs = radius * np.sin(phi) * np.cos(theta)
        ys = radius * np.sin(phi) * np.sin(theta)
        zs = radius * np.cos(phi)

        top = np.stack([xs, ys, zs + length/2], axis=-1)
        bottom = np.stack([xs, ys, -zs - length/2], axis=-1)

        # --- Apply rotation and translation ---
        def transform(points):
            pts = points.reshape(-1, 3).T
            pts = R @ pts
            pts = pts.T.reshape(points.shape)
            pts += np.array(center)
            return pts

        cyl = transform(cyl)
        top = transform(top)
        bottom = transform(bottom)

        # --- Plot ---
        ax.plot_surface(cyl[...,0], cyl[...,1], cyl[...,2],
                        color=color, alpha=alpha)
        ax.plot_surface(top[...,0], top[...,1], top[...,2],
                        color=color, alpha=alpha)
        ax.plot_surface(bottom[...,0], bottom[...,1], bottom[...,2],
                        color=color, alpha=alpha)

    def plot2d(
        self,
        samples: Array = None,
        ax: Axes = None,
        markersize: float = 1.0,
        alpha: float = 0.6,
        plt_show: bool = True,
        title: str = None,
        **kwargs
    ) -> Axes:
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')

        self._plot_capsule(ax, radius=self.radius, length=self.length, direction=self.orientation.flatten())

        # Samples
        if samples is not None:
            samples = samples.reshape(samples.shape[0], -1, 6)
            num_fingers = samples.shape[1]
            pts = samples[:, :, 0:3].reshape(-1, 3)
            forces = samples[:, :, 3:6].reshape(-1, 3)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=markersize, alpha=alpha, color='green')
            ax.quiver(
                pts[:, 0], pts[:, 1], pts[:, 2],
                forces[:, 0], forces[:, 1], forces[:, 2],
                length=0.1, color='orange', alpha=alpha/2
            )

            # Antipodal connections for 2-finger grasps
            if num_fingers == 2:
                p0 = samples[:, 0, 0:3]   # (N, 3) first finger contact points
                p1 = samples[:, 1, 0:3]   # (N, 3) second finger contact points
                nan_col = np.full((len(p0), 1), np.nan)
                xs = np.concatenate([p0[:, 0:1], p1[:, 0:1], nan_col], axis=1).ravel()
                ys = np.concatenate([p0[:, 1:2], p1[:, 1:2], nan_col], axis=1).ravel()
                zs = np.concatenate([p0[:, 2:3], p1[:, 2:3], nan_col], axis=1).ravel()
                ax.plot(xs, ys, zs, color='steelblue', alpha=0.4, linewidth=0.4, label='Antipodal pair')

            if num_fingers == 3:
                p0 = samples[:, 0, 0:3]   # (N, 3) first finger contact points
                p1 = samples[:, 1, 0:3]   # (N, 3) second finger contact points
                p2 = samples[:, 2, 0:3]   # (N, 3) third finger contact points
                nan_col = np.full((len(p0), 1), np.nan)
                xs = np.concatenate([p0[:, 0:1], p1[:, 0:1], p2[:, 0:1], nan_col], axis=1).ravel()
                ys = np.concatenate([p0[:, 1:2], p1[:, 1:2], p2[:, 1:2], nan_col], axis=1).ravel()
                zs = np.concatenate([p0[:, 2:3], p1[:, 2:3], p2[:, 2:3], nan_col], axis=1).ravel()
                ax.plot(xs, ys, zs, color='steelblue', alpha=0.4, linewidth=0.4, label='Grasp connections')

        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(-1.5, 1.5)
        fig.legend()
        if title is not None:
            ax.set_title(title)
        
        ax = fig.add_subplot(122)
        # how many fingers are above, in or below the infeasible region?
        if samples is not None:
            samples = samples.reshape(samples.shape[0], -1, 6)
            num_fingers = samples.shape[1]
            axis_pos = jnp.sum(samples[:, :, 0:3] * self.orientation[None], axis=-1) # (N, num_fingers)
            center_axis_vec = samples[:, :, 0:3] - axis_pos[..., None] * self.orientation[None]  # (N, num_fingers, 3)
            normal = center_axis_vec / self.radius
            rotated_normal = Rotation.from_rotvec(jnp.pi/4 * self.orientation.flatten()).apply(normal.reshape(-1, 3)).reshape(normal.shape)
            angle = jnp.arctan2(rotated_normal[..., 1], rotated_normal[..., 0])  # angle in plane perpendicular to capsule axis
            is_infeas = jnp.abs(axis_pos) < 0.125
            ax.scatter(axis_pos, angle, alpha=0.5, s=2, c=is_infeas.reshape(-1).astype(float), cmap='coolwarm')
            ax.set_xlabel("Axis position")
            ax.set_ylabel("Angle to capsule axis")
            ax.set_title("Finger positions relative to infeasible region")
        
            print(f"Fraction of fingers in infeasible region: {jnp.mean(is_infeas):.2%}")
            print(f"Fraction of fingers above infeasible region: {jnp.mean(axis_pos > 0.125):.2%}")
            print(f"Fraction of fingers below infeasible region: {jnp.mean(axis_pos < -0.125):.2%}")

        if plt_show:
            plt.show()

        return ax
