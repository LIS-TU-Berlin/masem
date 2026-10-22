import jax
import jax.numpy as jnp


def knn_log_radius_topk(
    x: jax.Array,
    k: int,
    eps: float = 1e-12,
    use_kmean: bool = True
) -> jax.Array:
    """x: (N, d)
    Returns (N,) array where entry i is (1/k) * sum_{j=1..k} log r_j(x_i),
    with r_j the distance to the j-th nearest neighbor of x_i.

    This is the avg of k Kozachenko-Leonenko entropy estimators for lower var.
    """
    N = x.shape[0]
    diffs = x[:, None, :] - x[None, :, :]
    d2 = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))
    masked = jnp.where(jnp.eye(N, dtype=bool), -jnp.inf, -d2)
    vals, _ = jax.lax.top_k(masked, k)  # k largest of (-d) = k smallest distances

    if use_kmean:
        log_dists = jnp.log(jnp.maximum(-vals, eps))
        ent_est = jnp.mean(log_dists, axis=-1)
    else:
        ent_est = jnp.log(jnp.maximum(-vals[:, -1], eps))

    return ent_est
