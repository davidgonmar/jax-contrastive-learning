import flax.linen as nn
from flax.linen import relu, log_softmax
from jaxtyping import Array, Float
import jax.numpy as jnp
import jax
import optax

# ================== Similarity measures ==================
def cosine_similarity(x: Float[Array, "b d"], y: Float[Array, "b d"]) -> Float[Array, "b b"]:
    """Compute cosine similarity between two sets of vectors."""
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
    return jnp.dot(x, y.T)

def euclidean_distance(x: Float[Array, "b d"], y: Float[Array, "b d"]) -> Float[Array, "b"]:
    """Compute euclidean distance between two sets of vectors."""
    return jnp.linalg.norm(x - y, axis=-1)

def nxent_loss(features: Float[Array, "n d"], temperature: float = 0.1, reduction = jnp.mean) -> Float[Array, "b"]:
    """Compute the normalized cross-entropy loss for contrastive learning."""
    # b = 2 * N, where N is the real batch size, B is the augmented batch size (each image in N is augmented twice)
    # they are concatenated so inputs are like [x1, x2, x1', x2']
    b, d = features.shape
    n = b // 2
    
    # matrix of cosine similarities
    similarities: Float[Array, "b b"] = cosine_similarity(features, features)
    
    # assume N = 2 for the visualization below

    # of form
    # |  x1_x1   x1_x2   x1_x1'   x1_x2'  |
    # |  x2_x1   x2_x2   x2_x1'   x2_x2'  |
    # | x1'_x1  x1'_x2  x1'_x1'  x1'_x2'  |
    # | x2'_x1  x2'_x2  x2'_x1'  x2'_x2'  |
    # where a_b means similarity between a and b
    # the diagonals should not count towards the loss

    # mask out the diagonals
    similarities = similarities - jnp.eye(b) * 1e9
    # for each row in similarities, the objective is that a_a' is selected
    N = b // 2
    labels = jnp.concatenate([N + jnp.arange(N), jnp.arange(N)], axis=0)
    # so for the above example, the labels would be
    # [2, 3, 0, 1]
    loss = optax.softmax_cross_entropy_with_integer_labels(similarities / temperature, labels)
    loss2 = jax.nn.log_softmax(similarities / temperature, axis=-1)[jnp.arange(b), labels]
    assert jnp.allclose(loss, -loss2)
    return -reduction(jax.nn.log_softmax(similarities / temperature, axis=-1)[jnp.arange(b), labels])


