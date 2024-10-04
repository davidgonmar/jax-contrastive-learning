import jax
import jax.numpy as jnp
from jax import grad, jit
import tensorflow as tf
import tensorflow_datasets as tfds
import optax
from modeling import SmallConvEncoder, MLPHead, get_model_for_contrastive_learning
from losses import nxent_loss
import numpy as np


model = get_model_for_contrastive_learning(SmallConvEncoder, MLPHead, hidden_dim=128, representation_dim=128)
rng = jax.random.PRNGKey(0)
optimizer = optax.adam(0.001)

def random_apply(func, x, p):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x
    )

def color_distortion(image, s=1.0):
    def color_jitter(x):
        x = tf.image.random_brightness(x, max_delta=0.8 * s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_hue(x, max_delta=0.2 * s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x

    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image

def preprocess(image, _):
    image1 = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image1 = tf.image.random_flip_left_right(image1)
    image1 = tf.image.resize(image1, [224, 224])
    image1 = color_distortion(image1)

    image2 = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image2 = tf.image.random_flip_left_right(image2)
    image2 = tf.image.resize(image2, [224, 224])
    image2 = color_distortion(image2)

    return image1, image2

def load_dataset(batch_size):
    dataset = tfds.load('cifar100', split='train', as_supervised=True).shuffle(1000)
    dataset = dataset.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



def forward_pass(params, optim_state, batch_stats, images):
    def compute_loss(params, batch_stats, images):
        outputs, updated_stats = model.apply({"params": params, "batch_stats": batch_stats}, images, mutable=["batch_stats"])
        loss = nxent_loss(outputs)
        return loss, updated_stats

    (loss, updated_stats), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, batch_stats, images)
    updates, optim_state = optimizer.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return loss, new_params, updated_stats["batch_stats"], optim_state 

def train(batch_size=256, num_epochs=10):
    init = model.init(rng, jnp.ones((1, 224, 224, 3)))
    
    optim_state = optimizer.init(init["params"])

    dataset = load_dataset(batch_size)

    params = init["params"]
    batch_stats = init["batch_stats"]

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for images1, images2 in dataset:
            images1, images2 = images1.numpy(), images2.numpy()
            images = jnp.concatenate([jax.device_put(images1), jax.device_put(images2)], axis=0)
            loss, params, batch_stats, optim_state = forward_pass(params, optim_state, batch_stats, images)
            
            epoch_loss += loss
            num_batches += 1

            print(f"Epoch {epoch + 1}, Loss: {loss}")

if __name__ == "__main__":
    train()
