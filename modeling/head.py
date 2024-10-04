import flax.linen as nn
from flax.linen import relu, log_softmax


class MLPHead(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = relu(x)
        x = nn.Dense(512)(x)
        x = relu(x)
        x = nn.Dense(self.num_classes)(x)
        x = log_softmax(x)
        return x