from modeling.head import MLPHead
from modeling.resnet import ResNet50Encoder, ResNet18Encoder
import flax.linen as nn
from flax.linen import relu

class SmallConvEncoder(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = relu(x)
        
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = relu(x)

        x = nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = relu(x)

        x = nn.avg_pool(x, (4, 4))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.out_dim)(x)
        return x


def get_model_for_contrastive_learning(encoder_cls: nn.Module, head_cls: nn.Module, hidden_dim: int, representation_dim: int):
    """Get the model for contrastive learning."""
    encoder = encoder_cls(out_dim=representation_dim)
    head = head_cls(num_classes=hidden_dim)
    return nn.Sequential([encoder, head])