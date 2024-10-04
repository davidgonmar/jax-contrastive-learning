import flax.linen as nn
from flax.linen import relu, log_softmax

class ConvBlock(nn.Module):
    kernel_size: int
    filters: list
    strides: tuple = (2, 2)

    @nn.compact
    def __call__(self, x):
        filters1, filters2, filters3 = self.filters
        
        main = nn.Sequential([
            nn.Conv(filters1, (1, 1), self.strides), 
            nn.BatchNorm(use_running_average=False), 
            relu,
            nn.Conv(filters2, (self.kernel_size, self.kernel_size), padding='SAME'), 
            nn.BatchNorm(use_running_average=False), 
            relu,
            nn.Conv(filters3, (1, 1)), 
            nn.BatchNorm(use_running_average=False)
        ])

        shortcut = nn.Sequential([
            nn.Conv(filters3, (1, 1), self.strides), 
            nn.BatchNorm(use_running_average=False)
        ])
        
        return relu(main(x) + shortcut(x))


class IdentityBlock(nn.Module):
    kernel_size: int
    filters: list

    @nn.compact
    def __call__(self, x):
        filters1, filters2 = self.filters
        
        main = nn.Sequential([
            nn.Conv(filters1, (1, 1)), 
            nn.BatchNorm(use_running_average=False), 
            relu,
            nn.Conv(filters2, (self.kernel_size, self.kernel_size), padding='SAME'), 
            nn.BatchNorm(use_running_average=False), 
            relu,
            nn.Conv(x.shape[-1], (1, 1)),
            nn.BatchNorm(use_running_average=False)
        ])
        
        return relu(main(x) + x)


class ResNet50Encoder(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), (2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        x = ConvBlock(3, [64, 64, 256], strides=(1, 1))(x)
        x = IdentityBlock(3, [64, 64])(x)
        x = IdentityBlock(3, [64, 64])(x)
        
        x = ConvBlock(3, [128, 128, 512])(x)
        x = IdentityBlock(3, [128, 128])(x)
        x = IdentityBlock(3, [128, 128])(x)
        x = IdentityBlock(3, [128, 128])(x)

        x = ConvBlock(3, [256, 256, 1024])(x)
        x = IdentityBlock(3, [256, 256])(x)
        x = IdentityBlock(3, [256, 256])(x)
        x = IdentityBlock(3, [256, 256])(x)
        x = IdentityBlock(3, [256, 256])(x)
        x = IdentityBlock(3, [256, 256])(x)

        x = ConvBlock(3, [512, 512, 2048])(x)
        x = IdentityBlock(3, [512, 512])(x)
        x = IdentityBlock(3, [512, 512])(x)

        x = nn.avg_pool(x, (7, 7))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.out_dim)(x)
        return x


class ConvBlock2(nn.Module):
    kernel_size: int
    filters: list
    strides: tuple = (2, 2)

    @nn.compact
    def __call__(self, x):
        filters1, filters2 = self.filters
        
        main = nn.Sequential([
            nn.Conv(filters1, (self.kernel_size, self.kernel_size), self.strides, padding='SAME'), 
            nn.BatchNorm(use_running_average=False), 
            relu,
            nn.Conv(filters2, (self.kernel_size, self.kernel_size), padding='SAME'), 
            nn.BatchNorm(use_running_average=False)
        ])

        shortcut = nn.Conv(filters2, (1, 1), self.strides)(x)
        shortcut = nn.BatchNorm(use_running_average=False)(shortcut)
        
        return relu(main(x) + shortcut)


class IdentityBlock2(nn.Module):
    kernel_size: int
    filters: list

    @nn.compact
    def __call__(self, x):
        filters1, filters2 = self.filters
        
        main = nn.Sequential([
            nn.Conv(filters1, (self.kernel_size, self.kernel_size), padding='SAME'), 
            nn.BatchNorm(use_running_average=False), 
            relu,
            nn.Conv(filters2, (self.kernel_size, self.kernel_size), padding='SAME'), 
            nn.BatchNorm(use_running_average=False)
        ])
        
        return relu(main(x) + x)


class ResNet18Encoder(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x):
        # Initial layers
        x = nn.Conv(64, (7, 7), (2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        # Residual blocks
        x = ConvBlock2(3, [64, 64], strides=(1, 1))(x)
        x = IdentityBlock2(3, [64, 64])(x)
        
        x = ConvBlock2(3, [128, 128])(x)
        x = IdentityBlock2(3, [128, 128])(x)

        x = ConvBlock2(3, [256, 256])(x)
        x = IdentityBlock2(3, [256, 256])(x)

        x = ConvBlock2(3, [512, 512])(x)
        x = IdentityBlock2(3, [512, 512])(x)

        # Final pooling and dense layer
        x = nn.avg_pool(x, (7, 7))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.out_dim)(x)
        return x
