import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, bottleneck_dim):
        super(VAE, self).__init__()

        self.bottleneck_dim = bottleneck_dim
        self.encoder = None
        self.decoder = None

    def forward(self, inputs):
        mu, log_sigma = self.encode(inputs)
        latent_code = self.bottleneck(mu, log_sigma)
        outputs = self.decode(latent_code)

        return outputs

    def encode(self, inputs):
        latent_parameters = self.encoder(inputs)
        mu, log_sigma = torch.split(latent_parameters, self.bottleneck_dim, dim=1)

        return mu, log_sigma

    def bottleneck(self, mu, log_sigma):
        noise = torch.randn_like(mu)
        latent_code = log_sigma.exp() * noise + mu

        return latent_code

    def decode(self, latent_code):
        return self.decoder(latent_code)


class CNNVAE(VAE):
    def __init__(self, input_shape, bottleneck_dim):
        super(CNNVAE, self).__init__(bottleneck_dim)

        in_channels = input_shape[0]
        hw = input_shape[1]
        hw_before_linear = hw // 4
        flat_dim = 64 * hw_before_linear ** 2

        self.encoder = nn.Sequential(nn.Conv2d(in_channels, out_channels=16, kernel_size=5, padding=2),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     nn.ReLU(True),
                                     nn.Flatten(),
                                     nn.Linear(flat_dim, 2*bottleneck_dim))

        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, flat_dim),
                                     nn.ReLU(True),
                                     Unflatten((64, hw_before_linear, hw_before_linear)),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                                                        padding=1, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                                                        padding=1, output_padding=1),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=16, out_channels=in_channels, kernel_size=5, padding=2),
                                     nn.Tanh())


class MLPVAE(VAE):
    def __init__(self, input_shape, bottleneck_dim):
        super(MLPVAE, self).__init__(bottleneck_dim)

        in_channels = input_shape[0]
        hw = input_shape[1]
        flat_dim = in_channels * (hw ** 2)

        self.encoder = nn.Sequential(nn.Flatten(),
                                     nn.Linear(flat_dim, out_features=512),
                                     nn.ReLU(True),
                                     nn.Linear(512, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 2*bottleneck_dim))

        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 512),
                                     nn.ReLU(True),
                                     nn.Linear(512, flat_dim),
                                     Unflatten(input_shape),
                                     nn.Tanh())


class Unflatten(nn.Module):
    def __init__(self, shape):
        """
        Reshapes a batch of flat tensors to the given shape.

        :param shape: expected output shape without batch dimension
        """
        super(Unflatten, self).__init__()

        self.shape = shape

    def forward(self, inputs):
        return torch.reshape(inputs, (-1,) + self.shape)
