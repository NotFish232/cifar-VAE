import torch as T
from torch import nn
from typing_extensions import Self


LATENT_DIM = 256


class Encoder(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 128, 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1024),
            nn.LeakyReLU(),
            nn.Dropout(),
        )

        self.mean_layer = nn.Linear(1024, LATENT_DIM)
        self.logvar_layer = nn.Linear(1024, LATENT_DIM)

    def forward(self: Self, x: T.Tensor) -> tuple[T.Tensor, tuple[T.Tensor, T.Tensor]]:
        x = self.convolutional_layers(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.fully_connected_layers(x)

        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = mean + T.exp(0.5 * logvar) * T.randn_like(logvar)

        return z, (mean, logvar)


class Decoder(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 1024 * 3 * 3), nn.LeakyReLU(), nn.Dropout()
        )

        self.deconvolutional_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 128, 4, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 3, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = self.fully_connected_layers(x)
        x = x.view(-1, 1024, 3, 3)
        x = self.deconvolutional_layers(x)

        return x


class VAE(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        z, (mean, logvar) = self.encoder(x)
        out = self.decoder(z)
        return out, (mean, logvar)


if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "gpu")

    v = VAE().to(device)

    print(f"Parameters: {sum(i.numel() for i in v.parameters()):,}")

    inp = T.randn(32 * 32 * 3, device=device).reshape(1, 3, 32, 32)


    print(v(inp))
