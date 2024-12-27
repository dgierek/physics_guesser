from torch import cat
import torch.nn as nn


class EvolutionOperator(nn.Module):
    def __init__(self, latent_dim):
        """
        Initializes the evolution operator.
        Args:
            latent_dim: Dimensionality of the latent space.
        """
        super(EvolutionOperator, self).__init__()
        self.evolution = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),  # Input: Concatenated z_i and z_{i-1}
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, latent_dim)  # Output: Predicted z_{i+1}
        )

    def forward(self, z_i, z_prev):
        """
        Predicts the next latent vector z_{i+1}.
        Args:
            z_i: Current latent vector.
            z_prev: Previous latent vector.
        Returns:
            Predicted next latent vector z_{i+1}.
        """
        input_vector = cat((z_i, z_prev), dim=1)
        z_next = self.evolution(input_vector)
        return z_next


class AutoencoderWithEvolution(nn.Module):
    def __init__(self, input_channels=3, input_size=64, latent_dim=10):
        """
        Initializes the Autoencoder with Evolution Operator.
        Args:
            input_channels: Number of input channels (e.g., 3 for RGB images).
            input_size: Height/width of input image (assumes square images).
            latent_dim: Dimensionality of the latent space.
        """
        super(AutoencoderWithEvolution, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * ((input_size // 16 - 1) ** 2), latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * ((input_size // 16 - 1) ** 2)),
            nn.ReLU(),
            nn.Unflatten(1, (256, input_size // 16 - 1, input_size // 16 - 1)),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # Evolution Operator
        self.evolution_operator = EvolutionOperator(latent_dim)

    def forward(self, x, z_prev=None):
        """
        Forward pass through the autoencoder and evolution operator.
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
            z_prev: Previous latent vector. If None, only performs encoding/decoding.
        Returns:
            Reconstructed input and predicted latent vector z_{i+1}.
        """
        z_i = self.encoder(x)  # Encode input to latent space
        if z_prev is not None:
            z_next = self.evolution_operator(z_i, z_prev)  # Predict next latent vector
        else:
            z_next = None
        reconstructed = self.decoder(z_i)  # Decode latent vector
        return reconstructed, z_i, z_next

