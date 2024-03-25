import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Adjusted initialization for generator and discriminator
input_size = 100  # Latent space dimension
output_channels = 1  # Assuming univariate time series
sequence_length = 8400  # Target sequence length

generator = Generator(input_size=100, output_channels=1, sequence_length=8400).to(device)
discriminator = Discriminator(input_channels=1).to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()  # Binary cross-entropy loss for real/fake classification
optimizer_G = optim.Adam(generator.parameters(), lr=0.0003)  # Adam optimizer for generator
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)  # Adam optimizer for discriminator
