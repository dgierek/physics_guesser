import torch.nn as nn
from torch import device, cuda
from torch.optim import Adam
from autoencoder import AutoencoderWithEvolution
from datasets import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Implementing GPU support (if aveilable)
device = device('cuda' if cuda.is_available() else 'cpu')
print('Device:', str(device))

# Initialising our autoencoder:
model = AutoencoderWithEvolution().to(device)

# Getting data from dataloader:
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # down sampling the resolution of the images
    transforms.ToTensor()])

dataset = ImageDataset(directory=r'simulation_frames/gravity_test_0/simulation_snapshots', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Example Training Loop
criterion_reconstruction = nn.MSELoss()  # Reconstruction loss
criterion_prediction = nn.MSELoss()  # Prediction loss
optimizer = Adam(model.parameters(), lr=1e-3)

num_epochs = 1
for epoch in range(num_epochs):
    loss = 0
    for data in tqdm(dataloader):
        x_prev, x_curr, x_next = data.squeeze(0).to(device)
        x_prev = x_prev.unsqueeze(0)
        x_curr = x_curr.unsqueeze(0)
        x_next = x_next.unsqueeze(0)
        # print('x_prev.shape', x_prev.shape)  # gives torch.Size([1, 3, 64, 64])
        # print('x_curr.shape', x_curr.shape)  # gives torch.Size([1, 3, 64, 64])
        # print('x_next.shape', x_next.shape)  # gives torch.Size([1, 3, 64, 64])

        optimizer.zero_grad()

        # Encode and decode current frame
        reconstructed, z_curr, z_next_pred = model(x_curr, z_prev=model.encoder(x_prev))

        # Encode next frame to get ground truth z_next
        z_next_true = model.encoder(x_next)

        # Compute losses
        loss_reconstruction = criterion_reconstruction(reconstructed, x_curr)
        loss_prediction = criterion_prediction(z_next_pred, z_next_true)
        loss = loss_reconstruction + loss_prediction

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
