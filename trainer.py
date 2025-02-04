import torch.nn as nn
from torch import device, cuda
from torch.optim import Adam
from autoencoder import AutoencoderWithEvolution
from datasets import ImageDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluator import CombinedLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

device = device('cuda' if cuda.is_available() else 'cpu')
print('Device:', str(device))

# Initialising our autoencoder:
model = AutoencoderWithEvolution().to(device)

# Defining transformation of images:
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # down sampling the resolution of the images
    transforms.ToTensor()])

# Example Training Loop
criterion = CombinedLoss(alpha=1.0, beta=1.0, gamma=1.0)
optimizer = Adam(model.parameters(), lr=1e-3)

num_epochs = 10 # 1 epoch is equivalent to training on one whole simulation

total_Loss = []
total_reconstruction_Loss = []
total_prediction_error_Loss = []
total_non_linearity_Loss = []
total_acceleration_Loss = []

for epoch in range(num_epochs):
    # Getting data out of one of the gravity simulations:
    dataset = ImageDataset(directory=rf'simulation_frames/gravity_{epoch}/simulation_snapshots', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    total_loss = 0
    total_reconstruction_loss, total_prediction_error_loss, total_non_linearity_loss, total_acceleration_loss = (
        0, 0 ,0 ,0)
    for _, data in tqdm(dataloader):
        x_prev, x_curr, x_next = data.squeeze(0).to(device)
        x_prev = x_prev.unsqueeze(0).requires_grad_(True)
        x_curr = x_curr.unsqueeze(0).requires_grad_(True)
        x_next = x_next.unsqueeze(0).requires_grad_(True)
        # print('x_prev.shape', x_prev.shape)  # gives torch.Size([1, 3, 64, 64])
        # print('x_curr.shape', x_curr.shape)  # gives torch.Size([1, 3, 64, 64])
        # print('x_next.shape', x_next.shape)  # gives torch.Size([1, 3, 64, 64])

        optimizer.zero_grad()

        # Encode and decode current frame
        reconstructed, z_curr, z_next_pred = model(x_curr, z_prev=model.encoder(x_prev))

        # Encode next frame to get ground truth z_next
        z_next_true = model.encoder(x_next)

        # Compute losses
        reconstruction_loss, prediction_error_loss, non_linearity_loss, acceleration_loss, loss = criterion(
            reconstructed, x_curr, x_next, z_prev=model.encoder(x_prev), z_curr=z_curr,
            z_next_pred=z_next_pred, z_next_true=z_next_true, latent_dim=10)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_prediction_error_loss += prediction_error_loss.item()
        total_non_linearity_loss += non_linearity_loss.item()
        total_acceleration_loss += acceleration_loss.item()

    print(f"Epoch {epoch + 1}, total loss: {total_loss / len(dataloader):.4f}")
    print(f"Epoch {epoch + 1}, reconstruction loss: {total_reconstruction_loss / len(dataloader):.4f}")
    print(f"Epoch {epoch + 1}, prediction error loss: {total_prediction_error_loss / len(dataloader):.4f}")
    print(f"Epoch {epoch + 1}, non linearity loss: {total_non_linearity_loss / len(dataloader):.4f}")
    print(f"Epoch {epoch + 1}, acceleration loss: {total_acceleration_loss / len(dataloader):.4f}")

    total_Loss.append(total_loss / len(dataloader))
    total_reconstruction_Loss.append(total_reconstruction_loss / len(dataloader))
    total_prediction_error_Loss.append(total_prediction_error_loss / len(dataloader))
    total_non_linearity_Loss.append(total_non_linearity_loss / len(dataloader))
    total_acceleration_Loss.append(total_acceleration_loss / len(dataloader))

# Visualizing the change of losses:
plt.plot(range(num_epochs), total_Loss, label="total loss")
plt.plot(range(num_epochs), total_reconstruction_Loss, label="reconstruction loss")
plt.plot(range(num_epochs), total_prediction_error_Loss, label="prediction error loss")
plt.plot(range(num_epochs), total_non_linearity_Loss, label="non linearity loss loss")
plt.plot(range(num_epochs), total_acceleration_Loss, label="acceleration loss loss")
plt.xlabel('Epoch number')
plt.ylabel('logarithmic Loss')
plt.yscale('log')
plt.legend()
plt.show()