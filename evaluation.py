# Third party imports:
import torch
import matplotlib.pyplot as plt

# Self written library imports:
import datasets
import autoencoder

# Load the trained model
model = autoencoder.AutoencoderWithEvolution(latent_dim=10)
model.load_state_dict(torch.load("saved_models\\trained_on_400_simul_lr_params_0.001_25_0.9_recon_loss_only_latent_dim_10.pth"))
model.eval()  # Set to evaluation mode

# Select a sample image from the dataset
dataset = datasets.LoadedDataset()
sample_idx = 0  # Change index to visualize different images
sample_image = dataset[sample_idx][1].unsqueeze(0)  # Selecting the middle image in the triplet
sample_image = sample_image.to("cuda" if torch.cuda.is_available() else "cpu")

# Run the image through the model
with torch.no_grad():
    reconstructed, _, _ = model(sample_image)

# Convert tensors to numpy for plotting
original_img = sample_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
reconstructed_img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Plot the original vs reconstructed image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(original_img)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(reconstructed_img)
ax[1].set_title("Reconstructed Image")
ax[1].axis("off")

plt.tight_layout()
plt.show()