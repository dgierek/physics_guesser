# Third party imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision

# Self written library imports:
import autoencoder
import evaluator
import datasets

'''SPRAWDŹ JAK TO MA BYĆ Z TYM WZOREM NA LOSSS!!!!!!!!!!!!!!!!'''

def fit_model_gravity(learning_rate=1.e-3, alpha=1.e-3, beta=1.e-3, gamma=0., latent_dim=10, num_epochs=30,
              recon_loss_only=False, save_model=False):

    """
    The function performs training of the model on gravity simulations for given number of epochs. One epoch is
    equivalent to training on one whole simulation.
    :param float learning_rate: learning_rate at which model will be trained
    :param float alpha: prediction loss impact factor in the total loss function
    :param float beta: non-linearity loss impact factor in the total loss function
    :param float gamma: acceleration loss impact factor in the total loss function
    :param int latent_dim: dimension of the model's latent space
    :param int num_epochs: number of epochs for which the model will be trained
    :param bool recon_loss_only: if True uses only the reconstruction part of the loss function
    :param bool save_model: if True saves the model in saved_models directory
    """
    # Switching to gpu if available:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', str(device))

    # Initialising our autoencoder:
    model = autoencoder.AutoencoderWithEvolution().to(device)

    # Defining transformation of images:
    transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((64, 64)),  # down sampling the resolution of the images
      torchvision.transforms.ToTensor()])

    # Initialising the optimizer:
    criterion = evaluator.CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialising losses history:
    total_loss_history = []
    recon_loss_history = []
    pred_loss_history = []
    nonlin_loss_history = []
    accel_loss_history = []

    for epoch in range(num_epochs):
      # Getting data out of one of the gravity simulations:
      dataset = datasets.ImageDataset(directory=rf'simulation_frames/gravity_{epoch % 145}/simulation_snapshots',
                                      transform=transform)
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

      recon_loss_per_epoch = torch.tensor(0.0, device=device, requires_grad=True)
      pred_loss_per_epoch = torch.tensor(0.0, device=device, requires_grad=True)
      nonlin_loss_per_epoch = torch.tensor(0.0, device=device, requires_grad=True)
      accel_loss_per_epoch = torch.tensor(0.0, device=device, requires_grad=True)

      for _, data in tqdm(dataloader):
        x_prev, x_curr, x_next = data.squeeze(0).to(device)
        x_prev = x_prev.unsqueeze(0).requires_grad_(True)
        x_curr = x_curr.unsqueeze(0).requires_grad_(True)
        x_next = x_next.unsqueeze(0).requires_grad_(True)

        # Encode and decode current frame
        reconstructed, z_curr, z_next_pred = model(x_curr, z_prev=model.encoder(x_prev))

        # Encode next frame to get ground truth z_next
        z_next_true = model.encoder(x_next)

        # Compute losses
        recon_loss, pred_loss, nonlin_loss, accel_loss = criterion(reconstructed=reconstructed, x_curr=x_curr,
                                                                   x_next=x_next, z_prev=model.encoder(x_prev),
                                                                   z_curr=z_curr, z_next_pred=z_next_pred,
                                                                   z_next_true=z_next_true, latent_dim=latent_dim)

        recon_loss_per_epoch = recon_loss_per_epoch + recon_loss
        # pred_loss_per_epoch = pred_loss_per_epoch + pred_loss
        # nonlin_loss_per_epoch = nonlin_loss_per_epoch + nonlin_loss
        # accel_loss_per_epoch = accel_loss_per_epoch + accel_loss

      if not recon_loss_only:
        '''SPRAWDŹ JAK TO MA BYĆ!!!!!!!!!!!!!!!!'''
        total_loss_per_epoch = (recon_loss_per_epoch + alpha * pred_loss_per_epoch + beta *
                      1 / (2 * latent_dim ** 2) * torch.norm(nonlin_loss_per_epoch, p=2).pow(2) +
                      gamma * 1 / latent_dim * torch.norm(accel_loss_per_epoch, p=1)
                      )

        print(f"Epoch {epoch + 1}/{num_epochs}, total loss: {total_loss_per_epoch / len(dataloader):.4f}")
        print(f"Epoch {epoch + 1}, reconstruction loss: {recon_loss_per_epoch / len(dataloader):.4f}")
        print(f"Epoch {epoch + 1}, prediction error loss: {pred_loss_per_epoch / len(dataloader):.4f}")
        print(f"Epoch {epoch + 1}, non linearity loss: {nonlin_loss_per_epoch / len(dataloader):.4f}")
        print(f"Epoch {epoch + 1}, acceleration loss: {accel_loss_per_epoch / len(dataloader):.4f}")

        total_loss_history.append(total_loss_per_epoch / len(dataloader))
        recon_loss_history.append(recon_loss_per_epoch / len(dataloader))
        pred_loss_history.append(pred_loss_per_epoch / len(dataloader))
        nonlin_loss_history.append(nonlin_loss_per_epoch / len(dataloader))
        accel_loss_history.append(accel_loss_per_epoch / len(dataloader))

      else:
        total_loss_per_epoch = recon_loss_per_epoch / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, reconstruction loss (total loss): "
              f"{recon_loss_per_epoch / len(dataloader):.4f}")

        recon_loss_history.append(total_loss_per_epoch.item() / len(dataloader))

      optimizer.zero_grad()
      total_loss_per_epoch.backward()
      optimizer.step()


    # Saving the model
    if save_model:
      if recon_loss_only:
        torch.save(model.state_dict(),
                   fr'saved_models/trained_on_{num_epochs}_simul_lr{learning_rate}_recon_loss_only.pth')
      else:
        torch.save(model.state_dict(),
                   fr'saved_models/trained_on_{num_epochs}_simul_lr{learning_rate}.pth')

    # Visualizing the change of losses:
    if not recon_loss_only:
      plt.plot(range(num_epochs), total_loss_history, label="total loss")
      plt.plot(range(num_epochs), recon_loss_history, label="reconstruction loss")
      plt.plot(range(num_epochs), pred_loss_history, label="prediction error loss")
      plt.plot(range(num_epochs), nonlin_loss_history, label="non linearity loss")
      plt.plot(range(num_epochs), accel_loss_history, label="acceleration loss")

    else:
      plt.plot(range(num_epochs), recon_loss_history, label="reconstruction loss")

    plt.xlabel('Epoch number')
    plt.ylabel('logarithmic Loss')
    plt.yscale('log')
    plt.legend()
    plt.show()