# Third party imports:
import torch
from tqdm import tqdm


# Self written library imports:
import datasets
import autoencoder
import visualization
import evaluator


def train_autoencoder(model, train_loader, val_loader, epochs=10, initial_lr=0.001, lr_decay_step=10,
                      lr_decay_factor=0.9, save_model=False, device='cuda' if torch.cuda.is_available() else 'cpu'):

    # Move model to GPU if available:
    model.to(device)
    # Define optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Define learning rate scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

    # Store loss values
    history = {"loss": [], "val_loss": []}

    # Instantiate the loss function:
    recon_loss_fn = evaluator.ReconstructionError()

    for epoch in range(epochs):
        total_loss = 0.0
        val_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            batch = batch.to(device)

            x_prev, x_i, x_next = batch[:, 0], batch[:, 1], batch[:, 2]

            optimizer.zero_grad()
            reconstructed, z_i, z_next_pred = model(x_i, z_prev=model.encoder(x_prev))

            # Calculating loss:
            loss = recon_loss_fn(x_i, reconstructed)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update tqdm progress bar with loss
            progress_bar.set_postfix(loss=f"{total_loss:.4f}")

        history["loss"].append(total_loss / len(train_loader))

        # Step the scheduler after every epoch
        scheduler.step()

        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_prev, x_i, x_next = batch[:, 0], batch[:, 1], batch[:, 2]

                reconstructed, z_i, z_next_pred = model(x_i, z_prev=model.encoder(x_prev))

                # Calculating loss:
                loss = recon_loss_fn(x_i, reconstructed)
                val_loss = loss.item()

        model.train()  # Set model back to training mode
        history["val_loss"].append(val_loss / len(val_loader))

        # Print current loss and learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} - Avg. Train Loss: {history['loss'][-1]:.4f}, "
              f"Avg. Val Loss: {history['val_loss'][-1]:.4f}, LR: {current_lr:.6f}")

    print("Training complete!")

    if save_model:
        torch.save(model.state_dict(),fr'saved_models/trained_on_{epochs}_simul_lr_params_{initial_lr}_'
                                      fr'{lr_decay_step}_{lr_decay_factor}_recon_loss_only_latent_dim_'
                                      fr'{model.latent_dim}.pth')
        print("Model saved!")

    return history


if __name__ == '__main__':
    # Instantiate the model:
    model = autoencoder.AutoencoderWithEvolution(latent_dim=10)

    # Load dataset using DataLoader:
    dataset = datasets.LoadedDataset()

    # Split dataset into training and validation sets:
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Train the model:
    history = train_autoencoder(model, train_loader, val_loader, epochs=400, save_model=True, initial_lr=1.e-3,
                                lr_decay_step=25)

    # Plot the loss history:
    visualization.plot_train_history(history)
