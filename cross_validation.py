from torch import device, cuda, save
from torch.optim import Adam
from autoencoder import AutoencoderWithEvolution
from datasets import ImageDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluator import CombinedLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import ParameterGrid, KFold
import numpy as np

'''IDEA CODE - NEEDS DEBUGGING!'''

# Parameters grid:
param_grid = {
    'lr_change': [0.1, 0.01, 0.001],
    'lr_change_period': [5, 10, 15],
    'alpha_beta_change': [100 ,10, 5, 1],
    'alpha_beta_change_period': [5, 10, 15, 20],
}

# Number of folds for K-Fold Cross-Validation
n_folds = 5
kf = KFold(n_splits=n_folds)

# Placeholder for results
results = []

# Iterate over each combination of hyperparameters
for params in ParameterGrid(param_grid):
    total_val_loss = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Create DataLoaders for training and validation
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Reinitialize model, criterion, optimizer, and scheduler for each fold
        model = AutoencoderWithEvolution().to(device)
        criterion = CombinedLoss(alpha=initial_alpha, beta=initial_beta, gamma=gamma)
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=params['lr_change_period'], gamma=params['lr_change'])

        for epoch in range(num_epochs):
            model.train()
            for _, data in tqdm(train_loader):
                x_prev, x_curr, x_next = data.squeeze(0).to(device)
                x_prev, x_curr, x_next = [x.unsqueeze(0).requires_grad_(True) for x in [x_prev, x_curr, x_next]]
                optimizer.zero_grad()
                reconstructed, z_curr, z_next_pred = model(x_curr, z_prev=model.encoder(x_prev))
                z_next_true = model.encoder(x_next)
                _, _, _, _, loss = criterion(
                    reconstructed, x_curr, x_next, z_prev=model.encoder(x_prev),
                    z_curr=z_curr, z_next_pred=z_next_pred, z_next_true=z_next_true, latent_dim=10)
                loss.backward()
                optimizer.step()
            scheduler.step()
            update_criterion(epoch)

        # Validate on validation fold
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, data in tqdm(val_loader):
                x_prev, x_curr, x_next = data.squeeze(0).to(device)
                x_prev, x_curr, x_next = [x.unsqueeze(0).requires_grad_(True) for x in [x_prev, x_curr, x_next]]
                reconstructed, z_curr, z_next_pred = model(x_curr, z_prev=model.encoder(x_prev))
                z_next_true = model.encoder(x_next)
                _, _, _, _, loss = criterion(
                    reconstructed, x_curr, x_next, z_prev=model.encoder(x_prev),
                    z_curr=z_curr, z_next_pred=z_next_pred, z_next_true=z_next_true, latent_dim=10)
                val_loss += loss.item()
        total_val_loss += val_loss / len(val_loader)

    avg_val_loss = total_val_loss / n_folds
    results.append({'params': params, 'val_loss': avg_val_loss})

# Find the best hyperparameters
best_params = min(results, key=lambda x: x['val_loss'])
print('Best hyperparameters:', best_params)