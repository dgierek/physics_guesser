import torch
import torch.nn as nn
import torch.autograd as autograd


class CombinedLoss(nn.Module):
    def __init__(self):
        """
        Creates loss function as defined in the article: https://arxiv.org/pdf/2005.11212
        """
        super(CombinedLoss, self).__init__()

    def compute_jacobian(self, output, input):
        jacobian = []
        for i in range(output.size(1)):
            grad_outputs = torch.zeros_like(output)
            grad_outputs[:, i] = 1
            jacobian_i = autograd.grad(outputs=output, inputs=input, grad_outputs=grad_outputs, retain_graph=True,
                                       create_graph=True, allow_unused=True)[0]
            jacobian.append(jacobian_i if jacobian_i is not None else torch.zeros_like(input))
        jacobian = torch.stack(jacobian, dim=1)
        return jacobian

    def forward(self, reconstructed, x_curr, x_next, z_prev, z_curr, z_next_pred, z_next_true, latent_dim):
        x_curr.requires_grad_(True)
        x_next.requires_grad_(True)
        z_prev.requires_grad_(True)
        z_curr.requires_grad_(True)
        z_next_pred.requires_grad_(True)
        z_next_true.requires_grad_(True)

        # Reconstruction Loss
        # reconstruction_loss = (torch.linalg.vector_norm(reconstructed - x_curr) /
        #                        (torch.linalg.vector_norm(x_curr) + 1.e-8))
        reconstruction_loss = torch.linalg.vector_norm(reconstructed - x_curr)

        # Prediction Error Loss
        prediction_error_loss = (torch.linalg.vector_norm(z_next_true - z_next_pred) /
                                 (torch.linalg.vector_norm(z_curr - z_prev) + 1e-8))

        # Non-Linearity Loss
        w_curr = torch.cat((z_prev, z_curr), dim=1)
        w_next = torch.cat((z_curr, z_next_pred), dim=1)
        jacobian_w_curr = self.compute_jacobian(w_curr, x_curr)
        jacobian_w_next = self.compute_jacobian(w_next, x_next)
        '''
        Previous version:
        non_linearity_loss = 1 / (2 * latent_dim**2) * torch.norm(jacobian_w_next -
                                                                  jacobian_w_curr, p=2).pow(2)
        '''
        non_linearity_loss = jacobian_w_next - jacobian_w_curr

        # Acceleration Loss
        I = torch.eye(latent_dim)
        M = torch.cat((-I, 2*I), dim=1)
        '''
        Previous version:
        acceleration_loss = 1 / latent_dim * torch.norm(z_next_pred - M @ w_curr.T, p=1)
        '''
        acceleration_loss = z_next_pred - M @ w_curr.T

        return reconstruction_loss, prediction_error_loss, non_linearity_loss, acceleration_loss