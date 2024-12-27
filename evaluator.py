import torch
import torch.nn as nn
import torch.autograd as autograd


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Creates loss function as defined in the article: https://arxiv.org/pdf/2005.11212
        :param alpha: weight for prediction error loss
        :param beta: weight for nonlinear error loss
        :param gamma: weight for acceleration error loss
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # weight for prediction error loss
        self.beta = beta    # weight for nonlinear error loss
        self.gamma = gamma  # weight for acceleration error loss

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
        reconstruction_loss = torch.mean(torch.abs(reconstructed - x_curr) / (torch.abs(x_curr) + 1e-8))

        # Prediction Error Loss
        prediction_error_loss = torch.mean(torch.abs(z_next_true - z_next_pred) / (torch.abs(z_curr - z_prev) + 1e-8))

        # Non-Linearity Loss
        w_curr = torch.cat((z_prev, z_curr), dim=1)
        w_next = torch.cat((z_curr, z_next_pred), dim=1)
        jacobian_w_curr = self.compute_jacobian(w_curr, x_curr)
        jacobian_w_next = self.compute_jacobian(w_next, x_next)
        non_linearity_loss = 1 / (2 * latent_dim**2) * torch.norm(jacobian_w_next - jacobian_w_curr, p=2).pow(2)
        # Acceleration Loss
        I = torch.eye(latent_dim)
        M = torch.cat((-I, 2*I), dim=1)
        acceleration_loss = 1 / latent_dim * torch.norm(z_next_pred - M @ w_curr.T, p=1)

        # Combined Loss
        combined_loss = (reconstruction_loss + self.alpha * prediction_error_loss + self.beta * non_linearity_loss +
                         self.gamma * acceleration_loss)

        return combined_loss


'Example usage'
# combined_loss = CombinedLoss(alpha=1.0, beta=1.0, gamma=1.0)
# reconstructed = torch.randn(1, 3, 64, 64, requires_grad=True)
# x_curr = torch.randn(1, 3, 64, 64, requires_grad=True)
# x_next = torch.randn(1, 3, 64, 64, requires_grad=True)
# z_prev = torch.randn(1, 10, requires_grad=True)
# z_curr = torch.randn(1, 10, requires_grad=True)
# z_next_pred = torch.randn(1, 10, requires_grad=True)
# z_next_true = torch.randn(1, 10, requires_grad=True)
#
# loss = combined_loss(reconstructed, x_curr, x_next, z_prev, z_curr, z_next_pred, z_next_true, 10)
# print('Loss:', loss.item())
