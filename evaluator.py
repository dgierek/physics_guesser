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


class ReconstructionError(nn.Module):
    """Computes normalized reconstruction error"""
    def __init__(self):
        super(ReconstructionError, self).__init__()

    def forward(self, x_i, reconstructed):
        return torch.mean(torch.linalg.vector_norm(x_i - reconstructed) / (torch.linalg.vector_norm(x_i) + 1e-4))  # Avoid division by zero


class PredictionError(nn.Module):
    """Computes the prediction error"""
    def __init__(self):
        super(PredictionError, self).__init__()

    def forward(self, z_next, z_next_pred, z_i, z_prev):
        torch.mean(torch.linalg.vector_norm(z_next - z_next_pred) /
                   (torch.linalg.vector_norm(z_i - z_prev) + 1e-4))


class NonlinearityError(nn.Module):
    def __init__(self, evolution_operator, latent_dim):
        """Initializes the loss computation with an evolution operator."""
        super(NonlinearityError, self).__init__()
        self.evolution_operator = evolution_operator  # Pass EvolutionOperator model
        self.scale_factor = 1 / (4 * latent_dim**3)  # Scaling constant

    def compute_jacobian_norm(self, z_prev, z_i):
        """Computes the Jacobian matrix and its L1 norm."""
        inputs = (z_i.requires_grad_(), z_prev.requires_grad_())

        # Compute Jacobian using PyTorch autograd
        J_matrix = torch.autograd.functional.jacobian(lambda x: self.evolution_operator(x[0], x[1]), inputs)

        # Compute L1 norm of Jacobian
        jacobian_norm = torch.norm(J_matrix[0], p=1)  # L1 norm
        return jacobian_norm

    def forward(self, z_i, z_prev):
        """Computes nonlinearity loss based on input latent variables."""
        diff = torch.abs(z_i - z_prev)  # |z_i - z_{i-1}|
        jacobian_norm = self.compute_jacobian_norm(z_prev, z_i)

        # Nonlinearity error calculation
        loss_value = self.scale_factor * diff * jacobian_norm
        return torch.mean(loss_value)  # Aggregate loss


class AccelerationError(nn.Module):
    """Computes the acceleration error"""
    def __init__(self, evolution_operator, latent_dim):
        super(AccelerationError, self).__init__()
        self.evolution_operator = evolution_operator
        self.latent_dim = latent_dim

    def forward(self, z_i, z_prev):
        w = torch.cat((z_i, z_prev), dim=1) # MAYBE IT SHOULD BE Z_PREV, Z_I
        u_of_w = self.evolution_operator(z_i, z_prev)
        I = torch.eye(self.latent_dim)
        M = torch.cat((-I, 2 * I), dim=1)

        accel_loss = (1 / self.latent_dim) * torch.norm(u_of_w - M @ w, p=1)
        return torch.mean(accel_loss)
