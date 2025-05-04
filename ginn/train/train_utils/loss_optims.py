import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# ================= Loss Balancer (Uncertainty-Based Weighting) =================
class LossBalancer(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        # Learnable log-variances (initialized to 0)
        # self.log_vars = nn.Parameter(torch.zeros(num_losses))
        if weights is not None:  # Explicit check for None
            self.log_vars = nn.Parameter(weights.clone().detach())  # Ensure it's a proper tensor
        else:
            self.log_vars = nn.Parameter(torch.zeros(num_losses))  # Trainable loss weights

    def forward(self, losses):
        # Compute weights as exp(-log_var), ensuring positivity
        weights = torch.exp(-self.log_vars)
        balanced_losses = weights * losses + self.log_vars  # log_vars term prevents zero weights
        return balanced_losses.sum()

# ================= GradNorm (Gradient Normalization) =================

class GradNormBalancer(nn.Module):
    def __init__(self, num_losses, alpha=1e-3, min_weight=1e-12, max_weight=1.0, weights=None):
        super().__init__()
        self.num_losses = num_losses
        self.alpha = alpha  # Controls update speed
        self.min_weight = min_weight
        self.max_weight = max_weight

        if weights is not None:
            self.loss_weights = nn.Parameter(weights.clone().detach().clamp(min=min_weight, max=max_weight))
        else:
            self.loss_weights = nn.Parameter(torch.ones(num_losses))  # Trainable loss weights

    def forward(self, losses, model_params):
        weighted_losses = self.loss_weights * losses
        total_loss = weighted_losses.sum()

        grads = torch.autograd.grad(total_loss, model_params, retain_graph=True, create_graph=True)
        grad_norms = torch.stack([torch.norm(g) for g in grads if g is not None])
        mean_norm = grad_norms.mean().detach()

        loss_ratios = losses.detach() / mean_norm
        update_factor = torch.exp(self.alpha * (loss_ratios - 1))

        # Apply the update safely
        self.loss_weights.data *= update_factor

        # Apply a **soft constraint** instead of hard clamping
        self.loss_weights.data = self.min_weight + (self.max_weight - self.min_weight) * torch.sigmoid(self.loss_weights.data)

        return total_loss

# ================= Main Execution =================
if __name__ == "__main__":
    class DummyModel(nn.Module):  # Example Model
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "max_epochs": 100,
    }

    model = DummyModel()
    
    # Train using Loss Balancer
    train(model, config, method="loss_balancer")
    
    # Train using GradNorm
    train(model, config, method="gradnorm")
