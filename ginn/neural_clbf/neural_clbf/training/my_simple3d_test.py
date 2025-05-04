from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

import torch.nn as nn

from neural_clbf.controllers import NeuralCLBFController, NeuralCBFController
from neural_clbf.controllers.simple_neural_cbf_controller import SimpleNeuralCBFController
from neural_clbf.systems.simple3d import Simple3DRobot 

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [0.5, 0.5, 0.5],
    ]
)
simulation_dt = 0.01

class CustomNN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_size))
        self.layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)


def generate_batch_with_masks(bounds: tuple, N: int, goal_fraction: float = .2) -> tuple:
    upper_limit, lower_limit = bounds
    
    dim = upper_limit.shape[0]  # Dimension inferred from limits
    upper_broadcast = upper_limit.unsqueeze(0).expand(N, dim)
    lower_broadcast = lower_limit.unsqueeze(0).expand(N, dim)
    result = torch.rand((N, dim)) * (upper_broadcast - lower_broadcast) + lower_broadcast
    
    # Create safe mask randomly
    safe_mask = torch.rand(N) > 0.5  # True/False randomly with 50% probability
    unsafe_mask = ~safe_mask
    
    # Create goal mask by randomly selecting a fraction of true values in the safe mask
    safe_indices = torch.nonzero(safe_mask).squeeze()
    num_goal_points = int(len(safe_indices) * goal_fraction)
    goal_indices = torch.randperm(len(safe_indices))[:num_goal_points]
    goal_mask = torch.zeros(N, dtype=torch.bool)
    goal_mask[safe_indices[goal_indices]] = True
    
    return result, goal_mask, safe_mask, unsafe_mask

def main():
    # Define the scenarios
    nominal_params = {}
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = Simple3DRobot(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    input_dim = len(dynamics_model.angle_dims) + dynamics_model.n_dims
    hidden_size = 48
    num_layers = 2

    # Create a custom neural network
    V_nn = CustomNN(input_dim, hidden_size, num_layers)

    # Initialize the controller
    cbf_controller = SimpleNeuralCBFController(
        dynamics_model,
        scenarios,
        V_nn,
        cbf_lambda=1.0,
        cbf_relaxation_penalty=1
    )

    # print(dynamics_model._f(torch.Tensor([[1, 1]]), nominal_params))
    # print(dynamics_model._g(torch.Tensor([[1, 1]]), nominal_params))

    trials = 5
    for _ in range(trials):
        batch = generate_batch_with_masks(dynamics_model.state_limits, 64)
        x, goal_mask, safe_mask, unsafe_mask = batch

        losses_list = cbf_controller.descent_loss(
            x
        )

        total_loss = 0
        for _, loss in losses_list:
            if not loss.isnan():
                total_loss += loss

        print(total_loss)

if __name__ == "__main__":
    main()
