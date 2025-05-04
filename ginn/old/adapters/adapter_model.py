import torch
import torch.nn as nn

from GINN.problem_sampler import ProblemSampler
from train.train_utils.latent_sampler import sample_new_z
from utils import get_stateless_net_with_partials, get_model
from models.model_utils import tensor_product_xz

from collections import defaultdict
import time
from copy import deepcopy

class AdapterMLP(nn.Module):
    def __init__(self, layer_sizes, activation):
        """
        :param layer_sizes: List of integers defining the number of neurons in each layer
                            Example: [3, 128, 128, 1] (input, hidden, hidden, output)
        :param activation: Activation function (as a nn.Module)
        """
        super(AdapterMLP, self).__init__()
        
        assert len(layer_sizes) >= 2, "layer_sizes must have at least input and output dimensions."
        
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation())  # Apply activation function
        
        # Final layer should always be linear
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def get_activation(activation_name):
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "none": lambda: nn.Identity()  # No activation
    }

    if activation_name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {activation_name}")
    
    return activations[activation_name.lower()]

def create_adapter_mlp(layer_sizes=[256, 512, 128, 1], activation_name="relu", siren_config=None):
    """
    Factory method to create an AdapterMLP instance.

    :param layer_sizes: List of layer dimensions, e.g., [3, 128, 128, 1]
    :param activation_name: String name of the activation function (default: "relu")
    :return: An instance of AdapterMLP
    """
    if activation_name == "sine":
        siren_config_tmp = deepcopy(siren_config)
        print(siren_config_tmp)
        siren_config_tmp['layers'] = layer_sizes
        siren_model = get_model(siren_config_tmp)
        return siren_model
    else:
        activation = get_activation(activation_name)
        return AdapterMLP(layer_sizes, activation)

class ConditionalSIRENWithAdapter(nn.Module):
    """
    Conditional SIREN with an adapter replacing the final layer.
    """
    def __init__(self, siren_model, adapter_model):
        super(ConditionalSIRENWithAdapter, self).__init__()
        
        # Remove final linear layer from SIREN
        self.siren = nn.Sequential(*list(siren_model.network.children())[:-1])
        self.adapter = adapter_model  # Adapter MLP replaces final layer
        self.jacobian = None

        # Freeze all but adapter
        for param in self.siren.parameters():
            param.requires_grad = False

    def forward(self, x, z, calc_jacobian=False):
        xz = torch.cat([x, z], dim=-1)  # Ensure concatenation happens before passing to the model
        features = self.siren(xz)  # Pass through modified SIREN layers
        h_x = self.adapter(features)  # Apply adapter MLP

        if calc_jacobian:
            self.jacobian = torch.autograd.functional.jacobian(lambda x: self.forward(x, z, calc_jacobian=False), x)

        return h_x

class LossTimer:
    def __init__(self):
        self.times = defaultdict(list)  # Store loss computation times
        self.start_times = {}  # Store start times for ongoing loss calculations

    def start(self, loss_name):
        """Start timing for a specific loss."""
        self.start_times[loss_name] = time.time()

    def stop(self, loss_name):
        """Stop timing and log duration for a specific loss."""
        if loss_name in self.start_times:
            elapsed_time = time.time() - self.start_times.pop(loss_name)
            self.times[loss_name].append(elapsed_time)

    def print_summary(self):
        """Prints the average and all recorded times for each loss."""
        print("=== Loss Timing Summary ===")
        for loss_name, timings in self.times.items():
            avg_time = sum(timings) / len(timings)
            print(f"{loss_name}: Avg {avg_time:.6f}s")

def get_adapter_model(config):
    # config = load_yaml_to_dict(adapter_config_path)
    config_siren = get_config_from_yml(config["paths"]["siren_config_path"])
    config_siren["device"] = device

    siren_model = get_model(config_siren).to(device)
    siren_model.load_state_dict(torch.load(config["paths"]["pretrained_siren_path"], map_location=device))

    final_layer_size = list(siren_model.network.children())[-3].out_features
    layer_sizes = [final_layer_size] + config["training"]["adapter_mid_layers"]
    activation_name = config["training"]["activation_name"]

    adapter_model = create_adapter_mlp(layer_sizes, activation_name=activation_name, siren_config=config_siren).to(device)
    model = ConditionalSIRENWithAdapter(siren_model, adapter_model).to(device)
    return model

if __name__ == "__main__":
    layers = [128, 128, 128, 1]
    mlp = create_adapter_mlp(layers, activation_name="tanh")