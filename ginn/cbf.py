import os
## For setup
import torch
from configs.get_config import get_config_from_yml
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from GINN.helpers.mp_manager import MPManager
from GINN.helpers.timer_helper import TimerHelper
from GINN.morse.scc_surfacenet_manager import SCCSurfaceNetManager
from GINN.problem_sampler import ProblemSampler
from GINN.visualize.plotter_3d import Plotter3d
from train.train_utils.autoclip import AutoClip
from utils import get_model, get_stateless_net_with_partials

## For extracting and plotting a mesh
import k3d
from notebooks.notebook_utils import get_mesh_for_latent

## For running a training loop
import einops
from tqdm import trange
from models.model_utils import tensor_product_xz
from train.losses import closest_shape_diversity_loss, eikonal_loss, envelope_loss, interface_loss, normal_loss_euclidean, obstacle_interior_loss, strain_curvature_loss
from train.train_utils.latent_sampler import sample_new_z
from utils import set_all_seeds, get_is_out_mask

import ipywidgets
import os

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, jacfwd, vmap
from utils import get_model, get_stateless_net_with_partials

import k3d
import matplotlib.pyplot as plt
import numpy as np

#################################################
#
#   CLASSES
#   CLASSES
#   CLASSES
#
##################################################

import torch.nn as nn
from typing import Tuple, Optional
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

DEFAULT_Z_VAL = 1.0

def batch_jacobian(f, x, z):
    """
    Compute the Jacobian using forward-mode differentiation (jacrev).
    """
    x = x.clone().detach().requires_grad_(True)
    z = z.clone().detach()

    return jacrev(f, argnums=0)(x, z)  # Computes d(f)/dx efficiently

# def smooth_mask(x, a, b, k):
#     sig_a = torch.sigmoid(k * (x - a))
#     sig_b = torch.sigmoid(k * (b - x))
#     return sig_a * sig_b

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

def load_yaml_to_dict(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

from torch.func import jacrev, vmap

def smooth_mask(x, a, b, k):
    # k is a distance metric now
    sig_a = torch.sigmoid((x - a) * (4.0 / k))
    sig_b = torch.sigmoid((b - x) * (4.0 / k))
    return sig_a * sig_b

# Define the combined 3D mask for a prism
def prism_mask(coords, x_range, y_range, z_range, k):
    x_mask = smooth_mask(coords[..., 0], x_range[0], x_range[1], k)
    y_mask = smooth_mask(coords[..., 1], y_range[0], y_range[1], k)
    z_mask = smooth_mask(coords[..., 2], z_range[0], z_range[1], k)
    
    mask = x_mask * y_mask * z_mask
    return mask
    
class CBFModel(nn.Module):
    def __init__(self, n, configs, bounds=[], scale_factors = [], centers_for_translations=[], alpha=50, device='cpu', scale = 10, mask_dist=8, upper_bound=1, do_mask=True, use_adapters=False, netp=None):
        super(CBFModel, self).__init__()

        self.n = n
        self.configs = configs
        self.bounds = bounds  # active bounds (may be subset)
        self.scale_factors = scale_factors
        self.centers_for_translations = centers_for_translations
        if not use_adapters:
            self.models = nn.ModuleList([get_model(config).to(device) for config in configs])
            self.all_models = nn.ModuleList([get_model(config).to(device) for config in configs])
        else:
            self.models = nn.ModuleList([get_adapter_model(config).to(device) for config in configs])
            self.all_models = nn.ModuleList([get_adapter_model(config).to(device) for config in configs])

        self.all_n = n
        self.all_configs = configs.copy() if isinstance(configs, list) else configs
        self.all_bounds = bounds.copy() if isinstance(bounds, list) else bounds
        self.all_scale_factors = scale_factors.copy() if isinstance(scale_factors, list) else scale_factors
        self.all_centers_for_translations = centers_for_translations.copy() if isinstance(centers_for_translations, list) else centers_for_translations
        self.all_models = self.models  # original full list of submodels

        self.scale = 1
        self.alpha = alpha
        self.mask_dist = mask_dist
        self.device = device
        self.jacobian = None
        self.upper_bound = upper_bound
        self.do_mask = do_mask
        self.netp = netp
        self.model2mesh = {}

        self.netp = get_stateless_net_with_partials(self, use_x_and_z_arg=True)

    def compute_spectral_norm(self) -> float:
        total_L = 0.0
        for submodel in self.models:
            submodel_L = 1.0
            for layer in submodel.modules():
                if isinstance(layer, torch.nn.Linear):
                    W = layer.weight
                    s = torch.linalg.svdvals(W)[0].item()  # largest singular value
                    print(s)
                    submodel_L *= s
            total_L += submodel_L  # could use max/submodel_L if using max-based aggregation
        return total_L

    def compute_softmax_lipschitz(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate an upper bound on the gradient norm of h(x)  
        using the softmax-weighted Jacobians from get_jacobian.  
        Returns a (B,1) tensor of ∥∇h(x_i)∥ for each sample i.
        """
        for m in self.models:
            m.eval()

        x = x.to(self.device)
        z = z.to(self.device)

        alpha = self.alpha
        B = x.shape[0]

        outputs, jacobians = [], []
        for i, submodel in enumerate(self.models):
            x_pre = (x - self.centers_for_translations[i]) / self.scale_factors[i]
            out = submodel(x_pre, z)            # (B,1)
            outputs.append(-alpha * out)

            J = self.get_jacobian(x_pre, z)     # (B,3)
            J_scaled = J / self.scale_factors[i]
            jacobians.append(J_scaled)

        outputs   = torch.stack(outputs, dim=0)    # (M, B, 1)
        jacobians = torch.stack(jacobians, dim=0)  # (M, B, 3)

        weights = torch.softmax(outputs, dim=0)    # (M, B, 1)
        weighted_jacobian = (weights * jacobians).sum(dim=0)  # (B, 3)

        # per-sample norm → shape (B,1)
        lips = torch.norm(weighted_jacobian, dim=1, keepdim=True)
        return lips

    def get_jacobian(self, x, z):
        def f_closed(x, z):
            return self.netp.f(x, z)

        vf_x_closed = vmap(jacrev(f_closed, argnums=0), in_dims=(0, 0), out_dims=0)
        jacobian_vf_x = vf_x_closed(x, z)
        if torch.isnan(jacobian_vf_x).any():
            jacobian_vf_x = torch.nan_to_num(jacobian_vf_x, nan=0.0)
            # print("NaN Jacobian, V:", self.forward(x, z))
        return jacobian_vf_x

    def forward(self, xog, z, mask_dist=None):
        # xog.requires_grad_(True)  # Ensure gradients are tracked
        outs = torch.zeros(len(self.models), xog.shape[0], 1)
        if not mask_dist: mask_dist = self.mask_dist

        self.centers_for_translations = torch.stack([c.to(self.device) for c in self.centers_for_translations])
        self.bounds = torch.stack([b.to(self.device) for b in self.bounds])
        self.scale_factors = torch.stack([s.to(self.device) for s in self.scale_factors])
        xog = xog.to(self.device)
        z = z.to(self.device)

        def output_func(x, z):
            # x: input tensor (could be batched)
            # z: additional argument to the network
            out_list = []
            for i, model in enumerate(self.models):
                # Compute mask if masking is enabled
                if self.do_mask:
                    m = prism_mask(x, self.bounds[i][0], self.bounds[i][1], self.bounds[i][2], mask_dist)
                    # When x is a single sample (0D case), m may be 0D.
                    if m.dim() == 0:
                        mask = m.unsqueeze(0)
                    else:
                        mask = m.unsqueeze(1)
                else:
                    mask = 1

                # Preprocess x for the current submodel
                x_preprocessed = (x - self.centers_for_translations[i]) / self.scale_factors[i]
                model_output = model(x_preprocessed, z)
                temp = mask * torch.exp(-self.alpha * model_output)

                # Avoid in-place squeeze issues: if temp is scalar or has a singleton batch dim,
                # adjust accordingly.
                if temp.dim() == 0:
                    out_list.append(temp.unsqueeze(0))
                elif temp.size(0) == 1:
                    out_list.append(temp.squeeze(0))
                else:
                    out_list.append(temp)
            
            # Stack along a new dimension so that out_values has shape: (num_models, ..., output_dim)
            out_values = torch.stack(out_list, dim=0)
            # Sum across models, then take the log and scale
            output = self.scale * -1 / self.alpha * torch.log(torch.sum(out_values, axis=0))
            output = torch.clamp(output, max=self.upper_bound)
            return output


        output = output_func(xog, z)
        return output

    def named_parameters(self, recurse=True, remove_duplicate=False):
        for i in range(self.n):
            for name, param in self.models[i].named_parameters(recurse=recurse, remove_duplicate=remove_duplicate):
                yield f'models.{i}.{name}', param

    def named_buffers(self, recurse=True, remove_duplicate=False):
        for i in range(self.n):
            for name, buffer in self.models[i].named_buffers(recurse=recurse, remove_duplicate=remove_duplicate):
                yield f'models.{i}.{name}', buffer
    
    def __iter__(self):
        return iter(self.models)
    
    def shift_models(self, indices: list[int], step: torch.Tensor):
        # step: (3,)
        step = step.to(device)
        for idx in indices:
            self.centers_for_translations[idx] += step
            self.bounds[idx] += step.unsqueeze(1)
    
    def move_model(self, index: int, new_center: torch.Tensor):
        new_center = new_center.to(self.device)
        current_center = self.centers_for_translations[index].to(self.device)
        translation = new_center - current_center

        # self.centers_for_translations[index] = new_center
        # self.bounds[index] = self.bounds[index] + translation.unsqueeze(1)

        self.centers_for_translations[index] =  new_center
        self.bounds[index] = self.bounds[index] + translation.unsqueeze(1)

    
    def get_dist_to_submodel(self, model_idx: int, point: torch.Tensor):
        point = point.to(self.device)  # Ensure correct device
        bounds = self.bounds[model_idx]

        clamped_point = torch.clamp(point, bounds[:, 0], bounds[:, 1])
        return torch.norm(point - clamped_point, dim=0)

    def get_submodel_in_rad(self, points: torch.Tensor, radius=0.5):
        points = points.to(self.device)
        if points.dim() == 1:
            points = points.unsqueeze(0)  # Convert (3,) → (1,3) for single point case
        indices = set()

        for i in range(len(self.bounds)):
            distances = torch.stack([self.get_dist_to_submodel(i, point) for point in points])
            if torch.any(distances < radius):
                indices.add(i)

        return sorted(indices) 

    def set_submodels(self, indices: list[int]):
        if len(indices) == 0:
            indices = [0]
        self.configs = [self.all_configs[i] for i in indices]
        self.bounds = [self.bounds[i] for i in indices]
        self.scale_factors = [self.all_scale_factors[i] for i in indices]
        self.centers_for_translations = [self.all_centers_for_translations[i] for i in indices]
        self.models = nn.ModuleList([self.all_models[i] for i in indices])
        self.n = len(indices)
        self.netp = get_stateless_net_with_partials(self, use_x_and_z_arg=True)

    def reset_submodels(self):
        self.configs = self.all_configs.copy() if isinstance(self.all_configs, list) else self.all_configs
        self.bounds = self.bounds.copy() if isinstance(self.bounds, list) else self.bounds
        self.scale_factors = self.all_scale_factors.copy() if isinstance(self.all_scale_factors, list) else self.all_scale_factors
        self.centers_for_translations = self.all_centers_for_translations.copy() if isinstance(self.all_centers_for_translations, list) else self.all_centers_for_translations
        self.models = nn.ModuleList(self.all_models)  # Ensure self.models remains an nn.ModuleList
        self.n = self.all_n
        self.netp = get_stateless_net_with_partials(self, use_x_and_z_arg=True)

    @classmethod
    def create_model(_, config_paths, sub_model_paths, alpha=50, device='cpu', upper_bound=8, all_scale_factors=torch.empty(0), all_center_translations=torch.empty(0), all_bounds=torch.empty(0), use_adapters=False):
        scale_name = "scale_factor.npy"
        center_translation_name = "center_for_translation.npy"
        bounds_name = "bounds.npy"

        if use_adapters:
            configs = [load_yaml_to_dict(path) for path in config_paths]
        else:
            configs = [get_config_from_yml(path) for path in config_paths]

        if all_scale_factors.shape[0] == 0:
            all_scale_factors = torch.stack([
                torch.from_numpy(np.load(os.path.join(config['dataset_dir'], scale_name))).float().to(device)
                for config in configs
            ])
            print("WARNING - Using from config")

        if all_center_translations.shape[0] == 0:
            all_center_translations = torch.stack([
                torch.from_numpy(np.load(os.path.join(config['dataset_dir'], center_translation_name))).float().to(device)
                for config in configs
            ])
            print("WARNING - Using from config")

        if all_bounds.shape[0] == 0:
            all_bounds = torch.stack([
                torch.from_numpy(np.load(os.path.join(config['dataset_dir'], bounds_name))).float().to(device)
                for config in configs
            ])
            print("WARNING - Using from config")

        model = CBFModel(len(configs), configs, bounds=all_bounds, scale_factors = all_scale_factors, centers_for_translations=all_center_translations, alpha=alpha, device=device, use_adapters=use_adapters)
        model.netp = get_stateless_net_with_partials(model, use_x_and_z_arg=True)

        for i in range(model.n): 
            model.models[i].load_state_dict(torch.load(sub_model_paths[i], map_location=device))

        # sub_model_netps = [get_stateless_net_with_partials(model.models[i], use_x_and_z_arg=configs[i]['use_x_and_z_arg']) for i in range(model.n)]

        z = torch.tensor([float(DEFAULT_Z_VAL)]).to(device)
        return model, z

    @classmethod
    def update_plot(_, model, z, plot, n = None, w = 35, mc_resolution=32, device='cpu', bound_pts=False):
        for obj in list(plot.objects):
            if "cbf" in str(obj.name):
                plot -= obj

        if n:
            xp = np.linspace(-w, w, n)
            yp = np.linspace(-w, w, n)
            zp = np.linspace(-w, w, n)

            X, Y, Z = np.meshgrid(xp, yp, zp, indexing='ij')

            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            points_tensor = torch.from_numpy(points).float().to(device)

            zs_tensor = torch.full((points.shape[0],), z.item()).unsqueeze(1).float()  # create z_tensor with same number of points

            cbf_outs = np.zeros(points.shape[0])
            cbf_outs = netp.f_(netp.params_, points_tensor, zs_tensor)
            
            colors = np.zeros(points.shape[0], dtype=np.uint32)

            for i in range(points.shape[0]):
                if cbf_outs[i] > 0: 
                    colors[i] = 0x00FF00  # Green for positive
                else: 
                    colors[i] = 0x000000  # Black for negative
            
            point_size = 0.25
            points_plot = k3d.points(positions=points.astype(np.float32), colors=colors, point_size = point_size)
            plot += points_plot

        # model.bounds[0] = torch.tensor([[-w, w], [-w, w], [-w, w]])
        # print(model.bounds.shape)

        # Compute global bounds
        global_lower_bound = torch.min(model.bounds[..., 0], dim=0).values  # Min of all lower bounds
        global_upper_bound = torch.max(model.bounds[..., 1], dim=0).values  # Max of all upper bounds
        global_bounds = torch.stack([global_lower_bound, global_upper_bound], dim=-1)

        delta = 2  # Amount to expand the bounds by
        bounds = torch.stack([
            global_bounds[..., 0] - delta,  # Lower bound decreases
            global_bounds[..., 1] + delta   # Upper bound increases
        ], dim=-1)

        vert, faces = get_mesh_for_latent(model.netp.f_, model.netp.params_, z, bounds, mc_resolution=mc_resolution, device=device, chunks=1, flip_faces=True)
        mesh_plot = k3d.mesh(vert, faces, color=0xff0000, side='double', opacity=1, name="cbf")
        plot += mesh_plot

        # for i in range(model.bounds.shape[0]):
        #     reshaped_bound = model.bounds[i].T.cpu().numpy()
        #     plot += k3d.points(positions=reshaped_bound, point_size = 0.25, name="cbf")

        if bound_pts:
            plot += k3d.points(positions=bounds.T.cpu().numpy(), point_size = 0.25, name="cbf")

    def get_dist_to_submodel(self, model_idx: int, point: torch.Tensor):
        """
        Returns the distance of the point to the submodel's bounding box.
        Uses the currently active bounds.
        """
        point = point.to(self.device)  # Ensure correct device
        bounds = self.bounds[model_idx]
        clamped_point = torch.clamp(point, bounds[:, 0], bounds[:, 1])
        return torch.norm(point - clamped_point, dim=0)
    
    def get_dist_to_submodel_original(self, model_idx: int, point: torch.Tensor):
        """
        Returns the distance of the point to the submodel's bounding box,
        using the original full configuration (self.all_bounds).
        """
        point = point.to(self.device)
        bounds = self.all_bounds[model_idx]
        clamped_point = torch.clamp(point, bounds[:, 0], bounds[:, 1])
        return torch.norm(point - clamped_point, dim=0)
    
    def get_submodel_in_rad(self, points: torch.Tensor, radius=0.5):
        """
        Returns the indexes (from the original configuration) of all submodels
        whose original bounds are within 'radius' of any of the provided points.
        This always uses the original configuration.
        """
        points = points.to(self.device)
        if points.dim() == 1:
            points = points.unsqueeze(0)  # (1,3) if a single point is provided
        indices = set()
        # Loop over the full set of submodels (original configuration)
        for i in range(len(self.all_bounds)):
            # Use the original bounds via the helper function
            distances = torch.stack([self.get_dist_to_submodel_original(i, point) for point in points])
            if torch.any(distances < radius):
                indices.add(i)
        return sorted(indices)

    def set_submodels(self, indices: list[int]):
        """
        Set the active submodels to only those whose original indices are given.
        All active attributes (configs, bounds, etc.) are updated from the original full configuration.
        """
        if len(indices) == 0:
            indices = [0]
        
        self.configs = [self.all_configs[i] for i in indices]
        self.bounds = [self.all_bounds[i] for i in indices]
        self.scale_factors = [self.all_scale_factors[i] for i in indices]
        self.centers_for_translations = [self.all_centers_for_translations[i] for i in indices]
        self.models = nn.ModuleList([self.all_models[i] for i in indices])
        self.n = len(indices)
        self.netp = get_stateless_net_with_partials(self, use_x_and_z_arg=True)

    def reset_submodels(self):
        """
        Reset the active submodels to the original full configuration.
        """
        
        self.configs = self.all_configs.copy() if isinstance(self.all_configs, list) else self.all_configs
        self.bounds = self.all_bounds.copy() if isinstance(self.all_bounds, list) else self.all_bounds
        self.scale_factors = self.all_scale_factors.copy() if isinstance(self.all_scale_factors, list) else self.all_scale_factors
        self.centers_for_translations = self.all_centers_for_translations.copy() if isinstance(self.all_centers_for_translations, list) else self.all_centers_for_translations
        self.models = nn.ModuleList(self.all_models)
        self.n = self.all_n
        self.netp = get_stateless_net_with_partials(self, use_x_and_z_arg=True)
