#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('ml cuda-toolkit-11.8.0')

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
from utils import set_all_seeds


# In[6]:


set_all_seeds(5)
## Set the device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

## Read the config
yml_path = '_quickstart/config_3d_cubehole.yml'
config = get_config_from_yml(yml_path)
config['device'] = device

print("DEVICE", device)
print("MODEL:", config['model'])
print("ACTIVATION", config.get('activation', None))

## Create the model and stateless functions and load a checkpoint
model = get_model(config)
netp = get_stateless_net_with_partials(model, use_x_and_z_arg=config['use_x_and_z_arg'])
# model.load_state_dict(torch.load('_quickstart/trained_model_3d.pt', map_location=device))

## Create different helpers for ...
## ... the problem definition
p_sampler = ProblemSampler(config)
## ... multiprocessing to create plots on a non-blocking thread
mp_manager = MPManager(config)
## ... recording timings
timer_helper = TimerHelper(config, lock=mp_manager.get_lock())
mp_manager.set_timer_helper(timer_helper)  ## weak circular reference
## ... plotting
plotter = Plotter3d(config)
## ... connectedness computation
scc_manager = SCCSurfaceNetManager(config, netp, mp_manager, plotter, timer_helper, p_sampler, device)
## ... sampling from the shape boundary
shapeb_helper = ShapeBoundaryHelper(config, netp, mp_manager, plotter, timer_helper, p_sampler.sample_from_interface()[0], device)
## ... clipping the gradients
auto_clip = AutoClip(config)


# In[3]:


z = torch.tensor([-0.1])
mesh_checkpoint = get_mesh_for_latent(netp.f_, netp.params_, z, config['bounds'], mc_resolution=128, device=device, chunks=1)

fig = k3d.plot()
fig += k3d.mesh(*mesh_checkpoint, color=0xff0000, side='double')
fig.display()
fig.camera_auto_fit = False
fig.camera = [0.8042741481976844,
            -1.040350835893895,
            0.7038650223301532,
            0.08252720725551285,
            -0.08146462547370059,
            -0.1973267630672968,
            -0.3986658507677483,
            0.39231188503442904,
            0.8289492893370278]


# In[7]:


from torch.utils.tensorboard import SummaryWriter
import datetime

ext = ""
if config['lambda_obst'] > 0:
    ext = "obst_" + str(config['lambda_obst']) + "_"
log_dir = "all_runs/runs_cubehole/" + ext + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
# opt.load_state_dict(torch.load('_quickstart/opt_3d.pt', map_location=device))
z = sample_new_z(config, is_init=True)

print(f'Initial z: {z}')
p_surface = None
cur_plot_epoch = 0
plots_dict_at_last_epoch = None
log_history_dict = {}

config['max_epochs'] = 100
prev_epoch = 0


# In[9]:


config['max_epochs'] = 50000
save_interval = 1000

for epoch in (pbar := trange(config['max_epochs'], leave=True, position=0, colour="yellow")):
    epoch += prev_epoch

    mp_manager.update_epoch(epoch)
    opt.zero_grad()
    
    # plotter.reset_output(p_sampler.recalc_output(netp.f_, netp.params_, z), epoch=epoch)
    # plotter.plot_shape(p_sampler.constr_pts_dict)
    
    loss_scc = torch.tensor(0.0)
    if config['lambda_scc'] > 0:
        success, res_tup = scc_manager.get_scc_pts_to_penalize(z, epoch)
        if success:
            p_penalize, p_penalties = res_tup
            print(f'penalize DCs with {len(p_penalize)} points')
            y_saddles_opt = model(p_penalize.data, p_penalize.z_in(z)).squeeze(1)
            loss_scc = config['lambda_scc'] *  (y_saddles_opt * p_penalties.data).mean()

    ## Design region loss                
    loss_env = torch.tensor(0.0)
    if config['lambda_env'] > 0:
        ys_env = model(*tensor_product_xz(p_sampler.sample_from_envelope(), z)).squeeze(1)
        loss_env = config['lambda_env'] * envelope_loss(ys_env)

    ## Interface loss
    loss_if = torch.tensor(0.0)
    if config['lambda_bc'] > 0:
        ys_BC = model(*tensor_product_xz(p_sampler.sample_from_interface()[0], z)).squeeze(1)
        loss_if = config['lambda_bc'] * interface_loss(ys_BC)
        
    ## Interface normal loss
    loss_if_normal = torch.tensor(0.0)
    if config['lambda_normal'] > 0:
        pts_normal, target_normal = p_sampler.sample_from_interface()
        ys_normal = netp.vf_x(*tensor_product_xz(pts_normal, z)).squeeze(1)
        loss_if_normal = config['lambda_normal'] * normal_loss_euclidean(ys_normal, torch.cat([target_normal for _ in range(config['batch_size'])]))

    ## Obstacle loss (for debugging purposes, it's not considered part of the envelope) TODO: do we leave it like this?
    loss_obst = torch.tensor(0.0)
    if config['lambda_obst'] > 0:
        ys_obst = model(*tensor_product_xz(p_sampler.sample_from_obstacles(), z))
        loss_obst = config['lambda_obst'] * obstacle_interior_loss(ys_obst)

    ## Sample points from the domain if necessary TODO: I think diversity doesnt need domain points anymore? TODO: can we move this up so that all the losses come after each other?
    if config['lambda_eikonal'] > 0 or config['lambda_div'] > 0:
        xs_domain = p_sampler.sample_from_domain()

    ## Eikonal loss    
    loss_eikonal = torch.tensor(0.0)
    if config['lambda_eikonal'] > 0:
        y_x_eikonal = netp.vf_x(*tensor_product_xz(xs_domain, z))
        loss_eikonal = config['lambda_eikonal'] * eikonal_loss(y_x_eikonal)

    ## Sample points from the 0-levelset if necessary TODO: can we move this up?
    if config['lambda_div'] > 0 or config['lambda_curv'] > 0:
        if p_surface is None or epoch % config['recompute_surface_pts_every_n_epochs'] == 0:
            p_surface, weights_surf_pts = shapeb_helper.get_surface_pts(z)
    
    ## Curvature loss
    loss_curv = torch.tensor(0.0)
    if config['lambda_curv'] > 0:
        if p_surface is None:
            print('No surface points found - skipping curvature loss')
        else:
            y_x_surf = netp.vf_x(p_surface.data, p_surface.z_in(z)).squeeze(1)
            y_xx_surf = netp.vf_xx(p_surface.data, p_surface.z_in(z)).squeeze(1)
            loss_curv = config['lambda_curv'] * strain_curvature_loss(y_x_surf, y_xx_surf, clip_max_value=config['strain_curvature_clip_max'],
                                                                            weights=weights_surf_pts)
    ## Diversity loss
    loss_div = torch.tensor(0.0)
    if config['lambda_div'] > 0 and config['batch_size'] > 1:
        if p_surface is None:
            print('No surface points found - skipping diversity loss')
        else:
            y_div = model(*tensor_product_xz(p_surface.data, z)).squeeze(1)  # [(bz k)] whereas k is n_surface_points; evaluate model at all surface points for each shape
            loss_div = config['lambda_div'] * closest_shape_diversity_loss(einops.rearrange(y_div, '(bz k)-> bz k', bz=config['batch_size']), 
                                                                                weights=weights_surf_pts)
            if torch.isnan(loss_div) or torch.isinf(loss_div):
                print(f'NaN or Inf loss_div: {loss_div}')
                loss_div = torch.tensor(0.0) if torch.isnan(loss_div) or torch.isinf(loss_div) else loss_div 

    loss = loss_env + loss_if + loss_if_normal + loss_obst + loss_eikonal + loss_scc + loss_curv + loss_div
    # print(f'loss_env: {loss_env}')
    # print(f'loss_if: {loss_if}')
    # print(f'loss_if_normal: {loss_if_normal}')
    # print(f'loss_obst: {loss_obst}')
    # print(f'loss_eikonal: {loss_eikonal}')
    # print(f'loss_scc: {loss_scc}')
    # print(f'loss_curv: {loss_curv}')
    # print(f'loss_div: {loss_div}')
    
    losses = {
        "loss_env": loss_env,
        "loss_if": loss_if,
        "loss_if_normal": loss_if_normal,
        "loss_obst": loss_obst,
        "loss_eikonal": loss_eikonal,
        "loss_scc": loss_scc,
        "loss_curv": loss_curv,
        "loss_div": loss_div,
        "loss": loss,
    }

    ## Gradients with clipping
    loss.backward()
    grad_norm = auto_clip.grad_norm(model.parameters())
    if auto_clip.grad_clip_enabled:
        auto_clip.update_gradient_norm_history(grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), auto_clip.get_clip_value())
        
    ## Update the parameters
    opt.step()

    for lname, l in losses.items():
        writer.add_scalar(f'Loss/{lname}', l, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    if (epoch % save_interval == 0 and epoch > 0) or (epoch - prev_epoch - 1) == config['max_epochs']:
        save_data = {
            # 'f_': netp.f_,              # Assuming this is a model or callable
            'params_': netp.params_,    # Model parameters or some state dict
            'z': z                      # Latent tensor
        }

        save_path_pts = os.path.join(log_dir, "pts")
        if not os.path.exists(save_path_pts):
            os.makedirs(save_path_pts)
        save_path = os.path.join(save_path_pts, f"nept_{epoch}.pt")  # Added .pt extension for convention
        torch.save(save_data, save_path)
    
    # Look at debugging plots
    # For this you have to enable plots in the config; note: this will slow down the training
    if mp_manager.are_plots_available_for_epoch(epoch):
        plots_dict_at_last_epoch = mp_manager.pop_plots_dict(epoch)

prev_epoch += config['max_epochs']


# In[ ]:


## NOTE: 8GB is not enough CUDA memory to perform marching cubes after training. Maybe we release some tensors? Alt: I would love to understand what these tensors
## TODO: smaller update?

z = torch.tensor([-0.1])
mesh_update = get_mesh_for_latent(netp.f_, netp.params_, z, config['bounds'], mc_resolution=128, device=device, chunks=1)

fig = k3d.plot()
# fig += k3d.mesh(*mesh_checkpoint, color=0xff0000, side='double', opacity=0.5, name='Original shape')
fig += k3d.mesh(*mesh_update, color=0x00ff00, side='double', opacity=0.5, name='Updated shape')
fig.display()
fig.camera_auto_fit = False
fig.camera = [0.8042741481976844,
            -1.040350835893895,
            0.7038650223301532,
            0.08252720725551285,
            -0.08146462547370059,
            -0.1973267630672968,
            -0.3986658507677483,
            0.39231188503442904,
            0.8289492893370278]

