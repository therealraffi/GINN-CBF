import time
import einops
import torch
from tqdm import trange
import logging
import time
import copy
import numpy as np
import os
import io
from PIL import Image

import wandb
from GINN.morse.scc_surfacenet_manager import SCCSurfaceNetManager
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from train.losses import closest_shape_diversity_loss, interface_loss, eikonal_loss, envelope_loss, normal_loss_euclidean, obstacle_interior_loss, strain_curvature_loss, domain_loss, outer_loss, uniformity_loss, total_variation_loss, boundary_loss, soft_margin_loss
from train.train_utils.autoclip import AutoClip
from train.train_utils.loss_optims import LossBalancer, GradNormBalancer
from models.model_utils import tensor_product_xz

from GINN.visualize.plotter_2d import Plotter2d
from GINN.visualize.plotter_3d import Plotter3d
from GINN.problem_sampler import ProblemSampler
from GINN.helpers.timer_helper import TimerHelper
from train.train_utils.latent_sampler import sample_new_z
from utils import get_stateless_net_with_partials, set_and_true

from neural_clbf.controllers.simple_neural_cbf_controller import SimpleNeuralCBFController
from neural_clbf.systems.simple3d import Simple3DRobot 

import matplotlib.pyplot as plt 
import torch.nn.functional as F

class Trainer():
    
    def __init__(self, config, model, mp_manager) -> None:
        self.config = config
        self.mpm = mp_manager
        self.logger = logging.getLogger('trainer')
        self.device = config['device']
        self.model = model.to(self.device)
        self.netp = get_stateless_net_with_partials(self.model, use_x_and_z_arg=self.config['use_x_and_z_arg'])
        
        print("Using dataset:", self.config['dataset_dir'])
        self.p_sampler = ProblemSampler(self.config)

        torch.set_default_device(self.device)

        self.timer_helper = TimerHelper(self.config, lock=mp_manager.get_lock())
        self.plotter = Plotter2d(self.config) if config['nx']==2 else Plotter3d(self.config)
        self.mpm.set_timer_helper(self.timer_helper)  ## weak circular reference
        self.scc_manager = SCCSurfaceNetManager(self.config, self.netp, self.mpm, self.plotter, self.timer_helper, self.p_sampler, self.device)
        self.shape_boundary_helper = ShapeBoundaryHelper(self.config, self.netp,self.mpm, self.plotter, self.timer_helper, 
                                                         self.p_sampler.sample_from_interface()[0], self.device)
        self.auto_clip = AutoClip(config)

        self.p_surface = None

    def train(self):
        ###########
        # z = sample_new_z(self.config, is_init=True).to(self.device)
        # print(z)
        z = torch.tensor([[1.0]])
        self.logger.info(f'Initial z: {z}')

        controller_period = 0.05
        simulation_dt = 0.01
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        torch.set_default_device('cpu')
        # Define the dynamics model
        dynamics_model = Simple3DRobot(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
        )

        cbf_controller = SimpleNeuralCBFController(
            dynamics_model,
            [{}],  # scenarios
            self.model,
            cbf_lambda=self.config["cbf_lambda"],
            cbf_relaxation_penalty=self.config["cbf_relaxation_penalty"],
            z=z,
            device=self.device
        )

        torch.set_default_device(self.device)

        ############## ############## ##############
        self.num_losses = 8
        loss_optim_mode = self.config["loss_optim"]
        if loss_optim_mode == "loss_balancer":
            loss_balancer = LossBalancer(self.num_losses).to(self.device)
            opt = torch.optim.Adam(
                list(self.model.parameters()) + list(loss_balancer.parameters()), 
                lr=self.config['lr']
            )
        elif loss_optim_mode == "gradnorm":
            
            # losses = torch.stack([
            #     loss_if, loss_if_normal, loss_eikonal, loss_scc,
            #     loss_curv, loss_dom, loss_descent, loss_small_control
            # ])

            initial_weights = torch.tensor([1, 0.1, 1.0e-9, 1.0e-2, 1.0e-8, 0.4, 0.1, 1.0e-6]).to(self.device)
            loss_balancer = GradNormBalancer(self.num_losses, weights=initial_weights).to(self.device)
            opt = torch.optim.Adam(
                list(self.model.parameters()) + list(loss_balancer.parameters()), 
                lr=self.config['lr']
            )
        else:
            loss_balancer = None
            opt = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        ############## ############## ##############

        if set_and_true('use_scheduler', self.config):
            def warm_and_decay_lr_scheduler(step: int):
                return self.config['scheduler_gamma'] ** (step / self.config['decay_steps'])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warm_and_decay_lr_scheduler)
        
        cur_plot_epoch = 0
        log_history_dict = {}
        prev_lost = 1
        min_loss = 1e10
        best_epoch = 1
        best_model = None

        for epoch in (pbar := trange(self.config['max_epochs'], leave=True, position=0, colour="yellow")):
            self.mpm.update_epoch(epoch)
            local_timer_helper = TimerHelper(self.config, lock=self.mpm.get_lock())

            print("\n============= " + self.config["problem"] + "_" + str(epoch), " =============")

            opt.zero_grad()
            if 'reset_zlatents_every_n_epochs' in self.config and epoch % self.config['reset_zlatents_every_n_epochs'] == 0:
                z = sample_new_z(self.config)
            
            if self.plotter.do_plot():
                self.plotter.reset_output(self.p_sampler.recalc_output(self.netp.f_, self.netp.params_, z), epoch=epoch)
                self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=[self.p_sampler.constr_pts_dict], kwargs_dict={})
        
            ## Descent loss
            loss_descent_env = torch.tensor(0.0)
            loss_descent_inf = torch.tensor(0.0)
            loss_descents = [loss_descent_env, loss_descent_inf]
            loss_small_control = torch.tensor(0.0)
            if self.config['lambda_descent'] > 0 or self.config['lambda_small_control'] > 0:
                with local_timer_helper.record("Descent"):
                    cbf_controller.z = z.to(self.device)
                    cbf_controller.set_V_nn(self.model)

                    # # xs_start, xs_goal = self.p_sampler.sample_for_descent()
                    # # u_refs = torch.nn.functional.normalize(xs_goal - xs_start)
                    # random_steps = self.config["controller_step_range"][0] + (self.config["controller_step_range"][1] - self.config["controller_step_range"][0]) * torch.rand(self.config["n_controller_steps"])
                    # for step_size in random_steps:
                    #     xs_start, u_refs = self.p_sampler.sample_for_descent()
                    #     u_refs = (u_refs / u_refs.norm(dim=1, keepdim=True)) * step_size
                    #     # print(xs_start.shape, u_refs.shape)
                    #     losses_list, u_opt = cbf_controller.descent_loss(xs_start, u_ref=u_refs, get_us=True)
                    #     # print("Loss list:", losses_list)
                    #     c = 1
                    #     cur_loss = torch.tensor(0.0)
                    #     for _, l in losses_list:
                    #         if not l.isnan():
                    #             cur_loss += max(l, 0)
                    #             c += 1
                    #         else:
                    #             print("FAILED DESCENT LOSS:", losses_list)

                    #     loss_descent += (torch.abs(cur_loss) / c)

                    # if self.config['lambda_small_control'] > 0:
                    #     u_norm = torch.norm(u_opt, p=2, dim=1)  # Compute L2 norm along control dimension (NxK → N)
                    #     cur_loss = torch.clamp(self.config['min_control_norm'] - u_norm, min=0)
                    #     loss_small_control += cur_loss.mean()  # Taking mean ensures proper scaling
                    
                    # random_steps = self.config["controller_step_range"][0] + (self.config["controller_step_range"][1] - self.config["controller_step_range"][0]) * torch.rand(self.config["n_controller_steps"]) 

                    xs_start_env, u_refs_env = self.p_sampler.sample_for_descent_env()
                    xs_start_onenv, u_refs_onenv = self.p_sampler.sample_for_descent_on_env()
                    xs_e = torch.cat([xs_start_env, xs_start_onenv], dim=0)
                    u_refs_e = torch.cat([u_refs_env, u_refs_onenv], dim=0)

                    xi, u_refsi = self.p_sampler.sample_for_descent_interface()
                    pairs = [(xs_e, u_refs_e), (xi, u_refsi)]
                    
                    for i, (xs_start, u_refs) in enumerate(pairs):
                        step_sizes = torch.empty(u_refs.shape[0], 1,
                            device=u_refs.device,
                            dtype=u_refs.dtype).uniform_(self.config["controller_step_range"][0], self.config["controller_step_range"][1])

                        u_refs = (u_refs / u_refs.norm(dim=1, keepdim=True)) * step_sizes
                        grad_phi = self.netp.vf_x(*tensor_product_xz(xs_start, z))
                        phi_vals = self.model(*tensor_product_xz(xs_start, z)).squeeze(-1)  # [N]
                        
                        # grad_phi is [N, 1, 3], u_refs is [N, 3]
                        inner = torch.bmm(grad_phi, u_refs.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [N]
                        inner = inner.squeeze(-1).squeeze(-1)  # [N]
                        loss_descent_vals = F.relu(inner + self.config['cbf_lambda'] * phi_vals)
                        loss_descents[i] += loss_descent_vals.mean()

                # loss_descent_env = loss_descent_env / self.config["n_controller_steps"]
                # loss_descent_inf = loss_descent_inf / self.config["n_controller_steps"]
                # loss_small_control = loss_small_control / self.config["n_controller_steps"]

                # if loss_small_control.mean() > 0:
                #     tensor = loss_small_control.view(-1)
                #     top_k_values, _ = torch.topk(tensor, 5)
                #     print("loss_small_control", top_k_values)
                # print("Sum loss descent:", loss_descent)

            vfx_loss = torch.tensor(0.0)
            jac_loss = torch.tensor(0.0)
            vfxx_loss = torch.tensor(0.0)
            # if self.config['lambda_vfx'] > -1:
            #     with local_timer_helper.record("Lie Norm"):
            #         xs_start, u_refs = self.p_sampler.sample_for_descent()
            #         Lf_V, Lg_V = cbf_controller.V_lie_derivatives(xs_start.to(self.device))
            #         vfx_loss = Lf_V.square().mean() + Lg_V.square().mean()

                    # jac_outs = self.netp.vf_x(*tensor_product_xz(xs_start.to(self.device), z)).squeeze(1)
                    # jac_loss = jac_outs.square().mean()
                    # vfx_loss = jac_loss

            if self.config['lambda_vx'] > -1:
                with local_timer_helper.record("Vf_x"):
                    xs_e = self.p_sampler.sample_from_envelope()
                    xs_i = self.p_sampler.sample_from_interface()[0]
                    # print(xs_i, xs_e)
                    xs = torch.cat([xs_e, xs_i], dim=0)

                    vfx_out = self.netp.vf_x(*tensor_product_xz(xs.to(self.device), z)).squeeze(1)
                    vfx_loss = vfx_out.square().mean()

            if self.config['lambda_vxx'] > -1:
                with local_timer_helper.record("Vf_xx"):
                    xs_e = self.p_sampler.sample_from_envelope()
                    xs_i = self.p_sampler.sample_from_interface()[0]
                    xs = torch.cat([xs_e, xs_i], dim=0)

                    vfxx_out = self.netp.vf_xx(*tensor_product_xz(xs.to(self.device), z)).squeeze(1)
                    vfxx_loss = vfxx_out.square().mean()
                    
            loss_scc = torch.tensor(0.0)
            if self.config['lambda_scc'] > 0:
                print("start loss_scc")
                with local_timer_helper.record("SCC"):
                    with self.timer_helper.record('train.get_scc_pts_to_penalize'):
                        success, res_tup = self.scc_manager.get_scc_pts_to_penalize(z, epoch)
                    if success:
                        print("============= loss_scc success =============")
                        p_penalize, p_penalties = res_tup
                        self.logger.debug(f'penalize DCs with {len(p_penalize)} points')
                        y_saddles_opt = self.model(p_penalize.data, p_penalize.z_in(z)).squeeze(1)
                        loss_scc = (y_saddles_opt * p_penalties.data).mean()
                    else:
                        print("============= loss_scc FAIL =============")

            # loss_env = torch.tensor(0.0)
            # if self.config['lambda_env'] > 0:
            #     with local_timer_helper.record("Envelope"):        
            #         ys_env = self.model(*tensor_product_xz(self.p_sampler.sample_from_envelope(), z)).squeeze(1)
            #         loss_env = self.config['lambda_env'] * envelope_loss(ys_env)

            loss_if = torch.tensor(0.0)
            if self.config['lambda_bc'] > 0:
                with local_timer_helper.record("BC"):
                    ys_BC = self.model(*tensor_product_xz(self.p_sampler.sample_from_interface()[0], z)).squeeze(1)
                    # loss_if = interface_loss(ys_BC, self.config['interface_delta'])
                    loss_if = ys_BC.square().mean().sqrt()
                    loss_if = torch.tensor(0.0) if torch.isnan(loss_if) else loss_if

            loss_if_normal = torch.tensor(0.0)
            if self.config['lambda_normal'] > 0:
                with local_timer_helper.record("Normal"):
                    pts_normal, target_normal = self.p_sampler.sample_from_interface()
                    ys_normal = self.netp.vf_x(*tensor_product_xz(pts_normal, z)).squeeze(1)
                    loss_if_normal = normal_loss_euclidean(ys_normal, torch.cat([target_normal for _ in range(self.config['batch_size'])]))

            if self.config['lambda_eikonal'] > 0 or self.config['lambda_div'] > 0:
                xs_domain = self.p_sampler.sample_from_domain()
            
            loss_eikonal = torch.tensor(0.0)
            if self.config['lambda_eikonal'] > 0:
                with local_timer_helper.record("Eikonal"):
                    ## Eikonal loss: NN should have gradient norm 1 everywhere
                    y_x_eikonal = self.netp.vf_x(*tensor_product_xz(xs_domain, z))
                    loss_eikonal = eikonal_loss(y_x_eikonal)

            if self.config['lambda_div'] > 0 or self.config['lambda_curv'] > 0:
                if self.p_surface is None or epoch % self.config['recompute_surface_pts_every_n_epochs'] == 0:
                    self.p_surface, weights_surf_pts = self.shape_boundary_helper.get_surface_pts(z)
                
            loss_div = torch.tensor(0.0)
            if self.config['lambda_div'] > 0 and self.config['batch_size'] > 1:
                with local_timer_helper.record("Diversity"):
                    if self.p_surface is None:
                        self.logger.debug('No surface points found - skipping diversity loss')
                    else:
                        y_div = self.model(*tensor_product_xz(self.p_surface.data, z)).squeeze(1)  # [(bz k)] whereas k is n_surface_points; evaluate model at all surface points for each shape
                        loss_div = closest_shape_diversity_loss(einops.rearrange(y_div, '(bz k)-> bz k', bz=self.config['batch_size']), 
                                                                                            weights=weights_surf_pts)
                        if torch.isnan(loss_div) or torch.isinf(loss_div):
                            self.logger.warning(f'NaN or Inf loss_div: {loss_div}')
                            loss_div = torch.tensor(0.0) if torch.isnan(loss_div) or torch.isinf(loss_div) else loss_div 
                        
            loss_curv = torch.tensor(0.0)
            if self.config['lambda_curv'] > 0:
                with local_timer_helper.record("Curv"):
                    # self.p_surface, weights_surf_pts = self.shape_boundary_helper.get_surface_pts(z)
                    if self.p_surface is None:
                        self.logger.info('No surface points found - skipping curvature loss')
                    else:
                        y_x_surf = self.netp.vf_x(self.p_surface.data, self.p_surface.z_in(z)).squeeze(1)
                        y_xx_surf = self.netp.vf_xx(self.p_surface.data, self.p_surface.z_in(z)).squeeze(1)
                        loss_curv = strain_curvature_loss(y_x_surf, y_xx_surf, clip_max_value=self.config['strain_curvature_clip_max'],
                                                                                    weights=weights_surf_pts)
                        # print("loss_curv", loss_curv)

            loss_dom = torch.tensor(0.0)
            if self.config['lambda_dom'] > 0:
                with local_timer_helper.record("Domain"):
                    x_dom   = self.p_sampler.sample_from_domain()              # (B, 3)
                    ys_dom  = self.model(*tensor_product_xz(x_dom, z)).squeeze(1)   # (B,)
                    mask_dom           = (ys_dom > self.config['max_domain_val']).squeeze(-1)     # Bool mask
                    penalized_pts_dom  = x_dom[mask_dom]                            # offending xyz
                    passed_pts_dom  = x_dom[~mask_dom]                            # offending xyz
                    penalized_vals_dom = ys_dom[mask_dom]      
                    loss_dom = domain_loss(ys_dom, self.config['max_domain_val'])
                    print("domain (avg failed passed loss min_val):", ys_dom.mean().item(), penalized_pts_dom.shape[0], passed_pts_dom.shape[0], loss_dom.item(), torch.min(ys_dom).item())

            # loss_outer_env = torch.tensor(0.0)
            loss_env_inner = torch.tensor(0.0)
            loss_env_outer = torch.tensor(0.0)
            if self.config['lambda_outer_env'] > 0:
                with local_timer_helper.record("Outer Env"):
                    x_inner = self.p_sampler.sample_from_envelope_inner()          # (B, 3)
                    ys_inner = self.model(*tensor_product_xz(x_inner, z))          # (B, 1) or (B,)
                    mask_inner = (ys_inner < self.config['min_env_val_inner']).squeeze(-1)       # Bool mask
                    penalized_pts_inner  = x_inner[mask_inner]                     # offending xyz
                    passed_pts_inner  = x_inner[~mask_inner]                     # offending xyz
                    penalized_vals_inner = ys_inner[mask_inner]                    # their SDF values
                    # loss_env_inner = envelope_loss(ys_inner, self.config['min_env_val_inner'])
                    # print("env inner (avg failed passed loss max_val):", ys_inner.mean().item(), penalized_pts_inner.shape[0], passed_pts_inner.shape[0], loss_env_inner.item(), torch.max(ys_inner).item())
                    print("env inner (avg failed passed max_val):", ys_inner.mean().item(), penalized_pts_inner.shape[0], passed_pts_inner.shape[0], torch.max(ys_inner).item())

                    # x_outer = self.p_sampler.sample_from_envelope_outer()
                    # ys_outer = self.model(*tensor_product_xz(x_outer, z))
                    # mask_outer = (ys_outer < self.config['min_env_val_outer']).squeeze(-1)
                    # penalized_pts_outer  = x_outer[mask_outer]
                    # passed_pts_outer = x_outer[~mask_outer]
                    # penalized_vals_outer = ys_outer[mask_outer]
                    # # loss_env_outer = envelope_loss(ys_outer, self.config['min_env_val_outer'])
                    # # print("env outer (avg failed passed loss max_val):", ys_outer.mean().item(), penalized_pts_outer.shape[0], passed_pts_outer.shape[0], loss_env_outer.item(), torch.max(ys_outer).item())
                    # print("env outer (avg failed passed max_val):", ys_outer.mean().item(), penalized_pts_outer.shape[0], passed_pts_outer.shape[0], torch.max(ys_outer).item())

                    # loss_outer_env = envelope_loss(ys_outer)

                    # loss_outer_env = torch.clamp(-ys_outer + self.config['min_outer_val'], min=0).square().mean()
                    # loss_outer_env = soft_margin_loss(ys_outer)
                    # loss_outer_env += torch.clamp(ys_outer - self.config['max_outer_val'], min=0).square().mean()

                    # pull shapes back to [N,1]
                    xs_outer, outer_targets = self.p_sampler.sample_from_envelope_wtargets()
                    ot  = outer_targets.unsqueeze(1)                 # [N,1]
                    tol = ot.abs() * self.config['env_val_tol']       # [N,1]
                    ys_outer = self.model(*tensor_product_xz(xs_outer, z))  # [N,1]

                    # compute lower- and upper-bound violations
                    lower_violation = F.relu(  ot - ys_outer ).clamp_min(1e-6)           # positive when ys_outer < ot
                    upper_violation = F.relu( ys_outer - (ot + tol) )    # positive when ys_outer > ot + tol
                    norm = ot.abs().clamp_min(1e-6)

                    violation = torch.cat([lower_violation / norm,
                                        upper_violation / norm], dim=1)  # [N,2]
                    loss_outer_env = violation.pow(2).mean().sqrt()
                    loss_outer_env = torch.tensor(0.0) if torch.isnan(loss_outer_env) else loss_outer_env
                    mask_good = ((ys_outer >= ot) & (ys_outer <= (ot + tol))).squeeze(-1)  # [N]

                    penalized_pts_outer = xs_outer[~mask_good]
                    passed_pts_outer    = xs_outer[ mask_good]

                    print("outer_targets", outer_targets.mean(), outer_targets.min(), outer_targets.max())
                    print("env outer (avg failed/passed/tol):",
                        ys_outer.mean().item(),
                        penalized_pts_outer.shape[0],
                        passed_pts_outer.shape[0],
                        loss_outer_env.item())
                    loss_env_outer = loss_outer_env

            ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######
            losses = torch.stack([
                loss_if, loss_if_normal, loss_eikonal, loss_scc,
                loss_curv, loss_dom, loss_descent_env, loss_descent_inf, loss_small_control, loss_env_inner, loss_env_outer, vfx_loss, 
                vfxx_loss
            ])

            if loss_optim_mode == "loss_balancer":
                assert self.num_losses == losses.shape[0]
                loss = loss_balancer(losses)
                learned_lambdas = torch.exp(-loss_balancer.log_vars).detach().cpu().numpy()
                print("lambdas", learned_lambdas)
            elif loss_optim_mode == "gradnorm":
                assert self.num_losses == losses.shape[0]
                loss = loss_balancer(losses, self.model.parameters())
                learned_lambdas = loss_balancer.loss_weights.detach().cpu().numpy()
                print("lambdas", learned_lambdas)
            else:  # Fixed lambdas (original behavior)
                lambdas = torch.tensor([
                    self.config['lambda_bc'], self.config['lambda_normal'],
                    self.config['lambda_eikonal'], self.config['lambda_scc'], self.config['lambda_curv'],
                    self.config['lambda_dom'], self.config['lambda_descent'], self.config['lambda_descent'],
                    self.config['lambda_small_control'], self.config['lambda_outer_env'], self.config['lambda_outer_env'],
                    self.config['lambda_vx'], self.config['lambda_vxx']
                ], device=self.device)
                loss = (lambdas * losses).sum()
            ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######

            ## gradients
            loss.backward()
            grad_norm = self.auto_clip.grad_norm(self.model.parameters())
            if self.auto_clip.grad_clip_enabled:
                self.auto_clip.update_gradient_norm_history(grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.auto_clip.get_clip_value())
                
            # take step
            opt.step()
            if set_and_true('use_scheduler', self.config):
                sched.step()
            
            ## Async Logging
            cur_log_dict = {
                'loss': loss,
                # 'loss_env': loss_env,
                'vfxx_loss': vfxx_loss,
                'loss_dom': loss_dom,
                'loss_descent_env': loss_descent_env,
                'loss_descent_inf': loss_descent_inf,
                'jac_loss': jac_loss,
                'vfx': vfx_loss,
                'loss_bc': loss_if,
                'loss_scc': loss_scc,
                'loss_eikonal': loss_eikonal,
                'loss_bc_normal': loss_if_normal,
                'loss_small_control': loss_small_control,
                # 'loss_div': loss_div,
                'loss_curv': loss_curv,
                'loss_env_inner': loss_env_inner,
                'loss_env_outer': loss_env_outer,
                'neg_loss_div': (-1) * loss_div,
                'grad_norm_pre_clip': grad_norm,
                'grad_norm_post_clip': self.auto_clip.get_last_gradient_norm(),
                'grad_clip': self.auto_clip.get_clip_value(),
                'lr': opt.param_groups[0]['lr'],
                'epoch': epoch
            }
            
            # Compute and print the average weight and bias value for each layer in the adapter model
            for name, param in self.model.named_parameters():
                if "weight" in name:  # Only consider weight tensors
                    avg_weight = param.data.mean().item()
                    cur_log_dict[name] = avg_weight
                elif "bias" in name:  # Only consider bias tensors
                    avg_bias = param.data.mean().item()
                    cur_log_dict[name] = avg_bias

            log_history_dict[epoch] = cur_log_dict
            cur_plot_epoch = self.log_to_wandb(cur_plot_epoch, log_history_dict, epoch)

            print("Current loss:", loss.item(), "Epoch:", epoch, "Delta (%):", (abs(loss - prev_lost) / loss).item() * 100)

            if (epoch % self.config['save_every_n_epochs'] == 0 or epoch >= self.config['max_epochs']) and not self.config['no_save']:
                if self.config['model_path'] is not None:
                    savename = self.config['model_path'].replace('.pth', f'_{epoch}.pth')
                    torch.save(self.model.state_dict(), savename)
                    print("Saving...", savename)

                else:
                    print("FAILED TO get SAVENAME!")
                    savename = self.config['model_save_path'].replace('.pth', f'_{epoch}.pth')
                    torch.save(self.model.state_dict(), savename)
                
                if penalized_pts_inner.numel() > 0:
                    inner_np = penalized_pts_inner.detach().cpu().numpy()
                    passed_inner_np = passed_pts_inner.detach().cpu().numpy()
                    np.save(
                        os.path.join(self.config['model_save_path'], "penalized_inner_pts.npy"),
                        inner_np
                    )
                    np.save(
                        os.path.join(self.config['model_save_path'], "passed_inner_pts.npy"),
                        passed_inner_np
                    )

                if penalized_pts_outer.numel() > 0:
                    outer_np = penalized_pts_outer.detach().cpu().numpy()
                    passed_outer_np = passed_pts_outer.detach().cpu().numpy()
                    np.save(
                        os.path.join(self.config['model_save_path'], "penalized_outer_pts.npy"),
                        outer_np
                    )
                    np.save(
                        os.path.join(self.config['model_save_path'], "passed_outer_pts.npy"),
                        passed_outer_np
                    )

                if penalized_pts_dom.numel() > 0:
                    dom_np = penalized_pts_dom.detach().cpu().numpy()
                    passed_dom_np = passed_pts_dom.detach().cpu().numpy()
                    np.save(
                        os.path.join(self.config['model_save_path'], "penalized_domain_pts.npy"),
                        dom_np
                    )
                    np.save(
                        os.path.join(self.config['model_save_path'], "passed_domain_pts.npy"),
                        passed_dom_np
                    )

            if loss < min_loss and not self.config['no_save']:
                best_epoch = epoch
                best_model = copy.deepcopy(self.model.state_dict())
                min_loss = loss

                savename = self.config['model_path'].replace('.pth', f'_best.pth')
                torch.save(self.model.state_dict(), savename)
                print("New best", epoch)

            pbar.set_description(f"{epoch}")
            # pbar.set_description(f"{epoch}: env:{loss_env.item():.1e} BC:{loss_if.item():.1e} obst:{loss_obst.item():.1e} eik:{loss_eikonal.item():.1e} cc:{loss_scc.item():.1e} div:{loss_div.item():.1e} curv:{loss_curv.item():.1e}")
            # print(torch.cuda.memory_summary())
            local_timer_helper.print_logbook()
            
            if abs(loss - prev_lost) / loss < self.config['loss_thresh'] and loss < self.config['min_loss_thresh']:
                print("Complete - Early Stop <", self.config['loss_thresh'])
                epoch = self.config['max_epochs']
                break

            prev_lost = loss

        savename = self.config['model_path'].replace('.pth', f'_best_{best_epoch}.pth')
        torch.save(best_model, savename)
        print("Best:", best_epoch)

        ## Finished
        self.timer_helper.print_logbook()

    def log_to_wandb(self, cur_plot_epoch, log_history_dict, epoch, await_all=False):
        with self.timer_helper.record('plot_helper.pool await async results'):
            ## Wait for plots to be ready, then log them
            while cur_plot_epoch <= epoch:
                if not self.mpm.are_plots_available_for_epoch(cur_plot_epoch):
                    ## no plots for this epoch; just log the current losses
                    wandb.log(log_history_dict[cur_plot_epoch])
                    del log_history_dict[cur_plot_epoch]
                    cur_plot_epoch += 1
                
                elif self.mpm.plots_ready_for_epoch(cur_plot_epoch):
                    # ## plots are available and ready
                    # print(log_history_dict[cur_plot_epoch] | self.mpm.pop_plots_dict(cur_plot_epoch))
                    # # wandb.log(log_history_dict[cur_plot_epoch] | self.mpm.pop_plots_dict(cur_plot_epoch))
                    # del log_history_dict[cur_plot_epoch]
                    # cur_plot_epoch += 1

                    plot_dict = self.mpm.pop_plots_dict(cur_plot_epoch)
                    for plot_name, plot_obj in plot_dict.items():
                        if isinstance(plot_obj, plt.Figure):  # Check if it's a Matplotlib figure
                            buf = io.BytesIO()
                            plot_obj.savefig(buf, format='png')  # Save plot to buffer as PNG
                            buf.seek(0)
                            image = Image.open(buf)
                            wandb.log({f"{plot_name}_plot": wandb.Image(image)})  # Log as an image

                    # Also log the usual metrics
                    wandb.log(log_history_dict[cur_plot_epoch])
                    del log_history_dict[cur_plot_epoch]
                    cur_plot_epoch += 1
                elif await_all or (epoch == self.config['max_epochs'] - 1):
                    ## plots are not ready yet - wait for them
                    self.logger.debug(f'Waiting for plots for epoch {cur_plot_epoch}')
                    time.sleep(1)
                else:
                    # print(f'Waiting for plots for epoch {cur_plot_epoch}')
                    break
        return cur_plot_epoch

    def test_plotting(self):
        """Minimal function to test plotting in 3D."""
        epoch = 0

        z = torch.eye(self.config['nz'])[:self.config['batch_size']]
        self.scc_manager.set_z(z)
        self.mpm.update_epoch(epoch)

        self.plotter.reset_output(self.p_sampler.recalc_output(self.f, self.params, z), epoch=epoch)      
        self.mpm.plot(self.plotter.plot_shape, 'plot_helper.plot_shape', arg_list=[None], kwargs_dict={})
        # self.mp_manager.plot(self.plot_helper.plot_shape, 'plot_helper.plot_shape', 
        #                     arg_list=[self.p_sampler.constr_pts_list], kwargs_dict={})
        


        ## Boiler for async wandb logging
        cur_plot_epoch = 0
        log_history_dict = {}
        ## Async Logging
        cur_log_dict = {
            'epoch': epoch,
            }
        log_history_dict[epoch] = cur_log_dict
        cur_plot_epoch = self.log_to_wandb(cur_plot_epoch, log_history_dict, epoch, await_all=True)
        
        