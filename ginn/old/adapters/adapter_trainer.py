import torch
import torch.nn as nn

from GINN.problem_sampler import ProblemSampler
from train.train_utils.latent_sampler import sample_new_z
from utils import get_stateless_net_with_partials, get_model
from neural_clbf.controllers.simple_neural_cbf_controller import SimpleNeuralCBFController
from neural_clbf.systems.simple3d import Simple3DRobot
from configs.get_adapter_config import build_config
from configs.get_config import get_config_from_yml
from models.model_utils import tensor_product_xz
from train.train_utils.loss_optims import GradNormBalancer
from torch.utils.tensorboard import SummaryWriter

import subprocess
import time
from datetime import datetime
import os
from copy import deepcopy
from tqdm import trange
from collections import defaultdict
import yaml

from adapter_model import LossTimer, ConditionalSIRENWithAdapter, AdapterMLP, create_adapter_mlp

def save_dict_to_yaml(data: dict, filename: str):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
        print("Saving config to", filename)

if __name__ == "__main__":
    config = build_config()
    device=config["training"]["device"]

    # Create model save path
    model_save_path = os.path.join(config["paths"]["model_save_path"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    config["paths"]["model_save_path"] = model_save_path
    os.makedirs(model_save_path, exist_ok=True)

    # Create TensorBoard log directory
    log_dir = os.path.join(config["paths"]["tensorboard_log_dir"], f"tensorboard_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir)
    cmd = " ".join(["tensorboard", "--logdir", log_dir, "--port", str(config["paths"]["tensorboard_port"])])
    # tensorboard_process = subprocess.Popen(
    #     cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    # )

    print("Run to see results:\n" + cmd)
    print(f"TensorBoard will run at: http://localhost:{config['paths']['tensorboard_port']}/")

    # Load and initialize models
    config_siren = get_config_from_yml(config["paths"]["siren_config_path"])
    config_siren["device"] = device

    siren_model = get_model(config_siren).to(device)
    siren_model.load_state_dict(torch.load(config["paths"]["pretrained_siren_path"], map_location=device))

    final_layer_size = list(siren_model.network.children())[-3].out_features
    layer_sizes = [final_layer_size] + config["training"]["adapter_mid_layers"]
    activation_name = config["training"]["activation_name"]

    adapter_model = create_adapter_mlp(layer_sizes, activation_name=activation_name, siren_config=config_siren).to(device)
    model = ConditionalSIRENWithAdapter(siren_model, adapter_model).to(device)
    opt = torch.optim.Adam(model.adapter.parameters(), lr=1e-3)

    p_sampler = ProblemSampler(config_siren)
    z = sample_new_z(config_siren, is_init=True).to(device)
    # netp = get_stateless_net_with_partials(siren_model, use_x_and_z_arg=True)

    dynamics_model = Simple3DRobot(
        {},  # nominal_params
        dt=config["simulation"]["simulation_dt"],
        controller_dt=config["simulation"]["controller_period"],
        scenarios=[{}]
    )

    cbf_controller = SimpleNeuralCBFController(
        dynamics_model,
        [{}],  # scenarios
        model,
        cbf_lambda=config["training"]["cbf_lambda"],
        cbf_relaxation_penalty=config["training"]["cbf_relaxation_penalty"],
        z=z,
        device=device
    )

    save_dict_to_yaml(config, os.path.join(config["paths"]["model_save_path"], "config.yml"))

    loss_timer = LossTimer()
    prev_loss = 1
    best_epoch, best_model = 1, None
    min_loss = 100

    # Loss Balancer
    if config['training']['loss_balancer_model'] == "gradnorm":
        loss_balancer = GradNormBalancer(num_losses=3).to(device) if config["training"]["loss_balancer_model"] == "gradnorm" else None
        print("loss_balancer range:", loss_balancer.max_weight, loss_balancer.min_weight)
    elif config['training']['loss_balancer_model'] == "fixed":
        loss_balancer = None
    else:
        raise ValueError("loss_balancer_model not recognized:", config['training']['loss_balancer_model'])

    for epoch in trange(config["training"]["max_epochs"], leave=True, position=0, colour="yellow"):
        opt.zero_grad()
        cbf_controller.set_V_nn(model)

        # Reconstruction Loss
        loss_timer.start("Reconstruction Loss")
        recon_points = {
            "interface": p_sampler.sample_from_interface()[0],
            "domain": p_sampler.sample_from_domain(),
            "outer": p_sampler.sample_from_outer()
        }
        recon_losses = {}
        for point_name, points in recon_points.items():
            siren_ys = siren_model(*tensor_product_xz(points, z)).squeeze(1)
            my_ys = model(*tensor_product_xz(points, z)).squeeze(1)
            if point_name == "outer":
                recon_losses[point_name] = torch.clamp(-my_ys + config['training']['min_outer_val'], min=0).norm().mean()
            elif point_name == "domain":
                recon_losses[point_name] = torch.clamp(my_ys - config['training']['max_domain_val'], min=0).norm().mean()
            else:
                recon_losses[point_name] = (siren_ys - my_ys).square().mean()
        
        loss_timer.stop("Reconstruction Loss")

        # Lie Derivative Norm Loss
        descent_points = {
            "interface": p_sampler.sample_for_descent_interface(),
            "on_env": p_sampler.sample_for_descent_on_env(),
            "env": p_sampler.sample_for_descent_env()
        }
        descent_losses = {}
        lie_norm_losses = {}
        small_control_losses = {}

        for point_name, (xs_start, u_refs) in descent_points.items():
            # Lie Derivative
            loss_timer.start("Lie Derivative Norm Loss")
            xs_start, u_refs = p_sampler.sample_for_descent()
            Lf_V, Lg_V = cbf_controller.V_lie_derivatives(xs_start)
            lie_norm_loss = Lf_V.square().mean() + Lg_V.square().mean()
            lie_norm_losses[point_name] = lie_norm_loss
            loss_timer.stop("Lie Derivative Norm Loss")

            # Descent Loss
            loss_timer.start("Descent Loss")
            loss_descent = torch.tensor(0.0, device=device)
            norms = torch.norm(u_refs, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            u_refs = u_refs / norms * config['training']['control_norm']
            losses_list, u_opt = cbf_controller.descent_loss(xs_start, u_ref=u_refs, get_us=True)
            loss_values = torch.stack([torch.clamp(l, min=0) for _, l in losses_list if not l.isnan()], dim=0)
            if loss_values.numel() > 0:
                loss_descent = loss_values.mean()
            descent_losses[point_name] = loss_descent
            loss_timer.stop("Descent Loss")

            # Small Control Loss
            loss_timer.start("Small Control Loss")
            loss_small_control = torch.tensor(0.0, device=device)
            u_norm = torch.norm(u_opt, p=2, dim=1)
            loss_small_controls = torch.clamp(config["training"]["min_control_norm"] - u_norm, min=0)
            loss_small_control = loss_small_controls.mean()
            small_control_losses[point_name] = loss_small_control
            loss_timer.stop("Small Control Loss")

        # Loss Balancer
        loss_timer.start("Loss Balancer Computation")
        losses = torch.cat(
            [ll.unsqueeze(0) for ll in lie_norm_losses.values()] + 
            [dl.unsqueeze(0) for dl in descent_losses.values()] + 
            [sl.unsqueeze(0) for sl in small_control_losses.values()] + 
            [rl.unsqueeze(0) for rl in recon_losses.values()]
        )

        # TODO: Fix
        if loss_balancer:
            lambdas = [l.item() for l in loss_balancer.loss_weights]
            loss = loss_balancer(losses, model.adapter.parameters())
            for i, lam in enumerate(lambdas):
                writer.add_scalar(f"Lambda/lambda_{i}", lam, epoch)
        else:
            lambdas = torch.tensor(
                    [config["training"]["lambda_descent"] / len(descent_losses)] * len(descent_losses) +
                    [config["training"]["lambda_lie_norm"] / len(lie_norm_losses)] * len(lie_norm_losses) +
                    [config["training"]["lambda_small_control"] / len(small_control_losses)] * len(small_control_losses) +
                    [config["training"]["lambda_recon"] / len(recon_points)] * len(recon_points)
                ).to(device)
            loss = (losses * lambdas).sum()

        loss_timer.stop("Loss Balancer Computation")

        loss.backward()
        opt.step()

        loss.backward()
        if config["training"]["grad_clipping_on"]:
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=config["training"]["grad_clip"])  # Clipping step
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach(), 2) for p in model.adapter.parameters() if p.grad is not None
            ]))
            writer.add_scalar("Grad Norm", grad_norm.item(), epoch)

        opt.step()

        # Logging losses
        # writer.add_scalar("Loss/Descent", loss_descent.item(), epoch)
        # writer.add_scalar("Loss/Small_Control", loss_small_control.item(), epoch)
        # writer.add_scalar("Loss/Lie_Norm", lie_norm_loss.item(), epoch)
        for loss_name, loss_val in descent_losses.items():
            writer.add_scalar(f"Loss/Descent/{loss_name}", loss_val.item(), epoch)
        writer.add_scalar("Loss/Descent", sum([dl.item() for _, dl in descent_losses.items()]), epoch)

        for loss_name, loss_val in lie_norm_losses.items():
            writer.add_scalar(f"Loss/Lie_Norm/{loss_name}", loss_val.item(), epoch)
        writer.add_scalar("Loss/Lie_Norm", sum([ll.item() for _, ll in lie_norm_losses.items()]), epoch)

        for loss_name, loss_val in small_control_losses.items():
            writer.add_scalar(f"Loss/Small_Control/{loss_name}", loss_val.item(), epoch)
        writer.add_scalar("Loss/Small_Control", sum([sl.item() for _, sl in small_control_losses.items()]), epoch)

        for loss_name, loss_val in recon_losses.items():
            writer.add_scalar(f"Loss/Reconstruction/{loss_name}", loss_val.item(), epoch)
        writer.add_scalar("Loss/Reconstruction", sum([lr.item() for _, lr in recon_losses.items()]), epoch)

        writer.add_scalar("Loss/Total", loss.item(), epoch)
        
        # Compute and print the average weight and bias value for each layer in the adapter model
        for name, param in model.adapter.named_parameters():
            if "weight" in name:  # Only consider weight tensors
                avg_weight = param.data.mean().item()
                writer.add_scalar(name, avg_weight, epoch)
            elif "bias" in name:  # Only consider bias tensors
                avg_bias = param.data.mean().item()
                writer.add_scalar(name, avg_bias, epoch)


        # Save model at intervals
        if epoch % config["training"]["save_n_epochs"] == 0 and epoch > 1:
            print("==============================================")
            savename = os.path.join(model_save_path, f"model_{epoch}.pth")
            torch.save(model.state_dict(), savename)
            print("Epoch:", epoch)
            print("Best Epoch", best_epoch)
            print("Saving...", savename)
            print("Loss", loss.item(), "Losses", [l.item() for l in losses])
            print("Lambdas:", lambdas)
            loss_timer.print_summary()
        
        # Save best model
        if loss < min_loss:
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())
            min_loss = loss

            savename = os.path.join(model_save_path, f"model_best.pth")
            torch.save(model.state_dict(), savename)
            print("New best", epoch)

        prev_loss = loss

    # Save the best model
    savename = os.path.join(model_save_path, f"model_best_{best_epoch}.pth")
    torch.save(best_model, savename)
    print("Best epoch:", best_epoch)