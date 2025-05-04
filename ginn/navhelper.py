from vis_utils import PointsManager
from scipy.spatial import cKDTree
import networkx as nx
import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch

from cbf import CBFModel
from neural_clbf.controllers.simple_neural_cbf_controller import SimpleNeuralCBFController
from neural_clbf.systems.simple3d import Simple3DRobot

def is_line_safe(p1, p2, points_manager, step=0.05, threshold=0.05, sdf=None, zs=None):
    """
    p1, p2: torch.Tensor of shape (3,)
    """
    direction = p2 - p1
    dist = torch.norm(direction).item()
    if dist == 0:
        return True

    direction = direction / dist
    num_steps = int(dist / step)

    for i in range(num_steps + 1):
        pt = p1 + i * step * direction

        if sdf is not None:
            pt = pt.unsqueeze(0).float()
            if sdf(pt, zs) <= 0:
                return False

        if points_manager.get_closest_distance(pt.detach().cpu().numpy()) < threshold:
            return False

    return True

def get_prm_samples(points_manager, start, goal, num_samples=1500, collision_threshold=0.05, z_range=0.5, sdf=None, zs=None, device='cpu'):
    all_points = torch.from_numpy(points_manager.get_all_points())
    bounds_min = all_points.min(dim=0).values
    bounds_max = all_points.max(dim=0).values

    samples = [start, goal]
    while len(samples) < num_samples:
        p = torch.rand(3) * (bounds_max - bounds_min) + bounds_min
        p[2] = torch.rand(1) * z_range + start[2]

        if sdf is not None:
            pt = p.unsqueeze(0).float().to(device)
            if sdf(pt, zs.to(device)) > 0.1:
                samples.append(p)

        elif sdf is None and points_manager.get_closest_distance(p) > collision_threshold:
            samples.append(p)

    return samples

def build_prm(points_manager, start, goal, num_samples=1500, neighbor_radius=1.0,
              collision_threshold=0.05, z_range=0, sdf=None, zs=None, device='cpu', samples=None):
    """
    start, goal: torch.Tensor of shape (3,)
    """

    if not samples:
        samples = get_prm_samples(points_manager, start, goal, num_samples, collision_threshold, z_range, sdf, zs, device)
    samples_tensor = torch.stack(samples)  # shape (N, 3)
    samples_np = samples_tensor.detach().cpu().numpy()  # for KDTree / networkx

    tree = cKDTree(samples_np)
    graph = nx.Graph()

    for i, p in enumerate(samples_tensor):
        graph.add_node(i, pos=p)
        dists, indices = tree.query(samples_np[i], k=15)
        for j in indices:
            if i == j:
                continue
            q = samples_tensor[j]
            if is_line_safe(p, q, points_manager, threshold=collision_threshold, sdf=sdf, zs=zs):
                weight = torch.norm(p - q).item()
                graph.add_edge(i, j, weight=weight)

    return graph, samples_tensor

def find_path(graph, samples, start, goal, max_waypoints=None):
    start_idx, goal_idx = 0, 1
    try:
        path_indices = nx.shortest_path(graph, source=start_idx, target=goal_idx, weight='weight')
        path = samples[path_indices]  # tensor slicing

        if max_waypoints is not None and len(path) > max_waypoints:
            idxs = torch.linspace(0, len(path) - 1, steps=max_waypoints).long()
            path = path[idxs]

        return path
    except nx.NetworkXNoPath:
        return None

def generate_safe_path(points_manager, start, goal, num_samples=500, threshold=0.05,
                       max_waypoints=None, sdf=None, zs=None, device='cpu', point_samples=None):

    if not point_samples is None:
        point_samples = get_prm_samples(
            points_manager, start, goal,
            num_samples=num_samples,
            collision_threshold=threshold,
            z_range=1e-10,
            sdf=sdf,
            zs=zs,
            device=device
        )

    graph, samples = build_prm(
        points_manager, start, goal,
        num_samples=num_samples,
        collision_threshold=threshold,
        sdf=sdf,
        zs=zs,
        device=device,
        samples=point_samples
    )

    path = find_path(graph, samples, start, goal, max_waypoints=max_waypoints)
    return path

def plan_single_pair(point_cloud, pm, plane, min_pcd_dist, max_waypoints, threshold, model, min_dist_range=[3, 5]):
    while True:
        num_pairs_left = num_pairs
        test_traj_pairs = []
        while num_pairs_left > 0:
            test_traj_pairs_tmp = sample_point_pairs(point_cloud, num_pairs=num_pairs_left, num_samples=750, min_pcd_dist=min_pcd_dist, 
                min_dist_range=min_dist_range, plane=plane, max_iterations=5000, reverse=True, plot=None, corner_scale=0.8)

            test_traj_pairs.append(test_traj_pairs_tmp)
            break

            start_positions = torch.tensor(test_traj_pairs_tmp[:, 0, :], dtype=torch.float32)
            goal_positions = torch.tensor(test_traj_pairs_tmp[:, 1, :], dtype=torch.float32)

            z_inputs = torch.full((start_positions.shape[0], 1), z_div.item())
            V_starts = model(start_positions, z_inputs)
            V_goals = model(goal_positions, z_inputs)
            mask = ((V_starts > 0.01) & (V_goals > 0.01))
            if len(mask.shape) > 1:
                mask = mask.squeeze(1)
            test_traj_pairs_tmp = test_traj_pairs_tmp[mask.cpu().numpy()]

            test_traj_pairs.append(test_traj_pairs_tmp)
            num_pairs_left -= len(test_traj_pairs_tmp)
            print("Left:", num_pairs_left)

        traj_pairs = np.vstack(test_traj_pairs)
        start_positions = traj_pairs[:, 0, :]
        goal_positions = traj_pairs[:, 1, :]

        start = torch.tensor(start_positions[0], dtype=torch.float32)
        goal = torch.tensor(goal_positions[0], dtype=torch.float32)

        waypoints = generate_safe_path(
            pm, start, goal,
            num_samples=2000,
            max_waypoints=max_waypoints,
            threshold=threshold
        )

        if waypoints is not None:
            return traj_pairs, waypoints
