#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import time
import glob
import yaml
import argparse
import logging
import pathlib
from pathlib import Path
import struct
import warnings
from random import randint
import json

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import ConvexHull, Delaunay, cKDTree, KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import open3d as o3d
import trimesh
import k3d
import wandb

from plyfile import PlyData

from concurrent.futures import ThreadPoolExecutor

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[20]:


import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plyfile import PlyData
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import k3d
import trimesh
import random

# === Paths ===
data_root = "/scratch/rhm4nj/cral/datasets/Replica-Dataset/ReplicaSDK"
room_name = "room_0"
room_path = Path(data_root) / room_name
segment_floor = True

def rgb_to_uint(rgb):
    r, g, b = rgb
    return (int(r) << 16) + (int(g) << 8) + int(b)

def calculate_bounds(points):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]]), (np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max]))

def get_upper_xy_plane_points(point_cloud, bbox_min, bbox_max, z_tol=1e-3):
    z_max = bbox_max[2]
    return np.abs(point_cloud[:, 2] - z_max) < z_tol

def get_lower_xy_plane_points(point_cloud, bbox_min, bbox_max, z_tol=1e-3):
    z_min = bbox_min[2]
    return np.abs(point_cloud[:, 2] - z_min) < z_tol

def downsample_random_indices(points, K):
    indices = np.random.choice(points.shape[0], K, replace=False)
    return indices


# In[21]:


# === Load semantic metadata ===
ply_path = room_path / "habitat" / "mesh_semantic.ply"
info_path = room_path / "habitat" / "info_semantic.json"

with open(info_path, "r") as f:
    info = json.load(f)

class_mapping = {}
for ele in info["classes"]:
    class_mapping[ele["id"]] = ele["name"]
for ele in info["objects"]:
    class_mapping[ele["id"]] = ele["class_name"]

# === Load mesh ===
plydata = PlyData.read(str(ply_path))
vertex_array = np.stack([
    plydata['vertex']['x'],
    plydata['vertex']['y'],
    plydata['vertex']['z']
], axis=-1)

face_data = plydata['face'].data
face_indices = [f[0] for f in face_data]
object_ids = np.array([f[1] for f in face_data], dtype=np.uint16)

# Assign vertex labels
vertex_object_ids = np.zeros(vertex_array.shape[0], dtype=np.uint16)
used = np.zeros(vertex_array.shape[0], dtype=bool)
for face, oid in zip(face_indices, object_ids):
    for v in face:
        if not used[v]:
            vertex_object_ids[v] = oid
            used[v] = True


# In[22]:


room_points = vertex_array  # treat entire mesh as room

_, (bbox_min, bbox_max) = calculate_bounds(room_points)

# Identify ceiling and floor
ceiling_mask = get_upper_xy_plane_points(room_points, bbox_min, bbox_max, z_tol=0.45)
exclude_mask = ceiling_mask

if segment_floor:
    floor_mask = get_lower_xy_plane_points(room_points, bbox_min, bbox_max, z_tol=1e-1)
    exclude_mask = ceiling_mask | floor_mask

# Exclude those points by index
excluded_indices = np.where(exclude_mask)[0]
excluded_index_set = set(excluded_indices.tolist())

print(f"Excluding {len(excluded_index_set)} vertices")
excluded_points = room_points[exclude_mask]

# Create a plot
plot = k3d.plot()
plot += k3d.points(
    positions=room_points[~exclude_mask],
    point_size=0.01,
    color=0xff0000,  # red for excluded
    name="all points"
)
plot += k3d.points(
    positions=room_points[ceiling_mask].astype(np.float32),
    point_size=0.01,
    color=0x0000ff,  # blue = ceiling
    name="ceiling"
)

if segment_floor:
    plot += k3d.points(
        positions=room_points[floor_mask].astype(np.float32),
        point_size=0.01,
        color=0x00ff00,  # green = floor
        name="floor"
    )

plot.display()


# In[23]:


from sklearn.cluster import DBSCAN

MIN_POINTS = 40  # filter threshold
all_fragments = []

oid_to_faces = defaultdict(list)
for face, oid in zip(face_indices, object_ids):
    oid_to_faces[oid].append(face)

for oid, faces in tqdm(oid_to_faces.items(), desc="Extracting object fragments"):
    valid_faces = []
    for face in faces:
        if all(v in excluded_index_set for v in face):
            continue  # skip face on ceiling or floor
        valid_faces.append(face)

    if not valid_faces:
        continue

    faces = np.array(valid_faces)

    unique_vertex_indices, inverse_indices = np.unique(faces.flatten(), return_inverse=True)
    local_vertices = vertex_array[unique_vertex_indices]
    local_faces = inverse_indices.reshape(faces.shape)

    db = DBSCAN(eps=0.05, min_samples=5).fit(local_vertices)
    labels = db.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    class_name = class_mapping.get(oid, f"unknown_{oid}")

    for cluster_id in range(num_clusters):
        mask = (labels == cluster_id)
        num_pts = mask.sum()

        if num_pts < MIN_POINTS:
            continue

        cluster_points = local_vertices[mask]
        all_fragments.append({
            "points": cluster_points,
            "centroid": cluster_points.mean(axis=0),
            "oid": oid,
            "class_name": class_name,
            "num_points": num_pts
        })


# In[24]:


# === Print 5 smallest fragments ===
sorted_frags = sorted(all_fragments, key=lambda f: f["num_points"])
print("5 smallest fragments (after filtering):")
for frag in sorted_frags[:10]:
    print(f"- Class: {frag['class_name']}, Points: {frag['num_points']}")


# In[25]:


from scipy.spatial import KDTree, cKDTree
import networkx as nx
from scipy.spatial.distance import cdist

# Precompute bounding boxes
bbox_centers = []
bbox_extents = []
for frag in all_fragments:
    pts = frag["points"]
    bounds = np.stack([pts.min(axis=0), pts.max(axis=0)])
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    bbox_centers.append(center)
    bbox_extents.append(extent)

bbox_centers = np.stack(bbox_centers)
bbox_extents = np.stack(bbox_extents)

# Build KDTree over bounding box centers
tree = KDTree(bbox_centers)
touch_threshold = 0.02
G = nx.Graph()
G.add_nodes_from(range(len(all_fragments)))

print("Building contact graph (fast)...")

for i in tqdm(range(len(all_fragments))):
    # Search neighbors within radius (center distance + extent fudge factor)
    extent_radius = np.linalg.norm(bbox_extents[i]) / 2 + 1.0  # generous margin
    neighbors = tree.query_ball_point(bbox_centers[i], r=extent_radius)

    pi = all_fragments[i]["points"]
    pi_tree = cKDTree(pi)

    for j in neighbors:
        if j <= i:
            continue

        pj = all_fragments[j]["points"]
        dists, _ = pi_tree.query(pj, k=1)
        if np.any(dists < touch_threshold):
            G.add_edge(i, j)
            # print("Connecting edges", all_fragments[i]['class_name'], all_fragments[j]['class_name'], dists[dists < touch_threshold].min())

components = list(nx.connected_components(G))
super_clusters = []

for i, component in enumerate(components):
    component = list(component)
    fragment_names = [f"{all_fragments[idx]['class_name']}_frag{idx}" for idx in component]
    fragment_sizes = [all_fragments[idx]['points'].shape[0] for idx in component]

    # Find largest fragment
    largest_idx = component[np.argmax(fragment_sizes)]
    key_name = f"{all_fragments[largest_idx]['class_name']}"

    # Concatenate all points in the component
    merged_points = np.vstack([all_fragments[idx]["points"] for idx in component])
    super_clusters.append((key_name, merged_points))

room_points = vertex_array  # assuming this is the full mesh still
super_clusters.append(("ceiling", room_points[ceiling_mask]))
if segment_floor:
    super_clusters.append(("floor", room_points[floor_mask]))

print(f"Found {len(super_clusters)} superclusters")


# In[26]:


plot = k3d.plot()
color_map = {key: random.randint(0, 0xFFFFFF) for key in range(len(super_clusters))}

for i, (key, component) in enumerate(super_clusters):
    plot += k3d.points(
        positions=component.astype(np.float32),
        point_size=0.01,
        color=color_map[i],
        name=key
    )

plot.display()

print([(key, component.shape) for key, component in super_clusters])


# In[27]:


def average_nearest_neighbor_distance(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nn_distances = distances[:, 1]
    return np.mean(nn_distances), np.std(nn_distances)

def remove_outliers(point_cloud, method="statistical", nb_neighbors=25, std_ratio=1.5, radius=0.1, min_neighbors=5, ret_indices=False):
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    if method == "statistical":
        # Statistical Outlier Removal (SOR)
        clean_pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "radius":
        # Radius Outlier Removal (ROR)
        clean_pcd, inlier_indices = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        raise ValueError("Invalid method. Use 'statistical' or 'radius'.")

    if ret_indices:
        return np.asarray(clean_pcd.points), np.asarray(inlier_indices)

    return np.asarray(clean_pcd.points)

def filtered_point_cloud_indices(A, B, min_distance=0.1, tree=None, ret_dist=False):
    if tree is None:
        tree = cKDTree(A)
    distances, _ = tree.query(B)
    if ret_dist:
        return distances > min_distance, distances[distances > min_distance]
    
    return distances > min_distance

def sample_states_and_controls_timed(inner_point_cloud, point_cloud, N, K, min_dot=0.25):
    def sample_vector():
        vec = np.random.randn(3)
        return vec / np.linalg.norm(vec)

    idx = np.random.choice(point_cloud.shape[0], N, replace=False)
    sampled_points = point_cloud[idx]
    kdtree = cKDTree(inner_point_cloud)
    distances, nearest_indices = kdtree.query(sampled_points, k=2)
    nearest_neighbors = inner_point_cloud[nearest_indices[:, 1]]
    sampled_data = []

    for i in range(N):
        x, y, z = sampled_points[i]
        uc = nearest_neighbors[i] - sampled_points[i]
        uc = uc / np.linalg.norm(uc)
        sampled_data.append([x, y, z, uc[0], uc[1], uc[2]])
        for _ in range(K - 1):
            random_vector = sample_vector()
            while np.dot(random_vector, uc) < min_dot:
                random_vector = -random_vector
                if np.dot(random_vector, uc) < min_dot:
                    random_vector = sample_vector()
            sampled_data.append([x, y, z, random_vector[0], random_vector[1], random_vector[2]])

    return np.array(sampled_data)

def compute_interface_normals(interface_filtered, normal_radius, max_nn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(interface_filtered)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
    )
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)
    print("Interface / Normals", interface_filtered.shape, normals.shape)
    return normals

def compute_pts_on_env(domain_filtered, all_outers_filtered, interface_density, pts_on_env_thickness):
    ao_env = all_outers_filtered[
        filtered_point_cloud_indices(domain_filtered, all_outers_filtered, interface_density)
    ]
    pts_on_env = ao_env[
        ~filtered_point_cloud_indices(domain_filtered, ao_env, pts_on_env_thickness)
    ]
    print("Done with pts_on_env", pts_on_env.shape)
    return pts_on_env

def compute_outer_points(domain_filtered, all_outers_filtered, outer_density):
    outers = all_outers_filtered[
        filtered_point_cloud_indices(domain_filtered, all_outers_filtered, outer_density)
    ]
    print("Done with outers/envelope", outers.shape)
    return outers

from sklearn.decomposition import PCA

def compute_oriented_bbox(points, margin=0.0):
    center = points.mean(axis=0)
    centered = points - center

    pca = PCA(n_components=3)
    pca.fit(centered)
    rotation = pca.components_.T  # each column is an axis

    rotated = centered @ rotation
    min_corner = rotated.min(axis=0) - margin
    max_corner = rotated.max(axis=0) + margin

    # Get 8 corners in PCA space
    corners_pca = np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
    ])

    # Transform corners back to world space
    corners_world = (corners_pca @ rotation.T) + center
    return corners_world

def sample_inside_oriented_bbox(corners, num_points=1000):
    origin = corners[0]
    edge_x = corners[1] - corners[0]  # along x
    edge_y = corners[3] - corners[0]  # along y
    edge_z = corners[4] - corners[0]  # along z
    u = np.random.uniform(0, 1, (num_points, 1))
    v = np.random.uniform(0, 1, (num_points, 1))
    w = np.random.uniform(0, 1, (num_points, 1))
    samples = origin + u * edge_x + v * edge_y + w * edge_z
    return samples


# In[ ]:


augemented_points = []
class_pts = super_clusters

def thicken_point_cloud_outward(points, num_augmented=3, noise_scale=0.01):
    N, _ = points.shape
    centroid = points.mean(axis=0)
    expanded_points = [points]  # include original
    for _ in range(num_augmented):
        # Direction from centroid to each point
        directions = points - centroid
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)

        # Apply directional noise
        noise = directions * (np.random.rand(N, 1) * noise_scale)
        new_points = points + noise
        expanded_points.append(new_points)

    return np.vstack(expanded_points)


for cname, points in class_pts:
    print(cname, points.shape)
    # if cname != 'wall': continue
    indices = downsample_random_indices(points, min(points.shape[0], 300_000))
    points = np.ascontiguousarray(points, dtype=np.float64)
    points = points[indices]

    interface_thickness=0.05
    pts_on_env_thickness = 0.05
    pts_on_env_gap = 0.075
    inner_thickness=0.025
    normal_radius=0.025
    max_nn=30

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
    )
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)

    # outer_extras = []
    # bounds, bounds_box = calculate_bounds(points)
    # widths = bounds[:, 1] - bounds[:, 0]  # [x_width, y_width, z_width]
    # outer_extras_thickness = np.max(widths)
    # for step in np.linspace(1e-1, outer_extras_thickness, 30):
    #     sampled_indices = 
    # 
    # 
    # 
    # (points, min(points.shape[0], 20_000))
    #     points_tmp = points[sampled_indices]
    #     normals_tmp = normals[sampled_indices]
        
    #     outer_extras.append(points_tmp + normals_tmp  * step)
    #     outer_extras.append(points_tmp - normals_tmp  * step)
    # envelope = np.vstack(outer_extras)

    inners = []
    for step in np.linspace(1e-3, inner_thickness, 5):
        inners.append(points + normals * step)
        inners.append(points - normals * step)

    domain = np.vstack(inners)  # And the opposite direction
    inner_m, inner_std = average_nearest_neighbor_distance(domain)
    inner_density = inner_m + inner_std * 3

    # interface_thickness = inner_thickness + interface_thickness
    # outers = []
    # normals_interface = []
    # for step in np.linspace(interface_thickness, interface_thickness + 0.25 + inner_density, 10):
    #     outers.append(points + normals  * step)
    #     outers.append(points - normals  * step)
    # interface_inp = np.vstack(outers)
    # normals_interface = np.vstack([normals] * len(outers))

    # pts_on_env_thickness = interface_thickness + pts_on_env_thickness
    # pts_on_envs = []
    # for step in np.linspace(pts_on_env_thickness, pts_on_env_thickness + 0.4 + inner_density):
    #     pts_on_envs.append(points + normals  * step)
    #     pts_on_envs.append(points - normals  * step)
    # pts_on_env = np.vstack(pts_on_envs)

    min_neighbors = 8

    # inner layer
    domain_filtered = domain[downsample_random_indices(domain, min(points.shape[0] * 5, 400_000))]
    domain_filtered = remove_outliers(domain_filtered, radius=inner_density, min_neighbors=min_neighbors)
    inner_m, inner_std = average_nearest_neighbor_distance(domain_filtered)
    inner_density = inner_m + inner_std * 3 + 0.05
    print("Done with inner layer", domain_filtered.shape)

    # all points
    bbox_offset = 0.25
    n_all_outers = min(points.shape[0] * 10, 1_000_000)
    _, bounds_box = calculate_bounds(domain_filtered)
    expanded_bounds = bounds_box[0] - bbox_offset, bounds_box[1] + bbox_offset
    all_outers = np.random.uniform(low=expanded_bounds[0], high=expanded_bounds[1], size=(n_all_outers, 3))

    domain_tree = cKDTree(domain_filtered)
    all_outers_filtered = all_outers[
        filtered_point_cloud_indices(domain_filtered, all_outers, inner_density, tree=domain_tree)
    ]
    print("Done with all points", all_outers_filtered.shape)

    # interface
    interface_thickness = inner_density + interface_thickness
    # interface_inp = thicken_point_cloud_outward(domain_filtered, 10, noise_scale=0.75)
    # interface_inp = interface_inp[downsample_random_indices(interface_inp, min(interface_inp.shape[0] * 5, 400_000))]
    interface_filtered = all_outers_filtered[
        ~filtered_point_cloud_indices(domain_filtered, all_outers_filtered, interface_thickness)
    ]
    print("Done with interface", interface_filtered.shape)

    # Prepare variables for parallel work
    interface_density = interface_thickness
    pts_on_env_thickness = interface_density + pts_on_env_thickness
    outer_density = pts_on_env_thickness + inner_density

    # # Run them in parallel
    # with ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(compute_interface_normals, interface_filtered, normal_radius, max_nn),
    #         executor.submit(compute_pts_on_env, domain_filtered, all_outers_filtered, interface_density, pts_on_env_thickness),
    #         executor.submit(compute_outer_points, domain_filtered, all_outers_filtered, outer_density)
    #     ]
    # normals_filtered = compute_interface_normals(interface_filtered, normal_radius, max_nn)
    # pts_on_env_filtered = compute_pts_on_env(domain_filtered, all_outers_filtered, interface_density, pts_on_env_thickness)
    # envelope_filtered = compute_outer_points(domain_filtered, all_outers_filtered, outer_density)

    # Normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(interface_filtered)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
    )
    pcd.normalize_normals()
    normals_filtered = np.asarray(pcd.normals)
    print("Interface / Normals", interface_filtered.shape, normals_filtered.shape)

    interface_tree = cKDTree(interface_filtered)
    # pts_on_env
    ao_env = all_outers_filtered[
        filtered_point_cloud_indices(domain_filtered, all_outers_filtered, interface_density + pts_on_env_gap)
    ]
    pts_on_env_filtered = ao_env[
        ~filtered_point_cloud_indices(domain_filtered, ao_env, pts_on_env_thickness + pts_on_env_gap)
    ]
    pts_on_env_filtered_dists, _ = interface_tree.query(pts_on_env_filtered)
    print("Done with pts_on_env", pts_on_env_filtered.shape, pts_on_env_filtered_dists.shape)

    # envelope_filtered
    envelope_filtered = all_outers_filtered[
        filtered_point_cloud_indices(domain_filtered, all_outers_filtered, outer_density)
    ]
    envelope_filtered_dists, _ = interface_tree.query(envelope_filtered)
    print("Done with outers/envelope", envelope_filtered.shape, envelope_filtered_dists.shape)

    # control
    N = 1000
    K = 30
    control_outs_env = sample_states_and_controls_timed(domain_filtered, envelope_filtered, N // 3, K)
    control_outs_on_env = sample_states_and_controls_timed(domain_filtered, pts_on_env_filtered, N // 3, K)
    control_outs_interface = sample_states_and_controls_timed(domain_filtered, interface_filtered, N // 3, K)
    control_outs = np.vstack([control_outs_env, control_outs_interface, control_outs_on_env])
    control_points, controls = control_outs[:, :3], control_outs[:, 3:]
    print("Done with control", control_points.shape)

    # bounds, scaling, translation
    all_points = np.vstack([domain, interface_filtered, pts_on_env_filtered, envelope_filtered])
    bounds_og, bounds_coords = calculate_bounds(all_points)
    bbox_min, bbox_max = bounds_coords
    bounds = bounds_og.copy()

    all_points_obj = np.vstack([domain, interface_filtered])
    bounds_obj, _ = calculate_bounds(all_points_obj)

    center_for_translation = (bbox_max + bbox_min) / 2
    scale_factor = max(bbox_max - bbox_min) / 2

    print("Done with bounds, scaling, translation:", bounds_coords, scale_factor, center_for_translation)

    augemented_points.append({
        "class": cname,
        "pts_inside": domain_filtered,
        "env_outside_pts": envelope_filtered,
        "pts_on_env": pts_on_env_filtered,
        "pts_on_env_dists": pts_on_env_filtered_dists,
        "env_outside_pts_dists": envelope_filtered_dists,
        "outside_points_dists": envelope_filtered_dists,
        "outside_points": envelope_filtered,
        "control_points": control_outs,
        "control_points_on_env": control_outs_on_env,
        "control_points_env": control_outs_env,
        "control_points_interface": control_outs_interface,
        "original": points,
        "interface_pts": interface_filtered,
        "interface_normals": normals_filtered,
        "bounds": bounds,
        "bounds_obj": bounds_obj,
        "scale_factor": scale_factor,
        "center_for_translation": center_for_translation
    })


# In[ ]:


import k3d
import numpy as np
import random

plot = k3d.plot()

colors = {
    # "original": 0xff0000,             # Red
    "pts_inside": 0x00ff00,           # Green
    "interface_pts": 0x0000ff,        # Blue
    # "env_outside_pts": 0xffff00,      # Yellow
    "pts_on_env": 0xff00ff,           # Magenta
    # "outside_points": 0x00ffff,       # Cyan
    # "control_points": 0x808080,       # Gray
    # "control_points_on_env": 0xFFA500,# Orange
    # "control_points_env": 0x800080,   # Purple
    # "control_points_interface": 0x008000 # Dark Green
}

# print(domain.shape, domain_normals.shape)
# # plot += k3d.vectors(domain[:domain_normals.shape[0]], domain_normals, color=0x0000ff, head_size=0.1)  # Blue
# print(interface_filtered.shape, normals_filtered.shape)
# plot += k3d.vectors(interface_filtered, normals_filtered / 10, color=0xFFA500, head_size=0.25, line_width=0.001)  # Blue
# plot += k3d.points(interface_tmp, point_size=0.01, color=0x808080)
# plot += k3d.points(pts_on_env_tmp, point_size=0.01, color=0x008000)

# plot += k3d.points(bounds_obj.T, point_size=0.05, color=0x008000)
plot.display()
for i, data_dict in enumerate(augemented_points):
    print(f"\nPlotting object {i}: {data_dict.get('class', 'Unknown')}")
    if data_dict.get('class', 'Unknown') != 'wall':
        continue
    
    for key, value in data_dict.items():
        if not isinstance(value, np.ndarray):
            continue
        
        if value.ndim == 2 and value.shape[1] >= 3:
            pts = value[:, :3]
            col = colors.get(key, -1)
            if col != -1:
                plot += k3d.points(pts, point_size=0.05, color=col)
                print(f"{key} bounds: min {pts.min(axis=0)}, max {pts.max(axis=0)}")
                print("ADDED", key, pts.shape)

# plot += k3d.points(interface_tmp_big, point_size=0.01, color=0x008000)


# In[ ]:


import k3d
from k3d.colormaps import matplotlib_color_maps
import numpy as np

# Step 1: Concatenate points and distances
env_pts_all = []
env_dists_all = []
int_all = []

for data_dict in augemented_points:
    if not isinstance(data_dict, dict):
        continue
    if data_dict.get('class', 'Unknown') != 'wall':
        continue

    env_pts = data_dict.get("env_outside_pts")
    env_dists = data_dict.get("env_outside_pts_dists")
    on_env_pts = data_dict.get("pts_on_env")
    on_env_dists = data_dict.get("pts_on_env_dists")
    interface = data_dict.get("interface_pts")

    if env_pts is not None and env_dists is not None and env_pts.shape[0] == env_dists.shape[0]:
        env_pts_all.append(env_pts)
        env_dists_all.append(env_dists)

    if on_env_pts is not None and on_env_dists is not None and on_env_pts.shape[0] == on_env_dists.shape[0]:
        env_pts_all.append(on_env_pts)
        env_dists_all.append(on_env_dists)
    
    int_all.append(interface)

# Step 2: Stack arrays
if env_pts_all and env_dists_all:
    all_pts = np.vstack(env_pts_all).astype(np.float32)
    all_dists = np.concatenate(env_dists_all).astype(np.float32)
    int_all = np.vstack(int_all).astype(np.float32)

    # Step 3: Create a separate plot
    env_plot = k3d.plot()
    env_plot += k3d.points(
        all_pts,
        attribute=all_dists,
        point_size=0.05,
        color_map=matplotlib_color_maps.Inferno
    )
    env_plot += k3d.points(int_all, point_size=0.05, color=0x0000ff)
    env_plot.display()
else:
    print("No envelope or on-env points found.")


# In[ ]:


import shutil

if segment_floor:
    suffix = "_objects"
else:
    suffix = "_single"

name = room_name + suffix
out_path = os.path.join(
    "/scratch/rhm4nj/cral/cral-ginn/ginn/myvis/data_gen", 
    "replica",
    name
)
print("Saving to:", out_path)

if not os.path.exists(out_path):
    os.makedirs(out_path)
else:
    shutil.rmtree(out_path)

skips_names = []

for idx, values in enumerate(augemented_points):
    folder_name = f"{idx}_{values['class']}"
    folder_path = os.path.join(out_path, folder_name)
    os.makedirs(folder_path)

    for name, arrays in values.items():
        if name in skips_names: continue

        print(f'Saving to {folder_name}:', name)
        np.save(f'{folder_path}/{name}.npy', arrays)

