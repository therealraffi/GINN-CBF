# standard library
import os
import sys
import time
import random
from itertools import cycle

# scientific / data
import numpy as np
import torch
import trimesh
import networkx as nx

# plotting / widgets
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import k3d

# progress bars
from tqdm.notebook import tqdm  # or `from tqdm import tqdm` if you prefer the console version

# SciPy / sklearn
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from sklearn.decomposition import PCA

from train.train_utils.autoclip import AutoClip
from train.train_utils.latent_sampler import sample_new_z

from utils import get_model, get_stateless_net_with_partials, set_all_seeds, get_is_out_mask
from notebooks.notebook_utils import get_mesh_for_latent

from models.model_utils import tensor_product_xz
from cbf import CBFModel
from simple_cbf import SimpleCBFModel

from torch.func import functional_call, jacrev, jacfwd, vmap

class PointsManager:
    def __init__(self, points_dict):
        """
        Args:
            points_dict: dict[str, np.ndarray], each array is (N_i, 3)
        """
        self.original_points = {k: v.copy() for k, v in points_dict.items()}
        self.translations = {k: np.zeros(3) for k in points_dict}
        self._combined_tree = None
        self._combined_tree_needs_update = True

    def _get_translated_points(self, name):
        return self.original_points[name] + self.translations[name]

    def _rebuild_combined_tree(self):
        all_points = np.vstack([
            self._get_translated_points(name) for name in self.original_points
        ])
        self._combined_tree = cKDTree(all_points)
        self._combined_tree_needs_update = False

    def move(self, name, translation):
        """Move a point cloud by translation (x, y, z)."""
        if name not in self.translations:
            raise KeyError(f"No point cloud named '{name}'")
        self.translations[name] += np.asarray(translation)
        self._combined_tree_needs_update = True

    def get_all_points(self):
        """Return all translated points from all clouds as a single (N, 3) array."""
        return np.vstack([self.get_points(name) for name in self.original_points])

    def get_points(self, name):
        """Get the current (translated) points for a named point cloud."""
        if name not in self.original_points:
            raise KeyError(f"No point cloud named '{name}'")
        return self._get_translated_points(name)

    def reset(self):
        """Reset all point clouds to their original position."""
        for name in self.translations:
            self.translations[name] = np.zeros(3)
        self._combined_tree_needs_update = True

    def get_closest_distance(self, point):
        """
        Returns the distance to the closest point across all clouds using the fastest method.
        """
        point = np.asarray(point).reshape(1, 3)

        if self._combined_tree is None or self._combined_tree_needs_update:
            self._rebuild_combined_tree()

        dist, _ = self._combined_tree.query(point)
        return dist[0]

    def plot(self, plot=None, point_size=0.02, display=False):
        """Plot all point clouds with k3d, using different colors."""
        if not plot:
            plot = k3d.plot()
        color_cycle = cycle([
            0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff,
            0x00ffff, 0x888888, 0xff8800, 0x8800ff, 0x00ff88
        ])
        for name, pts in self.original_points.items():
            translated_pts = self.get_points(name)
            color = next(color_cycle)
            plot += k3d.points(translated_pts.astype(np.float32), color=color, point_size=point_size, name=name)
        if display:
            plot.display()
        return plot

class InteractivePlane:
    def __init__(self, point_on_plane, normal, plot, point_size=0.01, size=5, color=0x00FF00):
        self.size = size
        self.color = color
        self.plot = plot

        self.z_slider = widgets.FloatSlider(value=point_on_plane[2], min=-10, max=10, step=0.1, description="Z Height")
        self.nx_slider = widgets.FloatSlider(value=normal[0], min=-1, max=1, step=0.1, description="Normal X")
        self.ny_slider = widgets.FloatSlider(value=normal[1], min=-1, max=1, step=0.1, description="Normal Y")
        self.nz_slider = widgets.FloatSlider(value=normal[2], min=-1, max=1, step=0.1, description="Normal Z")

        self.z_slider.observe(self.update_plane, names="value")
        self.nx_slider.observe(self.update_plane, names="value")
        self.ny_slider.observe(self.update_plane, names="value")
        self.nz_slider.observe(self.update_plane, names="value")

        self.point_on_plane = np.array([point_on_plane[0], point_on_plane[1], self.z_slider.value])
        self.normal = np.array([self.nx_slider.value, self.ny_slider.value, self.nz_slider.value])
        self.mesh = self.create_plane()

        self.plot += self.mesh
        display(self.z_slider)

    def create_plane(self):
        normal = self.normal / np.linalg.norm(self.normal)
        v1 = np.cross(normal, np.array([1, 0, 0]))
        if np.linalg.norm(v1) < 1e-6:
            v1 = np.cross(normal, np.array([0, 1, 0]))
        v1 = v1 / np.linalg.norm(v1) * self.size
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2) * self.size

        p1 = self.point_on_plane - v1 - v2
        p2 = self.point_on_plane + v1 - v2
        p3 = self.point_on_plane + v1 + v2
        p4 = self.point_on_plane - v1 + v2

        self.vertices = np.array([p1, p2, p3, p4], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        return k3d.mesh(self.vertices, faces, color=self.color, wireframe=False, name="interactive plane")

    def update_plane(self, _):
        self.point_on_plane[2] = self.z_slider.value
        self.normal = np.array([self.nx_slider.value, self.ny_slider.value, self.nz_slider.value])
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.vertices[:, 2] = self.point_on_plane[2]
        self.mesh.vertices = self.vertices
        print(self.normal, self.point_on_plane)

class InteractivePoint:
    def __init__(self, point, plot, point_size=0.1, color=0xFF0000):
        self.plot = plot
        self.color = color
        self.point_size = point_size

        self.x_slider = widgets.FloatSlider(value=point[0], min=-50, max=50, step=0.01, description="X")
        self.y_slider = widgets.FloatSlider(value=point[1], min=-50, max=50, step=0.01, description="Y")
        self.z_slider = widgets.FloatSlider(value=point[2], min=-50, max=50, step=0.01, description="Z")

        self.x_slider.observe(self.update_point, names="value")
        self.y_slider.observe(self.update_point, names="value")
        self.z_slider.observe(self.update_point, names="value")

        self.point = np.array([self.x_slider.value, self.y_slider.value, self.z_slider.value]).astype(np.float32)
        self.k3d_point = self.create_point()

        self.plot += self.k3d_point
        display(self.x_slider, self.y_slider, self.z_slider)

    def create_point(self):
        return k3d.points(positions=[self.point], point_size=self.point_size, color=self.color, name="interactive point")

    def update_point(self, _):
        self.point = np.array([self.x_slider.value, self.y_slider.value, self.z_slider.value]).astype(np.float32)
        self.k3d_point.positions = [self.point]

# plot = k3d.plot()
# plot.display()

# # Clean previous interactive objects from plot
# for obj in list(plot.objects):
#     print(obj.name)
#     if "interactive" in str(obj.name):
#         plot -= obj

# pm = PointsManager(points)
# point_cloud = pm.get_all_points()
# pm.plot(plot=plot)

# # Create interactive elements
# point_on_plane = np.mean(point_cloud, axis=0)
# # office_3: z = -1
# # room_0: -0.8
# # office_4: -0.85

# point_on_plane[2] = -0.85
# normal = np.array([0, 0, 1])
# my_plane = InteractivePlane(point_on_plane, normal, plot, size=5)

# # initial_point = np.array([-14, 39, 0.6])
# initial_point = point_cloud.mean(axis=0)
# interactive_point = InteractivePoint(initial_point, plot, point_size=0.15, color=0xFF0000)

class InteractivePlane:
    def __init__(self, point_on_plane, normal, plot, point_size=0.01, size=5, color=0x00FF00):
        self.size = size
        self.color = color
        self.plot = plot

        self.z_slider = widgets.FloatSlider(value=point_on_plane[2], min=-10, max=10, step=0.1, description="Z Height")
        self.nx_slider = widgets.FloatSlider(value=normal[0], min=-1, max=1, step=0.1, description="Normal X")
        self.ny_slider = widgets.FloatSlider(value=normal[1], min=-1, max=1, step=0.1, description="Normal Y")
        self.nz_slider = widgets.FloatSlider(value=normal[2], min=-1, max=1, step=0.1, description="Normal Z")

        self.z_slider.observe(self.update_plane, names="value")
        self.nx_slider.observe(self.update_plane, names="value")
        self.ny_slider.observe(self.update_plane, names="value")
        self.nz_slider.observe(self.update_plane, names="value")

        self.point_on_plane = np.array([point_on_plane[0], point_on_plane[1], self.z_slider.value])
        self.normal = np.array([self.nx_slider.value, self.ny_slider.value, self.nz_slider.value])
        self.mesh = self.create_plane()

        self.plot += self.mesh
        display(self.z_slider)

    def create_plane(self):
        normal = self.normal / np.linalg.norm(self.normal)
        v1 = np.cross(normal, np.array([1, 0, 0]))
        if np.linalg.norm(v1) < 1e-6:
            v1 = np.cross(normal, np.array([0, 1, 0]))
        v1 = v1 / np.linalg.norm(v1) * self.size
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2) * self.size

        p1 = self.point_on_plane - v1 - v2
        p2 = self.point_on_plane + v1 - v2
        p3 = self.point_on_plane + v1 + v2
        p4 = self.point_on_plane - v1 + v2

        self.vertices = np.array([p1, p2, p3, p4], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        return k3d.mesh(self.vertices, faces, color=self.color, wireframe=False, name="interactive plane")

    def update_plane(self, _):
        self.point_on_plane[2] = self.z_slider.value
        self.normal = np.array([self.nx_slider.value, self.ny_slider.value, self.nz_slider.value])
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.vertices[:, 2] = self.point_on_plane[2]
        self.mesh.vertices = self.vertices
        print(self.normal, self.point_on_plane)

class InteractivePoint:
    def __init__(self, point, plot, point_size=0.1, color=0xFF0000):
        self.plot = plot
        self.color = color
        self.point_size = point_size

        self.x_slider = widgets.FloatSlider(value=point[0], min=-50, max=50, step=0.01, description="X")
        self.y_slider = widgets.FloatSlider(value=point[1], min=-50, max=50, step=0.01, description="Y")
        self.z_slider = widgets.FloatSlider(value=point[2], min=-50, max=50, step=0.01, description="Z")

        self.x_slider.observe(self.update_point, names="value")
        self.y_slider.observe(self.update_point, names="value")
        self.z_slider.observe(self.update_point, names="value")

        self.point = np.array([self.x_slider.value, self.y_slider.value, self.z_slider.value]).astype(np.float32)
        self.k3d_point = self.create_point()

        self.plot += self.k3d_point
        display(self.x_slider, self.y_slider, self.z_slider)

    def create_point(self):
        return k3d.points(positions=[self.point], point_size=self.point_size, color=self.color, name="interactive point")

    def update_point(self, _):
        self.point = np.array([self.x_slider.value, self.y_slider.value, self.z_slider.value]).astype(np.float32)
        self.k3d_point.positions = [self.point]
