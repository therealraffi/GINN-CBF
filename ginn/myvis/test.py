import numpy as np
import torch
from scipy.spatial import KDTree
import time

# Example NumPy point cloud
# numpy_point_cloud = np.random.rand(10000, 3)  # A 3D point cloud with 10,000 points
numpy_point_cloud = torch.from_numpy(np.load(f'/scratch/rhm4nj/cral/ginn/myvis/cube/pts_on_env.npy')).float()
print(numpy_point_cloud)

# Example tensor of points to check (N x 3)
tensor_points = torch.rand(100, 3)  # A tensor with 100 3D points
print(tensor_points)

# Convert the PyTorch tensor to a NumPy array for KDTree processing
tensor_points_np = tensor_points.cpu().numpy()

start_time = time.time()

kdtree = KDTree(numpy_point_cloud)
tolerance = 1e-5  # Adjust as needed
distances, _ = kdtree.query(tensor_points_np, distance_upper_bound=tolerance)
inside_points_mask = distances < tolerance
inside_points = torch.tensor(inside_points_mask, dtype=torch.bool).tolist()

print(inside_points)

print("time taken", time.time() - start_time)