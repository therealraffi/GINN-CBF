import numpy as np
import struct
import collections
import struct
import collections
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import pickle

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
      """Read and unpack the next bytes from a binary file.
      :param fid:
      :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
      :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
      :param endian_character: Any of {@, =, <, >, !}
      :return: Tuple of read and unpacked values.
      """
      data = fid.read(num_bytes)
      return struct.unpack(endian_character + format_char_sequence, data)

    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def remove_outliers(point_cloud, radius=1, min_neighbors=3):
    """
    Removes outliers from the point cloud based on the number of neighbors within a certain radius.
    
    :param point_cloud: numpy array of shape (N, 3) representing the point cloud (x, y, z)
    :param radius: float, radius within which to count neighbors
    :param min_neighbors: int, minimum number of neighbors for a point to be considered valid
    :return: filtered point cloud (numpy array of shape (M, 3)), where M <= N
    """

    nbrs = NearestNeighbors(radius=radius).fit(point_cloud)
    indices = nbrs.radius_neighbors(point_cloud, return_distance=False)
    filtered_points = [point for i, point in enumerate(point_cloud) if len(indices[i]) >= min_neighbors]
    return np.array(filtered_points)

def align_point_cloud(point_cloud, qvec, tvec):
    """
    Aligns the point cloud based on the camera extrinsics (rotation and translation).
    
    :param point_cloud: numpy array of shape (N, 3) representing the point cloud (x, y, z)
    :param extrinsics: dictionary containing 'qvec' (quaternion) and 'tvec' (translation vector)
    :return: numpy array of aligned point cloud
    """
    rotation_matrix = R.from_quat(qvec).as_matrix()
    inverse_rotation = rotation_matrix.T
    inverse_translation = -np.dot(inverse_rotation, tvec)
    aligned_point_cloud = np.dot(point_cloud, inverse_rotation) + inverse_translation
    return aligned_point_cloud

def sample_points_on_cube_edges(cube_corners, N):
    """
    Samples N points along the edges of the cube formed by the 8 corners.

    :param cube_corners: numpy array of shape (8, 3) representing the 8 corners of the cube.
    :param N: Total number of points to sample along the edges.
    :return: numpy array of sampled points of shape (N, 3)
    """
    # Define the 12 edges of the cube (pairs of corners)
    edges = [
        (0, 1), (0, 2), (0, 3),  # Edges connected to corner 0
        (1, 4), (1, 5),           # Edges connected to corner 1
        (2, 4), (2, 6),           # Edges connected to corner 2
        (3, 5), (3, 6),           # Edges connected to corner 3
        (4, 7),                   # Edges connected to corner 4
        (5, 7),                   # Edges connected to corner 5
        (6, 7)                    # Edge between corners 6 and 7
    ]

    # Number of edges
    num_edges = len(edges)

    # Distribute N points across 12 edges equally (or near equally)
    points_per_edge = N // num_edges
    remaining_points = N % num_edges  # Extra points to distribute evenly

    sampled_points = []
    for i, (start_idx, end_idx) in enumerate(edges):
        # Get the start and end points of the edge
        start_point = cube_corners[start_idx]
        end_point = cube_corners[end_idx]
        
        # Determine how many points to sample on this edge
        if i < remaining_points:
            num_points = points_per_edge + 1  # Distribute extra points
        else:
            num_points = points_per_edge
        
        # Sample points along the edge using linear interpolation
        for t in np.linspace(0, 1, num_points, endpoint=False):
            sampled_point = (1 - t) * start_point + t * end_point
            sampled_points.append(sampled_point)

    return np.array(sampled_points)

def find_best_plane_ransac(point_cloud, tolerance=0.01, max_trials=1000):
    """
    Finds the best-fitting plane in the point cloud using RANSAC.

    :param point_cloud: numpy array of shape (N, 3) representing the point cloud (x, y, z)
    :param tolerance: float, the maximum allowed distance of points from the plane to be considered an intersection
    :param max_trials: int, the number of RANSAC iterations to find the best plane
    :return: best plane parameters (a, b, c, d), inliers count, and inlier points
    """
    # Extract x, y, and z coordinates from the point cloud
    X = point_cloud[:, :2]  # x and y
    z = point_cloud[:, 2]   # z
    
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)  # Add bias term for the plane equation
    
    ransac = RANSACRegressor(residual_threshold=tolerance, max_trials=max_trials)
    ransac.fit(X_poly, z)
    
    inlier_mask = ransac.inlier_mask_
    inliers = point_cloud[inlier_mask]
    
    a, b, d = ransac.estimator_.coef_[1], ransac.estimator_.coef_[2], ransac.estimator_.intercept_
    c = -1 
    
    return (a, b, c, d), np.sum(inlier_mask), inliers

def find_furthest_points(point_cloud):
    """
    Finds two points a and b in the point cloud that have the greatest distance between them.
    
    :param point_cloud: numpy array of shape (N, 3) representing the point cloud (x, y, z)
    :return: tuple containing the two points a and b with the greatest distance between them and the distance
    """
    # Calculate pairwise Euclidean distances between all points
    distances = squareform(pdist(point_cloud, 'euclidean'))
    
    # Find the indices of the two points with the maximum distance
    max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)
    
    # Get the points a and b corresponding to these indices
    point_a = point_cloud[max_dist_indices[0]]
    point_b = point_cloud[max_dist_indices[1]]
    
    # Calculate the maximum distance
    max_distance = distances[max_dist_indices]
    
    return point_a, point_b, max_distance

def sample_points_on_line(a, b, num_points):
    """
    Samples points from the line formed by points a and b.

    :param a: numpy array representing point a (x_a, y_a, z_a)
    :param b: numpy array representing point b (x_b, y_b, z_b)
    :param num_points: Number of points to sample along the line
    :return: numpy array of sampled points along the line
    """
    t_values = np.linspace(0, 1, num_points)
    sampled_points = np.array([(1 - t) * a + t * b for t in t_values])
    return sampled_points

def find_highest_point_above_plane(point_cloud, plane_params):
    """
    Finds the highest point above the plane and creates a parallel plane intersecting that point.
    
    :param point_cloud: numpy array of shape (N, 3) representing the point cloud (x, y, z)
    :param plane_params: tuple (a, b, c, d) representing the plane equation ax + by + cz + d = 0
    :return: the highest point and the new plane parameters (a, b, c, d')
    """
    a, b, c, d = plane_params
    
    distances = (np.dot(point_cloud, np.array([a, b, c])) + d) / np.sqrt(a**2 + b**2 + c**2)
    
    highest_point_idx = np.argmax(distances)
    highest_point = point_cloud[highest_point_idx]
    
    d_new = -(a * highest_point[0] + b * highest_point[1] + c * highest_point[2])
    new_plane_params = (a, b, c, d_new)
    
    return highest_point, new_plane_params

def align_plane_to_xy(point_cloud, plane_params):
    """
    Applies the rotation and translation needed to align a given plane with the XY plane.
    
    :param point_cloud: numpy array of shape (N, 3) representing the point cloud (x, y, z)
    :param plane_params: tuple (a, b, c, d) representing the plane ax + by + cz + d = 0
    :return: Transformed point cloud with the plane aligned to the XY plane
    """
    a, b, c, d = plane_params
    
    # Step 1: Translate the plane to pass through the origin
    normal_vector = np.array([a, b, c])
    plane_distance = d / np.linalg.norm(normal_vector)  # Signed distance to the plane
    translation_vector = plane_distance * normal_vector / np.linalg.norm(normal_vector)
    
    translated_cloud = point_cloud - translation_vector
    target_normal = np.array([0, 0, 1])  # Z-axis
    
    # Calculate the rotation axis (cross product of normal_vector and Z-axis)
    axis_of_rotation = np.cross(normal_vector, target_normal)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)  # Normalize the axis
    
    angle = np.arccos(np.dot(normal_vector, target_normal) / np.linalg.norm(normal_vector))
    rotation = R.from_rotvec(angle * axis_of_rotation)  # Rotation object
    rotated_cloud = rotation.apply(translated_cloud)
    
    return rotated_cloud

def sample_points_on_prism(corners, num_points_per_edge=100, num_points_per_face=100):
    # Corners should be ordered as follows:
    # 1st Plane: [bl_pt, br_pt, tl_pt, tr_pt]
    # 2nd Plane: [bl_pt2, br_pt2, tl_pt2, tr_pt2]
    
    bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 = corners

    # List of edges in the prism
    edges = [
        (bl_pt, br_pt),  # Edge on first plane
        (br_pt, tr_pt),
        (tr_pt, tl_pt),
        (tl_pt, bl_pt),
        (bl_pt2, br_pt2),  # Edge on second plane
        (br_pt2, tr_pt2),
        (tr_pt2, tl_pt2),
        (tl_pt2, bl_pt2),
        (bl_pt, bl_pt2),  # Connecting edges between the two planes
        (br_pt, br_pt2),
        (tr_pt, tr_pt2),
        (tl_pt, tl_pt2)
    ]
    
    # List of faces in the prism (each face is a list of 4 corner points)
    faces = [
        [bl_pt, br_pt, tr_pt, tl_pt],  # Front face
        [bl_pt2, br_pt2, tr_pt2, tl_pt2],  # Back face
        [bl_pt, bl_pt2, tl_pt2, tl_pt],  # Left face
        [br_pt, br_pt2, tr_pt2, tr_pt],  # Right face
        [tl_pt, tr_pt, tr_pt2, tl_pt2],  # Top face
        [bl_pt, br_pt, br_pt2, bl_pt2]  # Bottom face
    ]
    
    # Function to sample points between two points (for edges)
    def sample_edge(p1, p2, num_points):
        return np.linspace(p1, p2, num_points)

    # Function to sample points on a face (bilinear interpolation)
    def sample_face(p1, p2, p3, p4, num_points):
        u = np.linspace(0, 1, num_points)
        v = np.linspace(0, 1, num_points)
        points = []
        for ui in u:
            for vi in v:
                point = (1-ui)*(1-vi)*p1 + ui*(1-vi)*p2 + ui*vi*p3 + (1-ui)*vi*p4
                points.append(point)
        return np.array(points)
    
    # Sample points along each edge
    sampled_points_edges = []
    for edge in edges:
        p1, p2 = edge
        sampled_points_edges.append(sample_edge(p1, p2, num_points_per_edge))
    
    # Sample points on each face
    sampled_points_faces = []
    for face in faces:
        p1, p2, p3, p4 = face
        sampled_points_faces.append(sample_face(p1, p2, p3, p4, int(np.sqrt(num_points_per_face))))
    
    # Stack the sampled points into a single array
    sampled_points = np.vstack(sampled_points_edges + sampled_points_faces)
    
    return sampled_points

def scale_prism_outward(corners, scale_factor):
    center = np.mean(corners, axis=0)
    scaled_corners = []
    for corner in corners:
        vector_from_center = corner - center
        scaled_corner = center + scale_factor * vector_from_center
        scaled_corners.append(scaled_corner)
    return scaled_corners

import numpy as np

def scale_prism_outward_adapt(cube_vertices, point_cloud):
    """
    Adjust the vertices of a prism (or cube) such that the prism encompasses all points in the point cloud.

    :param cube_vertices: List of 8 vertices of the prism (numpy arrays).
    :param point_cloud: Nx3 numpy array containing the point cloud.
    :return: Updated vertices of the prism.
    """
    cube_vertices = np.array(cube_vertices)
    min_point = np.min(point_cloud, axis=0)
    max_point = np.max(point_cloud, axis=0)
    for i, vertex in enumerate(cube_vertices):
        for j in range(3):  # Loop through x, y, z coordinates
            if vertex[j] < (min_point[j] + max_point[j]) / 2:
                cube_vertices[i][j] = min_point[j]  # Move to the outermost "min" side
            else:
                cube_vertices[i][j] = max_point[j]  # Move to the outermost "max" side
    return cube_vertices


def uniformly_sample_points_in_cube(corners, num_points):
    # Get the minimum and maximum values in x, y, and z directions
    x_min, y_min, z_min = np.min(corners, axis=0)
    x_max, y_max, z_max = np.max(corners, axis=0)
    
    # Sample uniformly within these bounds
    sampled_points = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(num_points, 3)
    )
    
    return sampled_points

def construct_ray(tvec, qvec):
    """
    Constructs a ray in 3D space given the camera's translation and quaternion.
    
    :param tvec: numpy array of shape (3,), the camera's translation vector (position in 3D space)
    :param qvec: numpy array of shape (4,), the camera's quaternion [w, x, y, z] representing orientation
    :return: ray_origin, ray_direction (two numpy arrays of shape (3,))
    """
    ray_origin = tvec
    rotation_matrix = R.from_quat(qvec).as_matrix()
    ray_direction = np.dot(rotation_matrix, np.array([0, 0, 1]))  # Forward direction in world space
    return ray_origin, ray_direction

import trimesh

def build_scene_from_point_cloud(point_cloud, radius):
    """
    Builds a BVH (Bounding Volume Hierarchy) from the point cloud, inflating each point into a sphere.
    
    :param point_cloud: numpy array of shape (N, 3), the point cloud
    :param radius: float, the radius to inflate each point into a sphere
    :return: BVH tree (trimesh object)
    """
    spheres = [trimesh.creation.icosphere(radius=radius, subdivisions=2).apply_translation(point)
               for point in point_cloud]
    scene = trimesh.Scene(spheres)

    combined_mesh = trimesh.util.concatenate(spheres)
    # bvh_tree = trimesh.collision.BVH(scene.dump())  # Dump the scene into a format that can be used for BVH
    return scene

def find_first_intersection(ray_origin, ray_direction, scene):
    """
    Finds the first point in the BVH that intersects with the ray.
    
    :param ray_origin: numpy array of shape (3,), the origin of the ray
    :param ray_direction: numpy array of shape (3,), the direction of the ray
    :param bvh_tree: BVH tree (trimesh object) containing the inflated spheres
    :return: The closest intersection point or None if no intersection
    """
    # Perform ray-scene intersection
    locations, index_ray, index_tri = scene.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction],
        multiple_hits=False  # Only need the first intersection per ray
    )
    
    if len(locations) > 0:
        return locations[0]
    else:
        return None

def draw_ray(tvec, qvec, N, num_points=100):
    """
    Draws a ray of points starting from the given tvec (camera position) and extending in the 
    direction specified by the qvec (quaternion) over a distance of N.

    :param tvec: numpy array of shape (3,), the camera's translation vector (starting point of the ray)
    :param qvec: numpy array of shape (4,), the camera's quaternion [w, x, y, z] representing orientation
    :param N: float, the distance the ray should span
    :param num_points: int, the number of points to generate along the ray
    :return: numpy array of shape (num_points, 3), the points along the ray
    """
    rotation_matrix = R.from_quat(qvec).as_matrix()
    forward_direction = np.dot(rotation_matrix, np.array([0, 0, 1]))  # World-space forward direction
    distances = np.linspace(0, N, num_points)
    ray_points = np.array([tvec + d * forward_direction for d in distances])
    return ray_points

def quaternion_to_vector(q):
    """Convert a quaternion to a 3D direction vector."""
    # q is expected to be a 4-element array [w, x, y, z]
    w, x, y, z = q
    v = np.array([0, 0, 1])  # This can be any reference vector
    v_q = np.array([0, v[0], v[1], v[2]])
    q_v = quaternion_multiply(q, v_q)
    q_conj = np.array([w, -x, -y, -z])
    result_q = quaternion_multiply(q_v, q_conj)
    return result_q[1:]  # Skip the scalar part

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def calculate_normals(points, k=10):
    """
    Calculate normals for a given point cloud.

    :param points: Nx3 array of points (point cloud).
    :param k: Number of nearest neighbors to use for each point.
    :return: Nx3 array of normals.
    """
    num_points = points.shape[0]
    if k > num_points:
        k = num_points - 1  # Adjust k to avoid errors

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(points)
    _, indices = neighbors.kneighbors(points)

    normals = []

    for i, point_idx in enumerate(indices):
        neighborhood = points[point_idx]

        neighborhood_centered = neighborhood - np.mean(neighborhood, axis=0)

        pca = PCA(n_components=3)
        pca.fit(neighborhood_centered)

        normal = pca.components_[-1]
        normals.append(normal)

    normals = np.array(normals)
    for i in range(len(normals)):
        if np.dot(normals[i], points[i]) < 0:
            normals[i] = -normals[i]

    return normals