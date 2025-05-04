import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
from vis3d import downsample_points

class Ray:
    # direction is (1x3) NOT quarternion
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(direction)  # Normalize direction

    def point_at_distance(self, r):
        return self.origin + r * self.direction
    
    @classmethod
    def from_two_points(cls, point1, point2):
        """
        Create a ray given two points.
        :param point1: The origin of the ray.
        :param point2: A second point to determine the ray's direction.
        :return: An instance of the Ray class.
        """
        origin = np.array(point1)
        direction = np.array(point2) - np.array(point1)  # Direction is point2 - point1
        return cls(origin, direction)

class BVHNode:
    def __init__(self, bounding_box, points, left=None, right=None):
        self.bounding_box = bounding_box  # Axis-aligned bounding box (min, max)
        self.points = points  # Points contained in this node
        self.left = left
        self.right = right

def build_bvh(points, max_points_per_node=4, radius=1):
    """Recursively build a BVH."""
    if len(points) <= max_points_per_node:
        return BVHNode(bounding_box=create_bounding_box(points, radius), points=points)

    # Split points into two groups along the longest axis of the bounding box
    bounding_box = create_bounding_box(points, radius)
    axis = np.argmax(bounding_box[1] - bounding_box[0])  # Longest axis

    # Sort points along this axis
    sorted_points = sorted(points, key=lambda p: p[axis])
    mid = len(points) // 2

    left_node = build_bvh(sorted_points[:mid], max_points_per_node, radius)
    right_node = build_bvh(sorted_points[mid:], max_points_per_node, radius)

    return BVHNode(bounding_box, points, left=left_node, right=right_node)

def create_bounding_box(points, radius=0.5):
    """Create an axis-aligned bounding box (AABB) for a set of points with optional radius."""
    points = np.array(points)
    min_corner = np.min(points, axis=0) - radius
    max_corner = np.max(points, axis=0) + radius
    return min_corner, max_corner

def ray_intersects_aabb(ray, bounding_box):
    """Check if a ray intersects an axis-aligned bounding box (AABB)."""
    inv_dir = 1 / ray.direction
    # t_min = (bounding_box[0] - ray.origin) * inv_dir
    # t_max = (bounding_box[1] - ray.origin) * inv_dir
    t_min = (bounding_box[0] - ray.origin) * inv_dir
    t_max = (bounding_box[1] - ray.origin) * inv_dir

    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)

    t_near = np.max(t1)
    t_far = np.min(t2)

    return t_near <= t_far and t_far >= 0

def ray_intersects_sphere(ray, center, radius):
    """Check if a ray intersects a sphere and return True if it does."""
    # Vector from the ray origin to the center of the sphere
    oc = ray.origin - center
    
    # Coefficients for the quadratic equation
    a = np.dot(ray.direction, ray.direction)  # This is always 1 since the direction is normalized
    b = 2.0 * np.dot(oc, ray.direction)
    c = np.dot(oc, oc) - radius**2
    
    # Compute the discriminant
    discriminant = b**2 - 4 * c
    
    if discriminant < 0:
        return False  # No intersection
    else:
        # There are two intersections, but we just care if one is in front of the ray
        t1 = (-b - np.sqrt(discriminant)) / 2.0
        t2 = (-b + np.sqrt(discriminant)) / 2.0
        
        # Check if there is a valid intersection in the positive direction of the ray
        if t1 >= 0 or t2 >= 0:
            return True  # The ray intersects the sphere
        return False  # Intersection is behind the ray

def bvh_ray_intersection(ray, bvh_node, radius, intersections=[]):
    """Recursively check for intersections with the BVH, considering points as spheres."""
    if not ray_intersects_aabb(ray, bvh_node.bounding_box):
        return intersections  # No intersection with this node's bounding box

    if bvh_node.left is None and bvh_node.right is None:  # Leaf node
        for point in bvh_node.points:
            if ray_intersects_sphere(ray, point, radius):
                intersections.append(point)
    else:
        if bvh_node.left:
            bvh_ray_intersection(ray, bvh_node.left, radius, intersections)
        if bvh_node.right:
            bvh_ray_intersection(ray, bvh_node.right, radius, intersections)

    return intersections

def visualize_ray_sphere_intersections(points, ray_origin, ray_direction, intersections, radius):
    fig = go.Figure()

    # Add spheres for points
    for point in points:
        # If the point intersects, color it differently
        color = 'red' if any(np.array_equal(point, intersecting_point) for intersecting_point in intersections) else 'blue'
        
        # Add spheres as a cloud of points (for simplicity in visualization)
        sphere = go.Mesh3d(
            x=[point[0] + radius * np.sin(theta) * np.cos(phi) for theta in np.linspace(0, np.pi, 10) for phi in np.linspace(0, 2*np.pi, 10)],
            y=[point[1] + radius * np.sin(theta) * np.sin(phi) for theta in np.linspace(0, np.pi, 10) for phi in np.linspace(0, 2*np.pi, 10)],
            z=[point[2] + radius * np.cos(theta) for theta in np.linspace(0, np.pi, 10) for phi in np.linspace(0, 2*np.pi, 10)],
            color=color,
            opacity=0.5,
            showscale=False
        )
        fig.add_trace(sphere)

    # Add ray
    ray_end = ray_origin + ray_direction * 20  # Extend the ray
    fig.add_trace(go.Scatter3d(
        x=[ray_origin[0], ray_end[0]],
        y=[ray_origin[1], ray_end[1]],
        z=[ray_origin[2], ray_end[2]],
        mode='lines',
        line=dict(color='green', width=5),
        name='Ray'
    ))

    # Set up axis
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title="Ray-Sphere Intersection Visualization",
        showlegend=False
    )

    fig.show()

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

# plydata = PlyData.read('/u/rhm4nj/cral/gaussian-splatting/db/playroom/sparse/0/points3D.ply')
# vertex_data = np.array(plydata['vertex'].data)
# vertex_array = np.array([list(vertex) for vertex in vertex_data])
# points = vertex_array[:, :3]

# # points = np.random.randn(10000, 3) * 1

# ray = Ray(
#     direction=quaternion_to_vector([ 0.70183908, -0.04372278, -0.67398588, -0.22639183]), 
#     origin=[-0.10068968, -1.97592827, -1.0705704 ]
# )

# radius = .5  # Large radius for each sphere
# bvh_root = build_bvh(points, max_points_per_node=4, radius=radius)
# intersections = bvh_ray_intersection(ray, bvh_root, radius)

# # print("Intersecting spheres:", intersections)

# points = downsample_points(points, 5000)
# if intersections:
#     points = np.concatenate((points, intersections), axis=0)

# distances = [np.linalg.norm(point - ray.origin) for point in intersections]
# closest_point_index = np.argmin(distances)
# closest_point = intersections[closest_point_index]

# # intersections = [point for point in points.tolist() if ray_intersects_sphere(ray, point, radius)]
# visualize_ray_sphere_intersections(points, ray.origin, ray.direction, [closest_point], radius)
