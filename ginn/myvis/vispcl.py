import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
from vis3d import downsample_points
from visutils import *
import time
import numpy as np
from bvh import *


cam_extrinsics = read_extrinsics_binary("/u/rhm4nj/cral/gaussian-splatting/db/playroom/sparse/0/images.bin")
plydata = PlyData.read('/u/rhm4nj/cral/gaussian-splatting/db/playroom/sparse/0/points3D.ply')
qvec, tvec = cam_extrinsics[100].qvec, cam_extrinsics[100].tvec
print(qvec, tvec)

# plydata = PlyData.read("/u/rhm4nj/cral/gaussian-splatting/pretrained/playroom/point_cloud/iteration_7000/point_cloud.ply")

vertex_data = np.array(plydata['vertex'].data)
vertex_array = np.array([list(vertex) for vertex in vertex_data])
vertex_array = vertex_array[:, :3]
vertex_array = remove_outliers(vertex_array, .25, 10)
print(vertex_array.shape)

ray_pts = draw_ray(tvec, qvec, 5, 1000)
ray_start_pts = draw_ray(tvec, qvec, .2, 100)

origin, _ = construct_ray(tvec, qvec)
ray_dir = quaternion_to_vector(qvec)
origin_ray = Ray(direction=ray_dir, origin=origin)
print(origin_ray.origin, origin_ray.direction)

radius = 1

# Build BVH
print("start bvh...")
start_time = time.time()
bvh_root = build_bvh(vertex_array, radius=radius)
print("fin bvh", time.time() - start_time)

# Find intersections
print("start intersection...")
intersections = bvh_ray_intersection(origin_ray, bvh_root, radius)
print("Intersecting points:", intersections)

max_dist = 5
rays = []

if intersections:
    print("found intersections")
    distances = [np.linalg.norm(point - origin_ray.origin) for point in intersections]
    closest_point_index = np.argmin(distances)
    first_intersection = intersections[closest_point_index]
    max_dist = min(distances)

    for intersection in intersections:
        rays.append(Ray.from_two_points(origin_ray.origin, intersection))
else:
    print("did not find intersections")
    first_intersection = origin_ray.point_at_distance(max_dist)
    rays.append(origin_ray)

N = 3000
vertex_array = downsample_points(vertex_array, N)
if intersections:
    vertex_array = np.concatenate((vertex_array, intersections), axis=0)
print(vertex_array.shape)

print("intersection time:", time.time() - start_time)
print("First intersecting point:", first_intersection)

visualize_ray_sphere_intersections(vertex_array, origin_ray.origin, origin_ray.direction, intersections, radius)

plane, n_inliners, inliners = find_best_plane_ransac(vertex_array, .1, 500)
# vertex_array = align_plane_to_xy(vertex_array, plane)
# inliners = align_plane_to_xy(inliners, plane)

# tolerance = 2
# br_pt, tl_pt, max_distance = find_furthest_points(inliners)
# tr_pt = np.array([br_pt[0], tl_pt[1], tl_pt[2]]) # same z
# bl_pt = np.array([tl_pt[0], br_pt[1], tl_pt[2]]) # same z

# max_z_index = np.argmax(vertex_array[:, 2])
# height = vertex_array[max_z_index][2] - tl_pt[2]

# tl_pt2 = np.array([tl_pt[0], tl_pt[1], tl_pt[2] + height]) # same z
# br_pt2 = np.array([br_pt[0], br_pt[1], br_pt[2] + height]) # same z
# tr_pt2 = np.array([tr_pt[0], tr_pt[1], tr_pt[2] + height])
# bl_pt2 = np.array([bl_pt[0], bl_pt[1], bl_pt[2] + height])

# line1 = sample_points_on_line(br_pt, tl_pt, 1000)
# line2 = sample_points_on_line(bl_pt, tr_pt, 1000)
# line3 = sample_points_on_line(br_pt2, tl_pt2, 1000)
# line4 = sample_points_on_line(bl_pt2, tr_pt2, 1000)

# bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 = scale_prism_outward([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 ], 1.3)

# envelope = sample_points_on_prism(
#     np.vstack([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 ]), 1000, 10000
# )
# domain = uniformly_sample_points_in_cube([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2], 10000)

plots = [
    ["interface", vertex_array, 'blue'],
    ['ray', ray_pts, 'red'],
    ['ray_start', ray_start_pts, 'yellow'],
    ["inliners", inliners, "orange"]
    # ["envelope", envelope, 'orange'],
    # ["domain", domain, 'cyan'],
]

if intersections:
    plots.append([
        "on line", sample_points_on_line(origin_ray.origin, first_intersection, 1000), "cyan"
    ])

for ray in rays:
    end_point = ray.point_at_distance(max_dist)
    plots.append([
        "on line", sample_points_on_line(origin_ray.origin, end_point, 1000), "cyan"
    ])

intersections = align_plane_to_xy(intersections, plane)
for i, (name, data, color) in enumerate(plots):
    plots[i][1] = align_plane_to_xy(data, plane)

fig = go.Figure()
for plotname, data, color in plots:
    fig.add_trace(
        go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='markers',
            marker=dict(size=2, color=color),
            name=plotname
        )
    )
fig.show()

# for plotname, data, color in plots:
#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter3d(
#             x=data[:, 0], y=data[:, 1], z=data[:, 2],
#             mode='markers',
#             marker=dict(size=2, color=color),
#             name=plotname
#         )
#     )
#     fig.show()

# visualize_ray_sphere_intersections(vertex_array, origin_ray.origin, origin_ray.direction, intersections, radius)