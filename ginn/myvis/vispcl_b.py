import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
from vis3d import downsample_points
from visutils import *
import numpy as np
from bvh import *
import os

cam_extrinsics = read_extrinsics_binary("/u/rhm4nj/cral/gaussian-splatting/db/playroom/sparse/0/images.bin")
plydata = PlyData.read('/u/rhm4nj/cral/gaussian-splatting/db/playroom/sparse/0/points3D.ply')
qvec, tvec = cam_extrinsics[100].qvec, cam_extrinsics[100].tvec
print(qvec, tvec)

# plydata = PlyData.read("/u/rhm4nj/cral/gaussian-splatting/pretrained/playroom/point_cloud/iteration_7000/point_cloud.ply")

vertex_data = np.array(plydata['vertex'].data)
vertex_array = np.array([list(vertex) for vertex in vertex_data])
vertex_array = vertex_array[:, :3]
vertex_array_final = remove_outliers(vertex_array, .25, 10)
print(vertex_array_final.shape)

# align to plane
vertex_array = downsample_points(vertex_array_final, 3000)
plane, n_inliners, inliners = find_best_plane_ransac(vertex_array, .1, 500)
vertex_array = align_plane_to_xy(vertex_array, plane)
vertex_array_final = align_plane_to_xy(vertex_array_final, plane)
inliners = align_plane_to_xy(inliners, plane)

# get domain
tolerance = 2
br_pt, tl_pt, max_distance = find_furthest_points(inliners)
tr_pt = np.array([br_pt[0], tl_pt[1], tl_pt[2]]) # same z
bl_pt = np.array([tl_pt[0], br_pt[1], tl_pt[2]]) # same z

max_z_index = np.argmax(vertex_array[:, 2])
height = vertex_array[max_z_index][2] - tl_pt[2]

tl_pt2 = np.array([tl_pt[0], tl_pt[1], tl_pt[2] + height]) # same z
br_pt2 = np.array([br_pt[0], br_pt[1], br_pt[2] + height]) # same z
tr_pt2 = np.array([tr_pt[0], tr_pt[1], tr_pt[2] + height])
bl_pt2 = np.array([bl_pt[0], bl_pt[1], bl_pt[2] + height])

line1 = sample_points_on_line(br_pt, tl_pt, 1000)
line2 = sample_points_on_line(bl_pt, tr_pt, 1000)
line3 = sample_points_on_line(br_pt2, tl_pt2, 1000)
line4 = sample_points_on_line(bl_pt2, tr_pt2, 1000)

# bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 = scale_prism_outward([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 ], 1.35)
print("Expanding cube...")
bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 = scale_prism_outward_adapt([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 ], vertex_array_final)
domain = uniformly_sample_points_in_cube([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2], 10000)


# get envelope
bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 = scale_prism_outward([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 ], 1.05)
envelope = sample_points_on_prism(
    np.vstack([bl_pt, br_pt, tl_pt, tr_pt, bl_pt2, br_pt2, tl_pt2, tr_pt2 ]), 1000, 10000
)

# [[x_min, x_max], [y_min, y_max], [z_min, z_max]]  
bounds = [[bl_pt[0], br_pt[0]], [bl_pt[1], tr_pt2[1]], [bl_pt2[2], tr_pt[2]]]
for i, bound in enumerate(bounds): bounds[i] = sorted(bound)
for i, bound in enumerate(bounds): bounds[i] = [bounds[i][0] - 0.5, bounds[i][1] + 0.5]
bounds = np.array(bounds)

normals = calculate_normals(vertex_array_final, k=20)
print(normals)

plots = [
    ["interface", vertex_array_final, 'blue'],
    ["envelope", envelope, 'orange'],
    ["domain", domain, 'cyan'],
]
all_pts = plots + [
    ["normals", normals, '_'],
    ["bounds", bounds, "_"],
]

outputdir = "outs"
for plotname, data, _ in all_pts:
    out_path = os.path.join(outputdir, f"{plotname}_points")
    np.save(out_path, data)
    print(f"Saved {out_path}.npy")

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

for plotname, data, color in plots:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='markers',
            marker=dict(size=2, color=color),
            name=plotname
        )
    )
    fig.show()

# visualize_ray_sphere_intersections(vertex_array, origin_ray.origin, origin_ray.direction, intersections, radius)