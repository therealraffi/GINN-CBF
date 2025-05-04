import numpy as np
import torch
import plotly.graph_objects as go

# Downsample function to randomly select N points from a larger set
def downsample_points(points, N):
    if points.shape[0] > N:
        indices = np.random.choice(points.shape[0], N, replace=False)
        return points[indices]
    return points

# Function to load point clouds from file paths and downsample them
def load_and_downsample_point_clouds(file_paths, N):
    point_clouds = []
    for file_path in file_paths:
        points = torch.from_numpy(np.load(file_path)).cpu().numpy()
        point_clouds.append(downsample_points(points, N))
    return point_clouds

# Function to plot a 3D scatter plot for a given set of points
def plot_3d_points(points, color, point_name, marker_size=5):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=marker_size, color=color, opacity=0.8),
        name=point_name
    ))
    # Set axis ranges explicitly
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[-2, 2], title='X Axis'),
            yaxis=dict(nticks=10, range=[-2, 2], title='Y Axis'),
            zaxis=dict(nticks=10, range=[-2, 2], title='Z Axis'),
            aspectmode='cube'  # Ensure equal scaling of axes
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        title=point_name  # Add title to the plot
    )
    # Show the plot
    fig.show()

# Function to plot multiple point clouds together in a single figure
def plot_grouped_points(grouped_point_clouds, group_name, marker_size=5):
    fig = go.Figure()
    for points, name, color in grouped_point_clouds:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=marker_size, color=color, opacity=0.8),
            name=name
        ))
    # Set axis ranges explicitly
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[-2, 2], title='X Axis'),
            yaxis=dict(nticks=10, range=[-2, 2], title='Y Axis'),
            zaxis=dict(nticks=10, range=[-2, 2], title='Z Axis'),
            aspectmode='cube'  # Ensure equal scaling of axes
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        title=group_name  # Add group name as title
    )
    # Show the plot
    fig.show()

# Generalized function to handle grouping and rendering
def render_point_clouds_with_grouping(file_paths, names, colors, groups, N=1000):
    """
    :param file_paths: List of file paths to point clouds
    :param names: List of names for the point clouds
    :param colors: List of colors for each point cloud
    :param groups: Dictionary where keys are group names and values are lists of indices for the files to group together
    :param N: Number of points to downsample to
    """
    # Load and downsample all point clouds
    point_clouds = load_and_downsample_point_clouds(file_paths, N)
    
    # Keep track of files that have already been grouped
    grouped_indices = set()

    # Handle grouped plots
    for group_name, group_indices in groups.items():
        grouped_point_clouds = [(point_clouds[i], names[i], colors[i]) for i in group_indices]
        plot_grouped_points(grouped_point_clouds, group_name)

        # Add indices to the set of grouped indices
        grouped_indices.update(group_indices)
    
    # Handle separate plots for ungrouped files
    for i in range(len(file_paths)):
        if i not in grouped_indices:
            plot_3d_points(point_clouds[i], colors[i], names[i])

if __name__ == "__main__":
    # Bracket setup
    file_paths = [
        'GINN/simJEB/derived/pts_far_outside.npy',
        'GINN/simJEB/derived/pts_on_env.npy',
        'GINN/simJEB/derived/pts_inside.npy',
        'GINN/simJEB/derived/pts_outside.npy',
        'GINN/simJEB/derived/interface_points.npy',
        'GINN/simJEB/derived/pts_around_interface_outside_env_10mm.npy'
    ]

    names = [
        'pts_far_from_env_constraint', 
        'pts_on_envelope', 
        'inside_envelope', 
        'pts_outside_envelope', 
        'interface_points', 
        'envelope_around_interface'
    ]

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

    # Group definition: keys are the group names, values are lists of file indices to group together
    groups = {
        'Envelope': [0, 1, 3, 5],  
        'Domain': [2],   
        'Interface': [4]
    }

    # Render the grouped and ungrouped plots
    render_point_clouds_with_grouping(file_paths, names, colors, groups, N=10000)
