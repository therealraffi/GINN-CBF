import numpy as np
import torch
import plotly.graph_objects as go

# Downsample function to randomly select N points from a larger set
def downsample_points(points, N):
    if points.shape[0] > N:
        indices = np.random.choice(points.shape[0], N, replace=False)
        return points[indices]
    return points

# Function to load 2D arrays (from .npy files) and downsample them
def load_and_downsample_2d_arrays(file_paths, N):
    arrays = []
    for file_path in file_paths:
        points = np.load(file_path)  # Load the 2D numpy array
        arrays.append(downsample_points(points, N))
    return arrays

# Function to plot a 2D scatter plot for a given set of 2D points
def plot_2d_points(points, color, point_name, marker_size=5):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1],
        mode='markers',
        marker=dict(size=marker_size, color=color, opacity=0.8),
        name=point_name
    ))
    # Set axis ranges explicitly (can be adjusted)
    fig.update_layout(
        xaxis=dict(nticks=10, title='X Axis'),
        yaxis=dict(nticks=10, title='Y Axis'),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        title=point_name  # Add title to the plot
    )
    # Show the plot
    fig.show()

# Function to plot multiple 2D point arrays together in a single figure
def plot_grouped_2d_arrays(grouped_arrays, group_name, marker_size=5):
    fig = go.Figure()
    for points, name, color in grouped_arrays:
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers',
            marker=dict(size=marker_size, color=color, opacity=0.8),
            name=name
        ))
    # Set axis ranges explicitly (can be adjusted)
    fig.update_layout(
        xaxis=dict(nticks=10, title='X Axis'),
        yaxis=dict(nticks=10, title='Y Axis'),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        title=group_name  # Add group name as title
    )
    # Show the plot
    fig.show()

# Generalized function to handle grouping and rendering
def render_2d_arrays_with_grouping(file_paths, names, colors, groups={}, N=1000):
    """
    :param file_paths: List of file paths to 2D arrays
    :param names: List of names for the arrays
    :param colors: List of colors for each array
    :param groups: Dictionary where keys are group names and values are lists of indices for the files to group together
    :param N: Number of points to downsample to
    """
    # Load and downsample all arrays
    arrays = load_and_downsample_2d_arrays(file_paths, N)
    
    # Keep track of files that have already been grouped
    grouped_indices = set()

    # Handle grouped plots
    for group_name, group_indices in groups.items():
        grouped_arrays = [(arrays[i], names[i], colors[i]) for i in group_indices]
        plot_grouped_2d_arrays(grouped_arrays, group_name)

        # Add indices to the set of grouped indices
        grouped_indices.update(group_indices)
    
    # Handle separate plots for ungrouped files
    for i in range(len(file_paths)):
        if i not in grouped_indices:
            plot_2d_points(arrays[i], colors[i], names[i])

# Example usage
file_paths = [
    'points/simple_2d/domain.npy', 
    'points/simple_2d/envelope.npy', 
    'points/simple_2d/interface.npy', 
    'points/simple_2d/obstacles.npy', 

]

names = [
    'domain',
    'envelope',
    'interface',
    'obstacles'
]

colors = ['red', 'green', 'blue', 'cyan']

# Group definition: keys are the group names, values are lists of file indices to group together
groups = {
    "all": [0, 1, 2, 3]
}

# Render the grouped and ungrouped plots
render_2d_arrays_with_grouping(file_paths, names, colors, groups, N=10000)