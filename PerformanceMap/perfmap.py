'''
Used with data from random rollouts for one boundary condition
Create a 3D performance map 
    - x,y axis is frame grid clustered using UMAP for dimensionality reduction.
    - z axis is max deflection.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import umap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update, callback
import os
import base64
import pandas as pd
import plotly.express as px

def umap_reduce_framegrid(stacked_frame_grids, n_neighbors=15, min_dist=0.1):
    """
    Reduce the dimensionality of frame grids using UMAP.
    Input:
        - stacked_frame_grids: np.array (num_designs, num_rows, num_cols)
        - n_neighbors: int, optional
        UMAP parameter controlling local connectivity. Smaller neighborhoods may result in tighter, more isolated clusters. Larger neighborhoods can create smoother transitions between clusters but may blur local details.
        - min_dist: float, optional
    Output:
        - umap_2d: np.array (num_designs, 2)
    """
    # Flatten frame grids for UMAP input
    num_designs, num_rows, num_cols = stacked_frame_grids.shape
    flattened_grids = stacked_frame_grids.reshape(num_designs, -1)

    # Perform UMAP dimensionality reduction to 2D
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42, n_jobs=1)
    umap_2d = reducer.fit_transform(flattened_grids)
    return umap_2d

def create_cluster_image_plot(selected_cluster_idx, render_dir, name):
    """
    Create a plot of selected images where rows correspond to different clusters,
    and along the row, images are laid out in order of low, median, high max displacement.

    Parameters:
        selected_cluster_idx (list of lists): 2D list of selected episode indices per cluster.
        render_dir (str): Directory where rendered episode images are saved.
        name (str): Name for the resulting plot image file.
        n_select (int): Number of images selected per category (low, median, high).

    Returns:
        None: Saves the plot to a file in `render_dir`.
    """
    num_clusters = len(selected_cluster_idx)
    max_columns = max(len(cluster) for cluster in selected_cluster_idx)  # Ensure flexibility in column count

    fig, axs = plt.subplots(num_clusters, 1, figsize=(3 * max_columns, 2.2 * num_clusters))

    if num_clusters == 1:  # Ensure axs is iterable even for one cluster
        axs = [axs]

    for cluster_idx, cluster_eps_indices in enumerate(selected_cluster_idx):
        cluster_images = []
        for q_eps_idx in cluster_eps_indices:
            image_path = os.path.join(render_dir, f"end_{q_eps_idx}.png")  # Assuming images are saved as PNG
            if os.path.exists(image_path):
                cluster_images.append(mpimg.imread(image_path))
            else:
                print(f"Warning: Image not found for episode {q_eps_idx} at {image_path}")

        # Create a row of images for the current cluster
        if cluster_images:
            num_images = len(cluster_images)
            axs[cluster_idx].imshow(np.hstack(cluster_images[:num_images]))
            axs[cluster_idx].axis('off')
            axs[cluster_idx].set_title(f"Cluster {cluster_idx + 1}")

    # Adjust layout to minimize margins
    plt.subplots_adjust(wspace=0.0, hspace=0.05)  # Reduce space between images and rows
    plt.tight_layout(pad=0.0, rect=[0, 0, 1, 1])  # Further minimize padding

    # Save the final image plot
    output_path = os.path.join(render_dir, f"{name}_cluster_plot.png")
    plt.savefig(output_path, dpi=50*num_clusters, pad_inches=0.01)
    print(f"Cluster image plot saved at {output_path}")
     # Close the figure to release memory
    plt.close(fig)

def select_from_clusters(max_displacements, cluster_labels,  n_select=2):
    """
    Select indices of episodes per each density-based cluster. With n_select selected at the lowest,
    median, and max max displacement from each cluster.

    Parameters:
        umap_2d (np.array): UMAP-reduced 2D array of shape (num_designs, 2).
        max_displacements (np.array): Array of maximum displacements, aligned with UMAP indices.
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN parameter).
        num_core (int): Minimum number of samples in a neighborhood for a point to be considered a core point (DBSCAN parameter).
                        Higher min_samples enforces stricter density requirements, so only regions with sufficient density are identified as clusters. Lower min_samples may result in more scattered and loosely connected clusters. default value is min_samples = 4
        n_select (int): Number of indices to select near the lowest and max max displacement from each cluster.

    Returns:
        cluster_eps_indices (list of lists): 2D list where each sublist contains the indices of 2*n_select+3 episodes for a cluster.
    """

    # Initialize a 2D list for storing indices for each cluster
    cluster_eps_indices = []

    # Iterate through unique clusters (ignoring noise, labeled as -1)
    unique_clusters = set(cluster_labels) - {-1}  # Exclude noise points
    for cluster_id in unique_clusters:
        # Get indices belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Sort these indices by max displacement (ascending order)
        sorted_indices = sorted(
            cluster_indices, key=lambda idx: max_displacements[idx]
        )

        # Select n_select indices with the lowest max displacement
        low_displacement_indices = sorted_indices[:n_select]

        # Select the median value and one index below and one above
        median_idx = len(sorted_indices) // 2
        # median_displacement_indices = sorted_indices[max(0, median_idx - 1):median_idx + 2]

        # Select n_select indices near the max max displacement
        max_displacement_indices = sorted_indices[-n_select:]

        # Combine all selected indices for the current cluster
        cluster_eps_indices.append(
            # low_displacement_indices + median_displacement_indices + max_displacement_indices
            low_displacement_indices + [median_idx] + max_displacement_indices
        )

    return cluster_eps_indices

def create_interactive_performance_map_cluster(
    max_displacements,
    all_failed_elements,
    allowable_displacement,
    img_url_csv,
    umap_2d,
    cluster_labels=None,  # Added cluster_idx as an optional parameter
    selected_cluster_idx=None, # selected idx from clusters that are rendered
    gamma=20,
    num_z_ticks=6,
    marker_size=3,
    ):
    """
    Creates an interactive Dash app to visualize a 3D performance map using UMAP for dimensionality reduction.
    Displays corresponding toolpath images in a fixed location on hover, based on `img_url` in the data.csv file.

    Parameters:
        - max_displacements: np.array (num_designs,)
          Array of maximum displacements for each design.
        - all_failed_elements: list of list of int
            List of failed elements for each design.
        - allowable_displacement: float
            Maximum allowable displacement for the design.
        - img_url_csv: str
          Path to the CSV file containing columns [idx, img_url].
        - umap_2d: np.array (num_designs, 2)
            UMAP coordinates for each design.
        - cluster_idx: list of list of int
            List of indices of design instances with low mid high max deflection for each cluster from DBSCAN
            ex) [[32, 21, 52, 1, 35, 34, 48], [38, 42, 29, 7, 3, 3, 49], [14, 24, 24, 41, 11, 11, 30], [8, 28, 40, 53, 17, 17, 36], [43, 51, 51, 12, 2, 2, 6]]
            Updated to include clustering-based coloring of points.
        - gamma: float, optional
          Compression parameter for transforming max displacements.
        - num_z_ticks: int, optional
          Number of ticks to display along the z-axis using the original max displacements.
        - marker_size: int, optional
            Size of the markers in the 3D plot.

    """
    ## Load image data from CSV if provided ##
    if img_url_csv:
        data_df = pd.read_csv(img_url_csv)
    else:
        raise ValueError("img_url_csv is required")
    
    # Transform max displacements
    transformed_max_displacements = max_displacements / (1 + gamma * max_displacements)
    ## Get encoded values for each point (umap + transformed max disp) ##
    perf_encoded = np.hstack((umap_2d, transformed_max_displacements[:, None]))

    # Create ticks for the z-axis based on the number of instances in max_displacements
    sorted_max_displacements = np.sort(max_displacements)
    tick_indices = np.linspace(0, len(sorted_max_displacements) - 1, num_z_ticks, dtype=int)
    z_tick_original_values = sorted_max_displacements[tick_indices]
    z_tick_transformed_values = z_tick_original_values / (1 + gamma * z_tick_original_values)

    
    if isinstance(cluster_labels, (list, np.ndarray)):
        # print(f'cluster_labels : {cluster_labels}')
        # Map colors to cluster labels
        color_data = np.array(cluster_labels)
        colorscale = ['#D3D3D3'] + px.colors.qualitative.Light24 # Make noise (0) from color_data grey
    else:
        # Default to coloring by max deflection
        color_data = transformed_max_displacements
        colorscale = "bluyl_r"

    ## Separate failed and non-failed points ##
    idx_with_failed_elements = [i for i, failed in enumerate(all_failed_elements) if len(failed) > 0]
    percentage_design_failed = len(idx_with_failed_elements) / len(all_failed_elements) * 100
    # Separate points with failed elements
    failed_points = perf_encoded[idx_with_failed_elements]
    non_failed_points = np.delete(perf_encoded, idx_with_failed_elements, axis=0)
    print(f"failed_points : {failed_points.shape}")
    print(f"non_failed_points : {non_failed_points.shape}")
    # get original indices (later for labeling)
    org_idx = np.arange(perf_encoded.shape[0])  # Original indices
    failed_idx = org_idx[idx_with_failed_elements]
    non_failed_idx = np.delete(org_idx, idx_with_failed_elements, axis=0)

    # Create 3D scatter plot
    fig = go.Figure()

    ##  Create plane across z axis at allowable displacement  ##
    x_min, x_max = perf_encoded[:, 0].min(), perf_encoded[:, 0].max()
    y_min, y_max = perf_encoded[:, 1].min(), perf_encoded[:, 1].max()
    x_surface = np.linspace(x_min, x_max, 50)  # Grid for x
    y_surface = np.linspace(y_min, y_max, 50)  # Grid for y
    x_surface, y_surface = np.meshgrid(x_surface, y_surface)
    trans_allow_disp = allowable_displacement / (1 + gamma * allowable_displacement)
    z_surface = np.full_like(x_surface, trans_allow_disp)  # Constant z-plane
    # Add max deflection surface to the figure
    fig.add_trace(
        go.Surface(
            x=x_surface,
            y=y_surface,
            z=z_surface,
            colorscale='Reds',  # Choose a color
            opacity=0.1,  # Make it semi-transparent
            showscale=False,  # No color scale for the surface
            hoverinfo="skip",  # Disable hover info
            name='Allowable Deflection',
            visible=True  # Initially visible
        )
    )
    # Add annotation for the max deflection value next to surface
    fig.add_trace(
        go.Scatter3d(
            x=[x_min],  # Position the annotation near the edge
            y=[y_max],
            z=[trans_allow_disp],  # Match the surface value
            mode="text",
            text=[f"Allowable<br>Deflection:<br> {allowable_displacement:.3f}"],  # Plain text
            textposition="middle left",  # Adjust position relative to the point
            name="Annotation",
            showlegend=False,  # Do not include in the legend
            textfont=dict(
                color="red",  # Set text color to red
                size=12,      # Optionally adjust text size
            ),
            visible=True  # Keep annotation visible
        )
    )

    ## Add points ##
    # Add non-failed points with the colorscale
    fig.add_trace(
        go.Scatter3d(
            x=non_failed_points[:, 0],
            y=non_failed_points[:, 1],
            z=non_failed_points[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color_data[non_failed_idx],
                colorscale=colorscale,
                showscale=False, # Show the legend
            ),
            customdata=non_failed_idx,
            hoverinfo="skip",  # Disable default hover behavior
            hovertemplate=None,
            name="Non-Failed",
            visible=True  # Keep annotation visible
        )
    )
    # Add failed points with red unfilled circles
    fig.add_trace(
        go.Scatter3d(
            x=failed_points[:, 0],
            y=failed_points[:, 1],
            z=failed_points[:, 2],
            mode="markers",
            marker=dict(size=marker_size//2, color="red", symbol='x'),
            customdata=failed_idx,
            hoverinfo="skip",  # Disable default hover behavior
            hovertemplate=None,
            name='Failed',
            visible=True  # Keep annotation visible
        )
    )

    # Add black boundary circle for rendered idx selected from cluster
    if isinstance(selected_cluster_idx, (list, np.ndarray)): 
        selected_cluster_idx = np.unique(np.array(selected_cluster_idx).flatten())
        # print(f"selected_cluster_idx : {selected_cluster_idx}")
        selected_points = perf_encoded[selected_cluster_idx]
        fig.add_trace(
            go.Scatter3d(
                x=selected_points[:, 0],
                y=selected_points[:, 1],
                z=selected_points[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color='rgba(0, 0, 0, 1.0)', symbol="circle-open"),
                customdata=selected_cluster_idx,
                # customdata=np.column_stack((selected_cluster_idx, ["ignore"] * len(selected_cluster_idx))),  # Add a flag
                hoverinfo="none",  # Disable default hover behavior
                hovertemplate=None,
                name='Selected from Cluster',
                visible=True  # Keep annotation visible
            )
        )

    #  Set gridlines and axis labels
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="UMAP 1",
                showgrid=True,           # Show gridlines
                gridcolor="lightgrey",   # Set gridline color
                gridwidth=1,             # Set gridline width
                showticklabels=False,
            ),
            yaxis=dict(
                title="UMAP 2",
                showgrid=True,           # Show gridlines
                gridcolor="lightgrey",   # Set gridline color
                gridwidth=1,             # Set gridline width
                showticklabels=False,
            ),
            zaxis=dict(
                title="Max Displacement",
                showgrid=True,           # Show gridlines
                gridcolor="lightgrey",   # Set gridline color
                gridwidth=1,             # Set gridline width
                tickvals=z_tick_transformed_values,
                ticktext=[f"{val:.3f}" for val in z_tick_original_values],
            ),
            # Set the background color of the 3D scene to white
            bgcolor="white",    # Makes the entire 3D scene background white
            # Fix the camera view to lock the orientation of planes
            camera=dict(
                up=dict(x=0, y=0, z=1),  # Fix the "up" direction to z-axis
                eye=dict(x=1.25, y=1.25, z=1.25),  # Initial camera position
            ),
        ),
        title="Interactive 3D Performance Map",
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
    )

#     ## Add checkbox toggle for max allowable deflection surface ##
#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 type="buttons",  # Type of control
#                 showactive=True,  # Highlight the active button
#                 buttons=[
#                     dict(
#                         label="Show Surface",  # Button label
#                         method="update",  # Update trace properties
#                         args=[{
#                         "visible": [
#                             True,  # Allowable Max Deflection Surface
#                             True,  # Allowable Max Deflection Annotation
#                             True,  # Non-Failed Points
#                             True,  # Failed Points
#                             True,  # selected_cluster_idx
#                         ]
#                         }],
#                     ),
#                     dict(
#                         label="Hide Surface",  # Button label
#                         method="update",
#                         args=[{
#                         "visible": [
#                             False,  # Allowable Max Deflection Surface
#                             False,  # Allowable Max Deflection Annotation
#                             True,  # Non-Failed Points
#                             True,  # Failed Points
#                             True,  # selected_cluster_idx
#                         ]
#                         }],
#                     ),
#                 ],
#                 direction="left",  # Horizontal layout
#                 pad={"r": 10, "t": 10},  # Padding for alignment
#                 x=0.1,  # Position on x-axis (10% from left)
#                 # y=1.15,  # Position on y-axis (above the plot)
#                 y=0.15,  # Position on y-axis (above the plot)
#             )
#         ]
#     )

    ## Add checkbox toggle to hide failed points ##
    fig.update_layout(
    updatemenus=[
        # Checkbox-style toggle for failed points
        dict(
            type="dropdown",  # Use dropdown for checkbox-style behavior
            direction="up",  # Dropdown expands downward
            showactive=True,   # Highlight active selection
            buttons=[
                dict(
                    label="Show Failed Points + Deflection Surface",
                    method="update",
                    args=[{
                        "visible": [
                            True,  # Allowable Max Deflection Surface
                            True,  # Allowable Max Deflection Annotation
                            True,  # Non-Failed Points
                            True,  # Failed Points (show)
                            True,  # selected_cluster_idx
                        ]
                    }],
                ),
                dict(
                    label="Hide Failed Points + Deflection Surface",
                    method="update",
                    args=[{
                        "visible": [
                            False,  # Allowable Max Deflection Surface
                            False,  # Allowable Max Deflection Annotation
                            True,  # Non-Failed Points
                            False,  # Failed Points (hide)
                            True,  # selected_cluster_idx
                        ]
                    }],
                ),
            ],
            pad={"r": 10, "t": 10},  # Padding for alignment
            x=0.1,  # Position on x-axis (10% from left)
            y=0.15,  # Position on y-axis (above the plot)
        ),
    ]
)


    ## Initialize Dash app ##
    app = Dash(__name__)
    app.layout = html.Div(
        style={"display": "flex", "justify-content": "center", "align-items": "center", "height": "100vh"},
        children=[
            html.Div(
                [
                    # dcc.Graph(id="performance-map", figure=fig, clear_on_unhover=True),
                    dcc.Graph(
                            id="performance-map",
                            figure=fig,
                            clear_on_unhover=True,
                            style={
                                "width": "1000px",  # Full width of the container
                                "height": "700px",  # Set a fixed height
                            },
                        ),
                    html.P(
                        f"{percentage_design_failed:.2f}% of designs have failed elements.",
                        style={
                            "font-size": "16px",
                            "color": "#ff4d4d",  # Red text to indicate failure
                            "margin-top": "10px",
                            "text-align": "center",
                        },
                    ),
                ],
                style={
                    "width": "55%",
                    "height": "100vh",
                    "display": "flex",
                    "flex-direction": "column",  # Stack graph and text vertically
                    "justify-content": "center",
                    "align-items": "center",
                },
            ),
            html.Div(
                id="image-display",  # Placeholder for the image and text
                style={
                    "width": "45%",  # Fixed width for the image display section
                    "height": "auto",  # Automatically adjusts to content
                    "padding": "5px",
                    "border": "0px solid #ccc",
                    "background-color": "#f9f9f9",
                    "text-align": "center",
                    "display": "flex",
                    "justify-content": "center",  # Center horizontally
                    "align-items": "center",  # Center vertically
                    "flex-direction": "column",  # Stack content vertically
                },
                children="Hover over a point to see details.",
            ),
        ],
    )

    ## Define hover callback (display images on hover) ##
    @callback(
        Output("image-display", "children"),
        Input("performance-map", "hoverData"),
    )
    
    def display_hover(hoverData):
        # not hovering over a point
        if hoverData is None:
            return "Hover over a point to see details."
        # Extract hovered point index
        point_data = hoverData["points"][0]
        idx = point_data["customdata"]
        # Get corresponding row in imgurl csv
        try:
            row = data_df.loc[idx]
            img_url = row["img_url"]
            # Check if img_url is None
            if pd.isna(img_url):
            # if img_url is None:
                # Display a text message for indices where images were not queried 
                # return f"Index: {int(row['idx'])} \n No image is available."
                return html.Div([
                    html.P(
                        "No image available",
                        style={
                            "color": "black",
                            "font-size": "18px",
                            "text-align": "center",
                            "margin-top": "20px",
                        }
                    ),
                    html.P(
                        f"Index: {int(row['idx'])}",
                        style={"font-weight": "bold", "margin-top": "10px", "text-align": "center"}
                    ),
                ])
            else:
                # Display the actual image
                return html.Div([
                    html.Img(src=img_url, style={"width": "100%", "max-width": "1200px", "height": "auto"}),  # Image
                    html.P(f"Index: {int(row['idx'])}", style={"font-weight": "bold", "margin-top": "40px"}),
                ])
        except KeyError:
            # Handle missing or invalid index
            return f"Invalid point selected {int(idx)}."

    ## Run the app ##
    app.run(debug=False)

def create_interactive_performance_map(
    max_displacements, all_failed_elements, allowable_displacement, img_url_csv, umap_2d, gamma=20, num_z_ticks=6, marker_size=3):
    """
    Creates an interactive Dash app to visualize a 3D performance map using UMAP for dimensionality reduction.
    Displays corresponding toolpath images in a fixed location on hover, based on `img_url` in the data.csv file.

    Parameters:
        - max_displacements: np.array (num_designs,)
          Array of maximum displacements for each design.
        - all_failed_elements: list of list of int
            List of failed elements for each design.
        - allowable_displacement: float
            Maximum allowable displacement for the design.
        - img_url_csv: str
          Path to the CSV file containing columns [idx, img_url].
        - umap_2d: np.array (num_designs, 2)
            UMAP coordinates for each design.
        - cluster_idx: list of list of int
            List of indices of design instances with low mid high max deflection for each cluster from DBSCAN
            ex) [[32, 21, 52, 1, 35, 34, 48], [38, 42, 29, 7, 3, 3, 49], [14, 24, 24, 41, 11, 11, 30], [8, 28, 40, 53, 17, 17, 36], [43, 51, 51, 12, 2, 2, 6]]
        - gamma: float, optional
          Compression parameter for transforming max displacements.
        - num_z_ticks: int, optional
          Number of ticks to display along the z-axis using the original max displacements.
        - marker_size: int, optional
            Size of the markers in the 3D plot.

    """

    ## Load image data from CSV if provided ##
    if img_url_csv:
        data_df = pd.read_csv(img_url_csv)
    else:
        raise ValueError("img_url_csv is required")
    
    # Transform max displacements
    transformed_max_displacements = max_displacements / (1 + gamma * max_displacements)
    
    ## Get encoded values for each point (umap + transformed max disp) ##
    perf_encoded = np.hstack((umap_2d, transformed_max_displacements[:, None]))

    # Create ticks for the z-axis based on the number of instances in max_displacements
    sorted_max_displacements = np.sort(max_displacements)
    tick_indices = np.linspace(0, len(sorted_max_displacements) - 1, num_z_ticks, dtype=int)
    z_tick_original_values = sorted_max_displacements[tick_indices]
    z_tick_transformed_values = z_tick_original_values / (1 + gamma * z_tick_original_values)

    ## Separate failed and non-failed points ##
    # get elements that have failed elements 
    idx_with_failed_elements = [i for i, failed in enumerate(all_failed_elements) if len(failed) > 0]
    # get percentage of designs with failed element
    percentage_design_failed = len(idx_with_failed_elements)/len(all_failed_elements) * 100
    # Separate points with failed elements
    failed_points = perf_encoded[idx_with_failed_elements]
    non_failed_points = np.delete(perf_encoded, idx_with_failed_elements, axis=0)
    # get original indices (later for labeling)
    org_idx = np.arange(perf_encoded.shape[0])  # Original indices
    failed_idx = org_idx[idx_with_failed_elements]
    non_failed_idx = np.delete(org_idx, idx_with_failed_elements, axis=0)

    # Create a 3D scatter plot using Plotly
    fig = go.Figure()

    ##  Create plane across z axis at allowable displacement  ##
    x_min, x_max = perf_encoded[:, 0].min(), perf_encoded[:, 0].max()
    y_min, y_max = perf_encoded[:, 1].min(), perf_encoded[:, 1].max()
    x_surface = np.linspace(x_min, x_max, 50)  # Grid for x
    y_surface = np.linspace(y_min, y_max, 50)  # Grid for y
    x_surface, y_surface = np.meshgrid(x_surface, y_surface)
    trans_allow_disp = allowable_displacement / (1 + gamma * allowable_displacement)
    z_surface = np.full_like(x_surface, trans_allow_disp)  # Constant z-plane

    # Add max deflection surface to the figure
    fig.add_trace(
        go.Surface(
            x=x_surface,
            y=y_surface,
            z=z_surface,
            colorscale='Reds',  # Choose a color
            opacity=0.1,  # Make it semi-transparent
            showscale=False,  # No color scale for the surface
            hoverinfo="skip",  # Disable hover info
            name='Allowable Deflection',
            visible=True  # Initially visible
        )
    )
    # Add annotation for the max deflection value next to surface
    fig.add_trace(
        go.Scatter3d(
            x=[x_min],  # Position the annotation near the edge
            y=[y_max + 0.5],
            z=[trans_allow_disp],  # Match the surface value
            mode="text",
            text=[f"Allowable<br>Deflection:<br> {allowable_displacement:.3f}"],  # Plain text
            textposition="middle left",  # Adjust position relative to the point
            name="Annotation",
            showlegend=False,  # Do not include in the legend
            textfont=dict(
                color="red",  # Set text color to red
                size=12,      # Optionally adjust text size
            ),
            visible=True  # Keep annotation visible
        )
    )


    ## Add points ##
    # Add non-failed points with the colorscale
    fig.add_trace(
        go.Scatter3d(
            x=non_failed_points[:, 0],  # UMAP 1
            y=non_failed_points[:, 1],  # UMAP 2
            z=non_failed_points[:, 2],  # Transformed Max Displacement
            mode='markers',
            marker=dict(
                size=marker_size,
                color=non_failed_points[:, 2],  # Color by transformed max displacement
                colorscale='bluyl_r',
                showscale=False,
                colorbar=dict(
                    title="Max Displacement",
                    tickvals=z_tick_transformed_values,  # Show ticks for transformed values
                    ticktext=[f"{val:.3f}" for val in z_tick_original_values],  # Use original values as labels
                ),
            ),
            customdata=non_failed_idx,  # Attach original indices as custom data
            hoverinfo="none",  # Disable default hover behavior
            hovertemplate=None,
            name='Non-Failed',
            visible=True  # Keep annotation visible
        )
    )

    # Add failed points with red unfilled circles
    fig.add_trace(
        go.Scatter3d(
            x=failed_points[:, 0],  # UMAP 1
            y=failed_points[:, 1],  # UMAP 2
            z=failed_points[:, 2],  # Transformed Max Displacement
            mode='markers',
            marker=dict(
                size=marker_size,  # Slightly larger for visibility
                color='red',  # Fixed red color
                symbol='circle-open',  # Unfilled circle
            ),
            customdata=failed_idx,  # Attach original indices as custom data
            hoverinfo="none",  # Disable default hover behavior
            hovertemplate=None,
            name='Failed',
            visible=True  # Keep annotation visible
        )
    )
    ## Add all points (including failed) filled
    # # Create a 3D scatter plot using Plotly
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=performance_map[:, 0],  # UMAP 1
    #         y=performance_map[:, 1],  # UMAP 2
    #         z=performance_map[:, 2],  # Transformed Max Displacement
    #         mode='markers',
    #         marker=dict(
    #             size=6,
    #             color=performance_map[:, 2],  # Color by transformed max displacement
    #             colorscale='bluyl_r',
    #             showscale=False,
    #             colorbar=dict(
    #                 title="Max Displacement",
    #                 tickvals=z_tick_transformed_values,  # Show ticks for transformed values
    #                 ticktext=[f"{val:.3f}" for val in z_tick_original_values],  # Use original values as labels
    #             ),
    #         ),
    #         hoverinfo="none",  # Disable default hover behavior
    #         hovertemplate=None,
    #     )
    # )

    # Set gridlines and axis labels
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="UMAP 1",
                showgrid=True,           # Show gridlines
                gridcolor="lightgrey",   # Set gridline color
                gridwidth=1,             # Set gridline width
                showticklabels=False,
            ),
            yaxis=dict(
                title="UMAP 2",
                showgrid=True,           # Show gridlines
                gridcolor="lightgrey",   # Set gridline color
                gridwidth=1,             # Set gridline width
                showticklabels=False,
            ),
            zaxis=dict(
                title="Max Displacement",
                showgrid=True,           # Show gridlines
                gridcolor="lightgrey",   # Set gridline color
                gridwidth=1,             # Set gridline width
                tickvals=z_tick_transformed_values,
                ticktext=[f"{val:.3f}" for val in z_tick_original_values],
            ),
        ),
        title="Interactive 3D Performance Map",
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
    )

    ## Add checkbox toggle for max allowable deflection surface ##
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",  # Type of control
                showactive=True,  # Highlight the active button
                buttons=[
                    dict(
                        label="Show Surface",  # Button label
                        method="update",  # Update trace properties
                        args=[{
                        "visible": [
                            True,  # Allowable Max Deflection Surface
                            True,  # Allowable Max Deflection Annotation
                            True,  # Non-Failed Points
                            True,  # Failed Points
                        ]
                        }],
                    ),
                    dict(
                        label="Hide Surface",  # Button label
                        method="update",
                        args=[{
                        "visible": [
                            False,  # Allowable Max Deflection Surface
                            False,  # Allowable Max Deflection Annotation
                            True,  # Non-Failed Points
                            True,  # Failed Points
                        ]
                        }],
                    ),
                ],
                direction="left",  # Horizontal layout
                pad={"r": 10, "t": 10},  # Padding for alignment
                x=0.1,  # Position on x-axis (10% from left)
                # y=1.15,  # Position on y-axis (above the plot)
                y=0.15,  # Position on y-axis (above the plot)
            )
        ]
    )

    ## Initialize Dash app ##
    app = Dash(__name__)
    app.layout = html.Div(
        style={"display": "flex", "justify-content": "center", "align-items": "center", "height": "100vh"},
        children=[
            html.Div(
                [
                    # dcc.Graph(id="performance-map", figure=fig, clear_on_unhover=True),
                    dcc.Graph(
                            id="performance-map",
                            figure=fig,
                            clear_on_unhover=True,
                            style={
                                "width": "80%",  # Full width of the container
                                "height": "700px",  # Set a fixed height
                            },
                        ),
                    html.P(
                        f"{percentage_design_failed:.2f}% of designs have failed elements.",
                        style={
                            "font-size": "16px",
                            "color": "#ff4d4d",  # Red text to indicate failure
                            "margin-top": "10px",
                            "text-align": "center",
                        },
                    ),
                ],
                style={
                    "width": "50%",
                    "height": "100vh",
                    "display": "flex",
                    "flex-direction": "column",  # Stack graph and text vertically
                    "justify-content": "center",
                    "align-items": "center",
                },
            ),
            html.Div(
                id="image-display",  # Placeholder for the image and text
                style={
                    "width": "1200px",  # Fixed width for the image display section
                    "height": "auto",  # Automatically adjusts to content
                    "padding": "5px",
                    "border": "0px solid #ccc",
                    "background-color": "#f9f9f9",
                    "text-align": "center",
                    "display": "flex",
                    "justify-content": "center",  # Center horizontally
                    "align-items": "center",  # Center vertically
                    "flex-direction": "column",  # Stack content vertically
                },
                children="Hover over a point to see details.",
            ),
        ],
    )

    ## Define hover callback (display images on hover) ##
    @callback(
        Output("image-display", "children"),
        Input("performance-map", "hoverData"),
    )
    
    def display_hover(hoverData):
        # not hovering over a point
        if hoverData is None:
            return "Hover over a point to see details."
        # Extract hovered point index
        point_data = hoverData["points"][0]
        idx = point_data["customdata"]
        # Get corresponding row in imgurl csv
        try:
            row = data_df.loc[idx]
            img_url = row["img_url"]
            # Check if img_url is None
            if pd.isna(img_url):
                return html.Div([
                    html.P(
                        "No image available",
                        style={
                            "color": "black",
                            "font-size": "18px",
                            "text-align": "center",
                            "margin-top": "20px",
                        }
                    ),
                    html.P(
                        f"Index: {int(row['idx'])}",
                        style={"font-weight": "bold", "margin-top": "10px", "text-align": "center"}
                    ),
                ])
            else:
                # Display the actual image
                return html.Div([
                    html.Img(src=img_url, style={"width": "100%", "max-width": "1200px", "height": "auto"}),  # Image
                    html.P(f"Index: {int(row['idx'])}", style={"font-weight": "bold", "margin-top": "40px"}),
                ])
        except KeyError:
            # Handle missing or invalid index
            return f"Invalid point selected {int(idx)}."

    ## Run the app ##
    app.run(debug=True)