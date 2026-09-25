import numpy as np
import plotly.graph_objects as go


def visualize_unc_map(
    unc_metric: np.ndarray, output_file_path: str, map_type: str
) -> None:
    """
    Visualizes a 3D uncertainty map as an interactive HTML file with a slider to navigate through slices.
    Args:
      unc_metric (numpy.ndarray): A 3D array representing the uncertainty metric with dimensions (width, height, depth).
      output_file_path (str): The file path where the generated HTML visualization will be saved.
      map_type (str): Type of uncertainty map to add in the figure title
    Returns:
      None: The function saves the visualization as an HTML file at the specified output path.
    Notes:
      - The input "unc_metric" is expected to have dimensions (width, height, depth) and will be transposed to (depth, width, height).
      - The visualization includes a slider to navigate through the slices of the 3D uncertainty map.
      - The "map_type" parameter determines the color scheme of the visualization.
      - The output HTML file can be opened in a web browser to view the interactive visualization.
    """

    unc_metric = np.transpose(unc_metric, axes=(2, 0, 1))
    rows, columns = unc_metric[0].shape
    n_slices = unc_metric.shape[0]
    height = (n_slices - 1) / 10
    grid = np.linspace(0, height, n_slices)
    slice_step = grid[1] - grid[0]
    initial_slice = go.Surface(
        z=height * np.ones((rows, columns)),
        surfacecolor=np.flipud(unc_metric[-1]),
        colorscale="hot",
        showscale=True,
    )
    frames = [
        go.Frame(
            data=[
                dict(
                    type="surface",
                    z=(height - k * slice_step) * np.ones((rows, columns)),
                    surfacecolor=unc_metric[-1 - k],
                )
            ],
            name=f"frame{k+1}",
        )
        for k in range(1, n_slices)
    ]
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"frame{k+1}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=40, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f"{k+1}",
                )
                for k in range(n_slices)
            ],
            active=17,
            transition=dict(duration=0),
            x=0,  # Slider starting position
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="slice: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )  # Slider length
    ]
    layout3d = dict(
        title_text="Predictive uncertainty map",
        title_x=0.5,
        width=600,
        height=600,
        scene_zaxis_range=[-0.1, n_slices / 10],
        sliders=sliders,
    )
    fig = go.Figure(data=[initial_slice], layout=layout3d, frames=frames)
    fig.write_html(output_file_path)
