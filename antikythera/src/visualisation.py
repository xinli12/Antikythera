import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_hole_locations(data, ax=None, colors=None, excluded_sections=None):
    """
    Plot the hole locations with different colors for each section.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'hole_id', 'section', 'x', 'y' containing the data
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    colors : list, optional
        List of colors to use for the sections
    excluded_sections : list, optional
        List of section IDs to exclude from plotting

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Get unique sections
    sections = jnp.unique(data["section"])

    # Create a colormap
    if colors is None:
        colors_list = [
            "#E89BCF",  # Vivid Pink
            "#A8CFF0",  # Bright Sky Blue
            "#C8E2A7",  # Fresh Mint Green
            "#F0C7A1",  # Warm Peach
            "#B8BDF0",  # Lively Lavender
            "#E6A8D7",  # Soft Magenta
            "#A1E8D1",  # Aqua Pop
            "#E8D07A",  # Golden Vanilla
        ]
        colors = colors_list

    # Default excluded_sections to empty list if not provided
    if excluded_sections is None:
        excluded_sections = []

    # Plot each section with a different color
    for i, section in enumerate(sections):
        # Skip excluded sections
        section_id = int(section)
        if section_id in excluded_sections:
            continue

        mask = data["section"] == section
        ax.scatter(
            data["x"][mask],
            data["y"][mask],
            color=colors[section_id % len(colors)],  # Use section ID directly to index colors
            label=f"Section {section}",
            alpha=0.8,
        )

        # Add hole numbers
        for hole_id, x, y in zip(data["hole_id"][mask], data["x"][mask], data["y"][mask]):
            ax.annotate(
                str(hole_id),
                (x, y),
                xytext=(3, 7),
                textcoords="offset points",
                fontsize=6,
            )

    # Set plot properties
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    # ax.set_title('Antikythera Mechanism Calendar Ring Hole Locations')
    ax.legend()

    return ax


def plot_model_predictions(
    data,
    params,
    model,
    ax=None,
    color="red",
    marker="+",
    alpha=0.6,
    label=None,
    excluded_sections=None,
):
    """
    Plot the model predictions for the hole locations.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'hole_id', 'section', 'x', 'y' containing the data
    params : dict
        Dictionary of model parameters
    model : AntikytheraModel
        Model object
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    color : str, optional
        Color for the model predictions
    marker : str, optional
        Marker style for the model predictions
    alpha : float, optional
        Alpha value for the model predictions
    label : str, optional
        Label for the model predictions
    excluded_sections : list, optional
        List of section IDs to exclude from plotting

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Default excluded_sections to empty list if not provided
    if excluded_sections is None:
        excluded_sections = []

    # Get model predictions for each hole
    x_pred = []
    y_pred = []

    for i, hole_id in enumerate(data["hole_id"]):
        # Skip excluded sections
        section = data["section"][i]
        if int(section) in excluded_sections:
            continue

        m_i = model.get_model_prediction(params, hole_id)
        x_pred.append(m_i[0])
        y_pred.append(m_i[1])

    # Plot the model predictions
    label = label if label is not None else "Model Predictions"
    ax.scatter(
        x_pred,
        y_pred,
        color=color,
        marker=marker,
        s=30,
        linewidth=0.7,
        alpha=alpha,
        label=label,
    )
    ax.legend(loc="upper right")

    return ax


def plot_sectioned_model(
    data,
    model,
    params,
    posterior_samples=None,
    n_post_samples=50,
    highlight_hole=37,
    figsize=(12, 6),
    excluded_sections=[0, 4],
):
    """
    Plot the measured and modelled hole positions with posterior predictive samples.
    Sections are aligned and stacked for easier viewing.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'hole_id', 'section', 'x', 'y' containing the data
    model : BaseAntikytheraModel
        The model instance
    params : dict
        Dictionary of model parameters (typically median of posterior samples)
    posterior_samples : list or dict, optional
        List of parameter dictionaries from posterior sampling or
        dictionary with parameter arrays
    n_post_samples : int, optional
        Number of posterior samples to display for each hole
    highlight_hole : int, optional
        ID of the hole to highlight in a magnified view
    figsize : tuple, optional
        Figure size (width, height)
    excluded_sections : list, optional
        List of section IDs that were excluded from model fitting

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, width_ratios=[3, 3, 3, 2])

    # Main plot
    ax_main = fig.add_subplot(gs[:, :-1])

    # Magnified view in top right
    ax_mag = fig.add_subplot(gs[:, -1])

    # Unique sections and vertical spacing
    sections = jnp.unique(data["section"])
    spacing = 0.0  # vertical spacing between sections in mm

    # Colors
    measured_color = "gray"
    model_color = "red"

    # Define colors list for sections
    colors_list = [
        "#E89BCF",  # Vivid Pink
        "#A8CFF0",  # Bright Sky Blue
        "#C8E2A7",  # Fresh Mint Green
        "#F0C7A1",  # Warm Peach
        "#B8BDF0",  # Lively Lavender
        "#E6A8D7",  # Soft Magenta
        "#A1E8D1",  # Aqua Pop
        "#E8D07A",  # Golden Vanilla
    ]

    # Section offsets (for stacking)
    offsets = {}
    for i, section in enumerate(sections):
        section_id = int(section)
        offsets[section_id] = i * spacing

    # Store the highlight hole data
    highlight_data = None

    # Plot each section
    for section in sections:
        section_id = int(section)
        offset = offsets[section_id]

        # Get holes for this section
        mask = data["section"] == section
        section_holes = [int(h) for h in data["hole_id"][mask]]
        x_values = data["x"][mask]
        y_values = data["y"][mask] - offset  # Apply vertical offset

        # Get section color - use section_id to directly index the colors list
        posterior_color = colors_list[section_id % len(colors_list)]

        # Plot measured positions (grey circles)
        ax_main.scatter(x_values, y_values, s=50, facecolor="none", edgecolor=measured_color, linewidth=1, alpha=0.8)

        # # Label each section
        # ax_main.text(0, -offset + 0.5, f"S{section_id}", fontsize=10)

        # Add hole numbers
        for hole_id, x, y in zip(section_holes, x_values, y_values):
            ax_main.annotate(
                str(hole_id),
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

        # Skip model predictions for excluded sections
        if section_id in excluded_sections:
            continue

        # Plot model predictions and posterior samples
        for hole_id, x, y in zip(section_holes, x_values, y_values):
            # Model prediction (red cross)
            pred = model.get_model_prediction(params, hole_id)
            pred_x, pred_y = pred[0], pred[1] - offset

            ax_main.plot(pred_x, pred_y, "+", color=model_color, markersize=8, markeredgewidth=0.7)

            # Posterior predictive samples (using section-specific color)
            if posterior_samples is not None:
                post_x = []
                post_y = []

                # Check if posterior_samples is a dictionary or list
                if isinstance(posterior_samples, dict):
                    # Handle dictionary case - all samples stored as arrays
                    sample_count = len(posterior_samples[list(posterior_samples.keys())[0]])
                    indices = np.linspace(0, sample_count - 1, n_post_samples).astype(int)

                    for idx in indices:
                        # Extract parameters for this sample
                        sample_dict = {}
                        for key, values in posterior_samples.items():
                            sample_dict[key] = values[idx]

                        post_pred = model.get_model_prediction(sample_dict, hole_id)
                        post_x.append(post_pred[0])
                        post_y.append(post_pred[1] - offset)
                else:
                    # Original list case
                    indices = np.linspace(0, len(posterior_samples) - 1, n_post_samples).astype(int)

                    for idx in indices:
                        # Convert numpy int to python int for indexing
                        idx_int = int(idx)
                        sample_params = posterior_samples[idx_int]
                        post_pred = model.get_model_prediction(sample_params, hole_id)
                        post_x.append(post_pred[0])
                        post_y.append(post_pred[1] - offset)

                ax_main.scatter(post_x, post_y, s=5, color=posterior_color, alpha=0.3)

                # Store highlight hole data
                if hole_id == highlight_hole:
                    highlight_data = {
                        "hole_id": hole_id,
                        "measured": (x, y),
                        "predicted": (pred_x, pred_y),
                        "posterior_x": post_x,
                        "posterior_y": post_y,
                        "posterior_color": posterior_color,
                    }

                    # Draw highlight circle
                    highlight_circle = patches.Circle(
                        (x, y), 2, fill=False, edgecolor="red", linestyle="--", linewidth=0.5
                    )
                    ax_main.add_patch(highlight_circle)

    # Configure main plot
    ax_main.set_xlabel("distance (mm)")
    ax_main.set_ylabel("distance (mm)")
    ax_main.set_aspect("equal")
    ax_main.grid(alpha=0.3)

    # Create magnified view of highlighted hole
    if highlight_data:
        # Clear the axis
        ax_mag.clear()

        # Extract data
        h_id = highlight_data["hole_id"]
        h_x, h_y = highlight_data["measured"]
        h_pred_x, h_pred_y = highlight_data["predicted"]
        h_post_x = highlight_data["posterior_x"]
        h_post_y = highlight_data["posterior_y"]
        h_post_color = highlight_data["posterior_color"]

        # Plot measured position (grey circle)
        ax_mag.scatter(h_x, h_y, s=250, facecolor="none", edgecolor=measured_color, linewidth=2)

        # Plot model prediction (red cross)
        ax_mag.plot(h_pred_x, h_pred_y, "+", color=model_color, markersize=24, markeredgewidth=1)

        # Plot posterior samples (using section color)
        ax_mag.scatter(h_post_x, h_post_y, s=20, color=h_post_color, alpha=0.3)

        # Add hole number
        ax_mag.annotate(
            str(h_id),
            (h_x, h_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
        )

        # Set up the magnified view
        ax_mag.set_title(f"Hole {h_id}")

        # Zoom in around the hole
        zoom_radius = 0.7
        ax_mag.set_xlim(h_x - zoom_radius, h_x + zoom_radius)
        ax_mag.set_ylim(h_y - zoom_radius, h_y + zoom_radius)
        ax_mag.set_aspect("equal")
        ax_mag.grid(alpha=0.3)

        # # Add reference circle
        # gray_circle = patches.Circle(
        #     (h_x, h_y), 0.3, fill=False,
        #     edgecolor=measured_color, linewidth=1.5
        # )
        # ax_mag.add_patch(gray_circle)

        # Add dashed lines to the center
        ax_mag.axhline(y=h_y, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_mag.axvline(x=h_x, color="green", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add legend with section-specific colors
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=measured_color,
            markersize=10,
            label="Measured",
            linestyle="None",
        ),
        Line2D([0], [0], marker="+", color=model_color, markersize=10, label="Model", linestyle="None"),
    ]

    # Add color entries for each non-excluded section
    for section in sections:
        section_id = int(section)
        if section_id not in excluded_sections:
            section_color = colors_list[section_id % len(colors_list)]
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=section_color,
                    markersize=6,
                    alpha=0.5,
                    label=f"Section {section_id}",
                    linestyle="None",
                )
            )

    ax_main.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig
