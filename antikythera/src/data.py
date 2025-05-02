import jax.numpy as jnp
import pandas as pd


def load_data(file_path):
    """
    Load the hole measurement data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file

    Returns
    -------
    data : dict
        Dictionary with keys 'hole_id', 'section', 'x', 'y' containing the data
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Rename columns for consistency
    df = df.rename(
        columns={
            "Hole": "hole_id",
            "Section ID": "section",
            "Mean(X)": "x",
            "Mean(Y)": "y",
        },
    )

    # Convert to numpy arrays
    data = {
        "hole_id": jnp.array(df["hole_id"]),
        "section": jnp.array(df["section"]),
        "x": jnp.array(df["x"]),
        "y": jnp.array(df["y"]),
    }

    return data
