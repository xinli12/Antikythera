import jax.numpy as jnp
import numpy as np

from antikythera.src.data import load_data
from antikythera.src.model import SectionedModel, setup_optimization


def calculate_analytical_derivatives_isotropic(model, params):
    """
    Calculate analytical derivatives for the isotropic model.
    Based on the formulas from cw.md.
    """
    derivatives = {}
    sigma = params["sigma"]
    r = params["r"]
    N = params["N"]

    # Initialize derivatives
    derivatives["sigma"] = 0.0
    derivatives["r"] = 0.0
    derivatives["N"] = 0.0

    for section in model.section_ids_int:
        derivatives[f"x0_{section}"] = 0.0
        derivatives[f"y0_{section}"] = 0.0
        derivatives[f"alpha_{section}"] = 0.0

    # Calculate derivatives for each hole
    for hole_id in model.hole_ids:
        section = model.hole_to_section[int(hole_id)]
        idx = model.hole_to_idx[int(hole_id)]

        # Data point
        x_i, y_i = model.data["x"][idx], model.data["y"][idx]

        # Model prediction
        m_i = model.get_model_prediction(params, hole_id)
        x_ij, y_ij = m_i[0], m_i[1]

        # Error vector
        error_x = x_i - x_ij
        error_y = y_i - y_ij

        # Angular position
        phi_ij = 2 * np.pi * (hole_id - 1) / N + params[f"alpha_{section}"]

        # Center position derivatives
        derivatives[f"x0_{section}"] += (1.0 / sigma**2) * error_x
        derivatives[f"y0_{section}"] += (1.0 / sigma**2) * error_y

        # Alpha derivative
        derivatives[f"alpha_{section}"] += (r / sigma**2) * (error_y * np.cos(phi_ij) - error_x * np.sin(phi_ij))

        # Radius derivative
        derivatives["r"] += (1.0 / sigma**2) * (error_x * np.cos(phi_ij) + error_y * np.sin(phi_ij))

        # Sigma derivative contribution
        derivatives["sigma"] += (1.0 / sigma**3) * (error_x**2 + error_y**2)

        # N derivative contribution
        derivatives["N"] += (
            (2 * np.pi * r / (sigma**2 * N**2)) * (hole_id - 1) * (error_x * np.sin(phi_ij) - error_y * np.cos(phi_ij))
        )

    # Complete sigma derivative
    derivatives["sigma"] -= (2 * model.n_holes) / sigma

    return derivatives


def calculate_analytical_derivatives_radial_tangential(model, params):
    """
    Calculate analytical derivatives for the radial/tangential model.
    Based on the formulas from cw.md.
    """
    derivatives = {}
    sigma_r = params["sigma_r"]
    sigma_t = params["sigma_t"]
    r = params["r"]
    N = params["N"]

    # Initialize derivatives
    derivatives["sigma_r"] = 0.0
    derivatives["sigma_t"] = 0.0
    derivatives["r"] = 0.0
    derivatives["N"] = 0.0

    for section in model.section_ids_int:
        derivatives[f"x0_{section}"] = 0.0
        derivatives[f"y0_{section}"] = 0.0
        derivatives[f"alpha_{section}"] = 0.0

    # Calculate derivatives for each hole
    for hole_id in model.hole_ids:
        section = model.hole_to_section[int(hole_id)]
        idx = model.hole_to_idx[int(hole_id)]

        # Data point
        x_i, y_i = model.data["x"][idx], model.data["y"][idx]

        # Model prediction
        m_i = model.get_model_prediction(params, hole_id)
        x_ij, y_ij = m_i[0], m_i[1]

        # Get radial and tangential unit vectors
        r_hat, t_hat = model.get_radial_tangential_vectors(params, hole_id)

        # Error vector
        error_vec = jnp.array([x_i - x_ij, y_i - y_ij])
        e_r = jnp.dot(error_vec, r_hat)  # Radial error
        e_t = jnp.dot(error_vec, t_hat)  # Tangential error

        # Angular position
        phi_ij = 2 * np.pi * (hole_id - 1) / N + params[f"alpha_{section}"]

        # Center position derivatives
        derivatives[f"x0_{section}"] += (e_r / sigma_r**2) * np.cos(phi_ij) + (e_t / sigma_t**2) * np.sin(phi_ij)
        derivatives[f"y0_{section}"] += (e_r / sigma_r**2) * np.sin(phi_ij) - (e_t / sigma_t**2) * np.cos(phi_ij)

        # Alpha derivative
        derivatives[f"alpha_{section}"] += (e_r * e_t / sigma_r**2) - (e_t * (r + e_r) / sigma_t**2)

        # Radius derivative
        derivatives["r"] += e_r / sigma_r**2

        # Sigma derivatives
        derivatives["sigma_r"] += e_r**2 / sigma_r**3
        derivatives["sigma_t"] += e_t**2 / sigma_t**3

        # N derivative
        derivatives["N"] += (
            (-2 * np.pi / N**2) * (hole_id - 1) * ((e_r * e_t / sigma_r**2) - (e_t * (r + e_r) / sigma_t**2))
        )

    # Complete sigma derivatives
    derivatives["sigma_r"] -= model.n_holes / sigma_r
    derivatives["sigma_t"] -= model.n_holes / sigma_t

    return derivatives


def isotropic_derivatives_test():
    """Test that JAX derivatives match analytical derivatives for isotropic model."""
    # Load actual data from CSV file
    data_path = "antikythera/data/1-Fragment_C_Hole_Measurements.csv"
    data = load_data(data_path)

    # Initialize model
    model = SectionedModel(data)

    # Set up parameters for all sections (just guesses)
    params_dict = {
        "r": 77.0,
        "N": 355.0,
        "sigma": 0.1,
    }

    # Add parameters for each section
    for section in model.section_ids_int:
        params_dict[f"x0_{section}"] = 80.0
        params_dict[f"y0_{section}"] = 136.0
        params_dict[f"alpha_{section}"] = 3.7

    # Convert params_dict to vector for JAX gradient
    params_vec = []
    params_vec.append(params_dict["N"])
    params_vec.append(params_dict["r"])
    params_vec.append(params_dict["sigma"])

    for section in model.section_ids_int:
        params_vec.append(params_dict[f"x0_{section}"])
        params_vec.append(params_dict[f"y0_{section}"])
        params_vec.append(params_dict[f"alpha_{section}"])

    params_vec = jnp.array(params_vec)

    # Set up optimization
    _, grad_func = setup_optimization(model, "isotropic")

    # Calculate JAX gradients
    jax_gradients = grad_func(params_vec)

    # Calculate analytical gradients
    analytical_gradients = calculate_analytical_derivatives_isotropic(model, params_dict)

    # Create a vector of analytical gradients in the same order as JAX gradients
    analytical_vec = []
    analytical_vec.append(analytical_gradients["N"])
    analytical_vec.append(analytical_gradients["r"])
    analytical_vec.append(analytical_gradients["sigma"])

    for section in model.section_ids_int:
        analytical_vec.append(analytical_gradients[f"x0_{section}"])
        analytical_vec.append(analytical_gradients[f"y0_{section}"])
        analytical_vec.append(analytical_gradients[f"alpha_{section}"])

    analytical_vec = jnp.array(analytical_vec)

    # Test that they match within tolerance
    # Note: we negate analytical gradients since JAX calculates gradients of negative log-likelihood
    np.testing.assert_allclose(-analytical_vec, jax_gradients, rtol=1e-4, atol=1e-4)
    print("Isotropic derivatives test passed with real data!")


def radial_tangential_derivatives_test():
    """Test that JAX derivatives match analytical derivatives for radial/tangential model."""
    # Load actual data from CSV file
    data_path = "antikythera/data/1-Fragment_C_Hole_Measurements.csv"
    data = load_data(data_path)

    # Initialize model
    model = SectionedModel(data)

    # Set up parameters for all sections
    params_dict = {
        "r": 77.0,
        "N": 355.0,
        "sigma_r": 0.1,
        "sigma_t": 0.1,
    }

    # Add parameters for each section
    for section in model.section_ids_int:
        params_dict[f"x0_{section}"] = 80.0
        params_dict[f"y0_{section}"] = 136.0
        params_dict[f"alpha_{section}"] = 3.7

    # Convert params_dict to vector for JAX gradient
    params_vec = []
    params_vec.append(params_dict["N"])
    params_vec.append(params_dict["r"])
    params_vec.append(params_dict["sigma_r"])
    params_vec.append(params_dict["sigma_t"])

    for section in model.section_ids_int:
        params_vec.append(params_dict[f"x0_{section}"])
        params_vec.append(params_dict[f"y0_{section}"])
        params_vec.append(params_dict[f"alpha_{section}"])

    params_vec = jnp.array(params_vec)

    # Set up optimization
    _, grad_func = setup_optimization(model, "radial_tangential")

    # Calculate JAX gradients
    jax_gradients = grad_func(params_vec)

    # Calculate analytical gradients
    analytical_gradients = calculate_analytical_derivatives_radial_tangential(model, params_dict)

    # Create a vector of analytical gradients in the same order as JAX gradients
    analytical_vec = []
    analytical_vec.append(analytical_gradients["N"])
    analytical_vec.append(analytical_gradients["r"])
    analytical_vec.append(analytical_gradients["sigma_r"])
    analytical_vec.append(analytical_gradients["sigma_t"])

    for section in model.section_ids_int:
        analytical_vec.append(analytical_gradients[f"x0_{section}"])
        analytical_vec.append(analytical_gradients[f"y0_{section}"])
        analytical_vec.append(analytical_gradients[f"alpha_{section}"])

    analytical_vec = jnp.array(analytical_vec)

    # Test that they match within tolerance
    # Note: we negate analytical gradients since JAX calculates gradients of negative log-likelihood
    np.testing.assert_allclose(-analytical_vec, jax_gradients, rtol=1e-4, atol=1e-4)
    print("Radial tangential derivatives test passed with real data!")


if __name__ == "__main__":
    # Run tests directly
    isotropic_derivatives_test()
    radial_tangential_derivatives_test()
    print("All tests passed with the original Antikythera data!")
