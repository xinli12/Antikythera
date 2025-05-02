import jax.numpy as jnp
from jax import grad, jit


class BaseAntikytheraModel:
    """
    Base model for the location of holes on the Antikythera mechanism ring.
    """

    def __init__(self, data):
        # Validate required keys
        required_keys = {"hole_id", "section", "x", "y"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Missing required data keys: {missing}")

        # Convert all inputs to JAX arrays to ensure consistency
        self.data = {k: jnp.asarray(v) for k, v in data.items()}
        self.n_holes = len(self.data["x"])

        # Extract unique sections
        self.section_ids = jnp.unique(self.data["section"])
        # f-strings, dict keys, list indexing
        self.section_ids_int = [int(s) for s in self.section_ids]
        self.n_sections = len(self.section_ids)

        # Precompute hole IDs as a Python list
        self.hole_ids = [int(h) for h in self.data["hole_id"]]

        # Create mappings between holes and sections
        self._create_mappings()

    def _create_mappings(self):
        """Create mappings between holes and sections for efficient lookup."""
        # Create a mapping from hole ID to data index
        self.hole_to_idx = {}
        for i, hole_id in enumerate(self.data["hole_id"]):
            self.hole_to_idx[int(hole_id)] = i

        # Create a mapping from hole ID to section
        self.hole_to_section = {}
        for hole_id, section in zip(
            self.data["hole_id"],
            self.data["section"],
        ):
            self.hole_to_section[int(hole_id)] = int(section)

        # Group holes by section
        self.section_to_holes = {}
        for section in self.section_ids_int:
            mask = self.data["section"] == section
            self.section_to_holes[section] = [int(h) for h in self.data["hole_id"][mask]]

    def get_model_prediction(self, params, hole_id):
        """
        Get the model prediction for the location of a hole.
        This method must be implemented by derived classes.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to predict

        Returns
        -------
        m_i: jnp.ndarray
            Model prediction for the hole location [x, y]
        """
        raise NotImplementedError("Subclasses must implement get_model_prediction method")

    def get_error_vector(self, params, hole_id):
        """
        Get the error between the model prediction and the data.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to calculate the error for

        Returns
        -------
        e_i : jnp.ndarray
            Error vector
        """
        # Convert hole_id to int before using as dictionary key
        idx = self.hole_to_idx[int(hole_id)]
        d_i = jnp.array([self.data["x"][idx], self.data["y"][idx]])
        m_i = self.get_model_prediction(params, hole_id)

        # Calculate the error vector
        return m_i - d_i

    def get_radial_tangential_vectors(self, params, hole_id):
        """
        Get the radial and tangential unit vectors at the model prediction.
        This is a default implementation that might be overridden.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to calculate the vectors for

        Returns
        -------
        r_hat : jnp.ndarray
            Radial unit vector
        t_hat : jnp.ndarray
            Tangential unit vector
        """
        raise NotImplementedError("Subclasses must implement get_radial_tangential_vectors method")

    def _single_hole_log_likelihood_isotropic(self, params, hole_id):
        """
        Calculate log-likelihood contribution from a single hole
        for isotropic model.
        """
        sigma = params["sigma"]
        e_i = self.get_error_vector(params, hole_id)
        return -jnp.sum(e_i**2) / (2 * sigma**2)

    def log_likelihood_isotropic(self, params):
        """
        Calculate the log-likelihood for
        the isotropic covariance matrix model.
        """
        sigma = params["sigma"]

        # Use a plain Python loop instead of vectorized computation
        total_log_likelihood = 0.0
        for hole_id in self.hole_ids:
            total_log_likelihood += self._single_hole_log_likelihood_isotropic(params, hole_id)

        log_norm = -self.n_holes * jnp.log(2 * jnp.pi * sigma**2)

        return log_norm + total_log_likelihood

    def _single_hole_log_likelihood_radial_tangential(self, params, hole_id):
        """
        Calculate log-likelihood contribution from a single hole for R/T model.
        """
        sigma_r = params["sigma_r"]
        sigma_t = params["sigma_t"]

        e_i = self.get_error_vector(params, hole_id)
        r_hat, t_hat = self.get_radial_tangential_vectors(params, hole_id)

        e_r = jnp.dot(e_i, r_hat)
        e_t = jnp.dot(e_i, t_hat)

        return -(e_r**2 / (2 * sigma_r**2)) - (e_t**2 / (2 * sigma_t**2))

    def log_likelihood_radial_tangential(self, params):
        """
        Calculate the log-likelihood for the radial/tangential
        covariance matrix model.
        """
        sigma_r = params["sigma_r"]
        sigma_t = params["sigma_t"]

        # Use a plain Python loop instead of vectorized computation
        total_log_likelihood = 0.0
        for hole_id in self.hole_ids:
            method = self._single_hole_log_likelihood_radial_tangential
            log_likelihood = method(params, hole_id)
            total_log_likelihood += log_likelihood

        return -self.n_holes * jnp.log(2 * jnp.pi * sigma_r * sigma_t) + total_log_likelihood

    def vec_to_params_isotropic(self, vec):
        """
        Convert a parameter vector to a dictionary of
        parameters for isotropic model.
        Must be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses must implement vec_to_params_isotropic method")

    def vec_to_params_radial_tangential(self, vec):
        """
        Convert a parameter vector to a dictionary of parameters for R/T model.
        Must be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses must implement vec_to_params_radial_tangential method")


class SectionedModel(BaseAntikytheraModel):
    """
    Model that handles multiple sections with different
    centers and phase shifts.
    """

    def get_model_prediction(self, params, hole_id):
        """
        Get the model prediction for the location of a hole in
        a sectioned model.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to predict

        Returns
        -------
        m_i: jnp.ndarray
            Model prediction for the hole location [x, y]
        """
        # Convert hole_id to int before using as dictionary key
        section = self.hole_to_section[int(hole_id)]

        # Extract parameters
        r = params["r"]
        N = params["N"]
        x0 = params[f"x0_{section}"]
        y0 = params[f"y0_{section}"]
        alpha = params[f"alpha_{section}"]

        # Calculate the angular position
        phi = 2 * jnp.pi * (hole_id - 1) / N + alpha

        # Cartesian coordinates
        x = x0 + r * jnp.cos(phi)
        y = y0 + r * jnp.sin(phi)

        return jnp.array([x, y])

    def get_radial_tangential_vectors(self, params, hole_id):
        """
        Get the radial and tangential unit vectors at the model prediction.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to calculate the vectors for
        """
        section = self.hole_to_section[int(hole_id)]

        # Extract parameters
        N = params["N"]
        alpha = params[f"alpha_{section}"]

        # Calculate the angular position
        phi = 2 * jnp.pi * (hole_id - 1) / N + alpha

        # Calculate the radial and tangential unit vectors
        r_hat = jnp.array([jnp.cos(phi), jnp.sin(phi)])
        t_hat = jnp.array([jnp.sin(phi), -jnp.cos(phi)])

        return r_hat, t_hat

    def vec_to_params_isotropic(self, vec):
        """
        Convert a parameter vector to a dictionary of parameters
        for isotropic model.

        Parameters
        ----------
        vec : jnp.ndarray
            Parameter vector

        Returns
        -------
        params : dict
            Dictionary of parameters
        """
        params = {
            "N": vec[0],
            "r": vec[1],
            "sigma": vec[2],
        }

        # Ensure we use the actual section IDs from the data
        for j, section_id in enumerate(self.section_ids_int):
            params[f"x0_{section_id}"] = vec[3 + 3 * j]
            params[f"y0_{section_id}"] = vec[4 + 3 * j]
            params[f"alpha_{section_id}"] = vec[5 + 3 * j]

        return params

    def vec_to_params_radial_tangential(self, vec):
        """
        Convert a parameter vector to a dictionary of parameters for R/T model.

        Parameters
        ----------
        vec : jnp.ndarray
            Parameter vector

        Returns
        -------
        params : dict
            Dictionary of parameters
        """
        params = {
            "N": vec[0],
            "r": vec[1],
            "sigma_r": vec[2],
            "sigma_t": vec[3],
        }

        # Ensure we use the actual section IDs from the data
        for j, section_id in enumerate(self.section_ids_int):
            params[f"x0_{section_id}"] = vec[4 + 3 * j]
            params[f"y0_{section_id}"] = vec[5 + 3 * j]
            params[f"alpha_{section_id}"] = vec[6 + 3 * j]

        return params


# Specific model implementation for ideal model with single center
class IdealModel(BaseAntikytheraModel):
    """Simplified model with a single center and radius."""

    def get_model_prediction(self, params, hole_id):
        """
        Get the model prediction for the location of a hole in the ideal model.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to predict

        Returns
        -------
        m_i: jnp.ndarray
            Model prediction for the hole location [x, y]
        """
        r = params["r"]
        N = params["N"]
        theta_0 = params["theta_0"]
        x0 = params["x0"]
        y0 = params["y0"]

        # Calculate the angular position
        phi = 2 * jnp.pi * (hole_id - 1) / N + theta_0

        # Cartesian coordinates
        x = x0 + r * jnp.cos(phi)
        y = y0 + r * jnp.sin(phi)

        return jnp.array([x, y])

    def get_radial_tangential_vectors(self, params, hole_id):
        """
        Get the radial and tangential unit vectors at the model prediction.
        Override to use the ideal model's parameters.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters

        hole_id : int
            ID of the hole to calculate the vectors for

        Returns
        -------
        r_hat : jnp.ndarray
            Radial unit vector
        t_hat : jnp.ndarray
            Tangential unit vector
        """
        # Extract parameters
        N = params["N"]
        theta_0 = params["theta_0"]

        # Calculate the angular position
        phi = 2 * jnp.pi * (hole_id - 1) / N + theta_0

        # Calculate the radial and tangential unit vectors
        r_hat = jnp.array([jnp.cos(phi), jnp.sin(phi)])
        t_hat = jnp.array([jnp.sin(phi), -jnp.cos(phi)])

        return r_hat, t_hat

    def vec_to_params_isotropic(self, vec):
        """
        Convert a parameter vector to a dictionary of
        parameters for isotropic model.

        Parameters
        ----------
        vec : jnp.ndarray
            Parameter vector

        Returns
        -------
        params : dict
            Dictionary of parameters
        """
        params = {
            "N": vec[0],
            "r": vec[1],
            "sigma": vec[2],
            "x0": vec[3],
            "y0": vec[4],
            "theta_0": vec[5],
        }
        return params

    def vec_to_params_radial_tangential(self, vec):
        """
        Convert a parameter vector to a dictionary of parameters for R/T model.

        Parameters
        ----------
        vec : jnp.ndarray
            Parameter vector

        Returns
        -------
        params : dict
            Dictionary of parameters
        """
        params = {
            "N": vec[0],
            "r": vec[1],
            "sigma_r": vec[2],
            "sigma_t": vec[3],
            "x0": vec[4],
            "y0": vec[5],
            "theta_0": vec[6],
        }
        return params


def setup_optimization(model, model_type="isotropic"):
    """
    Set up JAX-based optimization for different model types.

    Parameters
    ----------
    model : BaseAntikytheraModel
        Model object
    model_type : str
        One of "isotropic", "radial_tangential", or "ideal"

    Returns
    -------
    objective : function
        JAX-based objective function
    grad_objective : function
        JAX-based gradient function
    """
    if model_type == "isotropic":

        @jit
        def neg_log_likelihood(params_flat):
            params_dict = model.vec_to_params_isotropic(params_flat)
            return -model.log_likelihood_isotropic(params_dict)

    elif model_type == "radial_tangential":

        @jit
        def neg_log_likelihood(params_flat):
            params_dict = model.vec_to_params_radial_tangential(params_flat)
            return -model.log_likelihood_radial_tangential(params_dict)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Define the gradient function
    grad_neg_log_likelihood = jit(grad(neg_log_likelihood))

    return neg_log_likelihood, grad_neg_log_likelihood
