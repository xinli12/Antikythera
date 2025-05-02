import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Set a seed for reproducibility
rng_key = jax.random.PRNGKey(0)


# Jeffreys prior for sigma (log-uniform)
class LogUniform(dist.Distribution):
    support = dist.constraints.positive

    def __init__(self, low=1e-3, high=1e2, validate_args=True):
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        low_shape = jnp.shape(self.low)
        high_shape = jnp.shape(self.high)
        batch_shape = jnp.broadcast_shapes(low_shape, high_shape)
        super(LogUniform, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        low_log = jnp.log(self.low)
        high_log = jnp.log(self.high)
        u = jax.random.uniform(key, shape, minval=low_log, maxval=high_log)
        return jnp.exp(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_normalization = jnp.log(jnp.log(self.high / self.low))
        return -jnp.log(value) - log_normalization


def numpyro_model_isotropic(model, data, prior_config=None):
    """
    NumPyro model for SectionedModel with isotropic uncertainty.

    Parameters
    ----------
    model : SectionedModel
        The SectionedModel instance
    data : dict
        Dictionary containing the hole data
    prior_config : dict, optional
        Dictionary of prior parameters
    """
    # Set default prior parameters if not provided
    if prior_config is None:
        prior_config = {
            "N_min": 300,
            "N_max": 400,
            "r_min": 50,
            "r_max": 100,
            "sigma_min": 1e-5,
            "sigma_max": 1e2,
            "x0_min": 80.0,
            "x0_max": 120.0,
            "y0_min": 100.0,
            "y0_max": 150.0,
            "alpha_min": jnp.pi,
            "alpha_max": 2 * jnp.pi,
        }

    # Sample global parameters
    N_min = prior_config["N_min"]
    N_max = prior_config["N_max"]
    r_min = prior_config["r_min"]
    r_max = prior_config["r_max"]
    N = numpyro.sample("N", dist.Uniform(N_min, N_max), rng_key=rng_key)
    r = numpyro.sample("r", dist.Uniform(r_min, r_max), rng_key=rng_key)

    # Sample error parameter with a Jeffrey's prior
    sigma = numpyro.sample(
        "sigma",
        LogUniform(prior_config["sigma_min"], prior_config["sigma_max"]),
        rng_key=rng_key,
    )

    # Define excluded sections
    excluded_sections = [0, 4]

    # Initialize params dictionary
    params = {"N": N, "r": r, "sigma": sigma}

    # Define reference values for excluded sections
    x0_ref = (prior_config["x0_min"] + prior_config["x0_max"]) / 2
    y0_ref = (prior_config["y0_min"] + prior_config["y0_max"]) / 2
    alpha_ref = (prior_config["alpha_min"] + prior_config["alpha_max"]) / 2

    # Set reference values for excluded sections
    for section_id in excluded_sections:
        if section_id in model.section_ids_int:
            params[f"x0_{section_id}"] = x0_ref
            params[f"y0_{section_id}"] = y0_ref
            params[f"alpha_{section_id}"] = alpha_ref

    # Sample parameters only for non-excluded sections
    for section_id in model.section_ids_int:
        if section_id not in excluded_sections:
            # Sample center coordinates
            params[f"x0_{section_id}"] = numpyro.sample(
                f"x0_{section_id}",
                dist.Uniform(prior_config["x0_min"], prior_config["x0_max"]),
                rng_key=rng_key,
            )
            params[f"y0_{section_id}"] = numpyro.sample(
                f"y0_{section_id}",
                dist.Uniform(prior_config["y0_min"], prior_config["y0_max"]),
                rng_key=rng_key,
            )

            # Sample rotation angle
            params[f"alpha_{section_id}"] = numpyro.sample(
                f"alpha_{section_id}",
                dist.Uniform(prior_config["alpha_min"], prior_config["alpha_max"]),
                rng_key=rng_key,
            )

    # Compute log likelihood for each hole
    for hole_id in model.hole_ids:
        idx = model.hole_to_idx[hole_id]
        obs_x, obs_y = data["x"][idx], data["y"][idx]

        # Get model prediction
        pred = model.get_model_prediction(params, hole_id)
        pred_x, pred_y = pred[0], pred[1]

        # Likelihood
        numpyro.sample(
            f"obs_{hole_id}",
            dist.Normal(jnp.array([pred_x, pred_y]), sigma),
            obs=jnp.array([obs_x, obs_y]),
            rng_key=rng_key,
        )


def numpyro_model_radial_tangential(model, data, prior_config=None):
    """
    NumPyro model for SectionedModel with radial/tangential uncertainty.

    Parameters
    ----------
    model : SectionedModel
        The SectionedModel instance
    data : dict
        Dictionary containing the hole data
    prior_config : dict, optional
        Dictionary of prior parameters
    """
    # Set default prior parameters if not provided
    if prior_config is None:
        prior_config = {
            "N_min": 300,
            "N_max": 400,
            "r_min": 50,
            "r_max": 100,
            "sigma_r_min": 1e-5,
            "sigma_r_max": 1e2,
            "sigma_t_min": 1e-5,
            "sigma_t_max": 1e2,
            "x0_min": 150.0,
            "x0_max": 250.0,
            "y0_min": 150.0,
            "y0_max": 250.0,
            "alpha_min": jnp.pi,
            "alpha_max": 2 * jnp.pi,
        }

    # Sample global parameters
    N_min = prior_config["N_min"]
    N_max = prior_config["N_max"]
    r_min = prior_config["r_min"]
    r_max = prior_config["r_max"]
    N = numpyro.sample("N", dist.Uniform(N_min, N_max), rng_key=rng_key)
    r = numpyro.sample("r", dist.Uniform(r_min, r_max), rng_key=rng_key)

    # Sample error parameters with Jeffrey's priors
    sigma_r = numpyro.sample(
        "sigma_r",
        LogUniform(prior_config["sigma_r_min"], prior_config["sigma_r_max"]),
        rng_key=rng_key,
    )
    sigma_t = numpyro.sample(
        "sigma_t",
        LogUniform(prior_config["sigma_t_min"], prior_config["sigma_t_max"]),
        rng_key=rng_key,
    )

    # Define excluded sections
    excluded_sections = [0, 4]

    # Initialize params dictionary
    params = {"N": N, "r": r, "sigma_r": sigma_r, "sigma_t": sigma_t}

    # Define reference values for excluded sections
    x0_ref = (prior_config["x0_min"] + prior_config["x0_max"]) / 2
    y0_ref = (prior_config["y0_min"] + prior_config["y0_max"]) / 2
    alpha_ref = (prior_config["alpha_min"] + prior_config["alpha_max"]) / 2

    # Set reference values for excluded sections
    for section_id in excluded_sections:
        if section_id in model.section_ids_int:
            params[f"x0_{section_id}"] = x0_ref
            params[f"y0_{section_id}"] = y0_ref
            params[f"alpha_{section_id}"] = alpha_ref

    # Sample parameters only for non-excluded sections
    for section_id in model.section_ids_int:
        if section_id not in excluded_sections:
            # Sample center coordinates
            params[f"x0_{section_id}"] = numpyro.sample(
                f"x0_{section_id}",
                dist.Uniform(prior_config["x0_min"], prior_config["x0_max"]),
                rng_key=rng_key,
            )
            params[f"y0_{section_id}"] = numpyro.sample(
                f"y0_{section_id}",
                dist.Uniform(prior_config["y0_min"], prior_config["y0_max"]),
                rng_key=rng_key,
            )

            # Sample rotation angle
            params[f"alpha_{section_id}"] = numpyro.sample(
                f"alpha_{section_id}",
                dist.Uniform(prior_config["alpha_min"], prior_config["alpha_max"]),
                rng_key=rng_key,
            )

    # Compute log likelihood for each hole
    for hole_id in model.hole_ids:
        idx = model.hole_to_idx[hole_id]
        obs_x, obs_y = data["x"][idx], data["y"][idx]

        # Get model prediction
        pred = model.get_model_prediction(params, hole_id)
        pred_x, pred_y = pred[0], pred[1]

        # Get radial and tangential unit vectors
        r_hat, t_hat = model.get_radial_tangential_vectors(params, hole_id)

        # Create the covariance matrix in the Cartesian basis
        # Start with covariance in r-t basis
        cov_rt = jnp.array([[sigma_r**2, 0], [0, sigma_t**2]])

        # Rotation matrix from r-t to x-y
        rot = jnp.vstack([r_hat, t_hat]).T

        # Covariance in x-y basis: R * cov_rt * R^T
        cov_xy = rot @ cov_rt @ rot.T

        # Likelihood using multivariate normal
        numpyro.sample(
            f"obs_{hole_id}",
            dist.MultivariateNormal(jnp.array([pred_x, pred_y]), cov_xy),
            obs=jnp.array([obs_x, obs_y]),
        )
