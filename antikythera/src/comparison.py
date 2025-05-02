import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def compute_savage_dickey_ratio(mcmc_samples, prior_config=None):
    """
    Compute the Savage-Dickey density ratio to compare the isotropic model (H0)
    to the radial/tangential model (H1).

    Parameters
    ----------
    mcmc_samples : dict
        MCMC samples from the radial/tangential model
    prior_config : dict, optional
        Configuration for priors, needed to compute prior density

    Returns
    -------
    bayes_factor : float
        Bayes factor in favor of the isotropic model (H0)
    """
    # Extract sigma_r and sigma_t from MCMC samples
    # mcmc_samples is a dict with keys 'sigma_r' and 'sigma_t'
    sigma_r = mcmc_samples["sigma_r"]
    sigma_t = mcmc_samples["sigma_t"]

    # Compute log difference
    log_diff = np.log(sigma_r) - np.log(sigma_t)

    # Compute posterior density at log_diff = 0 using KDE
    kde = stats.gaussian_kde(log_diff)
    posterior_density_at_zero = kde(0)[0]
    print(f"Posterior density at sigma_r = sigma_t: {posterior_density_at_zero}")

    # Compute prior density at log_diff = 0
    if prior_config is None:
        # Using default values from mcmc.py
        prior_config = {
            "sigma_r_min": 1e-5,
            "sigma_r_max": 1e2,
            "sigma_t_min": 1e-5,
            "sigma_t_max": 1e2,
        }
    prior_density_at_zero = analytical_prior_density_at_equal_sigmas(prior_config)
    print(f"Prior density at sigma_r = sigma_t: {prior_density_at_zero}")

    # Compute Bayes factor
    bayes_factor = posterior_density_at_zero / prior_density_at_zero

    return bayes_factor


def plot_savage_dickey_ratio(mcmc_samples):
    """
    Plot the posterior and prior densities of log(sigma_r) - log(sigma_t)
    to visualize the Savage-Dickey density ratio.
    """
    # Extract sigma_r and sigma_t
    if isinstance(mcmc_samples, az.InferenceData):
        sigma_r = mcmc_samples.posterior.sigma_r.values.flatten()
        sigma_t = mcmc_samples.posterior.sigma_t.values.flatten()
    else:
        sigma_r = mcmc_samples["sigma_r"]
        sigma_t = mcmc_samples["sigma_t"]

    # Compute log difference
    log_diff = np.log(sigma_r) - np.log(sigma_t)

    # Plot histogram of log differences
    plt.figure(figsize=(10, 6))

    # Plot histogram and KDE of posterior
    plt.hist(log_diff, bins=50, density=True, alpha=0.5, label="Posterior samples")

    # Plot posterior KDE
    kde = stats.gaussian_kde(log_diff)
    x_range = np.linspace(min(log_diff), max(log_diff), 1000)
    plt.plot(x_range, kde(x_range), "b-", label="Posterior density")

    # Highlight the density at log_diff = 0
    posterior_density_at_zero = kde(0)[0]
    plt.plot([0, 0], [0, posterior_density_at_zero], "r--", linewidth=2)
    plt.scatter(
        [0],
        [posterior_density_at_zero],
        color="red",
        s=100,
        label=f"Posterior density at 0: {posterior_density_at_zero:.4f}",
    )

    plt.axvline(x=0, color="k", linestyle="--", alpha=0.5, label=r"$\log \sigma_r = \log \sigma_t$")

    plt.xlabel(r"$\log \sigma_r - \log \sigma_t$")
    plt.ylabel("Density")
    # plt.title("Savage-Dickey Density Ratio for Model Comparison")
    plt.legend()

    return plt.gcf()


def analytical_prior_density_at_equal_sigmas(prior_config):
    """
    Analytically calculate the prior density at points where sigma_r = sigma_t.

    Parameters
    ----------
    prior_config : dict
        Dictionary containing prior configuration

    Returns
    -------
    float
        The prior density at the hypothesis sigma_r = sigma_t
    """
    a_r = prior_config["sigma_r_min"]
    b_r = prior_config["sigma_r_max"]
    a_t = prior_config["sigma_t_min"]
    b_t = prior_config["sigma_t_max"]

    log_a_r = np.log(a_r)
    log_b_r = np.log(b_r)
    log_a_t = np.log(a_t)
    log_b_t = np.log(b_t)

    overlap = max(0, min(log_b_r, log_b_t) - max(log_a_r, log_a_t))
    log_term_r = log_b_r - log_a_r
    log_term_t = log_b_t - log_a_t

    prior_density = overlap / (log_term_r * log_term_t)
    return prior_density
