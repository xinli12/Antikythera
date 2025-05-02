import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .model import (
    setup_optimization,
)


def _resample_data(original_data, resample_within_sections=False):
    """
    Resamples the hole data with replacement.

    Parameters
    ----------
    original_data : dict
        The original data dictionary.
    resample_within_sections : bool
        If True, resample hole indices within each section.
        Otherwise, resample from all holes globally.

    Returns
    -------
    dict
        A new data dictionary with resampled data.
    """
    n_holes = len(original_data["x"])
    resampled_indices = []

    if resample_within_sections:
        # Ensure section IDs are integers for dictionary lookup later if needed
        sections = np.unique(original_data["section"].astype(int))
        original_indices = np.arange(n_holes)
        for section_id in sections:
            section_mask = original_data["section"] == section_id
            indices_in_section = original_indices[section_mask]
            if len(indices_in_section) > 0:
                # Resample indices *within* this section
                sampled_section_indices = np.random.choice(
                    indices_in_section, size=len(indices_in_section), replace=True
                )
                resampled_indices.extend(sampled_section_indices)
    else:
        # Resample globally across all holes
        resampled_indices = np.random.choice(n_holes, size=n_holes, replace=True)

    resampled_indices = np.array(resampled_indices)
    # Ensure we use numpy arrays for indexing before converting back potentially
    bootstrap_data = {k: np.asarray(v)[resampled_indices] for k, v in original_data.items()}
    return bootstrap_data


def bootstrap_parameter_uncertainty(
    original_data,
    model_class,
    model_type,
    mle_result,
    n_bootstrap_samples=100,
    resample_within_sections=False,
    minimize_options=None,
    minimize_method="L-BFGS-B",  # Default method
    bounds=None,
):
    """
    Estimates parameter uncertainty using bootstrapping.

    Parameters
    ----------
    original_data : dict
        Dictionary containing the original dataset ('hole_id', 'section', 'x', 'y').
        Values should be NumPy arrays or convertible to them.
    model_class : class
        The model class to use (e.g., IdealModel, SectionedModel).
    model_type : str
        The type of covariance model ('isotropic' or 'radial_tangential').
    mle_result : OptimizeResult
        The result object from the initial scipy.optimize.minimize call.
    n_bootstrap_samples : int, optional
        Number of bootstrap iterations. Defaults to 100.
    resample_within_sections : bool, optional
        Whether to resample data within sections (True) or globally (False).
        Should be True for SectionedModel. Defaults to False.
    minimize_options : dict, optional
        Options dictionary passed to scipy.optimize.minimize.
    minimize_method : str, optional
        Optimization method for scipy.optimize.minimize. Defaults to 'L-BFGS-B'.
    bounds : sequence, optional
        Bounds for variables, passed to scipy.optimize.minimize.

    Returns
    -------
    np.ndarray
        An array where each row contains the parameter vector from a
        successful bootstrap optimization. Shape: (n_successful_samples, n_params).
    """
    mle_params_vec = mle_result.x
    bootstrap_parameter_vectors = []

    # Convert original data JAX arrays (if any) to NumPy for resampling
    np_original_data = {k: np.asarray(v) for k, v in original_data.items()}

    if minimize_options is None:
        minimize_options = {"disp": False}  # Default to not displaying progress per iteration

    # Create a progress bar with tqdm
    successful_samples = 0
    for i in tqdm(range(n_bootstrap_samples), desc="Bootstrap Progress"):
        try:
            # 1. Resample data (using NumPy arrays)
            bootstrap_data_np = _resample_data(np_original_data, resample_within_sections=resample_within_sections)

            # 2. Create model instance with resampled data (convert back to JAX arrays)
            # The model's __init__ already handles conversion to JAX arrays.
            bootstrap_model = model_class(bootstrap_data_np)

            # 3. Setup optimization
            neg_log_likelihood, grad_neg_log_likelihood = setup_optimization(bootstrap_model, model_type)

            # 4. Run optimization
            # Use the original MLE parameters as the starting point
            result_bootstrap = minimize(
                fun=neg_log_likelihood,
                x0=mle_params_vec,  # Start from original MLE
                method=minimize_method,
                jac=grad_neg_log_likelihood,
                options=minimize_options,
                bounds=bounds,  # Pass bounds if provided
            )

            # 5. Store result if successful
            if result_bootstrap.success:
                bootstrap_parameter_vectors.append(result_bootstrap.x)
                successful_samples += 1

        except Exception as e:
            print(f"Error in bootstrap iteration {i}: {e}")
            # Just continue to the next iteration - tqdm will show progress regardless
            continue

    print(f"Bootstrap finished. {successful_samples} successful samples out of {n_bootstrap_samples}.")

    if not bootstrap_parameter_vectors:
        print("Warning: No bootstrap samples converged successfully.")
        # Return an empty array with the correct number of columns if possible
        return np.empty((0, len(mle_params_vec)))

    return np.vstack(bootstrap_parameter_vectors)


def save_bootstrap_results(
    bootstrap_results,
    param_names,
    output_file,
    model_class_name=None,
    model_type=None,
    confidence_levels=0.68,
    original_mle=None,
    save_all_samples=False,
):
    """
    Calculate statistics from bootstrap results and save them to a file.

    Parameters
    ----------
    bootstrap_results : np.ndarray
        The array of bootstrap parameter vectors, with shape (n_samples, n_params)
    param_names : list
        List of parameter names corresponding to each column in bootstrap_results
    output_file : str
        File path to save the results to (CSV or JSON)
    model_class_name : str, optional
        Name of the model class used for bootstrapping
    model_type : str, optional
        Type of model used ('isotropic' or 'radial_tangential')
    confidence_levels : float or list, optional
        Confidence level(s) for interval calculation (default: 0.68)
        Can be a single value (e.g., 0.68 for 68% CI) or a list of values
        (e.g., [0.68, 0.95, 0.997] for 1σ, 2σ, and 3σ)
    original_mle : np.ndarray, optional
        The original maximum likelihood parameter vector for comparison
    save_all_samples : bool, optional
        Whether to save all individual bootstrap samples (default: False)
        If True, all samples will be saved regardless of the file size

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parameter statistics
    """
    if bootstrap_results.shape[0] == 0:
        print("No bootstrap results to save")
        return None

    if len(param_names) != bootstrap_results.shape[1]:
        raise ValueError(
            f"Number of parameter names ({len(param_names)}) does not match "
            f"number of parameters in bootstrap results ({bootstrap_results.shape[1]})"
        )

    # Calculate statistics
    means = np.mean(bootstrap_results, axis=0)
    stds = np.std(bootstrap_results, axis=0)
    median = np.median(bootstrap_results, axis=0)

    # Process confidence levels to ensure it's a list
    if not isinstance(confidence_levels, (list, tuple, np.ndarray)):
        confidence_levels = [confidence_levels]

    # Create DataFrame with basic results
    results_dict = {
        "Parameter": param_names,
        "Mean": means,
        "Median": median,
        "Std": stds,
    }

    # Add confidence intervals for each level
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_bounds = np.percentile(bootstrap_results, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_results, upper_percentile, axis=0)

        # Add to results dictionary
        results_dict[f"{conf_level*100:.1f}%_lower"] = lower_bounds
        results_dict[f"{conf_level*100:.1f}%_upper"] = upper_bounds

    # Add original MLE values if provided
    if original_mle is not None:
        results_dict["MLE"] = original_mle

    results_df = pd.DataFrame(results_dict)

    # Save results to file
    file_ext = os.path.splitext(output_file)[1].lower()

    if file_ext == ".csv":
        # For CSV, save main results and optionally samples in separate file
        results_df.to_csv(output_file, index=False)

        if save_all_samples:
            samples_file = f"{os.path.splitext(output_file)[0]}_samples.csv"
            samples_df = pd.DataFrame(bootstrap_results, columns=param_names)
            samples_df.to_csv(samples_file, index=False)
            print(f"All bootstrap samples saved to {samples_file}")

    elif file_ext == ".json":
        # Create a more comprehensive JSON structure
        json_data = {
            "model_info": {
                "model_class": model_class_name,
                "model_type": model_type,
                "confidence_levels": confidence_levels,
                "n_bootstrap_samples": bootstrap_results.shape[0],
            },
            "parameter_statistics": results_df.to_dict(orient="records"),
        }

        # Add raw bootstrap samples if requested or if size is reasonable
        if save_all_samples or bootstrap_results.shape[0] <= 1000:
            bootstrap_df = pd.DataFrame(bootstrap_results, columns=param_names)
            json_data["bootstrap_samples"] = bootstrap_df.to_dict(orient="records")

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)
    else:
        # Default to CSV if extension not recognized
        print(f"Unrecognized file extension: {file_ext}. Saving as CSV instead.")
        base_name = os.path.splitext(output_file)[0]
        results_df.to_csv(f"{base_name}.csv", index=False)

        if save_all_samples:
            samples_df = pd.DataFrame(bootstrap_results, columns=param_names)
            samples_df.to_csv(f"{base_name}_samples.csv", index=False)
            print(f"All bootstrap samples saved to {base_name}_samples.csv")

    print(f"Bootstrap results saved to {output_file}")
    return results_df
