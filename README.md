# The Antikythera Mechanism Calendar Ring Analysis

This repository contains statistical models and analyses for determining the original number of holes in the Antikythera mechanism's calendar ring, accounting for fragment displacements.

## Project Structure

```
antikythera/
├── data/                     # Raw data files
│   └── 1-Fragment_C_Hole_Measurements.csv # Hole location measurements
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_data_loading_exploration.ipynb  # Initial data exploration
│   ├── 02_maximum_likelihood.ipynb        # Maximum likelihood estimation
│   ├── 03_mcmc_sampling.ipynb             # MCMC sampling with numpyro
│   └── figures/                           # Generated figures
├── src/                      # Source code
│   ├── bootstrap.py          # Bootstrap confidence intervals
│   ├── comparison.py         # Model comparison utilities
│   ├── data.py               # Data loading and preprocessing
│   ├── mcmc.py               # MCMC modelling and sampling
│   ├── model.py              # Core statistical models
│   └── visualisation.py      # Plotting and visualisation functions
└── tests/                    # Unit tests
    └── model_test.py         # Model differentiation tests
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Setup

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

2. Install the package and dependencies:
   ```
   pip install -e .
   pip install -r requirements.txt
   ```

## Usage

The analysis workflow is organised into Jupyter notebooks:

1. **Data Exploration**:
   ```
   jupyter notebook antikythera/notebooks/01_data_loading_exploration.ipynb
   ```

2. **Maximum Likelihood Estimation**:
   ```
   jupyter notebook antikythera/notebooks/02_maximum_likelihood.ipynb
   ```

3. **MCMC Sampling**:
   ```
   jupyter notebook antikythera/notebooks/03_mcmc_sampling.ipynb
   ```

### Running Tests

To run the model validation tests:

```
pytest
```

## Dependencies

### Core Dependencies
- **NumPy (≥1.26.3)**: For numerical computing
- **JAX (≥0.4.25)** and **JAXlib (≥0.4.25)**: For automatic differentiation and gradient calculation
- **SciPy (≥1.11.4)**: For numerical optimisation
- **Pandas (≥2.1.4)**: For data manipulation
- **Matplotlib (≥3.8.2)**: For visualisation
- **NumPyro (≥0.13.2)**: For probabilistic programming and MCMC sampling
- **ArviZ (≥0.16.1)**: For MCMC diagnostics and visualisation
- **Jupyter (≥1.0.0)** and **Notebook (≥7.0.6)**: For interactive notebook analysis
- **pytest (≥7.4.4)**: For testing
- **tqdm (≥4.66.1)**: For progress bar visualisation

### Development Dependencies
- **pytest-cov (≥4.1.0)**: For test coverage reporting
- **black (≥23.12.1)**: For code formatting
- **isort (≥5.13.2)**: For import sorting
- **flake8 (≥7.0.0)**: For linting
- **pre-commit (≥3.6.0)**: For Git hooks

## Use of Auto-Generation Tools

This project made use of the following AI tools for generating code:

**Claude (Anthropic) / ChatGPT (OpenAI) / Grok (xAI)**: These tools were used to prototype model implementations, debug code, generate documentation, and refactor components. Specific contributions include:

- Drafting configuration files (`pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml`, `requirements.txt`, etc.)
- Designing the initial structure of statistical models
- Drafting the implementation of hypothesis tests
- Assisting in debugging gradient computations
- Generating template code for visualisations
- Refactoring code for improved readability
- Creating this README documentation
