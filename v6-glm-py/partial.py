"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""

import pandas as pd
from typing import Any

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import data


import pandas as pd
import statsmodels.api as sm


# TOOD create enum of allowed families
@data(1)
def compute_partial_betas(
    df: pd.DataFrame,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    is_first_iteration: bool,
    beta_coefficients: list[int],
    dstar: str,
    types: list[str],
    weights: list[int],
) -> Any:
    """Compute beta coefficients for a GLM model"""

    outcome_column = df[outcome_variable]

    # Create the design matrix X
    design_matrix = data[predictor_variables]
    design_matrix = sm.add_constant(design_matrix)  # Add an intercept term to the model
    print(design_matrix)

    # TODO redo
    result = df1
    return result.to_dict()


import pandas as pd
import numpy as np
import statsmodels.api as sm
from vantage6.tools.util import info, warn, error


def RPC_node_beta(
    data_path,
    subset_rules,
    formula,
    family,
    first_iteration,
    coeff,
    dstar=None,
    types=None,
    weights=None,
    extend_data=True,
):
    data = pd.read_csv(data_path)

    # Extract y and X variables name from formula
    y, X = dmatrices(formula, data, return_type="dataframe")

    # Extract the offset from formula (if exists)
    offset = data.eval(formula.split("~")[1].strip()) if "offset" in formula else None

    # Get the family required (Gaussian, Poisson, logistic,...)
    if family == "rs.poi":
        dstar = data[dstar]
    family = get_family(family, dstar)

    weights = np.ones(X.shape[0]) if weights is None else weights
    offset = np.zeros(X.shape[0]) if offset is None else offset

    nobs = X.shape[0]
    nvars = X.shape[1]

    if first_iteration:
        info("First iteration. Initializing variables.")

        if family.family == "rs.poi":
            mustart = np.maximum(y, dstar) + 0.1
        else:
            etastart = None
            family.initialize()
            mustart = family.start_params
        eta = family.link(mustart)
    else:
        eta = np.dot(X, coeff) + offset

    info("Calculating the Betas.")
    mu = family.link.inverse(eta)
    varg = family.variance(mu)
    gprime = family.link.deriv(eta)

    z = (eta - offset) + (y - mu) / gprime
    W = weights * (gprime**2 / varg)
    dispersion = np.sum(W * ((y - mu) / gprime) ** 2)

    output = {
        "v1": np.dot(X.T, W[:, np.newaxis] * X),
        "v2": np.dot(X.T, W * z),
        "dispersion": dispersion,
        "nobs": nobs,
        "nvars": nvars,
        "wt1": np.sum(weights * y),
        "wt2": np.sum(weights),
    }
