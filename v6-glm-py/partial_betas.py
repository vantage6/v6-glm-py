"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""

from typing import Any

import pandas as pd
import numpy as np
from pprint import pprint

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

from .common import GLMDataManager


# TODO weights are never set to custom value in the main function. Remove argument?
@data(1)
def compute_local_betas(
    df: pd.DataFrame,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    is_first_iteration: bool,
    beta_coefficients: dict[str, float] | None,
    category_reference_values: dict[str, str] | None = None,
    dstar: str = None,
    weights: list[int] = None,
) -> Any:
    """
    Compute beta coefficients for a GLM model

    TODO add docstring
    """
    info("Started function to compute beta coefficients")

    # convert input dicts to pandas
    if beta_coefficients is not None:
        beta_coefficients = pd.Series(beta_coefficients)

    data_mgr = GLMDataManager(
        df,
        outcome_variable,
        predictor_variables,
        family,
        category_reference_values,
        dstar,
        weights,
    )

    eta = data_mgr.compute_eta(is_first_iteration, beta_coefficients)

    # ensure that eta has the same column name as y to allow for subtraction/addition/..
    eta.columns = data_mgr.y.columns
    print(eta)

    info("Computing beta coefficients")
    mu = data_mgr.family.link.inverse(eta)
    # print("y", data_mgr.y)
    # print("mu", mu)
    varg = data_mgr.family.variance(mu)
    # print("varg", varg)

    # TODO below is what Copilot suggests but what is active is what is in the R version
    gprime = data_mgr.family.link.inverse(eta)  # for poisson, this is exp(eta)
    # gprime = family.link.deriv(eta)  # for poisson, this is 1 / eta
    # print("gprime", gprime)

    # compute Z matrix, weights and dispersion matrix
    y_minus_mu = data_mgr.y.sub(mu, axis=0)
    # print("y_minus_mu", y_minus_mu)

    z = eta + (y_minus_mu / gprime)
    # print("z", z)

    W = (gprime**2 / varg).mul(data_mgr.weights, axis=0)
    # print("W", W)
    # print(type(W), W.columns)
    dispersion_matrix = W * (y_minus_mu / gprime) ** 2
    dispersion = dispersion_matrix.sum()
    # dispersion = sum(W * ((y - mu) / gprime) ** 2) / sum(W)
    # print("dispersion", dispersion)

    # print("W*X", X.mul(W.iloc[:, 0], axis=0))
    # TODO there are some non-clear things in the code like `mul()` and `iloc[:, 0]`.
    # They are there to ensure proper multiplication etc of pandas Dataframes with
    # series. Make this code more clear and readable.
    return {
        "XTX": data_mgr.X.T.dot(data_mgr.X.mul(W.iloc[:, 0], axis=0)).to_dict(),
        "XTz": data_mgr.X.T.dot(W * z).to_dict(),
        "dispersion": float(dispersion),
        "num_observations": len(df),
        "num_variables": len(data_mgr.X.columns),
        "weighted_sum_of_y": float(
            np.sum(data_mgr.weights.mul(data_mgr.y.iloc[:, 0], axis=0))
        ),
        "weights_sum": float(np.sum(data_mgr.weights)),
    }
