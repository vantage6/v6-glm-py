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
import statsmodels.genmod.families as families

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

from .common import GLMDataManager


@data(1)
def compute_local_betas(
    df: pd.DataFrame,
    formula: str,
    family: str,
    is_first_iteration: bool,
    beta_coefficients: dict[str, float] | None,
    dstar: str = None,
) -> dict:
    """
    Compute beta coefficients for a GLM model

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    formula : str
        The formula specifying the model.
    family : str
        The family of the GLM (e.g., 'gaussian', 'binomial').
    is_first_iteration : bool
        Whether this is the first iteration of the model.
    beta_coefficients : dict[str, float] | None
        The beta coefficients. These must be provided if is_first_iteration is False.
    dstar : str, optional
        An optional parameter for additional model specifications.

    Returns
    -------
    dict
        The computed beta coefficients.
    """
    info("Started function to compute beta coefficients")

    # convert input dicts to pandas
    if beta_coefficients is not None:
        beta_coefficients = pd.Series(beta_coefficients)

    data_mgr = GLMDataManager(
        df,
        formula,
        family,
        dstar,
    )

    eta = data_mgr.compute_eta(is_first_iteration, beta_coefficients)

    # ensure that eta has the same column name as y to allow for subtraction/addition/..
    eta.columns = data_mgr.y.columns
    # print("eta", eta)

    info("Computing beta coefficients")
    mu = data_mgr.family.link.inverse(eta)
    # print("y", data_mgr.y)
    # print("mu", mu)
    varg = data_mgr.family.variance(mu)
    # print("varg", varg)

    # TODO below is what Copilot suggests but what is active is what is in the R version
    if isinstance(data_mgr.family, families.Poisson):
        gprime = data_mgr.family.link.inverse(eta)  # for poisson, this is exp(eta)
    else:
        # TODO below is new when testing with Gaussian!!!!
        gprime = data_mgr.family.link.deriv(eta)  # for poisson, this is 1 / eta
        if isinstance(gprime, np.ndarray):
            # for some families, gprime is a numpy array, so we need to convert it to a
            # DataFrame
            gprime = pd.DataFrame(gprime.flatten())
            gprime.columns = data_mgr.y.columns
    # gprime = pd.DataFrame(
    #     data_mgr.family.link.deriv(eta).flatten()
    # )
    # print("gprime", gprime)
    # print(type(gprime))
    # exit(0)

    # compute Z matrix and dispersion matrix
    y_minus_mu = data_mgr.y.sub(mu, axis=0)
    # print("y_minus_mu", y_minus_mu)

    z = eta + (y_minus_mu / gprime)
    # print("z", z)

    W = gprime**2 / varg

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
        "sum_y": float(np.sum(data_mgr.y)),
    }
