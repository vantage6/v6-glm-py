"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""

from typing import Any
from enum import Enum

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Family
from formulaic import Formula

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import data


class Family(str, Enum):
    # TODO add more families. Available from statsmodels.genmod.families:
    # from .family import Gaussian, Family, Poisson, Gamma, \
    #     InverseGaussian, Binomial, NegativeBinomial, Tweedie
    POISSON = "poisson"
    BINOMIAL = "binomial"
    GAUSSIAN = "gaussian"


# TOOD create enum of allowed families
@data(1)
def compute_partial_betas(
    df: pd.DataFrame,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    is_first_iteration: bool,
    beta_coefficients: list[int],
    dstar: str = None,
    types: list[str] = None,
    weights: list[int] = None,
) -> Any:
    """Compute beta coefficients for a GLM model"""
    info("Started function to compute beta coefficients")

    # Create the design matrix X and predictor variable y
    info("Creating design matrix X and predictor variable y")
    formula = f"{outcome_variable} ~ {' + '.join(predictor_variables)}"
    y, X = Formula(formula).get_model_matrix(df)
    # print("y", y)
    print("X", X)

    # Extract the offset if specified, otherwise set it to 0
    offset = df.get("offset", pd.Series([0] * len(df)))
    # print("offset", offset)

    # Get the requested family
    # TODO get dstar
    # if family == "rs.poi":
    #     dstar = eval(dstar, globals(), data.to_dict(orient='series'))
    family = _get_family(family, dstar)

    if weights is None:
        weights = pd.Series([1] * len(df))

    if is_first_iteration:
        # Initialize the model
        info("Initializing the model")

        # TODO There is an if-statement in the R code for relative survival Poission
        # models. Check how to implement this in Python
        # if isinstance(family, sm.families.Poisson):
        #     print(y)
        #     print(dstar)
        #     mu_start = np.maximum(y, dstar) + 0.1
        #     print(mu_start)
        # TODO check if this is correct - Copilot suggests the latter but what is
        # happening now is what happens in R version
        mu_start = y + 0.1
        # mu_start = np.maximum(y, 0.1)
        # print("mu_start", mu_start)

        # ???
        # model = sm.GLM(y, X, family=family, offset=offset, freq_weights=weights)
        # model.initialize()

        eta = family.link(mu_start)
    else:
        # TODO check if this correctly translates the dot product to a dataframe
        eta = pd.DataFrame(np.dot(X, beta_coefficients) + offset)
    # print("eta", eta)

    info("Computing beta coefficients")
    mu = family.link.inverse(eta)
    # print("mu", mu)
    varg = family.variance(mu)
    # print("varg", varg)

    # TODO below is what Copilot suggests but what is active is what is in the R version
    gprime = family.link.inverse(eta)  # for poisson, this is exp(eta)
    # gprime = family.link.deriv(eta)  # for poisson, this is 1 / eta
    # print("gprime", gprime)

    # compute Z matrix, weights and dispersion matrix
    y_minus_mu = y.sub(mu, axis=0)

    z = (eta.sub(offset, axis=0)) + (y_minus_mu / gprime)
    # print("z", z)
    # z = (eta - offset) + (y - mu) * gprime

    W = (gprime**2 / varg).mul(weights, axis=0)
    print("W", W)
    print(type(W), W.columns)
    dispersion_matrix = W * (y_minus_mu / gprime) ** 2
    dispersion = dispersion_matrix.sum()
    # dispersion = sum(W * ((y - mu) / gprime) ** 2) / sum(W)
    print("dispersion", dispersion)

    print("W*X", X.mul(W.iloc[:, 0], axis=0))
    # TODO there are some non-clear things in the code like `mul()` and `iloc[:, 0]`.
    # They are there to ensure proper multiplication etc of pandas Dataframes with
    # series. Make this code more clear and readable.
    return= {
        "v1": np.dot(X.T, X.mul(W.iloc[:, 0], axis=0)).tolist(),
        "v2": np.dot(X.T, W * z).tolist(),
        "dispersion": float(dispersion),
        "nobs": len(df),
        "nvars": len(X.columns),
        "wt1": float(np.sum(weights.mul(y.iloc[:, 0], axis=0))),
        "wt2": float(np.sum(weights)),
    }


def _get_family(family: str, dstar: str) -> Family:
    # TODO figure out which families are supported
    if family == Family.POISSON.value:
        return sm.families.Poisson()
    elif family == Family.BINOMIAL.value:
        return sm.families.Binomial()
    elif family == Family.GAUSSIAN.value:
        return sm.families.Gaussian()
    else:
        raise ValueError(f"Family {family} not supported")
