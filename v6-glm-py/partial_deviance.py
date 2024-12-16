import pandas as pd
import numpy as np

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

from .common import GLMDataManager


# TODO the weights seem not to be implemented completely in R. Maybe they are not needed?
@data(1)
def compute_local_deviance(
    df: pd.DataFrame,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    is_first_iteration: bool,
    dstar: str,
    beta_coefficients: list[int],
    beta_coefficients_previous: list[int],
    weighted_derivative_mu: list[int],
    category_reference_values: dict[str, str] | None = None,
    offset_column: str = None,
    types: list[str] = None,
    weights: list[int] = None,
) -> dict:
    """
    TODO add description
    """
    # TODO this function computes deviance_null which is never used. Why?
    info("Computing local deviance")

    data_mgr = GLMDataManager(
        df,
        outcome_variable,
        predictor_variables,
        family,
        category_reference_values,
        dstar,
        offset_column,
        weights,
    )

    beta_coefficients = pd.Series(beta_coefficients)
    beta_coefficients_previous = pd.Series(beta_coefficients_previous)

    # update mu and compute deviance, then compute eta
    eta_old = data_mgr.compute_eta(is_first_iteration, beta_coefficients_previous)
    if is_first_iteration:
        data_mgr.set_mu_start()
        mu_old = data_mgr.mu_start
        deviance_old = 0
    else:
        mu_old = data_mgr.family.link.inverse(eta_old)
        deviance_old = data_mgr.compute_deviance(mu_old)

    # update beta coefficients
    eta_new = data_mgr.compute_eta(is_first_iteration=False, betas=beta_coefficients)
    mu_new = data_mgr.family.link.inverse(eta_new)
    deviance_new = data_mgr.compute_deviance(mu_new)
    deviance_null = data_mgr.compute_deviance(weighted_derivative_mu)

    return {
        "deviance_old": float(deviance_old),
        "deviance_new": float(deviance_new),
        "deviance_null": float(deviance_null),
    }
