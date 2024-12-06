import pandas as pd
import numpy as np

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

from .common import GLMDataManager


@data(1)
def compute_local_deviance(
    df: pd.DataFrame,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    is_first_iteration: bool,
    dstar: str,
    beta_coefficients_overall: list[int],
    beta_coefficients_previous: list[int],
    offset_column: str = None,
    weighted_derivative_mu: list[int],
    types: list[str] = None,
    weights: list[int] = None,
) -> dict:
    """
    TODO add description
    """
    # TODO note that this function was not tested yet! No idea if it works as intended
    info("Computing local deviance")

    data_mgr = GLMDataManager(
        df,
        outcome_variable,
        predictor_variables,
        family,
        beta_coefficients_previous, # NB: beta coefficients from previous iteration!
        dstar,
        offset_column,
        weights,
    )

    # update mu and compute deviance, then compute eta
    eta_old = data_mgr.compute_eta(is_first_iteration)
    if is_first_iteration:
        data_mgr.set_mu_start()
        mu_old = data_mgr.mu_start
        deviance_old = 0
    else:
        mu_old = data_mgr.family.link.inverse(eta_old)
        deviance_old = data_mgr.compute_deviance(mu_old)

    # update beta coefficients
    eta_new = data_mgr.compute_eta(is_first_iteration=False, betas=beta_coefficients_overall)
    mu_new = data_mgr.family.link.inverse(eta_new)
    deviance_new = data_mgr.compute_deviance(mu_new)
    deviance_null = data_mgr.compute_deviance(weighted_derivative_mu)

    return {
        "deviance_old": deviance_old,
        "deviance_new": deviance_new,
        "deviance_null": deviance_null,
    }

