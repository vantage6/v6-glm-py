import pandas as pd

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

from .common import GLMDataManager, cast_to_pandas


@data(1)
def compute_local_deviance(
    df: pd.DataFrame,
    formula: str,
    family: str,
    is_first_iteration: bool,
    global_average_outcome_var: float,
    beta_coefficients: list[int],
    beta_coefficients_previous: list[int] | None = None,
    categorical_predictors: list[str] | None = None,
    survival_sensor_column: str | None = None,
    link_function: str | None = None,
) -> dict:
    """
    Compute the local deviance for a GLM model given the beta coefficients of the global
    model.

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
    global_average_outcome_var : float
        The global average of the response variable.
    beta_coefficients : list[int]
        The beta coefficients of the current model.
    beta_coefficients_previous : list[int]
        The beta coefficients of the previous model.
    categorical_predictors : list[str] | None
        Predictor variables that should be treated as categorical.
    survival_sensor_column : str | None
        An optional parameter for additional model specifications.
    link_function : str, optional
        The link function to use. For binomial family, can be 'logit' (default)
        or 'log' for relative risks.

    Returns
    -------
    dict
        The computed deviance values.
    """
    info("Computing local deviance")

    data_mgr = GLMDataManager(
        df,
        formula,
        family,
        categorical_predictors,
        survival_sensor_column,
        link_function=link_function,
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
        mu_old = data_mgr.compute_mu(eta_old)
        deviance_old = data_mgr.compute_deviance(mu_old)

    # update beta coefficients
    eta_new = data_mgr.compute_eta(is_first_iteration=False, betas=beta_coefficients)
    mu_new = data_mgr.compute_mu(eta_new)

    deviance_new = data_mgr.compute_deviance(mu_new)
    deviance_null = data_mgr.compute_deviance(global_average_outcome_var)

    return {
        "deviance_old": float(deviance_old),
        "deviance_new": float(deviance_new),
        "deviance_null": float(deviance_null),
    }
