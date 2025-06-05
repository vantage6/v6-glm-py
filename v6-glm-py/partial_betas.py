import pandas as pd
import numpy as np
import statsmodels.genmod.families as families

from vantage6.algorithm.tools.util import info, get_env_var
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation

from .common import Family, GLMDataManager, cast_to_pandas
from .constants import ENVVAR_MAX_PCT_PARAMS_OVER_OBS, DEFAULT_MAX_PCT_PARAMS_VS_OBS


@data(1)
def compute_local_betas(
    df: pd.DataFrame,
    formula: str,
    family: str,
    is_first_iteration: bool,
    beta_coefficients: dict[str, float] | None = None,
    categorical_predictors: list[str] | None = None,
    survival_sensor_column: str = None,
    link_function: str | None = None,
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
    categorical_predictors : list[str] | None
        Predictor variables that should be treated as categorical.
    survival_sensor_column : str, optional
        An optional parameter for additional model specifications.
    link_function : str, optional
        The link function to use. For binomial family, can be 'logit' (default)
        or 'log' for relative risks.

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
        categorical_predictors,
        survival_sensor_column,
        link_function=link_function,
    )
    y_column_names = data_mgr.y.columns

    eta = data_mgr.compute_eta(is_first_iteration, beta_coefficients)

    info("Computing beta coefficients")
    mu = data_mgr.compute_mu(eta, y_column_names)
    varg = data_mgr.family.variance(mu)
    varg = cast_to_pandas(varg, columns=y_column_names)

    # TODO in R, we can do gprime <- family$mu.eta(eta), but in Python I could not
    # find a similar function. It is therefore now implemented for each family
    if isinstance(data_mgr.family, families.Poisson):
        # for poisson, this is exp(eta)
        gprime = data_mgr.family.link.inverse(eta)
    elif isinstance(data_mgr.family, families.Binomial):
        # for binomial, this is mu * (1 - mu), which is the same as the variance func
        gprime = data_mgr.family.variance(mu)
    else:
        # For Gaussian family
        gprime = data_mgr.family.link.deriv(eta)
    gprime = cast_to_pandas(gprime, columns=y_column_names)

    # compute Z matrix and dispersion matrix
    y_minus_mu = data_mgr.y.sub(mu, axis=0)

    z = eta + (y_minus_mu / gprime)

    W = gprime**2 / varg

    if family == Family.GAUSSIAN.value:
        dispersion_matrix = W * (y_minus_mu / gprime) ** 2
        dispersion = float(dispersion_matrix.sum().iloc[0])
    else:
        # For non-Gaussian families, the dispersion is not estimated so we don't need to
        # share any information about it.
        dispersion = None

    _check_privacy(df, len(data_mgr.X.columns))

    XT_dot_X_times_W = data_mgr.X.T.dot(data_mgr.X.mul(W.iloc[:, 0], axis=0))
    XT_dot_W_times_z = data_mgr.X.T.dot(W * z)
    return {
        "XTX": XT_dot_X_times_W.to_dict(),
        "XTz": XT_dot_W_times_z.to_dict(),
        "dispersion": dispersion,
        "num_observations": len(df),
        "num_variables": len(data_mgr.X.columns),
        "sum_y": float(data_mgr.y.sum().iloc[0]),
    }


def _check_privacy(df: pd.DataFrame, num_variables: int):
    """
    Check that the privacy threshold is not violated.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    num_variables : int
        The number of variables in the model.

    Raises
    ------
    PrivacyThresholdViolation
        If the privacy threshold is violated.
    """
    # before returning the data, check that the model has limited risks of overfitting.
    # If too many variables are used, there is a chance the data will be reproducible.
    # This is a security measure to prevent data leakage.
    max_pct_vars_vs_obs = get_env_var(
        ENVVAR_MAX_PCT_PARAMS_OVER_OBS, DEFAULT_MAX_PCT_PARAMS_VS_OBS, as_type="int"
    )
    if num_variables * 100 / len(df) > max_pct_vars_vs_obs:
        raise PrivacyThresholdViolation(
            "Number of variables is too high compared to the number of observations. "
            f"This is not allowed to be more than {max_pct_vars_vs_obs}% but is "
            f"{num_variables * 100 / len(df)}%."
        )
