import pandas as pd
import numpy as np

from vantage6.algorithm.tools.util import info, warn, error


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
    weight_dmu: list[int],
    types: list[str] = None,
    weights: list[int] = None,
) -> dict:
    """
    TODO add description
    """
    info("Computing local deviance")
