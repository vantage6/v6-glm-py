import pandas as pd

from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.util import get_env_var
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation

from .constants import ENVVAR_MIN_ROWS_PER_CATEGORY, DEFAULT_MIN_ROWS_PER_CATEGORY


@data(1)
def get_categorical_levels(
    df: pd.DataFrame,
    columns: list[str],
    categorical_predictors: list[str] | None,
) -> dict:
    """
    Get the categorical levels for the categorical predictors

    Parameters
    ----------
    columns : list[str]
        The columns used in the computation. Part of these will be categorical and those
        will be used to return the categorical levels of
    categorical_predictors : list[str] | None
        Columns that may not appear categorical but are to be treated as such.

    Returns
    -------
    dict
        A dictionary containing the categorical levels for the categorical predictors
    """
    # check which columns are categorical - i.e. combine the categorical from the
    # dataframe with those forced to be categorical
    categorical_columns = list(
        set(
            categorical_predictors
            if categorical_predictors
            else [] + [col for col in columns if df[col].dtype == "object"]
        )
    )

    _check_privacy(df, categorical_columns)

    # get the categorical levels
    return {col: df[col].unique().tolist() for col in categorical_columns}


def _check_privacy(df: pd.DataFrame, categorical_columns: list[str]) -> None:
    """
    Check the privacy of the categorical columns - i.e. are there categories with too
    few values to share them?
    """

    min_rows = get_env_var(
        ENVVAR_MIN_ROWS_PER_CATEGORY,
        DEFAULT_MIN_ROWS_PER_CATEGORY,
        as_type="int",
    )

    for col in categorical_columns:
        for unique_value in df[col].unique():
            if unique_value is None:
                continue
            if df[col].value_counts()[unique_value] < min_rows:
                raise PrivacyThresholdViolation(
                    f"The column {col} has a category with too few matches to share "
                    "that it exists."
                )
