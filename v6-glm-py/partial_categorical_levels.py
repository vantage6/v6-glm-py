import pandas as pd

from vantage6.algorithm.tools.decorators import data


# TODO privacy guards
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
    print("--------------------------------")
    print(df)
    print(df.columns)
    print(df.dtypes)
    print(columns)
    print("--------------------------------")
    categorical_columns = list(
        set(
            categorical_predictors
            if categorical_predictors
            else [] + [col for col in columns if df[col].dtype == "object"]
        )
    )

    # get the categorical levels
    return {col: df[col].unique().tolist() for col in categorical_columns}
