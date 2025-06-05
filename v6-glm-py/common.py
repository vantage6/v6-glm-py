from enum import Enum
import re
from typing import Any
import pandas as pd
import numpy as np
import statsmodels.api as sm
from formulaic import Formula

from vantage6.algorithm.tools.util import info, get_env_var
from vantage6.algorithm.tools.exceptions import (
    UserInputError,
    PrivacyThresholdViolation,
    NodePermissionException,
)

from .constants import (
    DEFAULT_MINIMUM_ROWS,
    ENVVAR_ALLOWED_COLUMNS,
    ENVVAR_DISALLOWED_COLUMNS,
    ENVVAR_MINIMUM_ROWS,
)


class Family(str, Enum):
    """Enum for the exponential families supported in this algorithm"""

    POISSON = "poisson"
    BINOMIAL = "binomial"
    GAUSSIAN = "gaussian"
    SURVIVAL = "survival"


def get_family(family: str, link_function: str | None = None) -> Family:
    """Get the exponential family object from the statsmodels package
    Parameters
    ----------
    family : str
        Name of the family to use
    link_function : str | None
        The link function to use. For binomial family, can be 'logit' (default)
        or 'log' for relative risks. Ignored for other families.

    Returns
    -------
    sm.families.Family
        The family object configured with the appropriate link function

    Raises
    ------
    UserInputError
        If the family is not supported or if an invalid link function is specified
    """

    if family == Family.POISSON.value:
        return sm.families.Poisson()
    elif family == Family.BINOMIAL.value:
        # Handle custom link function for binomial family
        if link_function == 'log':
            return sm.families.Binomial(sm.families.links.log())
        elif link_function == 'logit' or link_function is None:
            return sm.families.Binomial()  # default uses logit link_function
        else:
            raise UserInputError(
                f"Invalid link function '{link_function}' for binomial family. "
                "Valid options are: 'logit' (default) or 'log'"
            )
    elif family == Family.GAUSSIAN.value:
        return sm.families.Gaussian()
    elif family == Family.SURVIVAL.value:
        return sm.families.Poisson()
    else:
        raise UserInputError(
            f"Family {family} not supported. Please provide one of the supported "
            f"families: {', '.join([fam.value for fam in Family])}"
        )


def get_formula(
    outcome_variable: str,
    predictor_variables: list[str],
    category_reference_variables: list[str],
    categorical_predictors: list[str] | None = None,
) -> str:
    """
    Get the formula for the GLM model from the outcome and predictor variables.

    If category_reference_variables is provided, the formula will be created with
    these variables as reference categories according to the formulaic package's
    syntax.

    Parameters
    ----------
    outcome_variable : str
        The outcome variable
    predictor_variables : list[str]
        The predictor variables
    category_reference_variables : list[str]
        The reference categories for the predictor variables
    categorical_predictors : list[str] | None
        Predictor variables that should be treated as categorical even though they are
        numerical.

    Returns
    -------
    str
        The formula for the GLM model
    """
    predictors = {}
    if category_reference_variables is not None:
        for var in predictor_variables:
            if var in category_reference_variables:
                ref_value = category_reference_variables[var]
                if (
                    categorical_predictors is None
                    or var not in categorical_predictors
                    or isinstance(ref_value, str)
                ):
                    ref_value = f"'{ref_value}'"
                predictors[var] = f"C({var}, Treatment(reference={ref_value}))"
            else:
                predictors[var] = var
    else:
        predictors = {var: var for var in predictor_variables}
    return f"{outcome_variable} ~ {' + '.join(predictors.values())}"


def cast_to_pandas(
    data_: np.ndarray | pd.Series | pd.DataFrame | Any,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cast a numpy array to a pandas Series.

    Parameters
    ----------
    data : np.ndarray | pd.Series | pd.DataFrame
        The data to cast. This function does nothing if the data is not a numpy array.
    columns : list[str] | None
        The column names to give in the resulting pandas Data frame

    Returns
    -------
    pd.Series
        The data as a pandas Series.
    """
    if isinstance(data_, np.ndarray):
        return pd.DataFrame(data_.flatten(), columns=columns)
    return pd.DataFrame(data_, columns=columns)


class GLMDataManager:
    """
    A class to manage data for Generalized Linear Models (GLM).

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    formula : str
        The formula specifying the model.
    family_str : str
        The family of the GLM (e.g., 'gaussian', 'binomial').
    survival_sensor_column : str, optional
        An optional parameter for additional model specifications.
    link_function : str, optional
        The link function to use for binomial family.
    y : pd.Series
        The response variable.
    X : pd.DataFrame
        The design matrix.
    family : Family
        The family object corresponding to the family_str.
    mu_start : pd.Series or None
        The initial values for the mean response.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        formula: str,
        family: str,
        categorical_predictors: list[str] | None,
        survival_sensor_column: str = None,
        link_function: str | None = None,
    ) -> None:
        """
        Initialize the GLMDataManager.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        formula : str
            The formula specifying the model.
        family : str
            The family of the GLM (e.g., 'gaussian', 'binomial').
        categorical_predictors : list[str] | None
            Predictor variables that should be treated as categorical.
        survival_sensor_column : str, optional
            An optional parameter for additional model specifications.
        link_function : str, optional
            The link function to use. For binomial family, can be 'logit' (default)
            or 'log' for relative risks.
        """

        self.df = df
        self.formula = formula
        self.family_str = family
        self.survival_sensor_column = survival_sensor_column
        self.link_function = link_function

        # User can indicate if there are numerical predictors that should be treated as
        # categorical.
        if categorical_predictors is not None:
            for predictor in categorical_predictors:
                self.df[predictor] = self.df[predictor].astype("category")

        self.y, self.X = self._get_design_matrix()
        self.y = cast_to_pandas(self.y)
        self.X = cast_to_pandas(self.X)

        # Initialize family with appropriate link function
        self.family = get_family(self.family_str, self.link_function)

        self.mu_start: pd.Series | None = None

        self._privacy_checks()

        # Additional check for binomial family with log link_function
        if self.family_str == Family.BINOMIAL.value and self.link_function == 'log':
            # Check if response variable contains only 0s and 1s
            unique_values = set(self.y.squeeze().unique())
            if not unique_values.issubset({0, 1}):
                raise UserInputError(
                    "For binomial family with log link_function (relative risks), "
                    "the response variable must contain only 0s and 1s."
                )

    def compute_eta(
        self, is_first_iteration: bool, betas: pd.Series | None
    ) -> pd.Series:
        """
        Compute the eta values for the GLM model.

        Parameters
        ----------
        is_first_iteration : bool
            Whether this is the first iteration of the model.
        betas : pd.Series | None
            The beta coefficients. These must be provided if is_first_iteration is
            False.

        Returns
        -------
        pd.Series
            The eta values for the GLM model.
        """
        info("Computing eta values")
        if is_first_iteration:
            if self.mu_start is None:
                self.set_mu_start()
            if self.family_str == Family.SURVIVAL:
                survival_sensor_column = self.df[self.survival_sensor_column]
                eta = (self.mu_start.squeeze() - survival_sensor_column).apply(np.log)
                eta = cast_to_pandas(eta)
            else:
                eta = self.family.link(self.mu_start)
        else:
            # dot product cannot be done with a series, so convert to numpy array and
            # reshape to get betas in correct format
            betas = betas.values.reshape(-1, 1)
            eta = self.X.dot(betas)
        eta.columns = self.y.columns
        return eta

    def compute_mu(self, eta: pd.Series, columns: list[str] | None = None) -> pd.Series:
        """
        Compute the mean response variable for the GLM model.

        Parameters
        ----------
        eta : pd.Series
            The eta values.
        columns : list[str] | None
            The column names of the response variable. Optional.

        Returns
        -------
        pd.Series
            The mean response variable.
        """
        if self.family_str == Family.SURVIVAL:
            # custom link function for survival models
            mu = self.df[self.survival_sensor_column].add(eta.squeeze().apply(np.exp))
        else:
            mu = self.family.link.inverse(eta)
        return cast_to_pandas(mu, columns=columns)

    def compute_deviance(self, mu: pd.Series) -> float:
        """
        Compute the deviance for the GLM model.

        Parameters
        ----------
        mu : pd.Series
            The mean response variable.

        Returns
        -------
        float
            The deviance for the GLM model.
        """
        y = self.y.squeeze()
        if isinstance(mu, pd.DataFrame):
            mu = mu.squeeze()
        return self.family.deviance(y, mu)

    def set_mu_start(self) -> None:
        """
        Set the initial values for the mean response variable.
        """
        if self.family_str == Family.SURVIVAL:
            self.mu_start = (
                np.maximum(self.y.squeeze(), self.df[self.survival_sensor_column]) + 0.1
            )
            self.mu_start = cast_to_pandas(self.mu_start)
        else:
            self.mu_start = self.family.starting_mu(self.y)

    def _get_design_matrix(self) -> tuple[pd.Series, pd.DataFrame]:
        """
        Create the design matrix X and predictor variable y

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing the predictor variable y and the design matrix X
        """
        info("Creating design matrix X and predictor variable y")
        y, X = Formula(self.formula).get_model_matrix(self.df)
        X.columns = self._simplify_column_names(X.columns)
        return y, X

    def _privacy_checks(self) -> None:
        """
        Do privacy checks on the data after initializing the GLMDataManager.

        Raises
        ------
        PrivacyThresholdViolation
            If the data contains too few values for at least one category of a
            categorical variable.
        """
        # check if dataframe is long enough
        min_rows = get_env_var(
            ENVVAR_MINIMUM_ROWS, default=DEFAULT_MINIMUM_ROWS, as_type="int"
        )
        if len(self.df) < min_rows:
            raise PrivacyThresholdViolation(
                f"Data contains less than {min_rows} rows. Refusing to "
                "handle this computation, as it may lead to privacy issues."
            )

        # check which columns the formula needs. These require some additional checks
        columns_used = Formula(self.formula).required_variables

        # check that a column has at least required number of non-null values
        for col in columns_used:
            if self.df[col].count() < min_rows:
                raise PrivacyThresholdViolation(
                    f"Column {col} contains less than {min_rows} non-null values. "
                    "Refusing to handle this computation, as it may lead to privacy "
                    "issues."
                )

        # Check if requested columns are allowed to be used for GLM by node admin
        allowed_columns = get_env_var(ENVVAR_ALLOWED_COLUMNS)
        if allowed_columns:
            allowed_columns = allowed_columns.split(",")
            for col in columns_used:
                if col not in allowed_columns:
                    raise NodePermissionException(
                        f"The node administrator does not allow '{col}' to be requested"
                        " in this algorithm computation. Please contact the node "
                        "administrator for more information."
                    )
        non_allowed_collumns = get_env_var(ENVVAR_DISALLOWED_COLUMNS)
        if non_allowed_collumns:
            non_allowed_collumns = non_allowed_collumns.split(",")
            for col in columns_used:
                if col in non_allowed_collumns:
                    raise NodePermissionException(
                        f"The node administrator does not allow '{col}' to be requested"
                        " in this algorithm computation. Please contact the node "
                        "administrator for more information."
                    )

    @staticmethod
    def _simplify_column_names(columns: pd.Index) -> pd.Index:
        """
        Simplify the column names of the design matrix

        Parameters
        ----------
        columns : pd.Index
            The column names of the design matrix
        predictors : list[str]
            The predictor variables

        Returns
        -------
        pd.Index
            The simplified column names
        """
        # remove the part of the column name that specifies the reference value
        # e.g. C(prog, Treatment(reference='General'))[T.Vocational] ->
        # prog[T.Vocational]
        pattern = r"C\(([^,]+), Treatment\(reference=[^\)]+\)\)\[([^\]]+)\]"
        replacement = r"\1[\2]"
        simplified_columns = [
            re.sub(pattern, replacement, column_name) for column_name in columns
        ]
        return pd.Index(simplified_columns)
