from enum import Enum
import re
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
    DEFAULT_PRIVACY_THRESHOLD_PER_CATEGORY,
    ENVVAR_ALLOWED_COLUMNS,
    ENVVAR_DISALLOWED_COLUMNS,
    ENVVAR_MINIMUM_ROWS,
    ENVVAR_PRIVACY_THRESHOLD_PER_CATEGORY,
)


class Family(str, Enum):
    """TODO docstring"""

    # TODO add more families. Available from statsmodels.genmod.families:
    # from .family import Gaussian, Family, Poisson, Gamma, \
    #     InverseGaussian, Binomial, NegativeBinomial, Tweedie
    POISSON = "poisson"
    BINOMIAL = "binomial"
    GAUSSIAN = "gaussian"
    SURVIVAL = "survival"


# TODO integrate with enum
def get_family(family: str) -> Family:
    """TODO docstring"""
    # TODO figure out which families are supported
    # TODO use dstar?
    if family.lower() == Family.POISSON.value:
        return sm.families.Poisson()
    elif family.lower() == Family.BINOMIAL.value:
        return sm.families.Binomial()
    elif family.lower() == Family.GAUSSIAN.value:
        return sm.families.Gaussian()
    else:
        raise UserInputError(
            f"Family {family} not supported. Please provide one of the supported "
            f"families: {Family.__members__.values()}"
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


def cast_numpy_to_pandas(
    data_: np.ndarray | pd.Series | pd.DataFrame, columns: list[str] | None = None
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
    if not isinstance(data_, np.ndarray):
        return data_
    return pd.DataFrame(data_.flatten(), columns=columns)


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
    dstar : str, optional
        An optional parameter for additional model specifications.
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
        dstar: str = None,
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
        dstar : str, optional
            An optional parameter for additional model specifications.
        """

        self.df = df
        self.formula = formula
        self.family_str = family
        self.dstar = dstar

        # User can indicate if there are numerical predictors that should be treated as
        # categorical.
        if categorical_predictors is not None:
            for predictor in categorical_predictors:
                self.df[predictor] = self.df[predictor].astype("category")

        self.y, self.X = self._get_design_matrix()
        self.family = get_family(self.family_str)

        self.mu_start = None

        self._privacy_checks()

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
            eta = self.family.link(self.mu_start)
        else:
            # dot product cannot be done with a series, so convert to numpy array and
            # reshape to get betas in correct format
            betas = betas.values.reshape(-1, 1)
            eta = self.X.dot(betas)
        eta.columns = self.y.columns
        return eta

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

        # check which words from the formula correspond to columns. These are the
        # columns assumed to be used in the dataframe - otherwise using the formula
        # would lead to other errors.
        formula_words = self._get_words(self.formula)
        columns_used = []
        for word in formula_words:
            if word in self.X.columns:
                columns_used.append(word)

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
    def _get_words(formula: str) -> list[str]:
        """
        Get the potential column names from the formula.

        Parameters
        ----------
        formula : str
            The formula specifying the model.

        Returns
        -------
        list[str]
            The potential column names
        """
        # This is currently implemented just by getting all words from the formula.
        # Those that are not existing column names will be ignored.
        pattern = r"\b\w+\b"
        words = re.findall(pattern, formula)
        return words

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
