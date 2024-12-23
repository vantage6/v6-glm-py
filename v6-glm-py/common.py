from enum import Enum
import re
import pandas as pd
import statsmodels.api as sm

from formulaic import Formula

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.exceptions import UserInputError


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

    Returns
    -------
    str
        The formula for the GLM model
    """
    predictors = {}
    if category_reference_variables is not None:
        for var in predictor_variables:
            if var in category_reference_variables:
                predictors[var] = (
                    f"C({var}, "
                    f"Treatment(reference='{category_reference_variables[var]}'))"
                )
            else:
                predictors[var] = var
    else:
        predictors = {var: var for var in predictor_variables}
    return f"{outcome_variable} ~ {' + '.join(predictors.values())}"


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
        dstar : str, optional
            An optional parameter for additional model specifications.
        """

        self.df = df
        self.formula = formula
        self.family_str = family
        self.dstar = dstar

        self.y, self.X = self._get_design_matrix()
        self.family = get_family(self.family_str)

        self.mu_start = None

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
        # TODO check if this is correct - Copilot suggests the latter but what is
        # happening now is what happens in R version
        # Also, note that R has separate if statement of Relative survival Poisson
        # models
        self.mu_start = self.y + 0.1
        # mu_start = np.maximum(y, 0.1)

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
        pattern = r"C\(([^,]+), Treatment\(reference='[^']+'\)\)\[([^\]]+)\]"
        replacement = r"\1[\2]"
        simplified_columns = [
            re.sub(pattern, replacement, column_name) for column_name in columns
        ]
        return pd.Index(simplified_columns)
