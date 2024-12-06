from enum import Enum
import pandas as pd
import statsmodels.api as sm

from formulaic import Formula
from statsmodels.genmod.families import Family
from formulaic import Formula

from vantage6.algorithm.tools.util import info


class Family(str, Enum):
    """TODO docstring"""

    # TODO add more families. Available from statsmodels.genmod.families:
    # from .family import Gaussian, Family, Poisson, Gamma, \
    #     InverseGaussian, Binomial, NegativeBinomial, Tweedie
    POISSON = "poisson"
    BINOMIAL = "binomial"
    GAUSSIAN = "gaussian"


# TODO integrate with enum
def get_family(self) -> Family:
    """TODO docstring"""
    # TODO figure out which families are supported
    # TODO use dstar?
    if self.family.lower() == Family.POISSON.value:
        return sm.families.Poisson()
    elif self.family.lower() == Family.BINOMIAL.value:
        return sm.families.Binomial()
    elif self.family.lower() == Family.GAUSSIAN.value:
        return sm.families.Gaussian()
    else:
        raise ValueError(f"Family {self.family} not supported")


class GLMDataManager:
    """TODO docstring"""

    def __init__(
        self,
        df: pd.DataFrame,
        outcome_variable: str,
        predictor_variables: list[str],
        family: str,
        dstar: str = None,
        offset_column: str = None,
        weights: list[int] = None,
    ):
        self.df = df
        self.outcome_variable = outcome_variable
        self.predictor_variables = predictor_variables
        self.family = family
        self.dstar = dstar
        self.offset_column = offset_column
        self.weights = weights if weights is not None else pd.Series([1] * len(df))

        self.y, self.X = self._get_design_matrix()
        self.offset = self._get_offset()
        self.family = get_family()

        self.mu_start = None

    def compute_eta(self, is_first_iteration: bool, betas: list[int]) -> pd.Series:
        """TODO docstring"""
        info("Computing eta values")
        if is_first_iteration:
            if self.mu_start is None:
                self.set_mu_start()
            eta = self.family.link(self.mu_start)
        else:
            eta = self.X.dot(betas) + self.offset

        return eta

    def compute_deviance(self, mu: pd.Series) -> float:
        """TODO docstring"""
        return self.family.deviance(self.y, mu, self.weights)

    def set_mu_start(self) -> None:
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
        formula = f"{self.outcome_variable} ~ {' + '.join(self.predictor_variables)}"
        y, X = Formula(formula).get_model_matrix(self.df)
        return y, X

    def _get_offset(self) -> pd.Series:
        """
        Extract the offset if specified, otherwise set it to 0

        Returns
        -------
        pd.Series
            The offset values

        Raises
        ------
        KeyError
            If the offset column is not found in the data frame
        """
        info("Extracting offset values")
        if self.offset_column is not None:
            try:
                offset = self.df[self.offset_column]
            except KeyError as exc:
                raise KeyError(
                    f"Offset column {self.offset_column} not found in data frame"
                ) from exc
        else:
            offset = pd.Series([0] * len(self.df))
        return offset
