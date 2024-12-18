from enum import Enum
import pandas as pd
import statsmodels.api as sm

from formulaic import Formula
from statsmodels.genmod.families import Family

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


class GLMDataManager:
    """TODO docstring"""

    def __init__(
        self,
        df: pd.DataFrame,
        outcome_variable: str,
        predictor_variables: list[str],
        family: str,
        category_reference_values: dict[str, str] = None,
        dstar: str = None,
        weights: list[int] = None,
    ):
        self.df = df
        self.outcome_variable = outcome_variable
        self.predictor_variables = predictor_variables
        self.family_str = family
        self.category_reference_values = category_reference_values
        self.dstar = dstar
        self.weights = weights if weights is not None else pd.Series([1] * len(df))

        self.y, self.X = self._get_design_matrix()
        self.family = get_family(self.family_str)

        self.mu_start = None

    def compute_eta(self, is_first_iteration: bool, betas: pd.Series) -> pd.Series:
        """TODO docstring"""
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
        """TODO docstring"""
        y = self.y.squeeze()
        if isinstance(mu, pd.DataFrame):
            mu = mu.squeeze()
        return self.family.deviance(y, mu, self.weights)

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
        # define the formula, including the reference values for categorical variables
        predictors = {}
        if self.category_reference_values is not None:
            for var in self.predictor_variables:
                if var in self.category_reference_values:
                    predictors[var] = (
                        f"C({var}, "
                        f"Treatment(reference='{self.category_reference_values[var]}'))"
                    )
                else:
                    predictors[var] = var
        else:
            predictors = {var: var for var in self.predictor_variables}
        formula = f"{self.outcome_variable} ~ {' + '.join(predictors.values())}"

        y, X = Formula(formula).get_model_matrix(self.df)
        X.columns = self._simplify_column_names(X.columns, predictors)
        return y, X

    @staticmethod
    def _simplify_column_names(columns: pd.Index, predictors: list[str]) -> pd.Index:
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
        simplified_columns = []
        for col in columns:
            simplified_col = col
            for pred_key, pred_value in predictors.items():
                if pred_value in col:
                    simplified_col = simplified_col.replace(pred_value, pred_key)
                    break
            simplified_columns.append(simplified_col)
        return pd.Index(simplified_columns)
