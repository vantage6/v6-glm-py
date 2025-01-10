"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

from functools import reduce
import numpy as np
import pandas as pd
import scipy.stats as stats

from vantage6.algorithm.tools.util import info, warn, get_env_var
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.exceptions import UserInputError, AlgorithmExecutionError

from .common import Family, get_formula
from .constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    ENVVAR_MINIMUM_ORGANIZATIONS,
    DEFAULT_MINIMUM_ORGANIZATIONS,
)


@algorithm_client
def glm(
    client: AlgorithmClient,
    family: str,
    outcome_variable: str | None = None,
    predictor_variables: list[str] | None = None,
    formula: str | None = None,
    categorical_predictors: list[str] = None,
    category_reference_values: dict[str, str] = None,
    dstar: str = None,
    tolerance_level: int = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    organizations_to_include: list[int] = None,
) -> dict:
    """
    Central part of the GLM algorithm

    This function creates subtasks for all the organizations involved in the GLM
    computation to compute partial results on their data and aggregates these, over
    multiple cycles, until the GLM is converged.

    Parameters
    ----------
    client : AlgorithmClient
        The client object to interact with the server
    family : str, optional
        The exponential family to use for computing the GLM. The available families are
        Gaussian, Poisson, Binomial, and Survival.
    outcome_variable : str, optional
        The name of the outcome variable column, by default None. If not provided, the
        formula must be provided.
    predictor_variables : list[str], optional
        The names of the predictor variable columns, by default None. If not provided,
        the formula must be provided.
    formula : str, optional
        The formula to use for the GLM, by default None. If not provided, the
        outcome_variable and predictor_variables must be provided.
    categorical_predictors : list[str], optional
        The column names of the predictor variables that are categorical. All columns
        with string values are considered categorical by default - this option should
        be used for columns with numerical values that should be treated as categorical.
    category_reference_values : dict[str, str], optional
        The reference values for the categorical variables, by default None. If, for
        instance, the predictor variable 'A' is a categorical variable with values
        'a', 'b', and 'c', and we want 'a' to be the reference value, this dictionary
        should be {'A': 'a'}.
    dstar : str, optional
        The dstar value, by default None. TODO improve explanation if we keep this
    tolerance_level : int, optional
        The tolerance level for the convergence of the algorithm, by default 1e-8.
    max_iterations : int, optional
        The maximum number of iterations for the algorithm, by default 25.
    organizations_to_include : list[int], optional
        The organizations to include in the computation, by default None. If not
        provided, all organizations in the collaboration are included.

    Returns
    -------
    dict
        The results of the GLM computation, including the coefficients and details of
        the computation.
    """
    # select organizations to include
    if not organizations_to_include:
        organizations = client.organization.list()
        organizations_to_include = [
            organization.get("id") for organization in organizations
        ]

    min_orgs = get_env_var(
        ENVVAR_MINIMUM_ORGANIZATIONS, DEFAULT_MINIMUM_ORGANIZATIONS, as_type="int"
    )
    if len(organizations_to_include) < min_orgs:
        raise UserInputError(
            "Number of organizations included in the computation is less than the "
            f"minimum required ({min_orgs})."
        )

    # Either formula or outcome and predictor variables should be provided
    if formula and (outcome_variable or predictor_variables):
        warn(
            "Both formula or outcome and predictor variables are provided - using "
            "the formula and ignoring the outcome/predictor."
        )
    if not formula and not (outcome_variable and predictor_variables):
        raise UserInputError(
            "Either formula or outcome and predictor variables should be provided. "
            "Neither is provided."
        )
    if not formula and outcome_variable:
        formula = get_formula(
            outcome_variable,
            predictor_variables,
            category_reference_values,
            categorical_predictors,
        )

    # Iterate to find the coefficients
    iteration = 1
    betas = None
    while iteration <= max_iterations:
        converged, new_betas, deviance = _do_iteration(
            iteration=iteration,
            client=client,
            formula=formula,
            family=family.lower(),
            categorical_predictors=categorical_predictors,
            dstar=dstar,
            tolerance_level=tolerance_level,
            organizations_to_include=organizations_to_include,
            betas_old=betas,
        )
        betas = new_betas["beta_estimates"]

        # terminate if converged or reached max iterations
        if converged:
            info(" - Converged!")
            break
        if iteration == max_iterations:
            warn(" - Maximum number of iterations reached!")
            break
        iteration += 1

    # after the iteration, return the final results
    info("Preparing final results")
    betas = pd.Series(betas)
    std_errors = pd.Series(new_betas["std_error_betas"])
    zvalue = betas / std_errors
    if new_betas["is_dispersion_estimated"]:
        pvalue = 2 * stats.t.cdf(
            -np.abs(zvalue), new_betas["num_observations"] - new_betas["num_variables"]
        )
    else:
        pvalue = 2 * stats.norm.cdf(-np.abs(zvalue))

    # add back indices to pvalue
    pvalue = pd.Series(pvalue, index=betas.index)

    # create dataframe with results
    results = pd.DataFrame(
        {
            "beta": betas,
            "std_error": std_errors,
            "z_value": zvalue,
            "p_value": pvalue,
        }
    )

    return {
        "coefficients": results.to_dict(),
        "details": {
            "converged": converged,
            "iterations": iteration,
            "dispersion": new_betas["dispersion"],
            "is_dispersion_estimated": new_betas["is_dispersion_estimated"],
            "deviance": deviance["new"],
            "null_deviance": deviance["null"],
            "num_observations": new_betas["num_observations"],
            "num_variables": new_betas["num_variables"],
        },
    }


def _do_iteration(
    iteration: int,
    client: AlgorithmClient,
    formula: str,
    family: str,
    categorical_predictors: list[str],
    dstar: str,
    tolerance_level: int,
    organizations_to_include: list[int],
    betas_old: dict | None = None,
) -> tuple[bool, dict, dict]:
    """
    Execute one iteration of the GLM algorithm

    Parameters
    ----------
    iteration : int
        The iteration number
    client : AlgorithmClient
        The client object to interact with the server
    formula : str
        The formula to use for the GLM
    family : str
        The family of the GLM
    categorical_predictors : list[str]
        The column names of the predictor variables to be treated as categorical
    dstar : str
        The dstar value
    tolerance_level : int
        The tolerance level for the convergence of the algorithm
    organizations_to_include : list[int]
        The organizations to include in the computation
    betas_old : dict, optional
        The beta coefficients from the previous iteration, by default None

    Returns
    -------
    tuple[bool, dict, dict]
        A tuple containing a boolean indicating if the algorithm has converged, a
        dictionary containing the new beta coefficients, and a dictionary containing
        the deviance.
    """
    # print iteration header to logs
    _log_header(iteration)

    # compute beta coefficients
    partial_betas = _compute_local_betas(
        client,
        formula,
        family,
        categorical_predictors,
        dstar,
        iter_num=iteration,
        organizations_to_include=organizations_to_include,
        betas=betas_old,
    )
    info(" - Partial betas obtained!")

    # compute central betas from the partial betas
    info("Computing central betas")
    new_betas = _compute_central_betas(partial_betas, family)
    info(" - Central betas obtained!")

    # compute the deviance for each of the nodes
    info("Computing deviance")
    deviance_partials = _compute_partial_deviance(
        client=client,
        formula=formula,
        family=family,
        categorical_predictors=categorical_predictors,
        iter_num=iteration,
        dstar=dstar,
        beta_estimates=new_betas["beta_estimates"],
        beta_estimates_previous=betas_old,
        global_average_outcome_var=new_betas["y_average"],
        organizations_to_include=organizations_to_include,
    )

    total_deviance = _compute_deviance(deviance_partials)
    info(" - Deviance computed!")

    # check if the algorithm has converged
    converged = False
    if total_deviance["new"] == 0:
        # Safety check to avoid division by zero
        converged = True
    convergence_criterion = (
        abs(total_deviance["old"] - total_deviance["new"]) / total_deviance["new"]
    )
    if convergence_criterion < tolerance_level:
        converged = True
    return converged, new_betas, total_deviance


def _compute_central_betas(
    partial_betas: list[dict],
    family: str,
) -> dict:
    """
    Compute the central beta coefficients from the partial beta coefficients

    Parameters
    ----------
    partial_betas : list[dict]
        The partial beta coefficients from the nodes
    family : str
        The family of the GLM

    Returns
    -------
    dict
        A dictionary containing the central beta coefficients and related metadata
    """
    # sum the contributions of the partial betas
    info("Summing contributions of partial betas")
    total_observations = sum([partial["num_observations"] for partial in partial_betas])
    sum_observations = sum([partial["sum_y"] for partial in partial_betas])

    y_average = sum_observations / total_observations

    XTX_sum = reduce(
        lambda x, y: x + y, [pd.DataFrame(partial["XTX"]) for partial in partial_betas]
    )
    XTz_sum = reduce(
        lambda x, y: x + y, [pd.DataFrame(partial["XTz"]) for partial in partial_betas]
    )
    dispersion_sum = sum([partial["dispersion"] for partial in partial_betas])
    num_observations = sum([partial["num_observations"] for partial in partial_betas])
    num_variables = partial_betas[0]["num_variables"]

    # TODO no idea if this is correct. It's just a translation of the R code
    if family == Family.GAUSSIAN.value:
        dispersion = dispersion_sum / (num_observations - num_variables)
        is_dispersion_estimated = True
    else:
        dispersion = 1
        is_dispersion_estimated = False

    info("Updating betas")

    XTX_np = XTX_sum.to_numpy()
    XTz_np = XTz_sum.to_numpy()

    beta_estimates = np.linalg.solve(XTX_np, XTz_np).flatten()
    std_error_betas = np.sqrt(np.diag(np.linalg.inv(XTX_np) * dispersion))

    # add the indices back to the beta estimates
    indices = pd.DataFrame(partial_betas[0]["XTX"]).index
    beta_estimates = pd.Series(beta_estimates, index=indices)
    std_error_betas = pd.Series(std_error_betas, index=indices)

    return {
        "beta_estimates": beta_estimates.to_dict(),
        "std_error_betas": std_error_betas.to_dict(),
        "dispersion": dispersion,
        "is_dispersion_estimated": is_dispersion_estimated,
        "num_observations": num_observations,
        "num_variables": num_variables,
        "y_average": y_average,
    }


def _compute_deviance(
    partial_deviances: list[dict],
) -> dict:
    """
    Compute the total deviance from the partial deviances

    Parameters
    ----------
    partial_deviances : list[dict]
        The partial deviances from the nodes

    Returns
    -------
    dict
        A dictionary containing the total deviance for the null, old, and new models
    """
    total_deviance_null = sum(
        [partial["deviance_null"] for partial in partial_deviances]
    )
    total_deviance_old = sum([partial["deviance_old"] for partial in partial_deviances])
    total_deviance_new = sum([partial["deviance_new"] for partial in partial_deviances])
    return {
        "null": total_deviance_null,
        "old": total_deviance_old,
        "new": total_deviance_new,
    }


def _compute_local_betas(
    client: AlgorithmClient,
    formula: str,
    family: str,
    categorical_predictors: list[str],
    dstar: str,
    iter_num: int,
    organizations_to_include: list[int],
    betas: list[int] | None = None,
) -> list[dict]:
    """
    Create a subtask to compute the partial beta coefficients for each organization
    involved in the task

    Parameters
    ----------
    client : AlgorithmClient
        The client object to interact with the server
    formula : str
        The formula to use for the GLM
    family : str
        The family of the GLM
    categorical_predictors : list[str]
        The column names of the predictor variables to be treated as categorical
    dstar : str
        The dstar value
    iter_num : int
        The iteration number
    organizations_to_include : list[int]
        The organizations to include in the computation
    betas : list[int], optional
        The beta coefficients from the previous iteration, by default None

    Returns
    -------
    list[dict]
        The results of the subtask
    """
    info("Defining input parameters")
    input_ = {
        "method": "compute_local_betas",
        "kwargs": {
            "formula": formula,
            "family": family,
            "is_first_iteration": iter_num == 1,
        },
    }
    if categorical_predictors:
        input_["kwargs"]["categorical_predictors"] = categorical_predictors
    if dstar:
        input_["kwargs"]["dstar"] = dstar
    if betas:
        input_["kwargs"]["beta_coefficients"] = betas

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial betas subtask",
        description=f"Subtask to compute partial betas - iteration {iter_num}",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # check that each node provided complete results
    _check_partial_results(
        results,
        ["XTX", "XTz", "dispersion", "num_observations", "num_variables", "sum_y"],
    )

    return results


def _compute_partial_deviance(
    client: AlgorithmClient,
    formula: str,
    family: str,
    categorical_predictors: list[str] | None,
    iter_num: int,
    dstar: str,
    beta_estimates: pd.Series,
    beta_estimates_previous: pd.Series | None,
    global_average_outcome_var: int,
    organizations_to_include: list[int],
) -> list[dict]:
    """
    Create a subtask to compute the partial deviance for each organization involved in
    the task

    Parameters
    ----------
    client : AlgorithmClient
        The client object to interact with the server
    formula : str
        The formula to use for the GLM
    family : str
        The family of the GLM
    categorical_predictors : list[str] | None
        The column names of the predictor variables to be treated as categorical
    iter_num : int
        The iteration number
    dstar : str
        The dstar value
    beta_estimates : pd.Series
        The beta coefficients from the current iteration
    beta_estimates_previous : pd.Series | None
        The beta coefficients from the previous iteration
    global_average_outcome_var : int
        The global average of the outcome variable
    organizations_to_include : list[int]
        The organizations to include in the computation

    Returns
    -------
    dict
        The results of the subtask
    """
    info("Defining input parameters")
    input_ = {
        "method": "compute_local_deviance",
        "kwargs": {
            "formula": formula,
            "family": family,
            "is_first_iteration": iter_num == 1,
            "beta_coefficients": beta_estimates,
            "global_average_outcome_var": global_average_outcome_var,
        },
    }
    if categorical_predictors:
        input_["kwargs"]["categorical_predictors"] = categorical_predictors
    if dstar:
        input_["kwargs"]["dstar"] = dstar
    if beta_estimates_previous:
        input_["kwargs"]["beta_coefficients_previous"] = beta_estimates_previous

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial deviance subtask",
        description=f"Subtask to compute partial deviance - iteration {iter_num}",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # check that each node provided complete results
    _check_partial_results(results, ["deviance_null", "deviance_old", "deviance_new"])

    return results


def _check_partial_results(results: list[dict], required_keys: list[str]) -> None:
    """
    Check that each of the partial results contains complete data
    """
    for result in results:
        if result is None:
            raise AlgorithmExecutionError(
                "At least one of the nodes returned invalid result. Please check the "
                "logs."
            )
        for key in required_keys:
            if key not in result:
                raise AlgorithmExecutionError(
                    "At least one of the nodes returned incomplete result. Please check"
                    " the logs."
                )


def _log_header(num_iteration: int) -> None:
    """
    Print header for the iteration to the logs
    """
    info("")
    info("#" * 60)
    info(f"# Starting iteration {num_iteration}")
    info("#" * 60)
    info("")
